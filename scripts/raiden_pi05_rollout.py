#!/usr/bin/env python3
"""Roll out a pi05 (openpi) policy on the YAM arm — monolithic bimanual runner on the robot host.

pi05 does NOT go through raiden's ModelBridge / `rd infer` (this raiden branch lacks
`raiden.inference`). Instead this runner reuses the SAME proven hardware scaffold as
`raiden_rollout.py` (background ZED capture, threaded proprio, footpedal e-stop,
command-to-command jump guard, homing on exit) and links to a running openpi policy
server (`scripts/serve_policy.py`) over its websocket — exactly what `deployment/
openpi_bridge.py` does, but driving THIS raiden's `RobotController` directly.

Two processes (both on the robot host — pi05 inference fits the local 4090, ~16GB):

    # 1) serve the checkpoint (GPU). openpi lives under robometer_policy_learning's third_party/dsrl_openpi.
    cd <robometer_policy_learning>/third_party/dsrl_openpi && uv run --no-sync scripts/serve_policy.py \
        policy:checkpoint --policy.config=pi05_yam_banana \
        --policy.dir=<outputs>/pi05_yam_banana/checkpoints/9000

    # 2) drive the arm (raiden venv), querying localhost:8000
    python -m robometer_policy_learning.scripts.raiden_pi05_rollout --host localhost --port 8000 \
        --prompt "pick up the banana and put it into the box" \
        --control-hz 15 --action-horizon 25 --max-steps 400 [--dry-run | --selftest]

pi05 is BIMANUAL: it consumes both arms' proprio (14-D state) + 3 cameras, and emits a
14-D joint action. Layouts (from deployment/openpi_bridge.py):
  raiden proprio per arm: [joints(6), gripper(1)]
  openpi state (14-D):    [l_joints(6), l_grip(1), r_joints(6), r_grip(1)]
  openpi action (14-D):   [l_joints(6), l_grip(1), r_joints(6), r_grip(1)]  (absolute joints)
  → command follower_l with action[0:7], follower_r with action[7:14].

Actions are ABSOLUTE joint targets (config use_delta_joint_actions=False; the server
un-normalizes). The ActionChunkBroker caches one server call and dispenses one action per
tick for `action_horizon` ticks, then re-queries — so per-tick latency is amortized and the
executed action is already smooth (no temporal ensembling needed, unlike the stochastic
flow policy).

Safety: --dry-run (inference on live hardware, never commands the arm), --selftest (no
hardware, just checks the server round-trips), footpedal e-stop, per-arm command jump guard,
homing on exit. openpi_bridge has no dry-run; this runner adds one.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np


def home_with_recovery(robot, tag="pi05"):
    """Return both arms home, escalating recovery ONLY as far as needed.

    CRITICAL ORDERING (fixes "arm collapses on stop"): the previous version reset the CAN interface
    (`ip link set <can> down`) on the FIRST home exception. But bringing the bus down DE-ENERGIZES
    the motors — so a single transient home hiccup would trigger a reset that COLLAPSES the arm, and
    if the post-reset retry failed (e.g. no passwordless sudo) it stayed collapsed. The working
    reference (act_infer) never resets CAN — it just retries/warns and keeps torque.

    So we escalate gently, keeping torque as long as possible:
      1. home;
      2. on failure: short settle + retry home (NO bus reset — motors stay energized);
      3. on failure: re-init the motor chain (no `ip link`) + retry;
      4. LAST RESORT ONLY: reset the CAN interface (this drops torque) + re-init + retry — reached
         only when the bus is genuinely wedged, never on a first transient.
    Best-effort — the caller always zeroes torque via close() afterward.
    """
    def _chain_alive():
        """Is the motor-chain SERVER THREAD actually running on both followers? A dead chain makes
        move_to_home_positions a SILENT no-op (prints 'reached home', applies zero torque → arm
        collapses). NOTE: controller.follower_l/r ARE the MotorChainRobot directly (verified: no
        ._robot wrapper — an earlier version wrongly indirected through ._robot and ALWAYS returned
        False, causing a redundant re-init that broke a healthy chain). On a live chain:
        motor_chain.running=True, _server_thread.is_alive()=True, _stop_event.is_set()=False."""
        for fol in (getattr(robot, "follower_l", None), getattr(robot, "follower_r", None)):
            if fol is None:
                return False
            chain = getattr(fol, "motor_chain", None)
            if chain is not None and getattr(chain, "running", True) is False:
                return False
            th = getattr(fol, "_server_thread", None)
            if th is not None and not th.is_alive():
                return False
            stop = getattr(fol, "_stop_event", None)
            if stop is not None and stop.is_set():
                return False
        return True

    def _try_home(label):
        # A dead chain won't raise on command — it just no-ops. So verify the chain is alive FIRST;
        # if not, this attempt can't really home (caller re-inits and retries).
        if not _chain_alive():
            print(f"[{tag}][WARN] motor chain not running{label} — cannot home (needs re-init).", flush=True)
            return False
        try:
            robot.move_to_home_positions()
            print(f"[{tag}] homed{label}.", flush=True)
            return _chain_alive()   # re-verify: a loss-comm mid-home kills the chain silently
        except Exception as e:
            print(f"[{tag}][WARN] home failed{label}: {e}", flush=True)
            return False

    # ROOT CAUSE (from live logs): during a driven rollout, a stop/transition loss-communication
    # crashes the i2rt motor-chain SERVER THREAD ("motor_chain is not running, exiting the robot
    # server"). After that, move_to_home_positions() drives a DEAD chain — no error, no torque —
    # so the arm COLLAPSES while the log falsely says "homed". Reset-arm works precisely because it
    # RE-INITIALIZES the chain from scratch. So make stop-homing do the same: recover the chain, then
    # home. This is the fix for "reset arm works but stop collapses".

    # 1. if the chain is already alive, a plain home works
    if _try_home(""):
        return True
    # 2. chain likely died on the stop transition → RE-INITIALIZE the motor chain (what Reset-arm
    #    does), NO destructive `ip link` reset, then home. This is the key recovery.
    print(f"[{tag}] motor chain down — re-initializing (like Reset-arm), then homing...", flush=True)
    try:
        robot.initialize_robots()
        time.sleep(0.5)
    except Exception as e:
        print(f"[{tag}][WARN] re-init failed: {e}", flush=True)
    if _try_home(" (after re-init)"):
        return True
    # 3. one more re-init + retry (transient) before the last-resort CAN reset
    try:
        robot.initialize_robots()
        time.sleep(0.5)
    except Exception as e:
        print(f"[{tag}][WARN] 2nd re-init failed: {e}", flush=True)
    if _try_home(" (after 2nd re-init)"):
        return True
    # 4. last resort: the bus is wedged — reset the CAN interface (THIS DROPS TORQUE), re-init, retry
    from raiden.robot.controller import reset_can_interface
    print(f"[{tag}][WARN] bus wedged — resetting CAN interface (torque will drop briefly)...", flush=True)
    for iface in ("can_follower_l", "can_follower_r"):
        ok = reset_can_interface(iface)
        print(f"[{tag}] reset {iface}: {'ok' if ok else 'FAILED (need sudo?)'}", flush=True)
    time.sleep(1.0)
    try:
        robot.initialize_robots()
        time.sleep(0.5)
        robot.move_to_home_positions()
        print(f"[{tag}] homed after CAN reset.", flush=True)
        return True
    except Exception as e:
        print(f"[{tag}][WARN] home still failed after CAN reset: {e} — arms may not be home!", flush=True)
        return False


# openpi mapping helpers live in the bridge; reuse them so obs/action layouts stay in one place.
OPENPI_HOME = Path.home() / "robometer_policy_learning" / "third_party" / "dsrl_openpi"
sys.path.insert(0, str(OPENPI_HOME / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(OPENPI_HOME / "deployment"))

MODEL_IMG_SIZE = 224

# raiden camera NAME -> openpi observation key. YAM teleop names its cams
# scene_camera/left_wrist_camera/right_wrist_camera; openpi wants head/left/right.
_CAM_TO_OPENPI = {
    "scene_camera": "observation/image_head",
    "left_wrist_camera": "observation/image_left_wrist",
    "right_wrist_camera": "observation/image_right_wrist",
}


def _raiden_to_openpi_state(r_pos: np.ndarray, l_pos: np.ndarray) -> np.ndarray:
    """raiden per-arm [joints6, grip1] pairs -> openpi 14-D [l6,lg,r6,rg]. (openpi_bridge.py)"""
    return np.concatenate([l_pos[:6], l_pos[6:7], r_pos[:6], r_pos[6:7]]).astype(np.float32)


def _openpi_action_split(action_14d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """openpi 14-D action [l6,lg,r6,rg] -> (left 7-D, right 7-D) raiden motor commands."""
    a = np.asarray(action_14d, dtype=np.float32)
    return a[0:7].copy(), a[7:14].copy()


class Cameras:
    """Background ZED capture (per-camera threads), BGR->RGB, keyed by camera NAME.
    Copied from raiden_rollout.py: synchronous grabs in the control loop starve the motor
    CAN bus, so grabbing runs off-thread and the loop only retrieves the last frame."""

    def __init__(self, cam_serials: dict, fps: int = 30):
        from raiden.cameras.zed import ZedCamera
        self._cams = {}
        for key, serial in cam_serials.items():
            cam = ZedCamera(camera_name=key, serial_number=int(serial), fps=fps)
            cam.open(enable_depth=False)
            self._cams[key] = cam
        self._latest = {}
        self._lock = threading.Lock()
        self._running = True
        self._threads = [threading.Thread(target=self._loop, args=(k, c), daemon=True)
                         for k, c in self._cams.items()]
        for t in self._threads:
            t.start()

    def _loop(self, key, cam):
        import cv2
        while self._running:
            if not cam.grab():
                continue
            frame = cam.get_frame()
            if frame.color is None or frame.color.size == 0:
                continue
            rgb = cv2.cvtColor(frame.color, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest[key] = rgb

    def latest(self) -> dict:
        for _ in range(500):
            with self._lock:
                if all(k in self._latest for k in self._cams):
                    return {k: v.copy() for k, v in self._latest.items()}
            time.sleep(0.01)
        raise RuntimeError(f"cameras produced no frames: have {list(self._latest)}, want {list(self._cams)}")

    def close(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        for cam in self._cams.values():
            cam.close()


class FramePublisher:
    """Echo the frames this runner grabs to the YAM Eval UI so it can show the rollout
    WITHOUT opening the (single-owner) ZEDs itself. Runs on its own thread and POSTs the
    latest JPEG per camera to /api/rollout/frame at a modest rate. Fully best-effort and
    OFF the control path: a slow/failed POST can never gate arm control — it just drops.
    """

    def __init__(self, cams, ui_url: str, hz: float = 6.0, quality: int = 70):
        self._cams = cams
        self._base = ui_url.rstrip("/")
        self._dt = 1.0 / max(1.0, hz)
        self._quality = quality
        self._running = True
        self._ok = None                       # None until first attempt; then bool
        try:
            import requests, cv2  # noqa: F401
            self._requests = requests
            self._cv2 = cv2
            requests.post(f"{self._base}/api/rollout/begin", timeout=2.0)  # reset buffers
        except Exception as e:
            print(f"[relay] disabled ({type(e).__name__}: {e})", flush=True)
            self._running = False
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[relay] streaming frames to {self._base}/api/rollout/frame", flush=True)

    def _loop(self):
        while self._running:
            t0 = time.monotonic()
            try:
                frames = self._cams.latest()
                for name, rgb in frames.items():
                    ok, buf = self._cv2.imencode(
                        ".jpg", self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR),
                        [self._cv2.IMWRITE_JPEG_QUALITY, self._quality])
                    if not ok:
                        continue
                    self._requests.post(f"{self._base}/api/rollout/frame",
                                        params={"camera": name}, data=buf.tobytes(),
                                        headers={"Content-Type": "image/jpeg"}, timeout=2.0)
                if self._ok is not True:
                    self._ok = True
            except Exception:
                if self._ok is not False:      # log the first failure only, then stay quiet
                    print("[relay] frame POST failing (UI not reachable?) — continuing", flush=True)
                    self._ok = False
            s = self._dt - (time.monotonic() - t0)
            if s > 0:
                time.sleep(s)

    def close(self):
        self._running = False


class BimanualProprioReader:
    """Reads BOTH followers' state in one thread (pi05 needs a 14-D bimanual state).
    Off the control thread so proprio/CAN reads aren't gated behind a camera grab/inference."""

    def __init__(self, robot, hz: float = 100.0):
        self._robot = robot
        self._dt = 1.0 / hz
        self._latest = None  # (r_pos(7), l_pos(7))
        self._lock = threading.Lock()
        self._running = True
        self._stopped = threading.Event()   # set when the loop is CONFIRMED off the CAN bus
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _arm7(obs_arm) -> np.ndarray:
        return np.concatenate([obs_arm["joint_pos"], obs_arm["gripper_pos"].reshape(1)]).astype(np.float32)

    def _loop(self):
        # Exception-safe: a transient CAN read failure must NOT kill the thread silently (a dead
        # reader leaves stale proprio) NOR wedge the bus. Always signals _stopped on exit so the
        # teardown can wait until this thread is provably done touching CAN before homing.
        try:
            while self._running:
                t0 = time.monotonic()
                # re-check the flag immediately before the CAN transaction: close() sets _running
                # False, and we must NOT start another get_all_observations() that would collide
                # with the 100Hz homing commands (that collision = "loss communication" → the first
                # home throws → old code reset CAN → arm collapsed).
                if not self._running:
                    break
                try:
                    obs = self._robot.get_all_observations()
                    r, l = self._arm7(obs["follower_r"]), self._arm7(obs["follower_l"])
                    with self._lock:
                        self._latest = (r, l)
                except Exception as e:
                    if self._running:
                        print(f"[proprio][WARN] read failed: {e}", flush=True)
                dt = self._dt - (time.monotonic() - t0)
                if dt > 0:
                    time.sleep(dt)
        finally:
            self._stopped.set()

    def latest(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(500):
            with self._lock:
                if self._latest is not None:
                    return self._latest[0].copy(), self._latest[1].copy()
            time.sleep(0.01)
        raise RuntimeError("no proprio data yet")

    def close(self):
        """Stop the reader and WAIT until it is provably off the CAN bus before returning, so the
        homing move that follows has the bus to itself. Overrides any thin close elsewhere."""
        self._running = False
        # wait on the loop's own 'stopped' signal (set in its finally), longer than one read period,
        # so an in-flight get_all_observations() finishes before we return and homing begins.
        self._stopped.wait(timeout=3.0)
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass


class Pi05Client:
    """Wraps the openpi websocket client + ActionChunkBroker. One infer() per tick returns a
    single 14-D action (the broker re-queries the server every action_horizon ticks)."""

    def __init__(self, host: str, port: int, action_horizon: int, prompt: str):
        from openpi_client import action_chunk_broker
        from openpi_client import websocket_client_policy as _ws
        print(f"[pi05] connecting to server {host}:{port}")
        ws = _ws.WebsocketClientPolicy(host=host, port=port)
        print(f"[pi05] server metadata: {ws.get_server_metadata()}")
        self._broker = action_chunk_broker.ActionChunkBroker(policy=ws, action_horizon=action_horizon)
        self._prompt = prompt

    def build_obs(self, frames_by_camname: dict, r_pos: np.ndarray, l_pos: np.ndarray) -> dict:
        import cv2
        obs = {}
        for cam_name, rgb in frames_by_camname.items():
            key = _CAM_TO_OPENPI.get(cam_name)
            if key is None:
                continue
            obs[key] = cv2.resize(rgb, (MODEL_IMG_SIZE, MODEL_IMG_SIZE))  # RGB uint8 (Cameras already BGR->RGB)
        obs["observation/state"] = _raiden_to_openpi_state(r_pos, l_pos)
        if self._prompt:
            obs["prompt"] = self._prompt
        return obs

    def predict(self, obs: dict) -> np.ndarray:
        """Return one 14-D absolute-joint action (openpi order [l6,lg,r6,rg])."""
        return np.asarray(self._broker.infer(obs)["actions"], dtype=np.float32)

    def reset(self):
        self._broker.reset()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost", help="openpi serve_policy host")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="pick up the banana and put it into the box")
    ap.add_argument("--action-horizon", type=int, default=25,
                    help="actions executed per server inference (broker re-queries after)")
    ap.add_argument("--control-hz", type=float, default=15.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-joint-delta", type=float, default=0.2,
                    help="per-arm command-to-command jump guard (rad/tick); exceed => e-stop")
    # camera serials keyed by raiden camera name (scene_camera/left_wrist_camera/right_wrist_camera)
    ap.add_argument("--cam-serials", nargs="+", required=False, default=[],
                    help="name=serial pairs, e.g. scene_camera=12345 left_wrist_camera=67890 ...")
    ap.add_argument("--dry-run", action="store_true",
                    help="inference on live hardware, NEVER command the arm")
    ap.add_argument("--selftest", action="store_true",
                    help="no hardware: send a synthetic obs to the server, assert a 14-D action")
    ap.add_argument("--relay-url", default="",
                    help="YAM Eval UI base URL (e.g. http://localhost:8081) to echo camera "
                         "frames to for the live rollout view + replay; empty disables")
    return ap.parse_args()


def selftest(args):
    """No hardware — confirm the server round-trips a synthetic YAM obs to a (14,) action."""
    client = Pi05Client(args.host, args.port, args.action_horizon, args.prompt)
    def img():
        return np.random.randint(0, 256, (MODEL_IMG_SIZE, MODEL_IMG_SIZE, 3), dtype=np.uint8)
    frames = {"scene_camera": img(), "left_wrist_camera": img(), "right_wrist_camera": img()}
    obs = client.build_obs(frames, np.zeros(7, np.float32), np.zeros(7, np.float32))
    a = client.predict(obs)
    assert a.shape == (14,), f"expected (14,) action, got {a.shape}"
    assert np.isfinite(a).all(), "action has non-finite values"
    lft, rgt = _openpi_action_split(a)
    print(f"[pi05][selftest] OK — action (14,): left j0={lft[0]:.3f} grip={lft[6]:.3f} | "
          f"right j0={rgt[0]:.3f} grip={rgt[6]:.3f}")


def main():
    args = parse_args()

    if args.selftest:
        selftest(args)
        return

    if not args.cam_serials:
        raise SystemExit("--cam-serials required for a live/dry run (e.g. scene_camera=<serial> ...)")
    cam_serials = dict(kv.split("=", 1) for kv in args.cam_serials)
    for needed in ("scene_camera", "left_wrist_camera", "right_wrist_camera"):
        if needed not in cam_serials:
            raise SystemExit(f"pi05 needs 3 cameras; missing {needed!r} in --cam-serials")

    from raiden.robot.controller import RobotController
    from raiden.robot.footpedal import try_open_footpedal

    estop = threading.Event()

    # Connect to the server (also warms the client) BEFORE motors init. Unlike the flow
    # runner there's no local CUDA warmup — inference is remote (the server owns the GPU) —
    # but we still init motors last to keep the CAN thread unstarved during setup.
    client = Pi05Client(args.host, args.port, args.action_horizon, args.prompt)

    print("[pi05] opening cameras (background capture threads)...")
    cams = Cameras(cam_serials, fps=30)
    cams.latest()  # block until all 3 cameras have produced a frame

    # echo frames to the UI (off the control path) so the rollout is watchable + replayable
    relay = FramePublisher(cams, args.relay_url) if args.relay_url else None

    print("[pi05] initializing robots (both followers, no leaders)...")
    robot = RobotController(use_right_leader=False, use_left_leader=False)
    robot.initialize_robots()
    robot.move_to_home_positions()
    time.sleep(0.5)
    print("[pi05] robot init OK")

    proprio = BimanualProprioReader(robot, hz=100.0)
    time.sleep(0.3)

    # COOPERATIVE stop, installed NOW (right after robot init) so it covers the whole run —
    # including the dry-run branch and the init→driven-loop gap. Previously the handler was only
    # installed inside the driven loop, so a Stop pressed earlier hit Python's DEFAULT SIGINT →
    # KeyboardInterrupt → process died with the arm ENERGIZED BUT NEVER HOMED. Now SIGINT only sets
    # a flag; every exit path flows through _home_and_close() below, which homes the arm.
    stop_requested = threading.Event()

    def _on_sigint(_sig, _frame):
        if not stop_requested.is_set():
            print("\n[pi05] stop requested — finishing tick, then homing.", flush=True)
        stop_requested.set()
    signal.signal(signal.SIGINT, _on_sigint)

    _cleaned = threading.Event()   # guard so homing/cleanup runs exactly once

    def _home_and_close():
        """Guaranteed teardown on ANY exit after robot init: drain proprio/cams off the CAN bus,
        settle, then home with CAN-reset recovery (unless e-stopped), then zero torque. Idempotent."""
        if _cleaned.is_set():
            return
        _cleaned.set()
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # a 2nd Ctrl-C must not interrupt homing
        try:
            if relay is not None:
                relay.close()
        except Exception as e:
            print(f"[pi05][WARN] relay close: {e}", flush=True)
        try:
            proprio.close()                    # WAITS until the reader is provably off the CAN bus
            cams.close()
            time.sleep(0.8)                    # settle: let the motor bus quiesce after the rollout's
                                               # command stream stops, BEFORE the 100Hz homing move —
                                               # firing home while the bus is still hot trips loss-comm
        except Exception as e:
            print(f"[pi05][WARN] proprio/cams close: {e}", flush=True)
        if not estop.is_set():
            print("[pi05] returning home...", flush=True)
            try:
                home_with_recovery(robot, tag="pi05")
            except Exception as e:
                print(f"[pi05][WARN] home failed: {e}", flush=True)
        try:
            robot.close()   # zero torque (NOT shutdown(): it re-homes at 100Hz)
        except Exception as e:
            print(f"[pi05][WARN] robot close: {e}", flush=True)

    # dry-run: inference on live hardware, never command the arm.
    if args.dry_run:
        n = args.max_steps
        print(f"[pi05][DRY-RUN] {n} ticks — inference on live hardware, NO arm motion")
        tick_dt = 1.0 / args.control_hz
        try:
            t_start = time.monotonic()
            for i in range(n):
                if stop_requested.is_set():
                    print("[pi05][DRY-RUN] stop — exiting cleanly"); break
                s = (t_start + i * tick_dt) - time.monotonic()
                if s > 0:
                    time.sleep(s)
                r_pos, l_pos = proprio.latest()
                t0 = time.monotonic()
                a = client.predict(client.build_obs(cams.latest(), r_pos, l_pos))
                infer_ms = (time.monotonic() - t0) * 1e3
                lft, rgt = _openpi_action_split(a)
                print(f"  [{i}] infer={infer_ms:5.1f}ms | L j0={lft[0]:.3f} grip={lft[6]:.3f} "
                      f"| R j0={rgt[0]:.3f} grip={rgt[6]:.3f}", flush=True)
        finally:
            # Home on exit even in dry-run: the arm sits at the init home pose, but a Stop pressed
            # during init/dry-run should still leave it verified-home (and _home_and_close is the
            # single teardown path). home_with_recovery is a safe no-op if already home.
            _home_and_close()
        print("[pi05][DRY-RUN] done.")
        return

    pedal = try_open_footpedal()
    if pedal is not None:
        pedal.on_press(lambda _c: (print("\n[pi05] FOOTPEDAL E-STOP"), estop.set(), robot.emergency_stop()))
        pedal.start()
        print("[pi05] footpedal e-stop armed")

    # (cooperative SIGINT handler + stop_requested already installed right after robot init above,
    # so a Stop pressed any time from init onward flows through _home_and_close and homes the arm.)

    tick_dt = 1.0 / args.control_hz
    prev_l = prev_r = None  # per-arm last commanded (6 joints) for the jump guard
    print(f"[pi05] driving | ticks={args.max_steps} control_hz={args.control_hz} "
          f"action_horizon={args.action_horizon} max_joint_delta={args.max_joint_delta}")
    try:
        t_start = time.monotonic()
        for i in range(args.max_steps):
            if estop.is_set():
                print("[pi05] e-stop set — aborting loop"); break
            if stop_requested.is_set():
                print("[pi05] stop — exiting loop cleanly"); break
            sleep = (t_start + i * tick_dt) - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

            r_pos, l_pos = proprio.latest()
            action = client.predict(client.build_obs(cams.latest(), r_pos, l_pos))  # (14,)
            l_cmd, r_cmd = _openpi_action_split(action)

            # Per-arm command-to-command jump guard (same rationale as raiden_rollout.py:
            # guard consecutive commands, not command-vs-measured, to avoid false trips).
            if prev_l is None:
                prev_l, prev_r = l_cmd[:6].copy(), r_cmd[:6].copy()
            for name, cmd, prev in (("left", l_cmd, prev_l), ("right", r_cmd, prev_r)):
                step = float(np.abs(cmd[:6] - prev).max())
                if step > args.max_joint_delta:
                    j = int(np.abs(cmd[:6] - prev).argmax())
                    print(f"\n[pi05][SAFETY] {name} joint {j} command jump {step:.3f} > "
                          f"{args.max_joint_delta} rad/tick — E-STOP")
                    estop.set(); robot.emergency_stop(); break
            if estop.is_set():
                break
            # stop may have arrived during predict/sleep — don't issue one more CAN command.
            if stop_requested.is_set():
                print("[pi05] stop — exiting loop cleanly"); break

            robot.follower_l.command_joint_pos(l_cmd)
            robot.follower_r.command_joint_pos(r_cmd)
            prev_l, prev_r = l_cmd[:6].copy(), r_cmd[:6].copy()
            if i % 30 == 0:
                print(f"  tick {i}/{args.max_steps} L j0={l_cmd[0]:.3f} g={l_cmd[6]:.2f} "
                      f"R j0={r_cmd[0]:.3f} g={r_cmd[6]:.2f}")
    except KeyboardInterrupt:
        # Fallback only — the flag handler above normally prevents this from ever firing.
        print("\n[pi05] Ctrl-C — stopping and homing.")
    finally:
        # Single teardown path: drain proprio/cams off the CAN bus, settle, home with CAN-reset
        # recovery (unless e-stopped, where emergency_stop already owns shutdown), then zero torque.
        if estop.is_set():
            print("[pi05] e-stopped — emergency_stop owns shutdown.", flush=True)
            _cleaned.set()   # skip _home_and_close's homing; e-stop path already handled the arm
        _home_and_close()
        print("[pi05] done.")


if __name__ == "__main__":
    main()