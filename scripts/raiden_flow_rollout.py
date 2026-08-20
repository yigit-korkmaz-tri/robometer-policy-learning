#!/usr/bin/env python3
"""Roll out a robometer-trained policy (flow / flow_mile / dp / bc) on the YAM arm.

Robot-host counterpart of scripts/serve_robometer_policy.py, and the direct analogue of
scripts/raiden_pi05_rollout.py: same proven hardware scaffold (background ZED capture, threaded
proprio, footpedal e-stop, command-to-command jump guard, homing on exit) and the same websocket
protocol (openpi-client), but pointed at a policy trained in THIS repo instead of at openpi.

Two processes:

    # 1) serve the checkpoint (GPU host, robometer venv)
    uv run python scripts/serve_robometer_policy.py --load-dir <run dir> --port 8000

    # 2) drive the arm (robot host, raiden venv)
    python -m robometer_policy_learning.scripts.raiden_flow_rollout \\
        --host <gpu-host> --port 8000 \\
        --cam-serials scene_camera=<serial> left_wrist_camera=<serial> right_wrist_camera=<serial> \\
        --max-steps 400 [--dry-run | --selftest]

Everything about the observation/action contract is taken from the SERVER's handshake metadata
(written at training time), not restated here, so the client cannot drift from training:

  * which cameras map to which observation keys (the dataset's conversion camera_map);
  * the image size + resize filter frames are downscaled with (shared resize_image);
  * the action horizon: --action-horizon defaults to the training-time n_action_steps, and the
    ActionChunkBroker therefore replans on exactly the cadence the policy was trained with;
  * action_mode: "delta" actions have the current state added back before commanding.

Low-dim normalization deliberately happens SERVER-side (it needs the training buffer's statistics),
so this client sends raw proprio -- exactly what the robot measures.

Layouts (identical to raiden_pi05_rollout.py / the LeRobot YAM conversion):
  raiden proprio per arm: [joints(6), gripper(1)]
  policy state (14-D):    [l_joints(6), l_grip(1), r_joints(6), r_grip(1)]
  policy action (14-D):   [l_joints(6), l_grip(1), r_joints(6), r_grip(1)]
  -> command follower_l with action[0:7], follower_r with action[7:14].

Safety: --dry-run (inference on live hardware, never commands the arm), --selftest (no hardware),
footpedal e-stop, per-arm command jump guard, homing on exit.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

# The hardware scaffold is shared verbatim with the pi0.5 runner; import it rather than forking it,
# so a fix to homing / capture / proprio benefits both runners.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from raiden_pi05_rollout import (  # noqa: E402
    BimanualProprioReader,
    Cameras,
    FramePublisher,
    home_with_recovery,
)

# openpi-client is a small standalone package (websocket + msgpack-numpy + ActionChunkBroker); the
# server speaks its protocol, so the robot side needs nothing from the full openpi.
OPENPI_HOME = _THIS.parent.parent / "third_party" / "dsrl_openpi"
sys.path.insert(0, str(OPENPI_HOME / "packages" / "openpi-client" / "src"))


def _resize(image: np.ndarray, size):
    """Resize a frame the way the training dataset was converted.

    Prefers the shared helper so there is literally one implementation; falls back to an inline
    INTER_AREA resize if robometer_policy_learning is not importable in the raiden venv.
    """
    try:
        from robometer_policy_learning.utils.policy_serving import resize_image

        return resize_image(image, size)
    except Exception:
        import cv2

        if size is None or image.shape[:2] == (size, size):
            return image
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _raiden_to_policy_state(r_pos: np.ndarray, l_pos: np.ndarray) -> np.ndarray:
    """raiden per-arm [joints6, grip1] pairs -> 14-D [l6,lg,r6,rg] (the LeRobot state layout)."""
    return np.concatenate([l_pos[:6], l_pos[6:7], r_pos[:6], r_pos[6:7]]).astype(np.float32)


def _policy_action_split(action_14d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """14-D action [l6,lg,r6,rg] -> (left 7-D, right 7-D) raiden motor commands."""
    a = np.asarray(action_14d, dtype=np.float32)
    return a[0:7].copy(), a[7:14].copy()


class RobometerClient:
    """openpi websocket client + ActionChunkBroker, configured from the server's handshake."""

    def __init__(self, host: str, port: int, action_horizon=None, prompt: str = ""):
        from openpi_client import action_chunk_broker
        from openpi_client import websocket_client_policy as _ws

        print(f"[flow] connecting to server {host}:{port}")
        self._ws = _ws.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self._ws.get_server_metadata() or {}

        self.image_keys = list(self.metadata.get("image_keys") or [])
        self.lowdim_keys = list(self.metadata.get("lowdim_keys") or ["state"])
        self.image_size = self.metadata.get("image_size")
        self.action_mode = self.metadata.get("action_mode", "absolute")
        self.action_dim = int(self.metadata.get("action_dim") or 14)
        chunk_size = int(self.metadata.get("chunk_size") or 1)
        trained_horizon = int(self.metadata.get("n_action_steps") or 1)

        # obs key -> robot camera name (metadata stores camera -> obs key).
        cam_map = self.metadata.get("camera_map") or {}
        self._obs_key_to_cam = {v: k for k, v in cam_map.items()}

        if action_horizon is None:
            action_horizon = trained_horizon
        elif action_horizon != trained_horizon:
            print(
                f"[flow][WARN] --action-horizon {action_horizon} != the training-time n_action_steps "
                f"{trained_horizon}: the policy will replan on a different cadence than it was trained for."
            )
        if action_horizon > chunk_size:
            # Fail here rather than mid-rollout: the broker would IndexError past the chunk end.
            raise SystemExit(
                f"--action-horizon {action_horizon} exceeds the policy's chunk_size {chunk_size}."
            )
        self.action_horizon = int(action_horizon)

        self._broker = action_chunk_broker.ActionChunkBroker(
            policy=self._ws, action_horizon=self.action_horizon
        )
        self._prompt = prompt or self.metadata.get("prompt", "")
        print(
            f"[flow] server contract: images={self.image_keys} lowdim={self.lowdim_keys} "
            f"image_size={self.image_size} action_dim={self.action_dim} chunk_size={chunk_size} "
            f"action_horizon={self.action_horizon} action_mode={self.action_mode}"
        )
        if self._prompt:
            print(f"[flow] prompt: {self._prompt!r}")
        if self.action_mode != "absolute":
            print(f"[flow][WARN] action_mode={self.action_mode!r}: adding current state to each action.")

    def build_obs(self, frames_by_camname: dict, r_pos: np.ndarray, l_pos: np.ndarray) -> dict:
        """Raw robot data -> the observation keys the policy was trained on.

        Frames are downscaled here (not sent at full 720p) purely for bandwidth; the server resizes
        again if needed and the filter is shared, so the result is identical either way. Proprio is
        sent RAW -- the server owns the training-time z-scoring.
        """
        obs = {}
        for obs_key in self.image_keys:
            cam_name = self._obs_key_to_cam.get(obs_key, obs_key)
            frame = frames_by_camname.get(cam_name)
            if frame is None:
                raise KeyError(
                    f"camera {cam_name!r} (for observation key {obs_key!r}) produced no frame; "
                    f"have {sorted(frames_by_camname)}"
                )
            obs[obs_key] = _resize(frame, self.image_size)  # RGB uint8 (Cameras already BGR->RGB)

        state = _raiden_to_policy_state(r_pos, l_pos)
        for key in self.lowdim_keys:
            obs[key] = state
        if self._prompt:
            obs["prompt"] = self._prompt
        return obs

    def predict(self, obs: dict, state: np.ndarray = None) -> np.ndarray:
        """Return one action for this tick, in the robot's action space."""
        action = np.asarray(self._broker.infer(obs)["actions"], dtype=np.float32).reshape(-1)
        if self.action_mode == "delta":
            if state is None:
                raise ValueError("delta action_mode requires the current state")
            action = action + np.asarray(state, dtype=np.float32).reshape(-1)
        return action

    def reset(self):
        self._broker.reset()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost", help="serve_robometer_policy.py host")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="", help="override the server's task prompt")
    ap.add_argument("--action-horizon", type=int, default=None,
                    help="actions executed per server inference (default: training n_action_steps)")
    ap.add_argument("--control-hz", type=float, default=15.0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--max-joint-delta", type=float, default=0.2,
                    help="per-arm command-to-command jump guard (rad/tick); exceed => e-stop")
    ap.add_argument("--cam-serials", nargs="+", default=[],
                    help="name=serial pairs, e.g. scene_camera=12345 left_wrist_camera=67890 ...")
    ap.add_argument("--dry-run", action="store_true",
                    help="inference on live hardware, NEVER command the arm")
    ap.add_argument("--selftest", action="store_true",
                    help="no hardware: send a synthetic obs to the server, assert a valid action")
    ap.add_argument("--relay-url", default="",
                    help="YAM Eval UI base URL to echo camera frames to; empty disables")
    return ap.parse_args()


def selftest(args):
    """No hardware — confirm the server round-trips a synthetic obs to a finite action."""
    client = RobometerClient(args.host, args.port, args.action_horizon, args.prompt)
    size = client.image_size or 224
    frames = {
        client._obs_key_to_cam.get(k, k): np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        for k in client.image_keys
    }
    zeros = np.zeros(7, np.float32)
    obs = client.build_obs(frames, zeros, zeros)
    a = client.predict(obs, state=_raiden_to_policy_state(zeros, zeros))
    assert a.shape == (client.action_dim,), f"expected ({client.action_dim},) action, got {a.shape}"
    assert np.isfinite(a).all(), "action has non-finite values"
    lft, rgt = _policy_action_split(a)
    print(f"[flow][selftest] OK — action ({client.action_dim},): left j0={lft[0]:.3f} grip={lft[6]:.3f} | "
          f"right j0={rgt[0]:.3f} grip={rgt[6]:.3f}")
    # Exercise the broker's re-query boundary: horizon+1 calls must span two server inferences.
    for _ in range(client.action_horizon):
        client.predict(obs, state=_raiden_to_policy_state(zeros, zeros))
    print(f"[flow][selftest] OK — broker re-queried after {client.action_horizon} actions")


def main():
    args = parse_args()

    if args.selftest:
        selftest(args)
        return

    if not args.cam_serials:
        raise SystemExit("--cam-serials required for a live/dry run (e.g. scene_camera=<serial> ...)")
    cam_serials = dict(kv.split("=", 1) for kv in args.cam_serials)

    from raiden.robot.controller import RobotController
    from raiden.robot.footpedal import try_open_footpedal

    estop = threading.Event()

    # Connect first (also validates the contract) so a misconfigured run fails before motors init.
    client = RobometerClient(args.host, args.port, args.action_horizon, args.prompt)
    for obs_key in client.image_keys:
        cam_name = client._obs_key_to_cam.get(obs_key, obs_key)
        if cam_name not in cam_serials:
            raise SystemExit(
                f"policy needs camera {cam_name!r} (observation key {obs_key!r}); missing from --cam-serials"
            )

    print("[flow] opening cameras (background capture threads)...")
    cams = Cameras(cam_serials, fps=30)
    cams.latest()  # block until all cameras have produced a frame

    relay = FramePublisher(cams, args.relay_url) if args.relay_url else None

    print("[flow] initializing robots (both followers, no leaders)...")
    robot = RobotController(use_right_leader=False, use_left_leader=False)
    robot.initialize_robots()
    robot.move_to_home_positions()
    time.sleep(0.5)
    print("[flow] robot init OK")

    proprio = BimanualProprioReader(robot, hz=100.0)
    time.sleep(0.3)

    # COOPERATIVE stop installed right after robot init, so every exit path homes the arm.
    stop_requested = threading.Event()

    def _on_sigint(_sig, _frame):
        if not stop_requested.is_set():
            print("\n[flow] stop requested — finishing tick, then homing.", flush=True)
        stop_requested.set()

    signal.signal(signal.SIGINT, _on_sigint)

    _cleaned = threading.Event()

    def _home_and_close():
        """Guaranteed teardown: drain proprio/cams off the CAN bus, settle, home, zero torque."""
        if _cleaned.is_set():
            return
        _cleaned.set()
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a 2nd Ctrl-C must not interrupt homing
        try:
            if relay is not None:
                relay.close()
        except Exception as e:
            print(f"[flow][WARN] relay close: {e}", flush=True)
        try:
            proprio.close()  # WAITS until the reader is provably off the CAN bus
            cams.close()
            time.sleep(0.8)  # let the motor bus quiesce before the 100Hz homing move
        except Exception as e:
            print(f"[flow][WARN] proprio/cams close: {e}", flush=True)
        if not estop.is_set():
            print("[flow] returning home...", flush=True)
            try:
                home_with_recovery(robot, tag="flow")
            except Exception as e:
                print(f"[flow][WARN] home failed: {e}", flush=True)
        try:
            robot.close()  # zero torque (NOT shutdown(): it re-homes at 100Hz)
        except Exception as e:
            print(f"[flow][WARN] robot close: {e}", flush=True)

    if args.dry_run:
        n = args.max_steps
        print(f"[flow][DRY-RUN] {n} ticks — inference on live hardware, NO arm motion")
        tick_dt = 1.0 / args.control_hz
        try:
            t_start = time.monotonic()
            for i in range(n):
                if stop_requested.is_set():
                    print("[flow][DRY-RUN] stop — exiting cleanly")
                    break
                s = (t_start + i * tick_dt) - time.monotonic()
                if s > 0:
                    time.sleep(s)
                r_pos, l_pos = proprio.latest()
                t0 = time.monotonic()
                obs = client.build_obs(cams.latest(), r_pos, l_pos)
                a = client.predict(obs, state=_raiden_to_policy_state(r_pos, l_pos))
                infer_ms = (time.monotonic() - t0) * 1e3
                lft, rgt = _policy_action_split(a)
                print(f"  [{i}] infer={infer_ms:5.1f}ms | L j0={lft[0]:.3f} grip={lft[6]:.3f} "
                      f"| R j0={rgt[0]:.3f} grip={rgt[6]:.3f}", flush=True)
        finally:
            _home_and_close()
        print("[flow][DRY-RUN] done.")
        return

    pedal = try_open_footpedal()
    if pedal is not None:
        pedal.on_press(lambda _c: (print("\n[flow] FOOTPEDAL E-STOP"), estop.set(), robot.emergency_stop()))
        pedal.start()
        print("[flow] footpedal e-stop armed")

    tick_dt = 1.0 / args.control_hz
    prev_l = prev_r = None  # per-arm last commanded (6 joints) for the jump guard
    print(f"[flow] driving | ticks={args.max_steps} control_hz={args.control_hz} "
          f"action_horizon={client.action_horizon} max_joint_delta={args.max_joint_delta}")
    try:
        t_start = time.monotonic()
        for i in range(args.max_steps):
            if estop.is_set():
                print("[flow] e-stop set — aborting loop")
                break
            if stop_requested.is_set():
                print("[flow] stop — exiting loop cleanly")
                break
            sleep = (t_start + i * tick_dt) - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

            r_pos, l_pos = proprio.latest()
            obs = client.build_obs(cams.latest(), r_pos, l_pos)
            action = client.predict(obs, state=_raiden_to_policy_state(r_pos, l_pos))
            l_cmd, r_cmd = _policy_action_split(action)

            # Per-arm command-to-command jump guard (guard consecutive commands, not
            # command-vs-measured, to avoid false trips).
            if prev_l is None:
                prev_l, prev_r = l_cmd[:6].copy(), r_cmd[:6].copy()
            for name, cmd, prev in (("left", l_cmd, prev_l), ("right", r_cmd, prev_r)):
                step = float(np.abs(cmd[:6] - prev).max())
                if step > args.max_joint_delta:
                    j = int(np.abs(cmd[:6] - prev).argmax())
                    print(f"\n[flow][SAFETY] {name} joint {j} command jump {step:.3f} > "
                          f"{args.max_joint_delta} rad/tick — E-STOP")
                    estop.set()
                    robot.emergency_stop()
                    break
            if estop.is_set():
                break
            # stop may have arrived during predict/sleep — don't issue one more CAN command.
            if stop_requested.is_set():
                print("[flow] stop — exiting loop cleanly")
                break

            robot.follower_l.command_joint_pos(l_cmd)
            robot.follower_r.command_joint_pos(r_cmd)
            prev_l, prev_r = l_cmd[:6].copy(), r_cmd[:6].copy()
            if i % 30 == 0:
                print(f"  tick {i}/{args.max_steps} L j0={l_cmd[0]:.3f} g={l_cmd[6]:.2f} "
                      f"R j0={r_cmd[0]:.3f} g={r_cmd[6]:.2f}")
    except KeyboardInterrupt:
        print("\n[flow] Ctrl-C — stopping and homing.")
    finally:
        if estop.is_set():
            print("[flow] e-stopped — emergency_stop owns shutdown.", flush=True)
            _cleaned.set()
        _home_and_close()
        print("[flow] done.")


if __name__ == "__main__":
    main()
