"""Human-in-the-loop (HITL) utilities for pi0 / pi0.5 policies on LIBERO envs.

The robomimic HITL worker in :mod:`robometer_policy_learning.utils.hitl_utils` is built around a
torch ``BaseActor`` (``actor.act``) and low-dim-normalized robomimic observations. pi0 / pi0.5 are
different on both counts:

  * inference is JAX/openpi-based and produces a whole ACTION CHUNK per query
    (``Pi0Wrapper.infer(obs, noise=None) -> {"actions": [horizon, action_dim]}``), executed with a
    receding-horizon window (``action_exec_len``);
  * LIBERO observations are already in pi0 format (``observation/image``, ``observation/wrist_image``,
    ``observation/state``, ``prompt``) and pi0 normalizes internally — there is no low-dim z-scoring.

So this module provides a dedicated :class:`Pi0LiberoHitlWorker` for collecting HITL rollouts with a
pi0 / pi0.5 policy under real keyboard / SpaceMouse teleoperation, reusing the device toggle, robosuite
env discovery, and control-mode helpers from :mod:`hitl_utils`.

LIBERO is robosuite-based, so teleop uses robosuite's ``input2action`` (delta OSC end-effector
commands) exactly like the robomimic worker; LIBERO's default controller is delta-mode OSC_POSE, so
the teleop deltas map straight onto the 7-dim env action space with no absolute-pose conversion.
"""

import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from robometer_policy_learning.utils.hitl_utils import (
    TakeoverToggle,
    _find_robosuite_env,
    _success_from_info,
)

# Intervention-label convention (matches the HITL buffers / MILE): 0 = autonomous policy, 1 = human.
ROLLOUT_LABEL, INTERVENTION_LABEL = 0, 1

# pi0 / pi0.5 LIBERO observation keys we feed to the policy AND store (raw; pi0 normalizes internally).
PI0_IMAGE_KEY = "observation/image"
PI0_WRIST_KEY = "observation/wrist_image"
PI0_STATE_KEY = "observation/state"
PI0_OBS_KEYS = (PI0_IMAGE_KEY, PI0_WRIST_KEY, PI0_STATE_KEY)


def _extract0(batched):
    """Extract env 0 from a vectorized obs dict / array (n_envs=1)."""
    if isinstance(batched, dict):
        return {k: (v[0] if isinstance(v, (np.ndarray, list, tuple)) else v) for k, v in batched.items()}
    return batched[0]


def _scalar(x):
    return np.asarray(x).reshape(-1)[0]


def _prompt_str(obs) -> str:
    """Pull a single prompt/language-instruction string out of a (possibly vectorized) obs dict."""
    p = obs.get("prompt") if isinstance(obs, dict) else None
    if isinstance(p, (list, tuple, np.ndarray)):
        p = p[0] if len(p) else None
    return str(p) if p is not None else "unknown task"


class Pi0LiberoHitlWorker:
    """Runs HITL rollouts on a single (n_envs=1) LIBERO env with a pi0 / pi0.5 policy.

    Autonomous control queries the pi0 policy for an action chunk and executes ``action_exec_len``
    actions open-loop (receding horizon) before replanning. A human can take over at any step with
    the takeover key (keyboard or SpaceMouse); while the human is in control the pi0 chunk is
    discarded and per-step teleop actions drive the env, and the policy replans from the corrected
    state once control is released.

    Each kept episode is appended to ``self.collected_episodes`` as a list of per-step transition
    dicts (raw pi0 obs, executed env action, reward, done/truncated, intervention label, prompt),
    ready to be written to an HDF5 dataset. Keyboard 'q' aborts (discards) the current episode; ESC
    raises KeyboardInterrupt.
    """

    def __init__(
        self,
        env,
        pi0_wrapper,
        action_dim: int,
        *,
        action_exec_len: int = 20,
        store_only_human: bool = False,
        rollout_pool_size: int = 0,
        enable_render: bool = True,
        teleop_device: str = "keyboard",
        takeover_key: str = "tab",
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        render_size: int = 512,
        camera: str = "agentview",
        wrist_camera: str = "robot0_eye_in_hand",
        show_wrist: bool = True,
        cmd_eps: float = 1e-6,
        record_video: bool = False,
        video_dir: str = None,
        video_fps: int = 20,
    ):
        self.env = env
        self.pi0 = pi0_wrapper
        self.action_dim = int(action_dim)
        self.action_exec_len = max(1, int(action_exec_len))
        self.store_only_human = bool(store_only_human)
        # Flow-MILE: if >0, precompute this many rollout-policy action chunks (a frozen baseline pool)
        # ONLY at HUMAN-intervened states -- that is the only place the pool feeds the Flow-MILE loss
        # (label-1 rows' observed_probs baseline; label-0 rows use the logged robot score, label-2 is
        # masked out). Policy/other frames are zero-filled so the stored array stays fixed-shape.
        self.rollout_pool_size = max(0, int(rollout_pool_size))
        # Horizon used to shape the zero-filled non-human baseline pools. Prefer the policy's
        # advertised `action_horizon`; if it's missing/None we resolve it lazily at collection
        # time from an actual sampled chunk (see `_resolve_rollout_horizon`).
        self._rollout_horizon = getattr(self.pi0, "action_horizon", None)
        self.cmd_eps = float(cmd_eps)
        self.enable_render = bool(enable_render)

        # Debug video recording (executed-step frames with the POLICY/HUMAN overlay).
        self.record_video = bool(record_video)
        self.video_dir = video_dir
        self.video_fps = int(video_fps)
        self._record_frames = None
        if self.record_video and self.video_dir:
            os.makedirs(self.video_dir, exist_ok=True)

        # Reach the underlying robosuite env (LIBERO is robosuite-based) for teleop + rendering.
        self.base_env = _find_robosuite_env(env)
        self.robot = self.base_env.robots[0]
        self.controller = self.robot.controller
        if bool(getattr(self.controller, "use_delta", True)) is False:
            raise NotImplementedError(
                "Pi0LiberoHitlWorker only supports delta-mode controllers (LIBERO default OSC_POSE); "
                f"got an absolute-pose controller ({getattr(self.controller, 'name', '?')})."
            )

        # Kept episodes accumulate here (each is a list of per-step transition dicts).
        self.collected_episodes: List[List[Dict]] = []

        # ---- Rendering / teleop window setup ----
        self.teleop_device = str(teleop_device).lower()
        if self.teleop_device in ("space_mouse", "3dmouse"):
            self.teleop_device = "spacemouse"
        self.render_size = int(render_size)
        self.camera = camera
        self.wrist_camera = wrist_camera
        self.show_wrist = bool(show_wrist) and wrist_camera != camera
        if self.teleop_device == "spacemouse":
            controls = "move/twist puck: move, left btn: grip, right btn: reset"
        else:
            controls = "wasd/rf+zx/tg/cv: move, space: grip, q: reset"
        self.window = f"HITL pi0 ({takeover_key}: take/release, {controls}, ESC: quit)"
        # NOTE: cv2 must be imported before torch (the caller does this), otherwise cv2's HighGUI
        # imshow/waitKey deadlocks against the pynput keyboard listener.

        # ---- Teleop device (keyboard or 3D SpaceMouse) + takeover toggle ----
        from robosuite.utils.input_utils import input2action

        if self.teleop_device == "keyboard":
            from robosuite.devices import Keyboard

            self._device = Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        elif self.teleop_device == "spacemouse":
            from robosuite.devices import SpaceMouse

            self._device = SpaceMouse(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        else:
            raise ValueError(f"teleop_device must be 'keyboard' or 'spacemouse', got {teleop_device!r}.")
        self._input2action = input2action
        self.toggle = TakeoverToggle(takeover_key)
        logger.info(f"HITL teleop device: {self.teleop_device} (takeover key: '{takeover_key}')")

    # -----------------------------------------------------------------------------------------
    def close(self):
        if self.toggle is not None:
            self.toggle.stop()
        dev = getattr(self, "_device", None)
        if dev is not None:
            # robosuite's SpaceMouse runs a daemon thread doing a blocking ``device.read(13)`` with no
            # stop flag; closing the HID handle below makes that read raise ``OSError: read error``
            # inside the thread. Disable the device and install a threading excepthook that swallows
            # exactly that reader thread's OSError (every other thread's exceptions pass through), so
            # shutdown is clean while the HID handle is still released for a later reopen.
            setattr(dev, "_enabled", False)
            reader = getattr(dev, "thread", None)
            if reader is not None:
                _orig_hook = threading.excepthook

                def _quiet_reader_hook(args, _reader=reader, _orig=_orig_hook):
                    if args.thread is _reader and isinstance(args.exc_value, OSError):
                        return  # harmless: HID handle closed out from under the blocking read
                    _orig(args)

                threading.excepthook = _quiet_reader_hook
            hid = getattr(dev, "device", None)
            if hid is not None and hasattr(hid, "close"):
                try:
                    hid.close()
                except Exception:  # noqa: BLE001
                    pass
            self._device = None
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass

    def _device_grasp(self):
        """Gripper state of the teleop device, normalized across Keyboard (``grasp``) / SpaceMouse."""
        dev = self._device
        g = getattr(dev, "grasp", None)
        if g is None:
            g = getattr(dev, "control_gripper", 0)
        return float(g)

    def _fit_action(self, a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.shape[0] == self.action_dim:
            return a
        fixed = np.zeros(self.action_dim, dtype=np.float32)
        n = min(self.action_dim, a.shape[0])
        fixed[:n] = a[:n]
        return fixed

    def _pi0_chunk(self, obs) -> np.ndarray:
        """Query the pi0 policy for a full action chunk, shape ``[horizon, action_dim]`` (env-space).

        Mirrors the DSRL eval path: feed the (batched, n_envs=1) obs with a scalar ``prompt`` and
        ``noise=None`` for plain pi0/pi0.5 flow sampling; ``result["actions"]`` is ``[horizon, dim]``.
        """
        obs = dict(obs)
        obs["prompt"] = _prompt_str(obs)
        result = self.pi0.infer(observations=obs, noise=None)
        actions = np.asarray(result["actions"], dtype=np.float32)
        return actions.reshape(-1, self.action_dim)

    def _pi0_pool(self, stored_obs, prompt) -> np.ndarray:
        """Draw ``rollout_pool_size`` rollout-policy action chunks for one stored state.

        Returns ``[P, H, action_dim]`` env-space float32 -- P independent plain flow-sampling draws
        (fresh x0 each call) from the CURRENT collection policy. Flow-MILE's frozen-rollout baseline is
        precomputed here at collection time so training can read the stored pool instead of keeping a
        resident copy of the rollout policy and sampling it every step (see
        scripts/export_hitl_to_lerobot.py and third_party/dsrl_openpi/scripts/train.py _flow_mile_grads).
        """
        obs = {**stored_obs, "prompt": prompt}
        chunks = [self._pi0_chunk(obs) for _ in range(self.rollout_pool_size)]
        pool = np.stack(chunks, axis=0).astype(np.float32)  # [P, H, action_dim]
        if self._rollout_horizon is None:
            self._rollout_horizon = int(pool.shape[1])
        return pool

    def _resolve_rollout_horizon(self, obs) -> int:
        """Return the rollout action-chunk horizon, deriving it lazily when needed.

        Prefers the policy's advertised ``action_horizon``; if that is unavailable, samples one
        pi0 chunk from ``obs`` (shape ``[H, action_dim]``) and caches ``H`` so the zero-filled
        non-human baseline pools match the real sampled pools.
        """
        if self._rollout_horizon is None:
            self._rollout_horizon = int(self._pi0_chunk(obs).shape[0])
        return self._rollout_horizon

    def _teleop_action(self):
        """Block until the human issues a deliberate teleop command; return it (env-space) or None.

        Returns ``(action, aborted)``. ``action`` is None if the human released control without a
        command (let the policy act this step). ``aborted`` is True if the human pressed reset ('q').
        """
        dev, toggle = self._device, self.toggle
        last_grasp = self._device_grasp()
        action = None
        while toggle.active and not dev._reset_state:
            ha, _ = self._input2action(device=dev, robot=self.robot, active_arm="right", env_configuration=None)
            if ha is None:
                break
            ha = self._fit_action(ha)
            grasp = self._device_grasp()
            # Gate on a deliberate move/grasp so an idle takeover does not freeze on a zero command.
            if np.linalg.norm(ha[:-1]) > self.cmd_eps or grasp != last_grasp:
                action = ha
                break
            last_grasp = grasp
            if self._render("COLLECT", self._cur_tag, self._cur_step, "HUMAN  (waiting)", False, record=False) == 27:
                raise KeyboardInterrupt
            time.sleep(0.01)
        return action, bool(dev._reset_state)

    def _render(self, phase, ep_tag, step, mode, success, record=True):
        """Render the agent view(s) with a who's-in-control banner (and buffer frames when recording)."""
        want_record = record and self._record_frames is not None
        if not self.enable_render and not want_record:
            return -1
        import cv2

        def _cam(name):
            img = self.base_env.sim.render(height=self.render_size, width=self.render_size, camera_name=name)
            img = np.ascontiguousarray(img[::-1, :, ::-1])  # robosuite renders upside-down RGB; flip + BGR
            cv2.putText(img, name, (8, self.render_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return img

        panels = [_cam(self.camera)]
        if self.show_wrist:
            try:
                panels.append(_cam(self.wrist_camera))
            except Exception:  # noqa: BLE001  (wrist camera unavailable)
                self.show_wrist = False
        frame = np.hstack(panels)
        is_human = mode.startswith("HUMAN")
        color = (0, 0, 255) if is_human else (0, 180, 0)  # BGR: red = human, green = policy
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), color, -1)
        cv2.putText(
            frame,
            f"{phase} {ep_tag}  step {step}  [{'HUMAN' if is_human else 'POLICY'} in control]"
            + ("  SUCCESS" if success else ""),
            (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
        if want_record:
            self._record_frames.append(frame.copy())
        if not self.enable_render:
            return -1
        cv2.imshow(self.window, frame)
        return cv2.waitKey(1) & 0xFF

    def _write_video(self, episode_id, phase, aborted=False):
        frames, self._record_frames = self._record_frames, None
        if not frames or not self.video_dir:
            return None
        import cv2

        h, w = frames[0].shape[:2]
        safe = str(episode_id).replace("/", "_")
        path = os.path.join(self.video_dir, f"{phase}_{safe}{'_aborted' if aborted else ''}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(self.video_fps), (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        logger.info(f"Saved HITL debug video ({len(frames)} frames) to {path}")
        return path

    @staticmethod
    def _store_obs(obs) -> Dict[str, np.ndarray]:
        """Copy the raw pi0 obs keys we persist (images uint8, state float); pi0 normalizes internally."""
        out = {}
        for k in PI0_OBS_KEYS:
            if k in obs:
                out[k] = np.asarray(obs[k])
        return out

    def rollout_episode(self, episode_id, phase="COLLECT", store=True,
                        require_success=False, require_intervention=False):
        """Run one HITL episode. Returns ``(steps, human_steps, stored, success)``.

        Transitions are buffered and, unless the episode is aborted ('q') or filtered out, appended
        to ``self.collected_episodes`` at episode end. ``require_success`` keeps only successful
        episodes; ``require_intervention`` keeps only episodes with >= 1 human step;
        ``store_only_human`` stores only the human-correction transitions.
        """
        env = self.env
        dev, toggle = self._device, self.toggle
        self._cur_tag, self._cur_step = episode_id, 0

        obs = _extract0(env.reset()[0])
        dev.start_control()
        toggle.reset(active=False)
        self._record_frames = [] if self.record_video else None

        chunk, chunk_pos = None, 0     # receding-horizon pi0 chunk state
        prev_human = False
        steps, human_steps, success, done = 0, 0, False, False
        pending, aborted = [], False

        while not done:
            self._cur_step = steps
            if toggle.active:
                # ---- Human takeover: per-step teleop, discard the pi0 chunk. ----
                if not prev_human:
                    dev.start_control()
                prev_human = True
                chunk = None
                action, aborted = self._teleop_action()
                if aborted:
                    break
                if action is None:  # released without a command -> let the policy act this step
                    prev_human = False
                    continue
                action = self._fit_action(action)
                mode, label = "HUMAN", INTERVENTION_LABEL
                human_steps += 1
            else:
                # ---- Autonomous pi0 control (receding-horizon chunk execution). ----
                if prev_human:  # just released -> replan from the corrected state
                    prev_human, chunk = False, None
                if chunk is None or chunk_pos >= len(chunk) or chunk_pos >= self.action_exec_len:
                    chunk = self._pi0_chunk(obs)
                    chunk_pos = 0
                action = self._fit_action(chunk[chunk_pos])
                chunk_pos += 1
                mode, label = "POLICY", ROLLOUT_LABEL

            cur = self._store_obs(obs)
            next_b, rew, term, trunc, info = env.step(action.reshape(1, self.action_dim).astype(np.float32))
            next_obs = _extract0(next_b)
            terminated, truncated = bool(_scalar(term)), bool(_scalar(trunc))
            done = terminated or truncated
            if terminated or _success_from_info(info):
                success = True

            if store and (not self.store_only_human or label == INTERVENTION_LABEL):
                pending.append(
                    dict(
                        obs=cur,
                        action=np.asarray(action, dtype=np.float32),
                        reward=float(_scalar(rew)),
                        done=bool(terminated),
                        truncated=bool(truncated),
                        intervention=int(label),
                    )
                )

            obs = next_obs
            steps += 1
            if self._render(phase, episode_id, steps, mode, success) == 27:
                raise KeyboardInterrupt

        # ---- Keep / discard the whole episode based on the filters. ----
        stored, kept = 0, False
        if store and not aborted and pending:
            n_interventions = sum(1 for t in pending if t["intervention"] == INTERVENTION_LABEL)
            if (success or not require_success) and (n_interventions > 0 or not require_intervention):
                prompt = _prompt_str(obs)
                for t in pending:
                    t["prompt"] = prompt
                    # Flow-MILE: the frozen-rollout baseline pool is only consumed for HUMAN (label-1)
                    # states, so sample it (env-space [P, H, action_dim]) only there; zero-fill every
                    # other frame to keep the per-demo array fixed-shape. Done once per kept episode.
                    if self.rollout_pool_size > 0:
                        if int(t["intervention"]) == INTERVENTION_LABEL:
                            t["rollout_samples"] = self._pi0_pool(t["obs"], prompt)
                        else:
                            horizon = self._resolve_rollout_horizon(t["obs"])
                            t["rollout_samples"] = np.zeros(
                                (self.rollout_pool_size, horizon, self.action_dim),
                                dtype=np.float32,
                            )
                self.collected_episodes.append(pending)
                stored, kept = len(pending), True

        if self.record_video:
            if kept:
                self._write_video(episode_id, phase)
            else:
                self._record_frames = None

        return steps, human_steps, stored, success
