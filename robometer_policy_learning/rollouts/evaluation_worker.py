import numpy as np
import cv2
import os
from datetime import datetime
from typing import Dict, List
import torch

from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.utils.gpu_utils import move_to_device, convert_to_tensor
from robometer_policy_learning.loggers.logger import Logger
from loguru import logger


class EvaluationWorker:
    """Worker class for running evaluations and recording videos."""

    def __init__(
        self,
        eval_env,
        device,
        num_episodes: int = 10,
        record_video: bool = True,
        logger: Logger = None,
        image_keys: List[str] = None,
        lowdim_obs_stats: Dict[str, Dict[str, np.ndarray]] = None,
    ):
        self.eval_env = eval_env
        self.device = device
        self.num_episodes = num_episodes
        self.record_video = record_video
        self.logger = logger
        self.image_keys = image_keys

        self.lowdim_obs_stats = lowdim_obs_stats or {}
        self._norm_tensors = None

        # Check if this is a chunked rollout (like in RolloutWorker)
        self.is_chunked_rollout = hasattr(self.eval_env, "is_chunk_empty")

    def run(self, actor: BaseActor):
        """Run evaluation episodes and optionally record video."""
        # Set actor to eval mode to disable dropout/batchnorm randomness
        was_training = actor.training
        actor.eval()

        try:
            # Run multiple evaluation episodes for statistics
            with torch.inference_mode():
                eval_metrics = self._run_evaluations(actor, num_episodes=self.num_episodes)

            # Record a video if requested
            if self.record_video:
                with torch.inference_mode():
                    video_metrics = self._record_evaluation_video(actor)
                eval_metrics.update(video_metrics)

            return eval_metrics
        finally:
            # Restore original training mode
            if was_training:
                actor.train()

    def _prepare_obs(self, obs):
        """Convert an extracted obs to a device tensor and z-score low-dim keys."""
        obs_device = convert_to_tensor(obs)
        obs_device = move_to_device(obs_device, self.device)
        return self._normalize_obs(obs_device)

    def _normalize_obs(self, obs):
        """Apply the training buffer's low-dim z-score stats to matching obs keys.

        No-op when no stats were provided. Image/embedding keys are absent from the stats
        dict and so are left untouched.
        """
        if not self.lowdim_obs_stats or not isinstance(obs, dict):
            return obs
        if self._norm_tensors is None:
            self._norm_tensors = {
                k: (
                    torch.as_tensor(st["mean"], dtype=torch.float32, device=self.device),
                    torch.as_tensor(st["std"], dtype=torch.float32, device=self.device),
                )
                for k, st in self.lowdim_obs_stats.items()
            }
        for k, (mean, std) in self._norm_tensors.items():
            if k in obs and torch.is_tensor(obs[k]):
                obs[k] = (obs[k].to(torch.float32) - mean) / std
        return obs

    def _run_evaluations(self, actor: BaseActor, num_episodes: int = 10):
        """Run multiple evaluation episodes without video recording for statistics."""
        all_rewards = []
        all_steps = []
        all_success = []

        for episode_idx in range(num_episodes):
            sum_reward = 0
            is_done = False
            obs, info = self.eval_env.reset()
            step_count = 0

            # Extract first env data for vectorized environments
            obs = self._extract_env_data(obs, 0)

            # Track if success was achieved at any point during the episode
            is_success = False

            while not is_done:
                # Get action and step environment
                obs_device = self._prepare_obs(obs)

                # Handle chunked actions (like in RolloutWorker)
                if self.is_chunked_rollout:
                    if self.eval_env.is_chunk_empty:
                        action, _ = actor.act(obs_device, deterministic=True)
                        action = action.detach().cpu().numpy()
                        next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                    else:
                        action = None
                        next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                    # Update action to grab the actual action that was taken
                    action = self.eval_env._get_last_action()
                else:
                    action, _ = actor.act(obs_device, deterministic=True)
                    action = action.detach().cpu().numpy()
                    next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)

                # Extract data for first environment (index 0)
                reward = self._extract_scalar(rewards, 0)
                done = self._extract_scalar(dones, 0)
                truncated = self._extract_scalar(truncateds, 0)
                obs = self._extract_env_data(next_obs, 0)
                info = self._extract_info(infos, 0)

                # Check for success at every step - once success is achieved, it stays true
                if isinstance(info, dict):
                    if info.get("is_success", False) or info.get("success", False):
                        is_success = True

                sum_reward += reward
                is_done = done or truncated
                step_count += 1

            all_rewards.append(sum_reward)
            all_steps.append(step_count)
            all_success.append(is_success)

        # Compute statistics
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        avg_steps = np.mean(all_steps)
        success_rate = np.mean(all_success)

        eval_metrics = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": np.min(all_rewards),
            "max_reward": np.max(all_rewards),
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "num_eval_episodes": num_episodes,
        }

        logger.info(f"Evaluation over {num_episodes} episodes:")
        logger.info(f"  Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
        logger.info(f"  Success Rate: {success_rate:.1%}")
        logger.info(f"  Average Steps: {avg_steps:.1f}")

        return eval_metrics

    def _record_evaluation_video(self, actor: BaseActor):
        """Record a single evaluation episode with video for visualization."""

        # Create videos directory (robust to missing logger or log_dir)
        if self.logger is not None and getattr(self.logger, "log_dir", None):
            base_dir = self.logger.log_dir
            video_dir = os.path.join(base_dir, "evaluation_videos")
        else:
            video_dir = os.path.join(os.getcwd(), "evaluation_videos")
        os.makedirs(video_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sum_reward = 0
        is_done = False
        obs, info = self.eval_env.reset()

        # Extract first env data for vectorized environments
        obs = self._extract_env_data(obs, 0)

        # Track if success was achieved at any point during the episode
        is_success = False

        # Identify image keys and initialize storage
        if self.image_keys is not None:
            image_keys = self.image_keys
        else:
            image_keys = (
                [key for key in obs.keys() if "image" in key.lower() or "cam" in key.lower()]
                if isinstance(obs, dict)
                else []
            )
        frames_dict = {"stacked": []}
        step_count = 0
        frame_skip_interval = 3

        while not is_done:
            # Capture frames (skip some for faster video)
            if step_count % frame_skip_interval == 0 and isinstance(obs, dict) and image_keys:
                collected_frames = {}

                for key in image_keys:
                    if key in obs:
                        frame = obs[key]

                        # Convert torch tensor to numpy if needed
                        if hasattr(frame, "detach"):
                            frame = frame.detach().cpu().numpy()

                        if isinstance(frame, np.ndarray):
                            # Convert to uint8
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

                            # Handle shape formats
                            if len(frame.shape) == 4:  # Remove batch dimension
                                frame = frame[0]
                            elif len(frame.shape) == 3 and frame.shape[0] in [
                                1,
                                3,
                            ]:  # CHW to HWC
                                frame = np.transpose(frame, (1, 2, 0))

                            # Convert grayscale to RGB
                            if len(frame.shape) == 2:
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                            collected_frames[key] = frame

                # Stack frames if any were collected
                if collected_frames:
                    stacked_frame = self._stack_frames(collected_frames, image_keys)
                    if stacked_frame is not None:
                        frames_dict["stacked"].append(stacked_frame)

            # Get action and step environment
            obs_device = self._prepare_obs(obs)

            # Handle chunked actions (like in RolloutWorker)
            if self.is_chunked_rollout:
                if self.eval_env.is_chunk_empty:
                    action, _ = actor.act(obs_device, deterministic=True)
                    action = action.detach().cpu().numpy()
                    next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                else:
                    action = None
                    next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)
                # Update action to grab the actual action that was taken
                action = self.eval_env._get_last_action()
            else:
                action, _ = actor.act(obs_device, deterministic=True)
                action = action.detach().cpu().numpy()
                next_obs, rewards, dones, truncateds, infos = self.eval_env.step(action)

            # Extract data for first environment (index 0)
            reward = self._extract_scalar(rewards, 0)
            done = self._extract_scalar(dones, 0)
            truncated = self._extract_scalar(truncateds, 0)
            obs = self._extract_env_data(next_obs, 0)
            info = self._extract_info(infos, 0)

            # Check for success at every step - once success is achieved, it stays true
            if isinstance(info, dict):
                if info.get("is_success", False) or info.get("success", False):
                    is_success = True

            sum_reward += reward
            is_done = done or truncated
            step_count += 1

            if step_count > 1000:
                break

        # Save video and log to TensorBoard
        video_saved = False
        if frames_dict["stacked"]:
            video_path = os.path.join(video_dir, f"eval_{timestamp}_all_cameras.mp4")

            # Log to TensorBoard if logger available
            if self.logger is not None:
                video_frames_rgb = []
                for frame_bgr in frames_dict["stacked"]:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_chw = np.transpose(frame_rgb, (2, 0, 1))
                    video_frames_rgb.append(frame_chw)

                video_tensor = torch.from_numpy(np.stack(video_frames_rgb)).unsqueeze(0)
                self.logger.log_video("eval_video", video=video_tensor, step=step_count, prefix="eval")

            # Save MP4 file
            self._save_video(frames_dict["stacked"], video_path, fps=20)
            video_saved = True
            logger.success(f"📹 Saved evaluation video: {video_path}")

        return {
            "video_reward": sum_reward,
            "video_steps": step_count,
            "video_saved": video_saved,
            "video_success": is_success,
        }

    def _stack_frames(self, collected_frames: Dict[str, np.ndarray], image_keys: List[str]) -> np.ndarray:
        """Stack multiple camera frames horizontally."""
        frames_to_stack = []
        for key in sorted(image_keys):
            if key in collected_frames:
                frame = collected_frames[key]
                # Resize to match height if needed
                if frames_to_stack and frame.shape[0] != frames_to_stack[0].shape[0]:
                    target_height = frames_to_stack[0].shape[0]
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    new_width = int(target_height * aspect_ratio)
                    frame = cv2.resize(frame, (new_width, target_height))
                frames_to_stack.append(frame)

        return cv2.cvtColor(np.hstack(frames_to_stack), cv2.COLOR_RGB2BGR) if frames_to_stack else None

    def _save_video(self, frames: List[np.ndarray], video_path: str, fps: int = 20):
        """Save frames as MP4 video."""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                video_writer.write(frame)
        finally:
            video_writer.release()

    def _extract_env_data(self, batched_data, env_idx: int):
        """Extract data for a specific environment from batched format (like RolloutWorker)."""
        if isinstance(batched_data, dict):
            return {key: value[env_idx] for key, value in batched_data.items()}
        elif isinstance(batched_data, (list, tuple)) and len(batched_data) > 0:
            return batched_data[env_idx]
        else:
            return batched_data

    def _extract_scalar(self, batched_data, env_idx: int):
        """Extract scalar value for a specific environment."""
        if isinstance(batched_data, (list, np.ndarray)):
            return float(batched_data[env_idx])
        else:
            return float(batched_data)

    def _extract_info(self, infos, env_idx: int):
        """Extract info dict for a specific environment."""
        if infos is None:
            return {}
        if isinstance(infos, list) and env_idx < len(infos):
            return infos[env_idx] if infos[env_idx] is not None else {}
        elif isinstance(infos, dict):
            # Could be a dict with batched values or a single info dict
            # Try to extract per-env if possible, otherwise return the whole dict
            return infos
        return {}


class BatchEvaluationWorker(EvaluationWorker):
    """Vectorized evaluation over ``eval_env.num_envs`` parallel envs.

    Drop-in replacement for :class:`EvaluationWorker` when ``eval_env`` is a
    ``gymnasium.vector.VectorEnv`` (built with ``n_envs > 1``). Instead of running the
    ``num_episodes`` episodes strictly one-at-a-time, it gives each of the ``num_envs`` envs a fixed
    QUOTA (``num_episodes`` split as evenly as possible) and runs them back-to-back via the vector
    env's autoreset, so envs never idle waiting for each other and every recorded episode runs to its
    natural termination (unbiased success rate). Exactly ``num_episodes`` episodes are recorded.

    Two independent speedups stack:
      * the policy's ``act()`` (a full reverse-diffusion sample for a DiffusionActor) is evaluated on
        a ``(num_envs, ...)`` batch in ONE GPU forward pass per decision, not ``num_envs`` passes;
      * with an ``AsyncVectorEnv`` the MuJoCo sims (and camera rendering) step CONCURRENTLY across
        subprocesses, which is the real lever when env stepping dominates (e.g. image obs).
    With a ``SyncVectorEnv`` only the first applies (the sims still step serially in-process), so the
    win there is modest; ``AsyncVectorEnv`` is what unlocks near-parallel env stepping.

    Returns the SAME metrics dict keys as :class:`EvaluationWorker` (``success_rate`` etc.), so it is
    interchangeable in the training scripts.
    """

    def __init__(self, *args, max_episode_steps: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-wave safety cap so a non-terminating env can't hang the loop. The env should truncate
        # itself at max_episode_steps; add a small margin. Falls back to a large constant if unknown.
        self.max_episode_steps = int(max_episode_steps) if max_episode_steps else None
        self.num_envs = int(getattr(self.eval_env, "num_envs", 1))

    def run(self, actor: BaseActor):
        """Run vectorized evaluation (+ optional video from env 0) and return metrics."""
        was_training = actor.training
        actor.eval()
        try:
            with torch.inference_mode():
                return self._run_batched(actor)
        finally:
            if was_training:
                actor.train()

    def _success_flags(self, info, num_envs: int) -> np.ndarray:
        """Per-env success flags from a (possibly vectorized) env info.

        Handles the gymnasium vector info dict (``info['is_success']`` is a length-``num_envs``
        array with a ``'_is_success'`` presence mask), a plain list of per-env info dicts, and the
        ``final_info`` nesting used by same-step autoreset. Missing entries default to False.
        """
        out = np.zeros(num_envs, dtype=bool)
        if isinstance(info, dict):
            for key in ("is_success", "success"):
                if key in info:
                    arr = np.asarray(info[key]).reshape(-1)
                    m = min(num_envs, arr.shape[0])
                    out[:m] |= np.array([bool(x) for x in arr[:m]], dtype=bool)
            fin = info.get("final_info")
            if fin is not None:
                fin = np.asarray(fin, dtype=object).reshape(-1)
                for i in range(min(num_envs, fin.shape[0])):
                    d = fin[i]
                    if isinstance(d, dict) and (d.get("is_success") or d.get("success")):
                        out[i] = True
        elif isinstance(info, (list, tuple)):
            for i in range(min(num_envs, len(info))):
                d = info[i] or {}
                if isinstance(d, dict) and (d.get("is_success") or d.get("success")):
                    out[i] = True
        return out

    def _act_batched(self, actor: BaseActor, obs):
        """Batched (num_envs, ...) policy actions as a numpy array (env space)."""
        obs_device = self._prepare_obs(obs)
        action, _ = actor.act(obs_device, deterministic=True)
        return action.detach().cpu().numpy()

    def _run_batched(self, actor: BaseActor):
        """Collect exactly ``num_episodes`` full episodes across ``num_envs`` parallel envs.

        Each env is assigned a fixed QUOTA of episodes (``num_episodes`` split as evenly as possible)
        and runs them back-to-back via the vector env's autoreset -- envs never wait for each other
        between episodes, so there is no synchronized-wave idle time. Every recorded episode runs to
        its NATURAL termination (success terminates early, failure runs to the horizon), so the
        success-rate estimate is unbiased. Once an env hits its quota it keeps stepping (harmless)
        until the slowest env finishes, so the only waste is a single bounded tail rather than a stall
        on every wave.

        Autoreset is ``AutoresetMode.NEXT_STEP`` (gymnasium default): the step where an env terminates
        carries the true final reward/info; the FOLLOWING step is that env's reset (fresh obs, reward
        0, action ignored). We track ``prev_done`` to skip accumulating that reset step and to start a
        clean accumulator for the next episode.
        """
        K = self.num_envs
        N = int(self.num_episodes)
        base, rem = divmod(N, K)
        # Envs [0, rem) get one extra so the quotas sum to exactly N.
        quota = np.array([base + (1 if i < rem else 0) for i in range(K)], dtype=np.int64)
        max_ep_len = self.max_episode_steps or 1000
        step_cap = max_ep_len * (int(quota.max()) + 1) + 100  # safety bound on total vector-steps

        rewards, steps, success = [], [], []
        completed = np.zeros(K, dtype=np.int64)
        ep_reward = np.zeros(K, dtype=np.float64)
        ep_steps = np.zeros(K, dtype=np.int64)
        ep_success = np.zeros(K, dtype=bool)
        prev_done = np.zeros(K, dtype=bool)  # True => this env's CURRENT step is an autoreset step

        obs, _ = self.eval_env.reset()
        image_keys = self._video_image_keys(obs) if self.record_video else []
        video_frames = []  # captured from env 0's FIRST episode only

        # Per-env open-loop action chunking. The shared VectorActionChunkingWrapper replans ALL envs
        # whenever ANY one needs a fresh chunk (and overwrites every env's buffer), so once episodes
        # desync it resamples every env almost every step. For a STOCHASTIC actor (diffusion / flow
        # sample from fresh noise per act() call) that injects high-frequency jitter and tanks success
        # vs the serial worker. So we drive chunking PER ENV here: each env samples a chunk and
        # executes n_action_steps of it open-loop, resampling only when ITS own buffer runs out or it
        # just reset -- exactly the serial (n_envs=1) semantics. We step with 2-D (num_envs,
        # action_dim) actions, which bypasses the wrapper's coupled chunking.
        chunked = bool(self.is_chunked_rollout)
        if chunked:
            n_exec = int(getattr(self.eval_env, "n_action_steps", 1) or 1)
            adim = int(np.prod(self.eval_env.single_action_space.shape))
            chunk_buf = np.zeros((K, n_exec, adim), dtype=np.float32)
            chunk_pos = np.full(K, n_exec, dtype=np.int64)  # >= n_exec => refill on first real step
        total = 0

        while (completed < quota).any() and total < step_cap:
            # Envs that terminated last step reset THIS step (NEXT_STEP autoreset): start a fresh
            # accumulator and don't count this reset step (its reward is 0 and its action is ignored).
            if prev_done.any():
                ep_reward[prev_done] = 0.0
                ep_steps[prev_done] = 0
                ep_success[prev_done] = False
            real = ~prev_done  # real steps of the running episode (exclude the reset step)

            # Video: capture env 0 only during its first episode (before it has completed any).
            if image_keys and completed[0] == 0 and not prev_done[0] and total % 3 == 0:
                frame = self._obs_to_frame(self._extract_env_data(obs, 0), image_keys)
                if frame is not None:
                    video_frames.append(frame)

            if chunked:
                # Refill only the envs whose own chunk is exhausted (on a real step); the rest keep
                # executing their existing open-loop chunk. act() is still batched over all envs (same
                # call frequency), but fresh chunks are adopted ONLY by the needy envs.
                need = real & (chunk_pos >= n_exec)
                if need.any():
                    pred = self._act_batched(actor, obs).reshape(K, -1, adim)[:, :n_exec, :]
                    chunk_buf[need] = pred[need]
                    chunk_pos[need] = 0
                actions = np.zeros((K, adim), dtype=np.float32)
                ridx = np.nonzero(real)[0]  # reset-step envs keep the dummy 0 action (env ignores it)
                actions[ridx] = chunk_buf[ridx, chunk_pos[ridx]]
                chunk_pos[ridx] += 1
                next_obs, r, term, trunc, info = self.eval_env.step(actions)
            else:
                next_obs, r, term, trunc, info = self.eval_env.step(self._act_batched(actor, obs))

            r = np.asarray(r, dtype=np.float64).reshape(-1)[:K]
            done = (np.asarray(term).astype(bool) | np.asarray(trunc).astype(bool)).reshape(-1)[:K]
            succ = self._success_flags(info, K)

            ep_reward[real] += r[real]
            ep_steps[real] += 1
            ep_success[real] |= succ[real]

            # Record episodes that ended this step, but only up to each env's quota.
            for i in np.nonzero(done & real)[0]:
                if completed[i] < quota[i]:
                    rewards.append(float(ep_reward[i]))
                    steps.append(int(ep_steps[i]))
                    success.append(bool(ep_success[i]))
                    completed[i] += 1

            if chunked:
                # A finished env resets NEXT step; force it to resample a fresh chunk from its new
                # initial obs on the first real step of its next episode.
                chunk_pos[done] = n_exec

            prev_done = done
            obs = next_obs
            total += 1

        if total >= step_cap:
            logger.warning(f"Batched eval hit step cap ({step_cap}); recorded {len(success)}/{N} episodes.")

        eval_metrics = {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "avg_steps": float(np.mean(steps)) if steps else 0.0,
            "success_rate": float(np.mean(success)) if success else 0.0,
            "num_eval_episodes": len(success),
        }
        logger.info(f"Batched evaluation over {len(success)} episodes ({K} envs, quota {base}-{base+1}):")
        logger.info(f"  Average Reward: {eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}")
        logger.info(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
        logger.info(f"  Average Steps: {eval_metrics['avg_steps']:.1f}")

        if self.record_video:
            eval_metrics.update(self._flush_video(video_frames))
        return eval_metrics

    def _video_image_keys(self, obs) -> List[str]:
        """Image keys to render for the video (explicit override, else auto-detected from env 0)."""
        if self.image_keys is not None:
            return list(self.image_keys)
        obs0 = self._extract_env_data(obs, 0)
        if not isinstance(obs0, dict):
            return []
        return [k for k in obs0.keys() if "image" in k.lower() or "cam" in k.lower()]

    def _obs_to_frame(self, obs0: dict, image_keys: List[str]):
        """Convert a single env's image obs into one stacked BGR frame (or None)."""
        if not isinstance(obs0, dict):
            return None
        collected = {}
        for key in image_keys:
            if key not in obs0:
                continue
            frame = obs0[key]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            if not isinstance(frame, np.ndarray):
                continue
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            if frame.ndim == 4:  # drop batch dim
                frame = frame[0]
            elif frame.ndim == 3 and frame.shape[0] in (1, 3):  # CHW -> HWC
                frame = np.transpose(frame, (1, 2, 0))
            if frame.ndim == 2:  # grayscale -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            collected[key] = frame
        if not collected:
            return None
        return self._stack_frames(collected, image_keys)

    def _flush_video(self, frames: List[np.ndarray]) -> Dict:
        """Save the captured env-0 frames to an mp4 and log to the logger (no-op if empty)."""
        if not frames:
            return {"video_saved": False}
        if self.logger is not None and getattr(self.logger, "log_dir", None):
            video_dir = os.path.join(self.logger.log_dir, "evaluation_videos")
        else:
            video_dir = os.path.join(os.getcwd(), "evaluation_videos")
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"eval_{timestamp}_all_cameras.mp4")

        if self.logger is not None:
            rgb = [np.transpose(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (2, 0, 1)) for f in frames]
            video_tensor = torch.from_numpy(np.stack(rgb)).unsqueeze(0)
            self.logger.log_video("eval_video", video=video_tensor, step=len(frames), prefix="eval")

        self._save_video(frames, video_path, fps=20)
        logger.success(f"📹 Saved evaluation video: {video_path}")
        return {"video_saved": True}
