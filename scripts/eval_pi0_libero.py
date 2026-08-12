#!/usr/bin/env python3
"""Evaluate a pi0 / pi0.5 policy across multiple LIBERO tasks from one suite.

Given a suite (``env.env_name``) and a list of task ids (``env.task_ids``), this runs the policy
autonomously (plain flow sampling, ``noise=None``) for ``eval.num_episodes`` episodes per task and
reports the per-task and aggregate success rate. Each task gets a freshly built LIBERO env; object
positions are randomized on every reset (LIBERO placement sampler), so the success rate is measured
over varied layouts (NOT the fixed benchmark init states).

A LoRA / HITL-fine-tuned checkpoint is loaded by setting ``pi0.config_name`` (e.g.
``pi05_libero_hitl_lora``) so ``Pi0Wrapper`` reconstructs the matching architecture; leave it null for
base checkpoints (config inferred from the path).

Usage (headless is fine; set eval.record_video=true to also dump rollout videos):
    uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
        env.env_name=libero_90 'env.task_ids=[57,58,59]' \
        pi0.checkpoint=gs://openpi-assets/checkpoints/pi05_libero/ eval.num_episodes=20

    # evaluate a fine-tuned LoRA checkpoint:
    uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
        'env.task_ids=[57]' pi0.checkpoint=/path/to/round0/ckpt pi0.config_name=pi05_libero_hitl_lora
"""

import os

# JAX/env setup must happen before importing JAX or any GPU libs (mirrors scripts/eval_pi0.py).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import jax

jax.devices()
del jax

import cv2  # noqa: F401  (video writing; import before torch for consistency with the other scripts)

import json
import time
from datetime import datetime

import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robometer_policy_learning.utils.logging_compat import get_logger, setup_loguru_logging
from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env
from robometer_policy_learning.utils.hitl_utils import _success_from_info
from robometer_policy_learning.utils.pi0_hitl_utils import (
    PI0_IMAGE_KEY,
    _extract0,
    _prompt_str,
    _scalar,
)
from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper

logger = get_logger()


def _pi0_chunk(pi0_wrapper, obs, action_dim: int) -> np.ndarray:
    """Full pi0 action chunk for the current obs (env-space, ``[horizon, action_dim]``), noise=None."""
    o = dict(obs)
    o["prompt"] = _prompt_str(obs)
    result = pi0_wrapper.infer(observations=o, noise=None)
    return np.asarray(result["actions"], dtype=np.float32).reshape(-1, action_dim)


def _run_episode(env, pi0_wrapper, action_exec_len: int, action_dim: int, record_frames=None):
    """One autonomous pi0 episode. Returns (success, steps, total_reward, prompt)."""
    obs = _extract0(env.reset()[0])
    prompt = _prompt_str(obs)  # LIBERO language instruction for this task (from the reset obs)
    done, success, ep_reward, ep_steps = False, False, 0.0, 0
    while not done:
        chunk = _pi0_chunk(pi0_wrapper, obs, action_dim)
        n_exec = min(int(action_exec_len), len(chunk))
        for a in chunk[:n_exec]:
            if record_frames is not None and PI0_IMAGE_KEY in obs:
                record_frames.append(np.asarray(obs[PI0_IMAGE_KEY]))
            next_b, rew, term, trunc, info = env.step(a.reshape(1, action_dim).astype(np.float32))
            obs = _extract0(next_b)
            ep_reward += float(_scalar(rew))
            ep_steps += 1
            terminated, truncated = bool(_scalar(term)), bool(_scalar(trunc))
            if terminated or _success_from_info(info):
                success = True
            if terminated or truncated:
                done = True
                break
    return success, ep_steps, ep_reward, prompt


def _write_video(frames, path, fps: int = 20):
    """Write stored pi0-format agentview frames to an mp4 (un-flip + RGB->BGR for a natural view)."""
    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in frames:
        # obs/image is pi0 format (already right-side-up; the 180deg rotation was applied upstream);
        # just convert RGB->BGR for cv2.
        writer.write(np.ascontiguousarray(np.asarray(f)[:, :, ::-1]))
    writer.release()
    logger.info(f"Saved eval video ({len(frames)} frames) to {path}")


def _eval_task(pi0_wrapper, env_name, task_id, cfg, device, output_dir, seed: int) -> dict:
    """Build the env for one task, run all episodes, return a metrics dict."""
    env, _ = setup_libero_env(
        task_suite_name=env_name,
        task_id=int(task_id),
        n_envs=1,
        dinov2_model=None,
        dinov2_processor=None,
        sentence_model=None,
        device=device,
        seed=seed,
        max_episode_steps=int(cfg.env.max_episode_steps),
        image_keys=list(OmegaConf.select(cfg, "env.image_keys", default=[PI0_IMAGE_KEY])),
    )
    action_dim = int(env.single_action_space.shape[0])
    action_exec_len = int(OmegaConf.select(cfg, "pi0.action_exec_len", default=20))
    num_episodes = int(OmegaConf.select(cfg, "eval.num_episodes", default=20))
    record_video = bool(OmegaConf.select(cfg, "eval.record_video", default=False))
    video_first_n = int(OmegaConf.select(cfg, "eval.video_first_n_episodes", default=1))

    successes, steps_l, rewards_l = [], [], []
    language_instruction = None
    try:
        for ep in tqdm(range(num_episodes), desc=f"{env_name} task {task_id}", unit="ep"):
            frames = [] if (record_video and ep < video_first_n) else None
            success, steps, reward, prompt = _run_episode(env, pi0_wrapper, action_exec_len, action_dim, frames)
            if language_instruction is None:
                language_instruction = prompt
            successes.append(float(success))
            steps_l.append(int(steps))
            rewards_l.append(float(reward))
            if frames:
                _write_video(frames, os.path.join(output_dir, "videos", f"task{task_id}_ep{ep}.mp4"))
    finally:
        env.close()

    return {
        "task_id": int(task_id),
        "seed": int(seed),
        "language_instruction": language_instruction,
        "num_episodes": num_episodes,
        "num_success": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_steps": float(np.mean(steps_l)) if steps_l else 0.0,
        "avg_reward": float(np.mean(rewards_l)) if rewards_l else 0.0,
    }


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="libero_eval")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = HydraConfig.get().runtime.output_dir
    setup_loguru_logging(log_level=OmegaConf.select(cfg, "logging.log_level", default="INFO"), output_dir=output_dir)

    # Retrieve seed from config; if None, randomly select it
    seed = OmegaConf.select(cfg, "seed", default=None)
    if seed is None:
        seed = int(time.time_ns() % (1 << 31))
        logger.info(f"No seed provided, selected seed={seed}")
    else:
        seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_name = str(cfg.env.env_name)
    # Accept either a list env.task_ids or a single env.task_id.
    task_ids = OmegaConf.select(cfg, "env.task_ids", default=None)
    if task_ids is None:
        task_ids = [OmegaConf.select(cfg, "env.task_id", default=0)]
    task_ids = [int(t) for t in task_ids]

    pi0_checkpoint = cfg.pi0.checkpoint
    pi0_config_name = OmegaConf.select(cfg, "pi0.config_name", default=None)
    logger.info(f"Loading pi0 policy from {pi0_checkpoint} (config_name={pi0_config_name})")
    pi0_wrapper = Pi0Wrapper(pi0_checkpoint, device=str(device), config_name=pi0_config_name)

    # ---- Optional wandb logging ----
    wandb_logger = None
    if bool(OmegaConf.select(cfg, "logging.wandb", default=False)):
        try:
            from robometer_policy_learning.loggers.wandb_logger import WandbLogger

            string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            wandb_logger = WandbLogger(
                exp_name=f"{cfg.logging.wandb_name}_{string_time}",
                offline=bool(OmegaConf.select(cfg, "logging.wandb_offline", default=False)),
                project=OmegaConf.select(cfg, "logging.wandb_project", default=None),
                entity=OmegaConf.select(cfg, "logging.wandb_entity", default=None),
                log_dir=f"{OmegaConf.select(cfg, 'logging.wandb_log_dir_base', default='logs/wandb')}/{string_time}",
                prefix="eval",
            )
            wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to init wandb logger: {e}")

    logger.info(
        f"Evaluating {env_name} tasks {task_ids} | {int(cfg.eval.num_episodes)} episodes each "
        f"| action_exec_len={int(cfg.pi0.action_exec_len)}"
    )

    results = []
    for idx, task_id in enumerate(task_ids):
        logger.info(f"===== [{idx + 1}/{len(task_ids)}] {env_name} task {task_id} =====")
        r = _eval_task(pi0_wrapper, env_name, task_id, cfg, device, output_dir, seed)
        results.append(r)
        logger.success(
            f"task {task_id} ({r['language_instruction']!r}): success_rate={r['success_rate']:.1%} "
            f"({r['num_success']}/{r['num_episodes']}) avg_steps={r['avg_steps']:.1f}"
        )
        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "task_id": r["task_id"],
                    "success_rate": r["success_rate"],
                    "avg_steps": r["avg_steps"],
                    "avg_reward": r["avg_reward"],
                },
                step=idx,
                prefix="eval",
            )

    # ---- Summary ----
    overall_sr = float(np.mean([r["success_rate"] for r in results])) if results else 0.0
    total_succ = int(np.sum([r["num_success"] for r in results]))
    total_eps = int(np.sum([r["num_episodes"] for r in results]))
    logger.info("=" * 60)
    logger.info(f"{'task_id':>10} | {'success':>8} | {'succ/eps':>9} | {'avg_steps':>9}")
    logger.info("-" * 60)
    for r in results:
        logger.info(
            f"{r['task_id']:>10} | {r['success_rate']:>7.1%} | "
            f"{r['num_success']:>4}/{r['num_episodes']:<4} | {r['avg_steps']:>9.1f}"
        )
    logger.info("-" * 60)
    logger.info(
        f"{'MEAN':>10} | {overall_sr:>7.1%} | {total_succ:>4}/{total_eps:<4} | "
        f"(micro success {total_succ / max(total_eps, 1):.1%})"
    )
    logger.info("=" * 60)
    # Legend: task_id -> language instruction (kept out of the numeric table to preserve alignment).
    logger.info("Task instructions:")
    for r in results:
        logger.info(f"  {r['task_id']:>3}: {r['language_instruction']!r}")

    # Persist results for programmatic use.
    summary = {
        "env_name": env_name,
        "task_ids": task_ids,
        "pi0_checkpoint": str(pi0_checkpoint),
        "pi0_config_name": pi0_config_name,
        "seed": int(seed),
        "num_episodes_per_task": int(cfg.eval.num_episodes),
        "per_task": results,
        "mean_success_rate": overall_sr,
        "micro_success_rate": total_succ / max(total_eps, 1),
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote results to {results_path}")

    if wandb_logger is not None:
        wandb_logger.log(
            {"mean_success_rate": overall_sr, "micro_success_rate": total_succ / max(total_eps, 1)},
            step=len(task_ids),
            prefix="eval",
        )
        try:
            wandb_logger.finish()
        except Exception:  # noqa: BLE001
            pass

    return summary


if __name__ == "__main__":
    main()
