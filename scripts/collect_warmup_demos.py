#!/usr/bin/env python3
"""Collect autonomous warmup demonstrations from a pretrained policy.

Rolls out the policy at ``<load_dir>/checkpoints/<checkpoint>/actor.pt`` AUTONOMOUSLY (no human, no
expert/criterion) in the robomimic env described by ``<load_dir>/.hydra/config.yaml``, keeps the
SUCCESSFUL episodes, and writes them to an HDF5 in the layout consumed by
``load_precollected_hitl`` / ``train_hitl.py``'s ``offline_mode=warmup``:

    /data/demo_{i}/actions       [N, action_dim]   env-space actions executed
    /data/demo_{i}/rewards       [N]
    /data/demo_{i}/dones         [N]
    /data/demo_{i}/intervention  [N]               all 0 here (autonomous); train_hitl.py relabels
                                                   warmup demos to offline (2) at load time
    /data/demo_{i}/obs/{key}     [N, ...]          RAW observations (low-dim un-normalized)

The HDF5 writer + checkpoint/config helpers are reused from ``collect_hitl_rollout.py`` so the file
format matches exactly what the warmup loader expects. Generate the demo set ONCE and reuse the same
file across every sweep config so they all share identical initial demos.

Usage:
    uv run python scripts/collect_warmup_demos.py \
        --load_dir /path/to/run --checkpoint best --num_rollouts 30 \
        --output data/warmup_square_30.hdf5
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2  # noqa: F401  (import order matters; keep above torch)

import argparse
import sys
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf

# Reuse the tested HDF5 writer + checkpoint/config helpers from the HITL collector so the file format
# is identical to what train_hitl.py's offline_mode=warmup (load_precollected_hitl) reads.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_hitl_rollout import (  # noqa: E402
    LABEL_OFFLINE,
    _load_pretrain_cfg,
    _read_source_env_args,
    _resolve_checkpoint_dir,
    _write_h5,
)

from robometer.utils.logger import get_logger, setup_loguru_logging  # noqa: E402
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer  # noqa: E402
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer  # noqa: E402
from robometer_policy_learning.buffers.samplers import RandomSampler  # noqa: E402
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env  # noqa: E402
from robometer_policy_learning.utils.hitl_utils import (  # noqa: E402
    HitlRolloutWorker,
    analyze_intervention_segments,
    describe_control_mode,
)

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Collect autonomous warmup demos from a pretrained policy (for offline_mode=warmup)."
    )
    parser.add_argument("--load_dir", required=True,
                        help="Pretraining run dir (has .hydra/config.yaml + checkpoints/<step>/actor.pt).")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint step (e.g. 50000 or 'best'); default = latest available.")
    parser.add_argument("--num_rollouts", type=int, default=30,
                        help="Number of SUCCESSFUL demos to collect.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--seed", type=int, default=0, help="Env seed.")
    parser.add_argument("--max_attempts", type=int, default=0,
                        help="Cap on total rollout attempts (0 = no cap; Ctrl+C writes what is collected).")
    parser.add_argument("--record_video", action="store_true",
                        help="Record per-episode debug videos next to --output.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    setup_loguru_logging(log_level="INFO", output_dir=out_dir)

    # ---- Env / training / model / policy come from the pretraining run's saved Hydra config so the
    # env build, chunking and obs handling match exactly what the actor was trained with. ----
    cfg = _load_pretrain_cfg(args.load_dir)

    # ---- Policy to roll out ----
    ckpt_dir = _resolve_checkpoint_dir(args.load_dir, args.checkpoint)
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor.pt not found in {actor_path}.")
    actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
    actor.eval()
    logger.info(f"Loaded actor {type(actor).__name__} from {actor_path}")
    if OmegaConf.select(cfg, "env.dino_image_keys", default=None):
        raise NotImplementedError("DINO-embedding (Mode A) policies are not supported by the HITL collector yet.")

    remove_obs_keys = list(
        getattr(actor, "remove_obs_keys", None) or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or []
    )

    # ---- Single, non-chunked collection env (the worker manages chunking manually). ----
    dataset_path = cfg.env.h5_dataset_path
    env, _ = setup_robomimic_env(
        dataset_path=dataset_path,
        n_envs=1,
        device=device,
        seed=args.seed,
        max_episode_steps=cfg.env.max_episode_steps,
        use_full_state=cfg.env.use_full_state,
        terminate_on_success=True,
        chunk_size=None,
        n_action_steps=1,
    )
    action_dim = int(env.single_action_space.shape[0])
    logger.info(f"Control mode: {describe_control_mode(env)}")
    asp = env.single_action_space
    if asp is not None and np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
        action_min = np.asarray(asp.low, dtype=np.float32)
        action_max = np.asarray(asp.high, dtype=np.float32)
    else:
        action_min = action_max = None

    n_exec = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)
    normalize_lowdim = bool(OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False))
    chunk_size = OmegaConf.select(cfg, "training.chunk_size", default=None)

    # ---- Low-dim obs normalization stats (from the source dataset, matching the policy). The worker
    # z-scores obs for act(); _write_h5 inverts that to store RAW obs (the dataset convention). ----
    lowdim_stats = {}
    if normalize_lowdim:
        logger.info("Building offline H5 buffer to recover low-dim obs normalization stats...")
        stats_buffer = H5ReplayBuffer(
            h5_paths=[dataset_path],
            sampler=RandomSampler(),
            remove_obs_keys=list(remove_obs_keys),
            min_action=action_min,
            max_action=action_max,
            normalize_lowdim_obs=True,
        )
        lowdim_stats = stats_buffer.lowdim_obs_stats
        logger.info(f"Low-dim obs normalization keys: {list(lowdim_stats.keys())}")

    # ---- Buffer + autonomous rollout worker (no teleop device, no expert/criterion). ----
    online_buffer = ReplayBuffer(
        capacity=200000,
        remove_obs_keys=list(remove_obs_keys),
        sampler=RandomSampler(),
        min_action=action_min,
        max_action=action_max,
    )
    video_dir = out_dir if args.record_video else None
    worker = HitlRolloutWorker(
        env=env,
        actor=actor,
        online_buffer=online_buffer,
        device=device,
        action_dim=action_dim,
        lowdim_stats=lowdim_stats,
        remove_obs_keys=remove_obs_keys,
        n_action_steps=n_exec,
        store_only_human=False,
        expert_policy=None,
        intervention_criteria=None,
        enable_render=False,
        human_teleop=False,  # autonomous: do NOT open a teleop device (keyboard/SpaceMouse)
        record_video=args.record_video,
        video_dir=video_dir,
    )

    # =====================================================================================
    # Collection loop: keep successful autonomous rollouts until the target is reached.
    # =====================================================================================
    logger.info(
        f"Collecting {args.num_rollouts} successful autonomous demos from {os.path.basename(ckpt_dir)} "
        f"-> {args.output}"
    )
    collected, attempt = 0, 0
    try:
        while collected < args.num_rollouts:
            if args.max_attempts and attempt >= args.max_attempts:
                logger.warning(
                    f"Reached max_attempts={args.max_attempts} with {collected}/{args.num_rollouts} kept; stopping."
                )
                break
            steps, _human, stored, success = worker.rollout_episode(
                f"warmup_{attempt}", phase="WARMUP", allow_human=False, store=True,
                require_success=True, require_intervention=False,
            )
            attempt += 1
            collected += int(stored > 0)
            logger.info(
                f"  attempt {attempt}: success={bool(success)} steps={steps} stored={stored} "
                f"| kept {collected}/{args.num_rollouts}"
            )
    except KeyboardInterrupt:
        logger.info("Interrupted; writing what has been collected so far.")
    finally:
        worker.close()
        env.close()

    if online_buffer.is_empty():
        logger.warning("No successful demos collected; nothing written.")
        return

    # Segment analysis (vs the policy's chunk_size) so you can confirm episodes are chunkable.
    analyze_intervention_segments(online_buffer, chunk_size=chunk_size, context="warmup demos")

    meta_extra = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "kind": "warmup_demos",
        "load_dir": args.load_dir,
        "checkpoint": args.checkpoint,
        "source_dataset": dataset_path,
        "human_mode": "none (autonomous)",
        "n_action_steps": n_exec,
        "chunk_size": chunk_size,
        "normalize_lowdim_obs": normalize_lowdim,
        "obs_are_raw": True,  # low-dim obs were un-normalized back to raw before writing
        "remove_obs_keys": list(remove_obs_keys),
        "intervention_labels": {"policy": 0, "human": 1, "offline": LABEL_OFFLINE},
        "num_attempts": attempt,
        "num_collected": collected,
    }
    env_args = _read_source_env_args(dataset_path)
    stats = _write_h5(
        online_buffer,
        args.output,
        lowdim_stats=lowdim_stats,
        action_min=action_min,
        action_max=action_max,
        env_args=env_args,
        meta_extra=meta_extra,
    )
    logger.success(
        f"Wrote {stats['num_demos']} demos / {stats['total_transitions']} transitions to {args.output} "
        f"({attempt} attempts). Use it with: hitl.offline_mode=warmup hitl.warmup_dataset_path={args.output}"
    )


if __name__ == "__main__":
    main()
