#!/usr/bin/env python3
"""Evaluate a trained FlowDAgger steering net (from scripts/train_flowdagger.py) on LIBERO.

For evals that were skipped during training (the "Evaluate ... now? (Y/S)" prompt), or to re-run one
with more episodes. ``--load-dir`` is the training run's Hydra dir: its ``.hydra/config.yaml`` gives the
frozen pi0.5 checkpoint, env settings and eval tasks, so the eval matches what train_flowdagger.py would
have run. ``--checkpoint`` picks the steering net in that dir (default: the latest ``steering_r{r}.pt``).

By default the results land in the run dir under the name the in-run eval would have used --
``steering_r{r}.pt`` -> ``eval_r{r+1}.json`` (+ ``videos/eval_r{r+1}/``), ``steering_import.pt`` ->
``eval_import.json`` -- with the same per-task lines, summary table and JSON fields, and a log in
``eval_r{r+1}.log``. An existing result is never overwritten unless ``--overwrite``.

Usage:
    # the skipped eval of round 1's net (steering_r1.pt -> eval_r2.json in the run dir)
    uv run python scripts/eval_flowdagger.py --load-dir outputs/fd_pi05_long --checkpoint steering_r1.pt

    # latest net, fewer tasks/episodes, written elsewhere
    uv run python scripts/eval_flowdagger.py --load-dir outputs/fd_pi05_long \
        --eval-task-ids 46 48 --num-episodes 10 --output-dir outputs/fd_pi05_long_evals

    # the untrained baseline (plain pi0.5, N(0, I) noise) under the run's eval settings
    uv run python scripts/eval_flowdagger.py --load-dir outputs/fd_pi05_long --checkpoint none
"""

import os

# JAX/env setup must happen before importing JAX or any GPU libs (mirrors train_flowdagger.py).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import jax

jax.devices()
del jax

import cv2  # noqa: F401  (import before torch, as in the other pi0 scripts)

import argparse
import glob
import re
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

# Reuse train_flowdagger.py's eval loop, steering wrapper and config helpers so the numbers are
# computed exactly as the in-run eval would have.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_flowdagger import EVAL_MAX_EPISODE_STEPS, Steering, _id_list, evaluate  # noqa: E402

from robometer_policy_learning.utils.flowdagger_utils import Pi0FlowOps, load_steering  # noqa: E402
from robometer_policy_learning.utils.logging_compat import get_logger, setup_loguru_logging  # noqa: E402
from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper  # noqa: E402

logger = get_logger()


def _resolve_checkpoint(load_dir: str, checkpoint):
    """``--checkpoint`` -> (steering .pt path or None for the untrained baseline, eval name)."""
    if checkpoint is None:
        rounds = [
            (int(m.group(1)), p) for p in glob.glob(os.path.join(load_dir, "steering_r*.pt"))
            if (m := re.search(r"steering_r(\d+)\.pt$", p))
        ]
        if rounds:
            checkpoint = max(rounds)[1]
        elif os.path.exists(os.path.join(load_dir, "steering_import.pt")):
            checkpoint = "steering_import.pt"
        else:
            raise FileNotFoundError(f"No steering_r*.pt or steering_import.pt in {load_dir}.")
    if str(checkpoint).lower() == "none":
        return None, "eval_base"
    path = checkpoint if os.path.isabs(checkpoint) or os.path.exists(checkpoint) else os.path.join(load_dir, checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Steering checkpoint not found: {path}")
    base = os.path.basename(path)
    m = re.fullmatch(r"steering_r(\d+)\.pt", base)
    if m:  # steering_r{r}.pt is evaluated after round r's training -> eval_r{r+1}, as in train_flowdagger.py
        return path, f"eval_r{int(m.group(1)) + 1}"
    if base == "steering_import.pt":
        return path, "eval_import"
    return path, f"eval_{os.path.splitext(base)[0]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--load-dir", required=True, help="train_flowdagger.py run dir (has .hydra/config.yaml).")
    parser.add_argument("--checkpoint", default=None,
                        help="steering_r{r}.pt / steering_import.pt (in --load-dir) or a path; 'none' = untrained "
                             "baseline. Default: the latest steering_r*.pt.")
    parser.add_argument("--eval-task-ids", type=int, nargs="+", default=None, help="Default: the run's eval tasks.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Per task. Default: the run's value.")
    parser.add_argument("--seed", type=int, default=None, help="Default: the run's seed (same as the in-run eval).")
    parser.add_argument("--max-episode-steps", type=int, default=EVAL_MAX_EPISODE_STEPS,
                        help=f"Eval episode budget (default {EVAL_MAX_EPISODE_STEPS}, as train_flowdagger.py's evals).")
    parser.add_argument("--video-first-n", type=int, default=None, help="Videos per task. Default: the run's value.")
    parser.add_argument("--no-video", action="store_true", help="Do not save rollout videos.")
    parser.add_argument("--output-dir", default=None, help="Where results go. Default: --load-dir.")
    parser.add_argument("--name", default=None, help="Result name (<name>.json). Default: from the checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing <name>.json.")
    args = parser.parse_args()

    cfg_path = os.path.join(args.load_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{cfg_path} not found: --load-dir must be a train_flowdagger.py run dir.")
    cfg = OmegaConf.load(cfg_path)
    ckpt, default_name = _resolve_checkpoint(args.load_dir, args.checkpoint)
    name = args.name or default_name
    out_dir = args.output_dir or args.load_dir
    out_path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it or --name to write another.")

    # Overrides go into the loaded run config, so evaluate() reads them exactly like the in-run eval.
    if args.no_video:
        cfg.flowdagger.eval_record_video = False
    elif "eval_record_video" not in cfg.flowdagger:  # runs from before eval videos existed: use today's default
        cfg.flowdagger.eval_record_video = True
    if args.video_first_n is not None:
        cfg.flowdagger.eval_video_first_n_episodes = args.video_first_n
    fd = cfg.flowdagger
    task_ids = args.eval_task_ids or _id_list(fd.eval_task_ids)
    num_episodes = args.num_episodes if args.num_episodes is not None else int(fd.eval_num_episodes)
    seed = args.seed if args.seed is not None else int(cfg.seed)

    # Own log file next to the results (not the run's training.log, which loguru would append to).
    setup_loguru_logging(log_level=OmegaConf.select(cfg, "logging.log_level", default="INFO"))
    os.makedirs(out_dir, exist_ok=True)
    logger.add(os.path.join(out_dir, f"{name}.log"), level="INFO",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n")
    logger.info(f"Evaluating {ckpt or 'the untrained baseline'} from run {args.load_dir} -> {out_path}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(fd.device) if torch.cuda.is_available() or str(fd.device) == "cpu" else "cpu")
    logger.info(f"Loading frozen pi0 policy from {cfg.pi0.checkpoint} (config_name={cfg.pi0.config_name})")
    pi0_wrapper = Pi0Wrapper(cfg.pi0.checkpoint, device=str(device), config_name=cfg.pi0.config_name)
    flow = Pi0FlowOps(pi0_wrapper.policy, num_steps=int(fd.num_denoise_steps), fp_iters=int(fd.fp_iters),
                      batch_size=int(fd.inversion_batch_size))
    steering = Steering(flow, fd, device, seed)
    if ckpt is not None:
        steering.net = load_steering(ckpt, device)
        # Name the checkpoint in the logs and the JSON's "policy" field.
        steering.describe = lambda: f"steering net ({os.path.basename(ckpt)})"
    evaluate(cfg, flow, steering, task_ids, num_episodes, seed, out_path, max_episode_steps=args.max_episode_steps)


if __name__ == "__main__":
    main()
