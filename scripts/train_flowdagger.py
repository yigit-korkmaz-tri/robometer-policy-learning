#!/usr/bin/env python3
"""FlowDAgger: human-in-the-loop adaptation of a FROZEN pi0.5 through its initial flow noise, on LIBERO.

A small torch STEERING network maps pi0.5's observation features (mean-pooled VLM prefix hidden state +
proprio state) to the initial noise ``w [10, 32]`` that pi0.5's 10-step Euler sampler integrates into
an action chunk. pi0.5 itself is never updated. Per round (all local, one process):

  0. eval     the starting policy on the eval tasks (skipped unless flowdagger.skip_init_eval=false)
  1. collect  ``collect_num_rollouts`` kept HITL rollouts per scheduled task: pi0.5 drives with the
              steering net's noise, a human takes over with the SpaceMouse / keyboard (Tab). Every
              autonomous step records the noise its chunk came from.
  2. invert   each human-correction window (<= 10 steps, normalized into pi0.5's action space) to the
              noise that reproduces it (per-step fixed-point inversion of the sampler, paper Eq. 3);
              windows that do not re-integrate to within ``inversion_mse_threshold`` are dropped.
  3. train    the steering net with MSE in noise space on all rounds' samples: inverted corrections
              plus the noises of the autonomous chunks (``human_ratio`` of each batch from corrections).
  4. eval     asks "Evaluate ... now? (Y/S)", then evals the steered policy -> <run dir>/eval_r{r+1}.json
              (per-task success rate, avg steps); then asks whether to continue to the next round.

Outputs in the Hydra run dir: round_{r}_task_{t}_rollouts.hdf5 (raw rollouts + per-step noise),
round_{r}_samples.h5 (features, noise targets, kind 0=autonomous / 1=human, inversion MSE),
steering_r{r}.pt, eval_r{k}.json, videos/eval_r{k}/task{T}_ep{i}.mp4 (flowdagger.eval_record_video).
A skipped eval can be run later with scripts/eval_flowdagger.py.

Reference: Murray et al., "FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in
Latent Space", CoRL 2026 (FlowDAgger/2607.08877v1.pdf); official code in FlowDAgger/flowdagger_pi05/.

Usage (local machine with a display for the teleop window):
    uv run python scripts/train_flowdagger.py --config-name libero_flowdagger \
        flowdagger.rounds=3 'flowdagger.eval_task_ids=[31,57,63,9]' 'flowdagger.collect_task_ids=[31,57]' \
        'flowdagger.collect_schedule="31;31 57"' flowdagger.collect_num_rollouts=5

Headless check on corrections collected earlier by scripts/collect_hitl_libero_pi0.py (no teleop:
rounds == start_round, so it only inverts, trains once and evals):
    uv run python scripts/train_flowdagger.py --config-name libero_flowdagger flowdagger.rounds=0 \
        'flowdagger.import_hdf5s=[outputs/<run>/round_0_task_31_rollouts.hdf5]' \
        'flowdagger.eval_task_ids=[31]' flowdagger.eval_num_episodes=10 flowdagger.train_steps_per_round=500

Resume after round 1 of an earlier run:
    uv run python scripts/train_flowdagger.py --config-name libero_flowdagger \
        flowdagger.resume_dir=outputs/<date>/<time> flowdagger.start_round=2
"""

import os

# JAX/env setup must happen before importing JAX or any GPU libs (mirrors collect_hitl_libero_pi0.py).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Initialize JAX early before any other GPU libraries so it gets a clean GPU context.
import jax

jax.devices()
del jax

# cv2 must be imported before torch so its HighGUI (imshow/waitKey) does not deadlock against the
# pynput keyboard listener used by the takeover toggle. Also writes the eval videos.
import cv2

import glob
import json
import time
from datetime import datetime

import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env
from robometer_policy_learning.utils.flowdagger_utils import (
    KIND_HUMAN,
    KIND_POLICY,
    ActionNormalizer,
    FlowDAggerPi0LiberoHitlWorker,
    Pi0FlowOps,
    SteeringNet,
    build_samples,
    concat_samples,
    episode_arrays,
    load_steering,
    read_rollouts_h5,
    read_samples_h5,
    save_steering,
    write_rollouts_h5,
    write_samples_h5,
)
from robometer_policy_learning.utils.hitl_utils import _success_from_info
from robometer_policy_learning.utils.logging_compat import get_logger, setup_loguru_logging
from robometer_policy_learning.utils.pi0_hitl_utils import PI0_IMAGE_KEY, _extract0, _prompt_str, _scalar
from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper

logger = get_logger()

# Eval episode budget, the same as scripts/eval_pi0_libero.py's libero_eval.yaml (what the
# hitl_pi0_loop*.sh evals use), so FlowDAgger success rates are comparable with those runs. Collection
# keeps env.max_episode_steps (800), which leaves the human room to finish a correction.
EVAL_MAX_EPISODE_STEPS = 600


# ---------------------------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------------------------
def _id_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (ListConfig, list, tuple)):
        return [int(x) for x in value]
    return [int(x) for x in str(value).replace(",", " ").split()]


def _build_schedule(schedule, collect_ids: list) -> list:
    """``"31 57; 63 9"`` (or a list of lists) -> ``[[31, 57], [63, 9]]``; empty => every collect task."""
    if schedule is None or (isinstance(schedule, str) and not schedule.strip()):
        return [list(collect_ids)]
    parts = list(schedule) if isinstance(schedule, (ListConfig, list, tuple)) else str(schedule).split(";")
    rounds = [_id_list(p) for p in parts]
    for i, ids in enumerate(rounds):
        if not ids:
            raise ValueError(f"flowdagger.collect_schedule round {i} is empty.")
        bad = [t for t in ids if t not in collect_ids]
        if bad:
            raise ValueError(
                f"flowdagger.collect_schedule round {i} has tasks {bad} not in collect_task_ids {collect_ids}."
            )
    return rounds


def _round_tasks(schedule: list, r: int) -> list:
    return schedule[min(r, len(schedule) - 1)]


def _confirm_next_round(r: int) -> bool:
    try:
        reply = input(f"\nRound {r - 1} finished. Continue to round {r}? [Y/n] ")
    except EOFError:  # no terminal: unattended, keep going
        return True
    return not reply.strip().lower().startswith("n")


def _ask_eval_now(what: str) -> bool:
    """Ask before an eval (as hitl_pi0_loop_seperated.sh does): True = evaluate now, False = skip.

    No terminal => evaluate (unattended keeps going)."""
    while True:
        try:
            reply = input(f"\nEvaluate {what} now? (Y (Yes) / S (Skip)) ").strip().lower()
        except EOFError:
            return True
        if reply in ("y", "yes"):
            return True
        if reply in ("s", "skip"):
            return False
        print("  Please answer Y or S.")


def _build_env(cfg, task_id: int, seed: int, max_episode_steps=None):
    """Single LIBERO env for one task (no DINO / sentence models: pi0 takes raw images + prompt).

    ``max_episode_steps`` defaults to ``env.max_episode_steps`` (collection)."""
    return setup_libero_env(
        task_suite_name=cfg.env.env_name,
        task_id=int(task_id),
        n_envs=1,
        seed=seed,
        max_episode_steps=int(max_episode_steps or cfg.env.max_episode_steps),
        image_keys=list(OmegaConf.select(cfg, "env.image_keys", default=[PI0_IMAGE_KEY])),
        init_state_index=OmegaConf.select(cfg, "env.init_state_index", default="auto"),
        settle_steps=int(OmegaConf.select(cfg, "env.settle_steps", default=10)),
    )[0]


# ---------------------------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------------------------
class Steering:
    """Holds the current steering net (None before any training) and produces per-obs noise."""

    def __init__(self, flow: Pi0FlowOps, fd_cfg, device: torch.device, seed: int):
        self.flow, self.cfg, self.device = flow, fd_cfg, device
        self.net = None
        self.init_noise = str(fd_cfg.init_noise)
        if self.init_noise not in ("randn", "zeros"):
            raise ValueError(f"flowdagger.init_noise must be randn or zeros, got {self.init_noise!r}")
        self._rng = np.random.default_rng(seed)

    def new_net(self, feature_dim: int) -> SteeringNet:
        return SteeringNet(
            feature_dim=feature_dim, state_dim=self.flow.state_dim, horizon=self.flow.horizon,
            noise_dim=self.flow.model_dim, hidden_dims=list(self.cfg.hidden_dims),
            latent_dim=int(self.cfg.latent_dim), noise_bound=float(self.cfg.noise_bound),
            output_bound=str(self.cfg.output_bound),
        ).to(self.device)

    def describe(self) -> str:
        return "steering net" if self.net is not None else f"untrained ({self.init_noise} noise = base pi0.5)"

    @torch.no_grad()
    def noise(self, obs: dict) -> np.ndarray:
        shape = (self.flow.horizon, self.flow.model_dim)
        if self.net is None:
            if self.init_noise == "zeros":
                return np.zeros(shape, np.float32)
            return self._rng.standard_normal(shape).astype(np.float32)
        feats = torch.from_numpy(self.flow.features([obs])).to(self.device)
        self.net.eval()
        return self.net(feats)[0].cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------------------------
def _run_episode(env, flow: Pi0FlowOps, steering: Steering, action_exec_len: int, action_dim: int,
                 record_frames=None):
    """One autonomous steered episode. Returns (success, steps, total_reward, prompt)."""
    obs = _extract0(env.reset()[0])
    prompt = _prompt_str(obs)  # LIBERO language instruction for this task (from the reset obs)
    done, success, ep_reward, steps = False, False, 0.0, 0
    while not done:
        o = dict(obs)
        o["prompt"] = prompt
        chunk = flow.sample(o, steering.noise(o)).reshape(-1, action_dim)
        for a in chunk[: min(action_exec_len, len(chunk))]:
            if record_frames is not None and PI0_IMAGE_KEY in obs:
                record_frames.append(np.asarray(obs[PI0_IMAGE_KEY]))
            next_b, rew, term, trunc, info = env.step(a.reshape(1, action_dim).astype(np.float32))
            obs = _extract0(next_b)
            ep_reward += float(_scalar(rew))
            steps += 1
            terminated, truncated = bool(_scalar(term)), bool(_scalar(trunc))
            if terminated or _success_from_info(info):
                success = True
            if terminated or truncated:
                done = True
                break
    return success, steps, ep_reward, prompt


def _write_video(frames, path, fps: int = 20):
    """Write stored pi0-format agentview frames to an mp4 (same as scripts/eval_pi0_libero.py)."""
    if not frames:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in frames:
        # obs/image is pi0 format (already right-side-up); just convert RGB->BGR for cv2.
        writer.write(np.ascontiguousarray(np.asarray(f)[:, :, ::-1]))
    writer.release()
    logger.info(f"Saved eval video ({len(frames)} frames) to {path}")


def _eval_task(cfg, flow, steering, task_id: int, num_episodes: int, seed: int, video_dir=None,
               max_episode_steps: int = EVAL_MAX_EPISODE_STEPS) -> dict:
    """All eval episodes of one task -> the per-task record of scripts/eval_pi0_libero.py.

    With ``video_dir``, the first ``flowdagger.eval_video_first_n_episodes`` episodes are saved there.
    """
    action_exec_len = int(cfg.pi0.action_exec_len)
    video_first_n = int(OmegaConf.select(cfg, "flowdagger.eval_video_first_n_episodes", default=1))
    env = _build_env(cfg, task_id, seed, max_episode_steps)
    task_info = getattr(env, "libero_task_info", None) or {}
    action_dim = int(env.single_action_space.shape[0])
    episodes, language_instruction = [], None
    try:
        for ep in tqdm(range(num_episodes), desc=f"{cfg.env.env_name} task {task_id}", unit="ep"):
            frames = [] if (video_dir and ep < video_first_n) else None
            success, steps, reward, prompt = _run_episode(env, flow, steering, action_exec_len, action_dim, frames)
            language_instruction = language_instruction or prompt
            episodes.append({"episode": ep, "success": bool(success), "steps": int(steps), "reward": float(reward)})
            if frames:
                _write_video(frames, os.path.join(video_dir, f"task{task_id}_ep{ep}.mp4"))
    finally:
        env.close()
    succ = [e["success"] for e in episodes]
    # Steps for the successful episodes only -- how long the policy takes when it does solve the task.
    succ_steps = [e["steps"] for e in episodes if e["success"]]
    return {
        "task_id": int(task_id),
        "task_name": task_info.get("task_name"),
        "seed": int(seed),
        "max_episode_steps": int(max_episode_steps),
        "language_instruction": language_instruction,
        "num_episodes": len(episodes),
        "num_success": int(sum(succ)),
        "success_rate": float(np.mean(succ)) if episodes else 0.0,
        "avg_steps": float(np.mean([e["steps"] for e in episodes])) if episodes else 0.0,
        "avg_steps_success": float(np.mean(succ_steps)) if succ_steps else 0.0,
        "avg_reward": float(np.mean([e["reward"] for e in episodes])) if episodes else 0.0,
        "episodes": episodes,
    }


def evaluate(cfg, flow, steering, task_ids, num_episodes, seed, out_path, wandb_logger=None, step=None,
             max_episode_steps: int = EVAL_MAX_EPISODE_STEPS):
    """Steered eval on ``task_ids``; logs the same per-task lines and summary table as eval_pi0_libero.py
    and writes a JSON with the same fields (plus ``policy``) to ``out_path``."""
    policy = steering.describe()
    name = os.path.splitext(os.path.basename(out_path))[0]
    video_dir = None
    if bool(OmegaConf.select(cfg, "flowdagger.eval_record_video", default=False)):
        video_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "videos", name)
    logger.info(
        f"Evaluating {policy} on {cfg.env.env_name} tasks {list(task_ids)} | {num_episodes} episodes each "
        f"| max_episode_steps={int(max_episode_steps)} | action_exec_len={int(cfg.pi0.action_exec_len)}"
    )
    results = []
    for idx, task_id in enumerate(task_ids):
        logger.info(f"===== [{idx + 1}/{len(task_ids)}] {cfg.env.env_name} task {task_id} ({name}) =====")
        r = _eval_task(cfg, flow, steering, task_id, num_episodes, seed, video_dir, max_episode_steps)
        results.append(r)
        logger.success(
            f"task {task_id} ({r['language_instruction']!r}): success_rate={r['success_rate']:.1%} "
            f"({r['num_success']}/{r['num_episodes']}) avg_steps={r['avg_steps']:.1f} "
            f"avg_steps_success={r['avg_steps_success']:.1f}"
        )

    # ---- Summary (same table as eval_pi0_libero.py) ----
    overall_sr = float(np.mean([r["success_rate"] for r in results])) if results else 0.0
    total_succ = int(sum(r["num_success"] for r in results))
    total_eps = int(sum(r["num_episodes"] for r in results))
    # Completion time pooled over every episode of every task (micro), and over the successful ones.
    all_eps = [e for r in results for e in r["episodes"]]
    succ_steps = [e["steps"] for e in all_eps if e["success"]]
    overall_steps = float(np.mean([e["steps"] for e in all_eps])) if all_eps else 0.0
    overall_steps_succ = float(np.mean(succ_steps)) if succ_steps else 0.0
    logger.info("=" * 60)
    logger.info(f"{name}: {policy}")
    logger.info(f"{'task_id':>10} | {'success':>8} | {'succ/eps':>9} | {'avg_steps':>9} | {'steps_succ':>10}")
    logger.info("-" * 60)
    for r in results:
        logger.info(
            f"{r['task_id']:>10} | {r['success_rate']:>7.1%} | {r['num_success']:>4}/{r['num_episodes']:<4} | "
            f"{r['avg_steps']:>9.1f} | {r['avg_steps_success']:>10.1f}"
        )
    logger.info("-" * 60)
    logger.info(
        f"{'MEAN':>10} | {overall_sr:>7.1%} | {total_succ:>4}/{total_eps:<4} | {overall_steps:>9.1f} | "
        f"{overall_steps_succ:>10.1f} (micro success {total_succ / max(total_eps, 1):.1%}; steps pooled over episodes)"
    )
    logger.info("=" * 60)
    # Legend: task_id -> language instruction (kept out of the numeric table to preserve alignment).
    logger.info("Task instructions:")
    for r in results:
        logger.info(f"  {r['task_id']:>3}: {r['language_instruction']!r}")

    summary = {
        "policy": policy,
        "env_name": str(cfg.env.env_name),
        "task_ids": [int(t) for t in task_ids],
        "max_episode_steps": int(max_episode_steps),
        "pi0_checkpoint": str(cfg.pi0.checkpoint),
        "pi0_config_name": cfg.pi0.config_name,
        "seed": int(seed),
        "num_episodes_per_task": int(num_episodes),
        "per_task": results,
        "mean_success_rate": overall_sr,
        "micro_success_rate": total_succ / max(total_eps, 1),
        "avg_steps": overall_steps,
        "avg_steps_success": overall_steps_succ,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote results to {out_path}")
    if wandb_logger is not None:
        metrics = {
            "mean_success_rate": overall_sr,
            "micro_success_rate": summary["micro_success_rate"],
            "avg_steps": overall_steps,
            "avg_steps_success": overall_steps_succ,
        }
        for r in results:
            metrics[f"task{r['task_id']}/success_rate"] = r["success_rate"]
            metrics[f"task{r['task_id']}/avg_steps_success"] = r["avg_steps_success"]
        wandb_logger.log_dict(metrics, step=step, prefix="eval")
    return summary


# ---------------------------------------------------------------------------------------------
# Episodes -> samples (features + noise targets)
# ---------------------------------------------------------------------------------------------
def episodes_to_samples(episodes, flow, normalizer, fd_cfg, task_id: int, round_idx: int):
    """Build, invert and featurize the training samples of a list of episodes (per-episode arrays)."""
    specs = []  # (obs dict, spec)
    for ep in episodes:
        for s in build_samples(
            ep, horizon=flow.horizon, use_policy=bool(fd_cfg.use_policy_samples),
            drop_pre_takeover_chunk=bool(fd_cfg.drop_pre_takeover_chunk),
            human_window_stride=int(fd_cfg.human_window_stride), min_human_window=int(fd_cfg.min_human_window),
        ):
            obs = {k: v[s["t"]] for k, v in ep["obs"].items()}
            obs["prompt"] = ep["prompt"]
            specs.append((obs, s))
    if not specs:
        return None

    targets = np.zeros((len(specs), flow.horizon, flow.model_dim), np.float32)
    inv_mse = np.zeros(len(specs), np.float32)
    kinds = np.asarray([s["kind"] for _, s in specs], np.int64)
    human = np.flatnonzero(kinds == KIND_HUMAN)
    for i in np.flatnonzero(kinds == KIND_POLICY):
        targets[i] = specs[i][1]["target"]
    keep = np.ones(len(specs), bool)
    if len(human):
        pairs = [normalizer.env_to_model(specs[i][1]["actions"]) for i in human]
        t0 = time.time()
        w, mse = flow.invert(
            [specs[i][0] for i in human], np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs]),
            env_dim=normalizer.env_dim, progress="Inverting corrections",
        )
        targets[human], inv_mse[human] = w, mse
        bad = mse > float(fd_cfg.inversion_mse_threshold)
        keep[human[bad]] = False
        p99 = float(np.percentile(np.abs(w), 99))
        logger.info(
            f"  inverted {len(human)} human windows in {time.time() - t0:.1f}s: mse median {np.median(mse):.2e} "
            f"max {mse.max():.2e}; dropped {int(bad.sum())} above {float(fd_cfg.inversion_mse_threshold):.0e}; "
            f"p99 |w*| = {p99:.2f}"
        )
        if str(fd_cfg.output_bound) == "tanh" and p99 >= 0.95 * float(fd_cfg.noise_bound):
            logger.warning(
                f"p99 |w*| = {p99:.2f} is at the steering bound {float(fd_cfg.noise_bound)}: tanh cannot reach "
                "these targets. Consider a larger flowdagger.noise_bound or output_bound=linear."
            )

    idx = np.flatnonzero(keep)
    if not len(idx):
        return None
    feats = flow.features([specs[i][0] for i in idx], progress="Featurizing samples")
    logger.info(
        f"  samples: {int((kinds[idx] == KIND_HUMAN).sum())} human + {int((kinds[idx] == KIND_POLICY).sum())} "
        f"autonomous (from {len(episodes)} episodes)"
    )
    return {
        "features": feats.astype(np.float32),
        "target": targets[idx],
        "kind": kinds[idx],
        "inv_mse": inv_mse[idx],
        "task_id": np.full(len(idx), int(task_id), np.int64),
        "round": np.full(len(idx), int(round_idx), np.int64),
    }


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
def train_steering(steering: Steering, data: dict, fd_cfg, wandb_logger=None, global_step: int = 0):
    """BC (MSE in noise space) on the aggregated samples. Returns the updated global step."""
    device = steering.device
    feats = torch.from_numpy(data["features"]).to(device)
    targets = torch.from_numpy(data["target"]).to(device)
    kinds = data["kind"]
    if steering.net is None or bool(fd_cfg.reinit_each_round):
        steering.net = steering.new_net(feats.shape[1])
    net = steering.net
    opt = torch.optim.Adam(net.parameters(), lr=float(fd_cfg.lr))

    human_idx = torch.from_numpy(np.flatnonzero(kinds == KIND_HUMAN)).to(device)
    policy_idx = torch.from_numpy(np.flatnonzero(kinds == KIND_POLICY)).to(device)
    ratio = fd_cfg.human_ratio
    mixed = ratio is not None and len(human_idx) > 0 and len(policy_idx) > 0
    batch_size, steps = int(fd_cfg.batch_size), int(fd_cfg.train_steps_per_round)
    n_human = round(batch_size * float(ratio)) if mixed else 0
    logger.info(
        f"Training steering net for {steps} steps on {len(human_idx)} human + {len(policy_idx)} autonomous samples "
        f"({'human_ratio=' + str(ratio) if mixed else 'uniform sampling'}, batch {batch_size})"
    )

    net.train()
    bar = tqdm(range(steps), desc="Training steering net", unit="step")
    for step in bar:
        if mixed:
            b = torch.cat([
                human_idx[torch.randint(len(human_idx), (n_human,), device=device)],
                policy_idx[torch.randint(len(policy_idx), (batch_size - n_human,), device=device)],
            ])
        else:
            b = torch.randint(len(kinds), (batch_size,), device=device)
        err = (net(feats[b]) - targets[b]).pow(2).flatten(1).mean(1)
        loss = err.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        global_step += 1
        if (step + 1) % 100 == 0 or step + 1 == steps:
            is_h = torch.isin(b, human_idx) if len(human_idx) else torch.zeros_like(b, dtype=torch.bool)
            metrics = {"loss": float(loss)}
            if is_h.any():
                metrics["loss_human"] = float(err[is_h].mean())
            if (~is_h).any():
                metrics["loss_policy"] = float(err[~is_h].mean())
            bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            if step + 1 == steps:
                logger.info(f"  step {step + 1}/{steps}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if wandb_logger is not None:
                wandb_logger.log_dict(metrics, step=global_step, prefix="train")
    net.eval()
    return global_step


# ---------------------------------------------------------------------------------------------
@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="libero_flowdagger")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir
    setup_loguru_logging(log_level=OmegaConf.select(cfg, "logging.log_level", default="INFO"), output_dir=run_dir)
    fd = cfg.flowdagger
    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(str(fd.device) if torch.cuda.is_available() or str(fd.device) == "cpu" else "cpu")

    eval_ids = _id_list(fd.eval_task_ids)
    collect_ids = _id_list(fd.collect_task_ids) or list(eval_ids)
    schedule = _build_schedule(fd.collect_schedule, collect_ids)
    rounds, start_round = int(fd.rounds), int(fd.start_round)
    logger.info(f"Eval tasks (every round): {eval_ids}")
    for r in range(start_round, rounds):
        logger.info(f"  round {r}: collect on {_round_tasks(schedule, r)}")

    # ---- frozen pi0.5 ----
    logger.info(f"Loading frozen pi0 policy from {cfg.pi0.checkpoint} (config_name={cfg.pi0.config_name})")
    pi0_wrapper = Pi0Wrapper(cfg.pi0.checkpoint, device=str(device), config_name=cfg.pi0.config_name)
    flow = Pi0FlowOps(
        pi0_wrapper.policy, num_steps=int(fd.num_denoise_steps), fp_iters=int(fd.fp_iters),
        batch_size=int(fd.inversion_batch_size),
    )
    normalizer = ActionNormalizer.from_policy(pi0_wrapper.policy, horizon=flow.horizon, model_dim=flow.model_dim)
    logger.info(
        f"pi0: action horizon {flow.horizon}, model action dim {flow.model_dim}, "
        f"{'quantile' if normalizer.use_quantiles else 'z-score'} action norm"
    )
    if int(cfg.pi0.action_exec_len) < flow.horizon:
        logger.warning(
            f"pi0.action_exec_len={cfg.pi0.action_exec_len} < chunk length {flow.horizon}: autonomous noise targets "
            "describe whole chunks, but only part of each is executed."
        )
    steering = Steering(flow, fd, device, seed)

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
            )
            wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to init wandb logger: {e}")

    eval_history = []  # (eval name, policy, summary) of every eval this run, for the closing table

    def run_eval(name: str, step: int):
        summary = evaluate(cfg, flow, steering, eval_ids, int(fd.eval_num_episodes), seed,
                           os.path.join(run_dir, f"{name}.json"), wandb_logger, step)
        eval_history.append((name, summary))
        return summary

    # ---- earlier data: resumed run's samples and/or imported HDF5s ----
    sample_parts = []
    resume_dir = OmegaConf.select(fd, "resume_dir", default=None)
    if resume_dir:
        for p in sorted(glob.glob(os.path.join(resume_dir, "round_*_samples.h5"))):
            part = read_samples_h5(p)
            if len(part["round"]) and int(part["round"][0]) < start_round:
                sample_parts.append(part)
                logger.info(f"Resumed {len(part['kind'])} samples from {p}")
        if start_round > 0:
            ckpt = os.path.join(resume_dir, f"steering_r{start_round - 1}.pt")
            steering.net = load_steering(ckpt, device)
            logger.info(f"Resumed steering net from {ckpt}")
    imports = list(OmegaConf.select(fd, "import_hdf5s", default=[]) or [])
    for p in imports:
        logger.info(f"Importing {p}")
        part = episodes_to_samples(read_rollouts_h5(p), flow, normalizer, fd, int(fd.import_task_id), -1)
        if part is not None:
            sample_parts.append(part)
    if imports:
        imported = concat_samples(sample_parts[-len(imports):])
        if imported is not None:
            write_samples_h5(os.path.join(run_dir, "round_import_samples.h5"), imported, {"sources": imports})

    global_step = 0
    if imports and concat_samples(sample_parts) is not None:
        # Train on the imported data before collecting, so round start_round already collects with it.
        global_step = train_steering(steering, concat_samples(sample_parts), fd, wandb_logger, global_step)
        ckpt = os.path.join(run_dir, "steering_import.pt")
        save_steering(ckpt, steering.net, {"sources": imports})
        if _ask_eval_now("the steering net trained on the imported data"):
            run_eval("eval_import", global_step)
        else:
            logger.info(
                f"Skipped eval_import. Run it later with: uv run python scripts/eval_flowdagger.py "
                f"--load-dir {run_dir} --checkpoint steering_import.pt"
            )

    # ---- HITL rounds ----
    worker = None
    last_trained_round = None
    try:
        for r in range(start_round, rounds):
            if r > start_round and not _confirm_next_round(r):
                logger.info(f"Stopping before round {r}.")
                break
            tasks = _round_tasks(schedule, r)
            logger.info(f"========== FLOWDAGGER ROUND {r} (collect on {tasks}; {steering.describe()}) ==========")
            if r == start_round and not bool(fd.skip_init_eval):
                run_eval(f"eval_r{r}", global_step)

            round_parts = []
            for task_id in tasks:
                env = _build_env(cfg, task_id, seed + r)
                if worker is None:
                    worker = FlowDAggerPi0LiberoHitlWorker(
                        env, pi0_wrapper, int(env.single_action_space.shape[0]),
                        flow=flow, noise_fn=steering.noise,
                        action_exec_len=int(cfg.pi0.action_exec_len),
                        enable_render=bool(cfg.teleop.enable_render),
                        teleop_device=str(cfg.teleop.device),
                        takeover_key=str(cfg.teleop.takeover_key),
                        pos_sensitivity=float(cfg.teleop.pos_sensitivity),
                        rot_sensitivity=float(cfg.teleop.rot_sensitivity),
                        camera=cfg.teleop.camera, wrist_camera=cfg.teleop.wrist_camera,
                        show_wrist=bool(cfg.teleop.show_wrist),
                        record_video=bool(cfg.debug),
                        video_dir=os.path.join(run_dir, "debug_videos") if bool(cfg.debug) else None,
                    )
                else:
                    worker.rebind_env(env)
                worker.collected_episodes = []
                kept, attempt = 0, 0
                logger.info(
                    f"Round {r}, task {task_id}: collecting {int(fd.collect_num_rollouts)} rollouts "
                    "(ESC = done with this task)"
                )
                try:
                    while kept < int(fd.collect_num_rollouts):
                        steps, human_steps, stored, success = worker.rollout_episode(
                            f"r{r}_t{task_id}_ep{attempt}", phase="COLLECT", store=True,
                            require_success=bool(cfg.hitl.require_success),
                            require_intervention=bool(cfg.hitl.keep_only_hitl_rollouts),
                        )
                        attempt += 1
                        kept += int(stored > 0)
                        logger.info(
                            f"  rollout {attempt}: success={bool(success)} steps={steps} human_steps={human_steps} "
                            f"stored={stored} | kept {kept}/{int(fd.collect_num_rollouts)}"
                        )
                except KeyboardInterrupt:
                    logger.info(f"ESC pressed: done with task {task_id} after {kept} kept rollouts.")
                finally:
                    env.close()
                if not worker.collected_episodes:
                    logger.warning(f"Round {r}, task {task_id}: nothing kept.")
                    continue
                write_rollouts_h5(
                    worker.collected_episodes, os.path.join(run_dir, f"round_{r}_task_{task_id}_rollouts.hdf5"),
                    {"created": datetime.now().isoformat(timespec="seconds"), "pi0_checkpoint": str(cfg.pi0.checkpoint),
                     "env_name": str(cfg.env.env_name), "task_id": int(task_id), "round": r,
                     "policy": steering.describe(), "action_exec_len": int(cfg.pi0.action_exec_len),
                     "intervention_labels": {"policy": 0, "human": 1}},
                )
                episodes = [episode_arrays(ep) for ep in worker.collected_episodes]
                part = episodes_to_samples(episodes, flow, normalizer, fd, task_id, r)
                if part is not None:
                    round_parts.append(part)

            round_samples = concat_samples(round_parts)
            if round_samples is None:
                logger.warning(f"Round {r}: no usable samples; skipping training.")
                continue
            write_samples_h5(os.path.join(run_dir, f"round_{r}_samples.h5"), round_samples, {"round": r})
            sample_parts.append(round_samples)

            global_step = train_steering(steering, concat_samples(sample_parts), fd, wandb_logger, global_step)
            ckpt = os.path.join(run_dir, f"steering_r{r}.pt")
            save_steering(ckpt, steering.net, {"round": r})
            last_trained_round = r
            # Eval round r's policy before the continue prompt (as the .sh loop does), unless skipped.
            if _ask_eval_now(f"round {r}'s steering net"):
                run_eval(f"eval_r{r + 1}", global_step)
            else:
                logger.info(
                    f"Skipped eval_r{r + 1}. Run it later with: uv run python scripts/eval_flowdagger.py "
                    f"--load-dir {run_dir} --checkpoint steering_r{r}.pt"
                )
    finally:
        if worker is not None:
            worker.close()
        if wandb_logger is not None:
            wandb_logger.finish()
    if eval_history:
        # Every eval of this run side by side, so rounds can be compared without opening the JSONs.
        logger.info("=" * 60)
        logger.info(f"{'eval':>12} | {'success':>8} | {'succ/eps':>9} | {'steps_succ':>10} | policy")
        logger.info("-" * 60)
        for name, sm in eval_history:
            n_succ = sum(t["num_success"] for t in sm["per_task"])
            n_eps = sum(t["num_episodes"] for t in sm["per_task"])
            logger.info(
                f"{name:>12} | {sm['mean_success_rate']:>7.1%} | {n_succ:>4}/{n_eps:<4} | "
                f"{sm['avg_steps_success']:>10.1f} | {sm['policy']}"
            )
        logger.info("=" * 60)
    logger.info(f"FlowDAgger done (last trained round: {last_trained_round}). Outputs in {run_dir}")


if __name__ == "__main__":
    main()
