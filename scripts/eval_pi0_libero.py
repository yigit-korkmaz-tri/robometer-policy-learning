#!/usr/bin/env python3
"""Evaluate a pi0 / pi0.5 policy across multiple LIBERO tasks from one suite.

Given a suite (``env.env_name``) and a list of task ids (``env.task_ids``), this runs the policy
autonomously (plain flow sampling, ``noise=None``) for ``eval.num_episodes`` episodes per task and
reports the per-task and aggregate success rate. Each task gets a freshly built LIBERO env.
``env.max_episode_steps`` is either one int for all tasks or a list with one entry per task id. By
default (``env.init_state_index=auto`` with ``env.libero_plus=false``) object positions are
randomized on every reset (LIBERO placement sampler), so the success rate is measured over varied
layouts rather than the fixed benchmark init states; set ``env.init_state_index`` to pin them.

With ``env.libero_plus=true`` the tasks come from the LIBERO-plus robustness benchmark instead. Setting
``env.perturbation`` (camera / light / noise / layout / robot / language / background) turns
``env.task_ids`` into BASE task ids and runs every episode under a DIFFERENT variant of that family for
that base task -- so one row of the summary is a (base task, perturbation) success rate rather than one
arbitrary perturbation instance. The summary also breaks results down per perturbation dimension. See
docs/LIBERO_PLUS.md.

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

    # LIBERO-plus: base tasks 3 and 7, 20 episodes each, every episode a different camera perturbation:
    uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
        env.libero_plus=true env.env_name=libero_spatial 'env.task_ids=[3,7]' \
        env.perturbation=camera eval.num_episodes=20

    # per-task episode budgets (one entry per task id, same order):
    uv run python scripts/eval_pi0_libero.py --config-name libero_eval \
        'env.task_ids=[57,58,59]' 'env.max_episode_steps=[400,600,900]'
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
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from robometer_policy_learning.utils.logging_compat import get_logger, setup_loguru_logging
from robometer_policy_learning.envs import libero_plus
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


def _resolve_task_ids(cfg) -> list:
    """Resolve which task ids to evaluate.

    Precedence: ``env.task_names`` > ``env.task_ids`` > ``env.task_id``. Under
    ``env.perturbation`` these are BASE task ids (plain-LIBERO ordering), not variant ids.
    """
    env_name = str(cfg.env.env_name)
    use_plus = bool(OmegaConf.select(cfg, "env.libero_plus", default=False))

    task_names = OmegaConf.select(cfg, "env.task_names", default=None)
    if task_names:
        if not use_plus:
            raise ValueError("env.task_names is only supported with env.libero_plus=true")
        if _perturbation_spec(cfg) is not None:
            # Ambiguous: names identify variants, while perturbation mode wants base tasks.
            raise ValueError(
                "env.task_names and env.perturbation are mutually exclusive: names pin one specific "
                "variant, while env.perturbation samples variants of a BASE task. Use env.task_ids "
                "(base ids) with env.perturbation."
            )
        return [libero_plus.resolve_task_id(env_name, str(n)) for n in task_names]

    task_ids = OmegaConf.select(cfg, "env.task_ids", default=None)
    if task_ids is None:
        task_ids = [OmegaConf.select(cfg, "env.task_id", default=0)]
    return [int(t) for t in task_ids]


def _perturbation_spec(cfg):
    """``env.perturbation`` normalized to a list of family names, or None when unset."""
    spec = OmegaConf.select(cfg, "env.perturbation", default=None)
    if spec is None:
        return None
    if isinstance(spec, (ListConfig, list, tuple)):
        spec = [str(x) for x in spec]
        return spec or None
    return [str(spec)]


def _steps_note(max_steps_per_task) -> str:
    """Render the resolved budgets compactly: a single value, or the per-task list."""
    unique = set(max_steps_per_task)
    return str(next(iter(unique))) if len(unique) == 1 else str(list(max_steps_per_task))


def _resolve_max_episode_steps(cfg, task_ids) -> list:
    """Per-task episode step budgets, as a list aligned with ``task_ids``.

    ``env.max_episode_steps`` is either a single int applied to every task, or a list with one entry
    per task id (handy when tasks in one run need very different budgets -- e.g. a long-horizon
    libero_10 task next to a short pick-and-place).
    """
    spec = OmegaConf.select(cfg, "env.max_episode_steps", default=None)
    if spec is None:
        raise ValueError("env.max_episode_steps must be set (an int, or a list with one entry per task id).")

    if isinstance(spec, (ListConfig, list, tuple)):
        steps = [int(x) for x in spec]
        if len(steps) != len(task_ids):
            raise ValueError(
                f"env.max_episode_steps has {len(steps)} entries but there are {len(task_ids)} task ids "
                f"({list(task_ids)}). Pass a single int to use one budget for every task, or a list of "
                f"exactly {len(task_ids)} values (order matches env.task_ids)."
            )
    else:
        steps = [int(spec)] * len(task_ids)

    if any(v <= 0 for v in steps):
        raise ValueError(f"env.max_episode_steps must be positive; got {steps}")
    return steps


def _resolve_variant_seed(cfg) -> int:
    """``env.variant_seed``, or a fresh random one when it is null.

    Resolved ONCE per run and reused for every base task, so a run has a single reproducible variant
    stream. The chosen value is logged and written to eval_results.json -- rerun with
    ``env.variant_seed=<value>`` to replay the exact same perturbation sequence.
    """
    variant_seed = OmegaConf.select(cfg, "env.variant_seed", default=None)
    if variant_seed is None:
        variant_seed = int(time.time_ns() % (1 << 31))
        logger.info(f"No env.variant_seed provided, selected variant_seed={variant_seed}")
    return int(variant_seed)


def _make_cycler(cfg, env_name: str, base_task_id: int, num_episodes: int, variant_seed: int):
    """Variant pool + sampler for one (base task, perturbation family) cell.

    Each episode draws a different variant, so the reported success rate covers the family rather than
    one arbitrary perturbation instance.
    """
    levels = OmegaConf.select(cfg, "env.variant_difficulty_levels", default=None)
    variants = libero_plus.variants_of(
        env_name,
        int(base_task_id),
        categories=_perturbation_spec(cfg),
        difficulty_levels=[int(x) for x in levels] if levels else None,
        name_contains=OmegaConf.select(cfg, "env.variant_name_contains", default=None),
    )
    cycler = libero_plus.VariantCycler(
        variants,
        seed=int(variant_seed),
        sampling=str(OmegaConf.select(cfg, "env.variant_sampling", default="shuffle")),
    )
    base_name = libero_plus.base_task_name(env_name, int(base_task_id))
    logger.info(
        f"base task {base_task_id} ({base_name}): {cycler.num_variants} variants in pool "
        f"[{', '.join(sorted({v.category for v in variants}))}], sampling={cycler.sampling}, "
        f"seed={cycler.seed}"
    )
    if num_episodes > cycler.num_variants:
        logger.warning(
            f"eval.num_episodes={num_episodes} exceeds the {cycler.num_variants}-variant pool for base "
            f"task {base_task_id}; variants will repeat (the pool is reshuffled once exhausted)."
        )
    return cycler


def _build_env(cfg, env_name: str, task_id: int, device, seed: int, max_episode_steps: int):
    """Build the single-env vector env for one concrete LIBERO(-plus) task id."""
    env, _ = setup_libero_env(
        task_suite_name=env_name,
        task_id=int(task_id),
        n_envs=1,
        dinov2_model=None,
        dinov2_processor=None,
        sentence_model=None,
        device=device,
        seed=seed,
        max_episode_steps=int(max_episode_steps),
        image_keys=list(OmegaConf.select(cfg, "env.image_keys", default=[PI0_IMAGE_KEY])),
        use_libero_plus=bool(OmegaConf.select(cfg, "env.libero_plus", default=False)),
        init_state_index=OmegaConf.select(cfg, "env.init_state_index", default="auto"),
        settle_steps=int(OmegaConf.select(cfg, "env.settle_steps", default=10)),
    )
    return env


def _eval_task(pi0_wrapper, env_name, task_id, cfg, device, output_dir, seed: int, max_episode_steps: int,
               cycler=None) -> dict:
    """Run all episodes for one task and return a metrics dict.

    Without ``cycler`` the env is built once and reused (classic behaviour). With one, every episode
    draws a different perturbation variant of the same family and the env is REBUILT for it (~1-2 s):
    a variant is a distinct MuJoCo scene -- different bddl, robot model, or post-render corruption --
    so it cannot be swapped in at reset. ``task_id`` is then the BASE task id.
    """
    action_exec_len = int(OmegaConf.select(cfg, "pi0.action_exec_len", default=20))
    num_episodes = int(OmegaConf.select(cfg, "eval.num_episodes", default=20))
    record_video = bool(OmegaConf.select(cfg, "eval.record_video", default=False))
    video_first_n = int(OmegaConf.select(cfg, "eval.video_first_n_episodes", default=1))

    successes, steps_l, rewards_l, episodes = [], [], [], []
    language_instruction = None
    env, task_info = None, {}
    try:
        if cycler is None:
            env = _build_env(cfg, env_name, task_id, device, seed, max_episode_steps)
            task_info = getattr(env, "libero_task_info", None) or {}
        action_dim = None

        for ep in tqdm(range(num_episodes), desc=f"{env_name} task {task_id}", unit="ep"):
            if cycler is not None:
                variant = cycler.next()
                if env is not None:
                    env.close()
                env = _build_env(cfg, env_name, variant.task_id, device, seed, max_episode_steps)
                task_info = getattr(env, "libero_task_info", None) or {}
            if action_dim is None:
                action_dim = int(env.single_action_space.shape[0])

            frames = [] if (record_video and ep < video_first_n) else None
            success, steps, reward, prompt = _run_episode(env, pi0_wrapper, action_exec_len, action_dim, frames)
            if language_instruction is None:
                language_instruction = prompt
            successes.append(float(success))
            steps_l.append(int(steps))
            rewards_l.append(float(reward))

            # Per-episode record: with per-episode variants this is the only way to trace a failure
            # back to the specific perturbation that caused it.
            pert = task_info.get("perturbation") or {}
            episodes.append({
                "episode": ep,
                "variant_task_id": task_info.get("task_id"),
                "variant_task_name": task_info.get("task_name"),
                "category": pert.get("category"),
                "difficulty_level": pert.get("difficulty_level"),
                "success": bool(success),
                "steps": int(steps),
            })
            if frames:
                _write_video(frames, os.path.join(output_dir, "videos", f"task{task_id}_ep{ep}.mp4"))
    finally:
        if env is not None:
            env.close()

    perturbation = task_info.get("perturbation") or {}
    # Steps for the successful episodes only -- how long the policy takes when it does solve the task.
    success_steps = [s for s, ok in zip(steps_l, successes) if ok]
    result = {
        "task_id": int(task_id),
        "task_name": task_info.get("task_name"),
        # LIBERO-plus only: which perturbation dimension this task exercises (None for plain LIBERO).
        "category": perturbation.get("category"),
        "difficulty_level": perturbation.get("difficulty_level"),
        "seed": int(seed),
        "max_episode_steps": int(max_episode_steps),
        "language_instruction": language_instruction,
        "num_episodes": num_episodes,
        "num_success": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_steps": float(np.mean(steps_l)) if steps_l else 0.0,
        "avg_steps_success": float(np.mean(success_steps)) if success_steps else 0.0,
        "avg_reward": float(np.mean(rewards_l)) if rewards_l else 0.0,
        "episodes": episodes,
    }
    if cycler is not None:
        # In perturbation mode the row describes a (base task, family) cell, not a single variant.
        distinct = {e["variant_task_id"] for e in episodes if e["variant_task_id"] is not None}
        result.update({
            "base_task_id": int(task_id),
            "base_task_name": libero_plus.base_task_name(env_name, int(task_id)),
            "task_name": None,
            "category": None if len({e["category"] for e in episodes}) > 1 else episodes[0]["category"],
            "difficulty_level": None,
            "num_variants": cycler.num_variants,
            "distinct_variants_run": len(distinct),
        })
    return result


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
    # Accepts env.task_ids / env.task_id, env.task_names, or LIBERO-plus perturbation filters.
    task_ids = _resolve_task_ids(cfg)
    # One episode step budget per task: env.max_episode_steps is an int, or a list matching task_ids.
    max_steps_per_task = _resolve_max_episode_steps(cfg, task_ids)

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

    perturbation = _perturbation_spec(cfg)
    num_episodes = int(cfg.eval.num_episodes)
    variant_seed = _resolve_variant_seed(cfg) if perturbation is not None else None
    if perturbation is not None:
        if not bool(OmegaConf.select(cfg, "env.libero_plus", default=False)):
            raise ValueError("env.perturbation requires env.libero_plus=true (it selects LIBERO-plus variants).")
        logger.info(
            f"Evaluating {env_name} BASE tasks {task_ids} under perturbation {perturbation} "
            f"| {num_episodes} episodes each, a different variant per episode "
            f"| max_episode_steps={_steps_note(max_steps_per_task)} "
            f"| action_exec_len={int(cfg.pi0.action_exec_len)}"
        )
    else:
        logger.info(
            f"Evaluating {env_name} tasks {task_ids} | {num_episodes} episodes each "
            f"| max_episode_steps={_steps_note(max_steps_per_task)} "
            f"| action_exec_len={int(cfg.pi0.action_exec_len)}"
        )

    results = []
    # strict: _resolve_max_episode_steps guarantees one budget per task id.
    for idx, (task_id, task_max_steps) in enumerate(zip(task_ids, max_steps_per_task, strict=True)):
        label = "base task" if perturbation is not None else "task"
        logger.info(
            f"===== [{idx + 1}/{len(task_ids)}] {env_name} {label} {task_id} "
            f"(max_episode_steps={task_max_steps}) ====="
        )
        cycler = (
            _make_cycler(cfg, env_name, task_id, num_episodes, variant_seed)
            if perturbation is not None else None
        )
        r = _eval_task(pi0_wrapper, env_name, task_id, cfg, device, output_dir, seed, task_max_steps,
                       cycler=cycler)
        results.append(r)
        variants_note = (
            f" over {r['distinct_variants_run']}/{r['num_variants']} variants"
            if perturbation is not None else ""
        )
        logger.success(
            f"{label} {task_id} ({r['language_instruction']!r}): success_rate={r['success_rate']:.1%} "
            f"({r['num_success']}/{r['num_episodes']}){variants_note} avg_steps={r['avg_steps']:.1f} "
            f"avg_steps_success={r['avg_steps_success']:.1f}"
        )
        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "task_id": r["task_id"],
                    "success_rate": r["success_rate"],
                    "avg_steps": r["avg_steps"],
                    "avg_steps_success": r["avg_steps_success"],
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
    variants_col = perturbation is not None
    steps_col = len(set(max_steps_per_task)) > 1  # per-task budgets differ -> show them
    header = f"{'task_id':>10} | {'success':>8} | {'succ/eps':>9} | {'avg_steps':>9} | {'steps_succ':>10}"
    header += f" | {'max_steps':>9}" if steps_col else ""
    logger.info(header + (f" | {'variants':>9}" if variants_col else ""))
    logger.info("-" * 60)
    for r in results:
        row = (
            f"{r['task_id']:>10} | {r['success_rate']:>7.1%} | "
            f"{r['num_success']:>4}/{r['num_episodes']:<4} | {r['avg_steps']:>9.1f} | "
            f"{r['avg_steps_success']:>10.1f}"
        )
        if steps_col:
            row += f" | {r['max_episode_steps']:>9}"
        if variants_col:
            row += f" | {r['distinct_variants_run']:>4}/{r['num_variants']:<4}"
        logger.info(row)
    logger.info("-" * 60)
    logger.info(
        f"{'MEAN':>10} | {overall_sr:>7.1%} | {total_succ:>4}/{total_eps:<4} | "
        f"(micro success {total_succ / max(total_eps, 1):.1%})"
    )
    logger.info("=" * 60)

    # ---- Per-perturbation breakdown (LIBERO-plus only: that is what the benchmark is for). ----
    per_category = {}
    for r in results:
        # Prefer per-episode records: under env.perturbation each episode is its own variant (and with
        # perturbation="all" they can even span families), so the task row has no single category.
        for e in r.get("episodes") or []:
            if e.get("category") is None:
                continue
            agg = per_category.setdefault(
                e["category"], {"num_success": 0, "num_episodes": 0, "num_tasks": 0, "_tasks": set()}
            )
            agg["num_success"] += int(bool(e["success"]))
            agg["num_episodes"] += 1
            agg["_tasks"].add(r["task_id"])
    for agg in per_category.values():
        agg["num_tasks"] = len(agg.pop("_tasks"))
    if per_category:
        for agg in per_category.values():
            agg["success_rate"] = agg["num_success"] / max(agg["num_episodes"], 1)
        logger.info(f"{'perturbation':>22} | {'success':>8} | {'succ/eps':>9} | {'tasks':>5}")
        logger.info("-" * 60)
        for cat in sorted(per_category, key=lambda c: per_category[c]["success_rate"]):
            agg = per_category[cat]
            logger.info(
                f"{cat:>22} | {agg['success_rate']:>7.1%} | "
                f"{agg['num_success']:>4}/{agg['num_episodes']:<4} | {agg['num_tasks']:>5}"
            )
        logger.info("=" * 60)
        if wandb_logger is not None:
            wandb_logger.log(
                {f"success_rate/{cat}": agg["success_rate"] for cat, agg in per_category.items()},
                step=len(task_ids),
                prefix="eval",
            )

    # Legend: task_id -> language instruction (kept out of the numeric table to preserve alignment).
    logger.info("Task instructions:")
    for r in results:
        logger.info(f"  {r['task_id']:>3}: {r['language_instruction']!r}")

    # Persist results for programmatic use.
    summary = {
        "env_name": env_name,
        "task_ids": task_ids,
        # Resolved per task, aligned with task_ids (env.max_episode_steps may be a scalar or a list).
        "max_episode_steps": max_steps_per_task,
        "libero_plus": bool(OmegaConf.select(cfg, "env.libero_plus", default=False)),
        "init_state_index": OmegaConf.select(cfg, "env.init_state_index", default="auto"),
        # Perturbation mode: task_ids above are BASE task ids and each episode ran a different variant.
        "perturbation": perturbation,
        "variant_sampling": OmegaConf.select(cfg, "env.variant_sampling", default="shuffle"),
        # The seed actually used (random when env.variant_seed was null) -- pass it back in to replay.
        "variant_seed": variant_seed,
        "per_category": per_category,
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
