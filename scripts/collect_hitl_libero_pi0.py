#!/usr/bin/env python3
"""Collect human-in-the-loop rollouts with a pi0 / pi0.5 policy on a LIBERO task.

Runs pi0 (or pi0.5) autonomously on a LIBERO task and lets a human take over with keyboard /
SpaceMouse teleoperation (LIBERO is robosuite-based). Every executed step is labelled
intervention=0 (policy) or 1 (human correction), and the kept rollouts are written to a
robomimic-style HDF5 dataset that stores the raw pi0-format observations:

    /data/demo_{i}/actions             [N, 7]      env-space actions actually executed
    /data/demo_{i}/rewards             [N]
    /data/demo_{i}/dones               [N]         1 at the terminal step
    /data/demo_{i}/intervention        [N]         per-step HITL label (0=policy, 1=human)
    /data/demo_{i}/obs/image           [N,224,224,3] uint8  (pi0 ``observation/image``)
    /data/demo_{i}/obs/wrist_image     [N,224,224,3] uint8  (pi0 ``observation/wrist_image``)
    /data/demo_{i}/obs/state           [N, 8]       (pi0 ``observation/state``)
    /data/demo_{i}.attrs["prompt"]     the LIBERO language instruction
    /data/demo_{i}.attrs["variant_*"]  LIBERO-plus perturbation variant this demo ran (env.perturbation)
    /meta.attrs["info"]                JSON provenance (pi0 checkpoint, task, action_exec_len, ...)

Observations are stored RAW (pi0 normalizes internally), so the dataset can later be exported to a
LeRobot dataset for pi0/pi0.5 fine-tuning, or loaded into the in-house buffers.

Usage (local machine with a display for the teleop window):
    uv run python scripts/collect_hitl_libero_pi0.py --config-name libero_collect_hitl \
        env.env_name=libero_90 env.task_id=57 \
        pi0.checkpoint=gs://openpi-assets/checkpoints/pi05_libero/ \
        teleop.device=keyboard hitl.collect_num_rollouts=50

Collecting on PERTURBED LIBERO-plus tasks (see docs/LIBERO_PLUS.md). With ``env.perturbation`` set,
``env.task_id`` is a BASE task id and every rollout runs a different variant of that family:
    uv run python scripts/collect_hitl_libero_pi0.py --config-name libero_collect_hitl \
        env.libero_plus=true env.env_name=libero_spatial env.task_id=3 env.perturbation=camera \
        teleop.device=keyboard hitl.collect_num_rollouts=20

To pin ONE specific variant instead, give its exact name and leave env.perturbation null:
    uv run python scripts/collect_hitl_libero_pi0.py --config-name libero_collect_hitl \
        env.libero_plus=true env.env_name=libero_spatial env.init_state_index=cycle \
        env.task_name=pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_light_1 \
        teleop.device=keyboard hitl.collect_num_rollouts=20

Controls:
  * Keyboard:   Tab = take/release control, wasd/rf + zx/tg/cv to move, space = gripper,
                q = abort episode, ESC = finish collection early (save & exit).
  * SpaceMouse: Tab = take/release control, move/twist the puck to move, left button = gripper,
                right button = abort episode, ESC = finish collection early (save & exit).
"""

import os

# JAX/env setup must happen before importing JAX or any GPU libs (mirrors scripts/eval_pi0.py).
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
# pynput keyboard listener used by the takeover toggle.
import cv2  # noqa: F401

import json
import time
from datetime import datetime

import h5py
import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from robometer_policy_learning.utils.logging_compat import get_logger, setup_loguru_logging
from robometer_policy_learning.envs import libero_plus
from robometer_policy_learning.envs.dsrl_env_wrappers import setup_libero_env
from robometer_policy_learning.utils.hitl_utils import describe_control_mode
from robometer_policy_learning.utils.pi0_hitl_utils import (
    PI0_IMAGE_KEY,
    PI0_STATE_KEY,
    PI0_WRIST_KEY,
    Pi0LiberoHitlWorker,
)
from robometer_policy_learning.utils.pi0_integration import Pi0Wrapper

logger = get_logger()

# pi0 obs key -> flattened HDF5 dataset name under /data/demo_i/obs.
_OBS_KEY_TO_H5 = {PI0_IMAGE_KEY: "image", PI0_WRIST_KEY: "wrist_image", PI0_STATE_KEY: "state"}


def _perturbation_spec(cfg):
    """``env.perturbation`` normalized to a list of family names, or None when unset."""
    spec = OmegaConf.select(cfg, "env.perturbation", default=None)
    if spec is None:
        return None
    if isinstance(spec, (ListConfig, list, tuple)):
        spec = [str(x) for x in spec]
        return spec or None
    return [str(spec)]


def _build_env(cfg, device, seed, task_id=None, task_name=None):
    """Build the single LIBERO(-plus) env (no DINO / sentence models: pi0 takes raw images + prompt)."""
    return setup_libero_env(
        task_suite_name=cfg.env.env_name,
        task_id=int(cfg.env.task_id) if task_id is None else int(task_id),
        n_envs=1,
        dinov2_model=None,
        dinov2_processor=None,
        sentence_model=None,
        device=device,
        seed=seed,
        max_episode_steps=int(cfg.env.max_episode_steps),
        image_keys=list(OmegaConf.select(cfg, "env.image_keys", default=[PI0_IMAGE_KEY])),
        # LIBERO-plus: perturbed task variants + benchmark init states (see docs/LIBERO_PLUS.md).
        use_libero_plus=bool(OmegaConf.select(cfg, "env.libero_plus", default=False)),
        task_name=task_name,
        init_state_index=OmegaConf.select(cfg, "env.init_state_index", default="auto"),
        settle_steps=int(OmegaConf.select(cfg, "env.settle_steps", default=10)),
    )[0]


def _resolve_variant_seed(cfg) -> int:
    """``env.variant_seed``, or a fresh random one when it is null.

    Logged and stored in the dataset's meta info -- rerun with ``env.variant_seed=<value>`` to replay
    the same perturbation sequence.
    """
    variant_seed = OmegaConf.select(cfg, "env.variant_seed", default=None)
    if variant_seed is None:
        variant_seed = int(time.time_ns() % (1 << 31))
        logger.info(f"No env.variant_seed provided, selected variant_seed={variant_seed}")
    return int(variant_seed)


def _make_cycler(cfg, variant_seed: int):
    """Variant pool + sampler for the configured (BASE task, perturbation family) cell."""
    base_id = int(cfg.env.task_id)
    levels = OmegaConf.select(cfg, "env.variant_difficulty_levels", default=None)
    variants = libero_plus.variants_of(
        str(cfg.env.env_name),
        base_id,
        categories=_perturbation_spec(cfg),
        difficulty_levels=[int(x) for x in levels] if levels else None,
        name_contains=OmegaConf.select(cfg, "env.variant_name_contains", default=None),
    )
    cycler = libero_plus.VariantCycler(
        variants,
        seed=int(variant_seed),
        sampling=str(OmegaConf.select(cfg, "env.variant_sampling", default="shuffle")),
    )
    logger.info(
        f"Perturbation sampling: base task {base_id} "
        f"({libero_plus.base_task_name(str(cfg.env.env_name), base_id)}) | "
        f"{cycler.num_variants} variants [{', '.join(sorted({v.category for v in variants}))}] | "
        f"sampling={cycler.sampling} seed={cycler.seed}"
    )
    return cycler


def _write_h5(episodes, output_path, *, action_min, action_max, meta_extra, episode_variants=None):
    """Write collected HITL episodes (lists of per-step transition dicts) to a robomimic-style HDF5."""
    if not episodes:
        raise RuntimeError("No episodes to write.")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    total, intervention_total = 0, 0
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        for i, trs in enumerate(episodes):
            n = len(trs)
            g = data_grp.create_group(f"demo_{i}")
            g.create_dataset("actions", data=np.stack([t["action"].reshape(-1) for t in trs]).astype(np.float32))
            g.create_dataset("rewards", data=np.asarray([t["reward"] for t in trs], dtype=np.float32))
            g.create_dataset("dones", data=np.asarray([float(t["done"] or t["truncated"]) for t in trs], dtype=np.float32))
            interv = np.asarray([int(t["intervention"]) for t in trs], dtype=np.int64)
            g.create_dataset("intervention", data=interv)

            # Flow-MILE: per-frame frozen-rollout baseline pool [N, P, H, action_dim] (env-space),
            # precomputed from the collection policy. Present only when hitl.rollout_pool_size > 0.
            if "rollout_samples" in trs[0]:
                pool = np.stack([np.asarray(t["rollout_samples"], dtype=np.float32) for t in trs], axis=0)
                g.create_dataset("rollout_samples", data=pool, compression="gzip")

            obs_grp = g.create_group("obs")
            for pi0_key, h5_name in _OBS_KEY_TO_H5.items():
                if pi0_key not in trs[0]["obs"]:
                    continue
                seq = np.stack([np.asarray(t["obs"][pi0_key]) for t in trs], axis=0)
                kwargs = {"compression": "gzip"} if seq.ndim >= 3 else {}
                obs_grp.create_dataset(h5_name, data=seq, **kwargs)

            g.attrs["num_samples"] = n
            g.attrs["prompt"] = str(trs[0].get("prompt", ""))
            # Which perturbation variant produced THIS demo. Under env.perturbation every rollout runs a
            # different variant, so this cannot live in /meta alone.
            if episode_variants is not None and i < len(episode_variants):
                for key, value in episode_variants[i].items():
                    g.attrs[key] = "" if value is None else value
            total += n
            intervention_total += int((interv == 1).sum())

        data_grp.attrs["total"] = total

        meta_grp = f.create_group("meta")
        if action_min is not None and action_max is not None:
            meta_grp.create_dataset("min_action", data=np.asarray(action_min, dtype=np.float32))
            meta_grp.create_dataset("max_action", data=np.asarray(action_max, dtype=np.float32))
        meta_grp.attrs["info"] = json.dumps(meta_extra)

    return {
        "num_demos": len(episodes),
        "total_transitions": total,
        "intervention_transitions": intervention_total,
        "intervention_fraction": (intervention_total / total) if total else 0.0,
    }


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="libero_collect_hitl")
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

    # ---- Load the pi0 / pi0.5 data-collection policy (JAX/openpi). ----
    pi0_checkpoint = cfg.pi0.checkpoint
    action_exec_len = int(OmegaConf.select(cfg, "pi0.action_exec_len", default=20))
    # config_name: set to the LoRA/HITL config (e.g. pi05_libero_hitl_lora) when collecting with a
    # fine-tuned checkpoint whose architecture is not inferable from the path; null => path heuristic.
    pi0_config_name = OmegaConf.select(cfg, "pi0.config_name", default=None)
    logger.info(f"Loading pi0 policy from {pi0_checkpoint} (config_name={pi0_config_name})")
    pi0_wrapper = Pi0Wrapper(pi0_checkpoint, device=str(device), config_name=pi0_config_name)

    # ---- LIBERO env. With env.perturbation set, every rollout runs a DIFFERENT variant of the same
    # perturbation family, which means rebuilding the env per rollout (a variant is a distinct MuJoCo
    # scene). The teleop device and window are owned by the worker and survive the rebuilds. ----
    perturbation = _perturbation_spec(cfg)
    task_name_cfg = OmegaConf.select(cfg, "env.task_name", default=None)
    cycler, variant_seed = None, None
    if perturbation is not None:
        if not bool(OmegaConf.select(cfg, "env.libero_plus", default=False)):
            raise ValueError("env.perturbation requires env.libero_plus=true (it selects LIBERO-plus variants).")
        if task_name_cfg:
            raise ValueError(
                "env.task_name and env.perturbation are mutually exclusive: task_name pins one specific "
                "variant, while env.perturbation samples variants of the BASE task in env.task_id."
            )
        variant_seed = _resolve_variant_seed(cfg)
        cycler = _make_cycler(cfg, variant_seed)

    def build_next_env():
        """Build the env for the next rollout; returns (env, task_info, variant_attrs)."""
        if cycler is None:
            e = _build_env(cfg, device, seed, task_name=task_name_cfg)
        else:
            e = _build_env(cfg, device, seed, task_id=cycler.next().task_id)
        info = getattr(e, "libero_task_info", None) or {}
        pert = info.get("perturbation") or {}
        return e, info, {
            "variant_task_id": int(info.get("task_id", -1)),
            "variant_task_name": info.get("task_name"),
            "perturbation_category": pert.get("category"),
            "difficulty_level": pert.get("difficulty_level"),
        }

    env, task_info, variant_attrs = build_next_env()
    action_dim = int(env.single_action_space.shape[0])
    # Resolved task provenance (task_name may have overridden task_id; perturbation is LIBERO-plus only).
    logger.info(
        f"LIBERO{'-plus' if task_info.get('libero_plus') else ''} {cfg.env.env_name} "
        f"task {task_info.get('task_id', cfg.env.task_id)} ({task_info.get('task_name')}) | action_dim={action_dim}"
    )
    if task_info.get("perturbation"):
        pinfo = task_info["perturbation"]
        logger.info(f"Perturbation: {pinfo['category']} (difficulty level {pinfo['difficulty_level']})")
    try:
        logger.info(f"Control mode: {describe_control_mode(env)}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not inspect control mode: {e}")

    asp = env.single_action_space
    if asp is not None and np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
        action_min = np.asarray(asp.low, dtype=np.float32)
        action_max = np.asarray(asp.high, dtype=np.float32)
    else:
        action_min = action_max = None

    # ---- HITL teleop worker ----
    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    video_dir = os.path.join(output_dir, "debug_videos") if debug else None
    worker = Pi0LiberoHitlWorker(
        env=env,
        pi0_wrapper=pi0_wrapper,
        action_dim=action_dim,
        action_exec_len=action_exec_len,
        store_only_human=bool(OmegaConf.select(cfg, "hitl.store_only_human", default=False)),
        rollout_pool_size=int(OmegaConf.select(cfg, "hitl.rollout_pool_size", default=0)),
        enable_render=bool(OmegaConf.select(cfg, "teleop.enable_render", default=True)),
        teleop_device=str(OmegaConf.select(cfg, "teleop.device", default="keyboard")),
        takeover_key=str(OmegaConf.select(cfg, "teleop.takeover_key", default="tab")),
        camera=OmegaConf.select(cfg, "teleop.camera", default="agentview"),
        wrist_camera=OmegaConf.select(cfg, "teleop.wrist_camera", default="robot0_eye_in_hand"),
        show_wrist=bool(OmegaConf.select(cfg, "teleop.show_wrist", default=True)),
        record_video=debug,
        video_dir=video_dir,
    )

    # ---- Collection loop: keep collecting until collect_num_rollouts KEPT rollouts. ----
    num_target = int(OmegaConf.select(cfg, "hitl.collect_num_rollouts", default=50))
    keep_only_hitl = bool(OmegaConf.select(cfg, "hitl.keep_only_hitl_rollouts", default=False))
    require_success = bool(OmegaConf.select(cfg, "hitl.require_success", default=True))
    # collect_output_path=null => collect but DO NOT save (dry-run / debugging teleop).
    output_path = OmegaConf.select(cfg, "hitl.collect_output_path", default=None)
    dest = output_path if output_path else "(not saving: hitl.collect_output_path is null)"
    logger.info(
        f"Collecting {num_target} rollouts (require_success={require_success}, "
        f"keep_only_hitl_rollouts={keep_only_hitl}, store_only_human={worker.store_only_human}) -> {dest}"
    )
    get_language_instruction = getattr(env, "get_language_instruction", None)
    task_instruction = get_language_instruction() if callable(get_language_instruction) else getattr(env, "language_instruction", None)
    logger.info(f"Task Instruction: {task_instruction}")

    collected, attempt, num_success = 0, 0, 0
    # Per-KEPT-episode variant provenance, index-aligned with worker.collected_episodes (each kept
    # rollout appends exactly one episode there).
    episode_variants = []
    # Early finish: press ESC in the teleop window at any time to STOP collecting and save what has
    # been collected so far. ESC raises KeyboardInterrupt from the worker's render loop, which we catch
    # here and fall through to the save path below. (The current in-progress rollout is discarded;
    # completed rollouts are kept.)
    finished_early = False
    try:
        while collected < num_target:
            if attempt > 0 and cycler is not None:
                # Next rollout, next perturbation variant: rebuild the env and re-point the worker at
                # it. The variant advances per ATTEMPT, not per kept rollout, so success / intervention
                # filtering cannot bias the collected set toward whichever variants are easiest.
                env.close()
                env, task_info, variant_attrs = build_next_env()
                worker.rebind_env(env)
            steps, human_steps, stored, success = worker.rollout_episode(
                f"ep{attempt}", phase="COLLECT", store=True,
                require_success=require_success, require_intervention=keep_only_hitl,
            )
            attempt += 1
            num_success += int(bool(success))
            collected += int(stored > 0)
            if stored > 0:
                episode_variants.append(dict(variant_attrs))
            variant_note = (
                f" variant={variant_attrs['variant_task_id']} ({variant_attrs['perturbation_category']})"
                if cycler is not None else ""
            )
            logger.info(
                f"  rollout {attempt}: success={bool(success)} steps={steps} human_steps={human_steps} "
                f"stored={stored}{variant_note} | kept {collected}/{num_target}"
            )
    except KeyboardInterrupt:
        finished_early = True
        logger.info("ESC pressed — finishing collection early; writing what has been collected so far.")
    finally:
        worker.close()
        env.close()
    if finished_early:
        logger.info(f"Finished early: kept {collected}/{num_target} target rollouts.")

    if not output_path:
        logger.warning(
            f"hitl.collect_output_path is null; not saving the {len(worker.collected_episodes)} "
            "collected episode(s). Set hitl.collect_output_path=<path.hdf5> to write them."
        )
        return

    if not worker.collected_episodes:
        logger.warning("No episodes collected; nothing written.")
        return

    # store_only_human keeps only human-correction steps: if no interventions were collected (e.g. an
    # early finish before any takeover), there is nothing meaningful to save and an empty correction
    # dataset would break downstream conversion. Skip writing entirely so no file is created (the
    # orchestrator then simply omits this missing path from the export step).
    n_human_total = sum(
        1 for ep in worker.collected_episodes for t in ep if int(t.get("intervention", 0)) == 1
    )
    if worker.store_only_human and n_human_total == 0:
        logger.warning(
            f"store_only_human=true but no human interventions were collected; not writing "
            f"{output_path} (empty correction dataset)."
        )
        return

    meta_extra = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "pi0_checkpoint": str(pi0_checkpoint),
        "env_name": str(cfg.env.env_name),
        "task_id": int(task_info.get("task_id", cfg.env.task_id)),
        # Which LIBERO checkout / perturbed task these corrections came from. Task ids are ambiguous
        # across the two checkouts, so the name + perturbation are what make the dataset traceable.
        "libero_task_info": task_info,
        # Perturbation sampling: task_id above is the BASE task and each rollout ran a different
        # variant of `perturbation` (see the per-demo variant_* attrs for which one).
        "perturbation": perturbation,
        "variant_sampling": str(OmegaConf.select(cfg, "env.variant_sampling", default="shuffle")),
        # The seed actually used (random when env.variant_seed was null) -- pass it back in to replay.
        "variant_seed": variant_seed,
        "variant_pool": [v.task_id for v in cycler.variants] if cycler is not None else None,
        "variant_pool_size": cycler.num_variants if cycler is not None else None,
        "seed": int(seed),
        "max_episode_steps": int(cfg.env.max_episode_steps),
        "action_exec_len": action_exec_len,
        "store_only_human": worker.store_only_human,
        "rollout_pool_size": worker.rollout_pool_size,
        "keep_only_hitl_rollouts": keep_only_hitl,
        "require_success": require_success,
        "obs_are_raw": True,
        "obs_key_map": {"image": PI0_IMAGE_KEY, "wrist_image": PI0_WRIST_KEY, "state": PI0_STATE_KEY},
        "intervention_labels": {"policy": 0, "human": 1},
        "num_attempts": attempt,
        "num_successful": num_success,
    }
    stats = _write_h5(
        worker.collected_episodes, output_path,
        action_min=action_min, action_max=action_max, meta_extra=meta_extra,
        episode_variants=episode_variants if cycler is not None else None,
    )
    logger.success(
        f"Wrote {stats['num_demos']} demos / {stats['total_transitions']} transitions "
        f"({stats['intervention_transitions']} human, {stats['intervention_fraction']:.1%}) to {output_path}"
    )


if __name__ == "__main__":
    main()
