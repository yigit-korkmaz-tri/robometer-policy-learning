#!/usr/bin/env python3
"""Iterative human-in-the-loop training (MILE and HG-DAgger).

Starting from a pretrained policy (and optionally the offline dataset used to train it), each
iteration:
  1. collects rollouts until ``hitl.rollouts_per_iter`` SUCCESSFUL trajectories are stored (failed
     attempts are run but discarded; no attempt cap). Every step is labelled intervention=1 (human
     correction) or 0 (policy). What gets stored to the online buffer is controlled by
     ``hitl.store_only_human``:
        * False (MILE)      -> store ALL transitions (labels 0/1),
        * True  (HG-DAgger) -> store ONLY the human corrections (label 1);
  2. trains the configured algorithm for ``hitl.train_steps_per_iter`` gradient steps over the
     training buffer (online, optionally mixed with offline demos labelled intervention=2 via
     MixedReplayBuffer);
then repeats.

``hitl.offline_mode`` selects the offline-demo source mixed in for anti-forgetting aggregation:
  * null          -> no offline data (train on the online buffer only);
  * "pretraining" -> mix the pretraining demonstration H5 (cfg.env.h5_dataset_path);
  * "warmup"      -> mix a user-provided demo H5 (``hitl.warmup_dataset_path``) as the offline anchor.
                     Same mechanism as "pretraining" but from an arbitrary path, so different
                     algorithms / sweep configs can reuse the SAME fixed set of initial demos.
  * "self"        -> collect ``hitl.self_num_rollouts`` SUCCESSFUL autonomous rollouts from the
                     initial (pretrained) policy and use them as the offline anchor. No external demo
                     source is needed: the policy's own successes act as the anti-forgetting anchor.

Alternatively, set ``hitl.precollected_hitl_dataset`` to a HITL HDF5 written by
``scripts/collect_hitl_rollout.py`` to train on precollected interventions instead of collecting
online. In that mode online collection is skipped (no expert/criterion/online-rollout worker) and
``hitl.num_iterations`` must be 1 (a single train phase over the loaded buffer). It is still
compatible with ``offline_mode='warmup'`` (which just loads its demo H5).

Algorithm is selected by ``alg.offline_alg_name`` (+ ``algorithm@offline_algorithm`` for its
hyperparameters), e.g. ``mile``, ``bc``, or ``dp``. For algorithms with a frozen rollout policy
(MILE), it is snapshotted from the current actor right before each training phase.

Gaussian-actor algorithms (``bc``, ``mile``) and diffusion-policy algorithms (``dp``,
``diffusion_mile``) are interchangeable depending on what the pretrained actor is. ``dp`` is the
diffusion-policy analogue of ``bc`` (denoising-loss cloning), so HG-DAgger / SIRIUS / IWR work on
diffusion policies via ``dp`` exactly as they do on Gaussian policies via ``bc``:

  * MILE:           alg.offline_alg_name=mile  hitl.store_only_human=false   (Gaussian actor)
  * Diffusion MILE: alg.offline_alg_name=diffusion_mile  hitl.store_only_human=false  (DiffusionActor)
  * Flow MILE:      alg.offline_alg_name=flow_mile  hitl.store_only_human=false  (FlowMatchingActor)
  * HG-DAgger:      alg.offline_alg_name=bc|dp  hitl.store_only_human=true
  * SIRIUS / IWR:   alg.offline_alg_name=bc|dp  hitl.store_only_human=false  hitl.reweighting=sirius|iwr
                    + offline_algorithm.use_weighted_bc=true (bc) / use_weighted_dp=true (dp)

``load_dir`` points at a *pretraining run output directory*. The env / training / model / policy
config is read from ``load_dir/.hydra/config.yaml`` so it matches the loaded model exactly, and the
actor weights are read from ``load_dir/checkpoints/<step>/actor.pt`` (``checkpoint=<step>`` selects
which; default = latest). This HITL config only carries the HITL algorithm, ``debug``, ``load_dir``
(+ ``checkpoint``), and the ``hitl`` / ``teleop`` / ``logging`` / ``eval`` sections.

Usage (local machine with a display, keyboard teleop):
    uv run python scripts/train_hitl.py --config-name robomimic_image_hitl \
        load_dir=/path/to/outputs/2026-06-04/16-21-35
    # pick a specific checkpoint step:
    uv run python scripts/train_hitl.py --config-name robomimic_image_hitl \
        load_dir=/path/to/run checkpoint=50000
    # HG-DAgger:
    uv run python scripts/train_hitl.py --config-name robomimic_image_hitl \
        load_dir=/path/to/run algorithm@offline_algorithm=bc alg.offline_alg_name=bc \
        hitl.store_only_human=true

Teleop device is selected by ``teleop.device`` (``keyboard`` or ``spacemouse``); the SpaceMouse
needs the ``hidapi`` package and a connected 3D mouse. The takeover toggle stays on the keyboard
for both devices.

Controls:
  * Keyboard:   Tab = take/release control, wasd/rf + zx/tg/cv to move, space = gripper,
                q = abort episode, ESC = quit.
  * SpaceMouse: Tab = take/release control, move/twist the puck to move, left button = gripper,
                right button = abort episode, ESC = quit.
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2  # noqa: F401  (import order matters; keep above torch)

import copy
from datetime import datetime

import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robometer.utils.logger import get_logger, setup_loguru_logging
from robometer_policy_learning.algorithms.bc import BCConfig
from robometer_policy_learning.algorithms.dp import DPConfig
from robometer_policy_learning.algorithms.dp.modeling_dp import DiffusionActor
from robometer_policy_learning.algorithms.diffusion_mile import DiffusionMILEConfig
from robometer_policy_learning.algorithms.flow_matching import FlowMatchingConfig
from robometer_policy_learning.algorithms.flow_matching.modeling_flow import FlowMatchingActor
from robometer_policy_learning.algorithms.flow_mile import FlowMILEConfig
from robometer_policy_learning.algorithms.mile import MILEConfig
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.mixed_replay_buffer import MixedReplayBuffer
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env
from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer_policy_learning.rollouts.evaluation_worker import BatchEvaluationWorker
from robometer_policy_learning.utils.hitl_utils import (
    HitlRolloutWorker,
    analyze_intervention_segments,
    compute_iwr_weights,
    compute_sirius_weights,
    describe_control_mode,
    get_intervention_criteria,
    load_precollected_hitl,
)
from robometer_policy_learning.utils.training_utils import save_checkpoint

logger = get_logger()

# Algorithms usable for HITL training. "bc"/"mile" need a Gaussian actor (explicit action
# distribution); "dp"/"diffusion_mile" need a DiffusionActor (DP-pretrained); "flow_mile" needs a
# FlowMatchingActor (Flow-Matching-pretrained). "dp" is the diffusion analogue of "bc" for
# HG-DAgger / SIRIUS / IWR (plain or weighted denoising cloning); "diffusion_mile" / "flow_mile"
# additionally score actions via the denoising-loss / flow-matching-loss log-prob proxy.
ALG_TO_CONFIG = {
    "bc": BCConfig,
    "mile": MILEConfig,
    "diffusion_mile": DiffusionMILEConfig,
    "flow_mile": FlowMILEConfig,
    "dp": DPConfig,
    "flow": FlowMatchingConfig,
}

# Algorithms that perform (optionally weighted) behavior cloning — usable for HG-DAgger and for
# SIRIUS / IWR reweighting. "bc" is Gaussian; "dp" is the diffusion-policy analogue.
BC_STYLE_ALGS = {"bc", "dp", "flow"}

# MILE-style algorithms: their probit intervention (BCE) loss needs BOTH policy (label 0) and
# human-correction (label 1) transitions, so they are incompatible with store_only_human=true.
MILE_STYLE_ALGS = {"mile", "diffusion_mile", "flow_mile"}

# Offline-demo intervention label (online labels 0=policy / 1=human are set by the rollout worker).
LABEL_OFFLINE = 2
# Robot (autonomous-policy) intervention label — same as the online policy label set by the worker.
LABEL_ROBOT = 0


def _load_pretrain_cfg(load_dir: str) -> DictConfig:
    """Load the Hydra config saved by the pretraining run at ``load_dir/.hydra/config.yaml``."""
    cfg_path = os.path.join(load_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Pretraining config not found at {cfg_path}. load_dir must be a pretraining run output "
            "directory (it should contain .hydra/config.yaml and checkpoints/<step>/actor.pt)."
        )
    return OmegaConf.load(cfg_path)


def _resolve_checkpoint_dir(load_dir: str, checkpoint=None) -> str:
    """Resolve the checkpoint directory (containing ``actor.pt``) inside a pretraining run dir.

    Expected layout is ``load_dir/checkpoints/<step>/actor.pt``. ``checkpoint`` picks a specific
    ``<step>`` (e.g. ``50000`` or ``latest``); when None, prefer ``latest`` else the largest numeric
    step. For backward compatibility, a ``load_dir`` that directly contains ``actor.pt`` is returned
    as-is.
    """
    if os.path.exists(os.path.join(load_dir, "actor.pt")):
        return load_dir
    ckpt_root = os.path.join(load_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"No 'checkpoints/' directory under {load_dir} (and no actor.pt directly in it).")
    if checkpoint is not None:
        cand = os.path.join(ckpt_root, str(checkpoint))
        if not os.path.exists(os.path.join(cand, "actor.pt")):
            raise FileNotFoundError(f"actor.pt not found in {cand} (checkpoint={checkpoint!r}).")
        return cand
    steps = [d for d in os.listdir(ckpt_root) if os.path.exists(os.path.join(ckpt_root, d, "actor.pt"))]
    if not steps:
        raise FileNotFoundError(f"No '<step>/actor.pt' checkpoints found under {ckpt_root}.")
    if "latest" in steps:
        chosen = "latest"
    else:
        numeric = [d for d in steps if d.isdigit()]
        chosen = max(numeric, key=int) if numeric else sorted(steps)[-1]
    return os.path.join(ckpt_root, chosen)


def _run_identity(path) -> str:
    """Machine-independent identity of a run dir for provenance matching.

    The dataset's ``/meta`` stores the collection ``load_dir`` as the absolute path on the machine it
    was collected on, but the same run lives under a different prefix on another machine. So compare
    only the tail from the last ``checkpoints`` path segment onward (e.g.
    ``checkpoints/square_low_dim/<timestamp>``), which is stable across machines. Falls back to the
    last two path components when there is no ``checkpoints`` segment.
    """
    parts = os.path.normpath(str(path)).split(os.sep)
    parts = [p for p in parts if p not in ("", ".")]
    if "checkpoints" in parts:
        idx = len(parts) - 1 - parts[::-1].index("checkpoints")  # last 'checkpoints' segment
        return "/".join(parts[idx:])
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else str(path))


def _read_precollected_meta(path: str) -> dict:
    """Return the provenance metadata dict written under ``/meta`` by collect_hitl_rollout.py (or {})."""
    import json

    import h5py

    try:
        with h5py.File(path, "r") as f:
            if "meta" in f and "info" in f["meta"].attrs:
                return json.loads(f["meta"].attrs["info"])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read /meta from precollected dataset {path}: {e}")
    return {}


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = HydraConfig.get().runtime.output_dir
    save_dir = os.path.join(output_dir, "checkpoints")

    # Setup loguru logging (console + file sink under the Hydra output dir).
    log_level = OmegaConf.select(cfg, "logging.log_level", default="INFO")
    setup_loguru_logging(log_level=log_level, output_dir=output_dir)

    # Setup wandb logger (same pattern as training_utils.setup_training).
    string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{cfg.logging.wandb_name}_{string_time}"
    wandb_logger = WandbLogger(
        exp_name=exp_name,
        offline=cfg.logging.wandb_offline,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        log_dir=f"{cfg.logging.wandb_log_dir_base}/{string_time}",
        prefix="offline",
    )

    # ---- Align with the loaded model: adopt env / training / model / policy from the pretraining
    # run's saved Hydra config so the env build, chunking, obs normalization and architecture match
    # exactly what the actor was trained with. 
    load_dir = OmegaConf.select(cfg, "load_dir", default=None)
    if not load_dir:
        raise ValueError(
            "Set load_dir=<pretraining run dir> (contains .hydra/config.yaml and checkpoints/<step>/actor.pt)."
        )
    pre_cfg = _load_pretrain_cfg(load_dir)
    OmegaConf.set_struct(cfg, False)
    for key in ("env", "training", "model", "policy"):
        if key in pre_cfg:
            cfg[key] = pre_cfg[key]
    OmegaConf.set_struct(cfg, True)
    OmegaConf.resolve(cfg)
    logger.info(f"Adopted env/training/model/policy from {os.path.join(load_dir, '.hydra', 'config.yaml')}")
    wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

    alg_name = str(OmegaConf.select(cfg, "alg.offline_alg_name", default="mile")).lower()
    if alg_name not in ALG_TO_CONFIG:
        raise ValueError(f"Unknown HITL algorithm '{alg_name}'. Choose from {sorted(ALG_TO_CONFIG)}.")

    # ---- Precollected HITL dataset: when set, train on a saved HITL HDF5 (collect_hitl_rollout.py)
    # instead of collecting interventions online. There is no online loop to iterate, so require
    # num_iterations == 1; the online collection (env/expert/criterion/worker) is skipped entirely. ----
    precollected_hitl_dataset = OmegaConf.select(cfg, "hitl.precollected_hitl_dataset", default=None)
    num_iterations = int(OmegaConf.select(cfg, "hitl.num_iterations", default=10))
    if precollected_hitl_dataset:
        if not os.path.exists(precollected_hitl_dataset):
            raise FileNotFoundError(f"hitl.precollected_hitl_dataset not found: {precollected_hitl_dataset}")
        assert num_iterations == 1, (
            "hitl.precollected_hitl_dataset is set, so there is no online collection to iterate over; "
            f"set hitl.num_iterations=1 (got {num_iterations})."
        )
        # Verify the dataset was collected with the SAME data-collection policy as this run's
        # load_dir, so the policy we train/snapshot is the one that produced the interventions.
        _meta = _read_precollected_meta(precollected_hitl_dataset)
        _ds_load_dir = _meta.get("load_dir")
        if _ds_load_dir is None:
            raise ValueError(
                f"Precollected dataset {precollected_hitl_dataset} has no 'load_dir' in its /meta; "
                "cannot verify it was collected with the same policy as load_dir."
            )
        elif _run_identity(_ds_load_dir) != _run_identity(load_dir):
            raise ValueError(
                "Precollected HITL dataset was collected with a different policy than the configured "
                f"load_dir:\n  dataset load_dir = {_ds_load_dir}\n  config  load_dir = {load_dir}\n"
                "These must match so the trained/rollout policy is the one that produced the "
                "interventions. Set load_dir to the dataset's collection run (or recollect the data)."
            )
        else:
            # Same run: also require the same checkpoint step so the trained/rollout policy weights
            # match the ones that produced the interventions (both null/latest is fine).
            _ds_ckpt = _meta.get("checkpoint")
            _cfg_ckpt = OmegaConf.select(cfg, "checkpoint", default=None)
            if str(_ds_ckpt) != str(_cfg_ckpt):
                raise ValueError(
                    "Precollected HITL dataset was collected with a different checkpoint than the "
                    f"configured one:\n  dataset checkpoint = {_ds_ckpt}\n  config  checkpoint = {_cfg_ckpt}\n"
                    "These must match (or both be null/latest) so the trained/rollout policy weights "
                    "are the ones that produced the interventions. Set checkpoint to the dataset's "
                    "collection step (or recollect the data)."
                )
            logger.info(f"Precollected dataset matches config load_dir={load_dir}, checkpoint={_cfg_ckpt}.")
        logger.info(f"Precollected HITL dataset: {precollected_hitl_dataset} (online collection skipped).")

    # ---- Pretrained actor (a deployable BaseActor pickled by Algorithm.save) ----
    ckpt_dir = _resolve_checkpoint_dir(load_dir, OmegaConf.select(cfg, "checkpoint", default=None))
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor.pt not found in {actor_path}.")
    actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
    actor.train()
    logger.info(f"Loaded pretrained actor {type(actor).__name__} from {actor_path}")
    if alg_name in ("diffusion_mile", "dp"):
        assert isinstance(actor, DiffusionActor), f"alg '{alg_name}' requires a DiffusionActor (DP-pretrained)"
    elif alg_name == "flow_mile":
        assert isinstance(actor, FlowMatchingActor), f"alg '{alg_name}' requires a FlowMatchingActor (Flow-Matching-pretrained)"
    elif alg_name in ("mile", "bc"):
        assert not isinstance(actor, (DiffusionActor, FlowMatchingActor)), f"alg '{alg_name}' requires a Gaussian actor"

    if OmegaConf.select(cfg, "env.dino_image_keys", default=None):
        raise NotImplementedError("DINO-embedding (Mode A) policies are not supported by the HITL collector yet.")

    # Keys to drop from stored/observed obs (match what the actor was trained with).
    remove_obs_keys = list(getattr(actor, "remove_obs_keys", None) or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or [])

    # ---- Single, non-chunked teleop/collection env (the HITL worker manages chunking manually).
    # Autonomous eval uses a separate EvaluationWorker + env (built below) so its videos are logged. ----
    env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path,
        n_envs=1,
        device=device,
        seed=OmegaConf.select(cfg, "hitl.seed", default=0),
        max_episode_steps=cfg.env.max_episode_steps,
        use_full_state=cfg.env.use_full_state,
        terminate_on_success=True,
        chunk_size=None,
        n_action_steps=1,
    )
    action_dim = int(env.single_action_space.shape[0])
    logger.info(f"Control mode: {describe_control_mode(env)}")

    # Action normalization bounds (buffers map stored env-space actions to [-1, 1] at sample time).
    asp = env.single_action_space
    if asp is not None and np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
        action_min = np.asarray(asp.low, dtype=np.float32)
        action_max = np.asarray(asp.high, dtype=np.float32)
    else:
        action_min = action_max = None

    chunk_size = OmegaConf.select(cfg, "training.chunk_size", default=None)
    n_exec = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)
    normalize_lowdim = bool(OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False))
    # Offline-data mode for aggregation (anti-forgetting). Four options:
    #   null         -> no offline data (train on the online buffer only)
    #   "pretraining"-> mix the pretraining demonstration H5 (cfg.env.h5_dataset_path)
    #   "warmup"     -> mix a user-provided demo H5 (hitl.warmup_dataset_path) as the offline anchor
    #                   buffer (lets different algorithms/sweeps reuse the same fixed demo set).
    #   "self"       -> collect hitl.self_num_rollouts SUCCESSFUL autonomous rollouts from the initial
    #                   policy and use them as the offline anchor (no external demo source needed).
    _offline_raw = OmegaConf.select(cfg, "hitl.offline_mode", default=None)
    offline_mode = str(_offline_raw).strip().lower()
    if offline_mode in ("none", "null", ""):
        offline_mode = None
    if offline_mode not in (None, "pretraining", "warmup", "self"):
        raise ValueError(
            f"hitl.offline_mode must be null, 'pretraining', 'warmup', or 'self' (got {_offline_raw!r})."
        )
    use_offline = offline_mode is not None
    store_only_human = bool(OmegaConf.select(cfg, "hitl.store_only_human", default=False))
    # MILE: split each trajectory into contiguous same-label segments (separate episode ids) so the
    # chunked sampler yields label-homogeneous chunks (no trailing-policy dilution) while keeping both
    # classes for the probit. Applies to the precollected load and the online collection worker.
    segment_by_intervention = bool(OmegaConf.select(cfg, "hitl.segment_by_intervention", default=False))
    # When true, online collection keeps only successful rollouts that contain >= 1 human
    # intervention (discarding intervention-free successes); false keeps all successful rollouts.
    keep_only_hitl_rollouts = bool(OmegaConf.select(cfg, "hitl.keep_only_hitl_rollouts", default=False))
    if store_only_human and alg_name in MILE_STYLE_ALGS:
        raise ValueError(
            f"hitl.store_only_human=true is incompatible with alg.offline_alg_name='{alg_name}'. "
            f"MILE-style algorithms {sorted(MILE_STYLE_ALGS)} need both policy (intervention=0) and "
            "human-correction (intervention=1) transitions for the probit intervention loss; storing "
            "only human steps leaves the BCE term degenerate. Set hitl.store_only_human=false, or use "
            "alg.offline_alg_name=bc for HG-DAgger-style human-only cloning."
        )

    # ---- Intervention source: keyboard, spacemouse, or a simulated human (expert + criterion).
    # Skipped entirely in precollected-dataset mode (no online collection). ----
    human_mode = str(OmegaConf.select(cfg, "hitl.human_mode", default="real")).lower()
    expert_policy, intervention_criteria = None, None
    if not precollected_hitl_dataset:
        if human_mode == "simulated":
            expert_dir = OmegaConf.select(cfg, "hitl.expert_load_dir", default=None)
            if not expert_dir:
                raise ValueError("hitl.human_mode=simulated requires hitl.expert_load_dir (a pretraining run dir or checkpoint dir).")
            expert_ckpt_dir = _resolve_checkpoint_dir(expert_dir, OmegaConf.select(cfg, "hitl.expert_checkpoint", default=None))
            expert_path = os.path.join(expert_ckpt_dir, "actor.pt")
            if not os.path.exists(expert_path):
                raise FileNotFoundError(f"actor.pt not found in {expert_path} (simulated-human expert).")
            expert_policy = torch.load(expert_path, map_location=device, weights_only=False).to(device)
            criteria_name = str(OmegaConf.select(cfg, "hitl.intervention_criteria", default="never"))
            # The denoising-score / flow-matching-score criteria need DiffusionActors / FlowMatchingActors
            # for BOTH the expert (judge) and the data-collection policy (the loaded ``actor``); fail
            # early with a clear message.
            if criteria_name.startswith("diffusion_mile"):
                for role, pol in (("expert", expert_policy), ("rollout/actor", actor)):
                    if not isinstance(pol, DiffusionActor):
                        raise TypeError(
                            f"hitl.intervention_criteria='{criteria_name}' requires a DiffusionActor {role}, "
                            f"got {type(pol).__name__}. Use a DP-pretrained policy, or pick a 'mile'/'mile_window' "
                            "criterion for Gaussian actors."
                        )
            elif criteria_name.startswith("flow_mile"):
                for role, pol in (("expert", expert_policy), ("rollout/actor", actor)):
                    if not isinstance(pol, FlowMatchingActor):
                        raise TypeError(
                            f"hitl.intervention_criteria='{criteria_name}' requires a FlowMatchingActor {role}, "
                            f"got {type(pol).__name__}. Use a Flow-Matching-pretrained policy, or pick a "
                            "'mile'/'mile_window' criterion for Gaussian actors."
                        )
            elif criteria_name.startswith("mile"):
                for role, pol in (("expert", expert_policy), ("rollout/actor", actor)):
                    if isinstance(pol, (DiffusionActor, FlowMatchingActor)):
                        raise TypeError(
                            f"hitl.intervention_criteria='{criteria_name}' requires a Gaussian Actor {role}, "
                            f"got {type(pol).__name__}. Use a Gaussian actor, or pick a 'diffusion_mile'/'flow_mile' "
                            "criterion for DiffusionActor/FlowMatchingActor experts."
                        )
            # NOTE: the intervention-hold is owned by the rollout worker (so it can skip the work a held
            # step would discard), not by the criterion wrapper — so it is NOT passed here.
            criteria_kwargs = dict(
                window=int(OmegaConf.select(cfg, "hitl.criteria_window", default=10)),
                intervention_cost=float(OmegaConf.select(cfg, "hitl.criteria_intervention_cost", default=0.0)),
            )
            # Optional knobs forwarded only when explicitly set, so each criterion keeps its own default
            # (e.g. MILE samples 50 by default, diffusion_mile samples 8). num_samples / probit_scale /
            # score_mc_samples mainly matter for the denoising-score criteria.
            for cfg_key, kw, cast in (
                ("hitl.criteria_num_samples", "num_samples", int),
                ("hitl.criteria_score_mc_samples", "score_monte_carlo_samples", int),
                ("hitl.criteria_probit_scale", "probit_scale", float),
            ):
                val = OmegaConf.select(cfg, cfg_key, default=None)
                if val is not None:
                    criteria_kwargs[kw] = cast(val)
            intervention_criteria = get_intervention_criteria(criteria_name, **criteria_kwargs)
            logger.info(f"Simulated human: expert={type(expert_policy).__name__} from {expert_path}, criteria='{criteria_name}'")
        elif human_mode != "real":
            raise ValueError(f"hitl.human_mode must be 'real' or 'simulated', got {human_mode!r}.")

    if chunk_size is None:
        sampler = RandomSampler()
    else:
        gamma = OmegaConf.select(cfg, "offline_algorithm.gamma", default=0.99)
        sampler = ChunkedSequentialSampler(chunk_size=int(chunk_size), obs_as_sequence=False, gamma=gamma)

    # ---- Offline H5 buffer ----
    lowdim_stats = {}
    offline_buffer = None
    if offline_mode == "pretraining" or normalize_lowdim:
        logger.info("Building offline H5 buffer (low-dim cache; images loaded lazily)...")
        offline_buffer = H5ReplayBuffer(
            h5_paths=[cfg.env.h5_dataset_path],
            sampler=sampler,
            remove_obs_keys=list(remove_obs_keys),
            min_action=action_min,
            max_action=action_max,
            normalize_lowdim_obs=normalize_lowdim,
            default_intervention_label=LABEL_OFFLINE,  # offline samples (MILE: action loss only)
        )
        lowdim_stats = offline_buffer.lowdim_obs_stats
        if normalize_lowdim:
            logger.info(f"Low-dim obs normalization keys: {list(lowdim_stats.keys())}")
        if offline_mode != "pretraining":
            offline_buffer = None  # only needed for stats

    online_buffer = ReplayBuffer(
        capacity=int(OmegaConf.select(cfg, "hitl.online_buffer_capacity", default=200000)),
        remove_obs_keys=list(remove_obs_keys),
        sampler=sampler,
        min_action=action_min,
        max_action=action_max,
    )

    # ---- Warmup buffer ----
    warmup_buffer = None
    if offline_mode == "warmup":
        warmup_dataset_path = OmegaConf.select(cfg, "hitl.warmup_dataset_path", default=None)
        if not warmup_dataset_path:
            raise ValueError("hitl.offline_mode='warmup' requires hitl.warmup_dataset_path (a demo H5 to load).")
        if not os.path.exists(warmup_dataset_path):
            raise FileNotFoundError(f"hitl.warmup_dataset_path not found: {warmup_dataset_path}")
        warmup_buffer = ReplayBuffer(
            capacity=int(OmegaConf.select(cfg, "hitl.online_buffer_capacity", default=200000)),
            remove_obs_keys=list(remove_obs_keys),
            sampler=sampler,
            min_action=action_min,
            max_action=action_max,
        )
        wstats = load_precollected_hitl(
            warmup_dataset_path, warmup_buffer, lowdim_stats, remove_obs_keys,
            keep_only_human=False, segment_by_intervention=False,
        )
        # Every warmup step is an offline anchor demo: label 2 => BC/action loss only for MILE
        for t in warmup_buffer.get_all_transitions():
            if t is not None:
                t.info["intervention"] = LABEL_OFFLINE
        offline_buffer = warmup_buffer  # plays the offline role in the MixedReplayBuffer below
        logger.info(
            f"Warmup demos: loaded {wstats['transitions']} transitions across {wstats['episodes']} "
            f"episodes from {warmup_dataset_path} (relabelled offline=2)."
        )
        if not warmup_buffer.is_empty():
            analyze_intervention_segments(warmup_buffer, chunk_size=chunk_size, context="warmup demos")

    # ---- Self-collected anchor ----
    self_buffer = None
    if offline_mode == "self":
        self_num_rollouts = int(OmegaConf.select(cfg, "hitl.self_num_rollouts", default=20))
        self_buffer = ReplayBuffer(
            capacity=int(OmegaConf.select(cfg, "hitl.online_buffer_capacity", default=200000)),
            remove_obs_keys=list(remove_obs_keys),
            sampler=sampler,
            min_action=action_min,
            max_action=action_max,
        )
        logger.info(
            f"offline_mode='self': collecting {self_num_rollouts} successful autonomous rollouts from "
            "the initial policy as the offline anchor buffer (headless, no human/expert)..."
        )
        self_worker = HitlRolloutWorker(
            env=env,
            actor=actor,
            online_buffer=self_buffer,
            device=device,
            action_dim=action_dim,
            lowdim_stats=lowdim_stats,
            remove_obs_keys=remove_obs_keys,
            n_action_steps=n_exec,
            store_only_human=False,
            segment_by_intervention=False,
            expert_policy=None,
            intervention_criteria=None,
            enable_render=False,
            human_teleop=False,
        )
        try:
            num_kept, attempt = 0, 0
            while num_kept < self_num_rollouts:
                _, _, stored, success = self_worker.rollout_episode(
                    f"self_r{attempt}", phase="SELF-COLLECT", allow_human=False, store=True,
                    require_success=True,
                )
                attempt += 1
                num_kept += int(stored > 0)
                logger.info(
                    f"  self-collect: kept {num_kept}/{self_num_rollouts} rollouts "
                    f"(attempt {attempt}, success={bool(success)}, stored={stored})"
                )
        finally:
            self_worker.close()

        for t in self_buffer.get_all_transitions():
            if t is not None:
                t.info["intervention"] = LABEL_ROBOT
        offline_buffer = self_buffer
        logger.info(
            f"Self-collected anchor: {len(self_buffer)} transitions across {self_num_rollouts}."
        )
        if not self_buffer.is_empty():
            analyze_intervention_segments(self_buffer, chunk_size=chunk_size, context="self-collected demos")

    if precollected_hitl_dataset:
        # HG-DAgger on a precollected dataset: when training a (plain, no-reweighting) cloning algo
        # -- BC or its diffusion analogue DP -- keep only the human-correction steps and drop
        # autonomous-policy steps (re-segmenting contiguous human runs so chunked sampling stays
        # valid). reweighting is read here (it is defined further down for the online path) so the
        # filter decision is available at load time.
        _reweighting = OmegaConf.select(cfg, "hitl.reweighting", default=None)
        keep_only_human = (alg_name in BC_STYLE_ALGS) and (_reweighting is None)
        stats = load_precollected_hitl(
            precollected_hitl_dataset, online_buffer, lowdim_stats, remove_obs_keys,
            keep_only_human=keep_only_human,
            segment_by_intervention=segment_by_intervention,
        )
        if keep_only_human:
            logger.info(
                f"HG-DAgger ({alg_name} + reweighting=null): filtered precollected dataset to human-only "
                f"steps -> {stats['transitions']} human-correction transitions across {stats['episodes']} "
                "contiguous human segments (autonomous-policy steps dropped)."
            )
            if stats["transitions"] == 0:
                logger.warning(
                    "No human-correction steps found in the precollected dataset after filtering; "
                    "the online buffer is empty and training will be skipped."
                )
        else:
            logger.info(
                f"Loaded precollected HITL dataset: {stats['transitions']} transitions across "
                f"{stats['episodes']} episodes ({stats['human']} human-correction steps)."
                f"Intervention rate in dataset: {stats['human'] / stats['transitions']}"
            )
            if segment_by_intervention:
                logger.info(
                    "segment_by_intervention=true: split trajectories into contiguous same-label "
                    f"segments ({stats['episodes']} segment-episodes) for label-homogeneous chunks "
                    "(no trailing-policy dilution); segments shorter than chunk_size are dropped."
                )
        # Segment analysis: intervention rate + human-segment length distribution vs chunk_size.
        # Short segments (relative to chunk_size) dilute/drop label-1 chunks and weaken the signal.
        if not online_buffer.is_empty():
            analyze_intervention_segments(
                online_buffer, chunk_size=chunk_size, context="precollected dataset"
            )

    # ---- Training buffer (online, optionally mixed with offline) + algorithm ----
    if use_offline:
        # offline_sample_ratio=null -> mix by the buffers' live size ratio (offline grows-relative).
        offline_ratio = OmegaConf.select(cfg, "hitl.offline_sample_ratio", default=None)
        train_buffer = MixedReplayBuffer(
            buffer_1=offline_buffer,
            buffer_2=online_buffer,
            sample_ratio=None if offline_ratio is None else float(offline_ratio),
            sampler=sampler,
        )
    else:
        train_buffer = online_buffer

    algo_dict = OmegaConf.to_container(OmegaConf.select(cfg, "offline_algorithm"), resolve=True)
    algo_config = ALG_TO_CONFIG[alg_name](**algo_dict)
    algo_config.actor = actor
    algo_config.buffer = train_buffer
    algo_config.logger = wandb_logger
    algo = algo_config.create()
    needs_rollout_policy = hasattr(algo, "set_rollout_policy")  # e.g. MILE
    logger.info(f"HITL algorithm: {alg_name} | store_only_human={store_only_human} | offline_mode={offline_mode}")

    reweighting = OmegaConf.select(cfg, "hitl.reweighting", default=None)
    # Reweighting (SIRIUS / IWR) per-sample weights are only consumed by a cloning algo with weighting
    # enabled: BC with use_weighted_bc=true, or DP with use_weighted_dp=true.
    _weighted_bc = alg_name == "bc" and OmegaConf.select(cfg, "offline_algorithm.use_weighted_bc", default=False)
    _weighted_dp = alg_name == "dp" and OmegaConf.select(cfg, "offline_algorithm.use_weighted_dp", default=False)
    if reweighting in ("sirius", "iwr") and not (_weighted_bc or _weighted_dp):
        logger.warning(
            f"hitl.reweighting={reweighting} computes per-sample weights, but they are only consumed by a "
            "weighted cloning algo: set (alg.offline_alg_name=bc, offline_algorithm.use_weighted_bc=true) "
            "or (alg.offline_alg_name=dp, offline_algorithm.use_weighted_dp=true)."
        )

    # ---- Rollout/teleop worker ----
    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    video_dir = os.path.join(output_dir, "debug_videos") if debug else None
    worker = None
    if not precollected_hitl_dataset:
        if debug:
            logger.info(f"debug=True: recording labelled HITL collection trajectory videos to {video_dir}")
        if human_mode == "simulated":
            enable_render = False
        else:
            enable_render = cfg.teleop.enable_render
        worker = HitlRolloutWorker(
            env=env,
            actor=algo.actor,
            online_buffer=online_buffer,
            device=device,
            action_dim=action_dim,
            lowdim_stats=lowdim_stats,
            remove_obs_keys=remove_obs_keys,
            n_action_steps=n_exec,
            store_only_human=store_only_human,
            segment_by_intervention=segment_by_intervention,
            expert_policy=expert_policy,
            intervention_criteria=intervention_criteria,
            intervention_hold=int(OmegaConf.select(cfg, "hitl.criteria_intervention_hold", default=1)),
            sim_execution_horizon=OmegaConf.select(cfg, "hitl.sim_execution_horizon", default=None),
            enable_render=enable_render,
            teleop_device=str(OmegaConf.select(cfg, "teleop.device", default="keyboard")),
            # Only open the teleop device (e.g. the single SpaceMouse) when a real human will actually
            # teleoperate online. Warmup/expert/simulated collection is autonomous, so a worker built
            # only for that must not grab the device (else concurrent/repeated runs collide on it).
            human_teleop=(not precollected_hitl_dataset and human_mode == "real"),
            takeover_key=str(OmegaConf.select(cfg, "teleop.takeover_key", default="tab")),
            camera=OmegaConf.select(cfg, "teleop.camera", default="agentview"),
            wrist_camera=OmegaConf.select(cfg, "teleop.wrist_camera", default="robot0_eye_in_hand"),
            show_wrist=bool(OmegaConf.select(cfg, "teleop.show_wrist", default=True)),
            record_video=debug,
            video_dir=video_dir,
        )

    # ---- Autonomous eval via BatchEvaluationWorker (separate env so chunked actors execute
    # open-loop; records + logs an eval video each round). The eval env is VECTORIZED with
    # eval.eval_num_envs parallel envs; the worker gives each env a quota of episodes and batches the
    # policy's act() (a full reverse-diffusion sample for DiffusionActors) across all of them in one
    # GPU forward pass. eval.eval_vectorization selects the backend:
    #   * "sync"  -> sub-envs step serially in-process; only act() is batched. Modest win for low-dim,
    #                ~none for images (env stepping/rendering dominates and stays serial).
    #   * "async" -> one subprocess per env (spawn) so the MuJoCo sims + rendering step CONCURRENTLY.
    #                Benchmarked ~2x at 10 envs / ~5x at 25 envs for image obs (unbiased success rate),
    #                at a one-time ~10s spawn cost (the eval env is built once and reused all run).
    # Async needs a picklable env (raw-image Mode B or low-dim; NOT DINO-embedding Mode A). ----
    eval_num_envs = int(OmegaConf.select(cfg, "eval.eval_num_envs", default=10))
    eval_vectorization = str(OmegaConf.select(cfg, "eval.eval_vectorization", default="sync")).lower()
    eval_env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path,
        n_envs=eval_num_envs,
        device=device,
        seed=OmegaConf.select(cfg, "hitl.seed", default=0)+14,
        max_episode_steps=cfg.env.max_episode_steps,
        use_full_state=cfg.env.use_full_state,
        terminate_on_success=True,
        chunk_size=chunk_size,
        n_action_steps=n_exec,
        vectorization=eval_vectorization,
    )
    eval_worker = BatchEvaluationWorker(
        eval_env=eval_env,
        device=device,
        num_episodes=int(cfg.eval.eval_num_episodes),
        record_video=cfg.eval.eval_record_video,
        logger=wandb_logger,
        lowdim_obs_stats=lowdim_stats,
        max_episode_steps=int(cfg.env.max_episode_steps),
    )

    # =====================================================================================
    # Iterative loop (a single pass in precollected-dataset mode)
    # =====================================================================================
    rollouts_per_iter = int(OmegaConf.select(cfg, "hitl.rollouts_per_iter", default=5))
    train_steps_per_iter = int(OmegaConf.select(cfg, "hitl.train_steps_per_iter", default=1000))
    clear_each_iter = bool(OmegaConf.select(cfg, "hitl.clear_buffer_each_iter", default=False))
    save_interval = int(OmegaConf.select(cfg, "hitl.save_interval", default=1))

    best_success_rate = -1.0       # global best across all iterations  -> save_dir/best
    iter_best_success_rate = -1.0  # best within the current iteration  -> save_dir/best_iter{it}
    keep_best = bool(OmegaConf.select(cfg, "hitl.keep_best", default=False))

    def _maybe_save_best(metrics, it=None):
        """Update the global best (always) and, when ``it`` is given, the current iteration's best."""
        nonlocal best_success_rate, iter_best_success_rate
        sr = metrics.get("success_rate") if metrics else None
        if sr is None:
            return
        sr = float(sr)
        if sr > best_success_rate:
            best_success_rate = sr
            save_checkpoint(algo, save_dir, "best")
            logger.info(f"New global best eval success_rate={sr:.3f}; saved checkpoint to {os.path.join(save_dir, 'best')}")
        if it is not None and sr > iter_best_success_rate:
            iter_best_success_rate = sr
            save_checkpoint(algo, save_dir, f"best_iter{it}")
            logger.info(f"New iter-{it + 1} best eval success_rate={sr:.3f}; saved checkpoint to {os.path.join(save_dir, f'best_iter{it}')}")

    def _load_best_weights(tag) -> bool:
        """Reload checkpoint ``save_dir/<tag>`` weights IN PLACE into actor / online_actor.

        Uses ``load_state_dict`` into the existing modules (rather than ``load_checkpoint``, which
        replaces the objects) so the rollout worker's ``actor`` reference and the optimizer's
        parameter bindings stay valid. The optimizer state is left as-is (its momentum is overwritten
        within a few steps). Returns True if anything was loaded.
        """
        src = os.path.join(save_dir, str(tag))
        reloaded = False
        for comp in ("actor", "online_actor"):
            comp_path = os.path.join(src, f"{comp}.pt")
            if hasattr(algo, comp) and os.path.exists(comp_path):
                loaded = torch.load(comp_path, map_location=device, weights_only=False)
                getattr(algo, comp).load_state_dict(loaded.state_dict())
                reloaded = True
        return reloaded

    try:
        if not precollected_hitl_dataset and expert_policy is not None and human_mode == "simulated":
            logger.info("Evaluating simulated human before training...")
            eval_metrics = eval_worker.run(expert_policy)
        if bool(OmegaConf.select(cfg, "eval.eval_on_first_step", default=True)):
            logger.info("Evaluating initial policy before training...")
            eval_metrics = eval_worker.run(algo.actor)
            wandb_logger.log(eval_metrics, step=algo.step_counter, prefix="eval")
            _maybe_save_best(eval_metrics)  # global best only (pre-training; not an iteration's best)

        for it in range(num_iterations):
            logger.info(f"===== HiTL iteration {it + 1}/{num_iterations} ({alg_name}) =====")
            # keep_best: revert to the PREVIOUS iteration's best before starting this iteration, so
            # collection + training build on that iteration's peak rather than its drifted end state.
            if keep_best and it > 0:
                if _load_best_weights(f"best_iter{it - 1}"):
                    logger.info(f"keep_best: reloaded iteration {it}'s best checkpoint before iteration {it + 1}")
                else:
                    logger.warning(
                        f"keep_best: no best checkpoint found for iteration {it} (no eval ran during it?); "
                        "keeping current weights"
                    )
            iter_best_success_rate = -1.0
            # In precollected-dataset mode the online buffer is already populated; do not clear it
            # and do not collect online.
            if clear_each_iter and not precollected_hitl_dataset:
                online_buffer.clear()

            # --- 1) Collect rollouts until rollouts_per_iter SUCCESSFUL trajectories are stored.
            if not precollected_hitl_dataset:
                # Count KEPT rollouts (stored>0): an episode is kept only if it passes the
                # require_success and keep_only_hitl_rollouts (>=1 intervention) filters.
                succ, lengths, human_frac, stored_total = [], [], [], 0
                num_success, num_kept, attempt = 0, 0, 0
                while num_kept < rollouts_per_iter:
                    steps, human_steps, stored, success = worker.rollout_episode(
                        f"it{it}_r{attempt}", phase="COLLECT", allow_human=True, store=True,
                        require_success=True, require_intervention=keep_only_hitl_rollouts,
                    )
                    attempt += 1
                    succ.append(float(success))
                    lengths.append(steps)
                    human_frac.append(human_steps / max(steps, 1))
                    stored_total += stored
                    num_success += int(bool(success))
                    num_kept += int(stored > 0)
                    logger.info(
                        f"  kept {num_kept}/{rollouts_per_iter} rollouts "
                        f"(attempt {attempt}, success={bool(success)}, stored={stored})"
                    )
                wandb_logger.log(
                    {
                        "mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
                        "intervention_fraction": float(np.mean(human_frac)) if human_frac else 0.0,
                        "transitions_stored": stored_total,
                        "online_buffer_size": len(online_buffer),
                        "iteration": it + 1,
                        "successful_attempt_rate": num_success / attempt,
                    },
                    step=algo.step_counter,
                    prefix="collect",
                )

            if online_buffer.is_empty():
                logger.warning("Online buffer is empty (no transitions stored yet); skipping training this iteration.")
                continue

            # Segment analysis on the accumulated online buffer: intervention rate + human-segment
            # length distribution vs chunk_size. Segments near/below chunk_size dilute (MILE) or are
            # dropped entirely (store_only_human / chunked sampler), silently shrinking the effective
            # training set — the usual cause of weak online HG-DAgger / MILE signal.
            analyze_intervention_segments(
                online_buffer, chunk_size=chunk_size, context=f"online buffer (iter {it + 1})"
            )

            # --- 1b) Optional class reweighting (recomputed each round as the dataset grows) ---
            if reweighting == "sirius":
                stats = compute_sirius_weights(
                    online_buffer,
                    offline_buffer if use_offline else None,
                )
                log_d = {f"count_{k}": v for k, v in stats["counts"].items()}
                log_d.update({f"weight_{k}": v for k, v in stats.get("weights", {}).items()})
                wandb_logger.log(log_d, step=algo.step_counter, prefix="sirius")
            elif reweighting == "iwr":
                stats = compute_iwr_weights(online_buffer)  # corrections (D_I) vs autonomy (D_R), online only
                log_d = {f"count_{k}": v for k, v in stats["counts"].items()}
                log_d.update({f"weight_{k}": v for k, v in stats.get("weights", {}).items()})
                wandb_logger.log(log_d, step=algo.step_counter, prefix="iwr")

            logger.info(f"Online buffer size after collection: {len(online_buffer)}")
       
            # --- 2) Train (rollout_policy = frozen snapshot of the data-collection policy, for MILE) ---
            if needs_rollout_policy:
                algo.set_rollout_policy(copy.deepcopy(algo.actor))
            buffer_size = len(train_buffer)
            batch_size = int(OmegaConf.select(cfg, "offline_algorithm.batch_size", default=256))
            steps_per_epoch = max(1, -(-buffer_size // batch_size))  # ceil(buffer_size / batch_size)
            eval_freq_steps = OmegaConf.select(cfg, "eval.eval_freq", default=None)
            logger.info(
                f"Training {train_steps_per_iter} steps this iteration; "
                f"eval every {eval_freq_steps} steps. "
                f"1 epoch is {steps_per_epoch} steps with the current dataset "
                f"(ceil({buffer_size}/{batch_size}))."
            )
            for i in tqdm(range(train_steps_per_iter), desc="Training", unit="step"):
                algo.train_step(logging_prefix="train")  # logs internally at algo.step_counter
                # --- 3) Autonomous (no-human) evaluation ---
                if eval_freq_steps and (i + 1) % eval_freq_steps == 0:
                    eval_metrics = eval_worker.run(algo.actor)
                    wandb_logger.log(eval_metrics, step=algo.step_counter, prefix="eval")
                    _maybe_save_best(eval_metrics, it=it+1)
            if (it + 1) % save_interval == 0:
                save_checkpoint(algo, save_dir, it + 1)
                logger.info(f"Saved checkpoint to {os.path.join(save_dir, str(it + 1))}")

    except KeyboardInterrupt:
        logger.info("Interrupted; saving final checkpoint and exiting.")
        save_checkpoint(algo, save_dir, "interrupted")
    finally:
        if worker is not None:
            worker.close()
        env.close()
        try:
            eval_env.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            wandb_logger.finish()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
