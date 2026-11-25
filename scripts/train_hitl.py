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
     MixedReplayBuffer); with ``hitl.use_epochs=true`` it is a number of epochs instead, one epoch
     being ceil(len(training buffer) / batch_size) steps, so the step count tracks the growing data;
     the policy is evaluated every ``eval.eval_freq`` steps (null => once, at the end of each iteration);
then repeats.

``hitl.recent_round_ratio`` sets the fraction of each ONLINE batch drawn from the current (newest)
iteration's rollouts, the rest uniformly from all earlier iterations (0.5 => the newest iteration
weighs as much as everything before it). null => uniform over the whole online buffer, under which the
newest iteration's share shrinks to ~1/num_iterations. Offline anchors are mixed in separately, by
``hitl.offline_sample_ratio``.

``hitl.human_ratio`` sets the fraction of each ONLINE batch drawn from human-correction (label-1)
samples, the rest from autonomous-policy (label-0) ones, like FlowDAgger's dual buffer;
``recent_round_ratio`` then applies within each label. A chunk counts as human when its first step
is. null (or either label missing) => label-agnostic, so the human share follows the intervention rate.
For MILE-style algorithms this also shifts the probit's class balance (its BCE is unweighted), so the
learned intervention probabilities are no longer calibrated to the true rate.

``hitl.pure_policy_chunks=true`` (chunked sampling, online buffer) keeps each rollout as one episode
but only admits chunks that either start on a human step (and may run into the policy steps after the
takeover, e.g. [H,H,R,R]) or are policy throughout ([R,R,R,R]). A policy-labelled chunk thus never hides
an upcoming takeover, while human takeovers shorter than the chunk are kept (unlike
``segment_by_intervention``, with which it cannot be combined). Takeovers with fewer than chunk_size
steps left before the episode ends still cannot form a chunk.

``hitl.retrain_from_init=true`` makes each iteration's training phase restart from the initial
(``load_dir``) weights with a fresh optimizer on the aggregated data, instead of fine-tuning the
previous iteration's policy. Collection still runs with the previous iteration's trained policy, and
the MILE rollout-policy snapshot / stored pools are taken before the reset. Pair with
``hitl.use_epochs=true`` so the training budget grows with the buffer.

``hitl.epoch_sampling=true`` makes every buffer hand out its samples (or chunks) in shuffled passes,
each exactly once per pass, instead of independent random draws. A pass restarts whenever a buffer
grows, i.e. after each collection phase.

``hitl.offline_mode`` selects the offline-demo source mixed in for anti-forgetting aggregation:
  * null          -> no offline data (train on the online buffer only);
  * "pretraining" -> mix the pretraining demonstration H5 (cfg.env.h5_dataset_path);
  * "warmup"      -> mix a user-provided demo H5 (``hitl.warmup_dataset_path``) as the offline anchor.
                     Same mechanism as "pretraining" but from an arbitrary path, so different
                     algorithms / sweep configs can reuse the SAME fixed set of initial demos.
  * "self"        -> collect ``hitl.self_num_rollouts`` SUCCESSFUL autonomous rollouts from the
                     initial (pretrained) policy and use them as the offline anchor. No external demo
                     source is needed: the policy's own successes act as the anti-forgetting anchor.

``hitl.save_round_datasets=true`` writes each round's collected transitions to
``<output_dir>/round_data/iter_{it}_rollouts.hdf5`` (collect_hitl_rollout.py's schema, so they can be
replayed below) and the actor that collected them to ``iter_{it}_collection_actor.pt``. On top of the
schema, each demo stores, per step:
  * flow policies: the initial noise behind every executed policy chunk. Round 0 (collected by the
    unmodified ``load_dir`` policy) stores it as ``noise`` / ``chunk_id``, which
    train_flowdagger_robomimic.py turns into policy samples; later rounds, collected by fine-tuned
    weights, store ``policy_noise`` / ``policy_chunk_id`` (FlowDAgger uses only their human windows);
  * flow_mile: ``rollout_samples [N, P, H, A]``, the baseline pool drawn from the round's collection
    policy at EVERY step (both labels), whether or not this run trains with
    ``offline_algorithm.use_stored_rollout_samples``.

Alternatively, set ``hitl.precollected_hitl_dataset`` to a HITL HDF5 written by
``scripts/collect_hitl_rollout.py`` (or a list of files, e.g. all of a run's ``round_data`` files) to
train on precollected interventions instead of collecting online. Each file is loaded as one round in
the order given (so ``hitl.recent_round_ratio`` treats the last as newest) and stored
``rollout_samples`` pools are restored, so FlowMILE scores each state against the policy that collected
it rather than the initial policy. In that mode online collection is skipped (no
expert/criterion/online-rollout worker) and ``hitl.num_iterations`` must be 1 (a single train phase over the loaded buffer). It is still
compatible with ``offline_mode='warmup'`` (which just loads its demo H5). ``hitl.store_only_human``
filters the loaded dataset the same way it filters online collection (true => human steps only).

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
    # save every round's data, then retrain another setting on exactly that data:
    uv run python scripts/train_hitl.py --config-name robomimic_hitl load_dir=/path/to/run \
        hitl.save_round_datasets=true
    uv run python scripts/train_hitl.py --config-name robomimic_hitl load_dir=/path/to/run \
        hitl.num_iterations=1 \
        'hitl.precollected_hitl_dataset=[<out>/round_data/iter_0_rollouts.hdf5,<out>/round_data/iter_1_rollouts.hdf5]'

Teleop device is selected by ``teleop.device`` (``keyboard`` or ``spacemouse``); the SpaceMouse
needs the ``hidapi`` package and a connected 3D mouse. The takeover toggle stays on the keyboard
for both devices.

``hitl.human_mode`` selects the intervention source:
  * "real"      -> a human teleoperates (keyboard/spacemouse) during takeover;
  * "simulated" -> a frozen expert + an intervention criterion decide takeover automatically (headless);
  * "toggle"    -> a human presses the takeover key to decide WHEN to intervene, but the frozen
                   ``hitl.expert_load_dir`` policy (not a teleop device) supplies the actions while in
                   control. Movement keys / the spacemouse are ignored; no criterion is used.

Controls:
  * Keyboard:   Tab = take/release control, wasd/rf + zx/tg/cv to move, space = gripper,
                n = skip episode, ESC = quit.
  * SpaceMouse: Tab = take/release control, move/twist the puck to move, left button = toggle gripper,
                n = skip episode, ESC = quit.
  * Toggle:     Tab = take/release control (expert drives while taken over), ESC = quit.
  'n' works for either device, including while the policy is still driving. Taking over keeps a
  closed gripper closed. Motion scale: ``teleop.pos_sensitivity`` / ``teleop.rot_sensitivity``
  (default 1.0). ``teleop.step_fps=<N>`` slows every step (policy and human) to ~N steps/s on screen
  so the human has time to take over and steer (null = full speed; only wall-clock pacing changes,
  not the sim). ``teleop.cmd_deadzone`` ignores teleop commands smaller than it (sensor drift).
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2  # noqa: F401  (import order matters; keep above torch)

import copy
import random
import sys
from collections import OrderedDict
from datetime import datetime

import h5py
import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robometer_policy_learning.utils.logging_compat import get_logger, setup_loguru_logging
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
from robometer_policy_learning.buffers.round_replay_buffer import RoundReplayBuffer
from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env
from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer_policy_learning.rollouts.evaluation_worker import BatchEvaluationWorker
from robometer_policy_learning.utils.hitl_utils import (
    HitlRolloutWorker,
    NoiseRecordingHitlWorker,
    analyze_intervention_segments,
    compute_iwr_weights,
    compute_sirius_weights,
    describe_control_mode,
    get_intervention_criteria,
    load_precollected_hitl,
)
from robometer_policy_learning.utils.training_utils import save_checkpoint
from robometer_policy_learning.utils.numpy_compat import allow_numpy2_rng_pickles

# collect_hitl_rollout.py's HDF5 writer (the precollected-dataset schema) lives next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_hitl_rollout import _read_source_env_args, _write_h5

logger = get_logger()
# Actors trained in the robometer (numpy 2) env pickle an RNG that numpy 1.26 can't rebuild.
allow_numpy2_rng_pickles()

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

# MILE-style algorithms: their probit intervention (BCE) loss needs BOTH policy (label 0) and
# human-correction (label 1) transitions, so they are incompatible with store_only_human=true.
MILE_STYLE_ALGS = {"mile", "diffusion_mile", "flow_mile"}

# Offline-demo intervention label (online labels 0=policy / 1=human are set by the rollout worker).
LABEL_OFFLINE = 2
# Robot (autonomous-policy) intervention label — same as the online policy label set by the worker.
LABEL_ROBOT = 0
# Per-transition info key of FlowMILE's collection-time baseline pool (FlowMILE.stored_rollout_key); also
# the per-step dataset name in saved round datasets, restored from there in precollected mode.
ROLLOUT_SAMPLES_KEY = "rollout_samples"


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


def _max_episode_steps(cfg: DictConfig, key: str) -> int:
    """Episode step cap for the env configured at ``key`` (``hitl.max_episode_steps`` for collection,
    ``eval.max_episode_steps`` for eval); null => the pretraining run's ``env.max_episode_steps``.
    """
    value = OmegaConf.select(cfg, key, default=None)
    return int(cfg.env.max_episode_steps) if value is None else int(value)


def _log_chunk_span_counts(online_buffer, chunk_size, pure_policy_chunks: bool, context: str):
    """Log how many valid chunk spans start on a human vs a policy step (after pure_policy_chunks filtering)."""
    if chunk_size is None or not hasattr(online_buffer, "chunk_span_counts"):
        return
    counts = online_buffer.chunk_span_counts(int(chunk_size))
    logger.info(
        f"[{context}] valid chunk spans (chunk_size={chunk_size}, pure_policy_chunks={pure_policy_chunks}): "
        f"{counts['human']} human-start, {counts['policy']} policy-start"
    )


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


class _TransitionList:
    """Minimal buffer view (``get_all_transitions``) so ``_write_h5`` can write a subset of a buffer."""

    def __init__(self, transitions):
        self._transitions = list(transitions)

    def get_all_transitions(self):
        return self._transitions


def _save_round_dataset(path, transitions, *, base_policy_noise: bool, meta: dict, **write_kwargs) -> dict:
    """Write one round's transitions as a precollected HITL HDF5 (collect_hitl_rollout.py's schema).

    On top of that schema, each demo gets these per-step datasets when every step carries them:
      * the initial flow noise behind each executed policy chunk (zeros + chunk id -1 on human steps).
        Stored as ``noise`` / ``chunk_id`` only when ``base_policy_noise`` (the round was collected by the
        unmodified load_dir policy), which is the name train_flowdagger_robomimic.py reads. Later rounds
        were collected by fine-tuned weights, through which alone their noise reproduces the actions, so
        they get ``policy_noise`` / ``policy_chunk_id`` instead and FlowDAgger cannot mistake them for
        base-policy noise (it keeps their human windows and drops their autonomous chunks);
      * ``rollout_samples [N, P, H, A]``: FlowMILE's baseline pool, P chunks sampled from the round's
        collection policy at each step's obs, in the policy's normalized action space.
    """
    stats = _write_h5(_TransitionList(transitions), path, meta_extra=meta, **write_kwargs)
    # Same demo order as _write_h5: episodes in first-seen order, steps sorted within each.
    episodes = OrderedDict()
    for t in transitions:
        episodes.setdefault(t.episode_id, []).append(t)
    noise_key, chunk_key = ("noise", "chunk_id") if base_policy_noise else ("policy_noise", "policy_chunk_id")
    n_noise = n_pool = 0
    with h5py.File(path, "a") as f:
        for i, trs in enumerate(episodes.values()):
            infos = [t.info or {} for t in sorted(trs, key=lambda x: int(x.step_in_episode))]
            g = f["data"][f"demo_{i}"]
            if all("noise" in d and "chunk_id" in d for d in infos):
                g.create_dataset(noise_key, data=np.stack([np.asarray(d["noise"], np.float32) for d in infos]),
                                 compression="gzip")
                g.create_dataset(chunk_key, data=np.asarray([int(d["chunk_id"]) for d in infos], np.int64))
                n_noise += 1
            if all(ROLLOUT_SAMPLES_KEY in d for d in infos):
                pools = np.stack([np.asarray(d[ROLLOUT_SAMPLES_KEY], np.float32) for d in infos])
                g.create_dataset(ROLLOUT_SAMPLES_KEY, data=pools, compression="gzip")
                n_pool += 1
    return {**stats, "demos_with_noise": n_noise, "demos_with_rollout_samples": n_pool}


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
    # Eval / collect curves are indexed by HITL iteration (0 = the pretrained policy) rather than by
    # gradient step, so runs with different train_steps_per_iter / use_epochs line up.
    wandb_logger.define_step_metric("eval", "iteration")
    wandb_logger.define_step_metric("collect", "iteration")

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

    # Seed buffer batch sampling (python random), flow/diffusion/probit noise and net init (torch), as
    # train_supervised.py does. The envs built below re-seed numpy's global RNG for scene placement; resets
    # keep advancing it, so each round sees new initial states while a same-seed rerun repeats them.
    seed = int(OmegaConf.select(cfg, "hitl.seed", default=0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

    alg_name = str(OmegaConf.select(cfg, "alg.offline_alg_name", default="mile")).lower()
    if alg_name not in ALG_TO_CONFIG:
        raise ValueError(f"Unknown HITL algorithm '{alg_name}'. Choose from {sorted(ALG_TO_CONFIG)}.")

    # ---- Precollected HITL dataset: when set, train on a saved HITL HDF5 (collect_hitl_rollout.py)
    # instead of collecting interventions online. There is no online loop to iterate, so require
    # num_iterations == 1; the online collection (env/expert/criterion/worker) is skipped entirely. ----
    # A single path or a list (e.g. the per-round files written by hitl.save_round_datasets); kept as a
    # list throughout, so an empty list means online collection.
    _precollected_raw = OmegaConf.select(cfg, "hitl.precollected_hitl_dataset", default=None)
    if isinstance(_precollected_raw, str):
        _precollected_raw = [_precollected_raw]
    precollected_hitl_dataset = [str(p) for p in (_precollected_raw or [])]
    num_iterations = int(OmegaConf.select(cfg, "hitl.num_iterations", default=10))
    if precollected_hitl_dataset:
        assert num_iterations == 1, (
            "hitl.precollected_hitl_dataset is set, so there is no online collection to iterate over; "
            f"set hitl.num_iterations=1 (got {num_iterations})."
        )
        for _path in precollected_hitl_dataset:
            if not os.path.exists(_path):
                raise FileNotFoundError(f"hitl.precollected_hitl_dataset not found: {_path}")
            # Verify the dataset was collected with the SAME data-collection policy as this run's
            # load_dir, so the policy we train/snapshot is the one that produced the interventions.
            _meta = _read_precollected_meta(_path)
            _ds_load_dir = _meta.get("load_dir")
            if _ds_load_dir is None:
                raise ValueError(
                    f"Precollected dataset {_path} has no 'load_dir' in its /meta; "
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
        logger.info(f"Precollected HITL dataset(s): {precollected_hitl_dataset} (online collection skipped).")

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
    elif alg_name in ("flow_mile", "flow"):
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
        max_episode_steps=_max_episode_steps(cfg, "hitl.max_episode_steps"),
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
    # Chunk-span filter on the online buffer: policy-start chunks must be all-policy, human-start chunks
    # may run into the following policy steps (keeps takeovers shorter than chunk_size).
    pure_policy_chunks = bool(OmegaConf.select(cfg, "hitl.pure_policy_chunks", default=False))
    if pure_policy_chunks and segment_by_intervention:
        raise ValueError(
            "hitl.pure_policy_chunks=true cannot be combined with hitl.segment_by_intervention=true: "
            "segmenting already makes chunks label-homogeneous and drops the takeovers shorter than "
            "chunk_size that pure_policy_chunks exists to keep. Set one of them to false."
        )
    # When true, online collection keeps only successful rollouts that contain >= 1 human
    # intervention (discarding intervention-free successes); false keeps all successful rollouts.
    keep_only_hitl_rollouts = bool(OmegaConf.select(cfg, "hitl.keep_only_hitl_rollouts", default=False))
    # Write each round's collected rollouts (+ flow noise, + FlowMILE baseline pools, + the collection
    # actor) under <output_dir>/round_data/, so other settings / baselines can be trained on the same data.
    save_round_datasets = bool(OmegaConf.select(cfg, "hitl.save_round_datasets", default=False))
    if save_round_datasets and precollected_hitl_dataset:
        logger.warning("hitl.save_round_datasets is ignored in precollected-dataset mode (nothing is collected).")
        save_round_datasets = False
    if store_only_human and alg_name in MILE_STYLE_ALGS:
        raise ValueError(
            f"hitl.store_only_human=true is incompatible with alg.offline_alg_name='{alg_name}'. "
            f"MILE-style algorithms {sorted(MILE_STYLE_ALGS)} need both policy (intervention=0) and "
            "human-correction (intervention=1) transitions for the probit intervention loss; storing "
            "only human steps leaves the BCE term degenerate. Set hitl.store_only_human=false, or use "
            "alg.offline_alg_name=bc for HG-DAgger-style human-only cloning."
        )

    # retrain_from_init + Flow-MILE: the online actor restarts at pi_0 every round, so the reference
    # policy must be pi_0 too (else the reference-relative score starts at an arbitrary offset each
    # round), and the probit baseline must then come from stored collection-time pools (else it would
    # be sampled from pi_0 for data later policies collected). Fail early, before any env is built.
    if bool(OmegaConf.select(cfg, "hitl.retrain_from_init", default=False)) and alg_name == "flow_mile":
        missing = [
            k for k in ("use_stored_rollout_samples", "fixed_initial_rollout_policy")
            if not OmegaConf.select(cfg, f"offline_algorithm.{k}", default=False)
        ]
        if missing:
            raise ValueError(
                "hitl.retrain_from_init=true with alg.offline_alg_name=flow_mile requires "
                + " and ".join(f"offline_algorithm.{k}=true" for k in missing)
                + " (the reference policy must be the initial policy the actor is reset to, with the "
                "probit baseline drawn from stored collection-time pools)."
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
        elif human_mode == "toggle":
            # Toggle mode: a real human presses the takeover key to decide WHEN to intervene, but a
            # frozen expert policy (not the keyboard/spacemouse) supplies the actions while in
            # control. No intervention criterion is used (the human owns the takeover timing).
            expert_dir = OmegaConf.select(cfg, "hitl.expert_load_dir", default=None)
            if not expert_dir:
                raise ValueError("hitl.human_mode=toggle requires hitl.expert_load_dir (a pretraining run dir or checkpoint dir for the expert policy).")
            expert_ckpt_dir = _resolve_checkpoint_dir(expert_dir, OmegaConf.select(cfg, "hitl.expert_checkpoint", default=None))
            expert_path = os.path.join(expert_ckpt_dir, "actor.pt")
            if not os.path.exists(expert_path):
                raise FileNotFoundError(f"actor.pt not found in {expert_path} (toggle-mode expert).")
            expert_policy = torch.load(expert_path, map_location=device, weights_only=False).to(device)
            logger.info(
                f"Toggle mode: human controls takeover timing; expert {type(expert_policy).__name__} "
                f"from {expert_path} supplies actions while in control."
            )
        elif human_mode != "real":
            raise ValueError(f"hitl.human_mode must be 'real', 'simulated', or 'toggle', got {human_mode!r}.")

    if pure_policy_chunks and chunk_size is None:
        logger.warning("hitl.pure_policy_chunks has no effect without training.chunk_size (no chunked sampling).")
    if pure_policy_chunks and store_only_human:
        logger.warning("hitl.pure_policy_chunks has no effect with store_only_human=true (no policy steps stored).")
    if chunk_size is None:
        sampler = RandomSampler()
    else:
        gamma = OmegaConf.select(cfg, "offline_algorithm.gamma", default=0.99)
        sampler = ChunkedSequentialSampler(chunk_size=int(chunk_size), obs_as_sequence=False, gamma=gamma)

    # true => every buffer visits each sample (or chunk) once per shuffled pass instead of i.i.d. draws.
    epoch_sampling = bool(OmegaConf.select(cfg, "hitl.epoch_sampling", default=False))

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
            epoch_sampling=epoch_sampling,
        )
        lowdim_stats = offline_buffer.lowdim_obs_stats
        if normalize_lowdim:
            logger.info(f"Low-dim obs normalization keys: {list(lowdim_stats.keys())}")
        if offline_mode != "pretraining":
            offline_buffer = None  # only needed for stats

    # Tracks each transition's collection iteration so hitl.recent_round_ratio can oversample the newest.
    recent_round_ratio = OmegaConf.select(cfg, "hitl.recent_round_ratio", default=None)
    # Fraction of each ONLINE batch from human-correction (label-1) samples; recent_round_ratio then
    # applies within each label. null => label-agnostic. Chunks are labelled by their first step.
    human_ratio = OmegaConf.select(cfg, "hitl.human_ratio", default=None)
    if human_ratio is not None and store_only_human:
        logger.warning("hitl.human_ratio has no effect with store_only_human=true (no policy-labelled samples).")
    online_buffer = RoundReplayBuffer(
        capacity=int(OmegaConf.select(cfg, "hitl.online_buffer_capacity", default=200000)),
        remove_obs_keys=list(remove_obs_keys),
        sampler=sampler,
        min_action=action_min,
        max_action=action_max,
        recent_ratio=None if recent_round_ratio is None else float(recent_round_ratio),
        human_ratio=None if human_ratio is None else float(human_ratio),
        pure_policy_chunks=pure_policy_chunks,
        epoch_sampling=epoch_sampling,
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
            epoch_sampling=epoch_sampling,
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
            epoch_sampling=epoch_sampling,
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
        # Same switch as online collection: hitl.store_only_human=true keeps only the human-correction
        # steps (HG-DAgger; contiguous human runs are re-segmented so chunked sampling stays valid),
        # false keeps every step incl. autonomous-policy ones (e.g. plain BC on all HITL data).
        keep_only_human = store_only_human
        # One round per file (in the order given), so hitl.recent_round_ratio treats the LAST file as the
        # newest round, as in online collection. Files with stored FlowMILE baseline pools restore them
        # onto info[ROLLOUT_SAMPLES_KEY]; precompute_rollout_samples then skips those transitions instead
        # of resampling them from the initial policy.
        stats = {"transitions": 0, "episodes": 0, "human": 0}
        for i_file, _path in enumerate(precollected_hitl_dataset):
            online_buffer.start_round(i_file)
            file_stats = load_precollected_hitl(
                _path, online_buffer, lowdim_stats, remove_obs_keys,
                keep_only_human=keep_only_human,
                segment_by_intervention=segment_by_intervention,
                episode_prefix=f"f{i_file}_",  # every file numbers its demos from demo_0
                extra_info_keys=(ROLLOUT_SAMPLES_KEY,),
            )
            stats = {k: stats[k] + file_stats[k] for k in stats}
            logger.info(f"  {_path}: {file_stats['transitions']} transitions, {file_stats['human']} human")
        n_pooled = sum(
            1 for t in online_buffer.get_all_transitions()
            if t is not None and ROLLOUT_SAMPLES_KEY in (t.info or {})
        )
        if n_pooled:
            logger.info(f"Restored stored rollout-sample pools for {n_pooled}/{stats['transitions']} transitions.")
        if keep_only_human:
            logger.info(
                f"store_only_human=true ({alg_name}): filtered precollected dataset to human-only "
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
            _log_chunk_span_counts(online_buffer, chunk_size, pure_policy_chunks, "precollected dataset")

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
    # FlowMILE fixed_initial_rollout_policy: freeze rollout_policy = pi_0 ONCE, before any training
    # (keep_best reloads later don't touch this copy). Per-round collection policies then only feed
    # the stored probit-baseline pools (see the training phase below).
    fixed_initial_rollout = bool(getattr(algo, "fixed_initial_rollout_policy", False))
    if fixed_initial_rollout:
        algo.set_rollout_policy(copy.deepcopy(algo.actor))
        logger.info("fixed_initial_rollout_policy=true: rollout_policy frozen to the initial (pretrained) policy.")
    logger.info(
        f"HITL algorithm: {alg_name} | store_only_human={store_only_human} | offline_mode={offline_mode} | "
        f"recent_round_ratio={recent_round_ratio} | human_ratio={human_ratio} | epoch_sampling={epoch_sampling}"
    )

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
        # Saving round datasets from a flow policy: also record the initial noise behind every policy chunk.
        worker_cls = HitlRolloutWorker
        if save_round_datasets:
            if isinstance(algo.actor, FlowMatchingActor):
                worker_cls = NoiseRecordingHitlWorker
            else:
                logger.warning(
                    f"hitl.save_round_datasets: {type(algo.actor).__name__} is not a FlowMatchingActor, so no "
                    "flow noise is recorded (the saved datasets are not usable as FlowDAgger policy samples)."
                )
            if not hasattr(algo, "precompute_rollout_samples"):
                logger.warning(
                    f"hitl.save_round_datasets: alg '{alg_name}' has no rollout-sample pools, so the saved "
                    f"datasets carry no '{ROLLOUT_SAMPLES_KEY}' (collect with flow_mile to get them)."
                )
        worker = worker_cls(
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
            expert_teleop=(human_mode == "toggle"),
            intervention_hold=int(OmegaConf.select(cfg, "hitl.criteria_intervention_hold", default=1)),
            sim_execution_horizon=OmegaConf.select(cfg, "hitl.sim_execution_horizon", default=None),
            enable_render=enable_render,
            teleop_device=str(OmegaConf.select(cfg, "teleop.device", default="keyboard")),
            pos_sensitivity=float(OmegaConf.select(cfg, "teleop.pos_sensitivity", default=1.0)),
            rot_sensitivity=float(OmegaConf.select(cfg, "teleop.rot_sensitivity", default=1.0)),
            step_fps=OmegaConf.select(cfg, "teleop.step_fps", default=None),
            cmd_eps=float(OmegaConf.select(cfg, "teleop.cmd_deadzone", default=1e-6)),
            # Only open the teleop device (e.g. the single SpaceMouse) when a real human will actually
            # teleoperate online. Warmup/expert/simulated collection is autonomous, so a worker built
            # only for that must not grab the device (else concurrent/repeated runs collide on it).
            human_teleop=(not precollected_hitl_dataset and human_mode in ("real", "toggle")),
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
    if (
        not precollected_hitl_dataset
        and human_mode in ("real", "toggle")
        and eval_vectorization == "async"
    ):
        logger.warning(
            "eval.eval_vectorization='async' is incompatible with on-screen real-teleop rendering. "
            "Forcing eval_vectorization='sync'."
        )
        eval_vectorization = "sync"
    eval_env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path,
        n_envs=eval_num_envs,
        device=device,
        seed=OmegaConf.select(cfg, "hitl.seed", default=0)+14,
        max_episode_steps=_max_episode_steps(cfg, "eval.max_episode_steps"),
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
        max_episode_steps=_max_episode_steps(cfg, "eval.max_episode_steps"),
    )

    # =====================================================================================
    # Iterative loop (a single pass in precollected-dataset mode)
    # =====================================================================================
    rollouts_per_iter = int(OmegaConf.select(cfg, "hitl.rollouts_per_iter", default=5))
    train_steps_per_iter = int(OmegaConf.select(cfg, "hitl.train_steps_per_iter", default=1000))
    # true => train_steps_per_iter counts epochs over the current training buffer, not gradient steps.
    use_epochs = bool(OmegaConf.select(cfg, "hitl.use_epochs", default=False))
    clear_each_iter = bool(OmegaConf.select(cfg, "hitl.clear_buffer_each_iter", default=False))
    # true => each iteration's training phase restarts from the initial weights + a fresh optimizer
    # (aggregate-and-retrain) instead of fine-tuning the previous iteration's policy.
    retrain_from_init = bool(OmegaConf.select(cfg, "hitl.retrain_from_init", default=False))
    if retrain_from_init and not use_epochs:
        logger.warning(
            "hitl.retrain_from_init=true with a fixed step budget: every iteration retrains from scratch "
            "on a growing buffer with the same train_steps_per_iter, so later iterations get fewer passes "
            "over the data. Consider hitl.use_epochs=true."
        )
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
            logger.info(f"New iter-{it} best eval success_rate={sr:.3f}; saved checkpoint to {os.path.join(save_dir, f'best_iter{it}')}")

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

    # Snapshot the INITIAL training state (before any gradient step) for retrain_from_init: every
    # module / optimizer in the algorithm's component_names (actor, online_actor when EMA is on,
    # actor_optimizer), kept on CPU so it doesn't double GPU memory. The EMA wrapper averages INTO
    # ``actor`` and has no state of its own, so restoring ``actor`` restores it too. The optimizer's
    # initial state_dict has no per-param state, so loading it wipes the Adam moments.
    init_state = {}
    if retrain_from_init:
        for comp in getattr(algo, "component_names", []):
            obj = getattr(algo, comp, None)
            if isinstance(obj, torch.nn.Module):
                init_state[comp] = {k: v.detach().cpu().clone() for k, v in obj.state_dict().items()}
            elif isinstance(obj, torch.optim.Optimizer):
                init_state[comp] = copy.deepcopy(obj.state_dict())

    def _reset_to_init():
        """Restore the initial weights + optimizer IN PLACE (keeps worker / optimizer bindings valid)."""
        for comp, sd in init_state.items():
            getattr(algo, comp).load_state_dict(sd)
        # MILE-family running stats of the probit gap: re-init so the first steps re-seed them from the
        # restarted policy rather than the end of the previous iteration's run.
        for reinit in ("_init_score_gap_normalization", "_init_logprob_gap_normalization"):
            if hasattr(algo, reinit):
                getattr(algo, reinit)()

    try:
        if not precollected_hitl_dataset and expert_policy is not None and human_mode in ("simulated", "toggle"):
            logger.info("Evaluating expert policy before training...")
            eval_metrics = eval_worker.run(expert_policy)
        if bool(OmegaConf.select(cfg, "eval.eval_on_first_step", default=True)):
            logger.info("Evaluating initial policy before training...")
            eval_metrics = eval_worker.run(algo.actor)
            wandb_logger.log({**eval_metrics, "iteration": 0}, step=algo.step_counter, prefix="eval")
            _maybe_save_best(eval_metrics)  # global best only (pre-training; not an iteration's best)

        for it in range(num_iterations):
            logger.info(f"===== HiTL iteration {it + 1}/{num_iterations} ({alg_name}) =====")
            if human_mode in ("real", "toggle"):
                input("Press Enter to begin rollout collection for this iteration...")
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
            if not precollected_hitl_dataset:
                online_buffer.start_round(it)  # this iteration's rollouts are the "recent" round

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
                        f"(attempt {attempt}, success={bool(success)}, stored={stored}, "
                        f"steps={steps}, successful attempt rate: {num_success / attempt:.2f})"
                    )
               
                wandb_logger.log(
                    {
                        "mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
                        "intervention_fraction": float(np.mean(human_frac)) if human_frac else 0.0,
                        "transitions_stored": stored_total,
                        "online_buffer_size": len(online_buffer),
                        "online_buffer_recent_size": online_buffer.recent_size(),
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
            _log_chunk_span_counts(online_buffer, chunk_size, pure_policy_chunks, f"online buffer (iter {it + 1})")

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
                # corrections (D_I) vs all non-intervention samples incl. offline demos (D_R)
                stats = compute_iwr_weights(online_buffer, offline_buffer if use_offline else None)
                log_d = {f"count_{k}": v for k, v in stats["counts"].items()}
                log_d.update({f"weight_{k}": v for k, v in stats.get("weights", {}).items()})
                wandb_logger.log(log_d, step=algo.step_counter, prefix="iwr")

            logger.info(f"Online buffer size after collection: {len(online_buffer)}")
       
            # --- 2) Train (rollout_policy = frozen snapshot of the data-collection policy, for MILE) ---
            if needs_rollout_policy:
                # The policy that just collected this round's data. Normally it becomes the frozen
                # rollout_policy; with fixed_initial_rollout_policy, rollout_policy stays pi_0 and this
                # snapshot is used only to draw the stored pools below.
                collection_policy = copy.deepcopy(algo.actor).eval()
                if not fixed_initial_rollout:
                    algo.set_rollout_policy(collection_policy)
                # FlowMILE with use_stored_rollout_samples: freeze each newly-collected state's probit
                # baseline pool to THIS round's collection policy (just snapshotted above). Existing
                # transitions keep the pool written the round they were collected, so a round-1
                # intervention state is never scored against a later policy already trained to imitate
                # its human action. Online buffer only (offline anchors fall back to fresh samples for
                # their discarded probit rows). No-op with clear_buffer_each_iter=true (buffer is fresh).
                # save_round_datasets also draws them (for every label) so they can be saved with the round,
                # even when this run's training does not use them.
                wants_pools = getattr(algo, "use_stored_rollout_samples", False) or save_round_datasets
                if wants_pools and hasattr(algo, "precompute_rollout_samples"):
                    n_written = algo.precompute_rollout_samples(online_buffer, pool_policy=collection_policy)
                    logger.info(
                        f"FlowMILE: stored rollout samples for {n_written} new transitions "
                        f"(from iteration {it + 1}'s collection policy)."
                    )
            # Save this round's data BEFORE any retrain_from_init reset: algo.actor is still the policy
            # that collected it (the saved actor is what reproduces its recorded noise / pools).
            if save_round_datasets and not any(
                t is not None and str(t.episode_id).startswith(f"it{it}_") for t in online_buffer.get_all_transitions()
            ):
                logger.warning(f"hitl.save_round_datasets: iteration {it + 1} stored no transitions; nothing saved.")
            elif save_round_datasets:
                round_dir = os.path.join(output_dir, "round_data")
                os.makedirs(round_dir, exist_ok=True)
                # Every episode id the worker stores this iteration starts with f"it{it}_" (see rollout_episode
                # calls above; the segment suffixes keep that prefix).
                round_trs = [
                    t for t in online_buffer.get_all_transitions()
                    if t is not None and str(t.episode_id).startswith(f"it{it}_")
                ]
                actor_file = f"iter_{it}_collection_actor.pt"
                torch.save(algo.actor, os.path.join(round_dir, actor_file))
                pool_info = None
                if hasattr(algo, "precompute_rollout_samples"):
                    pool_info = {
                        "pool_size": int(algo.stored_rollout_pool_size),
                        "num_inference_steps": algo.mc_num_inference_steps,  # None => actor default
                        "policy": "this round's collection policy",
                        "action_space": "policy-normalized [-1, 1]",
                    }
                round_path = os.path.join(round_dir, f"iter_{it}_rollouts.hdf5")
                round_stats = _save_round_dataset(
                    round_path, round_trs,
                    # Only iteration 0 is collected by the unmodified load_dir weights.
                    base_policy_noise=(it == 0),
                    lowdim_stats=lowdim_stats, action_min=action_min, action_max=action_max,
                    env_args=_read_source_env_args(cfg.env.h5_dataset_path),
                    meta={
                        "created": datetime.now().isoformat(timespec="seconds"),
                        "collector": "train_hitl.py", "iteration": it, "alg_name": alg_name,
                        "load_dir": load_dir, "checkpoint": OmegaConf.select(cfg, "checkpoint", default=None),
                        "collection_actor": actor_file, "noise_reproduces_base_policy": it == 0,
                        "rollout_samples": pool_info,
                        "human_mode": human_mode, "store_only_human": store_only_human,
                        "keep_only_hitl_rollouts": keep_only_hitl_rollouts,
                        "segment_by_intervention": segment_by_intervention,
                        "n_action_steps": n_exec, "chunk_size": chunk_size, "obs_are_raw": True,
                        "remove_obs_keys": list(remove_obs_keys),
                        "intervention_labels": {"policy": LABEL_ROBOT, "human": 1},
                    },
                )
                logger.info(
                    f"Saved round {it + 1} dataset to {round_path}: {round_stats['num_demos']} demos, "
                    f"{round_stats['total_transitions']} transitions "
                    f"({round_stats['intervention_transitions']} human); noise on {round_stats['demos_with_noise']} "
                    f"demos, rollout-sample pools on {round_stats['demos_with_rollout_samples']} demos."
                )
            # retrain_from_init: reset only NOW, after collection (done with the previous iteration's
            # policy) and after the rollout-policy snapshot + stored pools above, which must reflect
            # that collection policy rather than the initial weights.
            if retrain_from_init:
                _reset_to_init()
                logger.info(f"retrain_from_init: reset actor + optimizer to the initial weights for iteration {it + 1}")
            buffer_size = len(train_buffer)
            batch_size = int(OmegaConf.select(cfg, "offline_algorithm.batch_size", default=256))
            steps_per_epoch = max(1, -(-buffer_size // batch_size))  # ceil(buffer_size / batch_size)
            num_train_steps = train_steps_per_iter * steps_per_epoch if use_epochs else train_steps_per_iter
            # eval.eval_freq is in gradient steps; null => one eval at the end of each iteration.
            eval_freq = OmegaConf.select(cfg, "eval.eval_freq", default=None)
            eval_freq_steps = num_train_steps if eval_freq is None else max(1, int(eval_freq))
            logger.info(
                f"Training {num_train_steps} steps this iteration"
                + (f" ({train_steps_per_iter} epochs)" if use_epochs else "")
                + f"; eval every {eval_freq_steps} steps"
                + (" (end of iteration). " if eval_freq is None else ". ")
                + f"1 epoch is {steps_per_epoch} steps with the current dataset "
            )
            for i in tqdm(range(num_train_steps), desc="Training", unit="step"):
                algo.train_step(logging_prefix="train")  # logs internally at algo.step_counter
                # --- 3) Autonomous (no-human) evaluation ---
                if (i + 1) % eval_freq_steps == 0:
                    eval_metrics = eval_worker.run(algo.actor)
                    # Mid-iteration evals (eval_freq < num_train_steps) land at fractional iterations.
                    eval_iter = it + (i + 1) / num_train_steps
                    wandb_logger.log({**eval_metrics, "iteration": eval_iter}, step=algo.step_counter, prefix="eval")
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
