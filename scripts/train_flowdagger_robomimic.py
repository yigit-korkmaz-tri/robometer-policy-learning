#!/usr/bin/env python3
"""Iterative FlowDAgger on robomimic with a FROZEN Flow-Matching policy.

The robomimic counterpart of scripts/train_flowdagger.py (frozen pi0.5 on LIBERO). The pretrained
FlowMatchingActor (``load_dir``, a ``flow`` pretraining run) is never updated; a small steering net maps
its observation encoding to the initial noise ``x0 [H, A]`` the flow integrates from. Each iteration:
  1. collects rollouts with the steered policy until ``hitl.rollouts_per_iter`` are kept (same
     takeover / simulated-human / toggle modes and filters as scripts/train_hitl.py), recording the
     noise behind every autonomous chunk; they are written to ``iter_{it}_rollouts.hdf5``
     (collect_hitl_rollout.py's schema + ``noise`` / ``chunk_id``, so train_hitl.py's
     ``hitl.precollected_hitl_dataset`` can read it too);
  2. turns them into samples: autonomous chunks -> their recorded noise; human windows -> the noise
     found by inverting the flow's Euler sampler on the corrected chunk (dropped above
     ``flowdagger.inversion_mse_threshold``). Features (the frozen ``encode_obs``) are cached;
  3. trains the steering net by MSE on all iterations' samples (``flowdagger.human_ratio`` of each
     batch from human samples, the paper's dual buffer), saves it, evals the steered policy.

Before the first training the policy samples N(0, I) noise (``flowdagger.init_noise=randn``), i.e. it
IS the pretrained policy, so iteration 1's data and the initial eval match train_hitl.py's.

``hitl.offline_mode`` adds a fixed set of offline ANCHOR samples (round -1), mixed into every batch by
``hitl.offline_sample_ratio`` (null => their share of all samples), as in train_hitl.py:
  * null          -> none;
  * "pretraining" -> every chunk of the pretraining demos (cfg.env.h5_dataset_path), inverted;
  * "warmup"      -> every chunk of ``hitl.warmup_dataset_path``, inverted (all steps, whatever their label);
  * "self"        -> ``hitl.self_num_rollouts`` successful autonomous rollouts of the initial policy, each
                     chunk with the noise that produced it.
Inverted anchors use ``flowdagger.offline_window_stride``. Demo files are loaded fully into RAM.

``hitl.precollected_hitl_dataset`` (a HITL HDF5, or a list of them) trains on saved interventions instead
of collecting: ``hitl.num_iterations`` must be 1 and there is no expert / teleop. Human windows are
inverted. Autonomous chunks become samples only when the file carries their noise (written by this script)
AND its /meta ``load_dir`` / ``checkpoint`` match this run's, since the noise only reproduces those actions
through the same frozen policy. Plain collect_hitl_rollout.py files give human samples only.

Outputs (Hydra run dir): ``iter_{it}_rollouts.hdf5``, ``iter_{it}_samples.h5``, ``steering_it{it}.pt``
and ``checkpoints/{it}/actor.pt`` (+ ``checkpoints/best``): the pickled steered actor, loadable by any
robomimic eval script like a normal actor.

Usage (local machine with a display):
    uv run python scripts/train_flowdagger_robomimic.py --config-name robomimic_flowdagger \
        load_dir=checkpoints/square_low_dim/<flow run> checkpoint=10000 hitl.human_mode=real
    # headless, simulated human:
    uv run python scripts/train_flowdagger_robomimic.py --config-name robomimic_flowdagger \
        load_dir=<flow run> hitl.human_mode=simulated hitl.expert_load_dir=<expert run> \
        hitl.intervention_criteria=flow_mile teleop.enable_render=false
    # corrections only (no autonomous-chunk samples):
    ... flowdagger.use_policy_samples=false
    # shuffled passes over each sample pool instead of i.i.d. draws (train_hitl.py's hitl.epoch_sampling):
    ... flowdagger.epoch_sampling=true flowdagger.use_epochs=true
    # anchored on a fixed demo set, half of each batch:
    ... hitl.offline_mode=warmup hitl.warmup_dataset_path=<demos.hdf5> hitl.offline_sample_ratio=0.5
    # one training pass on a saved HITL dataset (e.g. the same one given to train_hitl.py):
    ... hitl.precollected_hitl_dataset=<hitl.hdf5> hitl.num_iterations=1

Controls are train_hitl.py's (Tab = take/release control, n = skip episode, ESC = quit).
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2  # noqa: F401  (import order matters; keep above torch)

import json
import sys
from datetime import datetime

import h5py
import numpy as np
import torch
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Reuse train_hitl.py's run-dir helpers and collect_hitl_rollout.py's HDF5 writer.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_hitl_rollout import _read_source_env_args, _write_h5
from train_hitl import _load_pretrain_cfg, _read_precollected_meta, _resolve_checkpoint_dir, _run_identity

from robometer_policy_learning.algorithms.flow_matching.modeling_flow import FlowMatchingActor
from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
from robometer_policy_learning.buffers.replay_buffer import ReplayBuffer
from robometer_policy_learning.buffers.samplers import RandomSampler
from robometer_policy_learning.envs.robosuite_wrappers import setup_robomimic_env
from robometer_policy_learning.loggers.wandb_logger import WandbLogger
from robometer_policy_learning.rollouts.evaluation_worker import BatchEvaluationWorker
from robometer_policy_learning.utils.flowdagger_utils import (
    KIND_HUMAN,
    EpochPoolSampler,
    FlowDAggerHitlWorker,
    SteeredFlowActor,
    SteeringNet,
    attach_recorded_noise,
    buffer_episodes,
    build_round_samples,
    concat_samples,
    relabel,
    sample_batch_indices,
    save_steering,
    write_samples_h5,
)
from robometer_policy_learning.utils.hitl_utils import (
    INTERVENTION_LABEL,
    analyze_intervention_segments,
    describe_control_mode,
    get_intervention_criteria,
    load_precollected_hitl,
)
from robometer_policy_learning.utils.logging_compat import get_logger, setup_loguru_logging

logger = get_logger()


def _new_steering(fd, actor: FlowMatchingActor, device) -> SteeringNet:
    return SteeringNet(
        feature_dim=actor.global_cond_dim, state_dim=0, horizon=actor.horizon, noise_dim=actor.action_dim,
        hidden_dims=list(fd.hidden_dims), latent_dim=int(fd.latent_dim), noise_bound=float(fd.noise_bound),
        output_bound=str(fd.output_bound),
    ).to(device)


def _load_expert(cfg, role: str):
    expert_dir = OmegaConf.select(cfg, "hitl.expert_load_dir", default=None)
    if not expert_dir:
        raise ValueError(f"hitl.human_mode={role} requires hitl.expert_load_dir (a pretraining run or checkpoint dir).")
    ckpt_dir = _resolve_checkpoint_dir(expert_dir, OmegaConf.select(cfg, "hitl.expert_checkpoint", default=None))
    path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"actor.pt not found in {path} ({role}-mode expert).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(path, map_location=device, weights_only=False).to(device), path


def _write_rollouts(buffer, path, *, lowdim_stats, action_min, action_max, env_args, meta):
    """collect_hitl_rollout.py's HDF5 + per-step ``noise [N, H, A]`` / ``chunk_id [N]`` in each demo."""
    stats = _write_h5(buffer, path, lowdim_stats=lowdim_stats, action_min=action_min, action_max=action_max,
                      env_args=env_args, meta_extra=meta)
    with h5py.File(path, "a") as f:
        for i, trs in enumerate(buffer_episodes(buffer).values()):  # same demo order as _write_h5
            g = f["data"][f"demo_{i}"]
            g.create_dataset("noise", data=np.stack([np.asarray(t.info["noise"], np.float32) for t in trs]),
                             compression="gzip")
            g.create_dataset("chunk_id", data=np.asarray([int(t.info["chunk_id"]) for t in trs], np.int64))
    return stats


def train_steering(net, opt, samples, num_steps, fd, rng, device, wandb_logger, step0, desc, offline_ratio=None):
    """MSE in noise space on ``samples``; returns the mean loss of the last 100 steps."""
    feats = torch.as_tensor(samples["features"], device=device)
    target = torch.as_tensor(samples["target"], device=device)
    is_human = torch.as_tensor(samples["kind"] == KIND_HUMAN, device=device)
    is_offline = torch.as_tensor(samples["round"] < 0, device=device)
    recent = fd.recent_round_ratio
    # Built per call, so each iteration's training starts fresh passes over the grown sample set.
    epoch_sampler = EpochPoolSampler(rng) if bool(OmegaConf.select(fd, "epoch_sampling", default=False)) else None
    net.train()
    recent_losses = []
    bar = tqdm(range(num_steps), desc=desc, unit="step")
    for i in bar:
        idx = sample_batch_indices(
            samples["kind"], samples["round"], int(fd.batch_size), rng,
            human_ratio=fd.human_ratio, recent_ratio=None if recent is None else float(recent),
            offline_ratio=offline_ratio, epoch_sampler=epoch_sampler,
        )
        idx_t = torch.as_tensor(idx, device=device)
        per_row = ((net(feats[idx_t]) - target[idx_t]) ** 2).mean((1, 2))
        loss = per_row.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        recent_losses = [*recent_losses, loss.item()][-100:]
        if (i + 1) % 100 == 0 or i + 1 == num_steps:
            h, off = is_human[idx_t], is_offline[idx_t]
            log = {"loss": float(np.mean(recent_losses))}
            for name, rows in (("human", h & ~off), ("policy", ~h & ~off), ("offline", off)):
                if rows.any():
                    log[f"loss_{name}"] = per_row[rows].mean().item()
            bar.set_postfix(loss=f"{log['loss']:.4f}")
            wandb_logger.log(log, step=step0 + i + 1, prefix="train")
    net.eval()
    return float(np.mean(recent_losses)) if recent_losses else float("nan")


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="robomimic_flowdagger")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = HydraConfig.get().runtime.output_dir
    save_dir = os.path.join(output_dir, "checkpoints")
    setup_loguru_logging(log_level=OmegaConf.select(cfg, "logging.log_level", default="INFO"), output_dir=output_dir)

    string_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_logger = WandbLogger(
        exp_name=f"{cfg.logging.wandb_name}_{string_time}",
        offline=cfg.logging.wandb_offline,
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        log_dir=f"{cfg.logging.wandb_log_dir_base}/{string_time}",
        prefix="offline",
    )
    # Eval / collect curves are indexed by iteration (0 = the pretrained policy), as in train_hitl.py.
    wandb_logger.define_step_metric("eval", "iteration")
    wandb_logger.define_step_metric("collect", "iteration")

    # ---- Adopt env / training / model / policy from the pretraining run (as train_hitl.py). ----
    load_dir = OmegaConf.select(cfg, "load_dir", default=None)
    if not load_dir:
        raise ValueError(
            "Set load_dir=<flow pretraining run dir> (has .hydra/config.yaml + checkpoints/<step>/actor.pt)."
        )
    pre_cfg = _load_pretrain_cfg(load_dir)
    OmegaConf.set_struct(cfg, False)
    for key in ("env", "training", "model", "policy"):
        if key in pre_cfg:
            cfg[key] = pre_cfg[key]
    OmegaConf.set_struct(cfg, True)
    OmegaConf.resolve(cfg)
    wandb_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
    fd = cfg.flowdagger
    seed = int(OmegaConf.select(cfg, "hitl.seed", default=0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if OmegaConf.select(cfg, "hitl.segment_by_intervention", default=False):
        raise ValueError("hitl.segment_by_intervention=true splits episodes at takeovers, which hides the "
                         "pre-takeover chunk from FlowDAgger's sample builder; leave it false.")
    if OmegaConf.select(cfg, "env.dino_image_keys", default=None):
        raise NotImplementedError("DINO-embedding (Mode A) policies are not supported by the HITL collector yet.")

    # ---- Frozen pretrained FlowMatchingActor, wrapped with the steering net ----
    ckpt_dir = _resolve_checkpoint_dir(load_dir, OmegaConf.select(cfg, "checkpoint", default=None))
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    base = torch.load(actor_path, map_location=device, weights_only=False).to(device)
    if not isinstance(base, FlowMatchingActor):
        raise TypeError(f"FlowDAgger needs a FlowMatchingActor (a `flow` pretraining run), got {type(base).__name__}.")
    actor = SteeredFlowActor.wrap(base, _new_steering(fd, base, device), steer_mode=str(fd.init_noise))
    actor.eval()
    logger.info(
        f"Frozen FlowMatchingActor from {actor_path}: horizon={actor.horizon}, action_dim={actor.action_dim}, "
        f"num_inference_steps={actor.num_inference_steps}, cond_dim={actor.global_cond_dim}"
    )
    remove_obs_keys = list(getattr(base, "remove_obs_keys", None)
                           or OmegaConf.select(cfg, "env.extra_keys_to_drop", default=[]) or [])

    # ---- Collection env (single, non-chunked; the worker chunks) ----
    env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path, n_envs=1, device=device, seed=seed,
        max_episode_steps=cfg.env.max_episode_steps + 200, use_full_state=cfg.env.use_full_state,
        terminate_on_success=True, chunk_size=None, n_action_steps=1,
    )
    action_dim = int(env.single_action_space.shape[0])
    logger.info(f"Control mode: {describe_control_mode(env)}")
    asp = env.single_action_space
    if np.all(np.isfinite(asp.low)) and np.all(np.isfinite(asp.high)):
        action_min, action_max = np.asarray(asp.low, np.float32), np.asarray(asp.high, np.float32)
    else:
        action_min = action_max = None
    chunk_size = OmegaConf.select(cfg, "training.chunk_size", default=None)
    n_exec = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)
    lowdim_stats = {}
    if bool(OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False)):
        lowdim_stats = H5ReplayBuffer(
            h5_paths=[cfg.env.h5_dataset_path], sampler=RandomSampler(), remove_obs_keys=list(remove_obs_keys),
            min_action=action_min, max_action=action_max, normalize_lowdim_obs=True,
        ).lowdim_obs_stats

    # ---- Offline anchors / precollected dataset (as train_hitl.py) ----
    _offline_raw = OmegaConf.select(cfg, "hitl.offline_mode", default=None)
    offline_mode = None if str(_offline_raw).strip().lower() in ("none", "null", "") else str(_offline_raw).lower()
    if offline_mode not in (None, "pretraining", "warmup", "self"):
        raise ValueError(f"hitl.offline_mode must be null, 'pretraining', 'warmup', or 'self' (got {_offline_raw!r}).")
    offline_ratio = OmegaConf.select(cfg, "hitl.offline_sample_ratio", default=None)
    offline_ratio = None if offline_ratio is None else float(offline_ratio)
    num_iterations = int(OmegaConf.select(cfg, "hitl.num_iterations", default=10))
    precollected = OmegaConf.select(cfg, "hitl.precollected_hitl_dataset", default=None)
    if isinstance(precollected, str):
        precollected = [precollected]
    precollected = [str(p) for p in (precollected or [])]
    for p in precollected:
        if not os.path.exists(p):
            raise FileNotFoundError(f"hitl.precollected_hitl_dataset not found: {p}")
    if precollected:
        assert num_iterations == 1, (
            "hitl.precollected_hitl_dataset is set, so there is no online collection to iterate over; "
            f"set hitl.num_iterations=1 (got {num_iterations})."
        )

    # ---- Intervention source (as train_hitl.py; none in precollected mode) ----
    human_mode = str(OmegaConf.select(cfg, "hitl.human_mode", default="real")).lower()
    expert_policy, criteria = None, None
    if precollected:
        pass
    elif human_mode == "simulated":
        expert_policy, expert_path = _load_expert(cfg, "simulated")
        criteria_name = str(OmegaConf.select(cfg, "hitl.intervention_criteria", default="never"))
        kwargs = dict(
            window=int(OmegaConf.select(cfg, "hitl.criteria_window", default=10)),
            intervention_cost=float(OmegaConf.select(cfg, "hitl.criteria_intervention_cost", default=0.0)),
        )
        for cfg_key, kw, cast in (
            ("hitl.criteria_num_samples", "num_samples", int),
            ("hitl.criteria_score_mc_samples", "score_monte_carlo_samples", int),
            ("hitl.criteria_probit_scale", "probit_scale", float),
        ):
            val = OmegaConf.select(cfg, cfg_key, default=None)
            if val is not None:
                kwargs[kw] = cast(val)
        criteria = get_intervention_criteria(criteria_name, **kwargs)
        logger.info(
            f"Simulated human: expert {type(expert_policy).__name__} from {expert_path}, criteria={criteria_name}"
        )
    elif human_mode == "toggle":
        expert_policy, expert_path = _load_expert(cfg, "toggle")
        logger.info(f"Toggle mode: expert {type(expert_policy).__name__} from {expert_path} acts while taken over.")
    elif human_mode != "real":
        raise ValueError(f"hitl.human_mode must be 'real', 'simulated', or 'toggle', got {human_mode!r}.")

    def _new_buffer():
        return ReplayBuffer(capacity=int(OmegaConf.select(cfg, "hitl.online_buffer_capacity", default=200000)),
                            remove_obs_keys=list(remove_obs_keys), sampler=RandomSampler(),
                            min_action=action_min, max_action=action_max)

    debug = bool(OmegaConf.select(cfg, "debug", default=False))
    store_only_human = bool(OmegaConf.select(cfg, "hitl.store_only_human", default=False))
    # store_only_human leaves no autonomous steps online; in precollected mode it filters them out here.
    use_policy = bool(fd.use_policy_samples) and not store_only_human

    def _make_worker(online: bool):
        """The HITL collection worker; ``online=False`` = autonomous-only (self anchors), no teleop device."""
        return FlowDAggerHitlWorker(
            env=env, actor=actor, online_buffer=_new_buffer(), device=device, action_dim=action_dim,
            lowdim_stats=lowdim_stats, remove_obs_keys=remove_obs_keys, n_action_steps=n_exec,
            store_only_human=store_only_human and online, segment_by_intervention=False,
            expert_policy=expert_policy if online else None, intervention_criteria=criteria if online else None,
            expert_teleop=online and human_mode == "toggle",
            intervention_hold=int(OmegaConf.select(cfg, "hitl.criteria_intervention_hold", default=1)),
            sim_execution_horizon=OmegaConf.select(cfg, "hitl.sim_execution_horizon", default=None),
            enable_render=online and human_mode != "simulated" and bool(cfg.teleop.enable_render),
            teleop_device=str(OmegaConf.select(cfg, "teleop.device", default="keyboard")),
            pos_sensitivity=float(OmegaConf.select(cfg, "teleop.pos_sensitivity", default=1.0)),
            rot_sensitivity=float(OmegaConf.select(cfg, "teleop.rot_sensitivity", default=1.0)),
            step_fps=OmegaConf.select(cfg, "teleop.step_fps", default=None),
            cmd_eps=float(OmegaConf.select(cfg, "teleop.cmd_deadzone", default=1e-6)),
            # Only the online worker opens the (single) teleop device; the self-anchor one must not grab it.
            human_teleop=online and human_mode in ("real", "toggle"),
            takeover_key=str(OmegaConf.select(cfg, "teleop.takeover_key", default="tab")),
            camera=OmegaConf.select(cfg, "teleop.camera", default="agentview"),
            wrist_camera=OmegaConf.select(cfg, "teleop.wrist_camera", default="robot0_eye_in_hand"),
            show_wrist=bool(OmegaConf.select(cfg, "teleop.show_wrist", default=True)),
            record_video=debug and online, video_dir=os.path.join(output_dir, "debug_videos") if debug else None,
        )

    def _samples(buffer, name, round_idx, *, policy, stride, file_tag, step=0):
        """build_round_samples + logging; saves ``<file_tag>_samples.h5``. Returns the samples (or None)."""
        samples, st = build_round_samples(
            buffer, actor, device, use_policy=policy, drop_pre_takeover_chunk=bool(fd.drop_pre_takeover_chunk),
            human_window_stride=stride, min_human_window=int(fd.min_human_window), fp_iters=int(fd.fp_iters),
            mse_threshold=fd.inversion_mse_threshold, round_idx=round_idx,
        )
        if samples is None:
            logger.warning(f"{name}: no samples.")
            return None
        logger.info(
            f"{name} samples: {st['num_policy']} policy chunks, {st['num_human']} inverted windows "
            f"(inversion mse median {st['inv_mse_median']:.2e}, max {st['inv_mse_max']:.2e}; "
            f"{st['num_dropped']} dropped), p99 |w*| = {st['human_abs_w_p99']:.2f}"
        )
        if fd.output_bound == "tanh" and st["human_abs_w_p99"] > 0.9 * float(fd.noise_bound):
            logger.warning(f"p99 |w*| {st['human_abs_w_p99']:.2f} is near/above flowdagger.noise_bound "
                           f"{fd.noise_bound}: the steering net cannot reach those targets; raise it.")
        wandb_logger.log(st, step=step, prefix="samples" if round_idx >= 0 else f"samples_{file_tag}")
        write_samples_h5(os.path.join(output_dir, f"{file_tag}_samples.h5"), samples, {"name": name})
        return samples

    # ---- Offline anchors (round -1) ----
    offline_samples = None
    if offline_mode in ("pretraining", "warmup"):
        path = (cfg.env.h5_dataset_path if offline_mode == "pretraining"
                else OmegaConf.select(cfg, "hitl.warmup_dataset_path", default=None))
        if not path:
            raise ValueError("hitl.offline_mode='warmup' requires hitl.warmup_dataset_path (a demo H5 to load).")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Offline demo file not found: {path}")
        off_buffer = _new_buffer()
        wst = load_precollected_hitl(path, off_buffer, lowdim_stats, remove_obs_keys)
        relabel(off_buffer, INTERVENTION_LABEL)  # every demo chunk is an inverted anchor
        logger.info(f"offline_mode={offline_mode}: {wst['transitions']} transitions / {wst['episodes']} demos "
                    f"from {path}; inverting windows with stride {fd.offline_window_stride}...")
        offline_samples = _samples(off_buffer, f"offline ({offline_mode})", -1, policy=False,
                                   stride=fd.offline_window_stride, file_tag="offline")
        del off_buffer
    elif offline_mode == "self":
        self_num = int(OmegaConf.select(cfg, "hitl.self_num_rollouts", default=10))
        logger.info(f"offline_mode='self': collecting {self_num} successful autonomous rollouts of the initial "
                    f"policy (steer_mode={actor.steer_mode}) as anchors...")
        self_worker = _make_worker(online=False)
        try:
            kept, attempt = 0, 0
            while kept < self_num:
                _, _, stored, success = self_worker.rollout_episode(
                    f"self_r{attempt}", phase="SELF-COLLECT", allow_human=False, store=True, require_success=True,
                )
                attempt += 1
                kept += int(stored > 0)
                logger.info(f"  self-collect: kept {kept}/{self_num} (attempt {attempt}, success={bool(success)})")
            offline_samples = _samples(self_worker.online_buffer, "offline (self)", -1, policy=True,
                                       stride=fd.human_window_stride, file_tag="offline")
        finally:
            self_worker.close()

    # ---- Precollected HITL dataset(s): the single training pass's data ----
    precollected_samples = None
    if precollected:
        pre_buffer = _new_buffer()
        for p in precollected:
            pst = load_precollected_hitl(p, pre_buffer, lowdim_stats, remove_obs_keys)
            meta = _read_precollected_meta(p)
            same_policy = (meta.get("load_dir") is not None
                           and _run_identity(meta["load_dir"]) == _run_identity(load_dir)
                           and str(meta.get("checkpoint")) == str(OmegaConf.select(cfg, "checkpoint", default=None)))
            n_noise = attach_recorded_noise(p, pre_buffer) if same_policy else 0
            logger.info(f"Precollected {p}: {pst['transitions']} transitions / {pst['episodes']} episodes "
                        f"({pst['human']} human); recorded noise for {n_noise} demos.")
            if not same_policy:
                logger.warning(
                    f"{p} was collected with load_dir={meta.get('load_dir')} checkpoint={meta.get('checkpoint')}, "
                    "not this run's policy: its autonomous chunks are skipped (their noise, if any, belongs to "
                    "another flow); the human corrections are still inverted."
                )
        if not pre_buffer.is_empty():
            analyze_intervention_segments(pre_buffer, chunk_size=chunk_size, context="precollected dataset")
        precollected_samples = _samples(pre_buffer, "precollected", 0, policy=use_policy,
                                        stride=fd.human_window_stride, file_tag="precollected")
        del pre_buffer

    worker = None if precollected else _make_worker(online=True)

    # ---- Autonomous eval (as train_hitl.py; sync only while a human watches the teleop window) ----
    eval_vectorization = str(OmegaConf.select(cfg, "eval.eval_vectorization", default="sync")).lower()
    if human_mode in ("real", "toggle") and eval_vectorization == "async":
        logger.warning("eval.eval_vectorization='async' is incompatible with on-screen teleop; using 'sync'.")
        eval_vectorization = "sync"
    eval_env, _ = setup_robomimic_env(
        dataset_path=cfg.env.h5_dataset_path, n_envs=int(OmegaConf.select(cfg, "eval.eval_num_envs", default=10)),
        device=device, seed=seed + 14, max_episode_steps=cfg.env.max_episode_steps+200,
        use_full_state=cfg.env.use_full_state, terminate_on_success=True, chunk_size=chunk_size,
        n_action_steps=n_exec, vectorization=eval_vectorization,
    )
    eval_worker = BatchEvaluationWorker(
        eval_env=eval_env, device=device, num_episodes=int(cfg.eval.eval_num_episodes),
        record_video=cfg.eval.eval_record_video, logger=wandb_logger, lowdim_obs_stats=lowdim_stats,
        max_episode_steps=int(cfg.env.max_episode_steps)+200,
    )

    rollouts_per_iter = int(OmegaConf.select(cfg, "hitl.rollouts_per_iter", default=5))
    keep_only_hitl = bool(OmegaConf.select(cfg, "hitl.keep_only_hitl_rollouts", default=False))
    env_args = _read_source_env_args(cfg.env.h5_dataset_path)
    net = actor.steering
    opt = torch.optim.Adam(net.parameters(), lr=float(fd.lr))
    all_samples, train_step, best_sr = concat_samples([offline_samples]), 0, -1.0
    eval_history = []

    def _eval(name, iteration):
        nonlocal best_sr
        logger.info(f"Evaluating {name} (steer_mode={actor.steer_mode}) ...")
        metrics = eval_worker.run(actor)
        wandb_logger.log({**metrics, "iteration": iteration}, step=train_step, prefix="eval")
        sr = metrics.get("success_rate")
        eval_history.append((name, sr))
        logger.info(f"[{name}] success_rate={sr}")
        if sr is not None and float(sr) > best_sr:
            best_sr = float(sr)
            os.makedirs(os.path.join(save_dir, "best"), exist_ok=True)
            torch.save(actor, os.path.join(save_dir, "best", "actor.pt"))
        return metrics

    def _collect_iteration(it):
        """Collect iteration ``it``'s rollouts, write them, and return their samples (or None)."""
        if human_mode in ("real", "toggle"):
            input("Press Enter to begin rollout collection for this iteration...")

        # --- 1) Collect (fresh buffer per iteration: only this iteration's episodes become samples) ---
        buffer = _new_buffer()
        worker.online_buffer = buffer
        lengths, human_frac, num_success, num_kept, attempt = [], [], 0, 0, 0
        while num_kept < rollouts_per_iter:
            steps, human_steps, stored, success = worker.rollout_episode(
                f"it{it}_r{attempt}", phase="COLLECT", allow_human=True, store=True,
                require_success=bool(OmegaConf.select(cfg, "hitl.require_success", default=True)),
                require_intervention=keep_only_hitl,
            )
            attempt += 1
            num_success += int(bool(success))
            num_kept += int(stored > 0)
            lengths.append(steps)
            human_frac.append(human_steps / max(steps, 1))
            logger.info(f"  kept {num_kept}/{rollouts_per_iter} (attempt {attempt}, success={bool(success)}, "
                        f"stored={stored}, steps={steps}, successful attempt rate {num_success / attempt:.2f})")
        wandb_logger.log({
            "mean_episode_len": float(np.mean(lengths)), "intervention_fraction": float(np.mean(human_frac)),
            "transitions_stored": len(buffer), "iteration": it + 1,
            "successful_attempt_rate": num_success / attempt,
        }, step=train_step, prefix="collect")
        analyze_intervention_segments(buffer, chunk_size=chunk_size, context=f"iteration {it + 1} rollouts")
        rollouts_path = os.path.join(output_dir, f"iter_{it}_rollouts.hdf5")
        _write_rollouts(buffer, rollouts_path, lowdim_stats=lowdim_stats, action_min=action_min,
                        action_max=action_max, env_args=env_args, meta={
                            "created": datetime.now().isoformat(timespec="seconds"),
                            "collector": "train_flowdagger_robomimic.py", "iteration": it,
                            "load_dir": load_dir, "checkpoint": OmegaConf.select(cfg, "checkpoint", default=None),
                            "collection_policy": actor.steer_mode, "source_dataset": cfg.env.h5_dataset_path,
                            "human_mode": human_mode, "store_only_human": store_only_human,
                            "keep_only_hitl_rollouts": keep_only_hitl, "n_action_steps": n_exec,
                            "chunk_size": chunk_size, "obs_are_raw": True,
                            "remove_obs_keys": list(remove_obs_keys),
                            "intervention_labels": {"policy": 0, "human": 1},
                        })
        logger.info(f"Wrote {rollouts_path}")

        # --- 2) Samples: recorded noise for autonomous chunks, inverted noise for human windows ---
        return _samples(buffer, f"Iteration {it + 1}", it, policy=use_policy, stride=fd.human_window_stride,
                        file_tag=f"iter_{it}", step=train_step)

    try:
        if bool(OmegaConf.select(cfg, "eval.eval_on_first_step", default=True)):
            _eval("initial policy", 0)

        for it in range(num_iterations):
            logger.info(f"===== FlowDAgger iteration {it + 1}/{num_iterations} =====")
            if precollected:
                samples = precollected_samples
            else:
                samples = _collect_iteration(it)
            all_samples = concat_samples([all_samples, samples])
            if all_samples is None or not (all_samples["round"] >= 0).any():
                # As train_hitl.py: no training on the offline anchors alone.
                logger.warning("No online/precollected samples yet; skipping training this iteration.")
                continue

            # --- 3) Train the steering net on every iteration's samples ---
            if bool(fd.reinit_each_iter):
                net = actor.steering = _new_steering(fd, base, device)
                opt = torch.optim.Adam(net.parameters(), lr=float(fd.lr))
            n = len(all_samples["kind"])
            steps_per_epoch = max(1, -(-n // int(fd.batch_size)))
            num_steps = int(fd.train_steps_per_iter) * (steps_per_epoch if bool(fd.use_epochs) else 1)
            n_off = int((all_samples["round"] < 0).sum())
            n_human = int(((all_samples["kind"] == KIND_HUMAN) & (all_samples["round"] >= 0)).sum())
            logger.info(f"Training the steering net {num_steps} steps on {n} samples ({n_human} human, "
                        f"{n - n_off - n_human} policy, {n_off} offline anchors; 1 epoch = {steps_per_epoch} steps)")
            loss = train_steering(net, opt, all_samples, num_steps, fd, rng, device, wandb_logger, train_step,
                                  desc=f"Training (iter {it + 1})", offline_ratio=offline_ratio)
            train_step += num_steps
            actor.set_steer_mode("steer")
            logger.info(f"Iteration {it + 1} final loss {loss:.4f}")
            save_steering(os.path.join(output_dir, f"steering_it{it + 1}.pt"), net, {"iteration": it + 1})
            os.makedirs(os.path.join(save_dir, str(it + 1)), exist_ok=True)
            torch.save(actor, os.path.join(save_dir, str(it + 1), "actor.pt"))

            # --- 4) Eval the steered policy ---
            _eval(f"iteration {it + 1}", it + 1)

    except KeyboardInterrupt:
        logger.info("Interrupted; saving the current steering net and exiting.")
        save_steering(os.path.join(output_dir, "steering_interrupted.pt"), actor.steering)
    finally:
        if eval_history:
            logger.info("Eval summary: " + ", ".join(f"{name}: {sr}" for name, sr in eval_history))
            with open(os.path.join(output_dir, "eval_history.json"), "w") as f:
                json.dump(eval_history, f, indent=2)
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
