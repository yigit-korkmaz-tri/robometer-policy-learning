#!/usr/bin/env python3
"""
Main RL training script.
"""

# Configure headless rendering for MuJoCo/GLFW before importing gym/metaworld
import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import copy

import numpy as np
import torch
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from rich import print as rprint

from robometer_policy_learning.buffers.samplers import ChunkedSequentialSampler, RandomSampler
from robometer_policy_learning.algorithms.bc import BC, BCConfig
from robometer_policy_learning.algorithms.iql import IQL, IQLConfig
from robometer_policy_learning.algorithms.sac import SAC, SACConfig
from robometer_policy_learning.algorithms.dp import DP, DPConfig
from robometer_policy_learning.algorithms.flow_matching import FlowMatching, FlowMatchingConfig
from robometer_policy_learning.algorithms.flow_matching.modeling_flow import FlowMatchingActor
from robometer_policy_learning.algorithms.flow_mile import FlowMILEConfig
from robometer_policy_learning.rollouts.evaluation_worker import EvaluationWorker
from robometer_policy_learning.utils.training_utils import (
    create_buffer,
    load_checkpoint,
    resolve_checkpoint_dir,
    save_checkpoint,
    setup_training,
)
from robometer_policy_learning.utils.policy_serving import build_policy_metadata
from robometer_policy_learning.utils.transitions_transforms import ImageAugmentationTransform

ALG_TO_CONFIG = {
    "iql": IQLConfig,
    "bc": BCConfig,
    "sac": SACConfig,
    "dp": DPConfig,
    "flow": FlowMatchingConfig,
    "flow_mile": FlowMILEConfig,
}

# MILE-style algorithms fine-tune a *pretrained* policy against a frozen snapshot of the policy that
# collected the data (the "rollout policy" / mental model), and their probit intervention loss needs
# both autonomous (label 0) and human-correction (label 1) transitions in the same dataset. Unlike
# the HITL loop in scripts/train_hitl.py -- which alternates collection and training over many
# rounds -- this script does a single fine-tuning pass over one static dataset, so the rollout policy
# is snapshotted once from the loaded checkpoint and never refreshed.
MILE_STYLE_ALGS = {"flow_mile"}

# Actor class each MILE-style algorithm requires, checked against the loaded checkpoint.
MILE_ACTOR_TYPES = {"flow_mile": FlowMatchingActor}

# HITL label semantics shared with scripts/train_hitl.py and the collectors.
LABEL_POLICY, LABEL_HUMAN, LABEL_OFFLINE = 0, 1, 2


def summarize_intervention_labels(buffer):
    """Count per-step HITL labels cached by an H5ReplayBuffer, as {label: count}.

    Returns None when the buffer exposes no per-demo cache (e.g. a non-H5 buffer).
    """
    cache = getattr(buffer, "hdf5_cache", None)
    if not cache:
        return None
    counts = {}
    for cached_demo in cache.values():
        labels = cached_demo.get("intervention")
        if labels is None:
            n = len(cached_demo.get("actions", []))
            key = getattr(buffer, "default_intervention_label", None)
            counts[key] = counts.get(key, 0) + n
            continue
        values, freqs = np.unique(np.asarray(labels), return_counts=True)
        for value, freq in zip(values, freqs):
            counts[int(value)] = counts.get(int(value), 0) + int(freq)
    return counts


@hydra_main(version_base=None, config_path="../robometer_policy_learning/configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    OmegaConf.resolve(cfg)

    # Setup all training components
    components = setup_training(cfg)

    # Extract components for easier access
    cfg = components.cfg
    device = components.device
    env = components.env
    eval_env = components.eval_env
    actor = components.actor
    remove_obs_keys = components.remove_obs_keys
    reward_model = components.reward_model
    reward_model_exp_cfg = components.reward_model_exp_cfg
    use_gt_rewards = components.use_gt_rewards
    use_relative_rewards = components.use_relative_rewards
    success_bonus_fn = components.success_bonus_fn
    save_dir = components.save_dir
    logger = components.logger
    wandb_logger = components.wandb_logger
    image_keys_to_be_used = components.dino_image_keys

    # Action bounds for buffer-side normalization: stored (env-space) actions are mapped to
    # the policy's [-1, 1] space for offline/online training, matching the actor's output.
    # Only set when the action space is finite (else the actor doesn't normalize either, so
    # the buffer shouldn't). The env still receives unnormalized actions (act() unnormalizes).
    # The space comes from setup_training: the simulator's when there is one, otherwise
    # synthesized from the dataset (cfg.env.offline_only).
    _action_space = components.action_space
    if (
        _action_space is not None
        and hasattr(_action_space, "low")
        and np.all(np.isfinite(_action_space.low))
        and np.all(np.isfinite(_action_space.high))
    ):
        action_min = np.asarray(_action_space.low, dtype=np.float32)
        action_max = np.asarray(_action_space.high, dtype=np.float32)
    else:
        action_min = action_max = None
    logger.info(f"Action normalization bounds: min={action_min}, max={action_max}")

    # Override num_offline_steps in debug mode
    if cfg.debug:
        rprint("Debug mode enabled")
        cfg.training.num_offline_steps = 10
        cfg.logging.wandb_offline = True

    offline_algo = None
    offline_buffer = None

    logger.info(f"Success bonus function: {success_bonus_fn}")
    offline_algorithm_cfg = OmegaConf.select(cfg, "offline_algorithm", default=None)
    observation_space = components.observation_space
    post_transforms = []
    if success_bonus_fn is not None:
        post_transforms.append(success_bonus_fn)
    use_image_transforms = OmegaConf.select(cfg, "training.use_image_transforms", default=False)
    if use_image_transforms:
        post_transforms.append(
            ImageAugmentationTransform(
                observation_space=observation_space,
                seed=cfg.training.seed,
            )
        )
    logger.info(f"Post transforms: {post_transforms}")

    # Initialize offline buffer
    if cfg.env.use_full_state:
        assert use_gt_rewards, "use_gt_rewards must be True when use_full_state is True"
        print(
            "⚠️ Using only the ground truth state observations, no DINO embeddings or language embeddings will be computed. No reward relabeling will be performed."
        )
    if cfg.training.chunk_size is None:
        sampler = RandomSampler()
    else:
        gamma = offline_algorithm_cfg.gamma if hasattr(offline_algorithm_cfg, "gamma") else 0.99
        sampler = ChunkedSequentialSampler(
            chunk_size=cfg.training.chunk_size, obs_as_sequence=False, gamma=gamma
        )
    logger.info(f"Offline Sampler: {sampler.__class__.__name__}")
    offline_buffer = create_buffer(
        sampler=sampler,
        use_eval_server=components.use_eval_server,
        eval_server_url=components.eval_server_url,
        eval_server_timeout=components.eval_server_timeout,
        reward_model=reward_model,
        reward_model_exp_cfg=reward_model_exp_cfg,
        use_gt_rewards=use_gt_rewards,
        use_relative_rewards=use_relative_rewards,
        capacity=0,  # Not used for H5 buffers
        remove_obs_keys=remove_obs_keys,
        post_transforms=post_transforms,
        h5_paths=[cfg.env.h5_dataset_path],
        use_full_state=cfg.env.use_full_state,
        sentence_model=components.sentence_model,
        dinov2_model=components.dinov2_model,
        dinov2_processor=components.dinov2_processor,
        image_keys_to_be_used=image_keys_to_be_used,
        min_action=action_min,
        max_action=action_max,
        normalize_lowdim_obs=OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False),
        lowdim_norm_eps=OmegaConf.select(cfg, "training.lowdim_norm_eps", default=1e-6),
        # Fallback HITL label for datasets with no per-step /data/demo_i/intervention (per-step
        # labels in the file always win). MILE-style algorithms need it; everything else ignores it.
        default_intervention_label=OmegaConf.select(cfg, "env.default_intervention_label", default=None),
        use_success_detection=cfg.reward_model.use_success_detection if reward_model is not None else False,
        success_detection_duration=cfg.reward_model.success_detection_duration
        if reward_model is not None
        else 2,
        success_detection_threshold=cfg.reward_model.success_detection_threshold
        if reward_model is not None
        else 0.65,
        add_estimated_reward=cfg.reward_model.add_estimated_reward
        if reward_model is not None
        else False,
    )

    alg_name = cfg.alg.offline_alg_name.lower()
    if alg_name not in ALG_TO_CONFIG:
        raise ValueError(f"Unknown offline algorithm: {alg_name}")
    is_mile = alg_name in MILE_STYLE_ALGS

    if is_mile:
        # MILE fine-tunes a *pretrained* policy, so the actor built from the config above is only a
        # correctly-shaped placeholder: replace it with the checkpoint's deployable actor (the same
        # object scripts/train_hitl.py loads). Loading the pickled actor rather than calling
        # load_checkpoint() matters -- load_checkpoint() would also restore the pretraining
        # optimizer over FlowMILE's (discarding its much lower fine-tuning lr) and would swap out
        # the `actor` object that FlowMILE's EMA already holds a reference to.
        if cfg.training.load_dir is None:
            raise ValueError(
                f"alg.offline_alg_name={alg_name} fine-tunes a pretrained policy: set "
                "training.load_dir=<flow pretraining run dir> (containing checkpoints/<step>/actor.pt)."
            )
        ckpt_dir = resolve_checkpoint_dir(
            cfg.training.load_dir, OmegaConf.select(cfg, "training.checkpoint", default=None)
        )
        actor_path = os.path.join(ckpt_dir, "actor.pt")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"actor.pt not found at {actor_path}")
        actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
        actor.train()
        expected_actor_type = MILE_ACTOR_TYPES[alg_name]
        if not isinstance(actor, expected_actor_type):
            raise TypeError(
                f"alg '{alg_name}' requires a {expected_actor_type.__name__}, but {actor_path} holds a "
                f"{type(actor).__name__}. Pretrain with alg.offline_alg_name=flow first."
            )
        logger.info(f"Loaded pretrained {type(actor).__name__} from {actor_path}")

    offline_algo_dict = OmegaConf.to_container(offline_algorithm_cfg)
    offline_algo_config = ALG_TO_CONFIG[alg_name](**offline_algo_dict)

    # Set runtime fields
    offline_algo_config.env = env
    offline_algo_config.actor = actor
    offline_algo_config.buffer = offline_buffer
    offline_algo_config.logger = wandb_logger

    offline_algo = offline_algo_config.create()

    # Persist the inference contract next to the checkpoints: low-dim z-score stats, action bounds,
    # obs keys, image size, camera map and n_action_steps. None of these are recoverable from
    # actor.pt, and every one of them silently changes the policy's input/output distribution if
    # inference gets it wrong. scripts/serve_robometer_policy.py reads this file.
    try:
        policy_metadata = build_policy_metadata(
            cfg=cfg,
            actor=offline_algo.actor,
            buffer=offline_buffer,
            observation_space=components.observation_space,
            action_space=components.action_space,
            dataset_path=cfg.env.h5_dataset_path,
        )
        logger.info(f"Policy metadata: {policy_metadata.describe()}")
        logger.info(f"Saved policy metadata to {policy_metadata.save(save_dir)}")
    except Exception as e:
        logger.warning(f"Could not write policy metadata (serving will need it): {type(e).__name__}: {e}")

    start_step = 0
    run_training = True

    if is_mile:
        # The frozen "rollout policy" (MILE's mental model) is the policy that produced the logged
        # rollouts and the human's decisions to intervene -- i.e. exactly the checkpoint just loaded.
        # In the iterative HITL loop this is re-snapshotted every collection round; here the dataset
        # is fixed and was collected by one policy, so it is snapshotted once and stays frozen.
        offline_algo.set_rollout_policy(copy.deepcopy(offline_algo.actor))
        logger.info(f"{alg_name}: frozen rollout policy snapshotted from {ckpt_dir}")

        # MILE's probit is trained by a BCE over labels {0, 1}; offline demos (label 2) contribute
        # only to the cloning term. Without both online classes the intervention loss is constant
        # and MILE degenerates to plain cloning, which is worth failing loudly on rather than
        # discovering after a long run.
        label_counts = summarize_intervention_labels(offline_buffer)
        logger.info(f"{alg_name}: intervention label counts {label_counts}")
        if label_counts is not None:
            missing = [
                name
                for name, label in (("policy(0)", LABEL_POLICY), ("human(1)", LABEL_HUMAN))
                if label_counts.get(label, 0) == 0
            ]
            if missing:
                raise ValueError(
                    f"{alg_name} needs both autonomous and human-correction transitions, but the "
                    f"dataset has none labelled {', '.join(missing)} (counts: {label_counts}). Write "
                    "per-step /data/demo_i/intervention labels (0=policy, 1=human, 2=offline demo) "
                    "into the HDF5, or use alg.offline_alg_name=flow for plain cloning."
                )

        if getattr(offline_algo, "use_stored_rollout_samples", False):
            # Collection-time sample pools exist to stop the probit baseline from drifting as the
            # rollout policy is re-snapshotted across HITL rounds. With a single frozen policy there
            # is no drift, so stored pools would be identical to fresh samples. They are also
            # unwritable here: H5ReplayBuffer builds transitions on the fly, so
            # precompute_rollout_samples() has no persistent transition objects to attach them to.
            logger.warning(
                "offline_algorithm.use_stored_rollout_samples=true has no effect for one-shot "
                "fine-tuning on a static HDF5 dataset (single frozen rollout policy, and the H5 "
                "buffer has no persistent transitions to store pools on). Disabling it."
            )
            offline_algo.use_stored_rollout_samples = False
    elif cfg.training.load_dir is not None:
        logger.info(f"Loading checkpoint from {cfg.training.load_dir}")
        start_step = load_checkpoint(offline_algo, cfg.training.load_dir)
        logger.info(f"Resuming from step {start_step}")

        # Create save directory
        # Save the offline algorithm
        logger.info(f"Saving checkpoint to {save_dir}/latest")
        save_checkpoint(offline_algo, save_dir, "latest")

        # If we loaded a checkpoint, we skip offline training step
        run_training = bool(cfg.training.continue_training)

    if run_training:
        # Rollout evaluation needs a simulator. Under cfg.env.offline_only (real-robot datasets)
        # there is no eval env, so evaluation is skipped entirely and progress is judged from the
        # training loss; evaluate such checkpoints on the robot instead.
        eval_freq = OmegaConf.select(cfg, "eval.eval_freq", default=None)
        eval_enabled = eval_env is not None and eval_freq is not None
        offline_evaluation_worker = None
        if eval_enabled:
            offline_evaluation_worker = EvaluationWorker(
                eval_env=eval_env,
                device=device,
                num_episodes=cfg.eval.eval_num_episodes,
                record_video=True,
                logger=wandb_logger,
                lowdim_obs_stats=getattr(offline_buffer, "lowdim_obs_stats", {}),
            )
        else:
            logger.info(
                f"Rollout evaluation disabled (eval_env={'None' if eval_env is None else 'set'}, "
                f"eval_freq={eval_freq})"
            )

        # Track the best eval success_rate and snapshot a "best" checkpoint whenever it improves.
        best_success_rate = -1.0

        def _maybe_save_best(metrics):
            nonlocal best_success_rate
            sr = metrics.get("success_rate") if metrics else None
            if sr is not None and float(sr) > best_success_rate:
                best_success_rate = float(sr)
                save_checkpoint(offline_algo, save_dir, "best")
                logger.info(f"New best eval success_rate={best_success_rate:.3f}; saved checkpoint to {save_dir}/best")

        if eval_enabled and cfg.eval.eval_on_first_step:
            offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
            wandb_logger.log(offline_eval_metrics, step=offline_algo.step_counter, prefix="eval")
        # Training loop
        logger.info(f"Training offline algorithm for {cfg.training.num_offline_steps} steps")
        with tqdm(total=cfg.training.num_offline_steps, desc="Offline Training", unit="step") as pbar:
            for i in range(start_step, cfg.training.num_offline_steps):
                metrics = offline_algo.train_step(logging_prefix="train")
                formatted_metrics = {k: f"{v:3.3f}" if isinstance(v, float) else v for k, v in metrics.items()}
                pbar.update(1)
                pbar.set_postfix(formatted_metrics)

                # Save checkpoint periodically
                if (i + 1) % cfg.training.save_interval == 0:
                    save_checkpoint(offline_algo, save_dir, i + 1)
                if eval_enabled and (i + 1) % eval_freq == 0:
                    offline_eval_metrics = offline_evaluation_worker.run(offline_algo.actor)
                    wandb_logger.log(offline_eval_metrics, step=offline_algo.step_counter, prefix="eval")
                    _maybe_save_best(offline_eval_metrics)

        # Without rollout evaluation there is no "best" snapshot, so always leave a final one.
        logger.info(f"Saving final checkpoint to {save_dir}/latest")
        save_checkpoint(offline_algo, save_dir, "latest")

    # clean up
    if env is not None:
        env.close()
    if eval_env is not None:
        eval_env.close()
    # finalize logger/run
    try:
        wandb_logger.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
