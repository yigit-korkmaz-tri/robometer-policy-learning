from dataclasses import dataclass
from typing import Any, Optional, Tuple

from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig


@dataclass
class FlowMatchingConfig(BaseAlgorithmConfig):
    """
    Configuration for the Flow Matching (FM) policy algorithm.

    Flow Matching (Lipman et al., 2023; Liu et al., 2022 "rectified flow") frames behavior
    cloning as learning a time-dependent velocity field that transports a simple noise
    distribution to the data distribution along a (conditional) probability path. Given a
    noise sample ``x0 ~ N(0, I)`` and an expert action chunk ``x1``, we follow the
    straight-line conditional OT path

        ``x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1``,   t in [0, 1]

    whose constant target velocity is ``u_t = x1 - (1 - sigma_min) * x0``. The network
    ``v_theta(x_t, t, obs)`` regresses ``u_t`` under an MSE loss. At inference time actions are
    produced by integrating ``dx/dt = v_theta(x_t, t, obs)`` from ``t = 0`` (Gaussian noise) to
    ``t = 1`` with a fixed-step Euler solver — typically far fewer steps than diffusion needs.

    This is the deterministic flow-matching analogue of :class:`DPConfig`: it shares the same
    conditional networks (U-Net / MLP / Transformer), observation encoder, EMA, weighted-cloning
    and augmentation machinery, but replaces the diffusers noise scheduler with the flow path
    above. Like DP, it operates entirely in the actor's normalized ([-1, 1]) action space; the
    replay buffer maps env-space actions into [-1, 1] and ``act()`` unnormalizes back.
    """

    # Runtime fields (inherited from BaseAlgorithmConfig: env, actor, critic, buffer, logger)
    action_space: Optional[Any] = None

    # ----- Generic training parameters -----
    learning_starts: int = 0
    batch_size: int = 256
    num_updates_per_train_step: int = 1

    # ----- Optimizer -----
    actor_optimizer_lr: float = 1e-4
    actor_optimizer_eps: float = 1e-8
    actor_optimizer_weight_decay: float = 1e-6
    actor_optimizer_betas: Tuple[float, float] = (0.95, 0.999)
    clip_grad_norm: float = 1.0  # <= 0 disables gradient clipping

    # ----- Flow matching path / solver -----
    # Number of Euler integration steps used to sample actions (t: 0 -> 1). Flow matching is
    # typically accurate with far fewer steps than diffusion (e.g. 5-20).
    num_inference_steps: int = 10
    # OT-CFM path width at t=0. sigma_min=0 recovers pure rectified flow (target u = x1 - x0);
    # a small positive value keeps the path from collapsing exactly onto the data at t=1.
    sigma_min: float = 0.0
    # Distribution used to sample the flow time t during training.
    #   "uniform"      : t ~ U(0, 1)
    #   "logit_normal" : t = sigmoid(s), s ~ N(logit_normal_mean, logit_normal_std^2); concentrates
    #                    samples near t=0.5, often improving sample quality (SD3-style).
    time_sampling: str = "uniform"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    # The conditional networks were designed for integer diffusion timesteps; the continuous
    # flow time t in [0, 1] is multiplied by this factor before the sinusoidal time embedding so
    # the embedding spans a useful frequency range.
    time_embed_scale: float = 1000.0
    # Clamp the integrated sample to [-clip_sample_range, clip_sample_range] at the end of sampling
    # (the actor works in normalized [-1, 1] action space). Set clip_sample=false to disable.
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # ----- Velocity prediction network (shared with Diffusion Policy) -----
    net_type: str = "unet"  # ["unet", "mlp", "transformer"]
    diffusion_step_embed_dim: int = 128
    # Conditional U-Net 1D. Downsampling factor is 2**(len(down_dims)-1); action sequences are
    # internally padded to a multiple of that factor.
    unet_down_dims: Tuple[int, ...] = (128, 256)
    unet_kernel_size: int = 5
    unet_n_groups: int = 8
    # Conditional MLP (robust fallback for arbitrary / very short horizons).
    mlp_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    # Conditional Transformer (attends over action-chunk tokens; conditioning is a prepended
    # [time + obs] token). Best suited to longer horizons (chunk_size > 1).
    transformer_d_model: int = 256
    transformer_nhead: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.0
    transformer_activation: str = "gelu"

    # ----- Observation conditioning encoder -----
    # MLP applied on top of the (featurized) observation to produce the global conditioning
    # vector. Empty tuple => use the raw featurized observation as conditioning.
    obs_encoder_hidden_dims: Tuple[int, ...] = (256, 256)

    # ----- Exponential moving average of weights (used for inference/eval) -----
    use_ema: bool = True
    ema_decay: float = 0.995

    # ----- Optional data-augmentation regularizers (parity with BC / DP) -----
    obs_noise_std: float = 0.0
    action_noise_std: float = 0.0

    # ----- Weighted cloning (HITL SIRIUS / IWR) -----
    use_weighted_fm: bool = False

    # ----- Diagnostics -----
    # If > 0, every N updates run the (expensive) Euler ODE sampler on the training batch and log
    # the resulting action MSE against the expert actions. 0 disables it.
    log_sample_metrics_every: int = 0

    @property
    def algorithm_class(self):
        from robometer_policy_learning.algorithms.flow_matching import FlowMatching

        return FlowMatching
