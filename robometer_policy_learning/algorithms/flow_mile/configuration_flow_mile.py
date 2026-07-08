from dataclasses import dataclass
from typing import Optional, Tuple

from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig


@dataclass
class FlowMILEConfig(BaseAlgorithmConfig):
    """
    Configuration for the Flow MILE (FlowMILE) algorithm.

    FlowMILE is the flow-matching analogue of :class:`DiffusionMILEConfig`: it is MILE for a
    :class:`FlowMatchingActor`, using the conditional flow-matching loss as the log-probability
    proxy instead of the diffusion denoising loss. It consumes a pre-built ``FlowMatchingActor``
    (loaded from a Flow Matching checkpoint), so it does NOT carry the actor's network / flow
    hyperparameters — those live on the actor (e.g. ``sigma_min``). Only fields read by
    ``FlowMILE`` are defined here, and they mirror :class:`DiffusionMILEConfig` field-for-field.
    """

    # Runtime fields (inherited from BaseAlgorithmConfig: env, actor, critic, buffer, logger)

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

    # ----- Exponential moving average of weights (used for inference/eval) -----
    use_ema: bool = True
    ema_decay: float = 0.995

    # ----- Optional data-augmentation regularizers (parity with BC) -----
    obs_noise_std: float = 0.0
    action_noise_std: float = 0.0

    # ----- Diagnostics -----
    # If > 0, every N updates run the (expensive) Euler ODE sampler on the training batch and log
    # the resulting action MSE against the expert actions. 0 disables it.
    log_sample_metrics_every: int = 0
    # Score diagnostics (human / non-intervention / rollout score means) each cost extra flow-matching
    # forward passes, so they are computed only every N gradient steps rather than every step. <= 0
    # disables them entirely. The free per-class intervention-prob diagnostics are always logged.
    log_score_metrics_every: int = 25

    # Convex weight of the probit intervention (BCE) loss.
    lambda_intervention: float = 1.0
    # Probit threshold offset `c` (higher => the human is modelled as intervening less readily).
    intervention_cost: float = 0.0
    # Probit scale 'beta'
    probit_scale: float = 1.0
    # Monte-Carlo samples K used to estimate the probit intervention probability.
    monte_carlo_samples: int = 50
    # Euler ODE steps for the Monte-Carlo action sampling (probit baseline / MILE marginal / anchor
    # chunks). None => use the actor's own num_inference_steps (as before). Set lower for cheaper MC
    # sampling; the deployment/eval sampler and the sample-MSE diagnostic keep the actor's full steps.
    mc_num_inference_steps: Optional[int] = None
    # Condition the intervention probability on the action for intervention steps.
    condition_intervention_on_action: bool = False
    # Condition the non-intervention baseline on the logged robot action: when True, the probit
    # baseline E_{a0~pi0} ell(a0,s) is replaced by ell(a_logged, s) on non-intervention (label-0)
    # transitions (the analogue of condition_intervention_on_action for the robot term).
    condition_nonintervention_on_robot: bool = False

    # ----- Flow-matching log-prob proxy (flow-matching loss) -----
    # Flow-matching samples (t, x0) used to estimate the per-sample score ell(a, s).
    # Higher => lower-variance score estimate at higher compute cost.
    score_monte_carlo_samples: int = 1
    # If True, use the reference-relative score ell_{theta,0} = loss_0 - loss_theta (online vs.
    # frozen rollout policy) instead of the plain score -loss_theta. Requires a rollout_policy.
    reference_relative_score: bool = True

    # ----- Anchor regularizer (fine-tuning stability) -----
    # Coefficient for matching the online velocity field to the frozen rollout velocity field on the
    # buffer's stored non-intervention (label-0) actions. 0 disables the anchor loss.
    anchor_loss_weight: float = 0.0
    # Flow-matching (t, x0) noise draws averaged per anchor action to estimate the anchor loss;
    # must be >= 1.
    anchor_monte_carlo_samples: int = 1

    # ----- Proximal score regularizer (score-magnitude stability) -----
    # Coefficient for the symmetric softplus proximal penalty
    # L_prox = mean 0.5 * [softplus(r) + softplus(-r)] applied to the online actor's score r of the
    # rollout-policy samples (the probit baseline term). It keeps that score near 0, penalizing the
    # tendency to satisfy the MILE gap by driving the rollout score strongly negative rather than by
    # raising the human-correction score. 0 disables it. Uses the same score definition as the probit
    # (reference-relative when reference_relative_score=True).
    proximal_loss_weight: float = 0.0

    # ----- Running-stats normalization of the marginal score gap (ell_theta - E[ell_rollout]) -----
    # When True, the per-state score gap fed to the probit is standardized by an EMA of its running
    # mean/variance BEFORE applying probit_scale / intervention_cost. This decouples those two
    # hyperparameters from the (arbitrary, drifting) absolute scale of the flow-matching score, so
    # the CDF argument stays near unit scale and probit_scale~1 / intervention_cost~0 are sensible
    # starting points.
    normalize_score_gaps: bool = False
    # EMA decay for the running score-gap mean/variance (closer to 1 => slower, smoother tracking).
    score_gap_ema_decay: float = 0.99
    # Numerical floor added to the running variance before taking the square root.
    score_gap_norm_eps: float = 1e-6
    # Lower bound on the normalization std (guards against div-by-tiny when gaps are near-constant).
    score_gap_std_min: float = 0.1
    # Clip the standardized gap to +/- this value (<= 0 disables clipping); keeps the CDF arg bounded.
    score_gap_clip: float = 5.0

    @property
    def algorithm_class(self):
        from robometer_policy_learning.algorithms.flow_mile import FlowMILE

        return FlowMILE
