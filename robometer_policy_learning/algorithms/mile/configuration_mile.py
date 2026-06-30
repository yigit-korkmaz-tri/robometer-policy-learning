from dataclasses import dataclass
from typing import Any, Optional

from robometer_policy_learning.algorithms.configuration_algorithm import BaseAlgorithmConfig


@dataclass
class MILEConfig(BaseAlgorithmConfig):
    """Configuration for MILE (Model-based Intervention LEarning).

    MILE trains a policy from human-in-the-loop data where each timestep carries an
    ``intervention`` flag (1 = the human took over and supplied the action). It combines:

      * a behaviour-cloning loss on the intervention steps (imitate the human's corrections),
      * a probit *intervention model* trained over all steps (BCE against the intervention
        labels) that compares the trainable policy to a frozen ``rollout_policy`` (the policy
        deployed during data collection; set via :meth:`MILE.set_rollout_policy`).

    The objective is ``action_loss + lambda_intervention * intervention_loss``.

    Works with both single-step and chunked actors.
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
    actor_optimizer_weight_decay: float = 0.0

    # ----- MILE-specific parameters -----
    # Loss used to imitate the human action on intervention steps.
    intervention_action_loss_type: str = "mse"  # ["mse", "nll", "huber", "smooth_l1"]
    # Convex weight of the probit intervention (BCE) loss.
    lambda_intervention: float = 1.0
    # Probit threshold offset `c` (higher => the human is modelled as intervening less readily).
    intervention_cost: float = 0.0
    # Probit scale 'beta'
    probit_scale: float = 1.0
    # Monte-Carlo samples K used to estimate the probit intervention probability.
    monte_carlo_samples: int = 50
    # Condition the intervention probability on the action for intervention steps.
    condition_intervention_on_action: bool = False
    # Stop gradient for monte carlo samples of the training policy.
    stop_gradient_for_monte_carlo_samples: bool = True

    # ----- Running-stats normalization of the log-prob gap (logπ_θ - E[logπ_θ(a₀)]) -----
    # When True, the per-state log-prob gap fed to the probit is standardized by an EMA of its
    # running mean/variance BEFORE applying probit_scale / intervention_cost. This decouples those
    # two hyperparameters from the (arbitrary, drifting) absolute scale of the log-probabilities, so
    # the CDF argument stays near unit scale and probit_scale~1 / intervention_cost~0 are sensible
    # starting points.
    normalize_logprob_gaps: bool = False
    # EMA decay for the running log-prob-gap mean/variance (closer to 1 => slower, smoother tracking).
    logprob_gap_ema_decay: float = 0.99
    # Numerical floor added to the running variance before taking the square root.
    logprob_gap_norm_eps: float = 1e-6
    # Lower bound on the normalization std (guards against div-by-tiny when gaps are near-constant).
    logprob_gap_std_min: float = 0.1
    # Clip the standardized gap to +/- this value (<= 0 disables clipping); keeps the CDF arg bounded.
    logprob_gap_clip: float = 5.0

    # ----- Regularization / anti-overfitting (parity with BC) -----
    l2_regularization: float = 0.0
    obs_noise_std: float = 0.0
    action_noise_std: float = 0.0
    gradient_penalty_weight: float = 0.0
    consistency_weight: float = 0.0
    clip_grad_norm: float = 10.0

    @property
    def algorithm_class(self):
        from robometer_policy_learning.algorithms.mile import MILE

        return MILE
