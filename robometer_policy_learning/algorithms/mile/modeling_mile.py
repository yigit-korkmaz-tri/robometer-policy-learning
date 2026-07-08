import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.algorithms.mile.configuration_mile import MILEConfig
from robometer_policy_learning.modules.base import BaseActor
from robometer_policy_learning.modules.base.distributions import SquashedDiagGaussianDistribution


EPS = 1e-4


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF Φ(x).
    Keep the x in the range [-3, 3] to avoid numerical instability.
    
    Args:
        x: Tensor of shape [B]

    Returns:
        Tensor of shape [B]
    """
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def probit_cdf_diagnostics(raw_diff: torch.Tensor, probit_arg: torch.Tensor) -> dict:
    """Per-step summary stats of the probit Gaussian-CDF input, for tuning probit_scale / intervention_cost.

    ``raw_diff`` is the UNNORMALIZED log-prob (MILE) or denoising-score (DiffusionMILE) gap;
    ``probit_arg`` is the NORMALIZED value ``probit_scale * raw_diff - intervention_cost`` actually
    passed to :func:`normal_cdf`. Keep ``probit_arg`` roughly within [-3, 3] (by tuning
    ``probit_scale`` / ``intervention_cost``) so Φ does not saturate and intervention probabilities
    do not collapse to 0/1. Returns mean/std/min/max for both tensors.
    """
    raw = raw_diff.detach()
    arg = probit_arg.detach()
    return {
        "diff_raw_mean": float(raw.mean()), "diff_raw_std": float(raw.std()),
        "diff_raw_min": float(raw.min()), "diff_raw_max": float(raw.max()),
        "cdf_arg_mean": float(arg.mean()), "cdf_arg_std": float(arg.std()),
        "cdf_arg_min": float(arg.min()), "cdf_arg_max": float(arg.max()),
    }


def aggregate_probit_diagnostics(diag_list: list, prefix: str = "probit_") -> dict:
    """Aggregate per-step :func:`probit_cdf_diagnostics` dicts into logged metrics.

    Means and stds are averaged across the gradient steps; mins/maxes are reduced by min/max so the
    logged extremes reflect the true observed range of the CDF input over the whole training step.
    Keys are prefixed with ``prefix`` (e.g. ``probit_diff_raw_mean``, ``probit_cdf_arg_max``).
    """
    if not diag_list:
        return {}
    out = {}
    for base in ("diff_raw", "cdf_arg"):
        out[f"{prefix}{base}_mean"] = float(np.mean([d[f"{base}_mean"] for d in diag_list]))
        out[f"{prefix}{base}_std"] = float(np.mean([d[f"{base}_std"] for d in diag_list]))
        out[f"{prefix}{base}_min"] = float(np.min([d[f"{base}_min"] for d in diag_list]))
        out[f"{prefix}{base}_max"] = float(np.max([d[f"{base}_max"] for d in diag_list]))
    return out


def proximal_score_loss(scores: torch.Tensor) -> torch.Tensor:
    """Symmetric softplus proximal penalty keeping a score r_theta(s, a) near zero.

        L_prox = mean_{s,a} 0.5 * [ softplus(r) + softplus(-r) ]

    Minimized at r = 0 and growing ~|r| for large |r| (a smooth, L1-like magnitude penalty),
    so it discourages the score from drifting to large positive or negative values.
    """
    return 0.5 * (F.softplus(scores) + F.softplus(-scores)).mean()


class MILE(BaseAlgorithm):
    """
    Model-based Intervention Learning (MILE) algorithm.
    A model-based intervention learning approach that trains a policy using supervised learning
    on interventions.
    """

    def __init__(self, config: MILEConfig):
        super().__init__(config)
        self.config = config
        self.actor = config.actor

        if self.actor is None:
            raise ValueError("Actor is required for MILE")

        self.device = next(self.actor.parameters()).device

        self.component_names = [
            "actor",
            "actor_optimizer",
        ]

        self.buffer = config.buffer
        self.rollout_policy = None
        self._last_probit_diagnostics = {}  # CDF-input stats from the latest _compute_intervention_probs

        # Store configuration
        self.batch_size = config.batch_size
        self.learning_starts = config.learning_starts
        self.intervention_action_loss_type = config.intervention_action_loss_type
        self.l2_regularization = config.l2_regularization
        self.monte_carlo_samples = config.monte_carlo_samples
        self.intervention_cost = config.intervention_cost
        self.probit_scale = config.probit_scale
        self.lambda_intervention = config.lambda_intervention
        self.condition_intervention_on_action = config.condition_intervention_on_action
        self.stop_gradient_for_monte_carlo_samples = config.stop_gradient_for_monte_carlo_samples

        # Optional running-stats standardization of the probit log-prob gap.
        self._init_logprob_gap_normalization()

        # if not 0.0 <= self.lambda_intervention <= 1.0:
        #     raise ValueError(
        #         f"lambda_intervention must be in [0, 1], got {self.lambda_intervention}"
        #     )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_optimizer_lr,
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        self.is_squashed = isinstance(
            self.actor.action_dist,
            SquashedDiagGaussianDistribution,
        )

        print(f"MILE: intervention_action_loss_type = {self.intervention_action_loss_type}")
        print(f"MILE: l2_regularization = {self.l2_regularization}")
        print(f"MILE: probit_scale = {self.probit_scale}")
        print(f"MILE: lambda_intervention = {self.lambda_intervention}")
        print(f"MILE: condition_intervention_on_action = {self.condition_intervention_on_action}")
        print(f"MILE: stop_gradient_for_monte_carlo_samples = {self.stop_gradient_for_monte_carlo_samples}")
        print(f"MILE: normalize_logprob_gaps = {self.normalize_logprob_gaps}")
        if self.normalize_logprob_gaps:
            print(
                f"MILE: logprob_gap_ema_decay = {self.logprob_gap_ema_decay}, "
                f"logprob_gap_std_min = {self.logprob_gap_std_min}, logprob_gap_clip = {self.logprob_gap_clip}"
            )

    def _init_logprob_gap_normalization(self):
        """Set up the optional EMA running-stats normalizer for the probit log-prob gap."""
        self.normalize_logprob_gaps = bool(self.config.normalize_logprob_gaps)
        self.logprob_gap_ema_decay = float(self.config.logprob_gap_ema_decay)
        self.logprob_gap_norm_eps = float(self.config.logprob_gap_norm_eps)
        self.logprob_gap_std_min = float(self.config.logprob_gap_std_min)
        self.logprob_gap_clip = float(self.config.logprob_gap_clip)

        self.logprob_gap_stats_initialized = False
        self.logprob_gap_mean = torch.zeros((), device=self.device)
        self.logprob_gap_var = torch.ones((), device=self.device)

    @torch.no_grad()
    def _update_logprob_gap_stats(self, gaps: torch.Tensor):
        """EMA-update the running mean/variance of the log-prob gap (no grad; detaches inputs)."""
        if not self.normalize_logprob_gaps:
            return

        gaps = gaps.detach().reshape(-1)
        if gaps.numel() == 0:
            return

        batch_mean = gaps.mean()
        batch_var = gaps.var(unbiased=False).clamp_min(self.logprob_gap_norm_eps)

        if not self.logprob_gap_stats_initialized:
            self.logprob_gap_mean.copy_(batch_mean)
            self.logprob_gap_var.copy_(batch_var)
            self.logprob_gap_stats_initialized = True
        else:
            d = self.logprob_gap_ema_decay
            self.logprob_gap_mean.mul_(d).add_(batch_mean, alpha=1.0 - d)
            self.logprob_gap_var.mul_(d).add_(batch_var, alpha=1.0 - d)

    def _normalize_logprob_gap(self, gap: torch.Tensor) -> torch.Tensor:
        """Standardize ``gap`` by the running mean/std (detached) and optionally clip.

        Gradients flow through ``gap``'s numerator only; the running stats are detached so the
        normalizer is non-differentiable. No-op when normalization is disabled.
        """
        if not self.normalize_logprob_gaps:
            return gap

        std = torch.sqrt(self.logprob_gap_var.detach() + self.logprob_gap_norm_eps)
        std = std.clamp_min(self.logprob_gap_std_min)

        gap = (gap - self.logprob_gap_mean.detach()) / std

        if self.logprob_gap_clip > 0:
            gap = gap.clamp(-self.logprob_gap_clip, self.logprob_gap_clip)

        return gap

    def set_rollout_policy(self, rollout_policy: BaseActor):
        """Set the frozen rollout policy / mental model for MILE."""
        self.rollout_policy = rollout_policy
        self.rollout_policy.eval()
        self.rollout_policy.to(self.device)

        for param in self.rollout_policy.parameters():
            param.requires_grad_(False)

    def _nll_target_actions(self, expert_actions: torch.Tensor) -> torch.Tensor:
        """Clamp expert actions just inside (-1, 1) for a tanh-squashed policy's NLL.

        For an unsquashed Gaussian the support is all of R, so no clamping is needed.
        """
        if self.is_squashed:
            return expert_actions.clamp(-1.0 + EPS, 1.0 - EPS)
        return expert_actions

    @staticmethod
    def _reduce_chunk_logprob(log_prob: torch.Tensor) -> torch.Tensor:
        """Collapse a per-sample log-prob to ``[K, B]`` (one scalar per Monte-Carlo sample, state).

        The distribution's ``log_prob`` sums over the action dim only, so for a chunked actor
        (mean/log_std shaped ``[B, chunk, A]``) it returns ``[K, B, chunk]`` — one log-prob per
        action step. We reduce the chunk by the MEAN (matching ``_compute_action_log_probs``),
        keeping the probit argument on a per-step scale so ``intervention_cost`` stays
        interpretable regardless of ``chunk_size``. Single-step actors return ``[K, B]`` unchanged.
        """
        if log_prob.dim() >= 3:
            return log_prob.mean(dim=tuple(range(2, log_prob.dim())))
        return log_prob

    def _compute_intervention_probs(
        self,
        obs: torch.Tensor,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor | None = None,
        interventions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use a probit model to compute the per-state probability of human intervention.

        Works for single-step (``mean_actions`` shaped ``[B, A]``) and chunked (``[B, chunk, A]``)
        actors; the chunk dim is averaged out so the returned probabilities are ``[B]`` (one per
        state, matching the per-step intervention labels).

        Args:
            obs: Observation
            mean_actions: Mean actions of the training policy
            log_std: Log standard deviation of the training policy

        Returns:
            Intervention probabilities, shape ``[B]``
        """
        if self.rollout_policy is None:
            raise RuntimeError(
                "MILE requires a frozen rollout_policy. Call set_rollout_policy() before training."
            )
        # Rollout policy samples:
        with torch.no_grad():
            rollout_mean_actions, rollout_log_std, _ = self.rollout_policy.get_action_dist_params(obs)
            rollout_dist = self.rollout_policy.action_dist.proba_distribution(rollout_mean_actions, rollout_log_std)
            rollout_samples = rollout_dist.sample(num_samples=self.monte_carlo_samples)

        training_dist = self.actor.action_dist.proba_distribution(mean_actions, log_std)

        # Log-prob of the rollout policy's samples under the training policy. For a chunked actor
        # log_prob is [K, B, chunk]; _reduce_chunk_logprob averages the chunk -> [K, B].
        log_pi_rollout_samples = self._reduce_chunk_logprob(training_dist.log_prob(rollout_samples))  # [K, B]
        expected_log_pi_rollout = log_pi_rollout_samples.mean(dim=0)  # [B]

        # ------------------------------------------------------------
        # Original MILE marginal:
        # p(nu=1 | s) = E_{a~pi_theta} Phi(log pi_theta(a) - E log pi_theta(a0) - c)
        # ------------------------------------------------------------
        training_samples = training_dist.sample(num_samples=self.monte_carlo_samples)
        if self.stop_gradient_for_monte_carlo_samples:
            training_samples = training_samples.detach()
        log_pi_training_samples = self._reduce_chunk_logprob(
            training_dist.log_prob(training_samples)
        )  # [K, B]

        probit_diff_marginal = log_pi_training_samples - expected_log_pi_rollout.unsqueeze(0)  # [K, B]

        # Optionally standardize the log-prob gap by its EMA running mean/std before the probit, so
        # probit_scale / intervention_cost are decoupled from the absolute (drifting) scale of the
        # log-probabilities. Stats are updated from (and only from) this marginal gap; the affine
        # transform keeps gradients flowing through the numerator (mean/std are detached).
        self._update_logprob_gap_stats(probit_diff_marginal)
        normalized_diff_marginal = self._normalize_logprob_gap(probit_diff_marginal)
        probit_arg_marginal = self.probit_scale * normalized_diff_marginal - self.intervention_cost

        # Record CDF-input diagnostics: the RAW (pre-normalization) gap vs the argument actually fed
        # to Φ, so probit_scale / intervention_cost can be tuned to keep the argument within ~[-3, 3].
        self._last_probit_diagnostics = probit_cdf_diagnostics(probit_diff_marginal, probit_arg_marginal)

        marginal_intervention_probs = normal_cdf(probit_arg_marginal)
        marginal_intervention_probs = marginal_intervention_probs.mean(dim=0)  # [B]

        intervention_probs = marginal_intervention_probs

        # ------------------------------------------------------------
        # Optional: for observed online interventions, use p(nu=1 | s, a_h).
        # This replaces only intervention == 1 samples.
        # No-intervention samples remain marginalized over a~pi_theta.
        # Offline samples intervention == 2 are ignored by the BCE later anyway.
        # ------------------------------------------------------------
        if self.condition_intervention_on_action:
            if actions is None or interventions is None:
                raise ValueError(
                    "condition_intervention_on_action=True requires actions and interventions."
                )

            observed_intervention_mask = (interventions > 0.5) & (interventions < 1.5)

            if observed_intervention_mask.any():
                log_pi_observed_actions = self._compute_action_log_probs(
                    actions=actions,
                    mean_actions=mean_actions,
                    log_std=log_std,
                )  # [B]

                # Standardize with the SAME running stats as the marginal (do not update them from
                # the observed gap) so both probit arguments live on the same scale.
                normalized_observed_gap = self._normalize_logprob_gap(
                    log_pi_observed_actions - expected_log_pi_rollout
                )
                probit_arg_observed = self.probit_scale * normalized_observed_gap - self.intervention_cost

                observed_intervention_probs = normal_cdf(probit_arg_observed)

                intervention_probs = intervention_probs.clone()
                intervention_probs[observed_intervention_mask] = observed_intervention_probs[
                    observed_intervention_mask
                ]

        intervention_probs = intervention_probs.clamp(EPS, 1.0 - EPS)
        return intervention_probs

    def _compute_action_log_probs(
        self,
        actions: torch.Tensor,
        mean_actions: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log pi_theta(actions | obs) with support clamping for squashed policies.

        Returns:
            log_prob_actions: Tensor [B]
        """
        nll_target_actions = self._nll_target_actions(actions)

        if actions.dim() == 3:
            # actions: [B, chunk_size, action_dim]
            batch_size, chunk_size, action_dim = actions.shape

            actions_flat = nll_target_actions.reshape(-1, action_dim)
            mean_actions_flat = mean_actions.reshape(-1, action_dim)

            if log_std is not None:
                if log_std.shape == mean_actions.shape:
                    log_std_flat = log_std.reshape(-1, action_dim)
                else:
                    # Handles global log_std, e.g. [D], [1, D], or broadcastable variants.
                    log_std_flat = log_std.reshape(-1, action_dim)

                    if log_std_flat.shape[0] == 1:
                        log_std_flat = log_std_flat.expand(mean_actions_flat.shape[0], -1)
                    elif log_std_flat.shape[0] != mean_actions_flat.shape[0]:
                        raise ValueError(
                            "Unsupported log_std shape for chunked actions: "
                            f"log_std.shape={tuple(log_std.shape)}, "
                            f"mean_actions.shape={tuple(mean_actions.shape)}"
                        )

                distribution_flat = self.actor.action_dist.proba_distribution(
                    mean_actions_flat,
                    log_std_flat,
                )
            else:
                distribution_flat = self.actor.action_dist.proba_distribution(mean_actions_flat)

            log_prob_actions = distribution_flat.log_prob(actions_flat)
            log_prob_actions = log_prob_actions.reshape(batch_size, chunk_size).mean(dim=1)
            return log_prob_actions

        distribution = (
            self.actor.action_dist.proba_distribution(mean_actions, log_std)
            if log_std is not None
            else self.actor.action_dist.proba_distribution(mean_actions)
        )

        log_prob_actions = distribution.log_prob(nll_target_actions)

        if log_prob_actions.dim() == 0:
            log_prob_actions = log_prob_actions.reshape(1)
        elif log_prob_actions.dim() > 1:
            log_prob_actions = log_prob_actions.reshape(log_prob_actions.shape[0], -1).mean(dim=-1)

        return log_prob_actions.reshape(-1)

    def _compute_gradient_penalty(self, obs) -> torch.Tensor:
        """Optional gradient penalty on actor outputs w.r.t. observations."""
        if isinstance(obs, dict):
            obs_for_grad = {
                k: v.detach().clone().requires_grad_(True)
                for k, v in obs.items()
            }
            grad_inputs = tuple(obs_for_grad.values())
        else:
            obs_for_grad = obs.detach().clone().requires_grad_(True)
            grad_inputs = obs_for_grad

        mean_actions_grad, _, _ = self.actor.get_action_dist_params(obs_for_grad)

        gradients = torch.autograd.grad(
            outputs=mean_actions_grad.sum(),
            inputs=grad_inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )

        if isinstance(obs_for_grad, dict):
            valid_grads = [g.flatten() for g in gradients if g is not None]
            if not valid_grads:
                return torch.zeros((), device=self.device)
            gradient_norm = torch.norm(torch.cat(valid_grads))
        else:
            gradient = gradients[0]
            if gradient is None:
                return torch.zeros((), device=self.device)
            gradient_norm = torch.norm(gradient)

        return self.config.gradient_penalty_weight * (gradient_norm - 1.0) ** 2

    def _extract_interventions(self, batch: dict) -> torch.Tensor:
        """Return the per-sample intervention label as a float tensor ``[B]``.

        Buffers batch ``info`` as a *list of per-transition dicts*
        (see ``BaseReplayBuffer._batch_transitions``), so the label normally lives at
        ``batch["info"][i]["intervention"]``. For robustness we also accept it already collated
        as ``batch["info"]["intervention"]`` or as a top-level ``batch["intervention"]``.

        Labels follow the 3-way convention: 0 = online (no intervention), 1 = online human
        intervention, 2 = offline-dataset sample.
        """
        info = batch.get("info", None)

        # (a) Already collated as a dict-of-arrays, or exposed as a top-level batch key.
        for container in (info, batch):
            if isinstance(container, dict) and "intervention" in container:
                return torch.as_tensor(
                    container["intervention"], dtype=torch.float32, device=self.device
                ).view(-1)

        # (b) The common case: a list/array of per-transition info dicts.
        if isinstance(info, (list, tuple, np.ndarray)):
            try:
                values = [float(np.asarray(d["intervention"]).reshape(-1)[0]) for d in info]
            except (KeyError, TypeError, IndexError) as e:
                raise KeyError(
                    "MILE expects an 'intervention' label in every transition's info dict "
                    "(batch['info'][i]['intervention'] in {0, 1, 2}); it was missing for at least "
                    "one sample. Ensure the buffer populates info['intervention']."
                ) from e
            return torch.tensor(values, dtype=torch.float32, device=self.device).view(-1)

        raise KeyError(
            "MILE could not find intervention labels. Expected batch['info'] to be a list of "
            "per-transition dicts each containing 'intervention' (0 = online no-intervention, "
            "1 = online intervention, 2 = offline)."
        )

    def train_step(self, logging_prefix: str = "mile") -> dict:
        """Perform one training step of model-based intervention learning."""
        if self.rollout_policy is None:
            raise RuntimeError(
                "MILE requires a frozen rollout_policy. Call set_rollout_policy() before training."
            )

        actor_losses = []
        intervention_losses = []
        total_losses = []
        actor_log_pis = []
        probit_diagnostics = []  # per-step CDF-input stats for probit_scale / intervention_cost tuning

        expert_action_means = []
        predicted_action_means = []

        mse_errors_all = []
        mse_errors_intervention = []

        unnormalized_mse_errors_all = []
        unnormalized_mse_errors_intervention = []

        unnormalized_max_predicted_actions = []
        unnormalized_min_predicted_actions = []

        gradient_steps = self.config.num_updates_per_train_step

        if gradient_steps != 1:
            print(f"Going to take {gradient_steps} training steps")
            print(self.buffer.size())

        for gradient_step in range(gradient_steps):
            batch = self.buffer.sample(self.batch_size, device=self.device)

            if not batch:
                print("Buffer is still empty. Skipping this training step")
                return {}

            obs = batch["obs"]
            actions = batch["action"]
            # Per-timestep intervention label: 0 = online (no intervention), 1 = online human
            # intervention, 2 = offline-dataset sample.
            #   * action imitation loss is applied to {1, 2} (human corrections + offline demos),
            #   * probit intervention BCE is applied to {0, 1} only (online steps with a real
            #     intervention signal); offline samples (2) carry no intervention label.
            interventions = self._extract_interventions(batch)
            action_mask = interventions > 0.5  # {1, 2}: imitate these actions
            online_mask = interventions < 1.5  # {0, 1}: include these in the probit BCE

            # Observation augmentation.
            if hasattr(self.config, "obs_noise_std") and self.config.obs_noise_std > 0:
                if isinstance(obs, dict):
                    obs = {
                        k: v + torch.randn_like(v) * self.config.obs_noise_std
                        for k, v in obs.items()
                    }
                else:
                    obs = obs + torch.randn_like(obs) * self.config.obs_noise_std

            # Action augmentation only for the imitated actions ({1, 2}).
            if hasattr(self.config, "action_noise_std") and self.config.action_noise_std > 0:
                actions = actions.clone()
                if action_mask.any():
                    actions[action_mask] = (
                        actions[action_mask]
                        + torch.randn_like(actions[action_mask]) * self.config.action_noise_std
                    )

            mean_actions, log_std, _ = self.actor.get_action_dist_params(obs)

            intervention_probs = self._compute_intervention_probs(
                obs=obs,
                mean_actions=mean_actions,
                log_std=log_std,
                actions=actions,
                interventions=interventions,
            )
            intervention_probs = intervention_probs.view_as(interventions)
            if self._last_probit_diagnostics:
                probit_diagnostics.append(self._last_probit_diagnostics)

            # Train the probit intervention model only on online steps {0, 1}; offline samples (2)
            # are excluded (they have no intervention signal, and a BCE target of 2 is invalid).
            if online_mask.any():
                intervention_loss = F.binary_cross_entropy(
                    intervention_probs[online_mask], interventions[online_mask]
                )
            else:
                intervention_loss = torch.zeros((), device=self.device)

            if log_std is not None:
                det_action = self.actor.action_dist.actions_from_params(
                    mean_actions,
                    log_std,
                    deterministic=True,
                )
            else:
                det_action = self.actor.action_dist.actions_from_params(
                    mean_actions,
                    deterministic=True,
                )

            if self.intervention_action_loss_type == "mse":
                if action_mask.any():
                    actor_loss = F.mse_loss(
                        det_action[action_mask],
                        actions[action_mask],
                    )
                else:
                    actor_loss = torch.zeros((), device=self.device)

                log_prob_actions = None

            elif self.intervention_action_loss_type == "nll":
                log_prob_actions = self._compute_action_log_probs(
                    actions=actions,
                    mean_actions=mean_actions,
                    log_std=log_std,
                )

                if action_mask.any():
                    actor_loss = -log_prob_actions[action_mask].mean()
                else:
                    actor_loss = torch.zeros((), device=self.device)

            elif self.intervention_action_loss_type == "huber":
                if action_mask.any():
                    actor_loss = F.huber_loss(
                        det_action[action_mask],
                        actions[action_mask],
                        delta=1.0,
                    )
                else:
                    actor_loss = torch.zeros((), device=self.device)

                log_prob_actions = None

            elif self.intervention_action_loss_type == "smooth_l1":
                if action_mask.any():
                    actor_loss = F.smooth_l1_loss(
                        det_action[action_mask],
                        actions[action_mask],
                    )
                else:
                    actor_loss = torch.zeros((), device=self.device)

                log_prob_actions = None

            else:
                raise ValueError(
                    f"Invalid intervention_action_loss_type: {self.intervention_action_loss_type}. "
                    "Must be 'mse', 'nll', 'huber', or 'smooth_l1'."
                )

            # L2 regularization.
            if self.l2_regularization > 0:
                l2_reg = torch.zeros((), device=self.device)
                for param in self.actor.parameters():
                    l2_reg = l2_reg + torch.norm(param) ** 2
                actor_loss = actor_loss + self.l2_regularization * l2_reg

            # Optional gradient penalty.
            if hasattr(self.config, "gradient_penalty_weight") and self.config.gradient_penalty_weight > 0:
                actor_loss = actor_loss + self._compute_gradient_penalty(obs)

            # Optional consistency regularization.
            if hasattr(self.config, "consistency_weight") and self.config.consistency_weight > 0:
                if isinstance(obs, dict):
                    obs_noisy = {
                        k: v + torch.randn_like(v) * 0.01
                        for k, v in obs.items()
                    }
                else:
                    obs_noisy = obs + torch.randn_like(obs) * 0.01

                mean_actions_noisy, _, _ = self.actor.get_action_dist_params(obs_noisy)
                consistency_loss = F.mse_loss(mean_actions, mean_actions_noisy)
                actor_loss = actor_loss + self.config.consistency_weight * consistency_loss

            total_loss = (
                actor_loss
                + self.lambda_intervention * intervention_loss
            )

            self.actor_optimizer.zero_grad()
            total_loss.backward()

            if hasattr(self.config, "clip_grad_norm") and self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.config.clip_grad_norm,
                )

            self.actor_optimizer.step()

            # --------------------
            # Logging / diagnostics
            # --------------------
            actor_losses.append(actor_loss.item())
            intervention_losses.append(intervention_loss.item())
            total_losses.append(total_loss.item())

            mse_error_all = F.mse_loss(det_action, actions)
            mse_errors_all.append(mse_error_all.item())

            unnormalized_det_action = self.actor.unnormalize_action(det_action)
            unnormalized_actions = self.actor.unnormalize_action(actions)

            unnormalized_mse_error_all = F.mse_loss(
                unnormalized_det_action,
                unnormalized_actions,
            )
            unnormalized_mse_errors_all.append(unnormalized_mse_error_all.item())

            if action_mask.any():
                mse_error_intervention = F.mse_loss(
                    det_action[action_mask],
                    actions[action_mask],
                )
                mse_errors_intervention.append(mse_error_intervention.item())

                unnormalized_mse_error_intervention = F.mse_loss(
                    unnormalized_det_action[action_mask],
                    unnormalized_actions[action_mask],
                )
                unnormalized_mse_errors_intervention.append(
                    unnormalized_mse_error_intervention.item()
                )

                intervention_actions = actions[action_mask]
                if intervention_actions.dim() == 3:
                    expert_action_means.append(
                        intervention_actions.mean(dim=1).mean().item()
                    )
                else:
                    expert_action_means.append(intervention_actions.mean().item())
            else:
                mse_errors_intervention.append(np.nan)
                unnormalized_mse_errors_intervention.append(np.nan)
                expert_action_means.append(np.nan)

            unnormalized_max_predicted_actions.append(unnormalized_det_action.max().item())
            unnormalized_min_predicted_actions.append(unnormalized_det_action.min().item())

            if self.intervention_action_loss_type == "nll" and log_prob_actions is not None:
                if action_mask.any():
                    actor_log_pis.append(log_prob_actions[action_mask].mean().item())
                else:
                    actor_log_pis.append(np.nan)
            else:
                actor_log_pis.append(actor_loss.item())

            predicted_action_means.append(det_action.mean().item())

        self.step_counter += gradient_steps

        metrics_dict = {
            "total_loss": np.mean(total_losses),
            "actor_loss": np.mean(actor_losses),
            "intervention_loss": np.mean(intervention_losses),

            "expert_action_mean": np.nanmean(expert_action_means),
            "predicted_action_mean": np.mean(predicted_action_means),

            "mse_error_all": np.mean(mse_errors_all),
            "mse_error_intervention": np.nanmean(mse_errors_intervention),

            "unnormalized_mse_error_all": np.mean(unnormalized_mse_errors_all),
            "unnormalized_mse_error_intervention": np.nanmean(
                unnormalized_mse_errors_intervention
            ),

            "unnormalized_max_predicted_actions": np.mean(
                unnormalized_max_predicted_actions
            ),
            "unnormalized_min_predicted_actions": np.mean(
                unnormalized_min_predicted_actions
            ),
        }

        if self.intervention_action_loss_type == "nll":
            metrics_dict["actor_log_pis_mean"] = np.nanmean(actor_log_pis)
        else:
            metrics_dict["action_loss_value"] = np.mean(actor_log_pis)

        # Probit CDF-input diagnostics: probit_diff_raw_* (unnormalized log-prob gap) and
        # probit_cdf_arg_* (the value fed to Φ); keep cdf_arg within ~[-3, 3] when tuning.
        metrics_dict.update(aggregate_probit_diagnostics(probit_diagnostics))

        # Running log-prob-gap normalization stats (only meaningful when normalize_logprob_gaps=True).
        if self.normalize_logprob_gaps and self.logprob_gap_stats_initialized:
            metrics_dict["logprob_gap_running_mean"] = float(self.logprob_gap_mean.item())
            metrics_dict["logprob_gap_running_std"] = float(
                torch.sqrt(self.logprob_gap_var + self.logprob_gap_norm_eps).item()
            )

        self.logger.log(metrics_dict, step=self.step_counter, prefix=logging_prefix)

        return metrics_dict


@torch.no_grad()
def mile_intervention_prob(judge_policy, other_policy, obs, num_samples: int = 50, intervention_cost: float = 0.0):
    """MILE probit probability that ``judge_policy`` intervenes on ``other_policy`` at ``obs``.

    Mirrors :meth:`MILE._compute_intervention_probs` (inference-only / no grad):

        p = E_{a_j~judge}[ Φ( logπ_judge(a_j) - E_{a_o~other}[logπ_judge(a_o)] - intervention_cost ) ]

    i.e. it is high when the *other* policy's actions are unlikely under the *judge*. For a
    simulated human, pass ``judge=expert_policy`` and ``other=rollout_policy``. Both policies must
    expose an explicit Gaussian ``action_dist`` (MLP/RNN/Transformer actors, not DiffusionActor).
    Returns a tensor of shape ``[B]`` (B inferred from ``obs``).
    """
    o_mean, o_log_std, _ = other_policy.get_action_dist_params(obs)
    o_dist = other_policy.action_dist.proba_distribution(o_mean, o_log_std)
    o_samples = o_dist.sample(num_samples=num_samples)

    j_mean, j_log_std, _ = judge_policy.get_action_dist_params(obs)
    j_dist = judge_policy.action_dist.proba_distribution(j_mean, j_log_std)

    # log π_judge over the other policy's samples (chunk dim, if any, averaged) -> [B]
    expected_log_j_other = MILE._reduce_chunk_logprob(j_dist.log_prob(o_samples)).mean(dim=0)
    # log π_judge over the judge's own samples -> [K, B]
    j_samples = j_dist.sample(num_samples=num_samples)
    log_j_self = MILE._reduce_chunk_logprob(j_dist.log_prob(j_samples))

    probit_arg = log_j_self - expected_log_j_other.unsqueeze(0) - intervention_cost
    return normal_cdf(probit_arg).mean(dim=0).clamp(EPS, 1.0 - EPS)


def make_mile_intervention_criteria(
    num_samples: int = 50, intervention_cost: float = 0.0, stochastic: bool = True, threshold: float = 0.5, **_
):
    """Build a simulated-human intervention criterion from MILE's probit model.

    The expert is the "judge": each step it computes ``p = mile_intervention_prob(expert,
    rollout, obs)``. If ``stochastic`` it intervenes with ``Bernoulli(p)``, else when
    ``p > threshold``. On intervention the expert action is executed; otherwise the rollout action.
    Returns a function with the HITL criterion signature
    ``(obs, rollout_policy, expert_policy, rollout_action, expert_action) -> (intervened, action)``.
    """

    def criteria(obs, rollout_policy, expert_policy, rollout_action, expert_action,
                 rollout_chunk=None, expert_chunk=None):
        p = float(
            mile_intervention_prob(
                expert_policy, rollout_policy, obs, num_samples=num_samples, intervention_cost=intervention_cost
            ).reshape(-1)[0]
        )
        intervene = bool(np.random.random() < p) if stochastic else bool(p > threshold)
        return intervene, (expert_action if intervene else rollout_action)

    return criteria


@torch.no_grad()
def _action_logprob_under(policy, obs, env_action):
    """log π_policy(env_action | obs) for a single env-space action.

    Maps the env-space action into the policy's normalized [-1, 1] space and evaluates it under the
    policy's action distribution at ``obs`` (the first predicted step if the policy is chunked).
    """
    device = next(policy.parameters()).device
    a_norm = policy.normalize_action(torch.as_tensor(np.asarray(env_action, dtype=np.float32), device=device))
    if isinstance(policy.action_dist, SquashedDiagGaussianDistribution):
        a_norm = a_norm.clamp(-1.0 + EPS, 1.0 - EPS)
    mean, log_std, _ = policy.get_action_dist_params(obs)
    if mean.dim() >= 3:  # [B, chunk, A] -> compare against the first predicted step
        mean = mean[:, 0]
        if log_std is not None and log_std.dim() >= 3:
            log_std = log_std[:, 0]
    mean = mean.reshape(1, -1)
    if log_std is not None:
        dist = policy.action_dist.proba_distribution(mean, log_std.reshape(1, -1))
    else:
        dist = policy.action_dist.proba_distribution(mean)
    return float(dist.log_prob(a_norm.reshape(1, -1)).reshape(-1)[0])


def make_mile_window_criteria(
    window: int = 10, intervention_cost: float = 0.0, stochastic: bool = True, threshold: float = 0.5, **_
):
    """Windowed MILE criterion: an EMA over the per-step expert-vs-robot log-prob gap.

    Per step ``g_t = logπ_expert(a_expert_t) - logπ_expert(a_robot_t) - intervention_cost`` (>= 0
    when the robot's *executed* action is less likely under the expert than the expert's own
    action). An EMA with decay ``alpha = 2/(window+1)`` smooths ``g_t`` so intervention reflects
    *sustained* divergence rather than a single-step blip; ``p = Φ(ema)``, then ``Bernoulli(p)``
    (stochastic) or ``p > threshold``. ``intervention_cost`` raises the bar (less intervention).

    Stateful (EMA per episode); exposes ``.reset()`` (called by the rollout worker each episode).
    """
    alpha = 2.0 / (float(max(1, window)) + 1.0)
    state = {"ema": None}

    def criteria(obs, rollout_policy, expert_policy, rollout_action, expert_action,
                 rollout_chunk=None, expert_chunk=None):
        lp_expert = _action_logprob_under(expert_policy, obs, expert_action)
        lp_robot = _action_logprob_under(expert_policy, obs, rollout_action)
        g = (lp_expert - lp_robot) - intervention_cost
        state["ema"] = g if state["ema"] is None else alpha * g + (1.0 - alpha) * state["ema"]
        p = float(normal_cdf(torch.tensor(state["ema"], dtype=torch.float32)))
        intervene = bool(np.random.random() < p) if stochastic else bool(p > threshold)
        return intervene, (expert_action if intervene else rollout_action)

    def reset():
        state["ema"] = None

    criteria.reset = reset
    return criteria
