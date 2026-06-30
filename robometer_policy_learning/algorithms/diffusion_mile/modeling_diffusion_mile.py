"""Diffusion MILE for diffusion-policy actors.

This module implements a diffusion/generative-policy analogue of MILE.  It is meant to
live next to the Gaussian ``modeling_mile.py`` implementation, but it does not require an
explicit action distribution.  Instead of Gaussian log probabilities it uses the standard
Diffusion Policy denoising objective as a log-probability proxy:

    ell_theta(a, s) = - E_{t, eps} || target - model_theta(noise(a, t, eps), t, s) ||^2

For online intervention samples, an optional flag uses p(nu=1 | s, a_h) with the observed
human action.  For online no-intervention samples, the human action is latent, so it uses
MILE's marginal p(nu=1 | s), estimated by Monte Carlo samples from the trainable policy.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.algorithms.diffusion_mile.configuration_diffusion_mile import DiffusionMILEConfig
from robometer_policy_learning.algorithms.dp.modeling_dp import DiffusionActor, EMAModel
from robometer_policy_learning.modules.base import BaseActor
from robometer_policy_learning.algorithms.mile.modeling_mile import (
    aggregate_probit_diagnostics,
    normal_cdf,
    probit_cdf_diagnostics,
)


EPS = 1e-4


class DiffusionMILE(BaseAlgorithm):
    """MILE for DiffusionActor policies using denoising-score proxies.

    Required config fields are intentionally aligned with the existing MILE/DP configs:

    * actor: DiffusionActor
    * buffer, batch_size, learning_starts, num_updates_per_train_step
    * monte_carlo_samples, intervention_cost, probit_scale, lambda_intervention
    * condition_intervention_on_action, actor_optimizer_lr
    * actor_optimizer_eps, actor_optimizer_weight_decay

    Optional fields:

    * score_monte_carlo_samples: number of diffusion-noising samples used to estimate ell(a,s)
    * reference_relative_score: use ell_{theta,0} = loss_0 - loss_theta instead of -loss_theta
    * anchor_loss_weight: coefficient for matching the online denoiser to the frozen rollout
      denoiser on actions sampled from the rollout policy
    * anchor_monte_carlo_samples: number of diffusion noising samples (t, eps) per stored
      no-intervention rollout action used for anchor loss
    * normalize_score_gaps: standardize the probit score gap by an EMA running mean/std before
      applying probit_scale / intervention_cost (with score_gap_ema_decay, score_gap_norm_eps,
      score_gap_std_min, score_gap_clip)
    * use_ema, ema_decay: maintain an EMA deployment actor, as in DP
    * actor_optimizer_betas: AdamW betas
    * obs_noise_std, action_noise_std, clip_grad_norm
    """

    def __init__(self, config: DiffusionMILEConfig):
        super().__init__(config)
        self.config = config

        online_actor = config.actor
        if online_actor is None:
            raise ValueError("A DiffusionActor is required for DiffusionMILE")
        if not isinstance(online_actor, DiffusionActor):
            raise TypeError(
                f"DiffusionMILE requires a DiffusionActor, got {type(online_actor).__name__}."
            )

        self.device = next(online_actor.parameters()).device
        self.online_actor = online_actor.to(self.device)
        self.horizon = int(self.online_actor.horizon)
        self.prediction_type = self.online_actor.scheduler.config.prediction_type

        # A DP-pretrained actor is typically saved as the frozen EMA deployment copy
        # (requires_grad=False). Re-enable gradients so the online actor is actually trainable;
        # otherwise total_loss has no grad_fn and backward() fails. The EMA copy built below is
        # frozen explicitly, so it stays frozen regardless.
        for p in self.online_actor.parameters():
            p.requires_grad_(True)

        # EMA weights are optional; training always updates online_actor.
        self.use_ema = config.use_ema
        if self.use_ema:
            self.actor = copy.deepcopy(self.online_actor).to(self.device)
            for p in self.actor.parameters():
                p.requires_grad_(False)
            self.ema = EMAModel(self.actor, decay=config.ema_decay)
            self.component_names = ["actor", "online_actor", "actor_optimizer"]
        else:
            self.actor = self.online_actor
            self.ema = None
            self.component_names = ["actor", "actor_optimizer"]

        self.buffer = config.buffer
        self.rollout_policy: Optional[DiffusionActor] = None
        self._last_probit_diagnostics = {}  # CDF-input stats from the latest _compute_intervention_probs

        self.batch_size = config.batch_size
        self.learning_starts = config.learning_starts
        self.monte_carlo_samples = int(config.monte_carlo_samples)
        self.score_monte_carlo_samples = int(config.score_monte_carlo_samples)
        self.intervention_cost = float(config.intervention_cost)
        self.probit_scale = float(config.probit_scale)
        self.lambda_intervention = float(config.lambda_intervention)
        self.condition_intervention_on_action = bool(
            config.condition_intervention_on_action
        )
        self.reference_relative_score = bool(config.reference_relative_score)
        self.anchor_loss_weight = float(config.anchor_loss_weight)
        self.anchor_monte_carlo_samples = int(config.anchor_monte_carlo_samples)

        # Optional running-stats standardization of the probit score gap.
        self._init_score_gap_normalization()

        if self.anchor_monte_carlo_samples < 1:
            raise ValueError(
                f"anchor_monte_carlo_samples must be >= 1, got {self.anchor_monte_carlo_samples}"
            )

        # if not 0.0 <= self.lambda_intervention <= 1.0:
        #     raise ValueError(
        #         f"lambda_intervention must be in [0, 1], got {self.lambda_intervention}"
        #     )

        self.actor_optimizer = torch.optim.AdamW(
            self.online_actor.parameters(),
            lr=config.actor_optimizer_lr,
            betas=config.actor_optimizer_betas,
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        print("DiffusionMILE: actor=DiffusionActor")
        print(f"DiffusionMILE: horizon={self.horizon}, action_dim={self.online_actor.action_dim}")
        print(f"DiffusionMILE: prediction_type={self.prediction_type}")
        print(f"DiffusionMILE: monte_carlo_samples={self.monte_carlo_samples}")
        print(f"DiffusionMILE: score_monte_carlo_samples={self.score_monte_carlo_samples}")
        print(f"DiffusionMILE: probit_scale={self.probit_scale}")
        print(f"DiffusionMILE: intervention_cost={self.intervention_cost}")
        print(f"DiffusionMILE: lambda_intervention={self.lambda_intervention}")
        print(f"DiffusionMILE: condition_intervention_on_action={self.condition_intervention_on_action}")
        print(f"DiffusionMILE: reference_relative_score={self.reference_relative_score}")
        print(f"DiffusionMILE: anchor_loss_weight={self.anchor_loss_weight}")
        print(f"DiffusionMILE: anchor_monte_carlo_samples={self.anchor_monte_carlo_samples}")
        print(f"DiffusionMILE: normalize_score_gaps={self.normalize_score_gaps}")
        if self.normalize_score_gaps:
            print(
                f"DiffusionMILE: score_gap_ema_decay={self.score_gap_ema_decay}, "
                f"score_gap_std_min={self.score_gap_std_min}, score_gap_clip={self.score_gap_clip}"
            )
        print(f"DiffusionMILE: use_ema={self.use_ema}")

    def _init_score_gap_normalization(self):
        self.normalize_score_gaps = bool(self.config.normalize_score_gaps)
        self.score_gap_ema_decay = float(self.config.score_gap_ema_decay)
        self.score_gap_norm_eps = float(self.config.score_gap_norm_eps)
        self.score_gap_std_min = float(self.config.score_gap_std_min)
        self.score_gap_clip = float(self.config.score_gap_clip)

        self.score_gap_stats_initialized = False
        self.score_gap_mean = torch.zeros((), device=self.device)
        self.score_gap_var = torch.ones((), device=self.device)

    @torch.no_grad()
    def _update_score_gap_stats(self, gaps: torch.Tensor):
        if not self.normalize_score_gaps:
            return

        gaps = gaps.detach().reshape(-1)
        if gaps.numel() == 0:
            return

        batch_mean = gaps.mean()
        batch_var = gaps.var(unbiased=False).clamp_min(self.score_gap_norm_eps)

        if not self.score_gap_stats_initialized:
            self.score_gap_mean.copy_(batch_mean)
            self.score_gap_var.copy_(batch_var)
            self.score_gap_stats_initialized = True
        else:
            d = self.score_gap_ema_decay
            self.score_gap_mean.mul_(d).add_(batch_mean, alpha=1.0 - d)
            self.score_gap_var.mul_(d).add_(batch_var, alpha=1.0 - d)

    def _normalize_score_gap(self, gap: torch.Tensor) -> torch.Tensor:
        if not self.normalize_score_gaps:
            return gap

        std = torch.sqrt(self.score_gap_var.detach() + self.score_gap_norm_eps)
        std = std.clamp_min(self.score_gap_std_min)

        gap = (gap - self.score_gap_mean.detach()) / std

        if self.score_gap_clip > 0:
            gap = gap.clamp(-self.score_gap_clip, self.score_gap_clip)

        return gap

    def set_rollout_policy(self, rollout_policy: BaseActor):
        """Set the frozen rollout policy / mental model for MILE."""
        if not isinstance(rollout_policy, DiffusionActor):
            raise TypeError(
                f"DiffusionMILE requires a DiffusionActor rollout_policy, got {type(rollout_policy).__name__}."
            )
        self.rollout_policy = rollout_policy.to(self.device)
        self.rollout_policy.eval()
        for param in self.rollout_policy.parameters():
            param.requires_grad_(False)

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Coerce buffer actions into ``(B, horizon, action_dim)``."""
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        elif actions.dim() != 3:
            raise ValueError(f"Unexpected action shape for DiffusionMILE: {tuple(actions.shape)}")
        return actions.float()

    def _extract_interventions(self, batch: dict) -> torch.Tensor:
        """Return intervention labels ``[B]``: 0=no intervention, 1=intervention, 2=offline."""
        info = batch.get("info", None)

        for container in (info, batch):
            if isinstance(container, dict) and "intervention" in container:
                return torch.as_tensor(
                    container["intervention"], dtype=torch.float32, device=self.device
                ).view(-1)

        if isinstance(info, (list, tuple, np.ndarray)):
            try:
                values = [float(np.asarray(d["intervention"]).reshape(-1)[0]) for d in info]
            except (KeyError, TypeError, IndexError) as e:
                raise KeyError(
                    "DiffusionMILE expects an 'intervention' label in every transition's info dict "
                    "(0 = online no-intervention, 1 = online intervention, 2 = offline)."
                ) from e
            return torch.tensor(values, dtype=torch.float32, device=self.device).view(-1)

        raise KeyError(
            "DiffusionMILE could not find intervention labels. Expected batch['info'] to contain "
            "per-transition dictionaries with an 'intervention' key."
        )

    @staticmethod
    def _flatten_score_shape(scores: torch.Tensor) -> torch.Tensor:
        """Reduce any non-batch dimensions by mean, returning [B]."""
        if scores.dim() > 1:
            return scores.reshape(scores.shape[0], -1).mean(dim=-1)
        return scores.reshape(-1)

    def _sample_policy_actions(self, policy: DiffusionActor, obs: Union[dict, torch.Tensor], num_samples: int) -> torch.Tensor:
        """Sample ``num_samples`` action chunks from a diffusion policy.

        Uses :meth:`DiffusionActor.sample_actions_batch` to draw all ``num_samples`` chunks in a
        single batched reverse-diffusion pass (one obs encode + one denoising loop over a ``K*B``
        batch) instead of a Python loop, so the Monte Carlo samples are parallelized on the GPU.

        Returns:
            Tensor ``[K, B, horizon, action_dim]`` in normalized action space.
        """
        with torch.no_grad():
            return policy.sample_actions_batch(obs, int(num_samples))

    def _denoising_losses(
        self,
        actor: DiffusionActor,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample diffusion denoising losses for observed actions.

        Args:
            actor: DiffusionActor to score with.
            obs: batch observations.
            actions: ``[B, horizon, action_dim]`` normalized actions.

        Returns:
            Tensor ``[B]``. Lower is better.
        """
        actions = self._prepare_actions(actions)
        batch_size = actions.shape[0]

        losses = []
        global_cond = actor.encode_obs(obs)
        num_train_timesteps = int(actor.scheduler.config.num_train_timesteps)
        prediction_type = actor.scheduler.config.prediction_type

        for _ in range(self.score_monte_carlo_samples):
            noise = torch.randn_like(actions)
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (batch_size,),
                device=actions.device,
            ).long()
            noisy_actions = actor.scheduler.add_noise(actions, noise, timesteps)
            model_pred = actor.predict_noise(noisy_actions, timesteps, global_cond)
            target = noise if prediction_type == "epsilon" else actions
            per_element = (model_pred - target).pow(2)
            losses.append(self._flatten_score_shape(per_element))

        return torch.stack(losses, dim=0).mean(dim=0)

    def _action_scores(
        self,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Diffusion log-prob proxy ell(a,s), returning ``[B]``.

        Plain score:
            ell_theta(a,s) = - denoising_loss_theta(a,s)

        Reference-relative score, if enabled:
            ell_{theta,0}(a,s) = denoising_loss_0(a,s) - denoising_loss_theta(a,s)

        The reference-relative variant is positive when the online actor denoises the action
        better than the frozen rollout policy/reference.
        """
        theta_loss = self._denoising_losses(self.online_actor, obs, actions)
        if not self.reference_relative_score:
            return -theta_loss

        if self.rollout_policy is None:
            raise RuntimeError("reference_relative_score=True requires a frozen rollout_policy.")
        with torch.no_grad():
            ref_loss = self._denoising_losses(self.rollout_policy, obs, actions)
        return ref_loss - theta_loss

    def _score_stacked_actions(
        self,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Score stacked actions ``[K, B, horizon, action_dim]`` -> ``[K, B]``."""
        if actions.dim() != 4:
            raise ValueError(f"Expected stacked actions [K, B, H, A], got {tuple(actions.shape)}")
        scores = []
        for k in range(actions.shape[0]):
            scores.append(self._action_scores(obs, actions[k]))
        return torch.stack(scores, dim=0)

    def _compute_anchor_loss(
        self,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
        interventions: torch.Tensor,
    ) -> torch.Tensor:
        """Anchor online actor to frozen rollout policy on stored no-intervention rollout actions.

        Uses logged online no-intervention samples:

            intervention == 0

        where batch["action"] is assumed to be the rollout/robot action that the human accepted.

        The loss is:

            E_{(s,a0)~D_nu=0,t,eps}
                || model_theta(a0_t,t,s) - model_0(a0_t,t,s) ||^2

        This preserves accepted robot behavior while avoiding expensive fresh rollout-policy sampling.
        """
        if self.anchor_loss_weight <= 0.0:
            return torch.zeros((), device=self.device)

        if self.rollout_policy is None:
            raise RuntimeError("anchor_loss_weight > 0 requires a frozen rollout_policy.")

        actions = self._prepare_actions(actions)

        # Anchor only on online no-intervention samples.
        # These should correspond to rollout-policy actions accepted by the human.
        anchor_mask = interventions < 0.5

        if not anchor_mask.any():
            return torch.zeros((), device=self.device)

        selected_obs = self._index_obs(obs, anchor_mask)
        selected_actions = self._prepare_actions(actions[anchor_mask]).float()
        batch_size = selected_actions.shape[0]

        anchor_losses = []

        # Here anchor_monte_carlo_samples means number of (t, epsilon) noising samples
        # per anchored action. Default should usually be 1.
        for _ in range(self.anchor_monte_carlo_samples):
            noise = torch.randn_like(selected_actions)
            timesteps = torch.randint(
                0,
                int(self.online_actor.scheduler.config.num_train_timesteps),
                (batch_size,),
                device=self.device,
            ).long()

            noisy_actions = self.online_actor.scheduler.add_noise(
                selected_actions,
                noise,
                timesteps,
            )

            online_global_cond = self.online_actor.encode_obs(selected_obs)
            online_pred = self.online_actor.predict_noise(
                noisy_actions,
                timesteps,
                online_global_cond,
            )

            with torch.no_grad():
                ref_global_cond = self.rollout_policy.encode_obs(selected_obs)
                ref_pred = self.rollout_policy.predict_noise(
                    noisy_actions,
                    timesteps,
                    ref_global_cond,
                )

            anchor_losses.append(F.mse_loss(online_pred, ref_pred))

        return torch.stack(anchor_losses).mean()

    def _compute_intervention_probs(
        self,
        obs: Union[dict, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        interventions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MILE intervention probabilities for diffusion actors.

        No-intervention uses the marginal:
            p(nu=1|s) = E_{a~pi_theta} Phi(beta(ell(a,s) - E_{a0~pi0}ell(a0,s)) - c)

        If ``condition_intervention_on_action`` is true, online intervention samples use:
            p(nu=1|s,a_h) = Phi(beta(ell(a_h,s) - E_{a0~pi0}ell(a0,s)) - c)
        """
        if self.rollout_policy is None:
            raise RuntimeError(
                "DiffusionMILE requires a frozen rollout_policy. Call set_rollout_policy() before training."
            )

        # Robot/mental-model expectation under frozen rollout policy.
        rollout_samples = self._sample_policy_actions(
            self.rollout_policy,
            obs,
            self.monte_carlo_samples,
        )  # [K, B, H, A]
        rollout_scores = self._score_stacked_actions(obs, rollout_samples)  # [K, B]
        expected_rollout_score = rollout_scores.mean(dim=0)  # [B]

        # Original MILE marginal over the trainable policy.
        # The expectation is over the trainable policy.  Even when EMA is enabled, use
        # online_actor here; EMA is only for deployment/evaluation.  sample_actions() itself
        # is no_grad in DiffusionActor, so gradients flow only through the score evaluation below.
        training_samples = self._sample_policy_actions(
            self.online_actor,
            obs,
            self.monte_carlo_samples,
        )
        training_scores = self._score_stacked_actions(obs, training_samples)  # [K, B]

        probit_diff_marginal = training_scores - expected_rollout_score.unsqueeze(0)  # [K, B]

        # Optionally standardize the score gap by its EMA running mean/std before the probit, so
        # probit_scale / intervention_cost are decoupled from the absolute (drifting) scale of the
        # denoising score. Stats are updated from (and only from) this marginal gap; the affine
        # transform keeps gradients flowing through the numerator (mean/std are detached).
        self._update_score_gap_stats(probit_diff_marginal)
        normalized_diff_marginal = self._normalize_score_gap(probit_diff_marginal)
        probit_arg_marginal = self.probit_scale * normalized_diff_marginal - self.intervention_cost

        # Record CDF-input diagnostics: the RAW (pre-normalization) gap vs the argument actually fed
        # to Φ, so probit_scale / intervention_cost can be tuned to keep the argument within ~[-3, 3].
        self._last_probit_diagnostics = probit_cdf_diagnostics(probit_diff_marginal, probit_arg_marginal)

        intervention_probs = normal_cdf(probit_arg_marginal).mean(dim=0)  # [B]

        if self.condition_intervention_on_action:
            if actions is None or interventions is None:
                raise ValueError(
                    "condition_intervention_on_action=True requires actions and interventions."
                )
            actions = self._prepare_actions(actions)
            observed_intervention_mask = (interventions > 0.5) & (interventions < 1.5)

            if observed_intervention_mask.any():
                observed_scores = self._action_scores(obs, actions)  # [B]
                # Standardize with the SAME running stats as the marginal (do not update them from
                # the observed gap) so both probit arguments live on the same scale.
                normalized_observed_gap = self._normalize_score_gap(observed_scores - expected_rollout_score)
                probit_arg_observed = self.probit_scale * normalized_observed_gap - self.intervention_cost
                observed_probs = normal_cdf(probit_arg_observed)
                intervention_probs = intervention_probs.clone()
                intervention_probs[observed_intervention_mask] = observed_probs[observed_intervention_mask]

        return intervention_probs.clamp(EPS, 1.0 - EPS)

    def _compute_actor_bc_loss(
        self,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diffusion behavior-cloning loss on intervention/offline actions.

        Returns:
            (loss, model_pred_mean)
        """
        if not mask.any():
            return torch.zeros((), device=self.device), torch.zeros((), device=self.device)

        selected_obs = self._index_obs(obs, mask)
        selected_actions = self._prepare_actions(actions[mask]).float()
        batch_size = selected_actions.shape[0]

        noise = torch.randn_like(selected_actions)
        timesteps = torch.randint(
            0,
            self.online_actor.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        noisy_actions = self.online_actor.scheduler.add_noise(selected_actions, noise, timesteps)
        global_cond = self.online_actor.encode_obs(selected_obs)
        model_pred = self.online_actor.predict_noise(noisy_actions, timesteps, global_cond)
        target = noise if self.prediction_type == "epsilon" else selected_actions
        return F.mse_loss(model_pred, target), model_pred.mean()

    @staticmethod
    def _index_obs(obs: Union[dict, torch.Tensor], mask: torch.Tensor) -> Union[dict, torch.Tensor]:
        if isinstance(obs, dict):
            return {k: v[mask] for k, v in obs.items()}
        return obs[mask]

    def train_step(self, logging_prefix: str = "diffusion_mile", rollout_step: int = None) -> dict:
        """Perform one training step of diffusion/generative MILE."""
        if rollout_step is not None and rollout_step < self.learning_starts:
            return {}
        if self.rollout_policy is None:
            raise RuntimeError(
                "DiffusionMILE requires a frozen rollout_policy. Call set_rollout_policy() before training."
            )

        actor_losses = []
        intervention_losses = []
        anchor_losses = []
        total_losses = []
        probit_diagnostics = []  # per-step CDF-input stats for probit_scale / intervention_cost tuning
        expert_action_means = []
        sample_mse_errors = []
        unnormalized_sample_mse_errors = []
        sample_mse_human_errors = []  # sampled vs human (label-1) actions only
        sample_mse_robot_errors = []  # sampled vs robot (label-0) actions only
        score_human_means = []
        score_rollout_means = []
        noise_pred_means = []
        # Per-class intervention probability, to detect a degenerate (base-rate-collapsed) probit:
        # a healthy probit has human(label 1) >> policy(label 0); if both sit near the dataset's
        # intervention rate, it is just predicting the prior and provides no per-state signal.
        iprob_policy_means = []  # mean p(nu=1|s) on label-0 (autonomous/policy) states
        iprob_human_means = []   # mean p(nu=1|s) on label-1 (human-correction) states

        gradient_steps = int(self.config.num_updates_per_train_step)

        for _ in range(gradient_steps):
            batch = self.buffer.sample(self.batch_size, device=self.device)
            if not batch:
                print("Buffer is still empty. Skipping this training step")
                return {}

            obs = batch["obs"]
            actions = self._prepare_actions(batch["action"])
            if len(actions) == 0:
                print("Buffer is still empty. Skipping this training step")
                return {}

            interventions = self._extract_interventions(batch)
            action_mask = interventions > 0.5  # {1, 2}: human corrections + offline demos
            online_mask = interventions < 1.5  # {0, 1}: valid intervention labels

            if self.config.obs_noise_std > 0:
                if isinstance(obs, dict):
                    obs = {k: v + torch.randn_like(v) * self.config.obs_noise_std for k, v in obs.items()}
                else:
                    obs = obs + torch.randn_like(obs) * self.config.obs_noise_std

            if self.config.action_noise_std > 0:
                actions = actions.clone()
                if action_mask.any():
                    actions[action_mask] = actions[action_mask] + torch.randn_like(actions[action_mask]) * self.config.action_noise_std

            intervention_probs = self._compute_intervention_probs(
                obs=obs,
                actions=actions,
                interventions=interventions,
            ).view_as(interventions)
            if self._last_probit_diagnostics:
                probit_diagnostics.append(self._last_probit_diagnostics)

            if online_mask.any():
                intervention_loss = F.binary_cross_entropy(
                    intervention_probs[online_mask],
                    interventions[online_mask],
                )
            else:
                intervention_loss = torch.zeros((), device=self.device)

            actor_loss, noise_pred_mean = self._compute_actor_bc_loss(obs, actions, action_mask)
            anchor_loss = self._compute_anchor_loss(
                obs=obs,
                actions=actions,
                interventions=interventions,
            )
            total_loss = (
                actor_loss
                + self.lambda_intervention * intervention_loss
                + self.anchor_loss_weight * anchor_loss
            )

            self.actor_optimizer.zero_grad()
            total_loss.backward()
            if self.config.clip_grad_norm and self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.online_actor.parameters(),
                    max_norm=self.config.clip_grad_norm,
                )
            self.actor_optimizer.step()

            if self.use_ema:
                self.ema.step(self.online_actor)

            self.step_counter += 1

            actor_losses.append(actor_loss.item())
            intervention_losses.append(intervention_loss.item())
            anchor_losses.append(anchor_loss.item())
            total_losses.append(total_loss.item())
            noise_pred_means.append(noise_pred_mean.item() if isinstance(noise_pred_mean, torch.Tensor) else float(noise_pred_mean))

            # Per-class intervention prob (label 0 = policy, label 1 = human). NaN when a class is
            # absent from this batch; nan-aggregated below so it survives sparse batches.
            policy_mask = interventions < 0.5
            human_mask = (interventions > 0.5) & (interventions < 1.5)
            iprob = intervention_probs.detach()
            iprob_policy_means.append(iprob[policy_mask].mean().item() if policy_mask.any() else np.nan)
            iprob_human_means.append(iprob[human_mask].mean().item() if human_mask.any() else np.nan)

            if action_mask.any():
                selected_actions = actions[action_mask]
                expert_action_means.append(selected_actions.mean().item())
                with torch.no_grad():
                    score_human_means.append(self._action_scores(self._index_obs(obs, action_mask), selected_actions).mean().item())
            else:
                expert_action_means.append(np.nan)
                score_human_means.append(np.nan)

            with torch.no_grad():
                rollout_samples = self._sample_policy_actions(self.rollout_policy, obs, 1)[0]
                score_rollout_means.append(self._action_scores(obs, rollout_samples).mean().item())

                if (
                    self.config.log_sample_metrics_every > 0
                    and self.step_counter % self.config.log_sample_metrics_every == 0
                ):
                    sampled = self.actor.sample_actions(obs)
                    sampled = self._prepare_actions(sampled)
                    if sampled.shape != actions.shape:
                        sampled = sampled.reshape(actions.shape)
                    sample_mse_errors.append(F.mse_loss(sampled, actions).item())
                    unnormalized_sample_mse_errors.append(
                        F.mse_loss(
                            self.actor.unnormalize_action(sampled),
                            self.actor.unnormalize_action(actions),
                        ).item()
                    )
                    # Per-class sample MSE: human (label-1) measures correction-fitting (should drop);
                    # robot (label-0) measures drift from the rollout policy (should stay small). The
                    # overall metric above averages both, so it can hide a real drop on the human subset.
                    per_sample_mse = (sampled - actions).pow(2).reshape(sampled.shape[0], -1).mean(dim=1)
                    if human_mask.any():
                        sample_mse_human_errors.append(per_sample_mse[human_mask].mean().item())
                    if policy_mask.any():
                        sample_mse_robot_errors.append(per_sample_mse[policy_mask].mean().item())

        metrics_dict = {
            "total_loss": float(np.mean(total_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "intervention_loss": float(np.mean(intervention_losses)),
            "anchor_loss": float(np.mean(anchor_losses)),
            "expert_action_mean": float(np.nanmean(expert_action_means)),
            "noise_pred_mean": float(np.mean(noise_pred_means)),
            "intervention_prob_mean": float(intervention_probs[online_mask].mean().item()) if online_mask.any() else float("nan"),
            # Per-class probit outputs: healthy => human_mean >> policy_mean; collapsed (base-rate)
            # => both ~ dataset intervention rate. The gap is the discrimination signal.
            "intervention_prob_policy_mean": float(np.nanmean(iprob_policy_means)) if np.any(~np.isnan(iprob_policy_means)) else float("nan"),
            "intervention_prob_human_mean": float(np.nanmean(iprob_human_means)) if np.any(~np.isnan(iprob_human_means)) else float("nan"),
            "human_score_mean": float(np.nanmean(score_human_means)),
            "rollout_score_mean": float(np.nanmean(score_rollout_means)),
        }
        if sample_mse_errors:
            metrics_dict["sample_mse_error"] = float(np.mean(sample_mse_errors))
            metrics_dict["unnormalized_sample_mse_error"] = float(np.mean(unnormalized_sample_mse_errors))
            metrics_dict["sample_mse_error_human"] = float(np.mean(sample_mse_human_errors)) if sample_mse_human_errors else float("nan")
            metrics_dict["sample_mse_error_robot"] = float(np.mean(sample_mse_robot_errors)) if sample_mse_robot_errors else float("nan")

        # Probit CDF-input diagnostics: probit_diff_raw_* (unnormalized denoising-score gap) and
        # probit_cdf_arg_* (the value fed to Φ); keep cdf_arg within ~[-3, 3] when tuning.
        metrics_dict.update(aggregate_probit_diagnostics(probit_diagnostics))

        # Running score-gap normalization stats (only meaningful when normalize_score_gaps=True).
        if self.normalize_score_gaps and self.score_gap_stats_initialized:
            metrics_dict["score_gap_running_mean"] = float(self.score_gap_mean.item())
            metrics_dict["score_gap_running_std"] = float(
                torch.sqrt(self.score_gap_var + self.score_gap_norm_eps).item()
            )

        if self.logger is not None:
            self.logger.log(metrics_dict, step=self.step_counter, prefix=logging_prefix)

        return metrics_dict


# =====================================================================================
# Simulated-human intervention criteria (denoising-score analogues of the MILE ones)
# =====================================================================================
# These mirror the Gaussian helpers in ``algorithms/mile/modeling_mile.py`` but score actions with
# the diffusion denoising proxy instead of explicit log-probabilities, so they work with
# DiffusionActor policies. They are inference-only (no grad) and plug into the HITL collector via
# ``hitl_utils.get_intervention_criteria`` (names "diffusion_mile" / "diffusion_mile_window").


@torch.no_grad()
def _denoising_score_under(
    policy: DiffusionActor,
    obs: Union[dict, torch.Tensor],
    actions: torch.Tensor,
    score_monte_carlo_samples: int = 1,
) -> torch.Tensor:
    """Denoising-score proxy ``ell(a, s) = -E_{t,eps}|| target - model_policy(noise(a,t,eps),t,s) ||^2``.

    Inference-only counterpart of :meth:`DiffusionMILE._denoising_losses`, returning the NEGATIVE
    loss so that higher => the action is more likely under ``policy``. ``actions`` may be ``[B, A]``
    (single step) or ``[B, H, A]``; a single-step action is tiled across the policy's horizon so the
    conditional network always sees a full-length chunk. Returns ``[B]``.
    """
    if actions.dim() == 2:
        actions = actions.unsqueeze(1)
    if actions.shape[1] == 1 and policy.horizon > 1:
        actions = actions.repeat(1, policy.horizon, 1)
    actions = actions.float()
    batch_size = actions.shape[0]

    global_cond = policy.encode_obs(obs)
    num_train_timesteps = int(policy.scheduler.config.num_train_timesteps)
    prediction_type = policy.scheduler.config.prediction_type

    losses = []
    for _ in range(int(score_monte_carlo_samples)):
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=actions.device).long()
        noisy_actions = policy.scheduler.add_noise(actions, noise, timesteps)
        model_pred = policy.predict_noise(noisy_actions, timesteps, global_cond)
        target = noise if prediction_type == "epsilon" else actions
        per_element = (model_pred - target).pow(2)
        losses.append(per_element.reshape(batch_size, -1).mean(dim=-1))
    return -torch.stack(losses, dim=0).mean(dim=0)


@torch.no_grad()
def diffusion_mile_intervention_prob(
    judge_policy: DiffusionActor,
    other_policy: DiffusionActor,
    obs: Union[dict, torch.Tensor],
    num_samples: int = 8,
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
) -> torch.Tensor:
    """Denoising-score analogue of :func:`...mile.modeling_mile.mile_intervention_prob`.

        p = E_{a_j~judge}[ Phi( beta * (ell_judge(a_j, s) - E_{a_o~other}[ell_judge(a_o, s)]) - c ) ]

    where ``ell_judge`` is the judge's denoising-score proxy (higher => more likely under the judge).
    It is high when the *other* policy's actions denoise poorly under the judge (i.e. the judge finds
    them unlikely). For a simulated human, pass ``judge=expert_policy`` and ``other=rollout_policy``.
    Both policies must be DiffusionActors. Returns a tensor ``[B]`` (B inferred from ``obs``).

    NOTE: each sample is a full reverse-diffusion rollout, so this costs ``2 * num_samples`` action
    samplings per call. Keep ``num_samples`` modest (and prefer a DDIM actor with few inference
    steps) for interactive simulated rollouts.
    """
    if not isinstance(judge_policy, DiffusionActor) or not isinstance(other_policy, DiffusionActor):
        raise TypeError(
            "diffusion_mile_intervention_prob requires DiffusionActor judge/other policies, got "
            f"judge={type(judge_policy).__name__}, other={type(other_policy).__name__}."
        )

    # Judge scores the OTHER policy's samples (the expectation/baseline term).
    other_scores = [
        _denoising_score_under(judge_policy, obs, other_policy.sample_actions(obs), score_monte_carlo_samples)
        for _ in range(num_samples)
    ]
    expected_score_other = torch.stack(other_scores, dim=0).mean(dim=0)  # [B]

    # Judge scores its OWN samples.
    self_scores = [
        _denoising_score_under(judge_policy, obs, judge_policy.sample_actions(obs), score_monte_carlo_samples)
        for _ in range(num_samples)
    ]
    score_j_self = torch.stack(self_scores, dim=0)  # [K, B]

    probit_arg = probit_scale * (score_j_self - expected_score_other.unsqueeze(0)) - intervention_cost
    return normal_cdf(probit_arg).mean(dim=0).clamp(EPS, 1.0 - EPS)


def make_diffusion_mile_intervention_criteria(
    num_samples: int = 8,
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
    stochastic: bool = True,
    threshold: float = 0.5,
    **_,
):
    """Build a simulated-human criterion from DiffusionMILE's denoising-score probit.

    Mirrors :func:`...mile.modeling_mile.make_mile_intervention_criteria` but scores actions with the
    diffusion denoising proxy instead of Gaussian log-probs. Each step the expert (judge) computes
    ``p = diffusion_mile_intervention_prob(expert, rollout, obs)``; it intervenes with
    ``Bernoulli(p)`` (``stochastic``) or when ``p > threshold``. On intervention the expert action is
    executed, otherwise the rollout action. Both the expert and the rollout policy must be
    DiffusionActors. Returns a function with the HITL criterion signature.
    """

    def criteria(obs, rollout_policy, expert_policy, rollout_action, expert_action,
                 rollout_chunk=None, expert_chunk=None):
        p = float(
            diffusion_mile_intervention_prob(
                expert_policy,
                rollout_policy,
                obs,
                num_samples=num_samples,
                score_monte_carlo_samples=score_monte_carlo_samples,
                intervention_cost=intervention_cost,
                probit_scale=probit_scale,
            ).reshape(-1)[0]
        )
        intervene = bool(np.random.random() < p) if stochastic else bool(p > threshold)
        return intervene, (expert_action if intervene else rollout_action)

    return criteria


@torch.no_grad()
def _action_score_under(
    policy: DiffusionActor,
    obs: Union[dict, torch.Tensor],
    env_action,
    score_monte_carlo_samples: int = 1,
) -> float:
    """Denoising score ``ell(a, s)`` for a single env-space action under a DiffusionActor (scalar).

    Maps the env-space action into the policy's normalized [-1, 1] space, tiles it across the
    policy's horizon (handled by :func:`_denoising_score_under`), and returns the per-state score
    for the first env in the batch.
    """
    device = next(policy.parameters()).device
    a_norm = policy.normalize_action(torch.as_tensor(np.asarray(env_action, dtype=np.float32), device=device))
    a_norm = a_norm.reshape(1, 1, -1)  # [B=1, H=1, A]
    score = _denoising_score_under(policy, obs, a_norm, score_monte_carlo_samples)
    return float(score.reshape(-1)[0])


@torch.no_grad()
def _chunk_score_under(
    policy: DiffusionActor,
    obs: Union[dict, torch.Tensor],
    env_chunk,
    score_monte_carlo_samples: int = 1,
) -> float:
    """Denoising score ``ell(a, s)`` for a full env-space action *chunk* under a DiffusionActor.

    Maps the env-space chunk ``[L, action_dim]`` into the policy's normalized [-1, 1] space and
    scores it as an in-distribution ``[1, horizon, action_dim]`` sequence (no per-action tiling).
    When the chunk length ``L`` differs from the scoring policy's horizon (e.g. the rollout and
    expert policies use different chunk sizes), the chunk is conformed to the policy horizon by
    truncation (``L`` too long) or last-step repeat-padding (``L`` too short). Returns the per-state
    score for the first env in the batch.
    """
    device = next(policy.parameters()).device
    chunk = torch.as_tensor(np.asarray(env_chunk, dtype=np.float32), device=device)
    if chunk.dim() == 1:
        chunk = chunk.unsqueeze(0)  # [A] -> [1, A]
    a_norm = policy.normalize_action(chunk)  # [L, A] in [-1, 1]

    horizon = int(policy.horizon)
    length = a_norm.shape[0]
    if length > horizon:
        a_norm = a_norm[:horizon]
    elif length < horizon:
        a_norm = torch.cat([a_norm, a_norm[-1:].expand(horizon - length, -1)], dim=0)

    a_norm = a_norm.reshape(1, horizon, -1)
    return float(_denoising_score_under(policy, obs, a_norm, score_monte_carlo_samples).reshape(-1)[0])


def make_diffusion_mile_paired_criteria(
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
    stochastic: bool = True,
    threshold: float = 0.5,
    **_,
):
    """Sampling-free DiffusionMILE criterion that scores the proposed human vs robot action.

    Unlike :func:`make_diffusion_mile_intervention_criteria` (which Monte-Carlo marginalizes over
    action chunks sampled from BOTH policies via expensive reverse diffusion), this directly scores
    the two *already-proposed* env actions under the expert's denoising model:

        p = Phi( probit_scale * (ell_expert(a_human) - ell_expert(a_robot)) - intervention_cost )

    i.e. it intervenes when the robot's actual action denoises worse under the expert than the
    human's own action would. This models a human with a **perfect model of the robot** — it judges
    the robot's specific action at this state rather than marginalizing over what the robot *might*
    do. It costs ~2 denoising-score evals per step (no ``sample_actions`` calls), so it is far
    faster than the marginal criterion and well suited to interactive simulated rollouts.

    ``score_monte_carlo_samples`` still controls the diffusion-noising draws used to estimate each
    ``ell(a, s)`` (set it to 1 for the cheapest, noisiest score). Both the expert and rollout policy
    must be DiffusionActors. Returns a function with the HITL criterion signature.

    When the rollout worker supplies the full predicted ``rollout_chunk`` / ``expert_chunk`` (env
    space, ``[L, action_dim]``), the in-distribution chunks are scored directly under the expert
    denoiser; otherwise it falls back to scoring the single proposed actions tiled across the
    horizon (exact only at ``horizon == 1``).
    """

    def criteria(obs, rollout_policy, expert_policy, rollout_action, expert_action,
                 rollout_chunk=None, expert_chunk=None):
        if rollout_chunk is not None and expert_chunk is not None:
            # Score the actual predicted chunks (in-distribution) under the expert's denoiser.
            s_human = _chunk_score_under(expert_policy, obs, expert_chunk, score_monte_carlo_samples)
            s_robot = _chunk_score_under(expert_policy, obs, rollout_chunk, score_monte_carlo_samples)
        else:
            # Fallback: score the single proposed actions (tiled across the horizon).
            s_human = _action_score_under(expert_policy, obs, expert_action, score_monte_carlo_samples)
            s_robot = _action_score_under(expert_policy, obs, rollout_action, score_monte_carlo_samples)
        arg = probit_scale * (s_human - s_robot) - intervention_cost
        p = float(normal_cdf(torch.tensor(arg, dtype=torch.float32)))
        intervene = bool(np.random.random() < p) if stochastic else bool(p > threshold)
        return intervene, (expert_action if intervene else rollout_action)

    return criteria


def make_diffusion_mile_window_criteria(
    window: int = 10,
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
    stochastic: bool = True,
    threshold: float = 0.5,
    **_,
):
    """Windowed DiffusionMILE criterion: an EMA over the per-step expert-vs-robot score gap.

    Mirrors :func:`...mile.modeling_mile.make_mile_window_criteria` with denoising scores. Per step
    ``g_t = ell_expert(a_expert_t) - ell_expert(a_robot_t) - c`` (>= 0 when the robot's *executed*
    action denoises worse under the expert than the expert's own action). An EMA with decay
    ``alpha = 2/(window+1)`` smooths ``g_t`` so intervention reflects *sustained* divergence rather
    than a single-step blip; ``p = Phi(probit_scale * ema)``, then ``Bernoulli(p)`` (stochastic) or
    ``p > threshold``. The single executed action is tiled across the expert's horizon for scoring.

    Stateful (EMA per episode); exposes ``.reset()`` (called by the rollout worker each episode).
    Much cheaper than the non-windowed criterion (no reverse-diffusion sampling — it scores the two
    already-proposed actions directly).
    """
    alpha = 2.0 / (float(max(1, window)) + 1.0)
    state = {"ema": None}

    def criteria(obs, rollout_policy, expert_policy, rollout_action, expert_action,
                 rollout_chunk=None, expert_chunk=None):
        s_expert = _action_score_under(expert_policy, obs, expert_action, score_monte_carlo_samples)
        s_robot = _action_score_under(expert_policy, obs, rollout_action, score_monte_carlo_samples)
        g = (s_expert - s_robot) - intervention_cost
        state["ema"] = g if state["ema"] is None else alpha * g + (1.0 - alpha) * state["ema"]
        p = float(normal_cdf(torch.tensor(probit_scale * state["ema"], dtype=torch.float32)))
        intervene = bool(np.random.random() < p) if stochastic else bool(p > threshold)
        return intervene, (expert_action if intervene else rollout_action)

    def reset():
        state["ema"] = None

    criteria.reset = reset
    return criteria