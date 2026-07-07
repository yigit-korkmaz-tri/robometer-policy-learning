"""Flow MILE for flow-matching-policy actors.

This module implements a flow-matching analogue of MILE. It mirrors the diffusion-policy
``modeling_diffusion_mile.py`` implementation but scores actions with the conditional
flow-matching objective instead of the diffusion denoising objective as a log-probability
proxy:

    ell_theta(a, s) = - E_{t, x0} || u_target - v_theta(x_t, t, s) ||^2

where ``x0 ~ N(0, I)``, ``t ~ U(0, 1)``, the OT interpolant is
``x_t = (1 - (1 - sigma_min) t) x0 + t a``, and the target velocity is
``u_target = a - (1 - sigma_min) x0``.

For online intervention samples, an optional flag uses p(nu=1 | s, a_h) with the observed
human action. For online no-intervention samples, the human action is latent, so it uses
MILE's marginal p(nu=1 | s), estimated by Monte Carlo samples from the trainable policy.
"""

from __future__ import annotations

import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.algorithms.flow_mile.configuration_flow_mile import FlowMILEConfig
from robometer_policy_learning.algorithms.dp.modeling_dp import EMAModel
from robometer_policy_learning.algorithms.flow_matching.modeling_flow import FlowMatchingActor
from robometer_policy_learning.modules.base import BaseActor
from robometer_policy_learning.algorithms.mile.modeling_mile import (
    aggregate_probit_diagnostics,
    normal_cdf,
    probit_cdf_diagnostics,
)


EPS = 1e-4


def _flow_matching_target(
    actor: FlowMatchingActor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw one conditional flow-matching sample for ``actions`` ``[B, H, A]``.

    Returns ``(x_t, t, target)`` where ``x0 ~ N(0, I)``, ``t ~ U(0, 1)`` per-sample,
    ``x_t = (1 - (1 - sigma_min) t) x0 + t * actions`` is the OT interpolant, and
    ``target = actions - (1 - sigma_min) x0`` is the (constant) target velocity. ``t`` is
    returned as ``[B]`` for the network's time embedding.
    """
    sigma_min = float(actor.sigma_min)
    batch_size = actions.shape[0]
    x0 = torch.randn_like(actions)
    t = torch.rand(batch_size, device=actions.device)
    t_expand = t.view(batch_size, *([1] * (actions.dim() - 1)))
    x_t = (1.0 - (1.0 - sigma_min) * t_expand) * x0 + t_expand * actions
    target = actions - (1.0 - sigma_min) * x0
    return x_t, t, target


class FlowMILE(BaseAlgorithm):
    """MILE for FlowMatchingActor policies using flow-matching-loss proxies.

    Required config fields are intentionally aligned with the existing MILE / DiffusionMILE
    configs:

    * actor: FlowMatchingActor
    * buffer, batch_size, learning_starts, num_updates_per_train_step
    * monte_carlo_samples, intervention_cost, probit_scale, lambda_intervention
    * condition_intervention_on_action, actor_optimizer_lr
    * actor_optimizer_eps, actor_optimizer_weight_decay

    Optional fields:

    * score_monte_carlo_samples: number of flow-matching samples used to estimate ell(a,s)
    * reference_relative_score: use ell_{theta,0} = loss_0 - loss_theta instead of -loss_theta
    * anchor_loss_weight: coefficient for matching the online velocity field to the frozen rollout
      velocity field on the buffer's stored non-intervention (label-0) actions
    * anchor_monte_carlo_samples: number of flow-matching (t, x0) noise draws averaged per anchor
      action when matching velocity fields
    * normalize_score_gaps: standardize the probit score gap by an EMA running mean/std before
      applying probit_scale / intervention_cost (with score_gap_ema_decay, score_gap_norm_eps,
      score_gap_std_min, score_gap_clip)
    * use_ema, ema_decay: maintain an EMA deployment actor, as in Flow Matching
    * actor_optimizer_betas: AdamW betas
    * obs_noise_std, action_noise_std, clip_grad_norm
    """

    def __init__(self, config: FlowMILEConfig):
        super().__init__(config)
        self.config = config

        online_actor = config.actor
        if online_actor is None:
            raise ValueError("A FlowMatchingActor is required for FlowMILE")
        if not isinstance(online_actor, FlowMatchingActor):
            raise TypeError(
                f"FlowMILE requires a FlowMatchingActor, got {type(online_actor).__name__}."
            )

        self.device = next(online_actor.parameters()).device
        self.online_actor = online_actor.to(self.device)
        self.horizon = int(self.online_actor.horizon)
        self.sigma_min = float(self.online_actor.sigma_min)

        # A flow-pretrained actor is typically saved as the frozen EMA deployment copy
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
        self.rollout_policy: Optional[FlowMatchingActor] = None
        self._last_probit_diagnostics = {}  # CDF-input stats from the latest _compute_intervention_probs
        self._last_score_diagnostics = {}   # raw score means (rollout baseline / observed human) from it

        self.batch_size = config.batch_size
        self.learning_starts = config.learning_starts
        self.monte_carlo_samples = int(config.monte_carlo_samples)
        # Euler steps for MC action sampling; None => use the actor's own num_inference_steps.
        self.mc_num_inference_steps = (
            int(config.mc_num_inference_steps) if config.mc_num_inference_steps is not None else None
        )
        self.score_monte_carlo_samples = int(config.score_monte_carlo_samples)
        self.intervention_cost = float(config.intervention_cost)
        self.probit_scale = float(config.probit_scale)
        self.lambda_intervention = float(config.lambda_intervention)
        self.condition_intervention_on_action = bool(
            config.condition_intervention_on_action
        )
        self.condition_nonintervention_on_robot = bool(
            config.condition_nonintervention_on_robot
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

        print("FlowMILE: actor=FlowMatchingActor")
        print(f"FlowMILE: horizon={self.horizon}, action_dim={self.online_actor.action_dim}")
        print(f"FlowMILE: sigma_min={self.sigma_min}")
        print(f"FlowMILE: monte_carlo_samples={self.monte_carlo_samples}")
        print(f"FlowMILE: mc_num_inference_steps={self.mc_num_inference_steps} (None => actor default)")
        print(f"FlowMILE: score_monte_carlo_samples={self.score_monte_carlo_samples}")
        print(f"FlowMILE: probit_scale={self.probit_scale}")
        print(f"FlowMILE: intervention_cost={self.intervention_cost}")
        print(f"FlowMILE: lambda_intervention={self.lambda_intervention}")
        print(f"FlowMILE: condition_intervention_on_action={self.condition_intervention_on_action}")
        print(f"FlowMILE: condition_nonintervention_on_robot={self.condition_nonintervention_on_robot}")
        print(f"FlowMILE: reference_relative_score={self.reference_relative_score}")
        print(f"FlowMILE: anchor_loss_weight={self.anchor_loss_weight}")
        print(f"FlowMILE: anchor_monte_carlo_samples={self.anchor_monte_carlo_samples}")
        print(f"FlowMILE: normalize_score_gaps={self.normalize_score_gaps}")
        if self.normalize_score_gaps:
            print(
                f"FlowMILE: score_gap_ema_decay={self.score_gap_ema_decay}, "
                f"score_gap_std_min={self.score_gap_std_min}, score_gap_clip={self.score_gap_clip}"
            )
        print(f"FlowMILE: use_ema={self.use_ema}")

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
        if not isinstance(rollout_policy, FlowMatchingActor):
            raise TypeError(
                f"FlowMILE requires a FlowMatchingActor rollout_policy, got {type(rollout_policy).__name__}."
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
            raise ValueError(f"Unexpected action shape for FlowMILE: {tuple(actions.shape)}")
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
                    "FlowMILE expects an 'intervention' label in every transition's info dict "
                    "(0 = online no-intervention, 1 = online intervention, 2 = offline)."
                ) from e
            return torch.tensor(values, dtype=torch.float32, device=self.device).view(-1)

        raise KeyError(
            "FlowMILE could not find intervention labels. Expected batch['info'] to contain "
            "per-transition dictionaries with an 'intervention' key."
        )

    @staticmethod
    def _flatten_score_shape(scores: torch.Tensor) -> torch.Tensor:
        """Reduce any non-batch dimensions by mean, returning [B]."""
        if scores.dim() > 1:
            return scores.reshape(scores.shape[0], -1).mean(dim=-1)
        return scores.reshape(-1)

    def _sample_policy_actions(self, policy: FlowMatchingActor, obs: Union[dict, torch.Tensor], num_samples: int) -> torch.Tensor:
        """Sample ``num_samples`` action chunks from a flow-matching policy.

        Uses :meth:`FlowMatchingActor.sample_actions_batch` to draw all ``num_samples`` chunks in a
        single batched Euler-integration pass (one obs encode + one ODE loop over a ``K*B`` batch)
        instead of a Python loop, so the Monte Carlo samples are parallelized on the GPU. The Euler
        step count is ``mc_num_inference_steps`` when set, else the actor's own num_inference_steps.

        Returns:
            Tensor ``[K, B, horizon, action_dim]`` in normalized action space.
        """
        with torch.no_grad():
            return policy.sample_actions_batch(
                obs, int(num_samples), num_inference_steps=self.mc_num_inference_steps
            )

    def _flow_losses_with_cond(
        self,
        actor: FlowMatchingActor,
        global_cond: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row flow-matching loss using a PRECOMPUTED global conditioning.

        ``actions`` ``[N, horizon, action_dim]`` and ``global_cond`` ``[N, cond]`` share the leading
        batch dim ``N`` (``N`` may be ``K*B`` for stacked scoring). The obs encode is done by the
        caller so it is not repeated per Monte Carlo sample / per stacked sample. Returns ``[N]``.
        """
        actions = self._prepare_actions(actions)

        losses = []
        for _ in range(self.score_monte_carlo_samples):
            x_t, t, target = _flow_matching_target(actor, actions)
            v_pred = actor.predict_velocity(x_t, t, global_cond)
            per_element = (v_pred - target).pow(2)
            losses.append(self._flatten_score_shape(per_element))

        return torch.stack(losses, dim=0).mean(dim=0)

    def _flow_losses(
        self,
        actor: FlowMatchingActor,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample flow-matching losses for observed actions.

        Args:
            actor: FlowMatchingActor to score with.
            obs: batch observations.
            actions: ``[B, horizon, action_dim]`` normalized actions.

        Returns:
            Tensor ``[B]``. Lower is better.
        """
        return self._flow_losses_with_cond(actor, actor.encode_obs(obs), actions)

    def _action_scores(
        self,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-matching log-prob proxy ell(a,s), returning ``[B]``.

        Plain score:
            ell_theta(a,s) = - flow_loss_theta(a,s)

        Reference-relative score, if enabled:
            ell_{theta,0}(a,s) = flow_loss_0(a,s) - flow_loss_theta(a,s)

        The reference-relative variant is positive when the online actor's velocity field fits the
        action better than the frozen rollout policy/reference.
        """
        theta_loss = self._flow_losses(self.online_actor, obs, actions)
        if not self.reference_relative_score:
            return -theta_loss

        if self.rollout_policy is None:
            raise RuntimeError("reference_relative_score=True requires a frozen rollout_policy.")
        with torch.no_grad():
            ref_loss = self._flow_losses(self.rollout_policy, obs, actions)
        return ref_loss - theta_loss

    def _score_stacked_actions(
        self,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Score stacked actions ``[K, B, horizon, action_dim]`` -> ``[K, B]`` in one batched pass.

        Flattens the ``K`` samples into a ``[K*B]`` batch, encodes obs ONCE per actor (tiled across
        ``K``), and runs a single velocity forward pass per actor instead of a ``K``-length Python
        loop that re-encoded obs every iteration. Mirrors :meth:`_action_scores` (plain or
        reference-relative score) but vectorized over the ``K`` Monte Carlo samples.
        """
        if actions.dim() != 4:
            raise ValueError(f"Expected stacked actions [K, B, H, A], got {tuple(actions.shape)}")
        K, B = actions.shape[0], actions.shape[1]
        flat = actions.reshape(K * B, *actions.shape[2:])  # row (k*B + b) -> sample k, obs b

        # encode_obs(obs) is [B, cond]; .repeat(K, 1) tiles the whole obs block K times, so row
        # (k*B + b) is conditioned on obs b — matching the reshape order above.
        online_cond = self.online_actor.encode_obs(obs).repeat(K, 1)  # [K*B, cond]
        theta_loss = self._flow_losses_with_cond(self.online_actor, online_cond, flat)  # [K*B]
        if not self.reference_relative_score:
            return (-theta_loss).reshape(K, B)

        if self.rollout_policy is None:
            raise RuntimeError("reference_relative_score=True requires a frozen rollout_policy.")
        with torch.no_grad():
            ref_cond = self.rollout_policy.encode_obs(obs).repeat(K, 1)
            ref_loss = self._flow_losses_with_cond(self.rollout_policy, ref_cond, flat)
        return (ref_loss - theta_loss).reshape(K, B)

    def _compute_anchor_loss(
        self,
        obs: Union[dict, torch.Tensor],
        actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Anchor online actor to frozen rollout policy on stored non-intervention (label-0) actions.

        Matches the online velocity field to the frozen rollout policy's field on the buffer's
        *accepted robot* actions (intervention label 0) for the states where they occurred, rather
        than on fresh chunks sampled from the frozen policy:

            E_{(s,a)~D_0, t, x0}
                || v_theta(a_t,t,s) - v_0(a_t,t,s) ||^2

        where D_0 is the non-intervention subset of the batch (``mask``). This pins the online actor's
        score of the actions it actually took autonomously to the frozen policy's. Without it, the
        online actor's flow-matching score of rollout-like actions (rollout_score_mean) drifts down,
        which inflates the MILE intervention probability even when the human score is roughly
        stationary.

        ``mask`` ``[B]`` selects the non-intervention (label-0) transitions. ``anchor_monte_carlo_samples``
        fresh flow-matching (t, x0) draws are averaged per anchor action (gradients flow only through
        the online velocity prediction; the frozen field is a no_grad target). Returns 0 when the batch
        has no non-intervention transitions.
        """
        if self.anchor_loss_weight <= 0.0:
            return torch.zeros((), device=self.device)

        if self.rollout_policy is None:
            raise RuntimeError("anchor_loss_weight > 0 requires a frozen rollout_policy.")

        if not mask.any():
            return torch.zeros((), device=self.device)

        selected_obs = self._index_obs(obs, mask)
        selected_actions = self._prepare_actions(actions[mask]).float()

        # Encode obs once per actor; the conditioning is reused across the MC noise draws.
        online_global_cond = self.online_actor.encode_obs(selected_obs)
        with torch.no_grad():
            ref_global_cond = self.rollout_policy.encode_obs(selected_obs)

        anchor_losses = []
        for _ in range(self.anchor_monte_carlo_samples):
            x_t, t, _ = _flow_matching_target(self.online_actor, selected_actions)

            online_pred = self.online_actor.predict_velocity(x_t, t, online_global_cond)
            with torch.no_grad():
                ref_pred = self.rollout_policy.predict_velocity(x_t, t, ref_global_cond)

            anchor_losses.append(F.mse_loss(online_pred, ref_pred))

        return torch.stack(anchor_losses).mean()

    def _compute_intervention_probs(
        self,
        obs: Union[dict, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        interventions: Optional[torch.Tensor] = None,
        rollout_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MILE intervention probabilities for flow-matching actors.

        No-intervention uses the marginal:
            p(nu=1|s) = E_{a~pi_theta} Phi(beta(ell(a,s) - E_{a0~pi0}ell(a0,s)) - c)

        If ``condition_intervention_on_action`` is true, online intervention samples use:
            p(nu=1|s,a_h) = Phi(beta(ell(a_h,s) - E_{a0~pi0}ell(a0,s)) - c)

        If ``condition_nonintervention_on_robot`` is true, the baseline E_{a0~pi0}ell(a0,s) is
        replaced by the logged robot action's score ell(a_logged,s) on non-intervention (label-0)
        transitions (label-1/2 keep the sampled baseline).

        ``rollout_samples`` ``[K, B, H, A]`` lets the caller pass frozen-rollout chunks already drawn
        this step (reused for the anchor loss / diagnostics) so the ODE sampling is not repeated; when
        None they are sampled here.
        """
        if self.rollout_policy is None:
            raise RuntimeError(
                "FlowMILE requires a frozen rollout_policy. Call set_rollout_policy() before training."
            )

        # Robot/mental-model expectation under frozen rollout policy.
        if rollout_samples is None:
            rollout_samples = self._sample_policy_actions(
                self.rollout_policy,
                obs,
                self.monte_carlo_samples,
            )  # [K, B, H, A]
        rollout_scores = self._score_stacked_actions(obs, rollout_samples)  # [K, B]
        expected_rollout_score = rollout_scores.mean(dim=0)  # [B]

        # Debug: raw (pre-probit) score means. The rollout baseline E_{a0~pi0} ell(a0,s) is recorded
        # here; the observed human-correction score is added below when it is computed.
        self._last_score_diagnostics = {
            "expected_rollout_score_mean": float(expected_rollout_score.mean().item()),
        }

        # Optionally replace the sampled rollout baseline E_{a0~pi0} ell(a0,s) with the score of the
        # actual LOGGED robot action on non-intervention (label-0) transitions. This conditions the
        # "robot did fine" baseline on what the robot actually did (the analogue of
        # condition_intervention_on_action for the robot term) instead of marginalizing over sampled
        # rollout chunks. Only label-0 entries are overwritten; label-1/2 keep the sampled baseline.
        if self.condition_nonintervention_on_robot:
            if actions is None or interventions is None:
                raise ValueError(
                    "condition_nonintervention_on_robot=True requires actions and interventions."
                )
            nonintervention_mask = interventions < 0.5
            if nonintervention_mask.any():
                robot_scores = self._action_scores(obs, self._prepare_actions(actions))  # [B]
                self._last_score_diagnostics["nonintervention_robot_score_mean"] = float(
                    robot_scores[nonintervention_mask].mean().item()
                )
                expected_rollout_score = expected_rollout_score.clone()
                expected_rollout_score[nonintervention_mask] = robot_scores[nonintervention_mask]

        # Original MILE marginal over the trainable policy.
        # The expectation is over the trainable policy.  Even when EMA is enabled, use
        # online_actor here; EMA is only for deployment/evaluation.  sample_actions() itself
        # is no_grad in FlowMatchingActor, so gradients flow only through the score evaluation below.
        training_samples = self._sample_policy_actions(
            self.online_actor,
            obs,
            self.monte_carlo_samples,
        )
        training_scores = self._score_stacked_actions(obs, training_samples)  # [K, B]

        probit_diff_marginal = training_scores - expected_rollout_score.unsqueeze(0)  # [K, B]

        # Optionally standardize the score gap by its EMA running mean/std before the probit, so
        # probit_scale / intervention_cost are decoupled from the absolute (drifting) scale of the
        # flow-matching score. Stats are updated from (and only from) this marginal gap; the affine
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
                # Debug: mean observed-action score on the human-correction (label-1) subset only.
                self._last_score_diagnostics["observed_human_score_mean"] = float(
                    observed_scores[observed_intervention_mask].mean().item()
                )
                # Standardize with the SAME running stats as the marginal (do not update them from
                # the observed gap) so both probit arguments live on the same scale.
                # observed_gap = observed_scores - expected_rollout_score.detach()
                # normalized_observed_gap = self._normalize_score_gap(observed_gap)                
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
        """Flow-matching behavior-cloning loss on intervention/offline actions.

        Returns:
            (loss, velocity_pred_mean)
        """
        if not mask.any():
            return torch.zeros((), device=self.device), torch.zeros((), device=self.device)

        selected_obs = self._index_obs(obs, mask)
        selected_actions = self._prepare_actions(actions[mask]).float()

        x_t, t, target = _flow_matching_target(self.online_actor, selected_actions)
        global_cond = self.online_actor.encode_obs(selected_obs)
        v_pred = self.online_actor.predict_velocity(x_t, t, global_cond)
        return F.mse_loss(v_pred, target), v_pred.mean()

    @staticmethod
    def _index_obs(obs: Union[dict, torch.Tensor], mask: torch.Tensor) -> Union[dict, torch.Tensor]:
        if isinstance(obs, dict):
            return {k: v[mask] for k, v in obs.items()}
        return obs[mask]

    def train_step(self, logging_prefix: str = "flow_mile", rollout_step: int = None) -> dict:
        """Perform one training step of flow-matching MILE."""
        if rollout_step is not None and rollout_step < self.learning_starts:
            return {}
        if self.rollout_policy is None:
            raise RuntimeError(
                "FlowMILE requires a frozen rollout_policy. Call set_rollout_policy() before training."
            )

        actor_losses = []
        intervention_losses = []
        anchor_losses = []
        total_losses = []
        probit_diagnostics = []  # per-step CDF-input stats for probit_scale / intervention_cost tuning
        # Raw (pre-probit) score means for debugging: rollout baseline and observed human-correction
        # score straight out of _compute_intervention_probs (free — already computed for the probit).
        expected_rollout_score_means = []
        observed_human_score_means = []
        nonintervention_robot_score_means = []  # logged robot-action score on label-0 (when conditioning on it)
        expert_action_means = []
        sample_mse_errors = []
        unnormalized_sample_mse_errors = []
        sample_mse_human_errors = []  # sampled vs human (label-1) actions only
        sample_mse_robot_errors = []  # sampled vs robot (label-0) actions only
        score_human_means = []
        score_non_intervention_means = []  # mean score of dataset label-0 (non-intervention) actions
        score_rollout_means = []
        velocity_pred_means = []
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
            policy_mask = interventions < 0.5  # {0}: non-intervention (robot/accepted) — anchor targets

            if self.config.obs_noise_std > 0:
                if isinstance(obs, dict):
                    obs = {k: v + torch.randn_like(v) * self.config.obs_noise_std for k, v in obs.items()}
                else:
                    obs = obs + torch.randn_like(obs) * self.config.obs_noise_std

            if self.config.action_noise_std > 0:
                actions = actions.clone()
                if action_mask.any():
                    actions[action_mask] = actions[action_mask] + torch.randn_like(actions[action_mask]) * self.config.action_noise_std

            # Sample frozen rollout-policy chunks ONCE this step; reused by the probit baseline and
            # the rollout-score diagnostic (same frozen policy + same obs). Sharing the samples across
            # these Monte Carlo estimates is a standard variance-reduction trick and does not bias
            # either of them, while saving an extra ODE rollout per step. (The anchor loss no longer
            # uses these — it anchors on the buffer's stored non-intervention actions instead.)
            rollout_samples = self._sample_policy_actions(
                self.rollout_policy, obs, self.monte_carlo_samples
            )  # [K, B, H, A]

            intervention_probs = self._compute_intervention_probs(
                obs=obs,
                actions=actions,
                interventions=interventions,
                rollout_samples=rollout_samples,
            ).view_as(interventions)
            if self._last_probit_diagnostics:
                probit_diagnostics.append(self._last_probit_diagnostics)
            expected_rollout_score_means.append(
                self._last_score_diagnostics.get("expected_rollout_score_mean", np.nan)
            )
            observed_human_score_means.append(
                self._last_score_diagnostics.get("observed_human_score_mean", np.nan)
            )
            nonintervention_robot_score_means.append(
                self._last_score_diagnostics.get("nonintervention_robot_score_mean", np.nan)
            )

            if online_mask.any():
                intervention_loss = F.binary_cross_entropy(
                    intervention_probs[online_mask],
                    interventions[online_mask],
                )
            else:
                intervention_loss = torch.zeros((), device=self.device)

            actor_loss, velocity_pred_mean = self._compute_actor_bc_loss(obs, actions, action_mask)
            anchor_loss = self._compute_anchor_loss(obs=obs, actions=actions, mask=policy_mask)
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
            velocity_pred_means.append(velocity_pred_mean.item() if isinstance(velocity_pred_mean, torch.Tensor) else float(velocity_pred_mean))

            # Per-class intervention prob (label 0 = policy, label 1 = human). NaN when a class is
            # absent from this batch; nan-aggregated below so it survives sparse batches.
            human_mask = (interventions > 0.5) & (interventions < 1.5)
            iprob = intervention_probs.detach()
            iprob_policy_means.append(iprob[policy_mask].mean().item() if policy_mask.any() else np.nan)
            iprob_human_means.append(iprob[human_mask].mean().item() if human_mask.any() else np.nan)

            if action_mask.any():
                expert_action_means.append(actions[action_mask].mean().item())
            else:
                expert_action_means.append(np.nan)

            # Score diagnostics each cost extra flow-matching passes, so compute them only every
            # log_score_metrics_every gradient steps (the rollout-score reuses this step's already-
            # sampled rollout chunks, so it adds only a scoring pass, not another ODE rollout).
            log_scores = (
                self.config.log_score_metrics_every > 0
                and self.step_counter % self.config.log_score_metrics_every == 0
            )
            if log_scores:
                with torch.no_grad():
                    if action_mask.any():
                        score_human_means.append(
                            self._action_scores(self._index_obs(obs, action_mask), actions[action_mask]).mean().item()
                        )
                    # Mean score of the non-intervention (label-0) dataset actions (robot/accepted).
                    if policy_mask.any():
                        score_non_intervention_means.append(
                            self._action_scores(self._index_obs(obs, policy_mask), actions[policy_mask]).mean().item()
                        )
                    score_rollout_means.append(self._action_scores(obs, rollout_samples[0]).mean().item())

            with torch.no_grad():
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
            "velocity_pred_mean": float(np.mean(velocity_pred_means)),
            "intervention_prob_mean": float(intervention_probs[online_mask].mean().item()) if online_mask.any() else float("nan"),
            # Per-class probit outputs: healthy => human_mean >> policy_mean; collapsed (base-rate)
            # => both ~ dataset intervention rate. The gap is the discrimination signal.
            "intervention_prob_policy_mean": float(np.nanmean(iprob_policy_means)) if np.any(~np.isnan(iprob_policy_means)) else float("nan"),
            "intervention_prob_human_mean": float(np.nanmean(iprob_human_means)) if np.any(~np.isnan(iprob_human_means)) else float("nan"),
            # Raw (pre-probit) score means straight from _compute_intervention_probs: the rollout
            # baseline E_{a0~pi0} ell(a0,s) and the observed human-correction action score ell(a_h,s).
            # observed_human_score_mean is NaN unless condition_intervention_on_action=True and the
            # batch contained label-1 human corrections.
            "expected_rollout_score_mean": float(np.nanmean(expected_rollout_score_means)) if np.any(~np.isnan(expected_rollout_score_means)) else float("nan"),
            "observed_human_score_mean": float(np.nanmean(observed_human_score_means)) if np.any(~np.isnan(observed_human_score_means)) else float("nan"),
            # Logged robot-action score on label-0 states, used as the baseline when
            # condition_nonintervention_on_robot=True (NaN otherwise / when no label-0 in the batch).
            "nonintervention_robot_score_mean": float(np.nanmean(nonintervention_robot_score_means)) if np.any(~np.isnan(nonintervention_robot_score_means)) else float("nan"),
        }
        # Score diagnostics are computed only on logging steps (log_score_metrics_every), so add them
        # only when collected this train_step; otherwise they are simply skipped for this log.
        if score_human_means:
            metrics_dict["human_score_mean"] = float(np.nanmean(score_human_means))
        if score_non_intervention_means:
            metrics_dict["non_intervention_score_mean"] = float(np.nanmean(score_non_intervention_means))
        if score_rollout_means:
            metrics_dict["rollout_score_mean"] = float(np.nanmean(score_rollout_means))
        if sample_mse_errors:
            metrics_dict["sample_mse_error"] = float(np.mean(sample_mse_errors))
            metrics_dict["unnormalized_sample_mse_error"] = float(np.mean(unnormalized_sample_mse_errors))
            metrics_dict["sample_mse_error_human"] = float(np.mean(sample_mse_human_errors)) if sample_mse_human_errors else float("nan")
            metrics_dict["sample_mse_error_robot"] = float(np.mean(sample_mse_robot_errors)) if sample_mse_robot_errors else float("nan")

        # Probit CDF-input diagnostics: probit_diff_raw_* (unnormalized flow-matching-score gap) and
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
# Simulated-human intervention criteria (flow-matching-score analogues of the MILE ones)
# =====================================================================================
# These mirror the Gaussian helpers in ``algorithms/mile/modeling_mile.py`` and the diffusion ones
# in ``algorithms/diffusion_mile/modeling_diffusion_mile.py``, but score actions with the
# flow-matching loss proxy, so they work with FlowMatchingActor policies. They are inference-only
# (no grad) and plug into the HITL collector via ``hitl_utils.get_intervention_criteria`` (names
# "flow_mile" / "flow_mile_paired" / "flow_mile_window").


@torch.no_grad()
def _flow_score_under(
    policy: FlowMatchingActor,
    obs: Union[dict, torch.Tensor],
    actions: torch.Tensor,
    score_monte_carlo_samples: int = 1,
) -> torch.Tensor:
    """Flow-matching score proxy ``ell(a, s) = -E_{t,x0}|| u_target - v_policy(x_t,t,s) ||^2``.

    Inference-only counterpart of :meth:`FlowMILE._flow_losses`, returning the NEGATIVE loss so that
    higher => the action is more likely under ``policy``. ``actions`` may be ``[B, A]`` (single step)
    or ``[B, H, A]``; a single-step action is tiled across the policy's horizon so the conditional
    network always sees a full-length chunk. Returns ``[B]``.
    """
    if actions.dim() == 2:
        actions = actions.unsqueeze(1)
    if actions.shape[1] == 1 and policy.horizon > 1:
        actions = actions.repeat(1, policy.horizon, 1)
    actions = actions.float()
    batch_size = actions.shape[0]
    sigma_min = float(policy.sigma_min)

    global_cond = policy.encode_obs(obs)

    losses = []
    for _ in range(int(score_monte_carlo_samples)):
        x0 = torch.randn_like(actions)
        t = torch.rand(batch_size, device=actions.device)
        t_expand = t.view(batch_size, *([1] * (actions.dim() - 1)))
        x_t = (1.0 - (1.0 - sigma_min) * t_expand) * x0 + t_expand * actions
        target = actions - (1.0 - sigma_min) * x0
        v_pred = policy.predict_velocity(x_t, t, global_cond)
        per_element = (v_pred - target).pow(2)
        losses.append(per_element.reshape(batch_size, -1).mean(dim=-1))
    return -torch.stack(losses, dim=0).mean(dim=0)


@torch.no_grad()
def flow_mile_intervention_prob(
    judge_policy: FlowMatchingActor,
    other_policy: FlowMatchingActor,
    obs: Union[dict, torch.Tensor],
    num_samples: int = 8,
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
) -> torch.Tensor:
    """Flow-matching-score analogue of :func:`...mile.modeling_mile.mile_intervention_prob`.

        p = E_{a_j~judge}[ Phi( beta * (ell_judge(a_j, s) - E_{a_o~other}[ell_judge(a_o, s)]) - c ) ]

    where ``ell_judge`` is the judge's flow-matching-score proxy (higher => more likely under the
    judge). It is high when the *other* policy's actions fit the judge's velocity field poorly (i.e.
    the judge finds them unlikely). For a simulated human, pass ``judge=expert_policy`` and
    ``other=rollout_policy``. Both policies must be FlowMatchingActors. Returns a tensor ``[B]``
    (B inferred from ``obs``).

    NOTE: each sample is a full Euler ODE rollout, so this costs ``2 * num_samples`` action samplings
    per call. Keep ``num_samples`` modest (and prefer a low ``num_inference_steps`` flow actor) for
    interactive simulated rollouts.
    """
    if not isinstance(judge_policy, FlowMatchingActor) or not isinstance(other_policy, FlowMatchingActor):
        raise TypeError(
            "flow_mile_intervention_prob requires FlowMatchingActor judge/other policies, got "
            f"judge={type(judge_policy).__name__}, other={type(other_policy).__name__}."
        )

    # Judge scores the OTHER policy's samples (the expectation/baseline term).
    other_scores = [
        _flow_score_under(judge_policy, obs, other_policy.sample_actions(obs), score_monte_carlo_samples)
        for _ in range(num_samples)
    ]
    expected_score_other = torch.stack(other_scores, dim=0).mean(dim=0)  # [B]

    # Judge scores its OWN samples.
    self_scores = [
        _flow_score_under(judge_policy, obs, judge_policy.sample_actions(obs), score_monte_carlo_samples)
        for _ in range(num_samples)
    ]
    score_j_self = torch.stack(self_scores, dim=0)  # [K, B]

    probit_arg = probit_scale * (score_j_self - expected_score_other.unsqueeze(0)) - intervention_cost
    return normal_cdf(probit_arg).mean(dim=0).clamp(EPS, 1.0 - EPS)


def make_flow_mile_intervention_criteria(
    num_samples: int = 8,
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
    stochastic: bool = True,
    threshold: float = 0.5,
    **_,
):
    """Build a simulated-human criterion from FlowMILE's flow-matching-score probit.

    Mirrors :func:`...mile.modeling_mile.make_mile_intervention_criteria` but scores actions with the
    flow-matching loss proxy instead of Gaussian log-probs. Each step the expert (judge) computes
    ``p = flow_mile_intervention_prob(expert, rollout, obs)``; it intervenes with ``Bernoulli(p)``
    (``stochastic``) or when ``p > threshold``. On intervention the expert action is executed,
    otherwise the rollout action. Both the expert and the rollout policy must be FlowMatchingActors.
    Returns a function with the HITL criterion signature.
    """

    def criteria(obs, rollout_policy, expert_policy, rollout_action, expert_action,
                 rollout_chunk=None, expert_chunk=None):
        p = float(
            flow_mile_intervention_prob(
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
    policy: FlowMatchingActor,
    obs: Union[dict, torch.Tensor],
    env_action,
    score_monte_carlo_samples: int = 1,
) -> float:
    """Flow-matching score ``ell(a, s)`` for a single env-space action under a FlowMatchingActor.

    Maps the env-space action into the policy's normalized [-1, 1] space, tiles it across the
    policy's horizon (handled by :func:`_flow_score_under`), and returns the per-state score for the
    first env in the batch.
    """
    device = next(policy.parameters()).device
    a_norm = policy.normalize_action(torch.as_tensor(np.asarray(env_action, dtype=np.float32), device=device))
    a_norm = a_norm.reshape(1, 1, -1)  # [B=1, H=1, A]
    score = _flow_score_under(policy, obs, a_norm, score_monte_carlo_samples)
    return float(score.reshape(-1)[0])


@torch.no_grad()
def _chunk_score_under(
    policy: FlowMatchingActor,
    obs: Union[dict, torch.Tensor],
    env_chunk,
    score_monte_carlo_samples: int = 1,
) -> float:
    """Flow-matching score ``ell(a, s)`` for a full env-space action *chunk* under a FlowMatchingActor.

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
    return float(_flow_score_under(policy, obs, a_norm, score_monte_carlo_samples).reshape(-1)[0])


def make_flow_mile_paired_criteria(
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
    stochastic: bool = True,
    threshold: float = 0.5,
    **_,
):
    """Sampling-free FlowMILE criterion that scores the proposed human vs robot action.

    Unlike :func:`make_flow_mile_intervention_criteria` (which Monte-Carlo marginalizes over action
    chunks sampled from BOTH policies via expensive Euler integration), this directly scores the two
    *already-proposed* env actions under the expert's velocity field:

        p = Phi( probit_scale * (ell_expert(a_human) - ell_expert(a_robot)) - intervention_cost )

    i.e. it intervenes when the robot's actual action fits the expert's velocity field worse than the
    human's own action would. This models a human with a **perfect model of the robot** — it judges
    the robot's specific action at this state rather than marginalizing over what the robot *might*
    do. It costs ~2 flow-matching-score evals per step (no ``sample_actions`` calls), so it is far
    faster than the marginal criterion and well suited to interactive simulated rollouts.

    ``score_monte_carlo_samples`` still controls the flow-matching (t, x0) draws used to estimate each
    ``ell(a, s)`` (set it to 1 for the cheapest, noisiest score). Both the expert and rollout policy
    must be FlowMatchingActors. Returns a function with the HITL criterion signature.

    When the rollout worker supplies the full predicted ``rollout_chunk`` / ``expert_chunk`` (env
    space, ``[L, action_dim]``), the in-distribution chunks are scored directly under the expert
    velocity field; otherwise it falls back to scoring the single proposed actions tiled across the
    horizon (exact only at ``horizon == 1``).
    """

    def criteria(obs, rollout_policy, expert_policy, rollout_action, expert_action,
                 rollout_chunk=None, expert_chunk=None):
        if rollout_chunk is not None and expert_chunk is not None:
            # Score the actual predicted chunks (in-distribution) under the expert's velocity field.
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


def make_flow_mile_window_criteria(
    window: int = 10,
    score_monte_carlo_samples: int = 1,
    intervention_cost: float = 0.0,
    probit_scale: float = 1.0,
    stochastic: bool = True,
    threshold: float = 0.5,
    **_,
):
    """Windowed FlowMILE criterion: an EMA over the per-step expert-vs-robot score gap.

    Mirrors :func:`...mile.modeling_mile.make_mile_window_criteria` with flow-matching scores. Per
    step ``g_t = ell_expert(a_expert_t) - ell_expert(a_robot_t) - c`` (>= 0 when the robot's
    *executed* action fits the expert's velocity field worse than the expert's own action). An EMA
    with decay ``alpha = 2/(window+1)`` smooths ``g_t`` so intervention reflects *sustained*
    divergence rather than a single-step blip; ``p = Phi(probit_scale * ema)``, then ``Bernoulli(p)``
    (stochastic) or ``p > threshold``. The single executed action is tiled across the expert's
    horizon for scoring.

    Stateful (EMA per episode); exposes ``.reset()`` (called by the rollout worker each episode).
    Much cheaper than the non-windowed criterion (no Euler-integration sampling — it scores the two
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
