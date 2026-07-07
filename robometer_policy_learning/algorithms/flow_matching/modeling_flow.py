"""Flow Matching (FM) policy implementation.

This module contains everything needed to train and deploy a Flow Matching policy
(Lipman et al., 2023, https://arxiv.org/abs/2210.02747; Liu et al., 2022 "rectified flow",
https://arxiv.org/abs/2209.03003) inside the robometer policy-learning framework:

* a conditional velocity-prediction network (the same 1D conditional U-Net, conditional MLP,
  and conditional Transformer used by Diffusion Policy),
* :class:`FlowMatchingActor`, a :class:`BaseActor` whose ``act()`` integrates the learned ODE
  (Euler) from Gaussian noise to an action chunk so it can be deployed/evaluated like any
  other actor,
* :class:`FlowMatching`, the :class:`BaseAlgorithm` that trains the actor by conditional
  flow matching (regressing the straight-line OT velocity).

This is the deterministic flow-matching analogue of :class:`DP`. Instead of a multi-step
diffusion noise schedule, training regresses a velocity field along the conditional OT path
``x_t = (1 - (1 - sigma_min) t) x0 + t x1`` and sampling integrates ``dx/dt = v(x_t, t, obs)``
from ``t=0`` to ``t=1``. The policy works in the actor's normalized ([-1, 1]) action space; the
replay buffer maps stored env-space actions into [-1, 1] and ``act()`` unnormalizes back.
"""

import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robometer_policy_learning.algorithms.dp.modeling_dp import EMAModel
from robometer_policy_learning.algorithms.flow_matching.configuration_flow import FlowMatchingConfig
from robometer_policy_learning.algorithms.modeling_algorithm import BaseAlgorithm
from robometer_policy_learning.modules.base import BaseActorConfig
from robometer_policy_learning.modules.base.modeling_actor import BaseActor
from robometer_policy_learning.modules.diffusion import ConditionalMLP, ConditionalTransformer, ConditionalUnet1D
from robometer_policy_learning.utils.featurizers import ObservationFeaturizer, _build_mlp_layers


# =====================================================================================
# Flow matching actor (deployable BaseActor)
# =====================================================================================
@dataclass
class FlowMatchingActorConfig(BaseActorConfig):
    """Config for :class:`FlowMatchingActor`.

    Carries the standard actor fields (obs/action space, normalization, featurizer/image
    encoder settings) plus the flow-matching hyperparameters from :class:`FlowMatchingConfig`.
    """

    # Featurizer / image-encoder settings (copied from the source actor config)
    featurizer: Optional[dict] = None
    activation: str = "relu"
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    image_encoder_type: Optional[str] = None
    finetune_image_encoder: bool = False
    image_feature_dim: int = 128
    resnet_backbone: str = "ResNet18"
    resnet_pretrained: bool = True
    resnet_pool: str = "spatial_softmax"
    spatial_softmax_num_kp: int = 32
    dinov2_model: object = None
    dinov2_processor: object = None
    impala_nn_scale: int = 1
    impala_num_blocks_per_stack: int = 2
    impala_use_smaller: bool = False
    impala_output_dim: Optional[int] = None

    # Flow matching hyperparameters
    horizon: int = 1
    num_inference_steps: int = 10
    sigma_min: float = 0.0
    time_embed_scale: float = 1000.0
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    net_type: str = "unet"
    diffusion_step_embed_dim: int = 128
    unet_down_dims: Tuple[int, ...] = (128, 256)
    unet_kernel_size: int = 5
    unet_n_groups: int = 8
    mlp_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    transformer_d_model: int = 256
    transformer_nhead: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 1024
    transformer_dropout: float = 0.0
    transformer_activation: str = "gelu"
    obs_encoder_hidden_dims: Tuple[int, ...] = (256, 256)

    @property
    def actor_class(self):
        return FlowMatchingActor


class FlowMatchingActor(BaseActor):
    """Actor whose ``act()`` produces actions by integrating a learned velocity field.

    Trainable parameters: an observation encoder (featurizer + MLP producing the global
    conditioning vector) and a velocity-prediction network (U-Net, MLP, or Transformer). The FM
    algorithm drives training through :meth:`encode_obs` / :meth:`predict_velocity`; deployment
    uses :meth:`_act` (called by :meth:`BaseActor.act`).
    """

    def __init__(self, config: FlowMatchingActorConfig):
        super().__init__(config)
        self.config = config
        self.preprocess_obs_transform = config.preprocess_obs_transform

        if not self.is_continuous:
            raise ValueError("FlowMatchingActor only supports continuous (Box) action spaces")

        self.action_dim = int(np.prod(config.action_space.shape))
        self.horizon = int(config.horizon)

        # --- Observation encoder: featurizer -> (B, obs_dim) -> MLP -> (B, global_cond_dim) ---
        self.obs_featurizer = ObservationFeaturizer(
            observation_space=config.observation_space,
            featurizer_cfg=config.featurizer,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            dropout_rate=config.dropout_rate,
            image_encoder_type=config.image_encoder_type,
            finetune_image_encoder=config.finetune_image_encoder,
            image_feature_dim=config.image_feature_dim,
            resnet_backbone=config.resnet_backbone,
            resnet_pretrained=config.resnet_pretrained,
            resnet_pool=config.resnet_pool,
            spatial_softmax_num_kp=config.spatial_softmax_num_kp,
            dinov2_model=config.dinov2_model,
            dinov2_processor=config.dinov2_processor,
            impala_nn_scale=config.impala_nn_scale,
            impala_num_blocks_per_stack=config.impala_num_blocks_per_stack,
            impala_use_smaller=config.impala_use_smaller,
            impala_output_dim=config.impala_output_dim,
        )
        obs_dim = int(self.obs_featurizer.output_dim)
        if obs_dim <= 0:
            raise ValueError("ObservationFeaturizer produced invalid output dimension for FlowMatchingActor.")

        if config.obs_encoder_hidden_dims:
            self.obs_encoder = nn.Sequential(
                *_build_mlp_layers(
                    obs_dim,
                    config.obs_encoder_hidden_dims,
                    config.activation,
                    config.use_layer_norm,
                    config.dropout_rate,
                )
            )
            self.global_cond_dim = int(config.obs_encoder_hidden_dims[-1])
        else:
            self.obs_encoder = nn.Identity()
            self.global_cond_dim = obs_dim

        # --- Velocity prediction network (shared architectures with Diffusion Policy) ---
        if config.net_type == "unet":
            self.net = ConditionalUnet1D(
                action_dim=self.action_dim,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                down_dims=config.unet_down_dims,
                kernel_size=config.unet_kernel_size,
                n_groups=config.unet_n_groups,
            )
        elif config.net_type == "mlp":
            self.net = ConditionalMLP(
                action_dim=self.action_dim,
                horizon=self.horizon,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                hidden_dims=config.mlp_hidden_dims,
            )
        elif config.net_type == "transformer":
            self.net = ConditionalTransformer(
                action_dim=self.action_dim,
                horizon=self.horizon,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=config.diffusion_step_embed_dim,
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                num_layers=config.transformer_num_layers,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
                activation=config.transformer_activation,
            )
        else:
            raise ValueError(f"Unknown net_type: {config.net_type!r} (expected 'unet', 'mlp', or 'transformer')")

        self.num_inference_steps = int(config.num_inference_steps)
        self.sigma_min = float(config.sigma_min)
        self.time_embed_scale = float(config.time_embed_scale)
        self.clip_sample = bool(config.clip_sample)
        self.clip_sample_range = float(config.clip_sample_range)

        # FlowMatchingActor has no Gaussian action distribution; act()/training use the flow ODE.
        self.action_dist = None

    # ------------------------------------------------------------------ helpers
    def encode_obs(self, obs: Union[dict, torch.Tensor]) -> torch.Tensor:
        """Featurize observations into the global conditioning vector ``(B, global_cond_dim)``."""
        if self.preprocess_obs_transform is not None:
            for transform in self.preprocess_obs_transform:
                obs = transform(obs)
        device = next(self.parameters()).device
        obs_flat = self.obs_featurizer.flatten_obs(obs, device=device)
        return self.obs_encoder(obs_flat.float())

    def predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """Network forward: predict the flow velocity ``v(x_t, t, obs)`` of the same shape as ``x_t``.

        ``t`` is the continuous flow time in [0, 1]; it is rescaled by ``time_embed_scale`` before
        the network's sinusoidal time embedding (which was designed for integer diffusion steps).
        """
        return self.net(x_t, t * self.time_embed_scale, global_cond)

    def _maybe_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_sample:
            return x.clamp(-self.clip_sample_range, self.clip_sample_range)
        return x

    @torch.no_grad()
    def sample_actions(
        self, obs: Union[dict, torch.Tensor], num_inference_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Integrate the learned ODE with Euler steps. Returns ``(B, horizon, action_dim)`` in [-1, 1].

        ``num_inference_steps`` overrides the actor's default step count for this call.
        """
        steps = int(num_inference_steps) if num_inference_steps is not None else self.num_inference_steps
        global_cond = self.encode_obs(obs)
        batch_size = global_cond.shape[0]
        device = global_cond.device

        x = torch.randn(batch_size, self.horizon, self.action_dim, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = i * dt
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            v = self.predict_velocity(x, t_batch, global_cond)
            x = x + dt * v
        return self._maybe_clip(x)

    @torch.no_grad()
    def sample_actions_batch(
        self, obs: Union[dict, torch.Tensor], num_samples: int, num_inference_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Sample ``num_samples`` independent action chunks per observation in ONE batched pass.

        Encodes the obs once, tiles the conditioning ``num_samples`` times, and runs a single Euler
        integration over a ``(num_samples * B)`` batch (each row gets its own initial noise), so the
        Monte Carlo samples are produced in parallel on the GPU rather than via a Python loop of
        ``sample_actions`` calls. Equivalent in distribution to ``num_samples`` independent
        ``sample_actions`` calls; only the obs encoding is shared (deterministic given obs).

        ``num_inference_steps`` overrides the actor's default step count for this call.

        Returns ``(num_samples, B, horizon, action_dim)`` in [-1, 1].
        """
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        steps = int(num_inference_steps) if num_inference_steps is not None else self.num_inference_steps

        global_cond = self.encode_obs(obs)  # (B, global_cond_dim)
        batch_size = global_cond.shape[0]
        device = global_cond.device
        kb = num_samples * batch_size

        # Tile conditioning K times so row (k * B + b) is conditioned on obs b; the final
        # .view(num_samples, B, ...) then recovers the [sample, obs] layout. Each row gets its own
        # initial noise, making the K chunks per obs independent draws.
        cond_rep = global_cond.repeat(num_samples, 1)  # (K*B, global_cond_dim)
        x = torch.randn(kb, self.horizon, self.action_dim, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t = i * dt
            t_batch = torch.full((kb,), t, device=device, dtype=torch.float32)
            v = self.predict_velocity(x, t_batch, cond_rep)
            x = x + dt * v
        x = self._maybe_clip(x)
        return x.view(num_samples, batch_size, self.horizon, self.action_dim)

    # ------------------------------------------------------------------ BaseActor API
    def _act(
        self, obs: Union[dict, torch.Tensor], deterministic: bool = False, actor_state: Any = None
    ) -> Tuple[torch.Tensor, Any]:
        # Flow-matching sampling starts from random noise so it is stochastic; ``deterministic`` is
        # accepted for API compatibility (a fixed seed could be added here for reproducibility).
        actions = self.sample_actions(obs)
        if self.horizon == 1:
            actions = actions.squeeze(1)  # (B, action_dim) for non-chunked deployment
        return actions, None

    def get_action_dist_params(self, obs, hidden=None):
        raise NotImplementedError("FlowMatchingActor does not expose an explicit action distribution.")

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))


# =====================================================================================
# Flow Matching algorithm
# =====================================================================================
class FlowMatching(BaseAlgorithm):
    """Flow Matching: behavior cloning via conditional flow matching (OT / rectified flow).

    ``config.actor`` must be a :class:`FlowMatchingActor` (built by
    ``training_utils.build_actor_critic_models`` like every other actor); it holds all trainable
    parameters. The (EMA) actor is exposed as ``self.actor`` so it can be evaluated and
    checkpointed like any other policy.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__(config)
        self.config = config

        online_actor = config.actor
        if online_actor is None:
            raise ValueError("A FlowMatchingActor is required for FlowMatching (built by build_actor_critic_models)")
        if not isinstance(online_actor, FlowMatchingActor):
            raise TypeError(
                f"FlowMatching requires a FlowMatchingActor, got {type(online_actor).__name__}. "
                "It is built in training_utils.build_actor_critic_models when offline_alg_name == 'flow'."
            )

        self.device = next(online_actor.parameters()).device
        self.horizon = online_actor.horizon
        self.online_actor = online_actor.to(self.device)

        for p in self.online_actor.parameters():
            p.requires_grad_(True)

        # EMA weights are used for deployment/eval; otherwise the online net is deployed.
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
        self.batch_size = config.batch_size
        self.learning_starts = config.learning_starts
        self.sigma_min = self.online_actor.sigma_min
        self.time_sampling = config.time_sampling
        self.logit_normal_mean = config.logit_normal_mean
        self.logit_normal_std = config.logit_normal_std
        self.use_weighted_fm = bool(getattr(config, "use_weighted_fm", False))

        self.actor_optimizer = torch.optim.AdamW(
            self.online_actor.parameters(),
            lr=config.actor_optimizer_lr,
            betas=config.actor_optimizer_betas,
            eps=config.actor_optimizer_eps,
            weight_decay=config.actor_optimizer_weight_decay,
        )

        print(
            f"FlowMatching: net_type={config.net_type}, horizon={self.horizon}, "
            f"action_dim={self.online_actor.action_dim}"
        )
        print(
            f"FlowMatching: num_inference_steps={config.num_inference_steps}, sigma_min={self.sigma_min}, "
            f"time_sampling={self.time_sampling}, use_ema={self.use_ema}, use_weighted_fm={self.use_weighted_fm}"
        )

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Coerce buffer actions into ``(B, horizon, action_dim)``."""
        if actions.dim() == 2:  # (B, action_dim) -> single-step chunk
            actions = actions.unsqueeze(1)
        elif actions.dim() != 3:
            raise ValueError(f"Unexpected action shape for FlowMatching: {tuple(actions.shape)}")
        return actions

    def _sample_time(self, batch_size: int) -> torch.Tensor:
        """Draw the flow time ``t`` in (0, 1) for each sample, per ``time_sampling``."""
        if self.time_sampling == "uniform":
            return torch.rand(batch_size, device=self.device)
        elif self.time_sampling == "logit_normal":
            s = torch.randn(batch_size, device=self.device) * self.logit_normal_std + self.logit_normal_mean
            return torch.sigmoid(s)
        raise ValueError(f"Unknown time_sampling: {self.time_sampling!r} (expected 'uniform' or 'logit_normal')")

    def train_step(self, logging_prefix: str = "flow", rollout_step: int = None) -> dict:
        if rollout_step is not None and rollout_step < self.learning_starts:
            return {}

        losses = []
        velocity_pred_means = []
        expert_action_means = []
        sample_mse_errors = []
        unnormalized_sample_mse_errors = []

        gradient_steps = self.config.num_updates_per_train_step

        for _ in range(gradient_steps):
            batch = self.buffer.sample(self.batch_size, device=self.device)
            if not batch:
                print("Buffer is still empty. Skipping this training step")
                return {}

            obs = batch["obs"]
            expert_actions = batch["action"]
            if len(expert_actions) == 0:
                print("Buffer is still empty. Skipping this training step")
                return {}

            expert_actions = self._prepare_actions(expert_actions).float()

            # Optional data augmentation (parity with BC / DP).
            if self.config.obs_noise_std > 0:
                if isinstance(obs, dict):
                    obs = {k: v + torch.randn_like(v) * self.config.obs_noise_std for k, v in obs.items()}
                else:
                    obs = obs + torch.randn_like(obs) * self.config.obs_noise_std
            if self.config.action_noise_std > 0:
                expert_actions = expert_actions + torch.randn_like(expert_actions) * self.config.action_noise_std

            batch_size = expert_actions.shape[0]

            # Conditional flow matching: sample noise x0, flow time t, form the OT interpolant x_t,
            # and regress the (constant) target velocity u = x1 - (1 - sigma_min) * x0.
            x0 = torch.randn_like(expert_actions)
            x1 = expert_actions
            t = self._sample_time(batch_size)
            t_expand = t.view(batch_size, *([1] * (x1.dim() - 1)))
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x0 + t_expand * x1
            target = x1 - (1.0 - self.sigma_min) * x0

            global_cond = self.online_actor.encode_obs(obs)
            v_pred = self.online_actor.predict_velocity(x_t, t, global_cond)

            if self.use_weighted_fm:
                per_sample = (v_pred - target).pow(2).reshape(batch_size, -1).mean(dim=1)
                w = batch.get("weight") if isinstance(batch, dict) else None
                if w is None:
                    w = torch.ones_like(per_sample)
                else:
                    w = torch.as_tensor(w, dtype=per_sample.dtype, device=per_sample.device).view(-1)
                loss = (w * per_sample).sum() / (w.sum() + 1e-8)
            else:
                loss = F.mse_loss(v_pred, target)

            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.config.clip_grad_norm and self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.online_actor.parameters(), max_norm=self.config.clip_grad_norm)
            self.actor_optimizer.step()

            if self.use_ema:
                self.ema.step(self.online_actor)

            losses.append(loss.item())
            velocity_pred_means.append(v_pred.mean().item())
            expert_action_means.append(expert_actions.mean().item())

            # Optional (expensive) end-to-end sampling diagnostic.
            if (
                self.config.log_sample_metrics_every > 0
                and (self.step_counter + 1) % self.config.log_sample_metrics_every == 0
            ):
                with torch.no_grad():
                    sampled = self.actor.sample_actions(obs)
                    if sampled.shape != expert_actions.shape:
                        sampled = sampled.reshape(expert_actions.shape)
                    sample_mse_errors.append(F.mse_loss(sampled, expert_actions).item())
                    unnormalized_sample_mse_errors.append(
                        F.mse_loss(
                            self.actor.unnormalize_action(sampled),
                            self.actor.unnormalize_action(expert_actions),
                        ).item()
                    )

            self.step_counter += 1

        metrics_dict = {
            "actor_loss": float(np.mean(losses)),
            "velocity_pred_mean": float(np.mean(velocity_pred_means)),
            "expert_action_mean": float(np.mean(expert_action_means)),
        }
        if sample_mse_errors:
            metrics_dict["sample_mse_error"] = float(np.mean(sample_mse_errors))
            metrics_dict["unnormalized_sample_mse_error"] = float(np.mean(unnormalized_sample_mse_errors))

        if self.logger is not None:
            self.logger.log(metrics_dict, step=self.step_counter, prefix=logging_prefix)

        return metrics_dict

    @torch.no_grad()
    def evaluate_policy(self, eval_buffer, num_eval_batches: int = 10) -> dict:
        """Flow-matching loss + one-shot sampling MSE on a held-out buffer (overfitting check)."""
        self.online_actor.eval()
        eval_losses = []
        eval_sample_mse = []
        for _ in range(num_eval_batches):
            batch = eval_buffer.sample(self.batch_size, device=self.device)
            if not batch:
                continue
            obs = batch["obs"]
            expert_actions = self._prepare_actions(batch["action"]).float()
            batch_size = expert_actions.shape[0]

            x0 = torch.randn_like(expert_actions)
            x1 = expert_actions
            t = self._sample_time(batch_size)
            t_expand = t.view(batch_size, *([1] * (x1.dim() - 1)))
            x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x0 + t_expand * x1
            target = x1 - (1.0 - self.sigma_min) * x0
            global_cond = self.online_actor.encode_obs(obs)
            v_pred = self.online_actor.predict_velocity(x_t, t, global_cond)
            eval_losses.append(F.mse_loss(v_pred, target).item())

            sampled = self.actor.sample_actions(obs)
            if sampled.shape != expert_actions.shape:
                sampled = sampled.reshape(expert_actions.shape)
            eval_sample_mse.append(F.mse_loss(sampled, expert_actions).item())

        self.online_actor.train()
        return {
            "eval_loss": float(np.mean(eval_losses)) if eval_losses else float("nan"),
            "eval_sample_mse_error": float(np.mean(eval_sample_mse)) if eval_sample_mse else float("nan"),
        }
