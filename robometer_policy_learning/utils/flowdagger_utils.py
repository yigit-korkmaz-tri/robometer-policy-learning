"""FlowDAgger building blocks: a frozen flow-matching policy steered through its initial noise.

FlowDAgger (Murray et al., CoRL 2026) keeps the flow-matching policy frozen and trains a small STEERING
network that outputs the initial noise ``w`` its sampler integrates from. Targets are behavior-cloned
in noise space:

  * human corrections -> the noise that would have produced the corrected action chunk, found by
    inverting the policy's discrete Euler sampler (per-step fixed-point inversion, paper Eq. 3);
  * autonomous chunks  -> the noise that actually produced them (recorded at collection time).

Two backends share the sample construction (:func:`build_samples`), the :class:`SteeringNet` and the
samples I/O:

  * pi0 / pi0.5 on LIBERO (``scripts/train_flowdagger.py``): :class:`Pi0FlowOps` (JAX),
    :class:`ActionNormalizer`, :class:`FlowDAggerPi0LiberoHitlWorker`. openpi convention: t=1 is noise and
    t=0 the action, ``x <- x + dt * v(x, t)`` with ``dt = -1/num_steps`` (10 steps for pi0.5).
  * robomimic FlowMatchingActor (``scripts/train_flowdagger_robomimic.py``): :class:`SteeredFlowActor`,
    :func:`invert_chunks`, :class:`FlowDAggerHitlWorker`, :func:`build_round_samples` (torch). Convention
    (``algorithms/flow_matching/modeling_flow.py``): t=0 is noise and t=1 the action, ``dt = +1/num_steps``,
    then an optional clip; actions are in [-1, 1] and ``BaseActor.act`` unnormalizes them.
    :class:`SteeredFlowActor` lives in this package (not in the script) so its pickled ``actor.pt`` loads
    from any eval script.

The pi0 inversion and the env-action -> model-action mapping are adapted from the official MIT-licensed
implementation (``FlowDAgger/flowdagger_pi05/flow_matching_inverter.py`` and ``action_converter.py``,
Copyright (c) Microsoft Corporation).
"""

import json
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from robometer_policy_learning.algorithms.flow_matching.modeling_flow import FlowMatchingActor
from robometer_policy_learning.utils.hitl_utils import HitlRolloutWorker
from robometer_policy_learning.utils.pi0_hitl_utils import (
    INTERVENTION_LABEL,
    PI0_IMAGE_KEY,
    PI0_STATE_KEY,
    PI0_WRIST_KEY,
    ROLLOUT_LABEL,
    Pi0LiberoHitlWorker,
)

# Training-sample kinds (``kind`` in the samples file).
KIND_POLICY, KIND_HUMAN = 0, 1


# ============================================================================================
# Shared: episodes -> training samples
# ============================================================================================
def build_samples(
    episode: Dict[str, np.ndarray],
    *,
    horizon: int,
    use_policy: bool = True,
    drop_pre_takeover_chunk: bool = True,
    human_window_stride: Optional[int] = None,
    min_human_window: int = 1,
) -> List[Dict]:
    """Turn one kept episode into sample specs ``{kind, t, target | actions}`` (``t`` = the obs step).

    ``episode`` holds per-step arrays: ``actions [N, A]``, ``intervention [N]`` and, for episodes collected
    by :class:`FlowDAggerPi0LiberoHitlWorker` / :class:`FlowDAggerHitlWorker`, ``chunk_id [N]`` and ``noise [N, H, D]``.

    * POLICY samples: one per autonomous chunk, at its first step ``t``, target = the stored noise. The
      last chunk before each takeover is dropped when ``drop_pre_takeover_chunk`` -- it is the behavior
      the human rejected. Episodes without ``noise`` (e.g. plain HITL HDF5s) contribute no policy samples.
    * HUMAN samples: every contiguous human segment is cut into windows of up to ``horizon`` steps
      starting at the takeover, every ``human_window_stride`` steps (default ``horizon``: back-to-back
      chunks, as the policy would have replanned). ``actions`` is that window's executed human chunk;
      it is inverted later. Windows shorter than ``min_human_window`` are dropped.
    """
    interv = np.asarray(episode["intervention"]).reshape(-1)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    n = len(interv)
    stride = int(human_window_stride or horizon)
    samples: List[Dict] = []

    if use_policy and "noise" in episode and "chunk_id" in episode:
        chunk_ids = np.asarray(episode["chunk_id"]).reshape(-1)
        noise = np.asarray(episode["noise"], dtype=np.float32)
        rejected = set()
        if drop_pre_takeover_chunk:
            for t in range(1, n):
                if interv[t] == INTERVENTION_LABEL and interv[t - 1] == ROLLOUT_LABEL:
                    rejected.add(int(chunk_ids[t - 1]))
        seen = set()
        for t in range(n):
            cid = int(chunk_ids[t])
            if interv[t] != ROLLOUT_LABEL or cid < 0 or cid in seen:
                continue
            seen.add(cid)
            if cid not in rejected:
                samples.append({"kind": KIND_POLICY, "t": t, "target": noise[t]})

    t = 0
    while t < n:
        if interv[t] != INTERVENTION_LABEL:
            t += 1
            continue
        end = t
        while end < n and interv[end] == INTERVENTION_LABEL:
            end += 1
        for s in range(t, end, stride):
            window = actions[s : min(s + horizon, end)]
            if len(window) >= min_human_window:
                samples.append({"kind": KIND_HUMAN, "t": s, "actions": window})
        t = end
    return samples


# ============================================================================================
# Shared: steering network
# ============================================================================================
class SteeringNet(nn.Module):
    """Observation features -> initial flow noise ``[horizon, noise_dim]`` (deterministic).

    The paper's ``vlm_pi0`` head: the pooled VLM feature goes through a ``Linear + LayerNorm + tanh``
    bottleneck, is concatenated with the proprio state (``state_dim=0``: none, as on robomimic, where the
    feature is the actor's ``encode_obs``), then an MLP. ``output_bound='tanh'`` squashes to
    ``[-noise_bound, noise_bound]``; ``'linear'`` leaves the output unbounded.
    """

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        horizon: int,
        noise_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        latent_dim: int = 256,
        noise_bound: float = 3.0,
        output_bound: str = "tanh",
    ):
        super().__init__()
        if output_bound not in ("tanh", "linear"):
            raise ValueError(f"output_bound must be 'tanh' or 'linear', got {output_bound!r}")
        self.kwargs = dict(
            feature_dim=int(feature_dim), state_dim=int(state_dim), horizon=int(horizon), noise_dim=int(noise_dim),
            hidden_dims=[int(h) for h in hidden_dims], latent_dim=int(latent_dim), noise_bound=float(noise_bound),
            output_bound=str(output_bound),
        )
        self.vlm_dim = int(feature_dim) - int(state_dim)
        self.state_dim, self.horizon, self.noise_dim = int(state_dim), int(horizon), int(noise_dim)
        self.noise_bound, self.output_bound = float(noise_bound), str(output_bound)
        self.bottleneck = nn.Sequential(nn.Linear(self.vlm_dim, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        layers, d = [], latent_dim + self.state_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(d, self.horizon * self.noise_dim)
        # Small init: start near zero noise, the mode of N(0, I).
        nn.init.orthogonal_(self.head.weight, gain=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        vlm, state = features[..., : self.vlm_dim], features[..., self.vlm_dim :]
        h = self.mlp(torch.cat([self.bottleneck(vlm), state], dim=-1))
        out = self.head(h)
        if self.output_bound == "tanh":
            out = self.noise_bound * torch.tanh(out)
        return out.view(*features.shape[:-1], self.horizon, self.noise_dim)


def save_steering(path: str, net: SteeringNet, extra: Optional[dict] = None):
    torch.save({"kwargs": net.kwargs, "state_dict": net.state_dict(), "extra": extra or {}}, path)


def load_steering(path: str, device="cpu") -> SteeringNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = SteeringNet(**ckpt["kwargs"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    return net


# ============================================================================================
# Shared: samples I/O and batch sampling
# ============================================================================================
SAMPLE_FIELDS = ("features", "target", "kind", "inv_mse", "task_id", "round")


def write_samples_h5(path: str, samples: Dict[str, np.ndarray], meta: Optional[dict] = None):
    with h5py.File(path, "w") as f:
        for k in SAMPLE_FIELDS:
            f.create_dataset(k, data=samples[k], **({"compression": "gzip"} if samples[k].ndim >= 2 else {}))
        f.attrs["info"] = json.dumps(meta or {})


def read_samples_h5(path: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in SAMPLE_FIELDS}


def concat_samples(parts: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    parts = [p for p in parts if p is not None and len(p["kind"])]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts], 0) for k in SAMPLE_FIELDS}


class EpochPoolSampler:
    """Draws from each pool in shuffled passes (every index once per pass) instead of i.i.d.

    One cursor per pool key; a pool reshuffles when its pass runs out, so pools mixed at fixed ratios
    each cycle at their own rate. A pool whose size changed starts a fresh pass. Make one per training
    phase so every phase starts fresh passes (train_hitl.py's ``hitl.epoch_sampling``).
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._passes: Dict[str, Tuple[np.ndarray, int]] = {}

    def __call__(self, key: str, idx: np.ndarray, k: int) -> np.ndarray:
        perm, pos = self._passes.get(key, (None, 0))
        if perm is None or len(perm) != len(idx):
            perm, pos = self.rng.permutation(idx), 0
        out = []
        while k > 0:
            if pos == len(perm):
                perm, pos = self.rng.permutation(idx), 0
            take = min(k, len(perm) - pos)
            out.append(perm[pos:pos + take])
            pos, k = pos + take, k - take
        self._passes[key] = (perm, pos)
        return np.concatenate(out)


def sample_batch_indices(
    kind: np.ndarray,
    rounds: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    human_ratio: Optional[float] = 0.5,
    recent_ratio: Optional[float] = None,
    offline_ratio: Optional[float] = None,
    epoch_sampler: Optional[EpochPoolSampler] = None,
) -> np.ndarray:
    """Batch indices over a sample set where ``rounds < 0`` marks offline anchors.

    ``offline_ratio`` of the batch comes uniformly from the anchors (null => their share of all
    samples). The online rest takes ``human_ratio`` from human samples (the paper's dual buffer; uniform
    when either kind is missing) and, within each kind, ``recent_ratio`` from the newest round (uniform
    when null or only one round exists). ``epoch_sampler`` swaps the i.i.d. draws within each pool for
    shuffled passes; the per-pool counts are unchanged.
    """

    def _pick(key: str, idx: np.ndarray, k: int) -> np.ndarray:
        return rng.choice(idx, k) if epoch_sampler is None else epoch_sampler(key, idx, k)

    offline = rounds < 0
    off_idx, on_idx = np.flatnonzero(offline), np.flatnonzero(~offline)
    if len(off_idx) and len(on_idx):
        share = len(off_idx) / len(rounds) if offline_ratio is None else float(offline_ratio)
        k_off = int(round(batch_size * share))
    else:
        k_off = batch_size if len(off_idx) else 0
    parts = [_pick("offline", off_idx, k_off)] if k_off > 0 else []
    k_on = batch_size - k_off
    if k_on <= 0 or not len(on_idx):
        return np.concatenate(parts)
    newest_round = rounds[on_idx].max()

    def _draw(name: str, idx: np.ndarray, k: int) -> List[np.ndarray]:
        if k <= 0 or len(idx) == 0:
            return []
        if recent_ratio is not None:
            newest = rounds[idx] == newest_round
            if newest.any() and (~newest).any():
                k_new = int(round(k * float(recent_ratio)))
                return [_pick(f"{name}_newest", idx[newest], k_new), _pick(f"{name}_older", idx[~newest], k - k_new)]
        return [_pick(name, idx, k)]

    human_idx = on_idx[kind[on_idx] == KIND_HUMAN]
    policy_idx = on_idx[kind[on_idx] == KIND_POLICY]
    if human_ratio is None or len(human_idx) == 0 or len(policy_idx) == 0:
        parts += _draw("online", on_idx, k_on)
    else:
        k_h = int(round(k_on * float(human_ratio)))
        parts += _draw("human", human_idx, k_h) + _draw("policy", policy_idx, k_on - k_h)
    return np.concatenate(parts)


# ============================================================================================
# pi0 / LIBERO: env-space <-> pi0 model-space actions
# ============================================================================================
# pi0 obs key -> flattened HDF5 dataset name under /data/demo_i/obs (same as collect_hitl_libero_pi0.py).
OBS_KEY_TO_H5 = {PI0_IMAGE_KEY: "image", PI0_WRIST_KEY: "wrist_image", PI0_STATE_KEY: "state"}


class ActionNormalizer:
    """Maps env-space action chunks into pi0's normalized, zero-padded model action space.

    Mirrors the policy's own transforms: Normalize (quantile for pi0.5, z-score for pi0) on the raw
    ``env_dim`` action, THEN zero-pad to ``model_dim`` -- so the padded dims are literal 0, exactly
    what the model saw in training. No DeltaActions: pi05_libero has ``extra_delta_transform=False``.
    """

    def __init__(self, action_stats, use_quantiles: bool, horizon: int, env_dim: int = 7, model_dim: int = 32):
        self.horizon, self.env_dim, self.model_dim = int(horizon), int(env_dim), int(model_dim)
        self.use_quantiles = bool(use_quantiles)
        if self.use_quantiles:
            self.lo = np.asarray(action_stats.q01, dtype=np.float32)[: self.env_dim]
            self.hi = np.asarray(action_stats.q99, dtype=np.float32)[: self.env_dim]
        else:
            self.mean = np.asarray(action_stats.mean, dtype=np.float32)[: self.env_dim]
            self.std = np.asarray(action_stats.std, dtype=np.float32)[: self.env_dim]

    @classmethod
    def from_policy(cls, policy, horizon: int, model_dim: int, env_dim: int = 7):
        """Read the action norm stats straight out of the policy's input ``Normalize`` transform."""
        from openpi import transforms as _transforms

        for t in getattr(policy._input_transform, "transforms", []):
            if isinstance(t, _transforms.Normalize) and t.norm_stats is not None:
                return cls(t.norm_stats["actions"], t.use_quantiles, horizon, env_dim, model_dim)
        raise ValueError("Could not find a Normalize transform with action stats on the pi0 policy.")

    def env_to_model(self, chunk: np.ndarray):
        """``chunk [T, env_dim]`` (T <= horizon) -> ``([horizon, model_dim] target, [horizon] real-row mask)``.

        A chunk shorter than the horizon is filled by repeating its last row (as the reference does);
        the mask marks the real rows so inversion quality is judged only on them.
        """
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1, self.env_dim)
        n = chunk.shape[0]
        if not 1 <= n <= self.horizon:
            raise ValueError(f"chunk length must be in [1, {self.horizon}], got {n}")
        if self.use_quantiles:
            normed = (chunk - self.lo) / (self.hi - self.lo + 1e-6) * 2.0 - 1.0
        else:
            normed = (chunk - self.mean) / (self.std + 1e-6)
        out = np.zeros((self.horizon, self.model_dim), dtype=np.float32)
        out[:n, : self.env_dim] = normed
        out[n:, : self.env_dim] = normed[-1]
        mask = np.zeros(self.horizon, dtype=bool)
        mask[:n] = True
        return out, mask

    def model_to_env(self, x: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`env_to_model` on the first ``env_dim`` dims (``[..., model_dim] -> [..., env_dim]``)."""
        x = np.asarray(x, dtype=np.float32)[..., : self.env_dim]
        if self.use_quantiles:
            return (x + 1.0) / 2.0 * (self.hi - self.lo + 1e-6) + self.lo
        return x * (self.std + 1e-6) + self.mean


# ============================================================================================
# pi0 / LIBERO: flow operations (features, steered sampling, inversion)
# ============================================================================================
class Pi0FlowOps:
    """JAX-side operations on a frozen openpi ``Policy`` (``Pi0Wrapper.policy``).

    Works on ``policy._model`` directly (``nnx.split`` shares its weights; nothing is copied).
    ``Pi0Wrapper.infer(noise=...)`` / ``.get_features`` are not used: they need ``action_horizon`` /
    ``get_prefix_rep`` on the Policy, which the current openpi fork does not have.

    Batched calls are padded to ``batch_size`` so each jitted function compiles once.
    """

    def __init__(self, policy, *, num_steps: int = 10, fp_iters: int = 5, batch_size: int = 8, state_dim: int = 8):
        import flax.nnx as nnx

        if getattr(policy, "_is_pytorch_model", False):
            raise NotImplementedError("FlowDAgger needs the JAX openpi model (got a PyTorch checkpoint).")
        self.policy = policy
        model = policy._model
        self.horizon = int(model.action_horizon)
        self.model_dim = int(model.action_dim)
        self.num_steps, self.fp_iters = int(num_steps), int(fp_iters)
        self.batch_size, self.state_dim = int(batch_size), int(state_dim)
        self._graphdef, self._state = nnx.split(model)
        self._features_fn = _build_features_fn(self._graphdef)
        self._invert_fn = _build_invert_fn(self._graphdef, self.num_steps, self.fp_iters)

    # ---- inputs ----
    def _model_obs(self, obs_list: Sequence[dict]):
        """Raw pi0 obs dicts (with ``prompt``) -> batched ``Observation`` via the policy's input transforms."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        inputs = [self.policy._input_transform(jax.tree.map(lambda x: x, dict(o))) for o in obs_list]
        batched = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *inputs)
        return _model.Observation.from_dict(batched)

    def _padded_batches(self, n: int, desc: Optional[str] = None):
        """Yield ``(real indices, indices padded to batch_size)``; ``desc`` shows a tqdm bar over samples."""
        bar = tqdm(total=n, desc=desc, unit="sample", leave=False) if desc else None
        for start in range(0, n, self.batch_size):
            idx = list(range(start, min(start + self.batch_size, n)))
            yield idx, idx + [idx[-1]] * (self.batch_size - len(idx))
            if bar is not None:
                bar.update(len(idx))
        if bar is not None:
            bar.close()

    # ---- features for the steering net ----
    def features(self, obs_list: Sequence[dict], progress: Optional[str] = None) -> np.ndarray:
        """``[B, feature_dim]`` float32: masked mean of pi0's prefix (image + language) hidden states,
        concatenated with the first ``state_dim`` dims of the normalized state (paper's ``vlm_pi0``)."""
        out = []
        for idx, padded in self._padded_batches(len(obs_list), progress):
            pooled, state = self._features_fn(self._state, self._model_obs([obs_list[i] for i in padded]))
            state = np.asarray(state, np.float32)[:, : self.state_dim]
            feats = np.concatenate([np.asarray(pooled, np.float32), state], 1)
            out.append(feats[: len(idx)])
        return np.concatenate(out, 0)

    # ---- steered sampling ----
    def sample(self, obs: dict, noise: np.ndarray) -> np.ndarray:
        """Env-space action chunk ``[horizon, env_dim]`` from pi0 started at ``noise [horizon, model_dim]``."""
        result = self.policy.infer(dict(obs), noise=np.asarray(noise, dtype=np.float32))
        return np.asarray(result["actions"], dtype=np.float32)

    # ---- inversion ----
    def invert(
        self, obs_list: Sequence[dict], targets: np.ndarray, masks: np.ndarray, env_dim: int = 7,
        progress: Optional[str] = None,
    ):
        """Invert model-space target chunks ``[B, H, model_dim]`` to noise.

        Returns ``(w [B, H, model_dim], mse [B])``. ``mse`` re-integrates pi0's sampler from ``w`` and
        compares to the target in normalized space, over the real rows (``masks``) and first ``env_dim`` dims.
        """
        import jax.numpy as jnp

        targets = np.asarray(targets, dtype=np.float32)
        ws, mses = [], []
        for idx, padded in self._padded_batches(len(obs_list), progress):
            obs = self._model_obs([obs_list[i] for i in padded])
            w, recon = self._invert_fn(self._state, obs, jnp.asarray(targets[padded]))
            w, recon = np.asarray(w, np.float32)[: len(idx)], np.asarray(recon, np.float32)[: len(idx)]
            err = np.square(recon[..., :env_dim] - targets[idx][..., :env_dim]).mean(-1)  # [b, H]
            m = masks[idx].astype(np.float32)
            ws.append(w)
            mses.append((err * m).sum(1) / np.maximum(m.sum(1), 1.0))
        return np.concatenate(ws, 0), np.concatenate(mses, 0)


def _build_features_fn(graphdef):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    @jax.jit
    def features_fn(state, observation):
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
        m = prefix_mask.astype(jnp.float32)[..., None]
        pooled = (prefix_out.astype(jnp.float32) * m).sum(1) / jnp.maximum(m.sum(1), 1.0)
        return pooled, observation.state

    return features_fn


def _build_invert_fn(graphdef, num_steps: int, fp_iters: int):
    """Jitted ``(state, observation, x_action [B,H,D]) -> (w, recon)``: per-step fixed-point inversion of the
    exact discrete Euler map, then the forward map from ``w`` (``recon``) to measure inversion error.

    The forward step from time t to t-dt is ``x_{t-dt} = x_t - dt * v(x_t, t)`` (dt = 1/num_steps). To invert
    it, solve ``x_t = x_{t-dt} + dt * v(x_t, t)`` per step: seed with an explicit reverse-Euler step, then
    refine with ``fp_iters`` fixed-point iterations (contraction ~0.1-0.3 per iteration at dt=0.1).
    """
    import einops
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    dt = 1.0 / num_steps

    @jax.jit
    def invert_fn(state, observation, x_action):
        model = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)
        batch = observation.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)

        def velocity(x_t, t):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(t, (batch,))
            )
            suffix_attn = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn, positions=pos, kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return model.action_out_proj(suffix_out[:, -model.action_horizon :]).astype(jnp.float32)

        def inv_step(k, x_prev):  # x at time k*dt -> x at time (k+1)*dt
            t_next = (k + 1) * dt
            x_next = x_prev + dt * velocity(x_prev, k * dt)
            return jax.lax.fori_loop(0, fp_iters, lambda _, x_est: x_prev + dt * velocity(x_est, t_next), x_next)

        w = jax.lax.fori_loop(0, num_steps, inv_step, x_action.astype(jnp.float32))
        recon = jax.lax.fori_loop(0, num_steps, lambda k, x: x - dt * velocity(x, 1.0 - k * dt), w)
        return w, recon

    return invert_fn


# ============================================================================================
# pi0 / LIBERO: a HITL worker that steers pi0 and records the noise it used
# ============================================================================================
class FlowDAggerPi0LiberoHitlWorker(Pi0LiberoHitlWorker):
    """``Pi0LiberoHitlWorker`` whose autonomous chunks start from ``noise_fn(obs)`` instead of N(0, I).

    Every stored step is tagged with ``chunk_id`` (-1 for human steps) and the ``noise [H, model_dim]``
    its chunk was sampled from (zeros for human steps), which :func:`build_samples` turns into BC targets.
    """

    def __init__(self, *args, flow: Pi0FlowOps, noise_fn: Callable[[dict], np.ndarray], **kwargs):
        self.flow, self.noise_fn = flow, noise_fn
        self._chunk_id, self._chunk_noise = -1, None
        super().__init__(*args, **kwargs)

    def _pi0_chunk(self, obs) -> np.ndarray:
        from robometer_policy_learning.utils.pi0_hitl_utils import _prompt_str

        obs = dict(obs)
        obs["prompt"] = _prompt_str(obs)
        noise = np.asarray(self.noise_fn(obs), dtype=np.float32)
        self._chunk_id += 1
        self._chunk_noise = noise
        return self.flow.sample(obs, noise).reshape(-1, self.action_dim)

    def _step_extras(self, label) -> dict:
        if label == INTERVENTION_LABEL:
            return {"chunk_id": -1, "noise": np.zeros((self.flow.horizon, self.flow.model_dim), np.float32)}
        return {"chunk_id": self._chunk_id, "noise": self._chunk_noise}


# ============================================================================================
# pi0 / LIBERO: rollouts I/O
# ============================================================================================
def write_rollouts_h5(episodes: List[List[Dict]], path: str, meta: dict):
    """Write collected episodes in the ``collect_hitl_libero_pi0.py`` schema, plus per-step ``noise`` and
    ``chunk_id`` so a run's raw data can be re-imported (``flowdagger.import_hdf5s``) without loss."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        total = 0
        for i, trs in enumerate(episodes):
            g = data.create_group(f"demo_{i}")
            g.create_dataset("actions", data=np.stack([t["action"].reshape(-1) for t in trs]).astype(np.float32))
            g.create_dataset("rewards", data=np.asarray([t["reward"] for t in trs], dtype=np.float32))
            g.create_dataset("dones", data=np.asarray([float(t["done"] or t["truncated"]) for t in trs], np.float32))
            g.create_dataset("intervention", data=np.asarray([int(t["intervention"]) for t in trs], dtype=np.int64))
            if "chunk_id" in trs[0]:
                g.create_dataset("chunk_id", data=np.asarray([int(t["chunk_id"]) for t in trs], dtype=np.int64))
                noise = np.stack([t["noise"] for t in trs]).astype(np.float32)
                g.create_dataset("noise", data=noise, compression="gzip")
            obs = g.create_group("obs")
            for key, name in OBS_KEY_TO_H5.items():
                if key in trs[0]["obs"]:
                    seq = np.stack([np.asarray(t["obs"][key]) for t in trs])
                    obs.create_dataset(name, data=seq, **({"compression": "gzip"} if seq.ndim >= 3 else {}))
            g.attrs["num_samples"] = len(trs)
            g.attrs["prompt"] = str(trs[0].get("prompt", ""))
            total += len(trs)
        data.attrs["total"] = total
        f.create_group("meta").attrs["info"] = json.dumps(meta)


def read_rollouts_h5(path: str) -> List[Dict[str, np.ndarray]]:
    """Load a HITL rollouts HDF5 (this module's or ``collect_hitl_libero_pi0.py``'s) as per-episode arrays."""
    episodes = []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for name in demos:
            g = f["data"][name]
            ep = {k: g[k][()] for k in ("actions", "intervention", "chunk_id", "noise") if k in g}
            ep["obs"] = {key: g["obs"][h5][()] for key, h5 in OBS_KEY_TO_H5.items() if h5 in g["obs"]}
            ep["prompt"] = str(g.attrs.get("prompt", ""))
            episodes.append(ep)
    return episodes


def episode_arrays(trs: List[Dict]) -> Dict[str, np.ndarray]:
    """Per-step transition dicts (as the worker collects them) -> the per-episode arrays of :func:`read_rollouts_h5`."""
    ep = {
        "actions": np.stack([np.asarray(t["action"], np.float32).reshape(-1) for t in trs]),
        "intervention": np.asarray([int(t["intervention"]) for t in trs], dtype=np.int64),
        "obs": {k: np.stack([np.asarray(t["obs"][k]) for t in trs]) for k in trs[0]["obs"]},
        "prompt": str(trs[0].get("prompt", "")),
    }
    if "chunk_id" in trs[0]:
        ep["chunk_id"] = np.asarray([int(t["chunk_id"]) for t in trs], dtype=np.int64)
        ep["noise"] = np.stack([t["noise"] for t in trs]).astype(np.float32)
    return ep


# ============================================================================================
# robomimic: steered FlowMatchingActor, inversion, HITL worker
# ============================================================================================
STEER_MODES = ("steer", "randn", "zeros")


class SteeredFlowActor(FlowMatchingActor):
    """A frozen FlowMatchingActor whose initial noise comes from a steering net.

    Built in place from a loaded actor with :meth:`wrap` (no copy). ``steer_mode``: ``steer`` = the
    steering net's noise; ``randn`` = N(0, I), i.e. the plain base policy (before any training);
    ``zeros`` = the noise mode. Still a FlowMatchingActor, so ``act`` / eval workers / the
    flow-MILE intervention criteria all work on it unchanged.
    """

    @classmethod
    def wrap(cls, actor: FlowMatchingActor, steering: SteeringNet, steer_mode: str = "randn") -> "SteeredFlowActor":
        if not isinstance(actor, FlowMatchingActor):
            raise TypeError(f"FlowDAgger needs a FlowMatchingActor, got {type(actor).__name__}.")
        for p in actor.parameters():
            p.requires_grad_(False)
        actor.__class__ = cls
        actor.steering = steering
        actor.set_steer_mode(steer_mode)
        actor.last_noise = None
        return actor

    def set_steer_mode(self, mode: str):
        if mode not in STEER_MODES:
            raise ValueError(f"steer_mode must be one of {STEER_MODES}, got {mode!r}")
        self.steer_mode = mode

    def initial_noise(self, global_cond: torch.Tensor) -> torch.Tensor:
        b = global_cond.shape[0]
        if self.steer_mode == "steer":
            return self.steering(global_cond)
        if self.steer_mode == "zeros":
            return torch.zeros(b, self.horizon, self.action_dim, device=global_cond.device)
        return torch.randn(b, self.horizon, self.action_dim, device=global_cond.device)

    def integrate(self, x: torch.Tensor, global_cond: torch.Tensor, num_inference_steps: Optional[int] = None):
        """The same Euler loop (and clip) as ``FlowMatchingActor.sample_actions``, from a given noise."""
        steps = int(num_inference_steps) if num_inference_steps is not None else self.num_inference_steps
        dt = 1.0 / steps
        for i in range(steps):
            t_batch = torch.full((x.shape[0],), i * dt, device=x.device, dtype=torch.float32)
            x = x + dt * self.predict_velocity(x, t_batch, global_cond)
        return self._maybe_clip(x)

    @torch.no_grad()
    def sample_actions(self, obs, num_inference_steps: Optional[int] = None) -> torch.Tensor:
        global_cond = self.encode_obs(obs)
        x0 = self.initial_noise(global_cond)
        self.last_noise = x0.detach().float().cpu().numpy()  # [B, H, A], read by the HITL worker
        return self.integrate(x0, global_cond, num_inference_steps)

    @torch.no_grad()
    def sample_actions_batch(self, obs, num_samples: int, num_inference_steps: Optional[int] = None):
        """Steered: the policy is deterministic, so all ``num_samples`` chunks are the same."""
        if self.steer_mode == "randn":
            return super().sample_actions_batch(obs, num_samples, num_inference_steps)
        global_cond = self.encode_obs(obs)
        x = self.integrate(self.initial_noise(global_cond), global_cond, num_inference_steps)
        return x.unsqueeze(0).expand(int(num_samples), *x.shape).contiguous()


@torch.no_grad()
def invert_chunks(
    actor: SteeredFlowActor, global_cond: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, fp_iters: int = 5
):
    """Noise ``w`` whose Euler integration gives ``target`` ``[B, H, A]`` (normalized actions).

    Walks the sampler backwards: for k = N-1..0 solve ``x_k = x_{k+1} - dt * v(x_k, k*dt)`` by fixed-point
    iteration seeded with ``v`` at ``x_{k+1}``. Returns ``(w, mse)``, ``mse`` = per-row reconstruction
    error of re-integrating ``w``, over the real (``mask``) rows of the chunk.
    """
    steps = actor.num_inference_steps
    dt = 1.0 / steps
    x = target
    for k in reversed(range(steps)):
        t_batch = torch.full((x.shape[0],), k * dt, device=x.device, dtype=torch.float32)
        y = x - dt * actor.predict_velocity(x, t_batch, global_cond)
        for _ in range(int(fp_iters)):
            y = x - dt * actor.predict_velocity(y, t_batch, global_cond)
        x = y
    recon = actor.integrate(x, global_cond)
    m = mask.to(target.dtype).unsqueeze(-1)
    mse = (((recon - target) ** 2) * m).sum((1, 2)) / (m.sum((1, 2)) * target.shape[-1]).clamp_min(1.0)
    return x, mse


class FlowDAggerHitlWorker(HitlRolloutWorker):
    """Robomimic ``HitlRolloutWorker`` that tags every policy step with its chunk id and the noise behind that chunk.

    Human steps get ``chunk_id=-1`` and zero noise. ``self.actor`` must be a :class:`SteeredFlowActor`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunk_id = -1
        self._chunk_noise = None

    def _policy_action(self, obs_t, st):
        a = super()._policy_action(obs_t, st)
        if st["pos"] == 1:  # the parent just replanned (pos is reset to 0, then advanced)
            self._chunk_id += 1
            self._chunk_noise = np.asarray(self.actor.last_noise[0], dtype=np.float32)
        return a

    def _step_extras(self, label) -> Dict:
        if label == ROLLOUT_LABEL and self._chunk_noise is not None:
            return {"chunk_id": self._chunk_id, "noise": self._chunk_noise}
        return {"chunk_id": -1, "noise": np.zeros((self.actor.horizon, self.action_dim), np.float32)}


# ============================================================================================
# robomimic: replay buffer -> training samples
# ============================================================================================
def buffer_episodes(buffer) -> "OrderedDict[str, list]":
    """Transitions grouped by ``episode_id`` (first-seen order, as ``collect_hitl_rollout._write_h5``)."""
    episodes = OrderedDict()
    for t in buffer.get_all_transitions():
        if t is not None:
            episodes.setdefault(t.episode_id, []).append(t)
    for eid in episodes:
        episodes[eid].sort(key=lambda x: int(x.step_in_episode))
    return episodes


def relabel(buffer, label: int):
    """Set every transition's ``info['intervention']`` to ``label`` and drop recorded chunk noise.

    Offline demos are relabelled human so that every step starts an inverted window.
    """
    for t in buffer.get_all_transitions():
        if t is not None:
            t.info = {k: v for k, v in (t.info or {}).items() if k not in ("noise", "chunk_id")}
            t.info["intervention"] = int(label)


def attach_recorded_noise(path: str, buffer) -> int:
    """Copy each demo's ``noise`` / ``chunk_id`` (files written by train_flowdagger_robomimic.py) onto the
    transitions ``load_precollected_hitl`` loaded from it (``episode_id`` = demo name, step = index).

    Returns the number of demos that carried noise; the others yield human samples only.
    """
    by_ep = buffer_episodes(buffer)
    n_demos = 0
    with h5py.File(path, "r") as f:
        for demo, g in f["data"].items():
            if "noise" not in g or "chunk_id" not in g or demo not in by_ep:
                continue
            noise, chunk_id = np.asarray(g["noise"], np.float32), np.asarray(g["chunk_id"], np.int64)
            for t in by_ep[demo]:
                k = int(t.step_in_episode)
                t.info["noise"], t.info["chunk_id"] = noise[k], int(chunk_id[k])
            n_demos += 1
    return n_demos


def _buffer_episode_arrays(trs: list, horizon: int, action_dim: int) -> Dict[str, np.ndarray]:
    info = [t.info or {} for t in trs]
    ep = {
        "actions": np.stack([np.asarray(t.action, np.float32).reshape(-1) for t in trs]),
        "intervention": np.asarray([int(round(float(i.get("intervention", 0)))) for i in info]),
    }
    if all("noise" in i and "chunk_id" in i for i in info):
        ep["chunk_id"] = np.asarray([int(i["chunk_id"]) for i in info])
        ep["noise"] = np.stack([np.asarray(i["noise"], np.float32).reshape(horizon, action_dim) for i in info])
    return ep


@torch.no_grad()
def build_round_samples(
    buffer,
    actor: SteeredFlowActor,
    device,
    *,
    use_policy: bool = True,
    drop_pre_takeover_chunk: bool = True,
    human_window_stride: Optional[int] = 1,
    min_human_window: int = 1,
    fp_iters: int = 5,
    mse_threshold: Optional[float] = None,
    round_idx: int = 0,
    batch_size: int = 512,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, float]]:
    """``buffer``'s episodes -> ``(samples, stats)``; samples has ``SAMPLE_FIELDS``.

    Policy samples keep their recorded noise; human windows (``build_samples``) are normalized, padded
    to the horizon by repeating the last action, inverted, and dropped above ``mse_threshold``.
    Features are the frozen actor's ``encode_obs`` of each sample's start obs, through the buffer's own
    collation so they match what ``act()`` sees.
    """
    H, A = actor.horizon, actor.action_dim
    specs = []  # (transition, spec)
    for trs in buffer_episodes(buffer).values():
        ep = _buffer_episode_arrays(trs, H, A)
        for s in build_samples(
            ep, horizon=H, use_policy=use_policy, drop_pre_takeover_chunk=drop_pre_takeover_chunk,
            human_window_stride=human_window_stride, min_human_window=min_human_window,
        ):
            specs.append((trs[s["t"]], s))
    if not specs:
        return None, {}

    was_training = actor.training
    actor.eval()
    feats, targets, kinds, mses = [], [], [], []
    for start in range(0, len(specs), batch_size):
        mini = specs[start : start + batch_size]
        batched = buffer._convert_batch_to_tensors(buffer._batch_transitions([t for t, _ in mini]), device=device)
        cond = actor.encode_obs(batched["obs"]).float()
        target = torch.zeros(len(mini), H, A, device=device)
        mask = torch.zeros(len(mini), H, dtype=torch.bool, device=device)
        human = torch.zeros(len(mini), dtype=torch.bool, device=device)
        for j, (_, s) in enumerate(mini):
            if s["kind"] == KIND_POLICY:
                target[j] = torch.as_tensor(s["target"], device=device)
                mask[j] = True
            else:
                a = actor.normalize_action(torch.as_tensor(s["actions"], device=device))
                n = len(a)
                target[j, :n], target[j, n:] = a, a[-1]
                mask[j, :n] = True
                human[j] = True
        mse = torch.zeros(len(mini), device=device)
        if human.any():
            w, m = invert_chunks(actor, cond[human], target[human], mask[human], fp_iters=fp_iters)
            target[human], mse[human] = w, m
        feats.append(cond.cpu().numpy())
        targets.append(target.cpu().numpy())
        kinds.append(np.where(human.cpu().numpy(), KIND_HUMAN, KIND_POLICY))
        mses.append(mse.cpu().numpy())
    if was_training:
        actor.train()

    out = {
        "features": np.concatenate(feats).astype(np.float32),
        "target": np.concatenate(targets).astype(np.float32),
        "kind": np.concatenate(kinds).astype(np.int64),
        "inv_mse": np.concatenate(mses).astype(np.float32),
    }
    n = len(out["kind"])
    out["task_id"] = np.zeros(n, np.int64)
    out["round"] = np.full(n, int(round_idx), np.int64)
    is_h = out["kind"] == KIND_HUMAN
    stats = {
        "num_policy": int((~is_h).sum()),
        "num_human": int(is_h.sum()),
        "inv_mse_median": float(np.median(out["inv_mse"][is_h])) if is_h.any() else 0.0,
        "inv_mse_max": float(out["inv_mse"][is_h].max()) if is_h.any() else 0.0,
        "human_abs_w_p99": float(np.percentile(np.abs(out["target"][is_h]), 99)) if is_h.any() else 0.0,
    }
    keep = ~(is_h & (out["inv_mse"] > mse_threshold)) if mse_threshold is not None else np.ones(n, bool)
    stats["num_dropped"] = int((~keep).sum())
    return {k: v[keep] for k, v in out.items()}, stats
