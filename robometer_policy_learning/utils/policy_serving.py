"""Shared plumbing for serving a trained policy to a real robot.

The single job of this module is to make **inference preprocessing byte-identical to training
preprocessing**. A policy trained by ``scripts/train_supervised.py`` on a converted LeRobot dataset
depends on four things that live nowhere in the actor's weights:

1. which robot camera feeds which observation key (``scripts/convert_lerobot_to_h5.py --camera-map``);
2. the image size and resize filter the frames were converted with;
3. the z-score statistics for low-dim keys (``training.normalize_lowdim_obs``), which the training
   buffer computes at load time and every eval script re-derives by rebuilding the buffer;
4. how many actions of each predicted chunk are executed before replanning
   (``training.n_action_steps``), which the training-time env wrapper applies but a robot does not.

Get any of them wrong and the policy silently sees a different input distribution than it trained
on. :class:`PolicyMetadata` captures all four next to the checkpoint at training time, and
:class:`RobometerPolicy` is the one implementation of that preprocessing, used by the server.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

# torch is imported lazily inside RobometerPolicy: the robot-side client only needs
# PolicyMetadata + resize_image, and the raiden environment it runs in may have no torch at all.

METADATA_FILENAME = "policy_metadata.json"

# Same keyword heuristic H5ReplayBuffer / the actor featurizers use to classify an obs key.
IMAGE_KEY_KEYWORDS = ("image", "rgb", "camera", "cam")


def is_image_key(key: str) -> bool:
    return any(kw in key.lower() for kw in IMAGE_KEY_KEYWORDS)


def resize_image(image: np.ndarray, size: Optional[int]) -> np.ndarray:
    """Resize an HWC uint8 frame to ``size x size``, matching the dataset converter exactly.

    ``scripts/convert_lerobot_to_h5.py`` downscales camera frames with ``cv2.INTER_AREA``; using a
    different filter here would feed the policy subtly different pixel statistics than it trained
    on. A no-op when the frame already has the target size (so it is safe to call on both the robot
    and the server).
    """
    if size is None or image.shape[:2] == (size, size):
        return image
    import cv2

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


@dataclasses.dataclass
class PolicyMetadata:
    """Everything an inference client/server needs that is not inside the actor's weights."""

    # Observation contract
    image_keys: List[str] = dataclasses.field(default_factory=list)
    lowdim_keys: List[str] = dataclasses.field(default_factory=list)
    remove_obs_keys: List[str] = dataclasses.field(default_factory=list)
    image_size: Optional[int] = None
    # Robot camera name -> observation key, inverted from the dataset's conversion camera_map, so a
    # client can name its cameras the way the robot does.
    camera_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    # key -> {"mean": [...], "std": [...]}; empty when training.normalize_lowdim_obs was false.
    lowdim_obs_stats: Dict[str, Dict[str, List[float]]] = dataclasses.field(default_factory=dict)

    # Action contract
    action_dim: int = 0
    action_low: List[float] = dataclasses.field(default_factory=list)
    action_high: List[float] = dataclasses.field(default_factory=list)
    # Chunk the actor predicts, and how many of it training executed before replanning.
    chunk_size: int = 1
    n_action_steps: int = 1
    # "absolute" | "delta" -- from the converted dataset's /meta. "delta" means the robot must add
    # its current state back to the returned action.
    action_mode: str = "absolute"

    # Provenance (not consumed by inference; for debugging a mismatch)
    alg: str = ""
    actor_class: str = ""
    dataset_path: str = ""
    prompt: str = ""
    normalize_lowdim_obs: bool = False

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyMetadata":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, METADATA_FILENAME)
        with open(path, "w") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def load(cls, directory: str) -> Optional["PolicyMetadata"]:
        """Load metadata from ``directory``, or from its parent (checkpoints/<step> -> checkpoints)."""
        for candidate in (directory, os.path.dirname(os.path.abspath(directory))):
            path = os.path.join(candidate, METADATA_FILENAME)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return cls.from_dict(json.load(f))
        return None

    def stats_as_arrays(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            key: {"mean": np.asarray(st["mean"], np.float32), "std": np.asarray(st["std"], np.float32)}
            for key, st in self.lowdim_obs_stats.items()
        }

    def describe(self) -> str:
        return (
            f"alg={self.alg} actor={self.actor_class} action_dim={self.action_dim} "
            f"chunk_size={self.chunk_size} n_action_steps={self.n_action_steps} "
            f"action_mode={self.action_mode} image_size={self.image_size} "
            f"images={self.image_keys} lowdim={self.lowdim_keys} "
            f"normalize_lowdim_obs={self.normalize_lowdim_obs}"
        )


def build_policy_metadata(
    *,
    cfg,
    actor,
    buffer,
    observation_space,
    action_space,
    dataset_path: str,
) -> PolicyMetadata:
    """Assemble :class:`PolicyMetadata` from a finished training setup.

    ``camera_map``/``action_mode``/``prompt`` are recovered from the converted dataset's ``/meta``
    and first demo when available, so a robot client can map its own camera names without the user
    restating the conversion arguments.
    """
    from omegaconf import OmegaConf

    keys = [k for k in observation_space.spaces if k not in (actor.remove_obs_keys or [])]
    image_keys = [k for k in keys if is_image_key(k)]
    lowdim_keys = [k for k in keys if not is_image_key(k)]

    image_size = None
    for key in image_keys:
        shape = observation_space.spaces[key].shape
        if len(shape) >= 2:
            image_size = int(shape[0])
            break

    stats = {}
    for key, st in (getattr(buffer, "lowdim_obs_stats", None) or {}).items():
        stats[key] = {
            "mean": np.asarray(st["mean"], np.float32).ravel().tolist(),
            "std": np.asarray(st["std"], np.float32).ravel().tolist(),
        }

    camera_map: Dict[str, str] = {}
    action_mode = "absolute"
    prompt = ""
    try:
        import h5py

        with h5py.File(dataset_path, "r") as f:
            if "meta" in f and "info" in f["meta"].attrs:
                info = json.loads(f["meta"].attrs["info"])
                action_mode = info.get("action_mode", "absolute")
                # conversion map is lerobot feature -> obs key; expose robot camera name -> obs key.
                for feature, obs_key in (info.get("camera_map") or {}).items():
                    camera_map[feature.rsplit(".", 1)[-1]] = obs_key
            demos = [k for k in f.get("data", {}) if k.startswith("demo")]
            if demos:
                demos.sort(key=lambda k: (len(k), k))
                prompt = str(f["data"][demos[0]].attrs.get("prompt", "") or "")
    except Exception:
        # Metadata provenance is best-effort: a dataset without /meta still yields a usable contract.
        pass

    chunk_size = int(OmegaConf.select(cfg, "training.chunk_size", default=None) or 1)
    n_action_steps = int(OmegaConf.select(cfg, "training.n_action_steps", default=1) or 1)

    return PolicyMetadata(
        image_keys=image_keys,
        lowdim_keys=lowdim_keys,
        remove_obs_keys=list(actor.remove_obs_keys or []),
        image_size=image_size,
        camera_map=camera_map,
        lowdim_obs_stats=stats,
        action_dim=int(np.prod(action_space.shape)),
        action_low=np.asarray(action_space.low, np.float32).ravel().tolist(),
        action_high=np.asarray(action_space.high, np.float32).ravel().tolist(),
        chunk_size=chunk_size,
        n_action_steps=min(n_action_steps, chunk_size),
        action_mode=action_mode,
        alg=str(OmegaConf.select(cfg, "alg.offline_alg_name", default="")),
        actor_class=type(actor).__name__,
        dataset_path=str(dataset_path),
        prompt=prompt,
        normalize_lowdim_obs=bool(OmegaConf.select(cfg, "training.normalize_lowdim_obs", default=False)),
    )


class RobometerPolicy:
    """Runs a trained actor on raw robot observations, applying the training-time preprocessing.

    Mirrors ``EvaluationWorker._prepare_obs`` -> ``actor.act(obs, deterministic=True)``: keys are
    renamed / resized, low-dim keys are z-scored with the training statistics, and the actor's own
    ``act()`` un-normalizes the action back into the environment's action space. The returned chunk
    is therefore directly commandable by the robot -- no further scaling.

    ``infer`` accepts either a plain observation dict or the ``{"method": "infer", "obs": {...}}``
    envelope that openpi-client's WebsocketClientPolicy wraps around it.
    """

    def __init__(self, actor, metadata: PolicyMetadata, device=None):
        import torch

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor.to(self.device).eval()
        self.metadata = metadata
        self._stats = {
            key: (
                torch.as_tensor(st["mean"], dtype=torch.float32, device=self.device),
                torch.as_tensor(st["std"], dtype=torch.float32, device=self.device),
            )
            for key, st in metadata.stats_as_arrays().items()
        }
        # Robot-side aliases accepted for each observation key: the key itself, the robot camera
        # name it was converted from, and the openpi-style "observation/<name>" spelling.
        self._aliases: Dict[str, str] = {}
        for obs_key in list(metadata.image_keys) + list(metadata.lowdim_keys):
            self._aliases[obs_key] = obs_key
        for cam_name, obs_key in (metadata.camera_map or {}).items():
            self._aliases[cam_name] = obs_key
            self._aliases[f"observation/{cam_name}"] = obs_key
        self._aliases.setdefault("observation/state", "state")

    def _resolve(self, obs: Dict[str, Any], wanted: str) -> Any:
        if wanted in obs:
            return obs[wanted]
        for incoming, mapped in self._aliases.items():
            if mapped == wanted and incoming in obs:
                return obs[incoming]
        raise KeyError(
            f"observation is missing {wanted!r}. Got keys {sorted(obs)}; the policy expects "
            f"images {self.metadata.image_keys} and low-dim {self.metadata.lowdim_keys} "
            f"(camera aliases: {self.metadata.camera_map})"
        )

    def preprocess(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Raw robot observation -> batched device tensors in the actor's expected format."""
        import torch

        prepared: Dict[str, Any] = {}

        for key in self.metadata.image_keys:
            frame = np.asarray(self._resolve(obs, key))
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
                frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
            if frame.dtype != np.uint8:
                # Training stored raw uint8 and the encoders divide by 255 themselves.
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            frame = resize_image(np.ascontiguousarray(frame), self.metadata.image_size)
            prepared[key] = torch.from_numpy(frame).to(self.device).unsqueeze(0)

        for key in self.metadata.lowdim_keys:
            value = np.asarray(self._resolve(obs, key), dtype=np.float32).reshape(-1)
            tensor = torch.from_numpy(value).to(self.device)
            if key in self._stats:
                mean, std = self._stats[key]
                tensor = (tensor - mean) / std
            prepared[key] = tensor.unsqueeze(0)

        return prepared

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``{"actions": [chunk_size, action_dim]}`` in the robot's action space."""
        import torch
        # openpi-client wraps observations in a method envelope; its own server does not unwrap it,
        # so accept both spellings rather than depending on which client version the robot runs.
        if isinstance(obs, dict) and "obs" in obs and isinstance(obs["obs"], dict):
            obs = obs["obs"]

        prepared = self.preprocess(obs)
        with torch.inference_mode():
            action, _ = self.actor.act(prepared, deterministic=True)
        action = action.detach().float().cpu().numpy()
        if action.ndim == 3:  # [1, chunk, dim]
            action = action[0]
        elif action.ndim == 2 and action.shape[0] == 1:  # [1, dim] (unchunked policy)
            action = action.reshape(1, -1)
        return {"actions": np.ascontiguousarray(action, dtype=np.float32)}

    def reset(self) -> None:
        """No-op: these actors are memoryless (chunk state lives in the client's broker)."""

    @property
    def server_metadata(self) -> Dict[str, Any]:
        """Handshake payload; the client configures its action horizon from this."""
        return dataclasses.asdict(self.metadata)
