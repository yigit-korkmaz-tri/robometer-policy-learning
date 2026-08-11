"""Synthesize gym observation/action spaces from an offline HDF5 dataset.

``build_actor_critic_models`` derives the actor's whole input structure from a gym
``observation_space``/``action_space``, which normally come from a simulator. Real-robot datasets
(e.g. LeRobot conversions) have no simulator, so the offline-only training path reads the shapes
straight out of the dataset instead.

This reads the HDF5 directly rather than going through ``H5ReplayBuffer``: the models are built
before the buffer exists, and only shapes/dtypes plus the ``/meta`` action bounds are needed.
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import h5py
import numpy as np

from robometer_policy_learning.utils.logging_compat import get_logger

logger = get_logger()

# Same modality heuristic H5ReplayBuffer uses, so a key classified as an image here is also read
# lazily as an image there.
IMAGE_KEY_KEYWORDS = ("image", "rgb", "camera", "cam")


def is_image_key(key: str, shape: Tuple[int, ...]) -> bool:
    return len(shape) >= 3 or any(kw in key.lower() for kw in IMAGE_KEY_KEYWORDS)


def widen_degenerate_bounds(
    min_action: np.ndarray, max_action: np.ndarray, eps: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure every action dim spans at least ``eps``.

    The buffer normalizes with ``(a - min) / (max - min)``, so a dim that never moves in the dataset
    would divide by zero and poison the whole batch with NaNs.
    """
    min_action = np.asarray(min_action, dtype=np.float32).copy()
    max_action = np.asarray(max_action, dtype=np.float32).copy()
    degenerate = np.flatnonzero((max_action - min_action) < eps)
    if degenerate.size:
        center = 0.5 * (min_action + max_action)
        min_action[degenerate] = center[degenerate] - eps / 2.0
        max_action[degenerate] = center[degenerate] + eps / 2.0
        logger.warning(f"Widened {degenerate.size} near-constant action dim(s) to span >= {eps}: {degenerate.tolist()}")
    return min_action, max_action


def _first_demo_name(data_grp: h5py.Group) -> str:
    demos = [k for k in data_grp.keys() if k.startswith("demo")]
    if not demos:
        raise ValueError("HDF5 /data group contains no demo_* groups")
    # demo_0, demo_1, ... -- sort numerically so "demo_10" doesn't come before "demo_2".
    demos.sort(key=lambda k: (len(k), k))
    return demos[0]


def build_spaces_from_h5(
    h5_path: str,
    remove_obs_keys: Optional[List[str]] = None,
    extra_obs_spaces: Optional[Dict[str, gym.Space]] = None,
    action_range_eps: float = 1e-3,
) -> Tuple[gym.spaces.Dict, gym.spaces.Box]:
    """Build ``(observation_space, action_space)`` from a robomimic-style HDF5.

    Args:
        h5_path: dataset written by e.g. ``scripts/convert_lerobot_to_h5.py``.
        remove_obs_keys: keys to leave out of the observation space (the same list the buffer drops).
        extra_obs_spaces: synthetic keys the buffer adds after loading (``dino_embedding``,
            ``language``); they must be present here or the actor's featurizer won't have an entry
            for them.
        action_range_eps: minimum per-dim action span (see :func:`widen_degenerate_bounds`).

    Returns:
        A ``spaces.Dict`` observation space (uint8 ``Box(0, 255, ...)`` for image keys, unbounded
        float32 ``Box`` otherwise) and a finite float32 ``Box`` action space taken from
        ``/meta/min_action`` and ``/meta/max_action`` (or from the first demo's actions).
    """
    remove = set(remove_obs_keys or [])
    obs_spaces: Dict[str, gym.Space] = {}

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{h5_path} has no /data group; expected a robomimic-style HDF5")
        data_grp = f["data"]
        demo = data_grp[_first_demo_name(data_grp)]
        if "obs" not in demo:
            raise ValueError(f"{h5_path}: {demo.name} has no obs group")

        for key, dataset in demo["obs"].items():
            if key in remove:
                continue
            shape = tuple(dataset.shape[1:])  # drop the leading time axis
            if is_image_key(key, shape):
                obs_spaces[key] = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            else:
                obs_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

        actions = demo["actions"]
        action_dim = int(actions.shape[1]) if actions.ndim > 1 else 1
        if "meta" in f and "min_action" in f["meta"] and "max_action" in f["meta"]:
            min_action = np.asarray(f["meta"]["min_action"][:], dtype=np.float32)
            max_action = np.asarray(f["meta"]["max_action"][:], dtype=np.float32)
        else:
            logger.warning(
                f"{h5_path} has no /meta/min_action|max_action; deriving action bounds from all demos"
            )
            all_actions = np.concatenate(
                [np.asarray(data_grp[k]["actions"][:], dtype=np.float32) for k in data_grp if k.startswith("demo")],
                axis=0,
            )
            min_action = all_actions.min(axis=0)
            max_action = all_actions.max(axis=0)

    if min_action.shape != (action_dim,) or max_action.shape != (action_dim,):
        raise ValueError(
            f"{h5_path}: action bounds have shape {min_action.shape}/{max_action.shape} but actions are "
            f"{action_dim}-dimensional"
        )
    min_action, max_action = widen_degenerate_bounds(min_action, max_action, eps=action_range_eps)

    for key, space in (extra_obs_spaces or {}).items():
        if key in remove:
            continue
        obs_spaces[key] = space

    observation_space = gym.spaces.Dict(obs_spaces)
    action_space = gym.spaces.Box(low=min_action, high=max_action, shape=(action_dim,), dtype=np.float32)
    logger.info(f"Synthesized spaces from {h5_path}")
    logger.info(f"  observation_space: {observation_space}")
    logger.info(f"  action_space: {action_space}")
    return observation_space, action_space
