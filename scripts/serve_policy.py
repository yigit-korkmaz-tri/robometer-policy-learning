#!/usr/bin/env python3
"""Serve a policy trained by scripts/train_supervised.py over a websocket, for real-robot rollouts.

The robometer counterpart of third_party/dsrl_openpi/scripts/serve_policy.py: same wire protocol
(websockets + openpi-client's msgpack-numpy), so a robot host can drive it with the exact client
scaffolding used for pi0.5 (`openpi_client.WebsocketClientPolicy` +
`openpi_client.action_chunk_broker.ActionChunkBroker`). See scripts/raiden_flow_rollout.py for the
robot side.

Consistency with training is the whole point of this script, and it is enforced by
`<run>/checkpoints/policy_metadata.json`, written during training:

  * low-dim observations are z-scored with the **training buffer's** statistics
    (training.normalize_lowdim_obs); no other component on the robot has them;
  * camera frames are resized to the dataset's image size with the dataset converter's filter;
  * robot camera names are mapped to observation keys via the dataset's conversion camera_map;
  * actions come back un-normalized into the robot's action space (the actor's own act() does this),
    so a client commands them directly;
  * the handshake advertises n_action_steps, so the client's ActionChunkBroker replans on the same
    cadence the policy was trained with.

Usage (GPU host):
    uv run python scripts/serve_policy.py --load-dir outputs/2026-08-10/18-13-47 --port 8000
    # a specific checkpoint, and a smoke test with synthetic observations:
    uv run python scripts/serve_policy.py --load-dir <run> --checkpoint 20000 --selftest

Older checkpoints with no policy_metadata.json can still be served by pointing --dataset at the
training HDF5, which re-derives the statistics exactly as the eval scripts do (by rebuilding the
buffer); --image-size / --camera-map fill in the rest.
"""

import argparse
import json
import logging
import os
import socket
import sys

import numpy as np
import torch

from robometer_policy_learning.utils.logging_compat import get_logger
from robometer_policy_learning.utils.policy_serving import (
    METADATA_FILENAME,
    PolicyMetadata,
    RobometerPolicy,
    is_image_key,
)
from robometer_policy_learning.utils.training_utils import resolve_checkpoint_dir

logger = get_logger()


def load_actor(load_dir: str, checkpoint=None, device: torch.device = None):
    ckpt_dir = resolve_checkpoint_dir(load_dir, checkpoint)
    actor_path = os.path.join(ckpt_dir, "actor.pt")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor.pt not found at {actor_path}")
    actor = torch.load(actor_path, map_location=device, weights_only=False).to(device)
    actor.eval()
    logger.info(f"Loaded {type(actor).__name__} from {actor_path}")
    return actor, ckpt_dir


def rebuild_metadata_from_dataset(actor, dataset_path: str, args) -> PolicyMetadata:
    """Fallback for checkpoints predating policy_metadata.json.

    Rebuilds an H5ReplayBuffer purely to recover ``lowdim_obs_stats`` -- the same trick
    scripts/eval_policy.py and scripts/rollout_hitl.py use -- so the z-scoring matches training.
    Images stay lazy, so this only reads the low-dim arrays.
    """
    from robometer_policy_learning.buffers.h5_replay_buffer import H5ReplayBuffer
    from robometer_policy_learning.buffers.samplers import RandomSampler
    from robometer_policy_learning.utils.dataset_spaces import build_spaces_from_h5

    remove_obs_keys = list(getattr(actor, "remove_obs_keys", None) or [])
    observation_space, action_space = build_spaces_from_h5(dataset_path, remove_obs_keys=remove_obs_keys)

    stats = {}
    if args.normalize_lowdim_obs:
        buffer = H5ReplayBuffer(
            h5_paths=[dataset_path],
            sampler=RandomSampler(),
            remove_obs_keys=remove_obs_keys,
            post_transforms=[],
            normalize_lowdim_obs=True,
        )
        stats = {
            k: {
                "mean": np.asarray(v["mean"], np.float32).ravel().tolist(),
                "std": np.asarray(v["std"], np.float32).ravel().tolist(),
            }
            for k, v in buffer.lowdim_obs_stats.items()
        }

    keys = [k for k in observation_space.spaces if k not in remove_obs_keys]
    image_keys = [k for k in keys if is_image_key(k)]
    image_size = args.image_size
    if image_size is None and image_keys:
        image_size = int(observation_space.spaces[image_keys[0]].shape[0])

    camera_map = {}
    if args.camera_map:
        for pair in args.camera_map.split(","):
            if pair.strip():
                cam, obs_key = pair.split("=", 1)
                camera_map[cam.strip()] = obs_key.strip()

    horizon = int(getattr(actor, "horizon", 1) or 1)
    return PolicyMetadata(
        image_keys=image_keys,
        lowdim_keys=[k for k in keys if not is_image_key(k)],
        remove_obs_keys=remove_obs_keys,
        image_size=image_size,
        camera_map=camera_map,
        lowdim_obs_stats=stats,
        action_dim=int(np.prod(action_space.shape)),
        action_low=np.asarray(action_space.low, np.float32).ravel().tolist(),
        action_high=np.asarray(action_space.high, np.float32).ravel().tolist(),
        chunk_size=horizon,
        n_action_steps=args.n_action_steps or max(1, horizon // 2),
        actor_class=type(actor).__name__,
        dataset_path=dataset_path,
        normalize_lowdim_obs=bool(args.normalize_lowdim_obs),
    )


def build_policy(args) -> RobometerPolicy:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    actor, ckpt_dir = load_actor(args.load_dir, args.checkpoint, device)

    metadata = PolicyMetadata.load(ckpt_dir)
    if metadata is None:
        if not args.dataset:
            raise FileNotFoundError(
                f"No {METADATA_FILENAME} found in {ckpt_dir} or its parent. This checkpoint predates "
                "metadata saving; re-run training, or pass --dataset <training h5> (plus "
                "--normalize-lowdim-obs / --image-size / --camera-map to match the training config) "
                "so the inference contract can be reconstructed."
            )
        logger.warning(f"No {METADATA_FILENAME} in {ckpt_dir}; reconstructing from {args.dataset}")
        metadata = rebuild_metadata_from_dataset(actor, args.dataset, args)
    else:
        logger.info(f"Loaded {METADATA_FILENAME} from {ckpt_dir}")

    # CLI overrides, for deliberately deviating from the training-time defaults.
    if args.n_action_steps:
        metadata.n_action_steps = int(args.n_action_steps)
    if args.prompt:
        metadata.prompt = args.prompt

    if metadata.n_action_steps > metadata.chunk_size:
        raise ValueError(
            f"n_action_steps ({metadata.n_action_steps}) > chunk_size ({metadata.chunk_size}): the "
            "policy cannot supply that many actions per inference."
        )
    if metadata.action_mode != "absolute":
        # The dataset stored action - state, so the robot must add its state back. Say so loudly:
        # commanding delta actions as absolute joint targets would drive the arm to ~zero.
        logger.warning(
            f"action_mode={metadata.action_mode!r}: served actions are RELATIVE to the current "
            "state. The client must add its current state before commanding the robot."
        )

    logger.info(f"Policy contract: {metadata.describe()}")
    return RobometerPolicy(actor, metadata, device=device)


def selftest(policy: RobometerPolicy):
    """No robot, no network: assert a synthetic observation produces a sane action chunk."""
    meta = policy.metadata
    obs = {k: np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for k in meta.image_keys}
    for key in meta.lowdim_keys:
        dim = len(meta.lowdim_obs_stats.get(key, {}).get("mean", [])) or meta.action_dim
        obs[key] = np.zeros(dim, dtype=np.float32)
    actions = policy.infer(obs)["actions"]
    assert actions.ndim == 2, f"expected [chunk, action_dim], got {actions.shape}"
    assert actions.shape[1] == meta.action_dim, f"action_dim {actions.shape[1]} != {meta.action_dim}"
    assert np.isfinite(actions).all(), "non-finite actions"
    low = np.asarray(meta.action_low, np.float32)
    high = np.asarray(meta.action_high, np.float32)
    within = bool((actions >= low - 1e-3).all() and (actions <= high + 1e-3).all())
    print(f"[serve][selftest] OK — actions {actions.shape}, within training action bounds: {within}")
    print(f"[serve][selftest] first action: {np.round(actions[0], 4).tolist()}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--load-dir", required=True, help="Training run dir (containing checkpoints/<step>/actor.pt)")
    p.add_argument("--checkpoint", default=None, help="Which checkpoints/<step> to serve (default: latest)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    p.add_argument("--prompt", default=None, help="Override the task prompt advertised to clients")
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override actions executed per inference (default: the training value from metadata)",
    )
    p.add_argument("--selftest", action="store_true", help="Run one synthetic inference and exit (no server)")
    # Fallback path for checkpoints without policy_metadata.json.
    p.add_argument("--dataset", default=None, help="Training HDF5, to reconstruct missing metadata")
    p.add_argument(
        "--normalize-lowdim-obs",
        action="store_true",
        help="With --dataset: the training run used training.normalize_lowdim_obs=true",
    )
    p.add_argument("--image-size", type=int, default=None, help="With --dataset: training image size")
    p.add_argument("--camera-map", default=None, help="With --dataset: robot_cam=obs_key,... aliases")
    return p.parse_args()


def main():
    args = parse_args()
    policy = build_policy(args)

    if args.selftest:
        selftest(policy)
        return

    # openpi-client vendors the msgpack-numpy codec this protocol uses; it is a small standalone
    # package, unlike the full `openpi` (which conflicts with the robometer extra), so prefer it and
    # fall back to the installed openpi only if the submodule is absent.
    client_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "dsrl_openpi", "packages", "openpi-client", "src",
    )
    if os.path.isdir(client_src) and client_src not in sys.path:
        sys.path.insert(0, client_src)
    from openpi.serving import websocket_policy_server

    hostname = socket.gethostname()
    logger.info(f"Serving on {args.host}:{args.port} (host={hostname})")
    logger.info(f"Client handshake metadata:\n{json.dumps(policy.server_metadata, indent=2)[:1200]}")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.server_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
