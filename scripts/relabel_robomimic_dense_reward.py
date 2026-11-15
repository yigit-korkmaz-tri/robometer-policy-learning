#!/usr/bin/env python3
"""Relabel a converted robomimic HDF5 with robosuite's dense (shaped) reward.

robomimic datasets store sparse 0/1 rewards, but online SAC in the reconstructed env
(``env.env_name=robomimic``) gets robosuite's dense shaped reward. For offline->online RL both land
in one replay buffer, so the offline rewards must use the same function. This replays each step's
stored sim state (``/data/demo_i/states``) in the env rebuilt from the dataset's own metadata (same
task / robot / controller / reward_scale / scene as ``setup_robomimic_env``) and recomputes
``env.reward()``.

Reward semantics match an online step: transition t's reward is evaluated at the state REACHED by
``actions[t]``, i.e. ``states[t+1]``. The last step has no recorded next state, so it is simulated
by stepping ``actions[T-1]`` from ``states[T-1]``.

The output is a copy of the input with ``rewards`` replaced; the original sparse rewards are kept
as ``sparse_rewards`` (the buffers ignore that key). ``dones`` are left unchanged.

Usage:
    uv run python scripts/relabel_robomimic_dense_reward.py \
        -i data/robomimic_datasets/robomimic_lowdim/square/ph/low_dim_converted.hdf5
    # -> .../low_dim_converted_dense.hdf5 (or pass -o); --verify N cross-checks N demos by stepping
    uv run python scripts/relabel_robomimic_dense_reward.py -i in.hdf5 -o out.hdf5 --verify 3
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import shutil
from copy import deepcopy

import h5py
import numpy as np
from tqdm import tqdm

from robometer_policy_learning.envs.robosuite_wrappers import load_robomimic_env_metadata
from robometer_policy_learning.utils.robomimic_scene import restore_dataset_scene


def build_reward_env(dataset_path: str):
    """Rebuild the dataset's env with dense reward shaping and no rendering.

    Mirrors setup_robomimic_env's kwargs (dataset env_kwargs + reward_shaping=True + restored
    dataset scene) so the reward function is exactly the one online training sees.
    """
    import robosuite as suite

    env_meta = load_robomimic_env_metadata(dataset_path)
    env_kwargs = deepcopy(env_meta.get("env_kwargs", {}))
    env_kwargs.pop("env_name", None)
    for k in ("camera_names", "camera_heights", "camera_widths", "camera_depths"):
        env_kwargs.pop(k, None)
    env_kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "use_object_obs": True,
            "reward_shaping": True,
            "ignore_done": True,  # we jump between arbitrary states; never stop on horizon
            "hard_reset": False,
        }
    )
    env = suite.make(env_meta["env_name"], **env_kwargs)
    restore_dataset_scene(env, dataset_path)
    env.reset()
    return env


def _set_state(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()


def dense_rewards_for_demo(env, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    T = len(actions)
    rewards = np.zeros(T, dtype=np.float64)
    for t in range(T - 1):
        _set_state(env, states[t + 1])
        rewards[t] = float(env.reward(actions[t]))
    _set_state(env, states[T - 1])
    _, r_last, _, _ = env.step(actions[T - 1])
    rewards[T - 1] = float(r_last)
    return rewards


def stepped_rewards_for_demo(env, states: np.ndarray, actions: np.ndarray, n: int) -> np.ndarray:
    """Reward from actually stepping actions[t] out of states[t] (for --verify)."""
    out = []
    for t in range(min(n, len(actions))):
        _set_state(env, states[t])
        _, r, _, _ = env.step(actions[t])
        out.append(float(r))
    return np.asarray(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="converted robomimic HDF5")
    parser.add_argument("-o", "--output", default=None, help="output HDF5 (default: <input>_dense.hdf5)")
    parser.add_argument("--verify", type=int, default=0, help="cross-check N demos against stepped rewards")
    parser.add_argument("--overwrite", action="store_true", help="overwrite an existing output file")
    args = parser.parse_args()

    out = args.output or os.path.splitext(args.input)[0] + "_dense.hdf5"
    if os.path.abspath(out) == os.path.abspath(args.input):
        raise ValueError("Output must differ from input (the input is copied, then relabelled).")
    if os.path.exists(out) and not args.overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it.")

    env = build_reward_env(args.input)
    print(f"Copying {args.input} -> {out}")
    shutil.copyfile(args.input, out)

    all_dense, n_success_steps, dense_at_success = [], 0, []
    with h5py.File(out, "r+") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)
        for i, k in enumerate(tqdm(demos, desc="Relabelling")):
            g = f["data"][k]
            if "states" not in g:
                raise KeyError(f"{k} has no 'states'; dense relabelling needs the recorded sim states.")
            states = np.asarray(g["states"])
            actions = np.asarray(g["actions"])
            sparse = np.asarray(g["rewards"]) if "rewards" in g else np.zeros(len(actions))
            dense = dense_rewards_for_demo(env, states, actions)

            if i < args.verify:
                stepped = stepped_rewards_for_demo(env, states, actions, n=len(actions) - 1)
                err = np.abs(stepped - dense[: len(stepped)])
                print(f"  verify {k}: max |stepped - replayed| = {err.max():.3e} (mean {err.mean():.3e})")

            if "sparse_rewards" not in g:
                g.create_dataset("sparse_rewards", data=sparse)
            del g["rewards"]
            g.create_dataset("rewards", data=dense)

            all_dense.append(dense)
            mask = sparse > 0.5
            n_success_steps += int(mask.sum())
            dense_at_success.extend(dense[mask].tolist())
        f["data"].attrs["reward_type"] = "dense_robosuite_shaped"

    env.close()
    d = np.concatenate(all_dense)
    print(f"Wrote {out}: {len(demos)} demos, {len(d)} steps")
    print(f"  dense reward: min {d.min():.3f} mean {d.mean():.3f} max {d.max():.3f}")
    if dense_at_success:
        print(
            f"  at the {n_success_steps} sparse-success steps: dense mean {np.mean(dense_at_success):.3f} "
            f"(min {np.min(dense_at_success):.3f})"
        )


if __name__ == "__main__":
    main()
