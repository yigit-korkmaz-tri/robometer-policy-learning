#!/usr/bin/env python3
"""Export collected pi0.5 HITL rollouts (robomimic-style HDF5) to a LeRobot dataset for openpi.

Bridges the HDF5 written by ``scripts/collect_hitl_libero_pi0.py`` into the LeRobot format that
openpi's training pipeline consumes (see ``third_party/dsrl_openpi/examples/libero/convert_libero_data_to_lerobot.py``).
Each ``/data/demo_i`` becomes one LeRobot episode; the per-demo ``prompt`` attr becomes the episode
task (openpi reconstructs the prompt from the task table via ``prompt_from_task=True``).

The dataset carries the standard LIBERO features (``image``, ``wrist_image``, ``state``, ``actions``)
plus an extra per-frame ``intervention`` label (0=policy, 1=human, 2=offline demo). ``intervention``
is ignored by the HG-DAgger config (dropped at repack) but preserved for the Flow-MILE data config.

Anti-forgetting aggregation — two base-demo sources (both folded into the SAME repo, since openpi
trains one repo):
  * ``--base-demos``: HDF5 file(s) in THIS repo's collection schema (obs/{image,wrist_image,state}).
  * ``--libero-base-*``: LIBERO expert demos from ``third_party/LIBERO/libero/datasets`` built ON THE FLY — give a
    suite, a list of task ids, and demos-per-task; the raw LIBERO demo obs
    (``agentview_rgb``/``eye_in_hand_rgb``/``ee_states``/``gripper_states``) are converted to pi0 format
    with the SAME transform used by ``examples/libero/main.py`` and the HITL corrector: images flipped
    ``[::-1,::-1]`` + ``resize_with_pad`` to 224, state = ``concat(ee_states, gripper_states)`` (8-dim
    ``[eef_pos, axis_angle, gripper_qpos]``). Base-demo frames are labelled 2 (offline).

Requires the ``lerobot`` package installed in the environment (the openpi env). Run with ``--frozen``:

Usage (openpi env):
    uv run --frozen python scripts/export_hitl_to_lerobot.py \
        --inputs 'outputs/**/hitl_libero_pi0_rollouts.hdf5' \
        --repo-id yourname/libero_hitl_r1 \
        --libero-base-suite libero_90 --libero-base-task-ids 57 58 59 --libero-base-num-demos 10

Then compute norm stats:
    uv run --frozen third_party/dsrl_openpi/scripts/compute_norm_stats.py --config-name pi05_libero_hitl_lora \
        --repo-id yourname/libero_hitl_r1
"""

import argparse
import glob
import importlib.util
import os
import shutil
import sys

import h5py
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Intervention labels (match collect_hitl_libero_pi0.py / the in-house HITL buffers).
LABEL_POLICY, LABEL_HUMAN, LABEL_OFFLINE = 0, 1, 2


def _demo_sort_key(name: str):
    tail = name.rsplit("_", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, name)


def _peek_rollout_shape(paths):
    """Return the stored rollout-sample pool shape ``(P, H, A)`` from the first HDF5 demo that has a
    ``rollout_samples`` dataset, or ``None`` if no file carries one (Flow-MILE baseline pools; see
    collect_hitl_libero_pi0.py). All frames in the exported repo must share this fixed shape."""
    for path in paths:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                continue
            for demo in sorted(f["data"].keys(), key=_demo_sort_key):
                g = f["data"][demo]
                if "rollout_samples" in g:
                    return tuple(np.asarray(g["rollout_samples"]).shape[1:])  # (P, H, A)
    return None


def _iter_demos(path, *, default_label, rollout_shape=None, warn_missing_rollout=False):
    """Yield (frames_dict, prompt) per demo in a collection-schema HITL/robomimic HDF5.

    frames_dict has arrays: image [N,H,W,3] uint8, wrist_image [N,H,W,3] uint8, state [N,8] f32,
    actions [N,7] f32, intervention [N] int64 (falls back to ``default_label`` if the dataset is
    absent — e.g. externally-supplied demos with no HITL labels). When ``rollout_shape`` (P,H,A) is
    given, also emits ``rollout_samples`` [N,P,H,A] f32 — the stored Flow-MILE baseline pool, or zeros
    if this file lacks it (warned once when ``warn_missing_rollout``).
    """
    warned = False
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{path} has no /data group; not a HITL/robomimic HDF5.")
        for demo in sorted(f["data"].keys(), key=_demo_sort_key):
            g = f["data"][demo]
            obs = g["obs"]
            n = len(np.asarray(g["actions"]))
            if n == 0:
                continue
            if "intervention" in g:
                interv = np.asarray(g["intervention"]).astype(np.int64).reshape(-1)
            else:
                interv = np.full((n,), int(default_label), dtype=np.int64)
            frames = {
                "image": np.asarray(obs["image"]),
                "wrist_image": np.asarray(obs["wrist_image"]),
                "state": np.asarray(obs["state"], dtype=np.float32),
                "actions": np.asarray(g["actions"], dtype=np.float32),
                "intervention": interv,
            }
            if rollout_shape is not None:
                if "rollout_samples" in g:
                    frames["rollout_samples"] = np.asarray(g["rollout_samples"], dtype=np.float32)
                else:
                    if warn_missing_rollout and not warned:
                        print(f"  WARNING: {os.path.basename(path)} has no rollout_samples; zero-filling "
                              f"{rollout_shape} (label-0 Flow-MILE baselines degrade to zero).")
                        warned = True
                    frames["rollout_samples"] = np.zeros((n, *rollout_shape), dtype=np.float32)
            prompt = g.attrs.get("prompt", "")
            if isinstance(prompt, bytes):
                prompt = prompt.decode()
            yield frames, str(prompt)


def _grab_language(task_name: str) -> str:
    """Derive a LIBERO task's language instruction from its name (mirrors libero's
    ``grab_language_from_filename``): strip the ``<ROOM>_SCENE<N>_`` prefix for LIBERO-100 tasks
    (names starting uppercase), else just replace underscores with spaces. This is the same value as
    ``benchmark.get_task(id).language`` and is robust to regenerated datasets that drop the
    ``/data.attrs['problem_info']`` metadata.
    """
    x = task_name + ".bddl"
    if x[0].isupper():  # LIBERO-100 (libero_90 / libero_10): names begin with the room in caps
        offset = 8 if "SCENE10" in x else 7  # skip "SCENE10_" (8) or "SCENE<d>_" (7)
        language = " ".join(x[x.find("SCENE") + offset:].split("_"))
    else:  # libero_spatial / object / goal: lowercase task names
        language = " ".join(x.split("_"))
    return language[: language.find(".bddl")]


def _load_libero_task_map():
    """Load libero_task_map WITHOUT importing the libero package (the openpi env may lack it)."""
    p = os.path.join(REPO_ROOT, "third_party", "LIBERO", "libero", "libero", "benchmark", "libero_suite_task_map.py")
    if not os.path.exists(p):
        raise FileNotFoundError(f"LIBERO task map not found at {p}")
    spec = importlib.util.spec_from_file_location("libero_suite_task_map", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.libero_task_map


def _to_pi0_images(rgb, image_hw, flip):
    """Convert LIBERO stored RGB frames [N,h0,w0,3] uint8 -> pi0 format [N,H,W,3] uint8.

    Matches ``examples/libero/main.py`` / ``preprocess_obs_for_pi0``: flip both axes (opengl->pi0),
    then ``resize_with_pad`` to (H,W) and cast uint8.
    """
    from openpi_client import image_tools

    h, w = image_hw
    out = []
    for frame in np.asarray(rgb):
        if flip:
            frame = np.ascontiguousarray(frame[::-1, ::-1])
        out.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, h, w)))
    return np.stack(out, axis=0)


def _iter_libero_demos(datasets_dir, suite, task_ids, num_demos, image_hw, base_label, flip, rollout_shape=None):
    """Yield (frames_dict, prompt) for LIBERO expert demos, converted to pi0 format on the fly.

    For each task id: resolve ``<datasets_dir>/<suite>/<TASK_NAME>_demo.hdf5`` via the task map, take
    the first ``num_demos`` demos, convert images + build the 8-dim state, and label every frame
    ``base_label`` (2=offline). The prompt is the task's ``language_instruction`` (from /data attrs).
    """
    task_map = _load_libero_task_map()
    if suite not in task_map:
        raise KeyError(f"Unknown LIBERO suite {suite!r}. Available: {sorted(task_map)}")
    names = task_map[suite]
    for tid in task_ids:
        if not (0 <= tid < len(names)):
            raise IndexError(f"task_id {tid} out of range for suite {suite} (0..{len(names) - 1}).")
        name = names[tid]
        path = os.path.join(datasets_dir, suite, f"{name}_demo.hdf5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"LIBERO demo file not found: {path}")
        # Derive the instruction from the task name (regenerated datasets drop the /data problem_info).
        prompt = _grab_language(name)
        with h5py.File(path, "r") as f:
            data = f["data"]
            demos = sorted(data.keys(), key=_demo_sort_key)[: int(num_demos)]
            for demo in demos:
                g = data[demo]
                obs = g["obs"]
                n = len(np.asarray(g["actions"]))
                if n == 0:
                    continue
                # State: [eef_pos(3), axis_angle(3), gripper_qpos(2)] == concat(ee_states, gripper_states).
                ee_states = np.asarray(obs["ee_states"], dtype=np.float32)      # [n,6] = pos + axis-angle
                gripper = np.asarray(obs["gripper_states"], dtype=np.float32)   # [n,2]
                state = np.concatenate([ee_states, gripper], axis=1)            # [n,8]
                frames = {
                    "image": _to_pi0_images(obs["agentview_rgb"], image_hw, flip),
                    "wrist_image": _to_pi0_images(obs["eye_in_hand_rgb"], image_hw, flip),
                    "state": state,
                    "actions": np.asarray(g["actions"], dtype=np.float32),
                    "intervention": np.full((n,), int(base_label), dtype=np.int64),
                }
                if rollout_shape is not None:
                    # Offline demos (label 2) have no rollout pool; zero-fill (excluded from every
                    # rollout-sample-dependent term). Keeps the feature fixed-shape across all frames.
                    frames["rollout_samples"] = np.zeros((n, *rollout_shape), dtype=np.float32)
                yield frames, str(prompt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=[],
                    help="Glob(s) for collected HITL HDF5 files (corrections).")
    ap.add_argument("--repo-id", required=True, help="Output LeRobot repo id (written under $HF_LEROBOT_HOME).")
    ap.add_argument("--base-demos", nargs="*", default=[],
                    help="Optional glob(s) for demo HDF5 file(s) (collection schema) to aggregate.")
    ap.add_argument("--base-label", type=int, default=LABEL_OFFLINE,
                    help="Intervention label for base-demo frames lacking an intervention dataset (default 2=offline).")
    # ---- On-the-fly LIBERO expert base demos ----
    ap.add_argument("--libero-base-suite", default=None,
                    help="LIBERO suite for on-the-fly base demos (e.g. libero_90, libero_10).")
    ap.add_argument("--libero-base-task-ids", type=int, nargs="*", default=[],
                    help="Task ids within --libero-base-suite to pull expert demos from.")
    ap.add_argument("--libero-base-num-demos", type=int, default=10,
                    help="Number of expert demos per task to include (first N).")
    ap.add_argument("--libero-datasets-dir", default=os.path.join(REPO_ROOT, "third_party", "LIBERO", "libero", "datasets"),
                    help="Root of LIBERO demo datasets (contains <suite>/<task>_demo.hdf5).")
    ap.add_argument("--libero-image-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="Flip LIBERO stored images [::-1,::-1] to pi0 orientation (default on; matches main.py).")
    # ---- Output dataset params ----
    ap.add_argument("--rollout-pool-size", type=int, default=None,
                    help="Flow-MILE rollout-sample pool size P. None (default) = auto (infer from the "
                         "corrections' stored rollout_samples; add the feature only if present). 0 = "
                         "force OFF (ignore stored pools). >0 = require stored pools of exactly this P.")
    ap.add_argument("--fps", type=int, default=10, help="Dataset fps (pi05_libero uses 10).")
    ap.add_argument("--image-hw", type=int, nargs=2, default=(224, 224), help="Image height width.")
    ap.add_argument("--robot-type", default="panda")
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()

    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    except ImportError:
        sys.exit(
            "ERROR: `lerobot` is not installed in this environment. Run from the openpi env with "
            "`uv run --frozen ...` (this repo's openpi copy pins lerobot as a project dependency)."
        )

    def _expand(globs):
        paths = []
        for gpat in globs:
            hits = sorted(glob.glob(gpat, recursive=True))
            if not hits:
                print(f"  WARNING: no files matched {gpat!r}")
            paths.extend(hits)
        return paths

    correction_files = _expand(args.inputs)
    base_files = _expand(args.base_demos)
    use_libero = bool(args.libero_base_suite and args.libero_base_task_ids)
    if not correction_files and not base_files and not use_libero:
        sys.exit("ERROR: no inputs (need --inputs, --base-demos, and/or --libero-base-suite + task ids).")
    print(
        f"Corrections: {len(correction_files)} file(s); base-demo files: {len(base_files)}; "
        f"LIBERO base: {args.libero_base_suite} tasks {args.libero_base_task_ids} "
        f"({args.libero_base_num_demos}/task)" if use_libero else
        f"Corrections: {len(correction_files)} file(s); base-demo files: {len(base_files)}; LIBERO base: none"
    )

    # Flow-MILE rollout-sample pool shape (P, H, A). Auto-inferred from the corrections unless
    # --rollout-pool-size 0 forces it off; must be uniform across the whole repo (fixed feature shape).
    rollout_shape = None if args.rollout_pool_size == 0 else _peek_rollout_shape(correction_files)
    if rollout_shape is not None and args.rollout_pool_size and args.rollout_pool_size != rollout_shape[0]:
        sys.exit(f"ERROR: --rollout-pool-size={args.rollout_pool_size} but stored pools have P={rollout_shape[0]}.")
    if args.rollout_pool_size and args.rollout_pool_size > 0 and rollout_shape is None:
        sys.exit("ERROR: --rollout-pool-size>0 but no input has a `rollout_samples` dataset "
                 "(collect with hitl.rollout_pool_size>0 first).")
    if rollout_shape is not None:
        print(f"Flow-MILE rollout pool: shape {rollout_shape} (P,H,A) — exporting `rollout_samples` feature.")

    h, w = args.image_hw
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    features = {
        "image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        # Extra per-frame HITL label (0=policy, 1=human, 2=offline). Preserved for Flow-MILE.
        "intervention": {"dtype": "int64", "shape": (1,), "names": ["intervention"]},
    }
    if rollout_shape is not None:
        # Per-frame frozen-rollout baseline pool (env-space, raw 7-dim; normalized/padded in-loader).
        features["rollout_samples"] = {
            "dtype": "float32", "shape": tuple(rollout_shape), "names": ["pool", "horizon", "action"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Unified list of (description, demo-stream). Each stream yields (frames_dict, prompt).
    streams = []
    for p in correction_files:
        streams.append((f"corrections:{os.path.basename(p)}",
                        _iter_demos(p, default_label=LABEL_HUMAN, rollout_shape=rollout_shape,
                                    warn_missing_rollout=True)))
    for p in base_files:
        streams.append((f"base:{os.path.basename(p)}",
                        _iter_demos(p, default_label=args.base_label, rollout_shape=rollout_shape)))
    if use_libero:
        streams.append((
            f"libero:{args.libero_base_suite}:{args.libero_base_task_ids}",
            _iter_libero_demos(
                args.libero_datasets_dir, args.libero_base_suite, args.libero_base_task_ids,
                args.libero_base_num_demos, (h, w), args.base_label, args.libero_image_flip,
                rollout_shape=rollout_shape,
            ),
        ))

    n_demos = n_frames = n_human = 0
    for desc, stream in streams:
        for frames, prompt in stream:
            n = len(frames["actions"])
            for i in range(n):
                frame = {
                    "image": frames["image"][i],
                    "wrist_image": frames["wrist_image"][i],
                    "state": frames["state"][i],
                    "actions": frames["actions"][i],
                    "intervention": np.asarray([frames["intervention"][i]], dtype=np.int64),
                    # lerobot >= 0cf8648 takes the task per frame; save_episode() no longer does.
                    "task": prompt or "libero task",
                }
                if rollout_shape is not None:
                    frame["rollout_samples"] = np.asarray(frames["rollout_samples"][i], dtype=np.float32)
                dataset.add_frame(frame)
            dataset.save_episode()
            n_demos += 1
            n_frames += n
            n_human += int((frames["intervention"] == LABEL_HUMAN).sum())
        print(f"  exported {desc} (running: {n_demos} demos / {n_frames} frames)")

    # NOTE: no dataset.consolidate() call -- removed in lerobot >= 0cf8648; save_episode() finalizes.
    print(
        f"Done: {n_demos} episodes / {n_frames} frames ({n_human} human, "
        f"{n_human / max(n_frames, 1):.1%}) -> {output_path}"
    )
    if args.push_to_hub:
        dataset.push_to_hub(tags=["libero", "panda", "hitl"], private=True, push_videos=True, license="apache-2.0")
        print(f"Pushed {args.repo_id} to the Hugging Face Hub.")


if __name__ == "__main__":
    main()
