#!/usr/bin/env python3
"""Visualize demonstrations stored in LIBERO datasets (robomimic-style HDF5) as videos.

Renders each ``/data/demo_i`` from a raw/regenerated LIBERO ``*_demo.hdf5`` into an mp4 that
shows the third-person ``agentview_rgb`` next to the wrist ``eye_in_hand_rgb`` camera, with a
text overlay of the task instruction, demo id, frame index, gripper state and current action.

Point ``--dataset`` at a single ``*_demo.hdf5`` file OR at a directory of them (e.g. a suite
folder ``third_party/LIBERO/libero/datasets/libero_10``); alternatively give ``--suite`` to
resolve the suite folder under ``--datasets-dir``. Select tasks with ``--task-ids`` (indices into
the suite's task map, same ordering as ``eval_pi0_libero.py``'s ``env.task_ids``) and pick
episodes with ``--demos`` (indices or ``all``) / ``--num-demos`` (first N).

Images stored by LIBERO are upside down on our platform (see regenerate_libero_dataset.py /
export_hitl_to_lerobot.py); ``--flip`` (default on) applies the ``[::-1, ::-1]`` correction so
demos render right-side-up, matching the pi0 preprocessing orientation.

Usage (openpi env, so h5py/opencv are available):
    uv run --frozen python scripts/visualize_libero_demos.py \
        --suite libero_90 --task-ids 57 58 59 --num-demos 3 --output-dir outputs/libero_viz

    uv run --frozen python scripts/visualize_libero_demos.py \
        --dataset third_party/LIBERO/libero/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 \
        --demos 0 5 12 --no-wrist
"""

import argparse
import glob
import importlib.util
import os

import cv2
import h5py
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASETS_DIR = os.path.join(REPO_ROOT, "third_party", "LIBERO", "libero", "datasets")


def _demo_sort_key(name: str):
    """Sort ``demo_<n>`` groups numerically (demo_2 before demo_10), non-numeric names last."""
    tail = name.rsplit("_", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, name)


def _grab_language(task_name: str) -> str:
    """Derive a LIBERO task's language instruction from its demo file stem (mirrors libero's
    ``grab_language_from_filename`` and export_hitl_to_lerobot.py): strip the ``<ROOM>_SCENE<N>_``
    prefix for LIBERO-100 tasks (names starting uppercase), else replace underscores with spaces.
    """
    name = task_name[: -len("_demo")] if task_name.endswith("_demo") else task_name
    x = name + ".bddl"
    if x[0].isupper():  # LIBERO-100 (libero_90 / libero_10): names begin with the room in caps
        offset = 8 if "SCENE10" in x else 7  # skip "SCENE10_" (8) or "SCENE<d>_" (7)
        language = " ".join(x[x.find("SCENE") + offset:].split("_"))
    else:  # libero_spatial / object / goal: lowercase task names
        language = " ".join(x.split("_"))
    return language[: language.find(".bddl")]


def _load_libero_task_map():
    """Load libero_task_map WITHOUT importing the libero package (mirrors export_hitl_to_lerobot.py).

    Returns a dict ``suite -> [task_name, ...]`` where the list index is the canonical task id (the
    same ordering used by ``eval_pi0_libero.py``'s ``env.task_ids`` and the LIBERO benchmark).
    """
    p = os.path.join(REPO_ROOT, "third_party", "LIBERO", "libero", "libero", "benchmark",
                     "libero_suite_task_map.py")
    if not os.path.exists(p):
        raise FileNotFoundError(f"LIBERO task map not found at {p}")
    spec = importlib.util.spec_from_file_location("libero_suite_task_map", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.libero_task_map


def _resolve_dataset_files(args) -> list[str]:
    """Return the list of ``*_demo.hdf5`` files to visualize, honoring --dataset/--suite/--task-ids."""
    if args.suite:
        root = os.path.join(args.datasets_dir, args.suite)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Suite directory not found: {root}")
        if args.task_ids:
            task_map = _load_libero_task_map()
            if args.suite not in task_map:
                raise KeyError(f"Unknown LIBERO suite {args.suite!r}. Available: {sorted(task_map)}")
            names = task_map[args.suite]
            files = []
            for tid in args.task_ids:
                if not (0 <= tid < len(names)):
                    raise IndexError(f"task_id {tid} out of range for {args.suite} (0..{len(names) - 1}).")
                path = os.path.join(root, f"{names[tid]}_demo.hdf5")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Demo file for task_id {tid} not found: {path}")
                files.append(path)
        else:
            files = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    elif args.dataset:
        if args.task_ids:
            raise ValueError("--task-ids requires --suite (task ids index into a suite's task map).")
        if os.path.isdir(args.dataset):
            files = sorted(glob.glob(os.path.join(args.dataset, "*.hdf5")))
        elif os.path.isfile(args.dataset):
            files = [args.dataset]
        else:
            raise FileNotFoundError(f"--dataset path not found: {args.dataset}")
    else:
        raise ValueError("Provide either --dataset (file or dir) or --suite.")

    if not files:
        raise FileNotFoundError("No matching *_demo.hdf5 files (check --dataset/--suite/--task-ids).")
    return files


def _select_demos(demo_names: list[str], args) -> list[str]:
    """Choose which demo groups to render from the sorted list, per --demos / --num-demos."""
    if args.demos and args.demos != ["all"]:
        chosen = []
        for d in args.demos:
            key = d if d.startswith("demo_") else f"demo_{d}"
            if key in demo_names:
                chosen.append(key)
            else:
                print(f"  WARNING: {key} not in file (has {len(demo_names)} demos); skipping.")
        return chosen
    if args.demos == ["all"] or args.num_demos <= 0:
        return demo_names
    return demo_names[: args.num_demos]


def _prep_rgb(frame, flip: bool):
    """LIBERO stored RGB [h,w,3] uint8 -> BGR for cv2, applying the upside-down correction."""
    frame = np.asarray(frame)
    if flip:
        frame = np.ascontiguousarray(frame[::-1, ::-1])
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _compose_frame(agent_bgr, wrist_bgr, header_lines, footer, panel_hw, show_wrist):
    """Stack agentview (+ wrist) side by side, upscale to panel_hw, and draw the text overlay."""
    ph, pw = panel_hw
    agent = cv2.resize(agent_bgr, (pw, ph), interpolation=cv2.INTER_NEAREST)
    if show_wrist and wrist_bgr is not None:
        wrist = cv2.resize(wrist_bgr, (pw, ph), interpolation=cv2.INTER_NEAREST)
        divider = np.full((ph, 4, 3), 255, dtype=np.uint8)
        canvas = np.hstack([agent, divider, wrist])
    else:
        canvas = agent

    w = canvas.shape[1]
    header_h = 18 * len(header_lines) + 8
    cv2.rectangle(canvas, (0, 0), (w, header_h), (0, 0, 0), -1)
    for i, line in enumerate(header_lines):
        cv2.putText(canvas, line, (6, 16 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if footer:
        fh = canvas.shape[0]
        cv2.rectangle(canvas, (0, fh - 22), (w, fh), (0, 0, 0), -1)
        cv2.putText(canvas, footer, (6, fh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 220, 255), 1, cv2.LINE_AA)
    return canvas


def _render_demo(g, prompt, demo_name, out_path, args):
    """Write one demo's video; returns the number of frames written."""
    obs = g["obs"]
    agent = np.asarray(obs["agentview_rgb"])
    wrist = np.asarray(obs["eye_in_hand_rgb"]) if "eye_in_hand_rgb" in obs else None
    actions = np.asarray(g["actions"], dtype=np.float32)
    gripper = np.asarray(obs["gripper_states"], dtype=np.float32) if "gripper_states" in obs else None
    n = len(actions)
    if n == 0:
        print(f"  {demo_name}: empty, skipping.")
        return 0

    show_wrist = args.wrist and wrist is not None
    panel_hw = (args.panel_size, args.panel_size)
    frames = []
    for t in range(n):
        a = actions[t]
        header = [
            f"{prompt}"[:64],
            f"{demo_name}  frame {t + 1}/{n}",
        ]
        grip = f" grip[{gripper[t][0]:+.2f},{gripper[t][1]:+.2f}]" if gripper is not None else ""
        footer = "act " + " ".join(f"{v:+.2f}" for v in a) + grip
        frames.append(_compose_frame(
            _prep_rgb(agent[t], args.flip),
            _prep_rgb(wrist[t], args.flip) if show_wrist else None,
            header, footer, panel_hw, show_wrist,
        ))

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("dataset source (give --dataset OR --suite)")
    src.add_argument("--dataset", default=None,
                     help="Path to a single *_demo.hdf5 file, or a directory containing them.")
    src.add_argument("--suite", default=None,
                     help="LIBERO suite name (e.g. libero_10, libero_90); resolved under --datasets-dir.")
    src.add_argument("--datasets-dir", default=DEFAULT_DATASETS_DIR,
                     help="Root of LIBERO demo datasets (contains <suite>/<task>_demo.hdf5).")
    src.add_argument("--task-ids", type=int, nargs="*", default=None,
                     help="Task ids to visualize (indices into the suite's task map, same ordering as "
                          "eval_pi0_libero.py's env.task_ids). Requires --suite. Default: all tasks.")

    sel = ap.add_argument_group("episode selection")
    sel.add_argument("--demos", nargs="*", default=None,
                     help="Demo ids to render: indices like '0 5 12', names like 'demo_3', or 'all'. "
                          "Default: first --num-demos.")
    sel.add_argument("--num-demos", type=int, default=3,
                     help="Number of demos per task to render when --demos is unset (default 3; <=0 = all).")
    sel.add_argument("--max-tasks", type=int, default=0,
                     help="Cap the number of task files processed (0 = no cap).")

    out = ap.add_argument_group("output / rendering")
    out.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "libero_viz"),
                     help="Directory to write the mp4 files into.")
    out.add_argument("--fps", type=int, default=20, help="Output video fps (default 20).")
    out.add_argument("--panel-size", type=int, default=256,
                     help="Per-camera panel size in px (images are upscaled from 128; default 256).")
    out.add_argument("--wrist", action=argparse.BooleanOptionalAction, default=True,
                     help="Show the wrist (eye-in-hand) camera beside agentview (default on).")
    out.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True,
                     help="Flip LIBERO images [::-1,::-1] so demos render right-side-up (default on).")
    args = ap.parse_args()

    files = _resolve_dataset_files(args)
    if args.max_tasks > 0:
        files = files[: args.max_tasks]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Visualizing {len(files)} task file(s) -> {args.output_dir}")

    total_videos = total_frames = 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        prompt = _grab_language(stem)
        with h5py.File(path, "r") as f:
            if "data" not in f:
                print(f"  SKIP {stem}: no /data group.")
                continue
            demo_names = sorted(f["data"].keys(), key=_demo_sort_key)
            chosen = _select_demos(demo_names, args)
            print(f"\n[{stem}]  '{prompt}'  ({len(demo_names)} demos, rendering {len(chosen)})")
            for demo_name in chosen:
                out_path = os.path.join(args.output_dir, f"{stem}__{demo_name}.mp4")
                n = _render_demo(f["data"][demo_name], prompt, demo_name, out_path, args)
                if n:
                    print(f"    {demo_name}: {n} frames -> {out_path}")
                    total_videos += 1
                    total_frames += n

    print(f"\nDone: wrote {total_videos} video(s) / {total_frames} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
