#!/usr/bin/env python3
"""Compare a saved correction rollout against a LIBERO expert demo, side-by-side (visual + numeric).

Use this BEFORE exporting/pushing to LeRobot to confirm the corrections and the offline expert demos
land in the SAME pi0 space — same image orientation/resolution, same 8-dim state convention, same
7-dim action space. The expert side is converted with the EXACT functions from
``scripts/export_hitl_to_lerobot.py`` (loaded as a module), so what you see here is exactly what the
export would write. Runs in the root env (no lerobot needed).

Outputs:
  * a numeric report (per-dim state/action ranges, shapes/dtypes, prompts) + a same/different verdict;
  * a side-by-side mp4: left = correction demo (POLICY/HUMAN banner), right = expert demo (OFFLINE).

Usage:
    uv run python scripts/compare_correction_vs_expert.py \
        --correction data/corrections_t9.hdf5 --correction-demo 0 \
        --libero-suite libero_90 --task-id 9 --expert-demo 0 --show-wrist
"""

import argparse
import importlib.util
import os

import cv2
import h5py
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {0: ("POLICY", (0, 180, 0)), 1: ("HUMAN", (0, 0, 255)), 2: ("OFFLINE", (200, 120, 0))}


def _demo_sort_key(name: str):
    tail = name.rsplit("_", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, name)


def _load_export_module():
    """Load export_hitl_to_lerobot.py as a module (single source of truth for LIBERO->pi0 conversion)."""
    p = os.path.join(REPO_ROOT, "scripts", "export_hitl_to_lerobot.py")
    spec = importlib.util.spec_from_file_location("export_hitl_to_lerobot", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _read_correction_demo(path, demo_idx):
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{path} has no /data group.")
        demos = sorted(f["data"].keys(), key=_demo_sort_key)
        if not (0 <= demo_idx < len(demos)):
            raise IndexError(f"correction demo {demo_idx} out of range (file has {len(demos)} demos).")
        demo = demos[demo_idx]
        g = f["data"][demo]
        o = g["obs"]
        n = len(np.asarray(g["actions"]))
        frames = {
            "image": np.asarray(o["image"]),
            "wrist_image": np.asarray(o["wrist_image"]) if "wrist_image" in o else None,
            "state": np.asarray(o["state"], dtype=np.float32),
            "actions": np.asarray(g["actions"], dtype=np.float32),
            "intervention": (
                np.asarray(g["intervention"]).astype(int).reshape(-1) if "intervention" in g
                else np.zeros(n, dtype=int)
            ),
        }
        prompt = g.attrs.get("prompt", "")
        prompt = prompt.decode() if isinstance(prompt, bytes) else str(prompt)
        return frames, prompt, demo


def _get_expert_demo(exp_mod, datasets_dir, suite, task_id, expert_demo, flip):
    gen = exp_mod._iter_libero_demos(datasets_dir, suite, [task_id], expert_demo + 1, (224, 224), 2, flip)
    for i, (fr, pr) in enumerate(gen):
        if i == expert_demo:
            return fr, pr
    raise IndexError(f"expert demo {expert_demo} not found for {suite} task {task_id} (too few demos).")


def _numeric_report(name, fr):
    print(f"[{name}] frames={len(fr['actions'])}")
    for key in ("state", "actions"):
        a = np.asarray(fr[key])
        rng = " ".join(f"[{a[:, d].min():+.2f},{a[:, d].max():+.2f}]" for d in range(a.shape[1]))
        print(f"   {key:8s} shape={a.shape} dtype={a.dtype}")
        print(f"     per-dim range: {rng}")
    img = np.asarray(fr["image"])
    print(f"   image    shape={img.shape} dtype={img.dtype} range=[{img.min()},{img.max()}]")


def _panel(img_rgb, wrist_rgb, label, color, show_wrist):
    p = cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)
    if show_wrist and wrist_rgb is not None:
        p = np.hstack([p, cv2.cvtColor(np.asarray(wrist_rgb), cv2.COLOR_RGB2BGR)])
    cv2.rectangle(p, (0, 0), (p.shape[1], 22), color, -1)
    cv2.putText(p, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--correction", required=True, help="Correction HDF5 (from collect_hitl_libero_pi0.py).")
    ap.add_argument("--correction-demo", type=int, default=0, help="Demo index in the correction file.")
    ap.add_argument("--libero-suite", required=True, help="LIBERO suite for the expert demo (e.g. libero_90).")
    ap.add_argument("--task-id", type=int, required=True, help="Expert task id (should match the corrections' task).")
    ap.add_argument("--expert-demo", type=int, default=0, help="Expert demo index within the task.")
    ap.add_argument("--libero-datasets-dir", default=os.path.join(REPO_ROOT, "third_party", "LIBERO", "libero", "datasets"))
    ap.add_argument("--libero-image-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="Flip LIBERO images to pi0 orientation (default on; must match the export setting).")
    ap.add_argument("--out", default=None, help="Output mp4 (default next to the correction file).")
    ap.add_argument("--show-wrist", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    exp_mod = _load_export_module()
    corr, corr_prompt, corr_name = _read_correction_demo(args.correction, args.correction_demo)
    exp, exp_prompt = _get_expert_demo(
        exp_mod, args.libero_datasets_dir, args.libero_suite, args.task_id, args.expert_demo, args.libero_image_flip
    )

    # ---- Numeric comparison ----
    print("=" * 70)
    print("NUMERIC COMPARISON")
    print(f"  correction prompt: {corr_prompt!r}")
    print(f"  expert     prompt: {exp_prompt!r}")
    if corr_prompt.strip().lower() != exp_prompt.strip().lower():
        print("  WARNING: prompts differ — are the correction task and --task-id the same task?")
    print("-" * 70)
    _numeric_report("CORRECTION", corr)
    _numeric_report("EXPERT", exp)
    keys = ["image", "state", "actions"]
    shapes_ok = all(np.asarray(corr[k]).shape[1:] == np.asarray(exp[k]).shape[1:] for k in keys)
    dtypes_ok = all(np.asarray(corr[k]).dtype == np.asarray(exp[k]).dtype for k in keys)
    print("-" * 70)
    print(f"  per-frame shapes identical: {shapes_ok} | dtypes identical: {dtypes_ok}")
    print(f"  => VERDICT: {'SAME format/space' if (shapes_ok and dtypes_ok) else 'MISMATCH — do not push'}")
    print("=" * 70)

    # ---- Side-by-side video ----
    nc, ne = len(corr["actions"]), len(exp["actions"])
    N = max(nc, ne)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.correction)),
        f"compare_{corr_name}_vs_{args.libero_suite}_t{args.task_id}_d{args.expert_demo}.mp4",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    frames = []
    for t in range(N):
        ci, ei = min(t, nc - 1), min(t, ne - 1)  # freeze the shorter one on its last frame
        clabel, ccolor = LABELS.get(int(corr["intervention"][ci]), ("?", (128, 128, 128)))
        cp = _panel(corr["image"][ci], corr["wrist_image"][ci] if corr["wrist_image"] is not None else None,
                    f"CORR t{ci} [{clabel}]", ccolor, args.show_wrist)
        ep = _panel(exp["image"][ei], exp["wrist_image"][ei], f"EXPERT t{ei} [OFFLINE]",
                    LABELS[2][1], args.show_wrist)
        divider = np.full((cp.shape[0], 4, 3), 255, dtype=np.uint8)
        frames.append(np.hstack([cp, divider, ep]))

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w, h))
    for fr in frames:
        writer.write(fr)
    writer.release()
    print(f"Wrote side-by-side video ({N} frames, corr={nc} / expert={ne}) to {out}")


if __name__ == "__main__":
    main()
