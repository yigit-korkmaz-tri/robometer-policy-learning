#!/usr/bin/env python3
"""Hyperparameter-sweep launcher for Flow-MILE HITL runs (scripts/train_hitl.py).

A single-node, SLURM-with-GRES-style scheduler: it queues all sweep runs and packs as many as fit
onto the available GPUs by VRAM, backfilling the queue as runs finish. Each run gets its own Hydra
output dir + wandb name + log file, and a ``.done`` marker so re-invoking resumes (skips finished
runs). Two strategies:

  * coord (default): coordinate descent around BASELINE — vary ONE axis at a time over its SWEEP
    values, holding the others at BASELINE. Cheap (~1 + sum(len(vals)-1) runs) and great for finding
    the regime before committing to a grid. For the axes below that's ~10 runs.
  * grid: full Cartesian product of every SWEEP axis (the product of the SWEEP axis sizes; with the
    current axes that is 2*1*2*2*1*1*3*2*2 = 96 runs). Usually too many; use only for a focused grid
    over 2-3 axes after trimming SWEEP.

Resource packing (how many run at once):
  * --per-gpu K        : run K runs per GPU (DEFAULT 3 -> 12 concurrent on the 4x A10G g5.12xlarge,
                         since ~3 low-dim flow-MILE runs fit in 24 GB). Set 1 for whole-GPU/exclusive.
  * --mem-per-run-mb M : instead, pack each GPU with floor(free_MB * mem_fraction / M) runs by VRAM
                         ("as many as fit"); measure M from one job's nvidia-smi VRAM (+ ~15%).
  * --max-parallel     : optional global cap on top of the per-GPU capacity.
  * --gpus 0,1         : restrict to a subset (default: all GPUs nvidia-smi reports).
GPU free memory is read once at startup, so this assumes a dedicated instance.

The "warmup amount" axis maps to the offline-demo source:
    0      -> hitl.offline_mode=null               (no offline anchor)
    N>0    -> hitl.offline_mode=warmup hitl.warmup_dataset_path=WARMUP_PATHS[N]
    "self" -> hitl.offline_mode=self hitl.self_num_rollouts=SELF_NUM_ROLLOUTS
              (anchor = the initial policy's own successful autonomous rollouts; no demo H5 needed)
(generate the warmup H5s once with scripts/collect_warmup_demos.py).

Usage:
    # dry-run: print the planned runs + the computed GPU concurrency, without launching
    uv run python scripts/sweep_flow_mile.py --mode coord --dry-run
    # launch (defaults to 3 runs/GPU = 12 concurrent on 4x A10G)
    uv run python scripts/sweep_flow_mile.py --mode coord
    # whole-GPU exclusive (one run per GPU = 4 concurrent)
    uv run python scripts/sweep_flow_mile.py --mode coord --per-gpu 1
    # or pack by VRAM instead of a fixed count
    uv run python scripts/sweep_flow_mile.py --mode coord --mem-per-run-mb 7000
    # full grid (asks for confirmation)
    uv run python scripts/sweep_flow_mile.py --mode grid --yes
"""

import argparse
import itertools
import os
import subprocess
import sys
import time

# Terminate quietly (like normal Unix tools) when output is piped into a reader that closes early,
# e.g. `... --dry-run | head`, instead of raising BrokenPipeError.
try:
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (ImportError, AttributeError, ValueError):
    pass

# =====================================================================================
# Edit this block to configure the sweep.
# =====================================================================================

# How to invoke training (the python launcher + script + hydra config name).
PYTHON = ["uv", "run", "python"]
TRAIN_SCRIPT = "scripts/train_hitl.py"
CONFIG_NAME = "robomimic_hitl"

# Fixed overrides shared by every run (Hydra key=value). Edit load_dir/dataset paths here.
BASE_OVERRIDES = [
    "algorithm@offline_algorithm=flow_mile",
    "alg.offline_alg_name=flow_mile",
    "load_dir=checkpoints/square_low_dim/2026-06-22_15-58-08",
    "checkpoint=10000",
    "offline_algorithm.condition_intervention_on_action=true",
    "offline_algorithm.condition_nonintervention_on_robot=true",
    "offline_algorithm.batch_size=256",
    "offline_algorithm.expected_rollout_score_weight=5",
    # "offline_algorithm.log_sample_metrics_every=200",
    # Standardize the probit score gap so intervention_cost / lambda are on a consistent scale across
    # runs (recommended when sweeping cost; comment out to sweep on the raw score scale).
    # "offline_algorithm.normalize_score_gaps=true",
    "hitl.segment_by_intervention=false",
    "hitl.num_iterations=1",
    "hitl.rollouts_per_iter=0",
    "hitl.train_steps_per_iter=30000",
    "hitl.save_interval=1",
    "eval.eval_freq=3000",
    "eval.eval_num_episodes=50",
    "eval.eval_num_envs=5",
    "eval.eval_vectorization=async",
]


# Swept axes -> candidate values.
# SWEEP = {
#     "lr": [1e-4, 5e-5, 1e-5],
#     "lambda": [1.0, 0.5, 0.1],
#     "cost": [1.0, 0.5, 0.0],
#     "scale": [1.0, 2.0, 5.0],     # probit scale beta
#     "anchor": [0.1],
#     "warmup": [30, 10, "self"],   # 30 = BASELINE; ablations: 10-demo warmup, "self" (self-collected anchor)
# }
SWEEP = {
    "lr": [1e-4, 5e-5],
    "lambda": [0.5],
    "cost": [0.0, 0.5],
    "scale": [1.0],     # probit scale beta
    "anchor": [0.1],
    "warmup": [0],   # 30 = BASELINE; ablations: 10-demo warmup, "self" (self-collected anchor)
    "demos": [30],   # precollected_hitl_dataset intervention-demo count (see DEMO_PATHS)
    "mc_steps": [10],     # offline_algorithm.mc_num_inference_steps (Euler steps for MC sampling)
    "prox": [0.0],      # offline_algorithm.proximal_loss_weight
}


# Center point for coordinate-descent mode (one value per axis; must be present in SWEEP[axis]).
BASELINE = {
    "lr": 1e-4,
    "lambda": 0.1,
    "cost": 0.5,
    "scale": 1.0,
    "anchor": 0.1,
    "warmup": 30,
    "demos": 30,
    "mc_steps": 10,
    "prox": 0.1,
}

# Hydra key for each non-warmup, non-demos axis (warmup + demos map to file paths below).
AXIS_TO_KEY = {
    "lr": "offline_algorithm.actor_optimizer_lr",
    "lambda": "offline_algorithm.lambda_intervention",
    "cost": "offline_algorithm.intervention_cost",
    "scale": "offline_algorithm.probit_scale",
    "anchor": "offline_algorithm.anchor_loss_weight",
    "mc_steps": "offline_algorithm.mc_num_inference_steps",
    "prox": "offline_algorithm.proximal_loss_weight",
}

# warmup amount -> demo H5 (0 means "no offline data"). Generate these once with
# scripts/collect_warmup_demos.py, then point each entry at the right file.
WARMUP_PATHS = {
    0: None,
    10: "data/warmup_square_10.hdf5",
    30: "data/warmup_square_30.hdf5",
}

# demos amount -> precollected HITL dataset H5 (the "demos" sweep axis -> hitl.precollected_hitl_dataset).
# Point each entry at the right file for the intervention-demo count.
DEMO_PATHS = {
    10: "data/real_square_hitl_rollouts_flow_10_demos.hdf5",
    30: "data/real_square_hitl_rollouts_flow_30_demos.hdf5",
    50: "data/real_square_hitl_rollouts_flow_50_demos.hdf5",
}

# Number of successful autonomous rollouts collected from the initial policy for the "self" anchor
# (warmup=="self" -> hitl.offline_mode=self hitl.self_num_rollouts=SELF_NUM_ROLLOUTS).
SELF_NUM_ROLLOUTS = 10

# Where per-run Hydra output dirs + logs go, and the wandb name prefix.
SWEEP_DIR = "sweeps/flow_mile_square"
WANDB_PREFIX = "fm_sweep"

# =====================================================================================


def _tag(axis, value):
    """Filesystem-safe short tag for an axis value (used in run names)."""
    if axis == "lr":
        return f"lr{value:.0e}"               # 1e-04 / 5e-05 / 1e-05
    if axis == "warmup":
        return "self" if value == "self" else f"warm{int(value)}"
    if axis == "demos":
        return f"demo{int(value)}"
    if axis == "mc_steps":
        return f"mc{int(value)}"
    short = {"lambda": "lam", "cost": "cost", "scale": "scale", "anchor": "anc", "prox": "prox"}[axis]
    return f"{short}{value:g}"


def run_name(cfg):
    return "_".join(
        _tag(a, cfg[a])
        for a in ("lr", "lambda", "cost", "scale", "anchor", "warmup", "demos", "mc_steps", "prox")
    )


def overrides_for(cfg):
    """Hydra overrides for one sweep point (the swept axes + warmup/demos path mappings)."""
    ov = [f"{AXIS_TO_KEY[a]}={cfg[a]}" for a in ("lr", "lambda", "cost", "scale", "anchor", "mc_steps", "prox")]

    # Precollected HITL dataset (intervention-demo count).
    demos = int(cfg["demos"])
    demo_path = DEMO_PATHS.get(demos)
    if not demo_path:
        raise ValueError(f"DEMO_PATHS has no H5 for demos={demos}; set it (or remove {demos} from SWEEP).")
    ov.append(f"hitl.precollected_hitl_dataset={demo_path}")

    warm = cfg["warmup"]
    if warm == "self":
        ov += ["hitl.offline_mode=self", f"hitl.self_num_rollouts={SELF_NUM_ROLLOUTS}"]
    elif int(warm) == 0:
        ov.append("hitl.offline_mode=null")
    else:
        warm = int(warm)
        path = WARMUP_PATHS.get(warm)
        if not path:
            raise ValueError(f"WARMUP_PATHS has no H5 for warmup={warm}; set it (or remove {warm} from SWEEP).")
        ov += ["hitl.offline_mode=warmup", f"hitl.warmup_dataset_path={path}"]
    return ov


def generate_runs(mode):
    """Return a list of sweep-point dicts for the chosen strategy (deduped, baseline first)."""
    axes = ["lr", "lambda", "cost", "scale", "anchor", "warmup", "demos", "mc_steps", "prox"]
    if mode == "grid":
        combos = itertools.product(*(SWEEP[a] for a in axes))
        return [dict(zip(axes, c)) for c in combos]

    # coordinate descent: baseline, then vary one axis at a time.
    for a in axes:
        if BASELINE[a] not in SWEEP[a]:
            raise ValueError(f"BASELINE[{a}]={BASELINE[a]} is not in SWEEP[{a}]={SWEEP[a]}.")
    seen, runs = set(), []
    for cfg in [dict(BASELINE)] + [
        {**BASELINE, a: v} for a in axes for v in SWEEP[a] if v != BASELINE[a]
    ]:
        key = tuple(cfg[a] for a in axes)
        if key not in seen:
            seen.add(key)
            runs.append(cfg)
    return runs


def build_command(cfg):
    name = run_name(cfg)
    out_dir = os.path.join(SWEEP_DIR, name)
    cmd = (
        PYTHON
        + [TRAIN_SCRIPT, "--config-name", CONFIG_NAME]
        + BASE_OVERRIDES
        + overrides_for(cfg)
        + [
            f"logging.wandb_name={WANDB_PREFIX}_{name}",
            f"hydra.run.dir={out_dir}",
        ]
    )
    return name, out_dir, cmd


def validate_warmup_paths(runs):
    """Fail early if any needed warmup H5 is missing."""
    missing = []
    for cfg in runs:
        if cfg["warmup"] == "self":
            continue  # self anchor collects rollouts at runtime; no warmup H5 needed
        warm = int(cfg["warmup"])
        if warm != 0:
            p = WARMUP_PATHS.get(warm)
            if not p or not os.path.exists(p):
                missing.append((warm, p))
    if missing:
        uniq = sorted(set(missing))
        raise FileNotFoundError(
            "Missing warmup demo H5(s): "
            + ", ".join(f"warmup={w} -> {p!r}" for w, p in uniq)
            + ". Generate them with scripts/collect_warmup_demos.py first (or edit WARMUP_PATHS / SWEEP)."
        )


def validate_demo_paths(runs):
    """Fail early if any needed precollected HITL dataset H5 (the "demos" axis) is missing."""
    missing = []
    for cfg in runs:
        demos = int(cfg["demos"])
        p = DEMO_PATHS.get(demos)
        if not p or not os.path.exists(p):
            missing.append((demos, p))
    if missing:
        uniq = sorted(set(missing))
        raise FileNotFoundError(
            "Missing precollected HITL dataset H5(s): "
            + ", ".join(f"demos={d} -> {p!r}" for d, p in uniq)
            + ". Set the correct paths in DEMO_PATHS (or remove those counts from SWEEP)."
        )


def query_gpus(gpu_filter):
    """Return {gpu_id: (total_mb, free_mb)} from nvidia-smi (filtered to gpu_filter if given).

    Returns {} when nvidia-smi is unavailable / reports no GPUs, so the caller can fall back to a
    single CPU lane.
    """
    import shutil

    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return {}
    info = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, total, free = (x.strip() for x in line.split(","))
        info[int(idx)] = (int(float(total)), int(float(free)))
    if gpu_filter:
        info = {g: info[g] for g in gpu_filter if g in info}
    return info


def compute_capacity(args):
    """Return (gpu_list, cap_per_gpu_dict). Reads GPU free VRAM once at startup."""
    gpu_filter = [int(g) for g in args.gpus.split(",") if g.strip()] if args.gpus else None
    info = query_gpus(gpu_filter)
    if not info:
        print("No nvidia-smi/GPU detected -> single CPU lane (per_gpu).")
        return [None], {None: max(1, args.per_gpu)}

    cap = {}
    print("GPU capacity (free VRAM at startup):")
    for g, (total, free) in info.items():
        if args.mem_per_run_mb:
            cap[g] = max(1, int((free * args.mem_fraction) // args.mem_per_run_mb))
        else:
            cap[g] = max(1, args.per_gpu)
        print(f"  gpu {g}: total={total}MB free={free}MB -> cap={cap[g]}")
    return list(info.keys()), cap


def schedule(runs, args):
    """SLURM-with-GRES-style scheduler: pack as many runs as fit per GPU, backfill as they finish."""
    # Build the pending FIFO queue, skipping already-completed runs (resume).
    pending, skipped = [], 0
    for cfg in runs:
        name, out_dir, cmd = build_command(cfg)
        if os.path.exists(os.path.join(out_dir, ".done")) and not args.force:
            skipped += 1
            continue
        pending.append((name, out_dir, cmd))
    if skipped:
        print(f"Resuming: skipping {skipped} run(s) with a .done marker (use --force to re-run).")

    gpu_list, cap = compute_capacity(args)
    total_cap = sum(cap.values())
    if args.max_parallel:
        total_cap = min(total_cap, args.max_parallel)
    print(f"Total concurrency: {total_cap} | queued: {len(pending)}\n")

    running = []  # {name, out_dir, popen, gpu, log}
    results = []
    t0 = time.time()

    def gpu_load(g):
        return sum(1 for r in running if r["gpu"] == g)

    def pick_gpu():
        free_gpus = [g for g in gpu_list if gpu_load(g) < cap[g]]
        return min(free_gpus, key=gpu_load) if free_gpus else None

    while pending or running:
        # Reap finished jobs.
        for r in list(running):
            rc = r["popen"].poll()
            if rc is None:
                continue
            status = "ok" if rc == 0 else f"FAILED(rc={rc})"
            if rc == 0:
                open(os.path.join(r["out_dir"], ".done"), "w").close()
            r["log"].close()
            running.remove(r)
            results.append((r["name"], status))
            print(f"[done ] {r['name']}: {status}  ({len(pending)} queued, {len(running)} running)")

        # Dispatch as many as fit.
        while pending and len(running) < total_cap:
            g = pick_gpu()
            if g is None:
                break
            name, out_dir, cmd = pending.pop(0)
            os.makedirs(out_dir, exist_ok=True)
            env = os.environ.copy()
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            log = open(os.path.join(out_dir, "run.log"), "w")
            log.write("# " + " ".join(cmd) + "\n\n")
            log.flush()
            popen = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append({"name": name, "out_dir": out_dir, "popen": popen, "gpu": g, "log": log})
            print(f"[start] {name}" + (f" (gpu {g})" if g is not None else "")
                  + f"  ({len(running)}/{total_cap} running, {len(pending)} queued)")

        if pending or running:
            time.sleep(args.poll)

    dt = time.time() - t0
    ok = sum(1 for _, s in results if s == "ok")
    failed = [n for n, s in results if s.startswith("FAILED")]
    print(f"\nSweep finished in {dt / 60:.1f} min: {ok} ok, {skipped} skipped, {len(failed)} failed.")
    for n in failed:
        print(f"  FAILED: {n}  (see {os.path.join(SWEEP_DIR, n, 'run.log')})")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description="Flow-MILE HITL hyperparameter sweep launcher (GPU-packing scheduler).")
    ap.add_argument("--mode", choices=["coord", "grid"], default="coord",
                    help="coord = coordinate descent around BASELINE (cheap); grid = full product.")
    ap.add_argument("--mem-per-run-mb", type=int, default=0,
                    help="VRAM per run (MB). If set, pack each GPU with floor(free*mem_fraction/this) runs "
                         "('as many as fit'). Measure via nvidia-smi on one run (+ ~15%% headroom).")
    ap.add_argument("--per-gpu", type=int, default=2,
                    help="Runs per GPU when --mem-per-run-mb is unset (default 3 = 3 jobs/GPU; on the "
                         "4x A10G g5.12xlarge that is 12 concurrent). Set 1 for whole-GPU/exclusive.")
    ap.add_argument("--mem-fraction", type=float, default=0.9,
                    help="Fraction of each GPU's free VRAM usable for packing (default 0.9; leaves headroom).")
    ap.add_argument("--max-parallel", type=int, default=0,
                    help="Optional global cap on concurrent runs (0 = no cap; use full per-GPU capacity).")
    ap.add_argument("--gpus", default="",
                    help="Comma-separated GPU ids to use, e.g. '0,1,2' (default: all GPUs nvidia-smi reports).")
    ap.add_argument("--poll", type=float, default=10.0, help="Scheduler poll interval in seconds (default 10).")
    ap.add_argument("--dry-run", action="store_true", help="Print the planned runs + capacity, then exit.")
    ap.add_argument("--force", action="store_true", help="Re-run even if a run's .done marker exists.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt for large sweeps.")
    args = ap.parse_args()

    runs = generate_runs(args.mode)
    validate_warmup_paths(runs)
    validate_demo_paths(runs)

    print(f"Sweep mode={args.mode}: {len(runs)} runs")
    for cfg in runs:
        print("  -", run_name(cfg))

    if args.dry_run:
        compute_capacity(args)
        print("\n--- commands ---")
        for cfg in runs:
            _, out_dir, cmd = build_command(cfg)
            print(f"\n# {run_name(cfg)}  (out: {out_dir})")
            print(" ".join(cmd))
        return

    if len(runs) > 20 and not args.yes:
        resp = input(f"\nLaunch {len(runs)} runs? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    sys.exit(schedule(runs, args))


if __name__ == "__main__":
    main()
