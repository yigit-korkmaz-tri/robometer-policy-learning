#!/usr/bin/env python3
"""Thin orchestrator for iterative pi0.5 HITL on LIBERO: collect -> export -> norm-stats -> train.

Each round runs four steps as SEPARATE subprocesses (so JAX training and robosuite teleop never share
a process), then hands the freshly fine-tuned checkpoint to the next round's data collection:

  1. collect  (root env):   scripts/collect_hitl_libero_pi0.py  -> round-R corrections HDF5
  2. export   (openpi env):  scripts/export_hitl_to_lerobot.py   -> LeRobot repo (all rounds + base demos)
  3. normstats(openpi env):  third_party/dsrl_openpi/scripts/compute_norm_stats.py
  4. train    (openpi env):  third_party/dsrl_openpi/scripts/train.py pi05_libero_hitl_lora (LoRA HG-DAgger)
  -> resolve the new checkpoint step dir; round R+1 collects with it (config_name=<train config>).

This is deliberately thin: it shells out with explicit commands (use ``--dry-run`` to print them and
tune before running). Prerequisites: (a) ``lerobot`` installed in the openpi env (this repo's openpi
copy pins it as a project dependency); (b) a display for the teleop collection step; (c) the base
pi0.5 checkpoint.

Round 0 collects with ``--base-pi0-checkpoint`` (default the LIBERO-tuned pi0.5) and trains initialized
from ``--init-weights``. Aggregate anti-forgetting demos with ``--base-demos`` (folded into every
round's LeRobot repo, since openpi trains a single repo). See [[dp-hgdagger-needs-aggregation]].

Example:
    uv run python scripts/hitl_pi0_loop.py --rounds 3 --task-id 57 \
        --collect-num-rollouts 10 --num-train-steps 3000 \
        --repo-id-prefix yourname/libero_hitl --exp-prefix hitl_t57 \
        --base-demos 'data/libero_t57_demos.hdf5'
"""

import argparse
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, *, cwd=None, dry, extra_env=None):
    prefix = " ".join(f"{k}={v}" for k, v in (extra_env or {}).items())
    printable = (prefix + " " if prefix else "") + " ".join(cmd)
    print(f"\n$ ({cwd or REPO_ROOT}) {printable}\n", flush=True)
    if dry:
        return
    env = {**os.environ, **extra_env} if extra_env else None
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True, env=env)


def _resolve_ckpt_step_dir(checkpoint_base_dir, config_name, exp_name):
    """Return the latest saved step dir under <base>/<config>/<exp> (orbax saves per-step subdirs)."""
    run_dir = os.path.join(checkpoint_base_dir, config_name, exp_name)
    steps = [d for d in glob.glob(os.path.join(run_dir, "*")) if os.path.basename(d).isdigit()]
    if not steps:
        raise FileNotFoundError(
            f"No numeric step checkpoints found under {run_dir}. Did training run and save a checkpoint?"
        )
    return max(steps, key=lambda d: int(os.path.basename(d)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--collect-num-rollouts", type=int, default=10)
    ap.add_argument("--num-train-steps", type=int, default=3000)
    ap.add_argument("--repo-id-prefix", required=True, help="LeRobot repo id prefix; round R -> <prefix>_r{R}.")
    ap.add_argument("--exp-prefix", required=True, help="openpi exp_name prefix; round R -> <prefix>_r{R}.")
    ap.add_argument("--train-config", default="pi05_libero_hitl_lora")
    ap.add_argument("--collect-config", default="libero_collect_hitl")
    ap.add_argument("--base-pi0-checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero/",
                    help="Round-0 collection policy checkpoint.")
    ap.add_argument("--base-pi0-config-name", default=None,
                    help="Round-0 collection config_name (null => path heuristic).")
    ap.add_argument("--init-weights", default="gs://openpi-assets/checkpoints/pi05_libero/params",
                    help="Round-0 training init params (weight loader).")
    ap.add_argument("--base-demos", nargs="*", default=[], help="Glob(s) for anti-forgetting demo HDF5s.")
    # On-the-fly LIBERO expert base demos (anti-forgetting) folded into every round's LeRobot repo.
    ap.add_argument("--libero-base-suite", default=None, help="LIBERO suite for on-the-fly base demos.")
    ap.add_argument("--libero-base-task-ids", type=int, nargs="*", default=[],
                    help="Task ids within --libero-base-suite for base demos (default: the eval --task-id).")
    ap.add_argument("--libero-base-num-demos", type=int, default=10, help="Expert demos per task.")
    ap.add_argument("--workdir", default=os.path.join(REPO_ROOT, "outputs", "hitl_pi0_loop"))
    ap.add_argument("--openpi-dir", default=os.path.join(REPO_ROOT, "third_party", "dsrl_openpi"))
    ap.add_argument("--checkpoint-base-dir", default=None,
                    help="openpi checkpoint_base_dir (default <openpi-dir>/checkpoints).")
    ap.add_argument("--weight-loader-flag", default="--weight-loader.params-path",
                    help="tyro flag to override the training init params per round (adjust if tyro differs).")
    ap.add_argument("--xla-mem-fraction", type=float, default=0.9,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION for JAX/GPU steps (JAX defaults to 0.75; 0.9 uses "
                         "more GPU memory to avoid OOM). Set <=0 to leave JAX's default.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument("--start-round", type=int, default=0, help="Resume from this round index.")
    args = ap.parse_args()

    ckpt_base = args.checkpoint_base_dir or os.path.join(args.openpi_dir, "checkpoints")
    os.makedirs(args.workdir, exist_ok=True)
    # Injected into JAX/GPU subprocess steps (collect, norm-stats, train) so JAX can use more GPU
    # memory than its 0.75 default -- the standard openpi OOM fix.
    jax_env = {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.xla_mem_fraction)} if args.xla_mem_fraction and args.xla_mem_fraction > 0 else None

    # Rolling state handed forward across rounds.
    collect_ckpt = args.base_pi0_checkpoint
    collect_cfg_name = args.base_pi0_config_name
    init_weights = args.init_weights
    round_hdf5s = []  # accumulate every round's corrections for aggregation

    for r in range(args.start_round, args.rounds):
        repo_id = f"{args.repo_id_prefix}_r{r}"
        exp_name = f"{args.exp_prefix}_r{r}"
        round_hdf5 = os.path.join(args.workdir, f"round_{r}_rollouts.hdf5")
        round_hdf5s.append(round_hdf5)
        print(f"\n========== HITL ROUND {r} ==========")

        # 1) Collect (root env, hydra overrides).
        collect_cmd = [
            "uv", "run", "python", "scripts/collect_hitl_libero_pi0.py",
            "--config-name", args.collect_config,
            f"env.task_id={args.task_id}",
            f"hitl.collect_num_rollouts={args.collect_num_rollouts}",
            f"hitl.collect_output_path={round_hdf5}",
            f"pi0.checkpoint={collect_ckpt}",
        ]
        if collect_cfg_name:
            collect_cmd.append(f"pi0.config_name={collect_cfg_name}")
        _run(collect_cmd, dry=args.dry_run, extra_env=jax_env)

        # openpi-env steps run with a plain `uv run` from third_party/dsrl_openpi. (These used to need
        # `--frozen` because `pyav` was unresolvable and re-locking bumped torch; both locks regenerate
        # cleanly now and `uv lock --check` passes, so the flag is no longer required. If you ever run
        # on a machine with a PARTIAL submodule checkout, set UV_FROZEN=1 instead -- uv validates every
        # path dependency in the lock, so an un-initialized submodule dir breaks a non-frozen sync.)
        # 2) Export/aggregate to a LeRobot repo (openpi env).
        export_cmd = [
            "uv", "run", "python", os.path.join(REPO_ROOT, "scripts", "export_hitl_to_lerobot.py"),
            "--inputs", *round_hdf5s,
            "--repo-id", repo_id,
        ]
        if args.base_demos:
            export_cmd += ["--base-demos", *args.base_demos]
        if args.libero_base_suite:
            # Default the base-demo task ids to the task being collected/trained on.
            lb_task_ids = args.libero_base_task_ids or [args.task_id]
            export_cmd += [
                "--libero-base-suite", args.libero_base_suite,
                "--libero-base-task-ids", *[str(t) for t in lb_task_ids],
                "--libero-base-num-demos", str(args.libero_base_num_demos),
            ]
        _run(export_cmd, cwd=args.openpi_dir, dry=args.dry_run)

        # 3) Norm stats (openpi env). compute_norm_stats takes --config-name + our added --repo-id
        # override (points at this round's exported repo).
        _run(
            ["uv", "run", "python", "scripts/compute_norm_stats.py",
             "--config-name", args.train_config, f"--repo-id={repo_id}"],
            cwd=args.openpi_dir, dry=args.dry_run, extra_env=jax_env,
        )

        # 4) Train (openpi env), initialized from the previous round's params. tyro flags are
        # hyphenated (see openpi README: --exp-name); --data.repo-id overrides the config's repo.
        _run(
            ["uv", "run", "python", "scripts/train.py", args.train_config,
             f"--exp-name={exp_name}", f"--data.repo-id={repo_id}",
             f"--num-train-steps={args.num_train_steps}",
             f"{args.weight_loader_flag}={init_weights}", "--overwrite"],
            cwd=args.openpi_dir, dry=args.dry_run, extra_env=jax_env,
        )

        # Hand the fresh checkpoint forward.
        if args.dry_run:
            step_dir = os.path.join(ckpt_base, args.train_config, exp_name, "<step>")
        else:
            step_dir = _resolve_ckpt_step_dir(ckpt_base, args.train_config, exp_name)
        print(f"Round {r} checkpoint: {step_dir}")
        collect_ckpt = step_dir
        collect_cfg_name = args.train_config  # fine-tuned LoRA arch => explicit config for Pi0Wrapper
        init_weights = os.path.join(step_dir, "params")

    print("\nHITL loop complete.")


if __name__ == "__main__":
    main()
