#!/usr/bin/env python3
"""Thin orchestrator for iterative pi0.5 HITL on LIBERO: collect -> export -> norm-stats -> train.

Each round runs its steps as SEPARATE subprocesses (so JAX training and robosuite teleop never share
a process), then hands the freshly fine-tuned checkpoint to the next round's data collection:

  0. eval     (root env):    scripts/eval_pi0_libero.py   -> success rate on the task suite (skip with --no-eval)
  1. collect  (root env):    scripts/collect_hitl_libero_pi0.py  -> per-task corrections HDF5 (one per task id)
  2. export   (openpi env):  scripts/export_hitl_to_lerobot.py   -> LeRobot repo (all rounds + base demos)
  3. normstats(openpi env):  third_party/dsrl_openpi/scripts/compute_norm_stats.py
  4. train    (openpi env):  third_party/dsrl_openpi/scripts/train.py pi05_libero_hitl_lora (LoRA HG-DAgger)
  -> resolve the new checkpoint step dir; round R+1 evals/collects with it (config_name=<train config>).

Works over a SUITE: pass a list of task ids (--task-ids); each round evals on all of them, collects
corrections for each, aggregates every task's corrections (plus optional base demos) into one LeRobot
repo, and trains a single policy. Eval runs at the START of each round, so round 0 measures the base
policy and later rounds measure the fine-tuned one -- a success-rate curve across rounds.

This is deliberately thin: it shells out with explicit commands (use ``--dry-run`` to print them and
tune before running). Prerequisites: (a) ``lerobot`` installed in the openpi env (this repo's openpi
copy pins it as a project dependency); (b) a display for the teleop collection step; (c) the base
pi0.5 checkpoint.

Round 0 collects with ``--base-pi0-checkpoint`` (default the LIBERO-tuned pi0.5) and trains initialized
from ``--init-weights``. Aggregate anti-forgetting demos with ``--base-demos`` (folded into every
round's LeRobot repo, since openpi trains a single repo). See [[dp-hgdagger-needs-aggregation]].

Example:
    uv run python scripts/hitl_pi0_loop.py --rounds 3 --env-name libero_90 --task-ids 57 58 59 \
        --collect-num-rollouts 5 --num-train-steps 3000 --eval-num-episodes 20 \
        --repo-id-prefix yourname/libero_hitl --exp-prefix hitl_t575859 \
        --libero-base-suite libero_90 --libero-base-num-demos 10  --train-config pi05_libero_hitl_lora
"""

import argparse
import glob
import os
import subprocess
import sys
import datetime

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
    ap.add_argument("--task-ids", type=int, nargs="+", required=True,
                    help="Task ids (from --env-name) to run each round: corrections are collected for "
                         "every task, aggregated into one LeRobot repo, and one policy is trained.")
    ap.add_argument("--env-name", default="libero_90", help="LIBERO suite for collection/eval.")
    ap.add_argument("--collect-num-rollouts", type=int, default=10, help="Rollouts collected PER task.")
    ap.add_argument("--num-train-steps", type=int, default=3000)
    # Per-round evaluation of the current policy on the task suite (round 0 = base policy).
    ap.add_argument("--eval-config", default="libero_eval")
    ap.add_argument("--eval-num-episodes", type=int, default=20, help="Eval episodes per task.")
    ap.add_argument("--no-eval", action="store_true", help="Skip the per-round eval step.")
    ap.add_argument("--repo-id-prefix", required=True, help="LeRobot repo id prefix; round R -> <prefix>_r{R}.")
    ap.add_argument("--exp-prefix", required=True, help="openpi exp_name prefix; round R -> <prefix>_r{R}.")
    ap.add_argument("--train-config", default="pi05_libero_hitl_lora")
    ap.add_argument("--collect-config", default="libero_collect_hitl")
    ap.add_argument("--store-only-human", type=str, default=None,
                    help="Store only human-correction steps during collection. Default (unset): derived "
                         "from --train-config (true for pi05_libero_hitl_lora / HG-DAgger, else false).")
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
    ap.add_argument("--workdir", default=os.path.join(REPO_ROOT, "outputs"))
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

    # When --store-only-human is unset, derive it from the train config.
    store_only_human = args.store_only_human
    if store_only_human is None:
        store_only_human = args.train_config == "pi05_libero_hitl_lora"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_base = args.checkpoint_base_dir or os.path.join(args.openpi_dir, "checkpoints")
    workdir = str(args.workdir) + "/" + args.train_config + "_" + timestamp
    os.makedirs(workdir, exist_ok=True)
    # Injected into JAX/GPU subprocess steps (collect, norm-stats, train) so JAX can use more GPU
    # memory than its 0.75 default -- the standard openpi OOM fix.
    jax_env = {"XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.xla_mem_fraction)} if args.xla_mem_fraction and args.xla_mem_fraction > 0 else None

    # Rolling state handed forward across rounds.
    collect_ckpt = args.base_pi0_checkpoint
    collect_cfg_name = args.base_pi0_config_name
    init_weights = args.init_weights
    round_hdf5s = []  # accumulate every round's corrections for aggregation

    # Resume: when starting past round 0, reconstruct the rolling state from the previous round's
    # checkpoint (so round <start_round> evals/collects with it and inits training from it) and
    # re-accumulate earlier rounds' correction HDF5s for aggregation. Without this, --start-round would
    # (wrongly) evaluate/collect with the BASE policy and train from base weights.
    if args.start_round > 0 and not args.dry_run:
        prev_exp = f"{args.exp_prefix}_r{args.start_round - 1}"
        prev_step = _resolve_ckpt_step_dir(ckpt_base, args.train_config, prev_exp)
        collect_ckpt = prev_step
        collect_cfg_name = args.train_config
        init_weights = os.path.join(prev_step, "params")
        for pr in range(args.start_round):
            for t in args.task_ids:
                p = os.path.join(workdir, f"round_{pr}_task_{t}_rollouts.hdf5")
                if os.path.exists(p):
                    round_hdf5s.append(p)
        print(f"Resuming at round {args.start_round}: eval/collect + train-init from {prev_step}; "
              f"aggregating {len(round_hdf5s)} prior correction file(s).")

    task_ids_str = "[" + ",".join(str(t) for t in args.task_ids) + "]"

    for r in range(args.start_round, args.rounds):
        repo_id = f"{args.repo_id_prefix}_r{r}"
        exp_name = f"{args.exp_prefix}_r{r}"
        print(f"\n========== HITL ROUND {r} ==========")

        # 0) Eval the current policy (round 0 = base) on the whole task suite, for a success-rate
        # curve across rounds. Root env; hydra.run.dir pins per-round eval_results.json.
        if not args.no_eval:
            eval_dir = os.path.join(workdir, f"eval_r{r}")
            eval_cmd = [
                "uv", "run", "python", "scripts/eval_pi0_libero.py",
                "--config-name", args.eval_config,
                f"env.env_name={args.env_name}",
                f"env.task_ids={task_ids_str}",
                f"eval.num_episodes={args.eval_num_episodes}",
                f"pi0.checkpoint={collect_ckpt}",
                f"hydra.run.dir={eval_dir}",
            ]
            if collect_cfg_name:
                eval_cmd.append(f"pi0.config_name={collect_cfg_name}")
            _run(eval_cmd, dry=args.dry_run, extra_env=jax_env)

        # 1) Collect corrections for EVERY task in the suite (root env, hydra overrides). One HDF5 per
        # (round, task); all rounds' HDF5s accumulate for aggregation in the export step.
        for t in args.task_ids:
            round_hdf5 = os.path.join(workdir, f"round_{r}_task_{t}_rollouts.hdf5")
            round_hdf5s.append(round_hdf5)
            collect_cmd = [
                "uv", "run", "python", "scripts/collect_hitl_libero_pi0.py",
                "--config-name", args.collect_config,
                f"env.env_name={args.env_name}",
                f"env.task_id={t}",
                f"hitl.collect_num_rollouts={args.collect_num_rollouts}",
                f"hitl.collect_output_path={round_hdf5}",
                f"pi0.checkpoint={collect_ckpt}",
                f"hitl.store_only_human={str(store_only_human).lower()}",
                f"hitl.rollout_pool_size={0 if args.train_config == 'pi05_libero_hitl_lora' else 15}",
            ]
            if collect_cfg_name:
                collect_cmd.append(f"pi0.config_name={collect_cfg_name}")
            _run(collect_cmd, dry=args.dry_run, extra_env=jax_env)

        # A collection step may legitimately write NO file (e.g. finished early with no interventions
        # under store_only_human). Only feed HDF5s that actually exist to the exporter so a missing
        # path never reaches conversion. (In --dry-run the files aren't created, so keep them all for
        # the printed command.)
        export_inputs = round_hdf5s if args.dry_run else [p for p in round_hdf5s if os.path.exists(p)]
        has_base = bool(args.base_demos or args.libero_base_suite)
        if not export_inputs and not has_base:
            print(f"Round {r}: no saved corrections and no base demos — skipping export/train.")
            continue

        # openpi-env steps run with `uv run --frozen` from third_party/dsrl_openpi -- pinning the lock
        # avoids torch/CUDA churn on re-resolve. (Both locks regenerate cleanly now, so --frozen is a
        # safe habit rather than a hard requirement; drop it only if you intend to re-lock.)
        # 2) Export/aggregate to a LeRobot repo (openpi env).
        export_cmd = [
            "uv", "run", "--frozen", "python",
            os.path.join(REPO_ROOT, "scripts", "export_hitl_to_lerobot.py"),
            "--repo-id", repo_id,
        ]
        if export_inputs:
            export_cmd += ["--inputs", *export_inputs]
        if args.base_demos:
            export_cmd += ["--base-demos", *args.base_demos]
        if args.libero_base_suite:
            # Default the base-demo task ids to the tasks being collected/trained on.
            lb_task_ids = args.libero_base_task_ids or args.task_ids
            export_cmd += [
                "--libero-base-suite", args.libero_base_suite,
                "--libero-base-task-ids", *[str(t) for t in lb_task_ids],
                "--libero-base-num-demos", str(args.libero_base_num_demos),
            ]
        _run(export_cmd, cwd=args.openpi_dir, dry=args.dry_run)

        # 3) Norm stats (openpi env). compute_norm_stats takes --config-name + our added --repo-id
        # override (points at this round's exported repo).
        _run(
            ["uv", "run", "--frozen", "python", "scripts/compute_norm_stats.py",
             "--config-name", args.train_config, f"--repo-id={repo_id}"],
            cwd=args.openpi_dir, dry=args.dry_run, extra_env=jax_env,
        )

        # 4) Train (openpi env), initialized from the previous round's params. tyro flags are
        # hyphenated (see openpi README: --exp-name); --data.repo-id overrides the config's repo.
        _run(
            ["uv", "run", "--frozen", "python", "scripts/train.py", args.train_config,
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

    # At the end of all rounds, eval the final policy one more time (unless --no-eval).
    if not args.no_eval:
        final_eval_r = args.rounds
        eval_dir = os.path.join(workdir, f"eval_r{final_eval_r}")
        eval_cmd = [
            "uv", "run", "python", "scripts/eval_pi0_libero.py",
            "--config-name", args.eval_config,
            f"env.env_name={args.env_name}",
            f"env.task_ids={task_ids_str}",
            f"eval.num_episodes={args.eval_num_episodes}",
            f"pi0.checkpoint={collect_ckpt}",
            f"hydra.run.dir={eval_dir}",
        ]
        if collect_cfg_name:
            eval_cmd.append(f"pi0.config_name={collect_cfg_name}")
        _run(eval_cmd, dry=args.dry_run, extra_env=jax_env)

    print("\nHITL loop complete.")


if __name__ == "__main__":
    main()
