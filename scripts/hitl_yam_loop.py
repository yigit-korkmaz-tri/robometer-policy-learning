#!/usr/bin/env python3
"""Iterative HG-DAgger / Flow-MILE on the REAL YAM rig: collect -> convert -> export -> train.

The real-robot counterpart of ``scripts/hitl_pi0_loop.py`` (LIBERO). Same idea -- each round
collects human corrections with the current policy, aggregates every round collected so far into
one LeRobot repo, fine-tunes, and hands the fresh checkpoint to the next round -- but the collection
step is a person at the rig instead of a simulator, so two steps are operator-gated:

  0. [PROMPT] start the policy server for this round's checkpoint (owns the GPU)
  1. [RIG]    rd infer --intervene --session   (raiden; operator ends the session with Ctrl+C)
  2. [PROMPT] stop the policy server, freeing the GPU for training
  3. [RIG]    rd convert --task <task> --append          raw SVO2 -> rgb/lowdim episodes
  4. [LOCAL]  convert_yam_data_to_lerobot.py --args.with-intervention   -> LeRobot repo <prefix>_r{R}
  5. [LOCAL|EC2] compute_norm_stats.py --config-name <cfg> --repo-id <repo>
  6. [LOCAL|EC2] train.py <cfg> --data.repo-id <repo> --weight-loader.params-path <round R-1 params>
  7.          resolve the new step dir -> next round serves it and inits training from it
  8. [PROMPT] print the eval command; you run it and press Enter to start the next round

DATA AGGREGATION is positional, not bookkept: every round records into the SAME
``<data-dir>/raw/<task>`` and step 4 re-exports that whole directory, so round R trains on rounds
0..R plus any ``--offline-dirs`` (anti-forgetting teleop demos, labelled 2). That also makes
``--start-round`` trivial -- the data is already on disk, so resuming only has to re-resolve the
previous round's checkpoint. Note the flip side: step 4 re-encodes every episode each round, so
export time grows linearly with rounds.

INTERVENTION LABELS. ``rd infer --intervene`` writes a per-frame ``control_source`` (0=policy,
1=human), ``rd convert`` carries it into each frame's lowdim pickle, and the exporter turns it into
the LeRobot ``intervention`` feature. ``pi05_yam_mugontree_hitl_lora`` drops it at repack (plain
aggregation BC); ``pi05_yam_mugontree_flow_mile_lora`` consumes it in the MILE intervention probit.

Example (local training):
    uv run python scripts/hitl_yam_loop.py --rounds 3 \
        --task mugontree_dagger --prompt "hang the mug on mug tree" \
        --repo-id-prefix ykorkmaz/yam_mug_hitl --exp-prefix mug_hitl \
        --base-checkpoint ~/robometer-policy-learning/third_party/dsrl_openpi/checkpoints/pi05_yam_mugontree_lora/mug_on_tree_lora/25000 \
        --base-serve-config pi05_yam_mugontree_lora \
        --offline-dirs ~/raiden_internal/data/processed/hang_mug_on_mug_tree \
        --num-train-steps 3000

``--train-config pi05_yam_mugontree_flow_mile_lora`` REQUIRES ``--compute ec2``: the MILE objective
keeps a frozen rollout-policy copy resident and does not fit the rig's 24 GB card (measured: a
41 GiB peak at batch_size=1).

Same run with fine-tuning on EC2 (collection and export stay local -- the episodes live on the rig):
    ... --compute ec2 --ec2-host ubuntu@10.161.51.28 \
        --ec2-repo /opt/dlami/nvme/robometer-policy-learning --ssh-key ~/.ssh/yigit.pem

``--dry-run`` prints every local and remote command (and both rsyncs) and skips all prompts.
"""

import argparse
import datetime
import glob
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENPI_DIR = REPO_ROOT / "third_party" / "dsrl_openpi"
YAM_EXPORTER = "yam_dataset_builder/yam_dataset_builder/convert_yam_data_to_lerobot.py"
BRIDGE = "raiden.bridges.openpi_bridge:OpenPIBridge"


# --------------------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------------------


def _show(where: str, text: str) -> None:
    print(f"\n$ [{where}] {text}\n", flush=True)


def run_local(cmd, *, cwd, dry: bool, env: dict | None = None, operator_ends: bool = False) -> None:
    """Run a command locally, aborting the loop if it fails.

    ``operator_ends``: the child is a rig session that the operator terminates with Ctrl+C (``rd
    infer --session``). Ctrl+C goes to the whole foreground process group, so the orchestrator
    ignores SIGINT for the duration -- otherwise ending an episode session also kills the loop. The
    child must NOT inherit that: an ignored disposition survives exec (``restore_signals`` only
    resets SIGPIPE/SIGXFZ/SIGXFSZ), so it is explicitly reset to SIG_DFL in the child, which raiden
    then replaces with its own handler. raiden catches KeyboardInterrupt, homes the arms, saves the
    episode and exits 0, so a clean end is indistinguishable from a normal exit here.
    """
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
    _show(f"LOCAL {cwd}", (prefix + " " if prefix else "") + shlex.join(str(c) for c in cmd))
    if dry:
        return
    full_env = {**os.environ, **(env or {})} if env else None
    if not operator_ends:
        result = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=full_env)
    else:
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd],
                cwd=str(cwd),
                env=full_env,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),  # noqa: PLW1509
            )
            proc.wait()
            result = subprocess.CompletedProcess(cmd, proc.returncode)
        finally:
            signal.signal(signal.SIGINT, previous)
    if result.returncode != 0:
        sys.exit(f"\nLOCAL step failed (exit {result.returncode}): {shlex.join(str(c) for c in cmd)}")


def _ssh_base(args) -> list[str]:
    ssh = ["ssh"]
    if args.ssh_key:
        ssh += ["-i", str(args.ssh_key)]
    if args.ssh_opts:
        ssh += shlex.split(args.ssh_opts)
    return ssh + [args.ec2_host]


def run_remote(args, command: str, *, dry: bool, capture: bool = False) -> str:
    """Run a shell command on the EC2 box. Non-interactive SSH has no uv on PATH -- hence the
    export -- and every openpi env var is set per command rather than trusted to remote dotfiles."""
    full = f'export PATH="$HOME/.local/bin:$PATH"; {command}'
    if not capture:
        _show(f"EC2 {args.ec2_host}", full)
    if dry:
        return ""
    proc = subprocess.run(_ssh_base(args) + [full], capture_output=capture, text=True)
    if proc.returncode != 0:
        sys.exit(f"\nREMOTE step failed (exit {proc.returncode}): {command}")
    return (proc.stdout or "").strip()


def rsync(src: str, dst: str, *, args, dry: bool, delete: bool = False) -> None:
    cmd = ["rsync", "-az"]
    if delete:
        cmd.append("--delete")
    if args.ssh_key:
        cmd += ["-e", f"ssh -i {args.ssh_key}"]
    cmd += [src, dst]
    _show("RSYNC", shlex.join(cmd))
    if dry:
        return
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"\nrsync failed: {src} -> {dst}")


def prompt(message: str, *, dry: bool) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{message}\n{bar}", flush=True)
    if dry:
        return
    input("[Enter to continue] ")


# --------------------------------------------------------------------------------------
# Checkpoint resolution
# --------------------------------------------------------------------------------------


def resolve_step_dir(args, config_name: str, exp_name: str, *, remote: bool, dry: bool) -> str:
    """Newest numeric step dir under <ckpt base>/<config>/<exp> (orbax saves one per step)."""
    if remote:
        run_dir = f"{args.remote_ckpt_base}/{config_name}/{exp_name}"
        if dry:
            return f"{run_dir}/<step>"
        step = run_remote(
            args,
            f"ls -1 '{run_dir}' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1",
            dry=False,
            capture=True,
        )
        if not step:
            sys.exit(f"No numeric step checkpoint under EC2:{run_dir} (did training save?)")
        return f"{run_dir}/{step}"

    run_dir = Path(args.local_ckpt_base) / config_name / exp_name
    if dry:
        return str(run_dir / "<step>")
    steps = [d for d in glob.glob(str(run_dir / "*")) if Path(d).name.isdigit()]
    if not steps:
        sys.exit(f"No numeric step checkpoints under {run_dir}. Did training run and save?")
    return max(steps, key=lambda d: int(Path(d).name))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--start-round", type=int, default=0, help="Resume at this round index.")
    ap.add_argument("--task", required=True, help="raiden dagger task name (data/raw/<task>).")
    ap.add_argument(
        "--prompt",
        required=True,
        help="Language instruction. Feeds the bridge's prompt, --dagger-instruction and (via the "
        "recording metadata) the LeRobot task string -- one flag so they cannot disagree.",
    )
    ap.add_argument("--repo-id-prefix", required=True, help="Round R -> <prefix>_r{R}.")
    ap.add_argument("--exp-prefix", required=True, help="openpi exp_name; round R -> <prefix>_r{R}.")
    ap.add_argument("--train-config", default="pi05_yam_mugontree_hitl_lora")
    ap.add_argument(
        "--base-checkpoint",
        required=True,
        help="Round-0 collection policy: the step dir served before any fine-tuning.",
    )
    ap.add_argument(
        "--base-serve-config",
        default=None,
        help="--policy.config for the round-0 checkpoint (default: --train-config). Later rounds "
        "always serve --train-config, since that is the arch they were trained as.",
    )
    ap.add_argument(
        "--init-weights",
        default="gs://openpi-assets/checkpoints/pi05_base/params",
        help="Round-0 training init params (later rounds init from the previous round).",
    )
    ap.add_argument("--num-train-steps", type=int, default=3000)
    ap.add_argument(
        "--offline-dirs",
        nargs="*",
        default=[],
        help="Processed dirs folded into every round's repo and labelled offline (2) -- "
        "anti-forgetting teleop demos.",
    )
    # Rig / raiden
    ap.add_argument("--raiden-repo", default=str(Path.home() / "raiden_internal"))
    ap.add_argument(
        "--raiden-data-dir",
        default=None,
        help="raiden --data-dir (default <raiden-repo>/data). One cumulative dir for all rounds.",
    )
    ap.add_argument("--raiden-cmd", default="uv run rd", help="How to invoke raiden's CLI.")
    ap.add_argument("--action-hz", type=float, default=30.0, help="Match the training data fps.")
    ap.add_argument("--serve-port", type=int, default=8000)
    ap.add_argument(
        "--w-intervention",
        type=float,
        default=1.0,
        help="rd convert per-sample loss weight on human-takeover frames (1.0 = no upweighting).",
    )
    # openpi
    ap.add_argument("--openpi-dir", default=str(OPENPI_DIR))
    ap.add_argument("--local-ckpt-base", default=None, help="Default <openpi-dir>/checkpoints.")
    ap.add_argument("--lerobot-home", default=None, help="Default $HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot.")
    ap.add_argument("--xla-mem-fraction", type=float, default=0.9)
    # Compute placement
    ap.add_argument("--compute", choices=["local", "ec2"], default="local",
                    help="Where norm-stats + training run. Collection and export are always local.")
    ap.add_argument("--ec2-host", default=None, help="user@host (required for --compute ec2).")
    ap.add_argument("--ec2-repo", default=None, help="Repo path on EC2.")
    ap.add_argument("--remote-openpi-dir", default=None, help="Default <ec2-repo>/third_party/dsrl_openpi.")
    ap.add_argument("--remote-ckpt-base", default=None, help="Default <remote-openpi-dir>/checkpoints.")
    ap.add_argument("--remote-lerobot-home", default="/opt/dlami/nvme/cache/huggingface/lerobot")
    ap.add_argument("--remote-openpi-data-home", default="/opt/dlami/nvme/cache/openpi")
    ap.add_argument("--remote-uv-cache-dir", default="/opt/dlami/nvme/cache/uv")
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--ssh-opts", default="")
    ap.add_argument("--wandb-api-key", default=os.environ.get("WANDB_API_KEY", ""))
    ap.add_argument("--fsdp-devices", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    args.base_serve_config = args.base_serve_config or args.train_config
    args.raiden_data_dir = args.raiden_data_dir or str(Path(args.raiden_repo) / "data")
    args.local_ckpt_base = args.local_ckpt_base or str(Path(args.openpi_dir) / "checkpoints")
    args.lerobot_home = args.lerobot_home or os.environ.get(
        "HF_LEROBOT_HOME", str(Path.home() / ".cache/huggingface/lerobot")
    )
    if "flow_mile" in args.train_config and args.compute == "local":
        # Not a hard error -- "local" may be a bigger box than the rig -- but on the 3090 this
        # wedges in XLA compilation rather than failing fast, which is a nasty way to lose a round.
        print(
            f"WARNING: {args.train_config} keeps a frozen rollout policy resident and needs ~41 GiB; "
            "it does not fit the rig's 24 GB card. Use --compute ec2 unless this host has a much "
            "larger GPU.",
            file=sys.stderr,
        )
    if args.compute == "ec2":
        if not (args.ec2_host and args.ec2_repo):
            ap.error("--compute ec2 requires --ec2-host and --ec2-repo")
        args.remote_openpi_dir = args.remote_openpi_dir or f"{args.ec2_repo}/third_party/dsrl_openpi"
        args.remote_ckpt_base = args.remote_ckpt_base or f"{args.remote_openpi_dir}/checkpoints"
    return args


# --------------------------------------------------------------------------------------
# Round steps
# --------------------------------------------------------------------------------------


def serve_command(args, config_name: str, ckpt: str) -> str:
    return (
        f"cd {args.openpi_dir} && uv run --no-sync python scripts/serve_policy.py policy:checkpoint "
        f"--policy.config={config_name} --policy.dir={ckpt} --port={args.serve_port}"
    )


def collect(args, r: int) -> None:
    cmd = shlex.split(args.raiden_cmd) + [
        "infer",
        "--bridge", BRIDGE,
        "--ckpt-path", f"localhost:{args.serve_port}",
        "--bridge-kwargs", json.dumps({"prompt": args.prompt}),
        "--action-type", "joint",
        "--action-hz", str(args.action_hz),
        # Native frames: the bridge letterboxes them itself with openpi's own resize_with_pad.
        "--resize-images", "",
        "--no-depth",
        "--intervene", "--session",
        "--dagger-task", args.task,
        "--dagger-instruction", args.prompt,
        "--data-dir", args.raiden_data_dir,
    ]
    print(
        f"\n>>> ROUND {r} COLLECTION. Enter starts an episode, RIGHT pedal / Ctrl+N ends it, "
        "s saves / d discards. Ctrl+C when you have enough episodes for this round.",
        flush=True,
    )
    run_local(cmd, cwd=args.raiden_repo, dry=args.dry_run, operator_ends=True)


def export(args, repo_id: str) -> None:
    processed = str(Path(args.raiden_data_dir) / "processed" / args.task)
    cmd = [
        "uv", "run", "--frozen", "python", YAM_EXPORTER,
        "--args.raw-dirs", processed,
    ]
    if args.offline_dirs:
        cmd += ["--args.offline-dirs", *args.offline_dirs]
    cmd += ["--args.with-intervention", "--args.repo-id", repo_id, "--args.overwrite"]
    run_local(cmd, cwd=args.openpi_dir, dry=args.dry_run, env={"HF_LEROBOT_HOME": args.lerobot_home})


def train_local(args, repo_id: str, exp_name: str, init_params: str) -> None:
    env = {
        "HF_LEROBOT_HOME": args.lerobot_home,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.xla_mem_fraction),
    }
    run_local(
        ["uv", "run", "--frozen", "python", "scripts/compute_norm_stats.py",
         "--config-name", args.train_config, f"--repo-id={repo_id}"],
        cwd=args.openpi_dir, dry=args.dry_run, env=env,
    )
    cmd = ["uv", "run", "--frozen", "python", "scripts/train.py", args.train_config,
           f"--exp-name={exp_name}", f"--data.repo-id={repo_id}",
           f"--num-train-steps={args.num_train_steps}",
           f"--weight-loader.params-path={init_params}", "--overwrite"]
    if not args.wandb_api_key:
        cmd.append("--no-wandb-enabled")
    else:
        env = {**env, "WANDB_API_KEY": args.wandb_api_key}
    run_local(cmd, cwd=args.openpi_dir, dry=args.dry_run, env=env)


def train_remote(args, repo_id: str, exp_name: str, init_params: str) -> None:
    # The episodes live on the rig, so the export ran locally; ship the finished repo up.
    local_ds = f"{args.lerobot_home}/{repo_id}"
    remote_ds = f"{args.remote_lerobot_home}/{repo_id}"
    run_remote(args, f"mkdir -p '{remote_ds}'", dry=args.dry_run)
    rsync(f"{local_ds}/", f"{args.ec2_host}:{remote_ds}/", args=args, dry=args.dry_run, delete=True)

    env = (
        f"HF_LEROBOT_HOME='{args.remote_lerobot_home}' "
        f"OPENPI_DATA_HOME='{args.remote_openpi_data_home}' "
        f"UV_CACHE_DIR='{args.remote_uv_cache_dir}' "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={args.xla_mem_fraction} "
    )
    if args.wandb_api_key:
        env += f"WANDB_API_KEY={args.wandb_api_key} "
    wandb_flag = "" if args.wandb_api_key else " --no-wandb-enabled"
    run_remote(
        args,
        f"cd '{args.remote_openpi_dir}' && {env}uv run --frozen python scripts/compute_norm_stats.py "
        f"--config-name {args.train_config} --repo-id={repo_id}",
        dry=args.dry_run,
    )
    run_remote(
        args,
        f"cd '{args.remote_openpi_dir}' && {env}uv run --frozen python scripts/train.py "
        f"{args.train_config} --exp-name={exp_name} --data.repo-id={repo_id} "
        f"--num-train-steps={args.num_train_steps} --fsdp-devices={args.fsdp_devices} "
        f"--weight-loader.params-path={init_params} --overwrite{wandb_flag}",
        dry=args.dry_run,
    )


def fetch_remote_checkpoint(args, exp_name: str, remote_step: str) -> str:
    """Copy the whole run dir back so serving (always local) is self-contained."""
    remote_run = f"{args.remote_ckpt_base}/{args.train_config}/{exp_name}"
    local_run = Path(args.local_ckpt_base) / args.train_config / exp_name
    if not args.dry_run:
        local_run.mkdir(parents=True, exist_ok=True)
    rsync(f"{args.ec2_host}:{remote_run}/", f"{local_run}/", args=args, dry=args.dry_run)
    return str(local_run / Path(remote_step).name)


# --------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    remote = args.compute == "ec2"
    started = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    serve_ckpt = args.base_checkpoint
    serve_config = args.base_serve_config
    init_params = args.init_weights

    # Resume: the cumulative data dir already holds every earlier round's episodes, so the only
    # state to rebuild is which checkpoint round <start-round> serves and initializes from. (The
    # LIBERO bash loop gets this wrong -- it keeps serving the BASE policy after --start-round.)
    if args.start_round > 0:
        prev_exp = f"{args.exp_prefix}_r{args.start_round - 1}"
        prev_step = (
            fetch_remote_checkpoint(args, prev_exp, resolve_step_dir(args, args.train_config, prev_exp, remote=True, dry=args.dry_run))
            if remote
            else resolve_step_dir(args, args.train_config, prev_exp, remote=False, dry=args.dry_run)
        )
        serve_ckpt, serve_config = prev_step, args.train_config
        init_params = f"{prev_step}/params" if not remote else f"{args.remote_ckpt_base}/{args.train_config}/{prev_exp}/{Path(prev_step).name}/params"
        print(f"Resuming at round {args.start_round}: serve + init from {prev_step}")

    print(f"\nHITL YAM loop | task={args.task!r} prompt={args.prompt!r} config={args.train_config} "
          f"compute={args.compute} | started {started}")

    for r in range(args.start_round, args.rounds):
        repo_id = f"{args.repo_id_prefix}_r{r}"
        exp_name = f"{args.exp_prefix}_r{r}"
        print(f"\n{'=' * 30} HITL ROUND {r} {'=' * 30}")

        # 0) The server owns the GPU and is started by hand, so a crashed or stale server is
        # visible to the operator rather than buried in this process's output.
        prompt(
            f"ROUND {r} — start the policy server in another terminal:\n\n"
            f"  {serve_command(args, serve_config, serve_ckpt)}\n\n"
            "Wait for 'server listening', then continue.",
            dry=args.dry_run,
        )

        # 1) Collect corrections (operator ends the session).
        collect(args, r)

        # 2) Training wants the GPU the server is holding.
        prompt(f"ROUND {r} — stop the policy server (Ctrl+C in its terminal) to free the GPU.",
               dry=args.dry_run)

        # 3) Decode this round's SVO2 into rgb/lowdim episodes. --append numbers the new episodes
        # after the existing ones instead of renumbering, which is what keeps the cumulative dir
        # (and therefore the aggregation) intact across rounds.
        run_local(
            shlex.split(args.raiden_cmd) + [
                "convert", "--data-dir", args.raiden_data_dir, "--task", args.task,
                "--append", "--w-intervention", str(args.w_intervention),
            ],
            cwd=args.raiden_repo, dry=args.dry_run,
        )

        # 4) Aggregate rounds 0..R (+ offline demos) into this round's LeRobot repo.
        export(args, repo_id)

        # 5+6) Norm stats and training.
        if remote:
            train_remote(args, repo_id, exp_name, init_params)
            remote_step = resolve_step_dir(args, args.train_config, exp_name, remote=True, dry=args.dry_run)
            step_dir = fetch_remote_checkpoint(args, exp_name, remote_step)
            init_params = f"{remote_step}/params"  # stays resident on EC2 for the next round
        else:
            train_local(args, repo_id, exp_name, init_params)
            step_dir = resolve_step_dir(args, args.train_config, exp_name, remote=False, dry=args.dry_run)
            init_params = f"{step_dir}/params"

        # 7) Hand the fresh checkpoint forward.
        print(f"\nRound {r} checkpoint: {step_dir}")
        serve_ckpt, serve_config = step_dir, args.train_config

        # 8) Eval is yours to run and score -- print it rather than pretend to measure it.
        eval_infer = " ".join(
            shlex.quote(str(c))
            for c in shlex.split(args.raiden_cmd) + [
                "infer", "--bridge", BRIDGE, "--ckpt-path", f"localhost:{args.serve_port}",
                "--bridge-kwargs", json.dumps({"prompt": args.prompt}),
                "--action-type", "joint", "--action-hz", str(args.action_hz),
                "--resize-images", "", "--no-depth", "--record",
                "--dagger-task", f"{args.task}_eval_r{r}",
                "--dagger-instruction", args.prompt, "--data-dir", args.raiden_data_dir,
            ]
        )
        prompt(
            f"ROUND {r} — evaluate the new checkpoint (autonomous, no --intervene):\n\n"
            f"  1) {serve_command(args, serve_config, serve_ckpt)}\n\n"
            f"  2) (cd {args.raiden_repo} && {eval_infer})\n\n"
            "Score the runs however you normally do, then continue to the next round.",
            dry=args.dry_run,
        )

    print(f"\nHITL loop complete. Final checkpoint: {serve_ckpt}")


if __name__ == "__main__":
    main()
