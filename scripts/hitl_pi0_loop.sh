#!/usr/bin/env bash
# ============================================================================================
# hitl_pi0_loop.sh — split-machine iterative pi0.5 HITL on LIBERO.
#
# Bash counterpart of scripts/hitl_pi0_loop.py, but the fine-tuning runs on a remote EC2 GPU box:
#
#   LOCAL  (this machine, needs a display for teleop):
#     0. eval     scripts/eval_pi0_libero.py           success rate on the task suite (skip --no-eval)
#     1. collect  scripts/collect_hitl_libero_pi0.py   per-task corrections HDF5 (ESC to finish early)
#     2. export   scripts/export_hitl_to_lerobot.py    LeRobot repo (LOCAL)  --rsync-->  EC2
#   EC2    (remote GPU, over SSH; dataset rsync'd from LOCAL, not the Hub):
#     3. normstats third_party/dsrl_openpi/scripts/compute_norm_stats.py
#     4. train     third_party/dsrl_openpi/scripts/train.py <config>   (LoRA HG-DAgger / Flow-MILE)
#   BACK:  rsync the fine-tuned checkpoint dir EC2 -> LOCAL; round R+1 evals/collects with it, and its
#          EC2 copy stays put to init round R+1's remote training.
#
# Assumes the SAME repo + uv envs on both machines. The exported LeRobot dataset is rsync'd to the same
# path on EC2 (under HF_LEROBOT_HOME); no Hugging Face Hub is used. If EC2's home/username differs from
# LOCAL, pass --remote-lerobot-home to point at EC2's LeRobot cache.
#
# Example:
#   scripts/hitl_pi0_loop.sh --rounds 3 --env-name libero_90 --task-ids "57 58 59" \
#     --repo-id-prefix you/libero_hitl --exp-prefix hitl_t575859 --train-config pi05_libero_flow_mile_lora \
#     --libero-base-suite libero_90 --libero-base-num-demos 10 \
#     --ec2-host ubuntu@ec2-1-2-3-4.compute.amazonaws.com --ec2-repo /home/ubuntu/robometer-policy-learning \
#     --ssh-key ~/.ssh/ec2.pem --remote-lerobot-home /home/ubuntu/.cache/huggingface/lerobot
#
# Use --dry-run to print every local/remote command (and both rsyncs) without executing.
# ============================================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- Defaults (mirror hitl_pi0_loop.py) ----
ROUNDS="2"
TASK_IDS="57 58"                       # space-separated, e.g. "57 58 59"
ENV_NAME="libero_90"
COLLECT_NUM_ROLLOUTS=2
NUM_TRAIN_STEPS=10
EVAL_CONFIG="libero_eval"
EVAL_NUM_EPISODES=2
NO_EVAL=false
REPO_ID_PREFIX="ykorkmaz/libero_ec2_test"                 # HF repo id prefix (namespace/name); round R -> <prefix>_r{R}
EXP_PREFIX="libero_ec2_test"
TRAIN_CONFIG="pi05_libero_hitl_lora"
COLLECT_CONFIG="libero_collect_hitl"
BASE_PI0_CHECKPOINT="gs://openpi-assets/checkpoints/pi05_libero/"
BASE_PI0_CONFIG_NAME=""
INIT_WEIGHTS="gs://openpi-assets/checkpoints/pi05_libero/params"
BASE_DEMOS=""                     # space-separated glob(s)
LIBERO_BASE_SUITE="libero_90"
LIBERO_BASE_TASK_IDS=""           # space-separated; default = TASK_IDS
LIBERO_BASE_NUM_DEMOS=2
WORKDIR="$REPO_ROOT/outputs"
OPENPI_DIR="$REPO_ROOT/third_party/dsrl_openpi"
LOCAL_CKPT_BASE=""                # default <OPENPI_DIR>/checkpoints
LEROBOT_HOME=""                   # local LeRobot cache root; default ${HF_LEROBOT_HOME:-~/.cache/huggingface/lerobot}
REMOTE_LEROBOT_HOME="/opt/dlami/nvme/cache/huggingface/lerobot"            # EC2 LeRobot cache root; default = same path as local (--remote-lerobot-home to override)
REMOTE_OPENPI_DATA_HOME="/opt/dlami/nvme/cache/openpi"
REMOTE_UV_CACHE_DIR="/opt/dlami/nvme/cache/uv"
WEIGHT_LOADER_FLAG="--weight-loader.params-path"
XLA_MEM=0.9
START_ROUND=0
DRY=false

# ---- Remote / SSH ----
EC2_HOST="ubuntu@10.161.51.28"                       # user@host  (required unless --dry-run)
EC2_REPO="/opt/dlami/nvme/robometer-policy-learning"                       # repo path on EC2 (required unless --dry-run)
REMOTE_OPENPI_DIR="$EC2_REPO/third_party/dsrl_openpi"              # default <EC2_REPO>/third_party/dsrl_openpi
REMOTE_CKPT_BASE=""               # default <REMOTE_OPENPI_DIR>/checkpoints
SSH_KEY="$HOME/.ssh/yigit.pem"
SSH_OPTS=""
HF_TOKEN=""
WANDB_API_KEY="${WANDB_API_KEY:-}"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

# ---- Arg parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds) ROUNDS="$2"; shift 2;;
    --task-ids) TASK_IDS="$2"; shift 2;;
    --env-name) ENV_NAME="$2"; shift 2;;
    --collect-num-rollouts) COLLECT_NUM_ROLLOUTS="$2"; shift 2;;
    --num-train-steps) NUM_TRAIN_STEPS="$2"; shift 2;;
    --eval-config) EVAL_CONFIG="$2"; shift 2;;
    --eval-num-episodes) EVAL_NUM_EPISODES="$2"; shift 2;;
    --no-eval) NO_EVAL=true; shift;;
    --repo-id-prefix) REPO_ID_PREFIX="$2"; shift 2;;
    --exp-prefix) EXP_PREFIX="$2"; shift 2;;
    --train-config) TRAIN_CONFIG="$2"; shift 2;;
    --collect-config) COLLECT_CONFIG="$2"; shift 2;;
    --base-pi0-checkpoint) BASE_PI0_CHECKPOINT="$2"; shift 2;;
    --base-pi0-config-name) BASE_PI0_CONFIG_NAME="$2"; shift 2;;
    --init-weights) INIT_WEIGHTS="$2"; shift 2;;
    --base-demos) BASE_DEMOS="$2"; shift 2;;
    --libero-base-suite) LIBERO_BASE_SUITE="$2"; shift 2;;
    --libero-base-task-ids) LIBERO_BASE_TASK_IDS="$2"; shift 2;;
    --libero-base-num-demos) LIBERO_BASE_NUM_DEMOS="$2"; shift 2;;
    --workdir) WORKDIR="$2"; shift 2;;
    --openpi-dir) OPENPI_DIR="$2"; shift 2;;
    --local-ckpt-base) LOCAL_CKPT_BASE="$2"; shift 2;;
    --lerobot-home) LEROBOT_HOME="$2"; shift 2;;
    --remote-lerobot-home) REMOTE_LEROBOT_HOME="$2"; shift 2;;
    --remote-openpi-data-home) REMOTE_OPENPI_DATA_HOME="$2"; shift 2;;
    --remote-uv-cache-dir) REMOTE_UV_CACHE_DIR="$2"; shift 2;;
    --weight-loader-flag) WEIGHT_LOADER_FLAG="$2"; shift 2;;
    --xla-mem-fraction) XLA_MEM="$2"; shift 2;;
    --start-round) START_ROUND="$2"; shift 2;;
    --dry-run) DRY=true; shift;;
    --ec2-host) EC2_HOST="$2"; shift 2;;
    --ec2-repo) EC2_REPO="$2"; shift 2;;
    --remote-openpi-dir) REMOTE_OPENPI_DIR="$2"; shift 2;;
    --remote-ckpt-base) REMOTE_CKPT_BASE="$2"; shift 2;;
    --ssh-key) SSH_KEY="$2"; shift 2;;
    --ssh-opts) SSH_OPTS="$2"; shift 2;;
    --hf-token) HF_TOKEN="$2"; shift 2;;
    --wandb-api-key) WANDB_API_KEY="$2"; shift 2;;
    -h|--help) usage 0;;
    *) echo "Unknown arg: $1" >&2; usage 1;;
  esac
done

# ---- Validate + derive ----
[[ -z "$ROUNDS" || -z "$TASK_IDS" || -z "$REPO_ID_PREFIX" || -z "$EXP_PREFIX" ]] && \
  { echo "ERROR: --rounds, --task-ids, --repo-id-prefix, --exp-prefix are required." >&2; usage 1; }
if ! $DRY; then
  [[ -z "$EC2_HOST" || -z "$EC2_REPO" ]] && \
    { echo "ERROR: --ec2-host and --ec2-repo are required (unless --dry-run)." >&2; usage 1; }
fi
: "${LOCAL_CKPT_BASE:=$OPENPI_DIR/checkpoints}"
: "${REMOTE_OPENPI_DIR:=$EC2_REPO/third_party/dsrl_openpi}"
: "${REMOTE_CKPT_BASE:=$REMOTE_OPENPI_DIR/checkpoints}"
# LeRobot dataset cache roots. The exporter writes to $LEROBOT_HOME/<repo_id>; we rsync that dir to the
# same path on EC2 and point the remote steps at it. Default the remote to the same path as local.
: "${LEROBOT_HOME:=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}}"
: "${REMOTE_LEROBOT_HOME:=$LEROBOT_HOME}"
TASK_IDS_CSV="$(echo "$TASK_IDS" | tr -s ' ' ',')"
[[ -z "$LIBERO_BASE_TASK_IDS" ]] && LIBERO_BASE_TASK_IDS="$TASK_IDS"
# HG-DAgger (pi05_libero_hitl_lora) trains on corrections only; Flow-MILE needs full trajectories.
if [[ "$TRAIN_CONFIG" == "pi05_libero_hitl_lora" ]]; then STORE_ONLY_HUMAN=true; else STORE_ONLY_HUMAN=false; fi
# Append the train config + a run timestamp so each invocation gets its own outputs dir.
WORKDIR="${WORKDIR}/${TRAIN_CONFIG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WORKDIR"

SSH=(ssh)
[[ -n "$SSH_KEY" ]] && SSH+=(-i "$SSH_KEY")
# shellcheck disable=SC2206  # intentional word-split of extra opts
[[ -n "$SSH_OPTS" ]] && SSH+=($SSH_OPTS)
[[ -n "$EC2_HOST" ]] && SSH+=("$EC2_HOST")
RSH="ssh"; [[ -n "$SSH_KEY" ]] && RSH="ssh -i $SSH_KEY"
# HF token exported to remote (and, if given, to local) so private-dataset download/push works.
REMOTE_HF_ENV=""; [[ -n "$HF_TOKEN" ]] && REMOTE_HF_ENV="HF_TOKEN=$HF_TOKEN HUGGING_FACE_HUB_TOKEN=$HF_TOKEN "
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
REMOTE_WANDB_ENV=""; [[ -n "$WANDB_API_KEY" ]] && REMOTE_WANDB_ENV="WANDB_API_KEY=$WANDB_API_KEY "
TRAIN_WANDB_FLAG="--wandb-enabled=false"; [[ -n "$WANDB_API_KEY" ]] && TRAIN_WANDB_FLAG=""

# ---- Runners ----
run_local() {  # run_local <cwd> <cmd...>
  local cwd="$1"; shift
  printf '\n$ (LOCAL %s) %s\n\n' "$cwd" "$*"
  $DRY && return 0
  ( cd "$cwd" && "$@" ) || { echo "LOCAL step failed: $*" >&2; exit 1; }
}
run_remote() {  # run_remote <remote-shell-command-string>
  local cmd="export PATH=\"\$HOME/.local/bin:\$PATH\"; $1"
  local shown="$cmd"
  [[ -n "$HF_TOKEN" ]] && shown="${shown//$HF_TOKEN/<HF_TOKEN>}"  # never echo the token
  [[ -n "$WANDB_API_KEY" ]] && shown="${shown//$WANDB_API_KEY/<WANDB_API_KEY>}"
  printf '\n$ [EC2 %s] %s\n\n' "${EC2_HOST:-<host>}" "$shown"
  $DRY && return 0
  "${SSH[@]}" "$cmd" || { echo "REMOTE step failed" >&2; exit 1; }
}
capture_remote() { "${SSH[@]}" "$1"; }  # stdout only, for resolving the step dir

# ============================================================================================
COLLECT_CKPT="$BASE_PI0_CHECKPOINT"      # LOCAL eval/collect policy
COLLECT_CFG_NAME="$BASE_PI0_CONFIG_NAME" # LOCAL config_name (empty => path heuristic)
REMOTE_INIT="$INIT_WEIGHTS"              # EC2 training init params
ROUND_HDF5S=()                           # accumulate every round's corrections

for (( r=START_ROUND; r<ROUNDS; r++ )); do
  REPO_ID="${REPO_ID_PREFIX}_r${r}"
  EXP_NAME="${EXP_PREFIX}_r${r}"
  echo -e "\n========== HITL ROUND ${r} =========="

  # 0) [LOCAL] eval the current policy on the whole suite.
  if ! $NO_EVAL; then
    EVAL_DIR="$WORKDIR/eval_r${r}"
    EVAL_CMD=(env XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM" uv run python scripts/eval_pi0_libero.py
      --config-name "$EVAL_CONFIG" "env.env_name=$ENV_NAME" "env.task_ids=[$TASK_IDS_CSV]"
      "eval.num_episodes=$EVAL_NUM_EPISODES" "pi0.checkpoint=$COLLECT_CKPT" "hydra.run.dir=$EVAL_DIR")
    [[ -n "$COLLECT_CFG_NAME" ]] && EVAL_CMD+=("pi0.config_name=$COLLECT_CFG_NAME")
    run_local "$REPO_ROOT" "${EVAL_CMD[@]}"
  fi

  # 1) [LOCAL] collect corrections for every task (ESC in the teleop window finishes early).
  for t in $TASK_IDS; do
    ROUND_HDF5="$WORKDIR/round_${r}_task_${t}_rollouts.hdf5"
    ROUND_HDF5S+=("$ROUND_HDF5")
    COLLECT_CMD=(env XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM" uv run python scripts/collect_hitl_libero_pi0.py
      --config-name "$COLLECT_CONFIG" "env.env_name=$ENV_NAME" "env.task_id=$t"
      "hitl.collect_num_rollouts=$COLLECT_NUM_ROLLOUTS" "hitl.collect_output_path=$ROUND_HDF5"
      "pi0.checkpoint=$COLLECT_CKPT" "hitl.store_only_human=$STORE_ONLY_HUMAN")
    [[ -n "$COLLECT_CFG_NAME" ]] && COLLECT_CMD+=("pi0.config_name=$COLLECT_CFG_NAME")
    run_local "$REPO_ROOT" "${COLLECT_CMD[@]}"
  done

  # Only feed HDF5s that actually exist (a collect may write nothing, e.g. early-finish w/o corrections).
  EXPORT_INPUTS=()
  if $DRY; then EXPORT_INPUTS=("${ROUND_HDF5S[@]}")
  else for p in "${ROUND_HDF5S[@]}"; do [[ -f "$p" ]] && EXPORT_INPUTS+=("$p"); done; fi
  HAS_BASE=false
  { [[ -n "$BASE_DEMOS" ]] || [[ -n "$LIBERO_BASE_SUITE" ]]; } && HAS_BASE=true
  if [[ ${#EXPORT_INPUTS[@]} -eq 0 ]] && ! $HAS_BASE; then
    echo "Round ${r}: no saved corrections and no base demos — skipping export/train."
    continue
  fi

  # 2) [LOCAL] export -> LeRobot repo under $LEROBOT_HOME/<repo_id> (openpi env). No Hub.
  EXPORT_CMD=(env "HF_LEROBOT_HOME=$LEROBOT_HOME" uv run --frozen python
    "$REPO_ROOT/scripts/export_hitl_to_lerobot.py" --repo-id "$REPO_ID")
  [[ ${#EXPORT_INPUTS[@]} -gt 0 ]] && EXPORT_CMD+=(--inputs "${EXPORT_INPUTS[@]}")
  [[ -n "$BASE_DEMOS" ]] && EXPORT_CMD+=(--base-demos $BASE_DEMOS)
  if [[ -n "$LIBERO_BASE_SUITE" ]]; then
    EXPORT_CMD+=(--libero-base-suite "$LIBERO_BASE_SUITE" --libero-base-task-ids $LIBERO_BASE_TASK_IDS
      --libero-base-num-demos "$LIBERO_BASE_NUM_DEMOS")
  fi
  run_local "$OPENPI_DIR" "${EXPORT_CMD[@]}"

  # 2b) rsync the LOCAL LeRobot dataset dir to the same repo_id path on EC2 (so the remote steps read
  # it locally, no Hub). --delete keeps the (round-specific) remote copy an exact mirror.
  LOCAL_DS="$LEROBOT_HOME/$REPO_ID"
  REMOTE_DS="$REMOTE_LEROBOT_HOME/$REPO_ID"
  printf '\n$ rsync %s/ -> %s:%s/\n\n' "$LOCAL_DS" "${EC2_HOST:-<host>}" "$REMOTE_DS"
  if ! $DRY; then
    capture_remote "mkdir -p '$REMOTE_DS'" || { echo "remote mkdir failed" >&2; exit 1; }
    rsync -az --delete -e "$RSH" "$LOCAL_DS/" "$EC2_HOST:$REMOTE_DS/" || { echo "dataset rsync failed" >&2; exit 1; }
  fi

  # 3) [EC2] compute norm stats (reads the rsync'd dataset via HF_LEROBOT_HOME).
  run_remote "cd '$REMOTE_OPENPI_DIR' && ${REMOTE_HF_ENV}HF_LEROBOT_HOME='$REMOTE_LEROBOT_HOME' OPENPI_DATA_HOME='$REMOTE_OPENPI_DATA_HOME' UV_CACHE_DIR='$REMOTE_UV_CACHE_DIR' \
XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM \
uv run --frozen python scripts/compute_norm_stats.py --config-name $TRAIN_CONFIG --repo-id=$REPO_ID"

  # 4) [EC2] LoRA fine-tune, initialized from REMOTE_INIT (base for r0, else prev round's EC2 ckpt).
  run_remote "cd '$REMOTE_OPENPI_DIR' && ${REMOTE_HF_ENV}${REMOTE_WANDB_ENV}HF_LEROBOT_HOME='$REMOTE_LEROBOT_HOME' OPENPI_DATA_HOME='$REMOTE_OPENPI_DATA_HOME' UV_CACHE_DIR='$REMOTE_UV_CACHE_DIR' \
XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM \
uv run --frozen python scripts/train.py $TRAIN_CONFIG --exp-name=$EXP_NAME --data.repo-id=$REPO_ID \
--num-train-steps=$NUM_TRAIN_STEPS $WEIGHT_LOADER_FLAG=$REMOTE_INIT $TRAIN_WANDB_FLAG --overwrite"

  # 5) Resolve the fresh EC2 checkpoint, rsync the run dir back to LOCAL, hand it forward.
  REMOTE_RUN_DIR="$REMOTE_CKPT_BASE/$TRAIN_CONFIG/$EXP_NAME"
  LOCAL_RUN_DIR="$LOCAL_CKPT_BASE/$TRAIN_CONFIG/$EXP_NAME"
  if $DRY; then
    STEP="<step>"
  else
    STEP="$(capture_remote "ls -1 '$REMOTE_RUN_DIR' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1")"
    [[ -z "$STEP" ]] && { echo "ERROR: no numeric step checkpoint under EC2:$REMOTE_RUN_DIR (did training save?)" >&2; exit 1; }
  fi
  # Copy the whole run dir back (step params + norm-stat assets) so LOCAL eval/collect is self-contained.
  printf '\n$ rsync %s:%s/ -> %s/\n\n' "${EC2_HOST:-<host>}" "$REMOTE_RUN_DIR" "$LOCAL_RUN_DIR"
  if ! $DRY; then
    mkdir -p "$LOCAL_RUN_DIR"
    rsync -az -e "$RSH" "$EC2_HOST:$REMOTE_RUN_DIR/" "$LOCAL_RUN_DIR/" || { echo "rsync back failed" >&2; exit 1; }
  fi

  LOCAL_STEP_DIR="$LOCAL_RUN_DIR/$STEP"
  REMOTE_STEP_DIR="$REMOTE_RUN_DIR/$STEP"
  echo "Round ${r} checkpoint: EC2:$REMOTE_STEP_DIR  ->  LOCAL:$LOCAL_STEP_DIR"
  COLLECT_CKPT="$LOCAL_STEP_DIR"       # LOCAL eval/collect next round
  COLLECT_CFG_NAME="$TRAIN_CONFIG"     # fine-tuned LoRA arch => explicit config for Pi0Wrapper
  REMOTE_INIT="$REMOTE_STEP_DIR/params"  # EC2 training init next round (stays resident on EC2)
done

# Final eval of the last policy (LOCAL), unless --no-eval.
if ! $NO_EVAL; then
  EVAL_DIR="$WORKDIR/eval_r${ROUNDS}"
  EVAL_CMD=(env XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM" uv run python scripts/eval_pi0_libero.py
    --config-name "$EVAL_CONFIG" "env.env_name=$ENV_NAME" "env.task_ids=[$TASK_IDS_CSV]"
    "eval.num_episodes=$EVAL_NUM_EPISODES" "pi0.checkpoint=$COLLECT_CKPT" "hydra.run.dir=$EVAL_DIR")
  [[ -n "$COLLECT_CFG_NAME" ]] && EVAL_CMD+=("pi0.config_name=$COLLECT_CFG_NAME")
  run_local "$REPO_ROOT" "${EVAL_CMD[@]}"
fi

echo -e "\nHITL loop complete."
