#!/usr/bin/env bash
# ============================================================================================
# eval_pi0_libero_suite.sh — evaluate one pi0 / pi0.5 checkpoint on a whole LIBERO suite and write a CSV.
#
# Runs scripts/eval_pi0_libero.py over the suite's task ids in chunks (one process per chunk, each with its
# own <out-dir>/chunk_<first>-<last>/eval_results.json), then merges every chunk into <out-dir>/results.csv:
#
#   task_id, task_name, language_instruction, num_episodes, num_success, success_rate,
#   avg_steps_success, avg_steps, max_episode_steps
#
# avg_steps_success is the mean episode length over SUCCESSFUL episodes only; it is left empty for a task
# with no successes (eval_pi0_libero.py reports 0.0 there, which would read as "instant success").
#
# Chunking keeps a long run resumable: eval_pi0_libero.py writes its results only when a process finishes,
# so a crash 80 tasks into a single 90-task process would lose everything. Re-running with the same
# --out-dir skips every chunk that already has eval_results.json; a failed chunk is reported and the
# remaining chunks still run (the CSV then covers the chunks that finished, and the script exits 1).
#
# Usage:
#   # base pi0.5-LIBERO on all 90 libero_90 tasks, 50 episodes each
#   scripts/eval_pi0_libero_suite.sh
#
#   # a fine-tuned LoRA checkpoint (needs its train config), 20 episodes, resuming into an existing dir
#   scripts/eval_pi0_libero_suite.sh --num-episodes 20 --config-name pi05_libero_hitl_lora \
#     --checkpoint third_party/dsrl_openpi/checkpoints/pi05_libero_hitl_lora/libero_hitl_pi05_r2/4999 \
#     --out-dir outputs/eval_suite_libero_90_hitl_r2
#
#   # a subset, plus extra Hydra overrides for eval_pi0_libero.py after a bare --
#   scripts/eval_pi0_libero_suite.sh --task-ids "19 35" -- env.max_episode_steps=800
#
# Flags: --checkpoint, --config-name (pi0.config_name; empty = inferred from the path, right for base
# checkpoints), --env-name (default libero_90), --task-ids (default: every task in the suite),
# --num-episodes (default 50), --chunk-size (default 10), --out-dir, --xla-mem-fraction (default 0.95),
# --dry-run (print the commands only).
# ============================================================================================
set -uo pipefail
# One { ... } block: bash parses it all before running, so editing this file mid-run is safe.
{
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT="gs://openpi-assets/checkpoints/pi05_libero/"
CONFIG_NAME=""
ENV_NAME="libero_90"
TASK_IDS=""                 # space-separated; empty => every task in the suite
NUM_EPISODES=50
CHUNK_SIZE=10
OUT_DIR=""
XLA_MEM=0.95
DRY=false
EXTRA=()

usage() { sed -n '2,35p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --config-name) CONFIG_NAME="$2"; shift 2;;
    --env-name) ENV_NAME="$2"; shift 2;;
    --task-ids) TASK_IDS="$2"; shift 2;;
    --num-episodes) NUM_EPISODES="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --xla-mem-fraction) XLA_MEM="$2"; shift 2;;
    --dry-run) DRY=true; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) usage 0;;
    *) echo "Unknown arg: $1" >&2; usage 1;;
  esac
done

# Task count per suite: libero_90 has 90 tasks, the other LIBERO suites 10.
if [[ -z "$TASK_IDS" ]]; then
  if [[ "$ENV_NAME" == "libero_90" ]]; then N_TASKS=90; else N_TASKS=10; fi
  TASK_IDS="$(seq -s ' ' 0 $(( N_TASKS - 1 )))"
fi
read -ra TASKS <<< "$(echo "$TASK_IDS" | tr ',' ' ')"
for t in "${TASKS[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] || { echo "ERROR: --task-ids must be integers, got '$t'." >&2; exit 1; }
done
[[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --chunk-size must be a positive integer." >&2; exit 1; }

# Default out dir: named after the checkpoint (its last path components) so different policies don't mix.
if [[ -z "$OUT_DIR" ]]; then
  TAG="$(echo "${CHECKPOINT%/}" | awk -F/ '{print (NF>1 ? $(NF-1)"_"$NF : $NF)}' | tr -c '[:alnum:]_.-' '_')"
  OUT_DIR="$REPO_ROOT/outputs/eval_suite_${ENV_NAME}_${TAG%_}_$(date +%Y%m%d_%H%M%S)"
fi
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$REPO_ROOT/$OUT_DIR"
# A relative local checkpoint path would be resolved from the eval's cwd; pin it to the repo root.
if [[ "$CHECKPOINT" != *://* && "$CHECKPOINT" != /* ]]; then CHECKPOINT="$REPO_ROOT/$CHECKPOINT"; fi
mkdir -p "$OUT_DIR"

echo "Checkpoint: $CHECKPOINT${CONFIG_NAME:+ (config_name=$CONFIG_NAME)}"
echo "Suite: $ENV_NAME, ${#TASKS[@]} task(s), $NUM_EPISODES episodes each, chunks of $CHUNK_SIZE -> $OUT_DIR"

FAILED=()
for (( i=0; i<${#TASKS[@]}; i+=CHUNK_SIZE )); do
  CHUNK=("${TASKS[@]:i:CHUNK_SIZE}")
  CHUNK_DIR="$OUT_DIR/chunk_${CHUNK[0]}-${CHUNK[-1]}"
  if [[ -f "$CHUNK_DIR/eval_results.json" ]]; then
    echo -e "\n== tasks ${CHUNK[*]}: already done ($CHUNK_DIR), skipping"
    continue
  fi
  IDS_CSV="$(IFS=,; echo "${CHUNK[*]}")"
  CMD=(env XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM" uv run python scripts/eval_pi0_libero.py
    --config-name libero_eval "env.env_name=$ENV_NAME" "env.task_ids=[$IDS_CSV]"
    "eval.num_episodes=$NUM_EPISODES" "pi0.checkpoint=$CHECKPOINT" "hydra.run.dir=$CHUNK_DIR")
  [[ -n "$CONFIG_NAME" ]] && CMD+=("pi0.config_name=$CONFIG_NAME")
  CMD+=("${EXTRA[@]}")
  echo -e "\n== tasks ${CHUNK[*]}\n\$ ${CMD[*]}"
  $DRY && continue
  ( cd "$REPO_ROOT" && "${CMD[@]}" ) || { echo "!! chunk ${CHUNK[0]}-${CHUNK[-1]} failed; continuing" >&2; FAILED+=("${CHUNK[0]}-${CHUNK[-1]}"); }
done
$DRY && exit 0

# ---- Merge every finished chunk into one CSV (sorted by task id) ----
python3 - "$OUT_DIR" <<'EOF'
import csv, glob, json, os, sys
out = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out, "chunk_*", "eval_results.json"))):
    for t in json.load(open(path))["per_task"]:
        rows.append({
            "task_id": t["task_id"],
            "task_name": t.get("task_name"),
            "language_instruction": t.get("language_instruction"),
            "num_episodes": t["num_episodes"],
            "num_success": t["num_success"],
            "success_rate": round(t["success_rate"], 4),
            # Mean steps over successful episodes; empty (not 0) when there were none.
            "avg_steps_success": round(t["avg_steps_success"], 1) if t["num_success"] > 0 else "",
            "avg_steps": round(t["avg_steps"], 1),
            "max_episode_steps": t.get("max_episode_steps"),
        })
rows.sort(key=lambda r: r["task_id"])
csv_path = os.path.join(out, "results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["task_id"])
    w.writeheader(); w.writerows(rows)
if rows:
    eps = sum(r["num_episodes"] for r in rows); succ = sum(r["num_success"] for r in rows)
    succ_steps = [(r["avg_steps_success"], r["num_success"]) for r in rows if r["num_success"] > 0]
    pooled = sum(s * n for s, n in succ_steps) / max(sum(n for _, n in succ_steps), 1)
    print(f"\n{len(rows)} task(s) -> {csv_path}")
    print(f"mean success rate (over tasks): {sum(r['success_rate'] for r in rows) / len(rows):.1%} | "
          f"micro: {succ}/{eps} = {succ / max(eps, 1):.1%} | avg steps on successes (pooled): {pooled:.1f}")
else:
    print(f"\nNo finished chunks under {out}; nothing to write.")
EOF

if (( ${#FAILED[@]} )); then
  echo "Failed chunk(s): ${FAILED[*]} -- re-run with --out-dir $OUT_DIR to retry just those." >&2
  exit 1
fi
exit 0
}
