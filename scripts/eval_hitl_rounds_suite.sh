#!/usr/bin/env bash
# Evaluate the base policy + every round of one HITL experiment on all libero_90 tasks, then put them side
# by side in <OUT>/comparison.csv. Each policy goes through scripts/eval_pi0_libero_suite.sh into its own
# <OUT>/<policy>/, so re-running this skips every chunk already evaluated (resumable, also per policy).
#
# Round checkpoints are read from <CK>/<EXP>_r<R>/<step>; any round missing there is first downloaded from
# its Hub model repo <MODEL_REPO_PREFIX>_r<R> (what scripts/hitl_pi0_loop_seperated.sh uploads: <step>/ with
# params/ + assets/). Private repos need HF_TOKEN in the environment (or a prior `hf auth login`).
#
# Every setting below can be overridden from the environment, e.g.:
#   # HG-DAgger rounds (the defaults)
#   scripts/eval_hitl_rounds_suite.sh
#   # Flow-MILE rounds trained on SageMaker, 20 episodes per task, without re-running the base policy
#   TRAIN_CONFIG=pi05_libero_flow_mile_lora EXP=fm_pi05 EPISODES=20 EVAL_BASE=false \
#     scripts/eval_hitl_rounds_suite.sh
set -uo pipefail
# One { ... } block: bash parses it all before running, so editing this file mid-run is safe.
{
cd "${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root

TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_libero_hitl_lora}"              # openpi train config of the rounds
EXP="${EXP:-libero_hitl_pi05}"                                      # rounds are <EXP>_r0, <EXP>_r1, ...
ROUNDS="${ROUNDS:-0 1 2}"
MODEL_REPO_PREFIX="${MODEL_REPO_PREFIX:-ykorkmaz/${EXP}_model}"     # Hub repos <prefix>_r<R>
CK="${CK:-third_party/dsrl_openpi/checkpoints/$TRAIN_CONFIG}"       # local checkpoint root
EPISODES="${EPISODES:-50}"
OUT="${OUT:-outputs/eval_suite_libero_90_${EXP}}"
EVAL_BASE="${EVAL_BASE:-true}"                                      # false to skip the base policy

echo "Experiment $EXP ($TRAIN_CONFIG), rounds: $ROUNDS | $EPISODES episodes/task -> $OUT"

# newest_step <run dir> -> highest numbered step dir that has both params/ and assets/ (empty if none)
newest_step() {
  local s
  for s in $(ls "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -rn); do
    [[ -d "$1/$s/params" && -d "$1/$s/assets" ]] && { echo "$s"; return; }
  done
}

POLICIES=()
if [[ "$EVAL_BASE" == "true" ]]; then
  scripts/eval_pi0_libero_suite.sh --num-episodes "$EPISODES" --out-dir "$OUT/base" || echo "!! base: some chunks failed" >&2
  POLICIES+=(base)
fi
for r in $ROUNDS; do
  run="$CK/${EXP}_r$r"
  step="$(newest_step "$run")"
  if [[ -z "$step" ]]; then
    repo="${MODEL_REPO_PREFIX}_r$r"
    echo -e "\n== round $r: not under $run, downloading $repo"
    uv run hf download "$repo" --repo-type model --local-dir "$run" \
      || { echo "!! round $r: download of $repo failed (HF_TOKEN set? repo name right?), skipping" >&2; continue; }
    step="$(newest_step "$run")"
    [[ -n "$step" ]] || { echo "!! round $r: $repo has no <step>/params + assets, skipping" >&2; continue; }
  fi
  echo -e "\n== round $r: $run/$step"
  scripts/eval_pi0_libero_suite.sh --num-episodes "$EPISODES" --config-name "$TRAIN_CONFIG" \
    --checkpoint "$run/$step" --out-dir "$OUT/r$r" || echo "!! round $r: some chunks failed" >&2
  POLICIES+=("r$r")
done

# ---- One row per task, success rate + avg steps on successes for every policy ----
python3 - "$OUT" "${POLICIES[@]}" <<'EOF'
import csv, os, sys
out, policies = sys.argv[1], sys.argv[2:]
tables = {}
for p in policies:
    path = os.path.join(out, p, "results.csv")
    if os.path.exists(path):
        tables[p] = {int(r["task_id"]): r for r in csv.DictReader(open(path))}
    else:
        print(f"(no results.csv for {p} yet)")
policies = [p for p in policies if p in tables]
if not policies:
    sys.exit("nothing to compare")
task_ids = sorted(set().union(*(t.keys() for t in tables.values())))
cols = ["task_id", "task_name"] + [f"{p}_{m}" for p in policies for m in ("success_rate", "avg_steps_success")]
path = os.path.join(out, "comparison.csv")
with open(path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for tid in task_ids:
        name = next((tables[p][tid]["task_name"] for p in policies if tid in tables[p]), "")
        row = [tid, name]
        for p in policies:
            r = tables[p].get(tid)
            row += [r["success_rate"], r["avg_steps_success"]] if r else ["", ""]
        w.writerow(row)
print(f"\n{len(task_ids)} task(s) x {len(policies)} policies -> {path}")
for p in policies:
    srs = [float(r["success_rate"]) for r in tables[p].values()]
    print(f"  {p:>5}: mean success {sum(srs) / len(srs):.1%} over {len(srs)} task(s)")
EOF
exit 0
}
