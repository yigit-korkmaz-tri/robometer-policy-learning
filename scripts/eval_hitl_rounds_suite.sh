#!/usr/bin/env bash
# Evaluate the base policy + every pi05_libero_hitl_lora round on all libero_90 tasks, then put them side
# by side in <OUT>/comparison.csv. Each policy goes through scripts/eval_pi0_libero_suite.sh into its own
# <OUT>/<policy>/, so re-running this skips every chunk already evaluated (resumable, also per policy).
#
# Usage (edit CK / EXP / ROUNDS / EPISODES below first; REPO overrides the repo root):
#   scripts/eval_hitl_rounds_suite.sh
set -uo pipefail
{
cd "${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root
CK=third_party/dsrl_openpi/checkpoints/pi05_libero_hitl_lora      # where the round checkpoints live
EXP=libero_hitl_pi05                                               # rounds are <EXP>_r0, <EXP>_r1, ...
ROUNDS="0 1 2"
EPISODES=50
OUT=outputs/eval_suite_libero_90_${EXP}
EVAL_BASE=true                                                     # false to skip the base policy

POLICIES=()
if $EVAL_BASE; then
  scripts/eval_pi0_libero_suite.sh --num-episodes "$EPISODES" --out-dir "$OUT/base" || echo "!! base: some chunks failed" >&2
  POLICIES+=(base)
fi
for r in $ROUNDS; do
  run="$CK/${EXP}_r$r"
  step=$(ls "$run" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  [[ -n "$step" && -d "$run/$step/params" ]] || { echo "!! round $r: no checkpoint under $run, skipping" >&2; continue; }
  scripts/eval_pi0_libero_suite.sh --num-episodes "$EPISODES" --config-name pi05_libero_hitl_lora \
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
