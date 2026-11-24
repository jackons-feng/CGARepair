#!/usr/bin/env bash
# Original CARE safety_moo_1.py + matching safety_care_test.py, five seeds.
# Usage: bash run_care_original_5seeds.sh repair|test|all
# Default models: a19 a29 a33; seeds: 256 42 123 1024 2026.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
MODE="${1:-all}"
case "$MODE" in
  repair|test|all) ;;
  *) echo "Usage: bash $0 repair|test|all" >&2; exit 2 ;;
esac
read -r -a MODELS <<< "${CARE_MODELS:-a19 a29 a33}"
read -r -a SEEDS <<< "${CARE_SEEDS:-256 42 123 1024 2026}"
LOG_ROOT="${CARE_LOG_ROOT:-logs_CARE_original_5seeds}"
mkdir -p "$LOG_ROOT"

[[ -f safety_care_test.py ]] || { echo "Missing safety_care_test.py" >&2; exit 2; }
if [[ "$MODE" != test ]]; then
  [[ -f safety_moo_1.py ]] || { echo "Missing original safety_moo_1.py" >&2; exit 2; }
fi

# Optional overrides for the *TEST loader*, not the original repair:
# CARE_TEST_SAMPLES=0 (full TEST), CARE_MODEL_ROOT, CARE_DATA_ROOT,
# CARE_EVALUATE_FRONT=1 (diagnostic only).
TEST_N="${CARE_TEST_SAMPLES:-0}"
FRONT="${CARE_EVALUATE_FRONT:-0}"
test_opts=(--test-samples "$TEST_N")
[[ -z "${CARE_MODEL_ROOT:-}" ]] || test_opts+=(--model-root "$CARE_MODEL_ROOT")
[[ -z "${CARE_DATA_ROOT:-}" ]] || test_opts+=(--data-root "$CARE_DATA_ROOT")
[[ "$FRONT" != 1 ]] || test_opts+=(--evaluate-front)

# Original repair completion: DO NOT require a fabricated train-provenance file.
repair_done() {
  local dir="$1"
  [[ -s "$dir/weights_best.h5" && -s "$dir/best_solution.npy" && -s "$dir/repair_result.txt" ]] || return 1
  python - "$dir/repair_result.txt" <<'PY_CHECK_REPAIR'
import csv,sys
try:
    with open(sys.argv[1], newline='') as f:
        x={r['metric']:r['value'] for r in csv.DictReader(f)}
    assert all(k in x for k in ('fitness','safety_rate','accuracy','elapsed_seconds','evaluations'))
    assert float(x['elapsed_seconds']) >= 0
except (OSError,KeyError,ValueError,TypeError,AssertionError):
    sys.exit(1)
PY_CHECK_REPAIR
}

test_done() {
  local dir="$1" model="$2" seed="$3"
  [[ -s "$dir/care_test_results.json" && -s "$dir/care_test_metrics.csv" && -s "$dir/care_time_summary.json" ]] || return 1
  python - "$dir/care_test_results.json" "$dir/weights_best.h5" "$model" "$seed" "$TEST_N" "$FRONT" <<'PY_CHECK_TEST'
import json,sys,hashlib
try:
    p,weights,model,seed,n,front=sys.argv[1:]
    with open(p,encoding='utf-8') as f: j=json.load(f)
    assert j['method']=='CARE_PSO'
    assert j['model']==model and int(j['seed'])==int(seed)
    assert int(j.get('test_samples_requested',-1))==int(n)
    assert bool(j.get('requested_evaluate_front',False))==(front=='1')
    assert bool(j.get('full_test_split_evaluated'))==(int(n)==0)
    assert 'original_test_metrics' in j and 'repaired_test_metrics' in j
    h=hashlib.sha256()
    with open(weights,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    assert j.get('checkpoint_sha256')==h.hexdigest()
except (OSError,ValueError,KeyError,AssertionError,TypeError):
    sys.exit(1)
PY_CHECK_TEST
}

failed=0
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    dir="loc_result/$model/$seed/pso"
    mkdir -p "$LOG_ROOT/$model"
    echo "==== [CARE] $model seed=$seed mode=$MODE ===="
    if [[ "$MODE" == repair || "$MODE" == all ]]; then
      if repair_done "$dir"; then
        echo "[SKIP REPAIR] Original CARE already complete"
      else
        # The original repair requires seed-specific CARE localization output.
        base="loc_result/$model/$seed"
        if [[ ! -s "$base/care_players.txt" || ! -s "$base/care_safety_vals.txt" ]]; then
          echo "[MISSING CARE LOCALIZATION] $base/{care_players.txt,care_safety_vals.txt}" >&2
          ((failed+=1)); continue
        fi
        echo "[REPAIR] safety_moo_1.py $model $seed"
        if ! python -u safety_moo_1.py "$model" "$seed" 2>&1 | tee "$LOG_ROOT/$model/repair_${seed}.log"; then
          echo "[FAILED REPAIR] $model seed=$seed" >&2
          ((failed+=1)); continue
        fi
        if ! repair_done "$dir"; then
          echo "[INCOMPLETE REPAIR] $model seed=$seed" >&2
          ((failed+=1)); continue
        fi
      fi
    fi
    if [[ "$MODE" == test || "$MODE" == all ]]; then
      if ! repair_done "$dir"; then
        echo "[NO CARE CHECKPOINT] $model seed=$seed (run repair first)" >&2
        ((failed+=1)); continue
      fi
      if test_done "$dir" "$model" "$seed"; then
        echo "[SKIP TEST] Matching full TEST result already exists"
      else
        echo "[TEST] safety_care_test.py $model $seed"
        if ! python -u safety_care_test.py "$model" "$seed" "${test_opts[@]}" --overwrite 2>&1 | tee "$LOG_ROOT/$model/test_${seed}.log"; then
          echo "[FAILED TEST] $model seed=$seed" >&2
          ((failed+=1)); continue
        fi
        if ! test_done "$dir" "$model" "$seed"; then
          echo "[INCOMPLETE TEST] $model seed=$seed" >&2
          ((failed+=1))
        fi
      fi
    fi
  done
done
echo "[DONE] CARE original, mode=$MODE, failed=$failed, logs=$LOG_ROOT"
[[ "$failed" == 0 ]]
