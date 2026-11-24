#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
MODE="${1:-train}"  # train, test, or both
if [[ "$MODE" != 'train' && "$MODE" != 'test' && "$MODE" != 'both' ]]; then
  echo "Usage: bash run_5seeds.sh train|test|both" >&2
  exit 2
fi
mkdir -p "$HERE/logs"
failed=0
for ds in mnist cifar10; do
  case "$ds" in
    mnist) eps="0.05" ;;
    cifar10) eps="0.1" ;;
    *) echo "Unknown dataset $ds" >&2; exit 2 ;;
  esac
  for seed in 42 123 256 1024 2026; do
    echo "==== $MODE dataset=$ds epsilon=$eps seed=$seed ===="
    if [[ "$MODE" == 'train' || "$MODE" == 'both' ]]; then
      python -u "$HERE/deepprov_train.py" "$ds" "$seed" --epsilon "$eps" --root "$PROJECT" \
        2>&1 | tee "$HERE/logs/train_${ds}_eps${eps}_${seed}.log"
      if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
        echo "[FAILED TRAIN] $ds $seed"
        failed=$((failed+1))
        continue
      fi
    fi
    if [[ "$MODE" == 'test' || "$MODE" == 'both' ]]; then
      python -u "$HERE/deepprov_test.py" "$ds" "$seed" --epsilon "$eps" --root "$PROJECT" \
        2>&1 | tee "$HERE/logs/test_${ds}_eps${eps}_${seed}.log"
      if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
        echo "[FAILED TEST] $ds $seed"
        failed=$((failed+1))
      fi
    fi
  done
done
python "$HERE/summarize_deepprov_port.py" --root "$PROJECT"
echo "[DONE] fail_count=$failed"
((failed==0))
