#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=0
exec "${PYTHON:-python3}" -u "$ROOT/adv/adv_famo_2.py" cifar10 0.1 --five-seeds --optimizer famo --loc-json "$ROOT/rq1/NoContrast/localization/cifar10/CGARepair_adv_samples512_seed256_eps0.1_v3.json" --loc-seed 256 --loc-samples 512 --run-tag paper_gamma001_steps5000 --output-root "$ROOT/rq1/NoContrast/famo_results" --threshold-acc 0.75 --threshold-adv-error 0.2 --max-steps 5000 --model-lr 0.001 --task-lr 0.025 --task-decay 0.001 --grad-batch-size 64 --eval-every 20 --search-radius 10.0 --max-grad-norm 1.0 --loss-eps 1e-08 --archive-size 60 --batch-size 256
