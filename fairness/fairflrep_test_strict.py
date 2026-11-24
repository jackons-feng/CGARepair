#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict TEST-only reporting for the CFRepair-aligned FairFLRep adapter.

This script is intentionally separate from fairflrep_repair_fulltrain.py.
It loads the fixed pretrained model and the saved FairFLRep weights, evaluates
both on the complete TEST split, and writes paper-ready Gap/Accuracy outputs.
It performs no localization, repair, PSO search, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

from tensorflow.compat.v1.keras.models import load_model

from fairflrep_repair_fulltrain import (
    SUPPORTED_DATASETS,
    configure_tensorflow,
    evaluate_model,
    load_fairness_split,
    print_metrics,
    validate_model_data_shape,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict TEST reporting for FairFLRep-CFRepair Gap adapter.")
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--model-root", default="../model")
    parser.add_argument("--data-root", default="../data/fairness")
    parser.add_argument("--output-root", default="./loc_result")
    parser.add_argument("--run-name", default="fairflrep_settdiff_gap_fulltrain")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_tensorflow()

    dataset = args.dataset.lower()
    model_path = Path(args.model_root) / f"{dataset}.h5"
    data_dir = Path(args.data_root) / dataset
    run_dir = Path(args.output_root) / dataset / args.run_name
    weights_path = run_dir / "weights_final.h5"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Fairness data directory not found: {data_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Repaired weights not found: {weights_path}. Run fairflrep_repair_fulltrain.py first."
        )

    x_test, y_test, g_test = load_fairness_split(data_dir, "test")

    model = load_model(str(model_path), compile=False)
    validate_model_data_shape(model, x_test, dataset)
    baseline = evaluate_model(model, x_test, y_test, g_test, "gap", args.batch_size)

    model.load_weights(str(weights_path))
    repaired = evaluate_model(model, x_test, y_test, g_test, "gap", args.batch_size)

    print_metrics("Original COMPLETE TEST metrics", baseline)
    print_metrics("Repaired COMPLETE TEST metrics", repaired)

    row = {
        "dataset": dataset,
        "method": "FairFLRep_SettDiff_PSO_Gap_fulltrain_adapter",
        "fairness_metric": "gap",
        "test_only_reporting": True,
        "test_used_for_localization": False,
        "test_used_for_repair": False,
        "test_used_for_selection": False,
        "before_test_gap": baseline["group_accuracy_gap"],
        "after_test_gap": repaired["group_accuracy_gap"],
        "delta_test_gap": float(repaired["group_accuracy_gap"]) - float(baseline["group_accuracy_gap"]),
        "before_test_accuracy": baseline["accuracy"],
        "after_test_accuracy": repaired["accuracy"],
        "delta_test_accuracy": float(repaired["accuracy"]) - float(baseline["accuracy"]),
        "before_test_f1": baseline["f1"],
        "after_test_f1": repaired["f1"],
        "weights_file": str(weights_path),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "test_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    txt = [
        f"dataset={dataset}",
        "method=fairflrep_settdiff_pso_gap_fulltrain_adapter",
        "fairness_metric=gap",
        "test_only_reporting=True",
        "test_used_for_selection=False",
        f"original_test_gap={float(baseline['group_accuracy_gap']):.10f}",
        f"repaired_test_gap={float(repaired['group_accuracy_gap']):.10f}",
        f"original_test_accuracy={float(baseline['accuracy']):.10f}",
        f"repaired_test_accuracy={float(repaired['accuracy']):.10f}",
        f"weights_file={weights_path}",
    ]
    (run_dir / "best_solution_full_test.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

    print(f"\nSaved strict TEST summary: {csv_path}")
    print(f"Saved paper-ready TEST text: {run_dir / 'best_solution_full_test.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
