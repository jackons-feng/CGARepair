import argparse
import ast
import csv
import importlib.util
import inspect
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rq1"
OUT = ROOT / "rq1/LocalizationTiming"

SOURCES = {
    "gradact": "loc_CGARepair_test.py",
    "ccbr_igd4": "loc_ccbr_test.py",
}

def prepare():
    pattern = re.compile(
        r"os\.path\.join\(\s*(['\"])result\1\s*,"
        r"\s*(['\"])loc_result\2\s*,"
    )

    for method, filename in SOURCES.items():
        original = (SRC / filename).read_text(encoding="utf-8")

        if method == "ccbr_igd4":
            if not re.search(r"(?m)^CALSAL_STEP\s*=\s*4\b", original):
                raise RuntimeError("CCBR integration step count changed")

        replacement = (
            "os.path.join('..', 'rq1', "
            "'LocalizationTiming', 'cache', "
            "os.environ['CGAT_METHOD'], "
            "os.environ['CGAT_DATASET'], "
            "os.environ['CGAT_TRIAL'], "
        )

        modified, count = pattern.subn(
            lambda _: replacement, original
        )

        if count < 1:
            raise RuntimeError(
                "Cannot isolate cache paths in " + filename
            )

        ast.parse(modified)

        target = SRC / ("_timing_only_" + method + ".py")

        if target.exists():
            if target.read_text(encoding="utf-8") != modified:
                raise RuntimeError(
                    "Existing timing source differs: " + str(target)
                )
        else:
            target.write_text(modified, encoding="utf-8")

        print("PREPARED", target, "isolated paths:", count)

def run_once(method, dataset, trial):
    os.environ["CGAT_METHOD"] = method
    os.environ["CGAT_DATASET"] = dataset
    os.environ["CGAT_TRIAL"] = trial

    os.chdir(SRC)
    sys.path.insert(0, str(SRC))

    path = SRC / ("_timing_only_" + method + ".py")
    spec = importlib.util.spec_from_file_location(
        "_cgarepair_benchmark", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.seed = 256
    random.seed(256)
    module.np.random.seed(256)
    module.tf.set_random_seed(256)

    timings = {}
    counts = {}

    if method == "gradact":
        names = [
            "_gradact_build_graph",
            "_gradact_average",
        ]
        function = module.compute_gradient_activation_values
    else:
        names = [
            "ig_diff_scores_for_layer",
            "causal_effect_scores_for_layer",
        ]
        function = module.compute_causal_intervention_values

    for name in names:
        if not hasattr(module, name):
            raise RuntimeError("Missing scorer: " + name)

        original = getattr(module, name)

        def wrapper(*args, _fn=original, _name=name, **kwargs):
            begin = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                duration = time.perf_counter() - begin
                timings[_name] = timings.get(_name, 0) + duration
                counts[_name] = counts.get(_name, 0) + 1

        setattr(module, name, wrapper)

    epsilon = "0.05" if dataset == "mnist" else "0.1"

    kwargs = {
        "task": "adv",
        "model_name": dataset,
        "num_examples": 512,
        "top_k": None,
        "resnet_18_flag": False,
        "epsilon": epsilon,
    }

    signature = inspect.signature(function)
    actual_kwargs = {
        key: value for key, value in kwargs.items()
        if key in signature.parameters
    }

    required = [
        key for key, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind not in (
            param.VAR_POSITIONAL, param.VAR_KEYWORD
        )
        and key not in actual_kwargs
    ]

    if required:
        raise RuntimeError(
            "Unsupported required arguments: " + repr(required)
        )

    begin = time.perf_counter()
    result = function(**actual_kwargs)
    function_sec = time.perf_counter() - begin

    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise RuntimeError("Unexpected ranking result")

    players, scores = result[:2]
    if not players or len(players) != len(scores):
        raise RuntimeError("Invalid ranking")

    core_sec = sum(timings.values())

    if core_sec <= 0 or core_sec > function_sec * 1.1:
        raise RuntimeError("Invalid attribution timing")

    if method == "ccbr_igd4":
        if counts.get("ig_diff_scores_for_layer", 0) == 0:
            raise RuntimeError(
                "IGD difference function was not executed"
            )

    record = {
        "dataset": dataset,
        "method": method,
        "trial": trial,
        "samples": 512,
        "seed": 256,
        "attribution_sec": round(core_sec, 6),
        "localization_function_sec": round(function_sec, 6),
        "neurons_ranked": len(players),
        "calls": counts,
    }

    print("@@TIMING@@" + json.dumps(record), flush=True)

def run_all(repeats):
    prepare()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT / ("run_" + stamp)
    run_dir.mkdir(parents=True, exist_ok=False)

    records = []

    for dataset in ("mnist", "cifar10"):
        for rep in range(repeats + 1):
            trial = "warmup" if rep == 0 else "rep%02d" % rep

            methods = (
                ("gradact", "ccbr_igd4")
                if rep % 2 == 0
                else ("ccbr_igd4", "gradact")
            )

            for method in methods:
                trial_key = stamp + "_" + dataset + "_" + trial
                logfile = run_dir / (
                    dataset + "_" + method + "_" + trial + ".log"
                )

                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--once", method, dataset, trial_key,
                ]

                print(
                    "RUN", dataset, method, trial,
                    flush=True
                )

                with logfile.open("w", encoding="utf-8") as f:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(SRC),
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )

                content = logfile.read_text(
                    encoding="utf-8", errors="replace"
                )

                found = [
                    line for line in content.splitlines()
                    if line.startswith("@@TIMING@@")
                ]

                if proc.returncode != 0 or len(found) != 1:
                    print(
                        "\n".join(content.splitlines()[-40:]),
                        flush=True,
                    )
                    raise RuntimeError(
                        "Benchmark failed: " + str(logfile)
                    )

                record = json.loads(
                    found[0].split("@@TIMING@@", 1)[1]
                )

                if rep > 0:
                    records.append(record)

                print(
                    "  core={:.3f}s function={:.3f}s".format(
                        record["attribution_sec"],
                        record["localization_function_sec"],
                    ),
                    flush=True,
                )

    summary = {}

    for dataset in ("mnist", "cifar10"):
        summary[dataset] = {}

        for method in SOURCES:
            selected = [
                row for row in records
                if row["dataset"] == dataset
                and row["method"] == method
            ]

            core = [
                row["attribution_sec"] for row in selected
            ]
            full = [
                row["localization_function_sec"]
                for row in selected
            ]

            summary[dataset][method] = {
                "core_mean": statistics.mean(core),
                "core_sd": (
                    statistics.stdev(core)
                    if len(core) > 1 else 0
                ),
                "function_mean": statistics.mean(full),
                "function_sd": (
                    statistics.stdev(full)
                    if len(full) > 1 else 0
                ),
            }

        a = summary[dataset]["gradact"]["core_mean"]
        b = summary[dataset]["ccbr_igd4"]["core_mean"]
        summary[dataset]["ratio"] = b / a

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    columns = [
        "dataset", "method", "trial", "seed", "samples",
        "attribution_sec", "localization_function_sec",
        "neurons_ranked",
    ]

    with (run_dir / "raw_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in records:
            writer.writerow({
                key: row.get(key) for key in columns
            })

    print("\n========== FINAL SUMMARY ==========")

    for dataset, values in summary.items():
        a = values["gradact"]
        b = values["ccbr_igd4"]

        print(
            "{} GradAct={:.3f}+/-{:.3f}s "
            "CCBR-IGD4={:.3f}+/-{:.3f}s ratio={:.2f}x"
            .format(
                dataset,
                a["core_mean"], a["core_sd"],
                b["core_mean"], b["core_sd"],
                values["ratio"],
            )
        )

    print("RESULTS:", run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", nargs=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.once:
        run_once(*args.once)
    else:
        run_all(args.repeats)
