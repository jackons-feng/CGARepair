#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RQ4 = ROOT / "cgarepair-rq4"
OUT = RQ4 / "results/famo_sensitivity"
LOGS = RQ4 / "logs/rq4c"

SEEDS = (42, 123, 256)

TASKS = {
    "adv": ("mnist", "adv/mnist/epsilon_0.05/CGARepair"),
    "backdoor": ("fashion", "fashion/CGARepair"),
    "safety": ("a33", "a33/CGARepair"),
    "fairness": ("bank", "bank/CGARepair"),
}

VARIANTS = {
    "decay": [("0001", 0.0001), ("01", 0.01)],
    "lr": [("0125", 0.0125), ("05", 0.05)],
}

manifest = json.loads(
    (RQ4 / "config/famo_runs.json").read_text(encoding="utf-8")
)

index = {
    (r["task"], r["model"], int(r["seed"])): r
    for r in manifest["records"]
}


def resolved(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read(path):
    if not path.is_file():
        raise RuntimeError("Missing: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def reference(task, seed):
    model = TASKS[task][0]
    r = index[(task, model, seed)]
    summary = read(resolved(r["summary"]))
    test = read(resolved(r["test"]))
    return summary, test


def hyperparams(task, summary):
    key = "optimizer_params" if task == "fairness" else "optimizer_parameters"
    hp = summary[key]

    if int(summary["gradient_steps_actual"]) != 5000:
        raise RuntimeError("RQ2 steps mismatch")
    if not math.isclose(float(hp["task_lr"]), 0.025, abs_tol=1e-10):
        raise RuntimeError("RQ2 task_lr mismatch")
    if not math.isclose(float(hp["task_decay"]), 0.001, abs_tol=1e-10):
        raise RuntimeError("RQ2 task_decay mismatch")
    return hp


def jobs(group):
    result = []
    for family in ("decay", "lr"):
        if group not in ("all", family):
            continue

        tasks = (
            ("adv", "backdoor", "safety", "fairness")
            if family == "decay" else ("adv", "fairness")
        )

        for task in tasks:
            for label, value in VARIANTS[family]:
                for seed in SEEDS:
                    result.append((family, task, label, value, seed))
    return result


def output_dir(job):
    family, task, label, value, seed = job
    relative = TASKS[task][1]
    return (
        OUT / relative / ("seed_" + str(seed))
        / ("famo_rq4c_{}_{}".format(family, label))
    )


def commands(job):
    family, task, label, value, seed = job
    model = TASKS[task][0]
    ref, _ = reference(task, seed)
    hp = hyperparams(task, ref)

    task_dir = ROOT / ("cfrepair_" + task)
    script = task_dir / (task + "_famo_2.py")
    test_script = task_dir / (task + "_famo_test_2.py")

    if not script.is_file() or not test_script.is_file():
        raise RuntimeError("FAMO source/test missing: " + task)

    loc = resolved(ref["localization"]["source_json"])
    if not loc.is_file():
        raise RuntimeError("Locator missing: {}".format(loc))

    tag = "rq4c_{}_{}".format(family, label)

    lr = value if family == "lr" else float(hp["task_lr"])
    decay = value if family == "decay" else float(hp["task_decay"])

    common = [
        "--loc-json", str(loc),
        "--loc-seed", "256",
        "--loc-samples", "4096" if task == "fairness" else "512",
        "--output-root", str(OUT),
        "--run-tag", tag,
        "--max-steps", "5000",
        "--model-lr", str(hp["model_lr"]),
        "--task-lr", str(lr),
        "--task-decay", str(decay),
        "--loss-eps", str(hp["loss_eps"]),
        "--max-grad-norm", str(hp["max_grad_norm"]),
        "--search-radius", str(hp["search_radius"]),
        "--archive-size", str(hp["archive_size"]),
        "--eval-every", str(hp["full_train_eval_every"]),
    ]

    if task == "adv":
        positional = ["mnist", "0.05", str(seed)]
        extra = [
            "--eval-samples", "10000",
            "--grad-batch-size", str(hp["grad_batch_size_per_task"]),
            "--batch-size", str(hp["evaluation_batch_size"]),
            "--output-mode", str(hp["output_mode"]),
            "--threshold-acc", str(ref["threshold_accuracy"]),
            "--threshold-adv-error", str(ref["threshold_adv_error"]),
        ]

    elif task == "backdoor":
        positional = ["fashion", str(seed)]
        extra = [
            "--eval-samples", str(hp["train_eval_samples"]),
            "--grad-batch-size", str(hp["grad_batch_size_per_task"]),
            "--batch-size", str(hp["evaluation_batch_size"]),
            "--output-mode", str(hp["output_mode"]),
            "--sr-definition", str(hp["sr_definition"]),
            "--threshold-acc", str(ref["threshold_acc"]),
            "--threshold-sr", str(ref["threshold_sr"]),
        ]

    elif task == "safety":
        positional = ["a33", str(seed)]
        extra = [
            "--eval-samples", str(hp["train_eval_samples"]),
            "--grad-batch-size", str(hp["grad_batch_size_per_task"]),
            "--batch-size", str(hp["evaluation_batch_size"]),
            "--safety-temperature", str(hp["safety_temperature"]),
            "--threshold-acc", str(ref["threshold_drawdown_retention"]),
            "--threshold-safety", str(ref["threshold_safety_violation"]),
        ]

    else:
        positional = ["bank", str(seed)]
        extra = [
            "--eval-samples", str(hp["train_eval_samples"]),
            "--grad-batch-size",
            str(hp["grad_batch_size_per_group_and_utility"]),
            "--eval-batch-size", str(hp["evaluation_batch_size"]),
            "--fair-temperature", str(hp["fair_temperature"]),
            "--output-mode", str(hp["output_mode"]),
            "--max-acc-drop", str(hp["max_acc_drop"]),
            "--gap-target", str(hp["gap_target"]),
            "--top-k", str(len(ref["selected_neurons"])),
        ]

    train = [
        sys.executable, "-u", str(script)
    ] + positional + common + extra

    out = output_dir(job)

    test = [
        sys.executable, "-u", str(test_script)
    ] + positional + [
        "--optimizer", "famo",
        "--run-dir", str(out)
    ]

    if task != "fairness":
        test += ["--test-samples", "0"]

    test += [
        "--batch-size", str(hp["evaluation_batch_size"])
    ]

    return task_dir, train, test


def validate(job, with_test):
    family, task, label, value, seed = job
    out = output_dir(job)

    original, original_test = reference(task, seed)
    s = read(out / "summary.json")
    hp = hyperparams_reference = (
        s["optimizer_params"] if task == "fairness"
        else s["optimizer_parameters"]
    )
    original_hp = hyperparams(task, original)

    if s.get("optimizer") != "famo":
        raise RuntimeError("Optimizer mismatch")
    if int(s.get("optimization_seed", s.get("seed", -1))) != seed:
        raise RuntimeError("Seed mismatch")
    if int(s.get("gradient_steps_actual", -1)) != 5000:
        raise RuntimeError("Step mismatch")

    expected_lr = value if family == "lr" else 0.025
    expected_decay = value if family == "decay" else 0.001

    if not math.isclose(float(hp["task_lr"]), expected_lr, abs_tol=1e-10):
        raise RuntimeError("task_lr mismatch")
    if not math.isclose(float(hp["task_decay"]), expected_decay, abs_tol=1e-10):
        raise RuntimeError("task_decay mismatch")

    for key in (
        "model_lr", "loss_eps", "max_grad_norm",
        "search_radius", "archive_size",
        "full_train_eval_every", "evaluation_batch_size"
    ):
        if not math.isclose(
            float(hp[key]), float(original_hp[key]),
            rel_tol=0, abs_tol=1e-10
        ):
            raise RuntimeError("Changed non-target parameter: " + key)

    loc1 = resolved(s["localization"]["source_json"])
    loc2 = resolved(original["localization"]["source_json"])

    if loc1.resolve() != loc2.resolve():
        raise RuntimeError("Locator mismatch")

    players_key = (
        "selected_neurons" if task == "fairness"
        else "selected_players"
    )
    if s[players_key] != original[players_key]:
        raise RuntimeError("Selected neuron mismatch")

    baseline_key = (
        "baseline_train" if task == "fairness"
        else "original_train_metrics"
    )

    metric_keys = {
        "adv": ("adv_error_rate", "clean_accuracy"),
        "backdoor": ("sr", "clean_accuracy"),
        "safety": ("safety_violation_rate", "drawdown_retention"),
        "fairness": ("fairness_gap", "accuracy"),
    }

    for key in metric_keys[task]:
        a = float(s[baseline_key][key])
        b = float(original[baseline_key][key])
        if abs(a - b) > 1e-7:
            raise RuntimeError("TRAIN baseline mismatch: " + key)

    best = out / "best_weights.h5"
    if not best.is_file() or best.stat().st_size == 0:
        raise RuntimeError("Missing checkpoint")

    if with_test:
        t = read(out / "test_results.json")

        if t.get("optimizer") != "famo":
            raise RuntimeError("TEST optimizer mismatch")

        ck = (
            "repaired_checkpoint" if task == "fairness"
            else "selected_checkpoint"
        )

        if Path(t[ck]).resolve() != best.resolve():
            raise RuntimeError("TEST checkpoint mismatch")

        if task == "fairness":
            if t.get("evaluation_scope") != "full_test":
                raise RuntimeError("Not independent full TEST")
            if t.get("test_used_to_select_checkpoint") is not False:
                raise RuntimeError("TEST selection leakage")
            before = t["original"]
            reference_before = original_test["original"]
        else:
            requested = t.get("test_samples_requested")

            if requested is not None:
                if int(requested) != 0:
                    raise RuntimeError("Not full TEST")

            elif task == "adv":
                data = t.get("test_data")
                reference_data = original_test.get("test_data")

                if not isinstance(data, dict):
                    raise RuntimeError("Missing TEST data evidence")

                if data != reference_data:
                    raise RuntimeError("TEST data differs from RQ2")

                if data.get("split") != "test":
                    raise RuntimeError("Not independent TEST split")

                for prefix in ("adv", "clean"):
                    total = data.get(prefix + "_total")
                    selected = data.get(prefix + "_selected")

                    if total is None or selected is None:
                        raise RuntimeError(
                            "Missing TEST count: " + prefix
                        )

                    if int(total) <= 0 or int(selected) != int(total):
                        raise RuntimeError(
                            "TEST was subsampled: " + prefix
                        )

            else:
                raise RuntimeError(
                    "Missing full-TEST evidence for task " + task
                )

            before = t["original_test_metrics"]
            reference_before = original_test["original_test_metrics"]

        test_keys = {
            "adv": ("adv_error_rate", "clean_accuracy"),
            "backdoor": ("standard_asr", "clean_accuracy"),
            "safety": ("safety_violation_rate", "drawdown_retention"),
            "fairness": ("fairness_gap", "accuracy"),
        }

        for key in test_keys[task]:
            if abs(float(before[key]) -
                   float(reference_before[key])) > 1e-7:
                raise RuntimeError("TEST baseline mismatch: " + key)

    return True


def execute(cmd, cwd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("COMMAND:", " ".join(cmd), flush=True)

    with logfile.open("a", encoding="utf-8") as f:
        f.write("\nCOMMAND: " + " ".join(cmd) + "\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()

        rc = process.wait()

    if rc:
        raise RuntimeError(
            "Exit status {} in {}".format(rc, logfile)
        )


def run(selected):
    total = 0

    for job in selected:
        family, task, label, value, seed = job
        out = output_dir(job)
        taskdir, train, test = commands(job)

        ident = "{}_{}_{}_seed{}".format(
            family, task, label, seed
        )
        print("\n========== {} ==========".format(ident),
              flush=True)

        summary_path = out / "summary.json"
        best_path = out / "best_weights.h5"
        test_path = out / "test_results.json"

        if summary_path.is_file() and best_path.is_file():
            validate(job, False)
            print("[SKIP] Verified TRAIN", flush=True)
        else:
            if out.exists() and any(out.iterdir()):
                raise RuntimeError(
                    "Incomplete output. Inspect: {}".format(out)
                )
            execute(
                train, taskdir,
                LOGS / (ident + "_train.log")
            )
            validate(job, False)

        if test_path.is_file():
            validate(job, True)
            print("[SKIP] Verified TEST", flush=True)
        else:
            execute(
                test, taskdir,
                LOGS / (ident + "_test.log")
            )
            validate(job, True)

        total += 1
        print("[SUCCESS] {}/{} {}".format(
            total, len(selected), ident
        ), flush=True)

    print("\n[PASS] Selected RQ4-C jobs complete:", total)


def summary():
    for family in ("decay", "lr"):
        tasks = (
            ("adv", "backdoor", "safety", "fairness")
            if family == "decay" else ("adv", "fairness")
        )

        print("\n========== {} ==========".format(family.upper()))

        for task in tasks:
            print("\nTASK:", task)

            metric = {
                "adv": ("adv_error_rate", "clean_accuracy"),
                "backdoor": ("standard_asr", "clean_accuracy"),
                "safety": ("safety_violation_rate", "drawdown_retention"),
                "fairness": ("fairness_gap", "accuracy"),
            }[task]

            values = (
                [("0001", 0.0001), ("001", 0.001), ("01", 0.01)]
                if family == "decay"
                else [("0125", 0.0125), ("025", 0.025), ("05", 0.05)]
            )

            center = 0.001 if family == "decay" else 0.025

            for label, value in values:
                defect = []
                utility = []

                for seed in SEEDS:
                    if abs(value - center) < 1e-12:
                        _, t = reference(task, seed)
                    else:
                        job = (family, task, label, value, seed)
                        validate(job, True)
                        t = read(
                            output_dir(job) / "test_results.json"
                        )

                    node = (
                        t["repaired"] if task == "fairness"
                        else t["repaired_test_metrics"]
                    )
                    defect.append(float(node[metric[0]]) * 100)
                    utility.append(float(node[metric[1]]) * 100)

                print(
                    "{}={}  defect={:.4f} +/- {:.4f}%  "
                    "utility={:.4f} +/- {:.4f}%".format(
                        family, value,
                        statistics.mean(defect),
                        statistics.stdev(defect),
                        statistics.mean(utility),
                        statistics.stdev(utility)
                    )
                )

    print("\n[PASS] RQ4-C paired three-seed summary completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("plan", "run", "summary"),
        default="plan"
    )
    parser.add_argument(
        "--group",
        choices=("all", "decay", "lr"),
        default="all"
    )
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    selected = jobs(args.group)

    if args.limit > 0:
        selected = selected[:args.limit]

    if args.action == "plan":
        for job in selected:
            commands(job)
            print(
                "{} task={} seed={} value={} output={}".format(
                    job[0], job[1], job[4],
                    job[3], output_dir(job)
                )
            )
        print("\n[PASS] RQ4-C PLAN: {} jobs".format(len(selected)))

    elif args.action == "run":
        run(selected)

    else:
        summary()


if __name__ == "__main__":
    main()
