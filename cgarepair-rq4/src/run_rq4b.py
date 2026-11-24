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
OUT = RQ4 / "results/famo_budget"
LOGS = RQ4 / "logs/rq4b"

SEEDS = (42, 123, 256, 1024, 2026)
BUDGETS = (1000, 2500)


def _resolve_release_paths(value, root):
    if isinstance(value, dict):
        return {k: _resolve_release_paths(v, root)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_release_paths(v, root) for v in value]
    if isinstance(value, str):
        prefixes = (
            "adv/", "backdoor/", "fairness/", "safety/",
            "rq1/", "model/", "cgarepair-rq4/"
        )
        if value.startswith(prefixes):
            return str((Path(root) / value).resolve())
    return value


MANIFEST = json.loads(
    (RQ4 / "config/famo_runs.json").read_text(encoding="utf-8")
)


MANIFEST = _resolve_release_paths(MANIFEST, ROOT)

INDEX = {
    (r["task"], r["model"], int(r["seed"])): r
    for r in MANIFEST["records"]
}


def path_from_record(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_json(path):
    if not path.is_file():
        raise RuntimeError("Missing file: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def reference(task, seed):
    model = "mnist" if task == "adv" else "bank"
    record = INDEX[(task, model, seed)]
    summary = read_json(path_from_record(record["summary"]))
    test = read_json(path_from_record(record["test"]))
    return summary, test


def output_dir(task, seed, budget):
    tag = "famo_rq4b_steps{}".format(budget)
    if task == "adv":
        return (
            OUT / "adv/mnist/epsilon_0.05/CGARepair"
            / ("seed_" + str(seed)) / tag
        )
    return (
        OUT / "bank/CGARepair"
        / ("seed_" + str(seed)) / tag
    )


def make_commands(task, seed, budget):
    ref, _ = reference(task, seed)

    hp = ref[
        "optimizer_parameters" if task == "adv"
        else "optimizer_params"
    ]

    loc = Path(ref["localization"]["source_json"])
    if not loc.is_file():
        raise RuntimeError("Missing locator: {}".format(loc))

    taskdir = ROOT / ("cfrepair_" + task)

    common = [
        "--loc-json", str(loc),
        "--loc-seed", "256",
        "--max-steps", str(budget),
        "--model-lr", str(hp["model_lr"]),
        "--task-lr", str(hp["task_lr"]),
        "--task-decay", str(hp["task_decay"]),
        "--loss-eps", str(hp["loss_eps"]),
        "--max-grad-norm", str(hp["max_grad_norm"]),
        "--search-radius", str(hp["search_radius"]),
        "--archive-size", str(hp["archive_size"]),
        "--output-mode", str(hp["output_mode"]),
        "--output-root", str(OUT),
        "--run-tag", "rq4b_steps{}".format(budget),
    ]

    output = output_dir(task, seed, budget)

    if task == "adv":
        train_script = taskdir / "adv_famo_2.py"
        test_script = taskdir / "adv_famo_test_2.py"

        train = [
            sys.executable, "-u", str(train_script),
            "mnist", "0.05", str(seed)
        ] + common + [
            "--loc-samples", "512",
            "--eval-samples", "10000",
            "--grad-batch-size",
            str(hp["grad_batch_size_per_task"]),
            "--batch-size",
            str(hp["evaluation_batch_size"]),
            "--eval-every",
            str(hp["full_train_eval_every"]),
            "--threshold-acc",
            str(ref["threshold_accuracy"]),
            "--threshold-adv-error",
            str(ref["threshold_adv_error"]),
        ]

        test = [
            sys.executable, "-u", str(test_script),
            "mnist", "0.05", str(seed),
            "--optimizer", "famo",
            "--run-dir", str(output),
            "--test-samples", "0",
            "--batch-size",
            str(hp["evaluation_batch_size"]),
        ]

    else:
        train_script = taskdir / "fairness_famo_2.py"
        test_script = taskdir / "fairness_famo_test_2.py"

        train = [
            sys.executable, "-u", str(train_script),
            "bank", str(seed)
        ] + common + [
            "--loc-samples", "4096",
            "--top-k", str(len(ref["selected_neurons"])),
            "--eval-samples", str(hp["train_eval_samples"]),
            "--grad-batch-size",
            str(hp["grad_batch_size_per_group_and_utility"]),
            "--eval-batch-size",
            str(hp["evaluation_batch_size"]),
            "--eval-every",
            str(hp["full_train_eval_every"]),
            "--fair-temperature",
            str(hp["fair_temperature"]),
            "--max-acc-drop",
            str(hp["max_acc_drop"]),
            "--gap-target",
            str(hp["gap_target"]),
        ]

        test = [
            sys.executable, "-u", str(test_script),
            "bank", str(seed),
            "--optimizer", "famo",
            "--run-dir", str(output),
            "--batch-size",
            str(hp["evaluation_batch_size"]),
        ]

    for script in (train_script, test_script):
        if not script.is_file():
            raise RuntimeError("Missing source: {}".format(script))

    return taskdir, output, train, test


def validate(task, seed, budget, output, need_test):
    ref, ref_test = reference(task, seed)
    s = read_json(output / "summary.json")
    best = output / "best_weights.h5"

    if not best.is_file() or best.stat().st_size == 0:
        raise RuntimeError("Missing checkpoint: {}".format(best))

    if s.get("optimizer") != "famo":
        raise RuntimeError("Optimizer mismatch")

    actual_seed = s.get(
        "optimization_seed", s.get("seed", -1)
    )

    if int(actual_seed) != seed:
        raise RuntimeError("Seed mismatch")

    if int(s.get("gradient_steps_actual", -1)) != budget:
        raise RuntimeError("Gradient-step budget mismatch")

    new_loc = Path(s["localization"]["source_json"]).resolve()
    old_loc = Path(ref["localization"]["source_json"]).resolve()

    if new_loc != old_loc:
        raise RuntimeError("Localization mismatch")

    selected_key = (
        "selected_players" if task == "adv"
        else "selected_neurons"
    )

    if s[selected_key] != ref[selected_key]:
        raise RuntimeError("Selected neurons mismatch")

    if task == "adv":
        baseline_key = "original_train_metrics"
        keys = ("adv_error_rate", "clean_accuracy")
    else:
        baseline_key = "baseline_train"
        keys = ("fairness_gap", "accuracy")

    for key in keys:
        a = float(s[baseline_key][key])
        b = float(ref[baseline_key][key])
        if abs(a - b) > 1e-7:
            raise RuntimeError(
                "TRAIN baseline mismatch: {}".format(key)
            )

    old_data = ref.get("train_data", {})
    new_data = s.get("train_data", {})

    for key in ("selection_index_sha256", "selected_count"):
        if key in old_data and key in new_data:
            if old_data[key] != new_data[key]:
                raise RuntimeError("TRAIN data mismatch: " + key)

    if need_test:
        t = read_json(output / "test_results.json")

        if t.get("optimizer") != "famo":
            raise RuntimeError("TEST optimizer mismatch")

        checkpoint_key = (
            "selected_checkpoint" if task == "adv"
            else "repaired_checkpoint"
        )

        if Path(t[checkpoint_key]).resolve() != best.resolve():
            raise RuntimeError("TEST checkpoint mismatch")

        if task == "adv":
            original_key = "original_test_metrics"
        else:
            original_key = "original"

        for key in keys:
            a = float(t[original_key][key])
            b = float(ref_test[original_key][key])
            if abs(a - b) > 1e-7:
                raise RuntimeError(
                    "TEST baseline mismatch: {}".format(key)
                )

        if task == "fairness":
            if t.get("evaluation_scope") != "full_test":
                raise RuntimeError("Not full TEST")
            if t.get("test_used_to_select_checkpoint") is not False:
                raise RuntimeError("TEST selection leakage")

    return True


def execute(command, cwd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)

    print("COMMAND:", " ".join(command), flush=True)

    with logfile.open("a", encoding="utf-8") as f:
        f.write("\nCOMMAND: " + " ".join(command) + "\n")
        f.flush()

        p = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()

        rc = p.wait()

    if rc != 0:
        raise RuntimeError(
            "Exit code {}. See {}".format(rc, logfile)
        )


def run():
    completed = 0

    for task in ("adv", "fairness"):
        for budget in BUDGETS:
            for seed in SEEDS:
                directory, out, train, test = make_commands(
                    task, seed, budget
                )

                name = "{}_steps{}_seed{}".format(
                    task, budget, seed
                )

                print("\n========== {} ==========".format(name),
                      flush=True)

                summary = out / "summary.json"
                checkpoint = out / "best_weights.h5"
                testfile = out / "test_results.json"

                if summary.is_file() and checkpoint.is_file():
                    validate(task, seed, budget, out, False)
                    print("[SKIP] Verified TRAIN", flush=True)
                else:
                    if out.exists() and any(out.iterdir()):
                        raise RuntimeError(
                            "Incomplete output; inspect: {}".format(out)
                        )

                    execute(
                        train, directory,
                        LOGS / (name + "_train.log")
                    )
                    validate(task, seed, budget, out, False)

                if testfile.is_file():
                    validate(task, seed, budget, out, True)
                    print("[SKIP] Verified TEST", flush=True)
                else:
                    execute(
                        test, directory,
                        LOGS / (name + "_test.log")
                    )
                    validate(task, seed, budget, out, True)

                completed += 1
                print("[SUCCESS] {} ({}/20)".format(
                    name, completed
                ), flush=True)

    print("\n[PASS] RQ4-B ALL 20 TRAIN + TEST COMPLETE")


def summary():
    for task in ("adv", "fairness"):
        print("\n========== {} ==========".format(
            task.upper()
        ))

        defect_key = (
            "adv_error_rate" if task == "adv"
            else "fairness_gap"
        )
        utility_key = (
            "clean_accuracy" if task == "adv"
            else "accuracy"
        )
        repaired_key = (
            "repaired_test_metrics" if task == "adv"
            else "repaired"
        )

        for budget in (1000, 2500, 5000):
            values = []

            for seed in SEEDS:
                if budget == 5000:
                    record = INDEX[
                        (task, "mnist" if task == "adv"
                         else "bank", seed)
                    ]
                    path = path_from_record(record["test"])
                else:
                    path = (
                        output_dir(task, seed, budget)
                        / "test_results.json"
                    )

                data = read_json(path)[repaired_key]
                values.append((
                    float(data[defect_key]) * 100,
                    float(data[utility_key]) * 100,
                ))

            d = [v[0] for v in values]
            u = [v[1] for v in values]

            print(
                "steps={:<5} defect={:8.4f} +/- {:7.4f}% "
                "utility={:8.4f} +/- {:7.4f}%".format(
                    budget,
                    statistics.mean(d),
                    statistics.stdev(d),
                    statistics.mean(u),
                    statistics.stdev(u),
                )
            )

    print("\n[PASS] RQ4-B summary completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("plan", "run", "summary"),
        default="plan"
    )
    args = parser.parse_args()

    if args.action == "plan":
        for task in ("adv", "fairness"):
            for seed in SEEDS:
                ref, _ = reference(task, seed)
                hp = ref[
                    "optimizer_parameters" if task == "adv"
                    else "optimizer_params"
                ]
                if int(ref["gradient_steps_actual"]) != 5000:
                    raise RuntimeError("RQ2 budget mismatch")
                if abs(float(hp["task_decay"]) - 0.001) > 1e-10:
                    raise RuntimeError("RQ2 task_decay mismatch")
                for budget in BUDGETS:
                    _, output, _, _ = make_commands(
                        task, seed, budget
                    )
                    print(
                        "{} seed={} steps={} output={}".format(
                            task, seed, budget, output
                        )
                    )

        print("\n[PASS] RQ4-B PLAN 20 JOBS READY")

    elif args.action == "run":
        run()
    else:
        summary()


if __name__ == "__main__":
    main()
