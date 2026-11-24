#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TEST evaluator for the ORIGINAL safety_moo_1.py (CARE-localized PSO).

CARE repair is NOT changed. Load the fixed PSO checkpoint saved by it:
    loc_result/<model>/<seed>/pso/weights_best.h5

Use the SAME held-out loading and class-decision metrics as safety_test_2.py:
    cgarepair_safety_utils.load_safety_data(model, split='test', ...)
    cgarepair_safety_utils.calculate_metrics(...)

IMPORTANT SCIENTIFIC CAVEAT: unmodified safety_moo_1.py samples its TRAIN
repair examples from the full counterexample.h5 and drawndown.h5 pools. If the
TEST split is derived from those same pools, leakage is possible. This script
warns and reports the unverified status; a test-only change cannot fix the
original training split. Never use TEST to choose a checkpoint.

Examples:
  python -u safety_care_test.py a19 256
  python -u safety_care_test.py a29 42
  python -u safety_care_test.py a33 2026 --evaluate-front
"""
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
MODELS = ('a19', 'a29', 'a33')


def get_parser():
    p = argparse.ArgumentParser(description='TEST original CARE PSO safety repair')
    p.add_argument('model_name', choices=MODELS)
    p.add_argument('seed', type=int)
    p.add_argument('--repair-dir', default=None, help='Directory with weights_best.h5')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--model-root', default=None)
    p.add_argument('--data-root', default=None)
    p.add_argument('--test-samples', type=int, default=0,
                   help='0 evaluates all TEST examples; >0 is smoke test only')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--evaluate-front', action='store_true',
                   help='Evaluate saved Pareto models only for diagnostics; never TEST-select')
    p.add_argument('--require-train-provenance', action='store_true',
                   help='Require independently generated TRAIN-only provenance file')
    p.add_argument('--allow-legacy-training', action='store_true',
                   help=argparse.SUPPRESS)  # backwards compatible, no longer required
    p.add_argument('--overwrite', action='store_true')
    return p


def write_json(path, obj):
    with Path(path).open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda x: x.item() if hasattr(x, 'item') else str(x))
        f.write('\n')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_repair_record(path):
    # The original CARE writes its training information as metric,value CSV.
    if not path.is_file():
        raise FileNotFoundError('CARE repair_result.txt missing: {}'.format(path))
    result = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            key = (row.get('metric') or '').strip()
            value = (row.get('value') or '').strip()
            if key and value:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    if not result or 'initial_safety' not in result:
        raise ValueError('CARE metadata format invalid: {}'.format(path))
    return result


def load_training_provenance(repair_dir, model_name, seed, require):
    path = repair_dir / 'care_train_provenance.json'
    if path.is_file():
        with path.open('r', encoding='utf-8') as f:
            info = json.load(f)
        verified = (info.get('split') == 'train' and
                    info.get('data_loader') == 'cgarepair_safety_utils.load_safety_data' and
                    info.get('model_name') == model_name and
                    str(info.get('seed')) == str(seed))
        if not verified:
            raise ValueError('Existing CARE train provenance does not match model/seed/split: {}'
                             .format(path))
        return info, True
    if require:
        raise RuntimeError('TRAIN provenance absent: {}'.format(path))
    print('[WARNING] Original safety_moo_1.py does NOT prove TRAIN/TEST disjointness. '
          'Its random training subset comes from full HDF5 pools; possible leakage. '
          'Results are exploratory until the split is audited.', flush=True)
    return None, False


def run(args):
    if args.test_samples < 0 or args.batch_size < 1:
        raise ValueError('--test-samples must be >=0 and --batch-size must be >=1')
    start = time.perf_counter()
    directory = (Path(args.repair_dir).expanduser().resolve() if args.repair_dir else
                 Path(args.output_root).expanduser() / args.model_name / str(args.seed) / 'pso')
    weights = directory / 'weights_best.h5'
    if not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError('Original CARE PSO checkpoint missing: {}'.format(weights))
    train_record = read_repair_record(directory / 'repair_result.txt')
    checkpoint_sha256 = sha256_file(weights)
    train_info, disjointness_verified_for_repair = load_training_provenance(
        directory, args.model_name, args.seed, args.require_train_provenance)

    outfile = directory / 'care_test_results.json'
    csvfile = directory / 'care_test_metrics.csv'
    timefile = directory / 'care_time_summary.json'
    if outfile.exists() and not args.overwrite:
        raise FileExistsError('{} already exists; use --overwrite'.format(outfile))

    # Import lazily to keep --help usable without TensorFlow installed.
    try:
        from cgarepair_safety_utils import (
            initialize_tf, load_safety_data, model_file, calculate_metrics)
    except ImportError as exc:
        raise ImportError('cgarepair_safety_utils.py is needed to use the same TEST '
                          'definition as safety_test_2.py. Put this script beside it. '
                          'Original import error: {}'.format(exc)) from exc

    t = time.perf_counter()
    _, _, sess = initialize_tf(args.seed)
    from tensorflow.keras.models import load_model
    original_file = model_file(args.model_name, args.model_root)
    model = load_model(str(original_file), compile=False)
    model_sec = time.perf_counter() - t

    t = time.perf_counter()
    ct, dr, test_provenance = load_safety_data(
        args.model_name, split='test', data_root=args.data_root,
        samples=args.test_samples, seed=args.seed)
    if len(ct) == 0 or len(dr) == 0:
        raise ValueError('Empty CARE safety TEST data')
    data_sec = time.perf_counter() - t

    t = time.perf_counter()
    baseline = calculate_metrics(model, args.model_name, ct, dr, sess, args.batch_size)
    model.load_weights(str(weights))
    repaired = calculate_metrics(model, args.model_name, ct, dr, sess, args.batch_size)
    eval_sec = time.perf_counter() - t

    print('[CARE TEST] model={} seed={} n_counterexample={} n_drawdown={}'.format(
        args.model_name, args.seed, len(ct), len(dr)), flush=True)
    print('[CARE TEST] before safety={:.6f} retention={:.6f}'.format(
        baseline['safety_violation_rate'], baseline['drawdown_retention']), flush=True)
    print('[CARE TEST] after  safety={:.6f} retention={:.6f}'.format(
        repaired['safety_violation_rate'], repaired['drawdown_retention']), flush=True)

    t = time.perf_counter()
    front = []
    if args.evaluate_front:
        for ckpt in sorted((directory / 'pareto_front').glob('weights_*.h5')):
            model.load_weights(str(ckpt))
            metrics = calculate_metrics(model, args.model_name, ct, dr, sess, args.batch_size)
            front.append({'checkpoint': str(ckpt.resolve()), 'test_metrics': metrics})
        if front:
            with (directory / 'care_pareto_test_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['checkpoint', 'test_safety_violation', 'test_drawdown_retention'])
                for r in front:
                    w.writerow([r['checkpoint'], r['test_metrics']['safety_violation_rate'],
                                r['test_metrics']['drawdown_retention']])
        model.load_weights(str(weights))
    front_sec = time.perf_counter() - t

    total = time.perf_counter() - start
    train_opt_sec = train_record.get('elapsed_seconds')
    timing = {
        'care_optimization_previous_sec': train_opt_sec,
        'care_localization_previous_sec': None,
        'test_model_load_sec': model_sec,
        'test_data_load_sec': data_sec,
        'test_before_after_eval_sec': eval_sec,
        'test_optional_front_eval_sec': front_sec,
        'test_total_wall_sec': total,
        'optimization_plus_test_sec': (train_opt_sec + total
                                       if isinstance(train_opt_sec, (int, float)) else None),
        'note': 'CARE optimization time excludes localization and some overhead; sum is '
                'not a measured end-to-end time. TEST never selects the checkpoint.',
    }
    record = {
        'schema': 'CARE_PSO_original_safety_moo_1_test_v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'CARE_PSO', 'task': 'safety',
        'model': args.model_name, 'seed': args.seed,
        'checkpoint': str(weights.resolve()),
        'selected_checkpoint': str(weights.resolve()),
        'checkpoint_sha256': checkpoint_sha256,
        'requested_evaluate_front': bool(args.evaluate_front),
        'checkpoint_selection': 'Original CARE PSO training fitness; fixed before TEST',
        'original_training_file': 'safety_moo_1.py',
        'original_pretrained_model': str(Path(original_file).resolve()),
        'train_only_provenance_certified': disjointness_verified_for_repair,
        'care_locator_disjointness_verified': False,
        'possible_train_test_overlap': not disjointness_verified_for_repair,
        'split_warning': ('Original safety_moo_1.py samples from full HDF5 pools. '
                          'Unless proven otherwise, its TRAIN data may overlap this TEST split.'),
        'train_provenance': train_info, 'train_metrics': train_record,
        'test_data': test_provenance,
        'test_samples_requested': int(args.test_samples),
        'full_test_split_evaluated': (args.test_samples == 0),
        'test_counterexample_count': len(ct), 'test_drawdown_count': len(dr),
        'original_test_metrics': baseline, 'repaired_test_metrics': repaired,
        'delta_safety_violation_rate': (repaired['safety_violation_rate'] -
                                        baseline['safety_violation_rate']),
        'delta_drawdown_retention': (repaired['drawdown_retention'] -
                                     baseline['drawdown_retention']),
        'optional_pareto_test_results': front,
        'timing_sec': timing, 'total_time_sec': total,
    }
    write_json(outfile, record)
    with csvfile.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'train_provenance_verified', 'n_counterexample_test',
                    'n_drawdown_test', 'safety_before', 'safety_after', 'delta_safety',
                    'retention_before', 'retention_after', 'delta_retention',
                    'optimization_sec', 'test_sec'])
        w.writerow([args.model_name, args.seed, int(disjointness_verified_for_repair),
                    len(ct), len(dr), baseline['safety_violation_rate'],
                    repaired['safety_violation_rate'], record['delta_safety_violation_rate'],
                    baseline['drawdown_retention'], repaired['drawdown_retention'],
                    record['delta_drawdown_retention'], train_opt_sec, total])
    write_json(timefile, {'schema': 'CARE_PSO_test_timing_v2',
                          'model': args.model_name, 'seed': args.seed,
                          'timing_sec': timing})
    print('[CARE TEST] Saved {}, {}, {}'.format(outfile, csvfile, timefile), flush=True)
    return record


def main():
    run(get_parser().parse_args())


if __name__ == '__main__':
    main()
