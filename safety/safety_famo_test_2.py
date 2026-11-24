"""Full independent TEST for CGARepair safety masked FAMO.

Reads the one best_weights.h5 selected on TRAIN by safety_famo_2.py.
TEST is never used to select or replace this checkpoint. Preserves the old
VR (argmin unsafe class) / drawdown retention metrics, adds strict Kernel mask
and pretrained model SHA256 audits, and optional snapshot diagnostics.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from read_CGARepair import model_name_checked
from cgarepair_safety_utils import (
    initialize_tf, load_safety_data, model_file, calculate_metrics, save_json,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def mask_audit(before, after, specs, tol=1.e-7):
    if len(before) != len(after):
        raise ValueError('Model weight tensor list size changed')
    allow = {}
    for spec in specs:
        allow.setdefault(int(spec['weight_index']), set()).add(int(spec['neuron_index']))
    changed, outside = 0, 0
    touched = []
    for idx, (a, b) in enumerate(zip(before, after)):
        if a.shape != b.shape or not np.all(np.isfinite(b)):
            raise ValueError('Repaired tensor invalid: {}'.format(idx))
        diff = np.abs(np.asarray(b, dtype=np.float64)-np.asarray(a, dtype=np.float64)) > tol
        n = int(np.count_nonzero(diff))
        if not n:
            continue
        changed += n
        if idx in allow:
            mask = np.zeros(a.shape, dtype=bool)
            for channel in allow[idx]:
                if not 0 <= channel < mask.shape[-1]:
                    raise ValueError('Selected neuron index outside Kernel shape')
                mask[..., channel] = True
            outside += int(np.count_nonzero(diff & ~mask))
        else:
            outside += n
        touched.append({'weight_array_index': idx, 'changed_parameters': n})
    if outside:
        raise RuntimeError('FAMO touched {} weights OUTSIDE GradAct selected Kernel'.format(outside))
    return {'changed_parameters': changed, 'changed_outside_selected_neurons': outside,
            'modified_weight_arrays': touched, 'comparison_tolerance': tol, 'passed': True}


def get_parser():
    p = argparse.ArgumentParser(description='Independent TEST for safety FAMO Kernel repair')
    p.add_argument('model_name', choices=('a19', 'a29', 'a33'))
    p.add_argument('seed', type=int)
    p.add_argument('--optimizer', '--algorithm', choices=('famo',), default='famo')
    p.add_argument('--run-tag', default='paper_gamma001_steps5000')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--run-dir', help='Explicit directory containing TRAIN summary.json')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--test-samples', type=int, default=0,
                   help='0 uses the FULL independent counterexample and drawdown TEST pools')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--evaluate-front', action='store_true',
                   help='Diagnostic eval of saved candidates; never changes selected weights')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--skip-model-hash-check', action='store_true')
    return p


def run(args):
    start = time.perf_counter()
    model_name = model_name_checked(args.model_name)
    suffix = ('_'+args.run_tag) if args.run_tag else ('_quick' if args.quick else '')
    directory = (Path(args.run_dir).expanduser().resolve() if args.run_dir else
                 Path(args.output_root).expanduser()/model_name/'CGARepair'/
                 ('seed_'+str(args.seed))/('famo'+suffix))
    summary_file = directory/'summary.json'
    if not summary_file.is_file():
        raise FileNotFoundError('Missing FAMO TRAIN summary: {}'.format(summary_file))
    with summary_file.open('r', encoding='utf-8') as f:
        repair = json.load(f)
    if (repair.get('task') != 'safety' or repair.get('model') != model_name or
            repair.get('optimizer') != 'famo' or
            int(repair.get('optimization_seed', -1)) != args.seed):
        raise ValueError('TEST model/optimizer/seed disagree with FAMO TRAIN summary')
    selected_file = directory/'best_weights.h5'
    if not selected_file.is_file():
        raise FileNotFoundError('Missing TRAIN-selected best_weights.h5: {}'.format(selected_file))
    test_file = directory/'test_results.json'
    if test_file.exists() and not args.overwrite:
        raise FileExistsError('Existing independent TEST result: {}'.format(test_file))
    if args.test_samples < 0 or args.batch_size <= 0:
        raise ValueError('Invalid TEST sample count or batch size')

    stage = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    try:
        K.set_learning_phase(0)
    except AttributeError:
        tf.compat.v1.keras.backend.set_learning_phase(0)
    from tensorflow.keras.models import load_model
    path = model_file(model_name, args.model_root)
    if (not args.skip_model_hash_check and
            sha256_file(path) != repair.get('model_sha256')):
        raise RuntimeError('Pretrained model SHA256 differs from TRAIN source model')
    model = load_model(path, compile=False)
    original_weights = [w.copy() for w in model.get_weights()]
    model_sec = time.perf_counter()-stage

    stage = time.perf_counter()
    ct, dr, data_info = load_safety_data(
        model_name, split='test', data_root=args.data_root,
        samples=args.test_samples, seed=args.seed)
    ct = np.asarray(ct)
    dr = np.asarray(dr)
    test_data_sec = time.perf_counter()-stage

    stage = time.perf_counter()
    original = calculate_metrics(model, model_name, ct, dr, session, args.batch_size)
    model.load_weights(str(selected_file))
    repaired_weights = model.get_weights()
    audit = mask_audit(original_weights, repaired_weights, repair['repair_specs'])
    repaired = calculate_metrics(model, model_name, ct, dr, session, args.batch_size)
    eval_sec = time.perf_counter()-stage
    print('[TEST] {} safety FAMO seed={} ct={} drawdown={}'.format(
        model_name, args.seed, len(ct), len(dr)), flush=True)
    print('[TEST] BEFORE VR={:.6f} Retention={:.6f}'.format(
        original['safety_violation_rate'], original['drawdown_retention']), flush=True)
    print('[TEST] AFTER  VR={:.6f} Retention={:.6f}'.format(
        repaired['safety_violation_rate'], repaired['drawdown_retention']), flush=True)
    print('[Mask] modified={} outside_selected=0'.format(audit['changed_parameters']), flush=True)

    diagnostic = []
    stage = time.perf_counter()
    if args.evaluate_front:
        for checkpoint in sorted((directory/'pareto_front').glob('weights_*.h5')):
            model.load_weights(str(checkpoint))
            m = calculate_metrics(model, model_name, ct, dr, session, args.batch_size)
            diagnostic.append({'checkpoint': str(checkpoint.resolve()), 'metrics': m})
        if diagnostic:
            with open(str(directory/'pareto_test_results.csv'), 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['checkpoint', 'test_violation', 'test_retention'])
                for item in diagnostic:
                    m = item['metrics']
                    writer.writerow([item['checkpoint'], m['safety_violation_rate'],
                                     m['drawdown_retention']])
        model.load_weights(str(selected_file))
    diagnostic_sec = time.perf_counter()-stage
    total = float(time.perf_counter()-start)
    before_times = repair.get('timing_sec', {})
    previous_loc = before_times.get('localization_previous_run_sec')
    previous_repair = before_times.get('total_repair_run_wall_sec')
    combined = (float(previous_loc)+float(previous_repair)+total if
                previous_loc is not None and previous_repair is not None else None)
    timing = {
        'localization_previous_run_sec': previous_loc,
        'repair_previous_run_sec': previous_repair,
        'test_model_load_sec': float(model_sec),
        'test_data_load_sec': float(test_data_sec),
        'test_selected_checkpoint_eval_sec': float(eval_sec),
        'test_optional_front_diagnostic_sec': float(diagnostic_sec),
        'test_total_wall_sec': total,
        'combined_historical_localization_repair_test_sec': combined,
        'notes': 'Independent TEST did not select/checkpoint. Historical times are from separate runs.',
    }
    result = {
        'schema': 'CGARepair_safety_FAMO_weight_test_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'safety', 'model': model_name, 'optimizer': 'famo',
        'method': 'CGARepair_FAMO', 'optimization_seed': int(args.seed),
        'test_samples_requested': int(args.test_samples),
        'test_data': data_info,
        'selected_checkpoint': str(selected_file.resolve()),
        'checkpoint_selection': 'TRAIN only; TEST never selects checkpoints',
        'checkpoint_selection_policy': repair['selection_policy'],
        'selected_players': repair['selected_players'], 'top_k': int(repair['top_k']),
        'original_test_metrics': original, 'repaired_test_metrics': repaired,
        'delta_safety_violation_rate': float(repaired['safety_violation_rate']-
                                              original['safety_violation_rate']),
        'delta_drawdown_retention': float(repaired['drawdown_retention']-
                                           original['drawdown_retention']),
        'mask_audit': audit,
        'optional_front_test_results': diagnostic,
        'repair_timing_sec': before_times,
        'timing_sec': timing, 'total_time_sec': total,
    }
    save_json(test_file, result)
    save_json(directory/'time_summary.json', {
        'schema': 'CGARepair_safety_FAMO_full_timing_v1',
        'model': model_name, 'optimizer': 'famo',
        'localization': {'source_json': repair['localization']['source_json'],
                         'historical_localization_elapsed_sec': previous_loc,
                         'not_included_in_repair_wall': True},
        'repair_timing_sec': before_times, 'test_timing_sec': timing,
        'combined_historical_localization_repair_test_sec': combined,
        'note': 'TRAIN-selected checkpoint only; TEST never used for optimization or checkpoint selection.',
    })
    print('[Files] {} {}'.format(test_file, directory/'time_summary.json'), flush=True)
    return result


def main():
    run(get_parser().parse_args())


if __name__ == '__main__':
    main()
