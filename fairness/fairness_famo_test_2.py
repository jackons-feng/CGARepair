"""Independent full TEST for CGARepair fairness masked FAMO.

TEST never selects the checkpoint; it evaluates exactly best_weights.h5
selected on the training split by fairness_famo_2.py. Reports absolute group
classification accuracy Gap, overall Accuracy, F1 and out-of-mask changes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import numpy as np

from cgarepair_fairness_utils import (
    check_dataset, load_fairness_split, model_path_for, set_up_tf_graph,
    evaluate_model, save_json, sha256_file,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def audit_selected_weights(before, after, specs, tol=1.e-7):
    if len(before) != len(after):
        raise ValueError('Weight tensor count differs')
    permitted = {}
    for s in specs:
        permitted.setdefault(int(s['kernel_weight_idx']), set()).add(int(s['channel_idx']))
    changed, outside, touched = 0, 0, []
    for i, (old, new) in enumerate(zip(before, after)):
        if old.shape != new.shape or not np.all(np.isfinite(new)):
            raise ValueError('Invalid repaired tensor {}'.format(i))
        diff = np.abs(np.asarray(new, dtype=np.float64)-np.asarray(old, dtype=np.float64)) > tol
        count = int(np.count_nonzero(diff))
        if not count:
            continue
        changed += count
        if i in permitted:
            mask = np.zeros(old.shape, dtype=bool)
            for channel in permitted[i]:
                if channel < 0 or channel >= mask.shape[-1]:
                    raise ValueError('Invalid selected channel in tensor {}'.format(i))
                mask[..., channel] = True
            outside += int(np.count_nonzero(diff & ~mask))
        else:
            outside += count
        touched.append({'weight_array_index': i, 'changed_parameters': count})
    if outside:
        raise RuntimeError('FAMO changed {} parameters outside GradAct mask'.format(outside))
    return {'changed_parameters': changed, 'changed_outside_selected_neurons': outside,
            'modified_weight_arrays': touched, 'comparison_tolerance': tol, 'passed': True}


def get_parser():
    p = argparse.ArgumentParser(description='Independent fairness FAMO full TEST')
    p.add_argument('dataset', choices=('bank', 'adult', 'default'))
    p.add_argument('seed', type=int)
    p.add_argument('--optimizer', choices=('famo',), default='famo')
    p.add_argument('--run-tag', default='paper_gamma001_steps5000')
    p.add_argument('--run-dir', help='Explicit directory with TRAIN summary.json and best_weights.h5')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--skip-model-hash-check', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p


def run(args):
    t0 = time.perf_counter()
    ds = check_dataset(args.dataset)
    suffix = ('_'+args.run_tag) if args.run_tag else ('_quick' if args.quick else '')
    if args.quick and args.run_tag and not suffix.endswith('_quick'):
        suffix += '_quick'
    directory = (Path(args.run_dir).expanduser().resolve() if args.run_dir else
                 Path(args.output_root).expanduser()/ds/'CGARepair'/
                 ('seed_'+str(args.seed))/('famo'+suffix))
    summary_path = directory/'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError('Missing FAMO TRAIN summary: {}'.format(summary_path))
    with summary_path.open('r', encoding='utf-8') as f:
        summary = json.load(f)
    if (summary.get('task') != 'fairness' or summary.get('optimizer') != 'famo' or
            summary.get('dataset') != ds or int(summary.get('seed', -1)) != args.seed or
            summary.get('test_used_for_selection') is not False):
        raise ValueError('FAMO TRAIN summary identity / test isolation mismatch')
    weight_file = directory/'best_weights.h5'
    if not weight_file.is_file():
        raise FileNotFoundError('Missing TRAIN-selected best_weights.h5: {}'.format(weight_file))
    result_file = directory/'test_results.json'
    if result_file.exists() and not args.overwrite:
        raise FileExistsError('Existing TEST result; use --overwrite deliberately: {}'.format(result_file))
    path = model_path_for(ds, args.model_root)
    if not args.skip_model_hash_check and sha256_file(path) != summary['model_sha256']:
        raise RuntimeError('Model checkpoint SHA256 differs from TRAIN')
    if args.batch_size <= 0:
        raise ValueError('TEST batch size must be positive')

    stage = time.perf_counter()
    X, Y, G, manifest = load_fairness_split(
        ds, 'test', args.data_root, sample_count=0, sample_seed=args.seed)
    test_data_sec = time.perf_counter()-stage
    stage = time.perf_counter()
    tf, K, session = set_up_tf_graph()
    try:
        K.set_learning_phase(0)
    except AttributeError:
        tf.compat.v1.keras.backend.set_learning_phase(0)
    from tensorflow.keras.models import load_model
    model = load_model(str(path), compile=False)
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if tuple(shape[1:]) != tuple(np.asarray(X).shape[1:]):
        raise ValueError('TEST feature shape differs from model input')
    original_weights = [w.copy() for w in model.get_weights()]
    original = evaluate_model(model, X, Y, G, session, args.batch_size, include_f1=True)
    model.load_weights(str(weight_file))
    repaired_weights = model.get_weights()
    audit = audit_selected_weights(original_weights, repaired_weights, summary['parameter_specs'])
    repaired = evaluate_model(model, X, Y, G, session, args.batch_size, include_f1=True)
    eval_sec = time.perf_counter()-stage
    total_wall = time.perf_counter()-t0
    print('[FULL TEST] {} FAMO seed={} n={} TRAIN-only checkpoint'.format(
        ds, args.seed, len(Y)), flush=True)
    print('[Original] Gap={:.6f} Acc={:.6f} F1={:.6f}'.format(
        original['fairness_gap'], original['accuracy'], original['f1']), flush=True)
    print('[Repaired] Gap={:.6f} Acc={:.6f} F1={:.6f}'.format(
        repaired['fairness_gap'], repaired['accuracy'], repaired['f1']), flush=True)
    print('[Audit] changed={} outside_mask=0'.format(audit['changed_parameters']), flush=True)
    output = {
        'schema': 'CGARepair_fairness_FAMO_masked_weight_test_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'fairness', 'dataset': ds, 'seed': int(args.seed),
        'method': 'CGARepair_FAMO', 'optimizer': 'famo',
        'evaluation_scope': 'full_test', 'train_data_loaded_by_test_script': False,
        'test_used_to_select_checkpoint': False, 'checkpoint_selection': 'TRAIN only',
        'selection_policy': summary['selection_policy'],
        'test_files': manifest['paths'], 'test_sample_count': int(len(Y)),
        'trained_model': str(path), 'repaired_checkpoint': str(weight_file.resolve()),
        'fairness_metric': 'absolute_gap_of_sensitive_group_classification_accuracies',
        'original': original, 'repaired': repaired,
        'fairness_gap_improvement': float(original['fairness_gap']-repaired['fairness_gap']),
        'fairness_gap_change': float(repaired['fairness_gap']-original['fairness_gap']),
        'accuracy_change': float(repaired['accuracy']-original['accuracy']),
        'f1_change': float(repaired['f1']-original['f1']),
        'weight_change_audit': audit,
        'timing_sec': {
            'test_data_load_sec': float(test_data_sec),
            'test_model_eval_and_weight_audit_sec': float(eval_sec),
            'full_test_wall_sec': float(total_wall),
        },
    }
    save_json(result_file, output)
    save_json(directory/'full_test_results.json', output)
    with open(str(directory/'full_test_results.txt'), 'w', encoding='utf-8') as f:
        for label, metrics in (('original', original), ('repaired', repaired)):
            for key in ('fairness_gap', 'accuracy', 'f1'):
                f.write('{}_{key}={:.10f}\n'.format(label, float(metrics[key]), key=key))
            for group in ('0', '1'):
                f.write('{}_group_{}_accuracy={:.10f}\n'.format(
                    label, group, metrics['group_accuracies'][group]))
        f.write('complete_test_samples={}\n'.format(len(Y)))
        f.write('test_checkpoint_selection=false\n')
        f.write('changed_outside_selected_neurons=0\n')
    timing = dict(summary['timing_sec'])
    timing['independent_test_wall_sec'] = float(total_wall)
    timing['test_data_load_sec'] = float(test_data_sec)
    timing['test_inference_and_audit_sec'] = float(eval_sec)
    timing['repair_plus_test_wall_sec'] = float(timing['repair_run_total_wall_sec']+total_wall)
    loc = timing.get('localization_previous_run_sec')
    timing['historical_locator_plus_repair_plus_test_sec'] = (
        float(loc)+timing['repair_plus_test_wall_sec'] if loc is not None else None)
    timing['notes'] = 'TEST never selected checkpoint; times measured across separate processes.'
    save_json(directory/'time_summary.json', timing)
    print('[Files] {} {}'.format(result_file, directory/'time_summary.json'), flush=True)
    return output


def main():
    run(get_parser().parse_args())


if __name__ == '__main__':
    main()
