"""CGARepair MOEA/D fairness independent FULL TEST (no test-based selection).

Read exactly ../data/fairness/{bank,adult,default}/x_test.h5, y_test.h5,
g_test.h5. Evaluate the baseline model and the SINGLE best_weights.h5 chosen
solely on TRAIN by fairness_moo_2.py. Report group accuracy gap, accuracy, F1.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from cgarepair_fairness_utils import (
    check_dataset, load_fairness_split, model_path_for,
    set_up_tf_graph, evaluate_model, save_json, sha256_file,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def verify_only_selected_weights_modified(before, after, specs, tol=1e-7):
    if len(before) != len(after):
        raise ValueError('Original and repaired model weight layouts disagree')
    permitted = {}
    for spec in specs:
        key = int(spec['kernel_weight_idx'])
        permitted.setdefault(key, []).append(int(spec['channel_idx']))
    changed_count, outside_count = 0, 0
    touched = []
    for idx, (old, new) in enumerate(zip(before, after)):
        if old.shape != new.shape:
            raise ValueError('Model weight {} shape changed'.format(idx))
        modified = np.abs(np.asarray(new, dtype=float) - np.asarray(old, dtype=float)) > tol
        n = int(np.count_nonzero(modified))
        if not n:
            continue
        changed_count += n
        if idx in permitted:
            accepted = np.zeros(old.shape, dtype=bool)
            for channel in permitted[idx]:
                accepted[..., channel] = True
            outside_count += int(np.count_nonzero(modified & (~accepted)))
        else:
            outside_count += n
        touched.append({'weight_array_index': idx, 'changed_parameters': n})
    result = {'changed_parameters': changed_count,
              'changed_outside_CGARepair_selected_neurons': outside_count,
              'modified_weight_arrays': touched,
              'comparison_tolerance': tol,
              'passed': outside_count == 0}
    if outside_count:
        raise RuntimeError('Repaired checkpoint changed {} parameters OUTSIDE '
                           'selected CGARepair kernel channels.'.format(outside_count))
    return result


def parser():
    p = argparse.ArgumentParser(description='Strict FULL TEST for CGARepair fairness MOEA/D')
    p.add_argument('dataset', choices=('bank', 'adult', 'default'))
    p.add_argument('seed', type=int)
    p.add_argument('--optimizer', choices=('moead', 'nsga3'), default='moead')
    p.add_argument('--run-tag', default='')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--skip-model-hash-check', action='store_true')
    return p.parse_args()


def run(args):
    t0 = time.perf_counter()
    ds = check_dataset(args.dataset)
    suffix = ('_' + args.run_tag) if args.run_tag else ('_quick' if args.quick else '')
    if args.quick and args.run_tag and not suffix.endswith('_quick'):
        suffix += '_quick'
    run_dir = Path(args.output_root) / ds / 'CGARepair' / ('seed_' + str(args.seed)) / (args.optimizer + suffix)
    summary_file = run_dir / 'summary.json'
    if not summary_file.is_file():
        raise FileNotFoundError('Repair summary missing: {}. Run fairness_moo_2.py first.'.format(summary_file))
    with open(str(summary_file), 'r', encoding='utf-8') as file:
        summary = json.load(file)
    if (summary.get('optimizer') != args.optimizer or summary.get('dataset') != ds or
            int(summary.get('seed', -1)) != args.seed):
        raise ValueError('Summary optimizer/dataset/seed mismatch; refusing stale checkpoint')
    if summary.get('test_used_for_selection') is not False:
        raise ValueError('Repair summary does not attest train-only checkpoint selection')
    best_weights = run_dir / 'best_weights.h5'
    if not best_weights.is_file():
        raise FileNotFoundError('Predetermined TRAIN-selected checkpoint missing: {}'.format(best_weights))
    model_file = model_path_for(ds, args.model_root)
    if not args.skip_model_hash_check and sha256_file(model_file) != summary.get('model_sha256'):
        raise RuntimeError('Base model SHA256 differs from repair source; wrong model or overwritten file')

    stage = time.perf_counter()
    X, Y, G, manifest = load_fairness_split(ds, 'test', args.data_root,
                                           sample_count=0, sample_seed=args.seed)
    test_data_load_sec = time.perf_counter() - stage
    stage = time.perf_counter()
    tf, K, session = set_up_tf_graph()
    from tensorflow.keras.models import load_model
    model = load_model(str(model_file), compile=False)
    declared = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if tuple(declared[1:]) != tuple(X.shape[1:]):
        raise ValueError('TEST model expected {}, but data shape={}'.format(declared, X.shape))
    base_weights = [v.copy() for v in model.get_weights()]
    original = evaluate_model(model, X, Y, G, session, args.batch_size, include_f1=True)
    model.load_weights(str(best_weights))
    modified_weights = model.get_weights()
    audit = verify_only_selected_weights_modified(base_weights, modified_weights,
                                                  summary['parameter_specs'])
    repaired = evaluate_model(model, X, Y, G, session, args.batch_size, include_f1=True)
    inference_sec = time.perf_counter() - stage

    output = {
        'dataset': ds, 'seed': args.seed, 'method': 'CGARepair', 'optimizer': args.optimizer,
        'evaluation_scope': 'full_test', 'train_data_loaded_by_test_script': False,
        'test_used_to_select_checkpoint': False,
        'selection_scope': 'training_only', 'selection_policy': summary['selection_policy'],
        'test_files': manifest['paths'], 'test_sample_count': len(Y),
        'trained_model': str(model_file), 'repaired_checkpoint': str(best_weights),
        'fairness_metric': 'absolute_gap_of_sensitive_group_classification_accuracies',
        'original': original, 'repaired': repaired,
        'fairness_gap_improvement': float(original['fairness_gap'] - repaired['fairness_gap']),
        'fairness_gap_change': float(repaired['fairness_gap'] - original['fairness_gap']),
        'accuracy_change': float(repaired['accuracy'] - original['accuracy']),
        'f1_change': float(repaired['f1'] - original['f1']),
        'weight_change_audit': audit,
        'timing_sec': {'test_data_load_sec': float(test_data_load_sec),
                       'baseline_and_repaired_test_inference_sec': float(inference_sec),
                       'full_test_wall_sec': float(time.perf_counter() - t0)}}
    save_json(run_dir / 'test_results.json', output)
    save_json(run_dir / 'full_test_results.json', output)
    with open(str(run_dir / 'full_test_results.txt'), 'w', encoding='utf-8') as f:
        for name, metrics in (('original', original), ('repaired', repaired)):
            f.write('{}_fairness_gap={:.10f}\n'.format(name, metrics['fairness_gap']))
            f.write('{}_accuracy={:.10f}\n'.format(name, metrics['accuracy']))
            f.write('{}_f1={:.10f}\n'.format(name, metrics['f1']))
            for group in ('0', '1'):
                f.write('{}_group_{}_accuracy={:.10f}\n'.format(
                    name, group, metrics['group_accuracies'][group]))
        f.write('complete_test_samples={}\n'.format(len(Y)))
        f.write('test_checkpoint_selection=false\n')
        f.write('changed_outside_selected_neurons={}\n'.format(
            audit['changed_outside_CGARepair_selected_neurons']))

    timing_file = run_dir / 'time_summary.json'
    prior = dict(summary['timing_sec'])
    prior['independent_test_wall_sec'] = float(time.perf_counter() - t0)
    prior['test_data_load_sec'] = float(test_data_load_sec)
    prior['test_inference_and_audit_sec'] = float(inference_sec)
    prior['repair_plus_test_wall_sec'] = float(
        prior['repair_run_total_wall_sec'] + prior['independent_test_wall_sec'])
    localization = prior.get('localization_previous_run_sec')
    prior['estimated_locator_plus_repair_plus_test_sec'] = (
        float(localization) + prior['repair_plus_test_wall_sec'] if localization is not None else None)
    prior['previous_localization_not_rerun_during_repair'] = True
    save_json(timing_file, prior)
    print('[FULL TEST] {} CGARepair {} samples={} (no test selection)'.format(ds, args.optimizer.upper(), len(X)))
    print('[Original] Gap={:.6f} Acc={:.6f} F1={:.6f}'.format(
        original['fairness_gap'], original['accuracy'], original['f1']))
    print('[Repaired] Gap={:.6f} Acc={:.6f} F1={:.6f}'.format(
        repaired['fairness_gap'], repaired['accuracy'], repaired['f1']))
    print('[Audit] changed={} outside_selected=0'.format(audit['changed_parameters']))
    print('[Timing] repair={:.2f}s, test={:.2f}s, loc(previous)={}'.format(
        prior['repair_run_total_wall_sec'], prior['independent_test_wall_sec'], localization))
    print('[Files] {} and {}'.format(run_dir / 'test_results.json', timing_file))
    return output


def main():
    run(parser())


if __name__ == '__main__':
    main()
