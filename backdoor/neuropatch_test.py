#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent hold-out TEST for neuropatch_backdoor.py.

Uses the same test data, Neural-Cleanse trigger, legacy SR, conditional ASR,
and optional evasion protocol as ccbr_test.py. These scripts must be installed
in the same directory. Neuropatch itself saves a full Keras weights_final.h5
(not the CCBR activation-scaling *.npy vectors).

Example:
  python -u neuropatch_test.py fashion 256 --epsilon 0.1 --evasion
  python -u neuropatch_test.py gtsrb_resnet 256 --epsilon 1.0 --evasion \
      --test-file ../data/gtsrb/test_data.pkl

No test data is used for neuron selection, action selection or checkpoint
selection. The weights_final.h5 was chosen by the repair-stage procedure.
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import ccbr_test as shared


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Independent TEST for neuropatch_backdoor.py')
    p.add_argument('model_name', help='fashion / gtsrb_resnet / gtsrb / mnist')
    p.add_argument('seed', type=int)
    p.add_argument('legacy_resnet', nargs='?', default='', help='Optional positional True (GTSRB only)')
    p.add_argument('--resnet', action='store_true')
    p.add_argument('--epsilon', default=None, help='Only a directory-name tag, matching repair run: fashion=0.1, gtsrb=1.0')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-dir', help='Explicit folder containing Neuropatch weights_final.h5 and run_stats.json')
    p.add_argument('--model-root', default='../model')
    p.add_argument('--model-path', help='Explicit original poisoned model, must match repair model')
    p.add_argument('--data-root', default='../data')
    p.add_argument('--test-file', help='Independent GTSRB TEST file, never train_data.pkl')
    p.add_argument('--img-dir', help='Neural-Cleanse trigger PNG directory (default: <data-root>/img)')
    p.add_argument('--test-samples', type=int, default=0, help='0=full hold-out TEST')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--evasion', action='store_true')
    p.add_argument('--evasion-samples', type=int, default=200)
    p.add_argument('--out-dir', help='Override test output directory')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    if args.test_samples < 0 or args.batch_size <= 0 or args.evasion_samples <= 0:
        p.error('test-samples >= 0, batch-size > 0 and evasion-samples > 0 are required')
    return args


def get_identity(args):
    m, tag = shared.model_identity(args)
    eps = str(args.epsilon) if args.epsilon is not None else ('1.0' if m == 'gtsrb' else '0.1')
    return m, tag, eps


def predict_labels(sess, model, images, batch_size):
    out = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        predictions = sess.run(model.output, feed_dict={model.input: batch})
        out.append(np.argmax(predictions, axis=-1).astype(np.int64).reshape(-1))
    if not out:
        raise ValueError('No samples to evaluate')
    return np.concatenate(out, axis=0)


def measure_pair(sess, model, clean, triggered, labels, target, batch_size):
    clean_pred = predict_labels(sess, model, clean, batch_size)
    triggered_pred = predict_labels(sess, model, triggered, batch_size)
    return shared.score_preds(clean_pred, triggered_pred, labels, target)


def measure_trigger(sess, model, triggered, labels, target, batch_size):
    p = predict_labels(sess, model, triggered, batch_size)
    eligible = labels != target
    cnt = int(np.sum((p == target) & eligible))
    return {
        'sr': float(cnt / len(labels)),
        'standard_asr': float(cnt / int(np.sum(eligible))) if np.any(eligible) else None,
        'successes': cnt,
        'non_target_count': int(np.sum(eligible)),
        'total_count': int(len(labels)),
    }


def run(args):
    t_start = time.perf_counter()
    model_name, tag, epsilon = get_identity(args)
    run_dir = (Path(args.run_dir) if args.run_dir else
               Path(args.output_root) / tag / str(args.seed) /
               ('singledefect_neuropatch_backdoor_bdonly_neuron_eps' + epsilon)).expanduser().resolve()
    weights = run_dir / 'weights_final.h5'
    stats_path = run_dir / 'run_stats.json'
    actions_path = run_dir / 'selected_actions.json'
    for p in (weights, stats_path, actions_path):
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError('Incomplete Neuropatch TRAIN repair run, missing: %s' % p)
    with stats_path.open('r', encoding='utf-8') as f:
        stats = json.load(f)
    with actions_path.open('r', encoding='utf-8') as f:
        selected_actions = json.load(f)
    if not isinstance(selected_actions, list):
        raise ValueError('Neuropatch selected_actions.json must be a list')
    if stats.get('method') != 'neuropatch' or stats.get('repair_task') != 'backdoor':
        raise ValueError('run_stats.json is not a Neuropatch backdoor repair')
    if stats.get('name_tag') != tag or int(stats.get('seed', -1)) != int(args.seed):
        raise ValueError('Neuropatch repair tag/seed do not match TEST arguments')
    if str(stats.get('epsilon')) != epsilon:
        raise ValueError('Neuropatch epsilon directory tag differs from repair metadata')
    if int(stats.get('selected_count', -1)) != len(selected_actions):
        raise ValueError('Neuropatch run_stats selected_count disagrees with selected_actions.json')

    outdir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / 'neuropatch_test_results.json',
             outdir / 'neuropatch_test_metrics.csv',
             outdir / 'neuropatch_time_summary.json']
    if any(p.exists() for p in paths) and not args.overwrite:
        raise FileExistsError('TEST outputs exist. Use --overwrite or let the batch runner skip completed TEST.')

    t = time.perf_counter()
    tf, sess = shared.setup_tf(args.seed)
    original_path = Path(args.model_path or (Path(args.model_root) / (tag + '.h5'))).expanduser().resolve()
    if not original_path.is_file():
        raise FileNotFoundError('Original model not found: %s' % original_path)
    repair_model_path = Path(stats.get('model_path', '')).expanduser().resolve()
    if repair_model_path != original_path:
        raise ValueError('Original model mismatches the model used by Neuropatch repair: %s vs %s' %
                         (original_path, repair_model_path))
    model = tf.keras.models.load_model(str(original_path), compile=False)
    original_weights = [w.copy() for w in model.get_weights()]
    t_load_model = time.perf_counter() - t

    t = time.perf_counter()
    clean, labels, test_source = shared.load_test_data(model_name, args, tf, sess)
    full_count = len(labels)
    if args.test_samples and args.test_samples < full_count:
        rng = np.random.RandomState(int(args.seed) + 424242)
        idx = rng.choice(full_count, args.test_samples, replace=False)
        clean, labels = clean[idx], labels[idx]
    expected_shape = tuple(shared.DATASET_CONFIG[model_name]['input_shape'])
    if tuple(clean.shape[1:]) != expected_shape:
        raise ValueError('TEST input shape mismatch: %s vs %s' % (clean.shape[1:], expected_shape))
    triggered, trigger_meta = shared.trigger_images(model_name, clean, args, tf)
    t_data = time.perf_counter() - t

    t = time.perf_counter()
    target = int(stats['target_label'])
    expected_target = shared.DATASET_CONFIG[model_name]['target_label']
    if target != expected_target:
        raise ValueError('TRAIN target label %d differs from TEST trigger target %d' % (target, expected_target))
    original = measure_pair(sess, model, clean, triggered, labels, target, args.batch_size)
    model.load_weights(str(weights))  # fixed before seeing any TEST result
    repaired = measure_pair(sess, model, clean, triggered, labels, target, args.batch_size)
    t_eval = time.perf_counter() - t
    print('[NEUROPATCH TEST] tag=%s seed=%d TEST=%d/%d' % (tag, args.seed, len(labels), full_count))
    print('[TRAIN CHECKPOINT] %s | selected_actions=%d' % (weights, len(selected_actions)))
    print('[BEFORE] ACC=%.4f%% SR=%.4f%% standard ASR=%s' % (
        100*original['clean_accuracy'], 100*original['sr'],
        ('%.4f%%' % (100*original['standard_asr'])) if original['standard_asr'] is not None else 'N/A'))
    print('[AFTER ] ACC=%.4f%% SR=%.4f%% standard ASR=%s' % (
        100*repaired['clean_accuracy'], 100*repaired['sr'],
        ('%.4f%%' % (100*repaired['standard_asr'])) if repaired['standard_asr'] is not None else 'N/A'))
    if not selected_actions:
        print('[NO-OP] Neuropatch accepted zero actions; the unchanged model is a valid outcome')

    t = time.perf_counter()
    evasion = None
    if args.evasion:
        evasion = {}
        for name, (images, test_y) in shared.evasion_cases(
                tf, sess, triggered, labels, model_name, args.evasion_samples, args.seed).items():
            model.set_weights(original_weights)
            before = measure_trigger(sess, model, images, test_y, target, args.batch_size)
            model.load_weights(str(weights))
            after = measure_trigger(sess, model, images, test_y, target, args.batch_size)
            evasion[name] = {'original': before, 'train_selected_repaired': after}
    t_evasion = time.perf_counter() - t
    total = time.perf_counter() - t_start
    timings = {'original_model_loading_sec': t_load_model,
               'test_data_trigger_loading_sec': t_data,
               'original_vs_repaired_eval_sec': t_eval,
               'test_evasion_sec': t_evasion,
               'test_total_wall_sec': total,
               'prior_repair_elapsed_sec': stats.get('elapsed_sec')}
    result = {
        'schema': 'Neuropatch_backdoor_holdout_test_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': 'Neuropatch', 'model': tag, 'seed': int(args.seed),
        'epsilon_directory_tag': epsilon, 'test_source': test_source,
        'test_count': int(len(labels)), 'test_full_count': int(full_count),
        'test_is_subset': bool(len(labels) != full_count),
        'trigger': trigger_meta,
        'original_model_file': str(original_path),
        'train_selected_checkpoint': str(weights),
        'train_checkpoint_sha256': shared.sha256_file(weights),
        'checkpoint_selection': 'TRAIN greedy action-selection only; fixed weights_final.h5',
        'selection_uses_TEST': False,
        'selected_action_count': int(len(selected_actions)),
        'selected_actions_file': str(actions_path),
        'repair_stats_file': str(stats_path),
        'sr_definition': 'triggered target predictions from non-target labels / ALL labels',
        'original_test_metrics': original, 'repaired_test_metrics': repaired,
        'delta_test_accuracy': float(repaired['clean_accuracy']-original['clean_accuracy']),
        'delta_test_sr': float(repaired['sr']-original['sr']),
        'evasion_results': evasion,
        'repair_timing_sec': stats.get('elapsed_sec'), 'timing_sec': timings,
    }
    shared.save_json(paths[0], result)
    with paths[1].open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model','seed','test_count','selected_actions','original_acc','repaired_acc',
                    'original_legacy_sr','repaired_legacy_sr','original_standard_asr',
                    'repaired_standard_asr','delta_acc','delta_legacy_sr','test_total_sec'])
        w.writerow([tag,args.seed,len(labels),len(selected_actions),
                    original['clean_accuracy'],repaired['clean_accuracy'],
                    original['sr'],repaired['sr'],original['standard_asr'],repaired['standard_asr'],
                    repaired['clean_accuracy']-original['clean_accuracy'],
                    repaired['sr']-original['sr'],total])
    shared.save_json(paths[2], {'method':'Neuropatch','model':tag,'seed':int(args.seed),
                                'timing_sec':timings,
                                'note':'Prior repair time and this hold-out TEST time are from separate runs.'})
    print('[FILES] %s / %s / %s' % tuple(map(str, paths)))
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
