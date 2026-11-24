"""Independent TEST evaluation for CGARepair MOEA/D or MOPSO weight repair.

Reads trained model weights (*.h5), NOT activation-scaling multipliers.
The best checkpoint is selected from TRAIN objective values and is never
reselected based on TEST results. JSON includes test timing and an experiment
wide time summary with provenance of prior localization timing.

Usage:
  python backdoor_test_2.py mnist 42 --optimizer moead --quick
  python backdoor_test_2.py gtsrb 42 --resnet --optimizer mopso
  python backdoor_test_2.py fashion 42 --optimizer moead --evasion --evaluate-front
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from read_CGARepair import resolve_model_name, model_tag
from cgarepair_backdoor_utils import (DATASET_CONFIG, initialize_tf, load_dataset,
    model_file, trigger_inputs, select_subset, calculate_metrics, predict_labels, save_json)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def run_dir_path(root, tag, seed, optimizer, run_tag='', quick=False):
    suffix = ('_' + run_tag) if run_tag else ('_quick' if quick else '')
    return Path(root).expanduser() / tag / 'CGARepair' / ('seed_' + str(seed)) / (optimizer + suffix)


def evasion_cases(triggered, labels, model_name, session, max_n, seed):
    """Retain four evasion transformations from original test, but on bounded samples.

    Noise scale follows input scaling (MNIST in [0,1], GTSRB in [0,255]).
    Returns name -> (X_perturbed, matching true labels).
    """
    from scipy.ndimage import rotate
    import tensorflow as tf
    rng = np.random.RandomState(int(seed)+1000)
    n = min(len(triggered), max_n)
    if n < 1:
        raise ValueError('No samples for robustness evaluation')
    indices = rng.choice(len(triggered), n, replace=False)
    base = triggered[indices]
    y = np.asarray(labels)[indices]
    h, w, channels = DATASET_CONFIG[model_name]['input_shape']
    max_val = 255.0 if model_name == 'gtsrb' else 1.0
    cases = {}
    # Original rotation angles -15, -10, 10, 15 degrees.
    rotations = []
    for angle in (-15, -10, 10, 15):
        rot = np.stack([rotate(img, angle, axes=(0, 1), reshape=False,
                               order=1, mode='nearest') for img in base])
        rotations.append(np.clip(rot, 0, max_val).astype(np.float32))
    cases['rotation'] = (np.concatenate(rotations), np.tile(y, 4))
    compressions = []
    for ratio in (0.5, 0.7):
        new_size = (max(1, int(h * ratio)), max(1, int(w * ratio)))
        smaller = session.run(tf.image.resize(base.astype('float32'), new_size))
        restored = session.run(tf.image.resize(smaller, (h, w)))
        compressions.append(np.clip(restored, 0, max_val).astype(np.float32))
    cases['compression'] = (np.concatenate(compressions), np.tile(y, 2))
    noises = []
    for sigma in ((10.0, 20.0) if model_name == 'gtsrb' else (0.04, 0.08)):
        noisy = np.clip(base + rng.normal(0, sigma, size=base.shape), 0, max_val)
        noises.append(noisy.astype(np.float32))
    cases['noise'] = (np.concatenate(noises), np.tile(y, 2))
    brightness = [np.clip(base * fac, 0, max_val).astype(np.float32) for fac in (0.7, 1.3)]
    cases['brightness'] = (np.concatenate(brightness), np.tile(y, 2))
    return cases


def eval_triggered_only(model, triggered, labels, target, session, sr_definition, batch_size):
    preds = predict_labels(model, triggered, session, batch_size)
    mask = labels != target
    successes = int(np.sum((preds == target) & mask))
    denom = int(mask.sum()) if sr_definition == 'standard' else len(labels)
    return {'sr': float(successes / max(denom, 1)), 'successes': successes,
            'denominator': denom, 'n_examples': int(len(labels))}


def parse_args():
    p = argparse.ArgumentParser(description='TEST saved CGARepair/MOEAD/MOPSO weight models')
    p.add_argument('model_name', help='mnist, fashion, gtsrb, gtsrb_resnet')
    p.add_argument('seed', type=int)
    p.add_argument('legacy_resnet', nargs='?', default=None)
    p.add_argument('--resnet', action='store_true')
    p.add_argument('--optimizer', '--algorithm', choices=('moead', 'nsga3', 'mopso'), default='moead')
    p.add_argument('--run-tag', default='')
    p.add_argument('--quick', action='store_true', help='Read optimizer_quick run directory')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-dir', help='Explicit repair run directory (containing summary.json)')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--test-samples', type=int, default=0,
                   help='0 uses full hold-out test split; use a small number for smoke testing')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--evasion', action='store_true', help='Enable original four robustness transforms')
    p.add_argument('--evasion-samples', type=int, default=200)
    p.add_argument('--evaluate-front', action='store_true',
                   help='Evaluate saved Pareto .h5 files, but NEVER select using test scores')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def run(args):
    start = time.perf_counter()
    resnet = args.resnet or str(args.legacy_resnet).lower() in ('true', '1', 'yes', 'y')
    model_name, resnet = resolve_model_name(args.model_name, resnet)
    tag = model_tag(model_name, resnet)
    run_dir = (Path(args.run_dir) if args.run_dir else
               run_dir_path(args.output_root, tag, args.seed, args.optimizer, args.run_tag, args.quick))
    summary_path = run_dir / 'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError('Repair summary missing: {}. Run backdoor_moo_2.py first.'.format(summary_path))
    with open(str(summary_path), 'r', encoding='utf-8') as f:
        repair = json.load(f)
    if repair.get('model_tag') != tag or int(repair.get('optimization_seed', -1)) != int(args.seed):
        raise ValueError('Test args do not match repair summary: {} vs {} / seed {}'.format(
            repair.get('model_tag'), tag, args.seed))
    if repair.get('optimizer') != args.optimizer:
        raise ValueError('Repair optimizer mismatch')
    target = DATASET_CONFIG[model_name]['target_label']
    test_json = run_dir / 'test_results.json'
    if test_json.exists() and not args.overwrite:
        raise FileExistsError('Test result exists: {}. Use --overwrite to update it.'.format(test_json))

    t = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    from tensorflow.keras.models import load_model
    actual_model_path = model_file(model_name, resnet, args.model_root)
    model = load_model(actual_model_path, compile=False)
    pretrained_weights = [w.copy() for w in model.get_weights()]
    best_file = run_dir / 'best_weights.h5'
    if not best_file.is_file():
        best_file = Path(repair.get('best_weights_file', ''))
    if not best_file.is_file():
        raise FileNotFoundError('Best train-selected checkpoint not found; cannot test: {}'.format(best_file))
    model_load_sec = time.perf_counter()-t

    t = time.perf_counter()
    clean, labels = load_dataset(model_name, 'test', args.data_root, session)
    if args.test_samples > 0:
        rng = np.random.RandomState(int(args.seed)+424242)
        idx = rng.choice(len(clean), size=min(args.test_samples, len(clean)), replace=False)
        clean, labels = clean[idx], labels[idx]
    poisoned = trigger_inputs(clean, model_name, args.data_root)
    data_load_sec = time.perf_counter()-t

    t = time.perf_counter()
    sr_definition = repair['sr_definition']
    original = calculate_metrics(model, clean, labels, poisoned, target, session,
                                 sr_definition, args.batch_size)
    model.load_weights(str(best_file))
    fixed_selected = calculate_metrics(model, clean, labels, poisoned, target, session,
                                       sr_definition, args.batch_size)
    baseline_selected_eval_sec = time.perf_counter()-t
    print('[TEST] {} / {} / optimizer={} / seed={}'.format(tag, len(clean), args.optimizer, args.seed))
    print('[TEST] original SR={:.6f} Acc={:.6f}'.format(original['sr'], original['clean_accuracy']))
    print('[TEST] train-selected SR={:.6f} Acc={:.6f}'.format(
        fixed_selected['sr'], fixed_selected['clean_accuracy']))

    front_results = []
    t = time.perf_counter()
    if args.evaluate_front:
        for p in sorted((run_dir / 'pareto_front').glob('weights_*.h5')):
            model.load_weights(str(p))
            m = calculate_metrics(model, clean, labels, poisoned, target, session,
                                  sr_definition, args.batch_size)
            front_results.append({'checkpoint': str(p), 'test_metrics': m})
        with open(str(run_dir / 'pareto_test_results.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['checkpoint', 'test_sr', 'test_accuracy', 'standard_asr'])
            for result in front_results:
                r = result['test_metrics']
                writer.writerow([result['checkpoint'], r['sr'], r['clean_accuracy'], r['standard_asr']])
        model.load_weights(str(best_file))
    pareto_eval_sec = time.perf_counter()-t

    t = time.perf_counter()
    evasion = None
    if args.evasion:
        cases = evasion_cases(poisoned, labels, model_name, session,
                              args.evasion_samples, args.seed)
        evasion = {}
        for kind, (images, targets) in cases.items():
            model.set_weights(pretrained_weights)
            before = eval_triggered_only(model, images, targets, target, session,
                                         sr_definition, args.batch_size)
            model.load_weights(str(best_file))
            after = eval_triggered_only(model, images, targets, target, session,
                                        sr_definition, args.batch_size)
            evasion[kind] = {'original': before, 'train_selected_repaired': after}
        model.load_weights(str(best_file))
    evasion_sec = time.perf_counter()-t

    test_total = float(time.perf_counter()-start)
    repair_timing = repair.get('timing_sec', {})
    prior_loc = repair_timing.get('localization_previous_run_sec')
    prior_repair = repair_timing.get('total_repair_run_wall_sec')
    timing = {
        'localization_previous_run_sec': prior_loc,
        'repair_previous_run_sec': prior_repair,
        'test_model_load_sec': float(model_load_sec),
        'test_data_and_trigger_load_sec': float(data_load_sec),
        'test_original_and_train_selected_eval_sec': float(baseline_selected_eval_sec),
        'test_saved_pareto_eval_sec': float(pareto_eval_sec),
        'test_evasion_eval_sec': float(evasion_sec),
        'test_total_wall_sec': test_total,
        'combined_previous_localization_plus_repair_plus_test_sec': (
            float(prior_loc)+float(prior_repair)+test_total
            if prior_loc is not None and prior_repair is not None else None),
        'notes': 'Previous localization and repair times are from earlier separate runs; '
                 'the combined field is a sum, NOT one uninterrupted wall-clock run. '
                 'No test data were used to choose the checkpoint.',
    }
    result = {
        'schema': 'CGARepair_backdoor_weight_test_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method': repair['method'],
        'model': tag,
        'optimizer': args.optimizer,
        'seed': int(args.seed),
        'localization': repair['localization'],
        'selected_players': repair['selected_players'],
        'top_k': int(repair['top_k']),
        'checkpoint_selection': 'TRAIN only',
        'checkpoint_selection_policy': repair['selection_policy'],
        'selected_checkpoint': str(best_file.resolve()),
        'test_sample_count': int(len(clean)),
        'sr_definition': sr_definition,
        'original_test_metrics': original,
        'repaired_test_metrics': fixed_selected,
        'delta_test_sr': float(fixed_selected['sr']-original['sr']),
        'delta_test_accuracy': float(fixed_selected['clean_accuracy']-original['clean_accuracy']),
        'front_test_results': front_results,
        'evasion_results': evasion,
        'repair_timing_sec': repair_timing,
        'timing_sec': timing,
        'total_time_sec': test_total,
    }
    save_json(test_json, result)
    save_json(run_dir / 'time_summary.json', {
        'schema': 'CGARepair_full_timing_summary_v1',
        'method': repair['method'], 'model': tag, 'optimizer': args.optimizer,
        'localization': {
            'source_json': repair['localization']['source_json'],
            'prior_elapsed_sec': prior_loc,
            'not_included_in_repair_run': True,
        },
        'repair_timing_sec': repair_timing,
        'test_timing_sec': timing,
        'combined_prior_localization_plus_repair_plus_test_sec': (
            timing['combined_previous_localization_plus_repair_plus_test_sec']),
        'important_note': 'Test selection is not used. Localization runtime is historical. '
                          'Evaluation time is already included in optimization wall time.',
    })
    print('[Timing] localization_previous={}s, repair_previous={}s, test={:.2f}s'.format(
        prior_loc, prior_repair, test_total))
    print('[Files] {}, {}'.format(test_json, run_dir / 'time_summary.json'))
    return result


def main():
    a = parse_args()
    run(a)


if __name__ == '__main__':
    main()
