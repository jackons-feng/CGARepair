"""Hold-out adversarial TEST for CGARepair/FAMO.

Load the one weight checkpoint picked on TRAIN. Never select using TEST data.
This is local kernel-weight repair, not activation scaling.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

from read_CGARepair import epsilon_string, model_name_checked
from cgarepair_adv_utils import (initialize_tf, load_adv_and_clean,
                                 model_file, calculate_metrics, save_json)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def run_dir_for(root, model, eps, seed, optimizer, run_tag='', quick=False):
    suffix = ('_'+run_tag) if run_tag else ('_quick' if quick else '')
    return (Path(root).expanduser() / 'adv' / model / ('epsilon_'+eps) /
            'CGARepair' / ('seed_'+str(seed)) / (optimizer+suffix))


def parse_args():
    p = argparse.ArgumentParser(description='TEST train-selected CGARepair ADV repair checkpoints')
    p.add_argument('model_name', choices=('mnist', 'cifar10'))
    p.add_argument('epsilon')
    p.add_argument('seed', type=int)
    p.add_argument('--optimizer', '--algorithm', choices=('famo',), default='famo')
    p.add_argument('--run-tag', default='')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--run-dir', help='Explicit repair output directory, containing summary.json')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--test-samples', type=int, default=0, help='0 uses full independent TEST split')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--evaluate-front', action='store_true',
                   help='Optional diagnostic; evaluates archived candidates but never changes best checkpoint')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def run(args):
    start = time.perf_counter()
    model = model_name_checked(args.model_name)
    eps = epsilon_string(args.epsilon)
    directory = (Path(args.run_dir).expanduser().resolve() if args.run_dir else
                 run_dir_for(args.output_root, model, eps, args.seed,
                             args.optimizer, args.run_tag, args.quick))
    summary_file = directory/'summary.json'
    if not summary_file.is_file():
        raise FileNotFoundError('Repair summary not found: {}. Run adv_famo_2.py first.'.format(summary_file))
    with open(str(summary_file), 'r', encoding='utf-8') as f:
        repair = json.load(f)
    if (repair.get('task') != 'adv' or repair.get('model') != model or
            epsilon_string(repair.get('epsilon')) != eps or
            int(repair.get('optimization_seed', -1)) != args.seed or
            repair.get('optimizer') != args.optimizer):
        raise ValueError('Model/epsilon/optimizer/seed do not match repair summary')
    result_path = directory/'test_results.json'
    if result_path.exists() and not args.overwrite:
        raise FileExistsError('Existing {}. Add --overwrite if deliberate.'.format(result_path))

    t = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    from tensorflow.keras.models import load_model
    path = model_file(model, args.model_root)
    pretrained = load_model(path, compile=False)
    saved = directory/'best_weights.h5'
    if not saved.is_file():
        saved = Path(repair.get('best_weights_file', ''))
    if not saved.is_file():
        raise FileNotFoundError('Train-selected repaired weights missing: {}'.format(saved))
    model_load_sec = time.perf_counter()-t

    t = time.perf_counter()
    adv_x, adv_y, clean_x, clean_y, test_data = load_adv_and_clean(
        model, eps, split='test', data_root=args.data_root,
        samples=args.test_samples, seed=args.seed)
    test_data_sec = time.perf_counter()-t

    t = time.perf_counter()
    original_metrics = calculate_metrics(pretrained, adv_x, adv_y,
                                         clean_x, clean_y, session, args.batch_size)
    pretrained.load_weights(str(saved))
    repaired_metrics = calculate_metrics(pretrained, adv_x, adv_y,
                                         clean_x, clean_y, session, args.batch_size)
    selected_eval_sec = time.perf_counter()-t
    print('[TEST] {} eps={} optimizer={} seed={} adv={} clean={}'.format(
        model, eps, args.optimizer, args.seed, len(adv_y), len(clean_y)), flush=True)
    print('[TEST] BEFORE adv_error={:.6f} robustness={:.6f} clean_acc={:.6f}'.format(
        original_metrics['adv_error_rate'], original_metrics['adv_robustness'],
        original_metrics['clean_accuracy']), flush=True)
    print('[TEST] AFTER  adv_error={:.6f} robustness={:.6f} clean_acc={:.6f}'.format(
        repaired_metrics['adv_error_rate'], repaired_metrics['adv_robustness'],
        repaired_metrics['clean_accuracy']), flush=True)

    front_t = time.perf_counter()
    front_results = []
    if args.evaluate_front:
        for file in sorted((directory/'pareto_front').glob('weights_*.h5')):
            pretrained.load_weights(str(file))
            met = calculate_metrics(pretrained, adv_x, adv_y, clean_x, clean_y,
                                    session, args.batch_size)
            front_results.append({'checkpoint': str(file.resolve()), 'metrics': met})
        if front_results:
            with open(str(directory/'pareto_test_results.csv'), 'w',
                      newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['checkpoint', 'test_adv_error', 'test_adv_robustness',
                                 'test_clean_acc'])
                for item in front_results:
                    met = item['metrics']
                    writer.writerow([item['checkpoint'], met['adv_error_rate'],
                                     met['adv_robustness'], met['clean_accuracy']])
        pretrained.load_weights(str(saved))
    front_eval_sec = time.perf_counter()-front_t

    test_total = float(time.perf_counter()-start)
    previous_timing = repair.get('timing_sec', {})
    previous_localization = previous_timing.get('localization_previous_run_sec')
    previous_repair = previous_timing.get('total_repair_run_wall_sec')
    combined = (float(previous_localization)+float(previous_repair)+test_total
                if previous_localization is not None and previous_repair is not None else None)
    timing = {
        'localization_previous_run_sec': previous_localization,
        'repair_previous_run_sec': previous_repair,
        'test_model_load_sec': float(model_load_sec),
        'test_data_load_sec': float(test_data_sec),
        'test_selected_checkpoint_eval_sec': float(selected_eval_sec),
        'test_optional_front_eval_sec': float(front_eval_sec),
        'test_total_wall_sec': float(test_total),
        'combined_historical_localization_repair_test_sec': combined,
        'notes': 'Test did not select best checkpoint. Historical times from separate runs; '
                 'combined value is a sum, not one continuous clock measurement.'
    }
    result = {
        'schema': 'CGARepair_adv_weight_test_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'adv', 'model': model, 'epsilon': eps,
        'optimizer': args.optimizer, 'method': repair['method'],
        'optimization_seed': int(args.seed),
        'selected_checkpoint': str(saved.resolve()),
        'checkpoint_selection': 'TRAIN only; never reselected using TEST',
        'checkpoint_selection_policy': repair['selection_policy'],
        'selected_players': repair['selected_players'],
        'top_k': int(repair['top_k']),
        'test_data': test_data,
        'original_test_metrics': original_metrics,
        'repaired_test_metrics': repaired_metrics,
        'delta_adv_error_rate': float(repaired_metrics['adv_error_rate']-
                                       original_metrics['adv_error_rate']),
        'delta_adv_robustness': float(repaired_metrics['adv_robustness']-
                                      original_metrics['adv_robustness']),
        'delta_clean_accuracy': float(repaired_metrics['clean_accuracy']-
                                       original_metrics['clean_accuracy']),
        'optional_front_test_results': front_results,
        'repair_timing_sec': previous_timing,
        'timing_sec': timing,
        'total_time_sec': test_total,
    }
    save_json(result_path, result)
    save_json(directory/'time_summary.json', {
        'schema': 'CGARepair_adv_full_timing_v1',
        'model': model, 'epsilon': eps, 'optimizer': args.optimizer,
        'localization': {'source_json': repair['localization']['source_json'],
                         'historical_localization_elapsed_sec': previous_localization,
                         'not_included_in_repair_wall': True},
        'repair_timing_sec': previous_timing,
        'test_timing_sec': timing,
        'combined_historical_localization_repair_test_sec': combined,
        'note': 'Fitness evaluation wall is already included inside optimization wall; '
                'do not add the two. TEST was never used for checkpoint choice.'
    })
    print('[Timing] previous_loc={} previous_repair={} TEST={:.2f}s'.format(
        previous_localization, previous_repair, test_total), flush=True)
    print('[Files] {} {}'.format(result_path, directory/'time_summary.json'), flush=True)
    return result


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
