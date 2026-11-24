"""Independent TEST evaluation of train-selected CGARepair safety repair.

Reads the best_weights.h5 checkpoint selected on TRAIN by MOEA/D or MOPSO.
Never chooses a new best model from the TEST split.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

from read_CGARepair import model_name_checked
from cgarepair_safety_utils import (initialize_tf, load_safety_data,
                                    model_file, calculate_metrics, save_json)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def run_directory(root, model, seed, optimizer, run_tag='', quick=False):
    suffix = ('_' + run_tag) if run_tag else ('_quick' if quick else '')
    return (Path(root).expanduser() / model / 'CGARepair' /
            ('seed_' + str(seed)) / (optimizer + suffix))


def get_parser():
    p = argparse.ArgumentParser(description='TEST train-selected CGARepair safety repair')
    p.add_argument('model_name', choices=('a19', 'a29', 'a33'))
    p.add_argument('seed', type=int)
    p.add_argument('--optimizer', '--algorithm', choices=('moead', 'nsga3', 'mopso'), default='moead')
    p.add_argument('--run-tag', default='')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--run-dir', help='Explicit directory containing repair summary.json')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--test-samples', type=int, default=0, help='0 uses full independent TEST pools')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--evaluate-front', action='store_true',
                   help='Optional diagnostic ONLY; cannot reselect checkpoint on TEST')
    p.add_argument('--overwrite', action='store_true')
    return p


def run(args):
    wall_start = time.perf_counter()
    model = model_name_checked(args.model_name)
    directory = (Path(args.run_dir).expanduser().resolve() if args.run_dir else
                 run_directory(args.output_root, model, args.seed,
                               args.optimizer, args.run_tag, args.quick))
    summary_file = directory/'summary.json'
    if not summary_file.is_file():
        raise FileNotFoundError('Missing {}. Run safety_moo_2.py first.'.format(summary_file))
    with open(str(summary_file), 'r', encoding='utf-8') as file:
        repair = json.load(file)
    if (repair.get('task') != 'safety' or repair.get('model') != model or
            int(repair.get('optimization_seed', -1)) != args.seed or
            repair.get('optimizer') != args.optimizer):
        raise ValueError('Safety model, optimizer or seed disagree with the repair summary')
    result_file = directory/'test_results.json'
    if result_file.exists() and not args.overwrite:
        raise FileExistsError('{} already exists; supply --overwrite or a new --run-tag.'.format(
            result_file))
    if args.test_samples < 0:
        raise ValueError('test-samples must be >=0')

    begin = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    from tensorflow.keras.models import load_model
    path = model_file(model, args.model_root)
    if os.stat(path).st_size != os.stat(repair['model_file']).st_size:
        raise ValueError('TEST pretrained model file size differs from training pretrained file')
    net = load_model(path, compile=False)
    selected = directory/'best_weights.h5'
    if not selected.is_file():
        selected = Path(repair.get('best_weights_file', ''))
    if not selected.is_file():
        raise FileNotFoundError('Train-selected best_weights.h5 missing: {}'.format(selected))
    model_load_sec = time.perf_counter()-begin

    begin = time.perf_counter()
    ct, dr, data_info = load_safety_data(model, split='test', data_root=args.data_root,
                                          samples=args.test_samples, seed=args.seed)
    test_data_sec = time.perf_counter()-begin

    begin = time.perf_counter()
    original = calculate_metrics(net, model, ct, dr, session, args.batch_size)
    net.load_weights(str(selected))
    repaired = calculate_metrics(net, model, ct, dr, session, args.batch_size)
    selected_eval_sec = time.perf_counter()-begin
    print('[TEST] {} {} opt={} seed={} counterexamples={} drawdown={}'.format(
        model, 'safety', args.optimizer, args.seed, len(ct), len(dr)), flush=True)
    print('[TEST] BEFORE safety={:.6f} retention={:.6f}'.format(
        original['safety_violation_rate'], original['drawdown_retention']), flush=True)
    print('[TEST] AFTER  safety={:.6f} retention={:.6f}'.format(
        repaired['safety_violation_rate'], repaired['drawdown_retention']), flush=True)

    front_begin = time.perf_counter()
    front_records = []
    if args.evaluate_front:
        for candidate in sorted((directory/'pareto_front').glob('weights_*.h5')):
            net.load_weights(str(candidate))
            metrics = calculate_metrics(net, model, ct, dr, session, args.batch_size)
            front_records.append({'checkpoint': str(candidate.resolve()), 'metrics': metrics})
        if front_records:
            with open(str(directory/'pareto_test_results.csv'), 'w',
                      newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['checkpoint', 'test_safety_violation', 'test_drawdown_retention'])
                for record in front_records:
                    writer.writerow([record['checkpoint'],
                                     record['metrics']['safety_violation_rate'],
                                     record['metrics']['drawdown_retention']])
        net.load_weights(str(selected))
    front_eval_sec = time.perf_counter()-front_begin

    previous = repair.get('timing_sec', {})
    prev_locator = previous.get('localization_previous_run_sec')
    prev_repair = previous.get('total_repair_run_wall_sec')
    test_total = float(time.perf_counter()-wall_start)
    combined = (float(prev_locator) + float(prev_repair) + test_total
                if prev_locator is not None and prev_repair is not None else None)
    timing = {
        'localization_previous_run_sec': prev_locator,
        'repair_previous_run_sec': prev_repair,
        'test_model_load_sec': float(model_load_sec),
        'test_data_load_sec': float(test_data_sec),
        'test_selected_checkpoint_eval_sec': float(selected_eval_sec),
        'test_optional_front_eval_sec': float(front_eval_sec),
        'test_total_wall_sec': test_total,
        'combined_historical_localization_repair_test_sec': combined,
        'notes': 'TEST does not select a new model. Historical times were measured '
                 'in different runs; the combined value is an arithmetic sum.',
    }
    result = {
        'schema': 'CGARepair_safety_weight_test_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'safety', 'model': model, 'optimizer': args.optimizer,
        'method': repair['method'], 'optimization_seed': int(args.seed),
        'selected_checkpoint': str(selected.resolve()),
        'checkpoint_selection': 'TRAIN only; TEST never used for best checkpoint selection',
        'checkpoint_selection_policy': repair['selection_policy'],
        'selected_players': repair['selected_players'], 'top_k': int(repair['top_k']),
        'test_data': data_info,
        'original_test_metrics': original, 'repaired_test_metrics': repaired,
        'delta_safety_violation_rate': float(repaired['safety_violation_rate']-
                                             original['safety_violation_rate']),
        'delta_drawdown_retention': float(repaired['drawdown_retention']-
                                         original['drawdown_retention']),
        'optional_front_test_results': front_records,
        'repair_timing_sec': previous,
        'timing_sec': timing, 'total_time_sec': test_total,
    }
    save_json(result_file, result)
    save_json(directory/'time_summary.json', {
        'schema': 'CGARepair_safety_full_timing_v1',
        'model': model, 'optimizer': args.optimizer,
        'localization': {
            'source_json': repair['localization']['source_json'],
            'historical_localization_elapsed_sec': prev_locator,
            'not_included_in_repair_wall': True,
        },
        'repair_timing_sec': previous,
        'test_timing_sec': timing,
        'combined_historical_localization_repair_test_sec': combined,
        'note': 'Fitness evaluations are included in optimization wall. '
                'Do not add them again. TEST did not choose best weights.',
    })
    print('[Timing] locator(previous)={}s repair(previous)={}s TEST={:.2f}s'.format(
        prev_locator, prev_repair, test_total), flush=True)
    print('[Files] {} {}'.format(result_file, directory/'time_summary.json'), flush=True)
    return result


def main():
    run(get_parser().parse_args())


if __name__ == '__main__':
    main()
