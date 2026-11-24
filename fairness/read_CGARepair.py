"""CGARepair fairness RQ1 ranking reader -- intentionally NOT named read.py.

Exact cache pattern from loc_CGARepair_test.py:
  CGARepair_fairness_samples4096_seed256_v3.json
Names contain actual model.layers indices, never sequential Conv/Dense indices.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

from cgarepair_fairness_utils import check_dataset, save_json, sha256_file

PLAYER_RE = re.compile(r'^(Conv\w*|Dense)_(\d+)_(\d+)$', re.IGNORECASE)


def adaptive_topk(n):
    if n < 1:
        raise ValueError('Empty localization ranking')
    return min(n, max(4, min(11, int(n * .01))))


def _candidate_dirs(loc_root, dataset):
    here = Path(__file__).resolve().parent
    candidates = []
    if loc_root:
        user = Path(loc_root).expanduser().resolve()
        candidates.extend((user, user / dataset, user / 'loc_result' / dataset))
    for start in (here, Path.cwd()):
        for parent in (start, start.parent):
            candidates.append(parent / 'rq1' / 'result' / 'loc_result' / dataset)
            candidates.append(parent / 'result' / 'loc_result' / dataset)
    unique = []
    for folder in candidates:
        folder = folder.resolve()
        if folder not in unique:
            unique.append(folder)
    return unique


def find_json(dataset, loc_seed=256, loc_samples=4096, loc_json=None, loc_root=None):
    dataset = check_dataset(dataset)
    if loc_json:
        candidate = Path(loc_json).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError('Specified --loc-json missing: {}'.format(candidate))
        return candidate
    filename = 'CGARepair_fairness_samples{}_seed{}_v3.json'.format(loc_samples, loc_seed)
    checked, found = [], []
    for folder in _candidate_dirs(loc_root, dataset):
        candidate = folder / filename
        checked.append(str(candidate))
        if candidate.is_file():
            found.append(candidate.resolve())
    unique = sorted(set(found), key=str)
    if not unique:
        raise FileNotFoundError('CGARepair fairness ranking missing: {}. Tried:\n  {}\n'
                                'If RQ1 used 512 rather than 4096, pass --loc-samples 512; '
                                'otherwise use --loc-json <exact filename>.'.format(
                                    filename, '\n  '.join(checked)))
    if len(unique) != 1:
        raise RuntimeError('Ambiguous CGARepair fairness caches:\n  {}\n'
                           'Choose one with --loc-json.'.format('\n  '.join(map(str, unique))))
    return unique[0]


def load_ranking(dataset, loc_seed=256, loc_samples=4096, loc_json=None, loc_root=None,
                 top_k=None):
    start = time.perf_counter()
    dataset = check_dataset(dataset)
    file = find_json(dataset, loc_seed, loc_samples, loc_json, loc_root)
    with open(str(file), 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError('Expected JSON object: {}'.format(file))
    if data.get('task') != 'fairness':
        raise ValueError('This ranking task={} is not fairness'.format(data.get('task')))
    if str(data.get('model')).lower() != dataset:
        raise ValueError('Ranking dataset={} differs from {}'.format(data.get('model'), dataset))
    method = str(data.get('method', ''))
    if not ('cgarepair' in method.lower() and 'gradact' in method.lower()):
        raise ValueError('Refusing non-CGARepair Gradient x Activation ranking: {}'.format(method))
    if data.get('seed') is None or int(data['seed']) != int(loc_seed):
        raise ValueError('CGARepair loc seed mismatch: {} != {}'.format(data.get('seed'), loc_seed))
    actual = data.get('num_examples', data.get('actual_sample_count'))
    if actual is None or int(actual) != int(loc_samples):
        raise ValueError('CGARepair loc sample count mismatch: {} != {}'.format(actual, loc_samples))
    players, scores = data.get('players'), data.get('scores')
    if not isinstance(players, list) or not isinstance(scores, list) or len(players) != len(scores) or not players:
        raise ValueError('Ranking players/scores missing or inconsistent in {}'.format(file))
    if len(set(players)) != len(players):
        raise ValueError('Duplicate neurons in CGARepair ranking')
    pairs = []
    for name, score in zip(players, scores):
        if PLAYER_RE.fullmatch(str(name)) is None:
            raise ValueError('Unexpected actual-layer neuron name {}'.format(name))
        value = float(score)
        if not math.isfinite(value):
            raise ValueError('Nonfinite CGARepair score for neuron {}'.format(name))
        pairs.append((name, value))
    pairs.sort(key=lambda item: item[1], reverse=True)
    k = adaptive_topk(len(pairs)) if top_k is None else int(top_k)
    if not 1 <= k <= len(pairs):
        raise ValueError('Top-k {} outside [1,{}]'.format(k, len(pairs)))
    top = pairs[:k]
    if sum(1 for _, v in top if v > 0.0) == 0:
        raise ValueError('CGARepair fairness Top-k scores all zero; inspect directional attribution')
    meta = {
        'task': 'fairness', 'localization_method': 'CGARepair', 'dataset': dataset,
        'source_json': str(file), 'source_sha256': sha256_file(file),
        'source_formula': method, 'source_model_signature': data.get('model_signature'),
        'source_data_selection': data.get('data_selection'),
        'source_fairness_sample_fingerprint': data.get('fairness_sample_fingerprint'),
        'source_direction': data.get('direction'),
        'source_runtime_sec': float(data['runtime_sec']) if data.get('runtime_sec') is not None else None,
        'source_runtime_is_previous_run': True,
        'loc_seed': int(loc_seed), 'loc_samples': int(loc_samples),
        'num_players': len(pairs), 'top_k': int(k),
        'selected_topk': [name for name, _ in top],
        'selected_scores': [float(score) for _, score in top],
        'read_time_sec': float(time.perf_counter() - start)
    }
    return meta['selected_topk'], meta['selected_scores'], meta


def verify_model_signature(meta, model_path):
    """Reject stale locator runs after a model has changed, when signature exists."""
    sig = meta.get('source_model_signature')
    if sig is None:
        return 'no_model_signature_in_locator'
    stat = Path(model_path).stat()
    observed = [int(stat.st_size), int(stat.st_mtime_ns)]
    if list(sig) != observed:
        raise RuntimeError('Model does not match localization model_signature. '
                           'locator={}, current={}. Rerun localization on the '
                           'same model or pass correct --model-root.'.format(sig, observed))
    return 'matched_file_size_and_mtime_ns'


def export_ranking(players, scores, meta, export_dir):
    folder = Path(export_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'players.txt').write_text(','.join(players), encoding='utf-8')
    (folder / 'fairness_vals.txt').write_text(
        ','.join('{:.17g}'.format(score) for score in scores), encoding='utf-8')
    save_json(folder / 'manifest.json', meta)
    return str(folder.resolve())


def main():
    parser = argparse.ArgumentParser(description='CGARepair fairness Top-k reader; original read.py unchanged')
    parser.add_argument('dataset', choices=('bank', 'adult', 'default'))
    parser.add_argument('--loc-seed', type=int, default=256)
    parser.add_argument('--loc-samples', type=int, default=4096)
    parser.add_argument('--loc-json')
    parser.add_argument('--loc-root')
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--export-dir')
    args = parser.parse_args()
    names, scores, meta = load_ranking(args.dataset, args.loc_seed, args.loc_samples,
                                        args.loc_json, args.loc_root, args.top_k)
    out = args.export_dir or str(Path('loc_result') / args.dataset / 'CGARepair' / 'localization_input')
    export_ranking(names, scores, meta, out)
    print('[Locator] {} Top-{} / {} from {} (loc_seed={})'.format(
        args.dataset, len(names), meta['num_players'], meta['source_json'], args.loc_seed))
    for name, score in zip(names, scores):
        print('  {}: {:.9g}'.format(name, score))
    print('Wrote {}. Original CFRepair read.py untouched.'.format(out))


if __name__ == '__main__':
    main()
