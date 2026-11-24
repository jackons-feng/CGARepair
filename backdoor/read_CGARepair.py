"""Read CGARepair backdoor GradAct ranking and export the exact adaptive Top-k.

Python >=3.7. Does not require TensorFlow. Run either as a CLI or import
load_ranking() from backdoor_moo_2.py.
"""
import argparse
import glob
import json
import math
import os
import re
import time
from pathlib import Path

METHOD = 'CGARepair'
NAME_RE = re.compile(r'^(Conv\w*|Dense)_(\d+)_(\d+)$', flags=re.IGNORECASE)


def resolve_model_name(model_name, resnet=False):
    raw = str(model_name).lower()
    if raw == 'gtsrb_resnet':
        return 'gtsrb', True
    if raw not in ('mnist', 'fashion', 'gtsrb'):
        raise ValueError('Unsupported backdoor model: {} (use mnist, fashion, gtsrb)'.format(model_name))
    if resnet and raw != 'gtsrb':
        raise ValueError('The original setup only supports --resnet for gtsrb')
    return raw, bool(resnet)


def model_tag(model_name, resnet=False):
    model_name, resnet = resolve_model_name(model_name, resnet)
    return model_name + ('_resnet' if resnet else '')


def adaptive_topk(n):
    if n < 1:
        raise ValueError('Ranking is empty')
    return min(n, max(4, min(11, int(n * 0.01))))


def _candidate_dirs(loc_root, tag):
    roots = []
    if loc_root:
        roots.append(Path(loc_root).expanduser().resolve())
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    for base in (cwd, here, cwd.parent, here.parent):
        roots.extend([base / 'result' / 'loc_result',
                      base / 'rq1' / 'result' / 'loc_result',
                      base / 'loc_result'])
    result = []
    for root in roots:
        for p in (root / tag, root):
            if str(p) not in result:
                result.append(str(p))
    return result


def find_location_json(model_name, resnet=False, loc_seed=256, loc_samples=512,
                       loc_json=None, loc_root=None):
    tag = model_tag(model_name, resnet)
    if loc_json:
        path = Path(loc_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Explicit --loc-json does not exist: {}'.format(path))
        return str(path)
    pattern = 'CGARepair_backdoor_samples{}_seed{}*.json'.format(loc_samples, loc_seed)
    matches = []
    searched = []
    for directory in _candidate_dirs(loc_root, tag):
        searched.append(str(Path(directory) / pattern))
        matches.extend(glob.glob(str(Path(directory) / pattern)))
    matches = sorted(set(str(Path(p).resolve()) for p in matches))
    if not matches:
        raise FileNotFoundError('No CGARepair ranking JSON for {}. Searched:\n  {}\n'
                                'Use --loc-json /full/path/to/CGARepair_backdoor_...json'.format(
                                    tag, '\n  '.join(searched)))
    if len(matches) > 1:
        # Never silently use an arbitrary version of the localization ranking.
        raise RuntimeError('Ambiguous ranking JSON files for {}:\n  {}\n'
                           'Supply --loc-json to select the exact experiment.'.format(
                               tag, '\n  '.join(matches)))
    return matches[0]


def load_ranking(model_name, resnet=False, loc_seed=256, loc_samples=512,
                 loc_json=None, loc_root=None, strict_topk=True):
    """Return selected_players, selected_scores, metadata, including ALL ranking information."""
    start = time.perf_counter()
    model_name, resnet = resolve_model_name(model_name, resnet)
    tag = model_tag(model_name, resnet)
    source = find_location_json(model_name, resnet, loc_seed, loc_samples, loc_json, loc_root)
    with open(source, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get('task') != 'backdoor':
        raise ValueError('Expected task=backdoor, got {} in {}'.format(data.get('task'), source))
    if str(data.get('model')).lower() not in (model_name, tag):
        raise ValueError('Wrong model in ranking JSON: {} vs {}'.format(data.get('model'), tag))
    if data.get('seed') is not None and int(data['seed']) != int(loc_seed):
        raise ValueError('Localization seed mismatch: {} != {}'.format(data['seed'], loc_seed))
    source_n = data.get('num_examples', data.get('num_examples_requested'))
    if source_n is not None and int(source_n) != int(loc_samples):
        raise ValueError('Localization sample count mismatch: {} != {}'.format(source_n, loc_samples))
    method = str(data.get('method', ''))
    if not ('gradact' in method.lower() or 'cgarepair' in method.lower()):
        raise ValueError('This does not look like a CGARepair/GradAct cache: method={}'.format(method))
    players = data.get('players')
    scores = data.get('scores')
    if not isinstance(players, list) or not isinstance(scores, list) or len(players) != len(scores):
        raise ValueError('players/scores missing or inconsistent in {}'.format(source))
    if not players:
        raise ValueError('Empty neuron ranking')
    if len(set(players)) != len(players):
        raise ValueError('Duplicate players found in ranking')
    pairs = []
    for player, score in zip(players, scores):
        m = NAME_RE.fullmatch(player)
        if not m:
            raise ValueError('Unexpected player notation: {}. Expected Conv2D_8_2 or Dense_12_4.'.format(player))
        value = float(score)
        if not math.isfinite(value):
            raise ValueError('Non-finite localization score for {}'.format(player))
        pairs.append((player, value))
    # Python's stable sort matches the CGARepair locator's items.sort(descending).
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    k = adaptive_topk(len(pairs))
    selected = pairs[:k]
    selected_names = [p for p, _ in selected]
    # Optionally validate the persisted Top-k text from the original locator.
    ablation_candidates = glob.glob(str(Path(source).parent.parent.parent / 'backdoor' /
                                        tag / 'top{}_samples{}_CGARepair_seed{}.txt'.format(
                                            k, loc_samples, loc_seed)))
    verified = False
    if ablation_candidates:
        with open(ablation_candidates[0], 'r', encoding='utf-8') as f:
            ablation = json.load(f)
        if strict_topk and int(ablation.get('top_k', -1)) != k:
            raise RuntimeError('CGARepair ablation Top-k disagrees with ranking')
        topk_file = ablation.get('selected_topk_path')
        if topk_file:
            possibilities = [Path(topk_file), Path(source).parent.parent.parent / topk_file,
                             Path(ablation_candidates[0]).parent / Path(topk_file).name]
            for path in possibilities:
                if path.is_file():
                    with open(path, 'r', encoding='utf-8') as f:
                        names = [s.strip() for s in f.read().split(',') if s.strip()]
                    if strict_topk and selected_names != names:
                        raise RuntimeError('CGARepair selected_topk_path differs from source scores:\n{}\n{}'.format(
                            selected_names, names))
                    verified = True
                    break
    meta = {
        'method': METHOD,
        'model': tag,
        'loc_seed': int(loc_seed),
        'loc_samples': int(loc_samples),
        'source_json': source,
        'source_method': method,
        'source_runtime_sec': (float(data['runtime_sec']) if data.get('runtime_sec') is not None else None),
        'source_runtime_note': 'Recorded by an earlier localization run; excluded from this repair run total',
        'source_model_signature': data.get('model_signature'),
        'source_data_selection': data.get('data_selection'),
        'num_players': len(pairs),
        'top_k': k,
        'selected_topk': selected_names,
        'selected_scores': [float(s) for _, s in selected],
        'ablation_selected_file_verified': verified,
        'read_time_sec': float(time.perf_counter()-start),
    }
    return selected_names, [float(s) for _, s in selected], meta


def export_ranking(players, scores, meta, export_dir=None):
    tag = meta['model']
    folder = Path(export_dir).expanduser() if export_dir else Path('loc_result') / tag / METHOD
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / 'players.txt', 'w', encoding='utf-8') as f:
        f.write(','.join(players))
    with open(folder / 'backdoor_vals.txt', 'w', encoding='utf-8') as f:
        f.write(','.join('{:.17g}'.format(s) for s in scores))
    with open(folder / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return str(folder.resolve())


def main():
    p = argparse.ArgumentParser(description='Export the exact Top-k CGARepair localization for repair.')
    p.add_argument('model_name', help='mnist, fashion, gtsrb or gtsrb_resnet')
    p.add_argument('--resnet', action='store_true')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--loc-json')
    p.add_argument('--loc-root')
    p.add_argument('--export-dir')
    a = p.parse_args()
    players, scores, meta = load_ranking(a.model_name, a.resnet, a.loc_seed, a.loc_samples,
                                         a.loc_json, a.loc_root)
    path = export_ranking(players, scores, meta, a.export_dir)
    print('[CGARepair] {} -> selected Top-{} / {} neurons'.format(meta['model'], meta['top_k'], meta['num_players']))
    for name, score in zip(players, scores):
        print('  {} {:.9g}'.format(name, score))
    print('[CGARepair] Localization time from earlier run: {}'.format(meta['source_runtime_sec']))
    print('[CGARepair] Saved players.txt, backdoor_vals.txt, manifest.json in {}'.format(path))


if __name__ == '__main__':
    main()
