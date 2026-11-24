"""Read CGARepair safety rankings without touching CFRepair's legacy read.py.

The original locator emits CGARepair_safety_samples512_seed256_v3.json
under rq1/result/loc_result/{a19,a29,a33}/.
Player labels are Keras model.layers actual indices, not sequential Conv/Dense indices.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

METHOD = 'CGARepair'
SUPPORTED = ('a19', 'a29', 'a33')
PLAYER_RE = re.compile(r'^(Conv\w*|Dense)_(\d+)_(\d+)$', re.IGNORECASE)


def model_name_checked(value):
    model = str(value).strip().lower()
    if model not in SUPPORTED:
        raise ValueError('Safety models are a19, a29, a33; received {}'.format(value))
    return model


def adaptive_topk(n):
    if n <= 0:
        raise ValueError('No neurons in CGARepair ranking')
    return min(n, max(4, min(11, int(n * .01))))


def _search_folders(model, loc_root=None):
    folders = []
    if loc_root:
        root = Path(loc_root).expanduser().resolve()
        folders.extend([root / model, root, root / 'result' / 'loc_result' / model,
                        root / 'rq1' / 'result' / 'loc_result' / model])
    here = Path(__file__).resolve().parent
    for base in (Path.cwd(), here, Path.cwd().parent, here.parent):
        folders.extend([
            base / 'result' / 'loc_result' / model,
            base / 'rq1' / 'result' / 'loc_result' / model,
            base.parent / 'rq1' / 'result' / 'loc_result' / model,
        ])
    result = []
    seen = set()
    for path in folders:
        normalized = str(path.resolve())
        if normalized not in seen:
            result.append(Path(normalized))
            seen.add(normalized)
    return result


def find_ranking_json(model, loc_seed=256, loc_samples=512, loc_json=None, loc_root=None):
    model = model_name_checked(model)
    if loc_json:
        f = Path(loc_json).expanduser().resolve()
        if not f.is_file():
            raise FileNotFoundError('Specified --loc-json does not exist: {}'.format(f))
        return str(f)
    name = 'CGARepair_safety_samples{}_seed{}_v3.json'.format(loc_samples, loc_seed)
    checked = [folder / name for folder in _search_folders(model, loc_root)]
    found = sorted(set(str(p.resolve()) for p in checked if p.is_file()))
    if not found:
        raise FileNotFoundError('CGARepair safety ranking missing for {}. Checked:\n  {}\n'
                                'Use --loc-json to specify an exact file.'.format(
                                    model, '\n  '.join(str(p) for p in checked)))
    if len(found) != 1:
        raise RuntimeError('Multiple safety rankings found:\n  {}\nUse --loc-json.'.format(
            '\n  '.join(found)))
    return found[0]


def load_ranking(model, loc_seed=256, loc_samples=512,
                 loc_json=None, loc_root=None, strict_topk=True):
    begin = time.perf_counter()
    model = model_name_checked(model)
    source = find_ranking_json(model, loc_seed, loc_samples, loc_json, loc_root)
    with open(source, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if str(data.get('task', '')).lower() != 'safety':
        raise ValueError('Locator task must be safety, found {}'.format(data.get('task')))
    if str(data.get('model', '')).lower() != model:
        raise ValueError('Model mismatch: {} != {}'.format(data.get('model'), model))
    if data.get('seed') is not None and int(data['seed']) != int(loc_seed):
        raise ValueError('Localization seed mismatch')
    if data.get('num_examples') is not None and int(data['num_examples']) != int(loc_samples):
        raise ValueError('Localization sample count mismatch')
    method = str(data.get('method', ''))
    if 'cgarepair' not in method.lower() and 'gradact' not in method.lower():
        raise ValueError('Not a CGARepair ranking: {}'.format(method))
    players, scores = data.get('players'), data.get('scores')
    if not isinstance(players, list) or not isinstance(scores, list) or not players or len(players) != len(scores):
        raise ValueError('Missing or inconsistent locator players/scores')
    if len(set(players)) != len(players):
        raise ValueError('Duplicate player names in localization')
    pairs = []
    for name, score in zip(players, scores):
        if not PLAYER_RE.fullmatch(str(name)):
            raise ValueError('Bad neuron name {}; expected Dense_9_3 or Conv2D_5_4'.format(name))
        value = float(score)
        if not math.isfinite(value):
            raise ValueError('Nonfinite locator score: {}'.format(name))
        pairs.append((name, value))
    pairs.sort(key=lambda t: t[1], reverse=True)
    k = adaptive_topk(len(pairs))
    selected = [t[0] for t in pairs[:k]]

    # Cross-check the independent localization ablation result where available.
    result_root = Path(source).parent.parent.parent
    ablation = result_root / 'safety' / model / (
        'top{}_samples{}_CGARepair_seed{}.txt'.format(k, loc_samples, loc_seed))
    verified = False
    if ablation.is_file():
        with open(str(ablation), 'r', encoding='utf-8') as f:
            record = json.load(f)
        if strict_topk and int(record.get('top_k', -1)) != k:
            raise RuntimeError('Top-k ablation result disagrees with ranking')
        potential = [ablation.with_name(ablation.stem + '_selected_topk.txt')]
        if record.get('selected_topk_path'):
            named = Path(record['selected_topk_path'])
            potential.extend([named, ablation.parent / named.name, result_root.parent / named])
        for candidate in potential:
            if candidate.is_file():
                listed = [p.strip() for p in candidate.read_text(encoding='utf-8').split(',') if p.strip()]
                if strict_topk and listed != selected:
                    raise RuntimeError('Cumulative Top-k and CGARepair ranking differ')
                verified = True
                break
    runtime = data.get('runtime_sec')
    meta = {
        'task': 'safety', 'method': METHOD, 'model': model,
        'source_json': source, 'source_method': method,
        'source_runtime_sec': float(runtime) if runtime is not None else None,
        'source_runtime_is_previous_run': True,
        'source_model_signature': data.get('model_signature'),
        'source_data_selection': data.get('data_selection'),
        'loc_seed': int(loc_seed), 'loc_samples': int(loc_samples),
        'num_players': len(pairs), 'top_k': k,
        'selected_topk': selected,
        'selected_scores': [float(p[1]) for p in pairs[:k]],
        'ablation_topk_verified': bool(verified),
        'read_time_sec': float(time.perf_counter() - begin),
    }
    return selected, [float(t[1]) for t in pairs[:k]], meta


def export_ranking(players, scores, meta, export_dir):
    folder = Path(export_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'players.txt').write_text(','.join(players), encoding='utf-8')
    (folder / 'safety_vals.txt').write_text(','.join('{:.17g}'.format(v) for v in scores),
                                           encoding='utf-8')
    with open(str(folder / 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, allow_nan=False)
    return str(folder.resolve())


def main():
    parser = argparse.ArgumentParser(description='Check CGARepair safety localization (no CFRepair read.py overwrite)')
    parser.add_argument('model_name', choices=SUPPORTED)
    parser.add_argument('--loc-seed', type=int, default=256)
    parser.add_argument('--loc-samples', type=int, default=512)
    parser.add_argument('--loc-json')
    parser.add_argument('--loc-root')
    parser.add_argument('--export-dir')
    args = parser.parse_args()
    players, scores, meta = load_ranking(args.model_name, args.loc_seed, args.loc_samples,
                                         args.loc_json, args.loc_root)
    out = args.export_dir or 'loc_result/{}/CGARepair/localization_input'.format(args.model_name)
    export_ranking(players, scores, meta, out)
    print('[CGARepair SAFETY] {} Top-{}/{} source={}'.format(
        args.model_name, meta['top_k'], meta['num_players'], meta['source_json']))
    for name, score in zip(players, scores):
        print('  {} {:.9g}'.format(name, score))
    print('[Timing] previous locator {}s, ranking read {:.4f}s'.format(
        meta['source_runtime_sec'], meta['read_time_sec']))
    print('[Files] exported {}'.format(out))


if __name__ == '__main__':
    main()
