"""CGARepair ADV neuron ranking adapter (MNIST epsilon .05 / CIFAR-10 epsilon .1).

Reads the existing CGARepair_adv_samples512_seed256_eps..._v3.json directly;
no legacy sequential-layer renumbering, and no dependency on backdoor read.py.
"""
import argparse
import glob
import json
import math
import re
import time
from pathlib import Path

METHOD = 'CGARepair'
PLAYER_RE = re.compile(r'^(Conv\w*|Dense)_(\d+)_(\d+)$', re.IGNORECASE)


def model_name_checked(model):
    model = str(model).strip().lower()
    if model not in ('mnist', 'cifar10'):
        raise ValueError('ADV model must be mnist or cifar10, received {}'.format(model))
    return model


def epsilon_string(epsilon):
    try:
        value = float(epsilon)
    except (ValueError, TypeError):
        raise ValueError('Invalid epsilon {}'.format(epsilon))
    if not 0.0 < value < 1.0:
        raise ValueError('Expected attack epsilon in (0,1), got {}'.format(epsilon))
    return format(value, '.12g')


def adaptive_topk(n):
    if n <= 0:
        raise ValueError('Neuron ranking must be nonempty')
    return min(n, max(4, min(11, int(n * .01))))


def _candidate_folders(loc_root, model):
    folders = []
    if loc_root:
        p = Path(loc_root).expanduser().resolve()
        folders.extend([p / model, p])
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for base in (cwd, here, cwd.parent, here.parent):
        for root in (base / 'result' / 'loc_result',
                     base / 'rq1' / 'result' / 'loc_result',
                     base.parent / 'rq1' / 'result' / 'loc_result'):
            folders.extend([root / model, root])
    unique = []
    for p in folders:
        s = str(p)
        if s not in unique:
            unique.append(s)
    return unique


def find_ranking_json(model, epsilon, loc_seed=256, loc_samples=512,
                      loc_json=None, loc_root=None):
    model = model_name_checked(model)
    eps = epsilon_string(epsilon)
    if loc_json:
        p = Path(loc_json).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError('Specified --loc-json missing: {}'.format(p))
        return str(p)
    filename = 'CGARepair_adv_samples{}_seed{}_eps{}_v3.json'.format(
        loc_samples, loc_seed, eps)
    found = []
    checked = []
    for folder in _candidate_folders(loc_root, model):
        p = Path(folder) / filename
        checked.append(str(p))
        if p.is_file():
            found.append(str(p.resolve()))
    found = sorted(set(found))
    if not found:
        raise FileNotFoundError('No CGARepair adversarial ranking for {} eps={}. Expected:\n  {}\n'
                                'Use --loc-json or --loc-root.'.format(model, eps, '\n  '.join(checked)))
    if len(found) > 1:
        raise RuntimeError('Ambiguous CGARepair ADV ranking files:\n  {}\n'
                           'Please supply --loc-json.'.format('\n  '.join(found)))
    return found[0]


def load_ranking(model, epsilon, loc_seed=256, loc_samples=512,
                 loc_json=None, loc_root=None, strict_topk=True):
    t = time.perf_counter()
    model = model_name_checked(model)
    eps = epsilon_string(epsilon)
    source = find_ranking_json(model, eps, loc_seed, loc_samples, loc_json, loc_root)
    with open(source, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if str(data.get('task')).lower() != 'adv':
        raise ValueError('Expected task=adv, got {}'.format(data.get('task')))
    if str(data.get('model')).lower() != model:
        raise ValueError('CGARepair model mismatch: {} != {}'.format(data.get('model'), model))
    if data.get('seed') is not None and int(data['seed']) != int(loc_seed):
        raise ValueError('Localization seed mismatch')
    count = data.get('num_examples', data.get('num_examples_requested'))
    if count is not None and int(count) != int(loc_samples):
        raise ValueError('Localization samples mismatch')
    if data.get('epsilon') is not None and epsilon_string(data['epsilon']) != eps:
        raise ValueError('CGARepair epsilon mismatch: {} != {}'.format(data['epsilon'], eps))
    method = str(data.get('method', ''))
    if 'gradact' not in method.lower() and 'cgarepair' not in method.lower():
        raise ValueError('Not a CGARepair/GradAct ranking: {}'.format(method))
    players = data.get('players')
    scores = data.get('scores')
    if not isinstance(players, list) or not isinstance(scores, list) or len(players) != len(scores) or not players:
        raise ValueError('Ranking players/scores absent or inconsistent')
    if len(set(players)) != len(players):
        raise ValueError('Duplicate neuron names in ranking')
    pairs = []
    for player, score in zip(players, scores):
        if PLAYER_RE.fullmatch(str(player)) is None:
            raise ValueError('Unexpected neuron notation {}. Need Conv2D_5_10 / Dense_9_3.'.format(player))
        val = float(score)
        if not math.isfinite(val):
            raise ValueError('Nonfinite score for {}'.format(player))
        pairs.append((player, val))
    pairs.sort(key=lambda p: p[1], reverse=True)
    k = adaptive_topk(len(pairs))
    selected = [x[0] for x in pairs[:k]]
    verified = False
    # The original locator writes this independent cumulative-ablation artifact.
    result_root = Path(source).parent.parent.parent
    ablation = (result_root / 'adv' / model / ('epsilon_' + eps) /
                'top{}_samples{}_CGARepair_seed{}.txt'.format(k, loc_samples, loc_seed))
    if ablation.is_file():
        with open(str(ablation), 'r', encoding='utf-8') as f:
            record = json.load(f)
        if strict_topk and int(record.get('top_k', -1)) != k:
            raise RuntimeError('CGARepair top-k ablation disagrees with ranking size')
        linked = record.get('selected_topk_path')
        candidates = [ablation.with_name(ablation.stem + '_selected_topk.txt')]
        if linked:
            candidates = [Path(linked), result_root.parent / linked,
                          ablation.parent / Path(linked).name] + candidates
        for named in candidates:
            if named.is_file():
                names = [x.strip() for x in named.read_text(encoding='utf-8').split(',') if x.strip()]
                if strict_topk and names != selected:
                    raise RuntimeError('Ablation selection and ranking selection differ: {} vs {}'.format(names, selected))
                verified = True
                break
    meta = {
        'task': 'adv', 'method': METHOD, 'model': model, 'epsilon': eps,
        'source_json': source, 'source_method': method,
        'source_runtime_sec': (float(data['runtime_sec']) if data.get('runtime_sec') is not None else None),
        'source_runtime_is_previous_run': True,
        'source_model_signature': data.get('model_signature'),
        'source_data_selection': data.get('data_selection'),
        'loc_seed': int(loc_seed), 'loc_samples': int(loc_samples),
        'num_players': len(pairs), 'top_k': k,
        'selected_topk': selected,
        'selected_scores': [float(p[1]) for p in pairs[:k]],
        'ablation_topk_verified': verified,
        'read_time_sec': float(time.perf_counter() - t),
    }
    return selected, [float(p[1]) for p in pairs[:k]], meta


def export_ranking(players, scores, meta, export_dir):
    folder = Path(export_dir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'players.txt').write_text(','.join(players), encoding='utf-8')
    (folder / 'adv_vals.txt').write_text(','.join('{:.17g}'.format(s) for s in scores), encoding='utf-8')
    with open(str(folder / 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return str(folder.resolve())


def main():
    p = argparse.ArgumentParser(description='Inspect/export CGARepair ADV Top-k ranking')
    p.add_argument('model_name', choices=('mnist', 'cifar10'))
    p.add_argument('epsilon', help='0.05 for MNIST, 0.1 for CIFAR-10')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--loc-json')
    p.add_argument('--loc-root')
    p.add_argument('--export-dir')
    a = p.parse_args()
    players, scores, meta = load_ranking(a.model_name, a.epsilon, a.loc_seed,
                                         a.loc_samples, a.loc_json, a.loc_root)
    out = a.export_dir or 'loc_result/adv/{}/epsilon_{}/CGARepair/localization_input'.format(
        a.model_name, epsilon_string(a.epsilon))
    export_ranking(players, scores, meta, out)
    print('[CGARepair ADV] model={} epsilon={} selected Top-{}/{} from {}'.format(
        a.model_name, meta['epsilon'], meta['top_k'], meta['num_players'], meta['source_json']))
    for name, score in zip(players, scores):
        print('  {} {:.9g}'.format(name, score))
    print('[CGARepair ADV] prior localization runtime (seconds): {}'.format(meta['source_runtime_sec']))
    print('[CGARepair ADV] exported {}'.format(out))


if __name__ == '__main__':
    main()
