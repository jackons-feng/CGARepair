"""BIRDNN-AD / CCBR-IGD localization adapter for formal FAMO.

Python >= 3.7. Original read_CGARepair.py remains unchanged.
Requires an explicit --loc-json. Only the four verified ablation tasks
are enabled. Never relabels BIRDNN or CCBR as GradAct.
"""
import inspect
import json
import math
import re
import time
from pathlib import Path
import read_CGARepair as _base

# Preserve the original reader's other public functions, especially
# export_ranking(), so existing FAMO scripts remain compatible.
for _key in dir(_base):
    if not _key.startswith('__') and _key not in globals():
        globals()[_key] = getattr(_base, _key)

_TASK = Path(__file__).resolve().parent.name.rsplit('cfrepair_', 1)[-1]
_ROOT = Path(__file__).resolve().parent.parent
_NAME = re.compile(r'^(Conv\w*|Dense)_(\d+)_(\d+)$', re.I)

_CASES = {
    'adv': {
        'mnist': ('mnist',
                  'CGARepair_adv_samples512_seed256_eps0.05_v3.json', 512),
    },
    'backdoor': {
        'fashion': ('fashion',
                    'CGARepair_backdoor_samples512_seed256_v3.json', 512),
    },
    'safety': {
        'a33': ('a33',
                'CGARepair_safety_samples512_seed256_v3.json', 512),
    },
    'fairness': {
        'bank': ('bank',
                 'CGARepair_fairness_samples4096_seed256_v3.json', 4096),
    },
}

def _read(path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError('Localization JSON too large: {}'.format(path))
    with path.open('r', encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError('Expected JSON object: {}'.format(path))
    return obj

def load_ranking(*args, **kwargs):
    """Drop-in replacement using the original reader's argument signature."""
    start = time.perf_counter()

    bound = inspect.signature(_base.load_ranking).bind_partial(
        *args, **kwargs
    )
    bound.apply_defaults()
    a = bound.arguments

    model = a.get('model_name', a.get('dataset', a.get('model')))
    if model is None:
        raise ValueError('Cannot identify model from reader arguments')

    model = str(model).lower()

    if model not in _CASES.get(_TASK, {}):
        raise ValueError(
            'Unsupported ablation task/model: {}/{}'.format(_TASK, model)
        )

    folder, ref_name, required_samples = _CASES[_TASK][model]

    seed = int(a.get('loc_seed', 256))
    samples = int(a.get('loc_samples', required_samples))

    if seed != 256 or samples != required_samples:
        raise ValueError(
            'Expected loc_seed=256, loc_samples={}'.format(required_samples)
        )

    source_name = a.get('loc_json')
    if not source_name:
        raise ValueError('AD/IGD requires an explicit --loc-json')

    path = Path(source_name).expanduser().resolve()
    loc_dir = (
        _ROOT / 'rq1' / 'result' / 'loc_result' / folder
    ).resolve()

    if path.parent != loc_dir:
        raise ValueError(
            'Use the verified RQ1 localization directory: {}'.format(loc_dir)
        )

    data = _read(path)
    ref = _read(loc_dir / ref_name)

    raw_method = str(data.get('method', ''))
    method_key = raw_method.lower()

    if method_key == 'birdnn':
        method = 'AD_BIRDNN'
    elif method_key == 'ccbr':
        method = 'IGD_CCBR'
    else:
        raise ValueError(
            'Expected BIRDNN or CCBR, got {}'.format(raw_method)
        )

    if method_key not in path.name.lower():
        raise ValueError('JSON filename/method mismatch')

    if data.get('task') != _TASK:
        raise ValueError('Task mismatch')

    if str(data.get('model', '')).lower() != model:
        raise ValueError('Model mismatch')

    if data.get('seed') is None or int(data['seed']) != seed:
        raise ValueError('Localization seed mismatch')

    n = data.get('num_examples', data.get('sample_n'))
    if n is None or int(n) != samples:
        raise ValueError('Localization sample count mismatch')

    if _TASK == 'adv':
        epsilon = a.get('epsilon', a.get('eps'))
        if epsilon is None or abs(float(epsilon) - 0.05) > 1e-9:
            raise ValueError('Expected MNIST epsilon=0.05')
        if data.get('epsilon') is not None:
            if abs(float(data['epsilon']) - 0.05) > 1e-9:
                raise ValueError('Source epsilon mismatch')

    if ref.get('task') != _TASK:
        raise ValueError('Reference task mismatch')

    if str(ref.get('model', '')).lower() != model:
        raise ValueError('Reference model mismatch')

    if int(ref.get('seed', -1)) != seed:
        raise ValueError('Reference seed mismatch')

    if int(ref.get('num_examples', -1)) != samples:
        raise ValueError('Reference sample count mismatch')

    players = data.get('players')
    scores = data.get('scores')

    if (not isinstance(players, list) or
        not isinstance(scores, list) or
        not players or len(players) != len(scores)):
        raise ValueError('Invalid players/scores')

    if len(set(players)) != len(players):
        raise ValueError('Duplicate neuron identifiers')

    if set(players) != set(ref.get('players', [])):
        raise ValueError(
            'Candidate neuron set differs from formal GradAct'
        )

    pairs = []
    for player, score in zip(players, scores):
        if not isinstance(player, str) or not _NAME.fullmatch(player):
            raise ValueError('Invalid player: {}'.format(player))

        value = float(score)
        if not math.isfinite(value):
            raise ValueError('Nonfinite score: {}'.format(player))

        pairs.append((player, value))

    fp = data.get('fairness_sample_fingerprint')
    ref_fp = ref.get('fairness_sample_fingerprint')

    if _TASK == 'fairness' and (not fp or fp != ref_fp):
        raise ValueError('Bank sample fingerprint mismatch')

    if fp is not None and ref_fp is not None and fp != ref_fp:
        raise ValueError('Sample fingerprint mismatch')

    # Same stable descending sort and adaptive Top-K as formal GradAct.
    pairs.sort(key=lambda item: item[1], reverse=True)

    k = min(len(pairs), max(4, min(11, int(len(pairs) * 0.01))))

    if a.get('top_k') is not None:
        if int(a['top_k']) != k:
            raise ValueError('Top-K override differs from formal protocol')

    chosen = pairs[:k]

    runtime = data.get('runtime_sec', data.get('time'))
    try:
        runtime = float(runtime) if runtime is not None else None
        if runtime is not None and not math.isfinite(runtime):
            runtime = None
    except (TypeError, ValueError):
        runtime = None

    # Baseline JSONs do not record model_signature.
    # The GradAct reference signature is used ONLY to check that FAMO
    # loads the original pretrained checkpoint. Its origin is recorded.
    baseline_signature = data.get('model_signature')
    reference_signature = ref.get('model_signature')

    meta = {
        'method': method,
        'model': model,
        'loc_seed': seed,
        'loc_samples': samples,
        'source_json': str(path),
        'source_method': raw_method,
        'source_runtime_sec': runtime,
        'source_runtime_note': 'Earlier RQ1 localization, not FAMO runtime',
        'source_model_signature': (
            baseline_signature or reference_signature
        ),
        'source_model_signature_origin': (
            'baseline_json' if baseline_signature
            else 'official_gradact_reference_only'
        ),
        'baseline_json_model_signature': baseline_signature,
        'reference_gradact_json': str(loc_dir / ref_name),
        'source_data_selection': data.get('data_selection'),
        'source_sample_fingerprint': fp,
        'sample_identity_proved': bool(
            fp and ref_fp and fp == ref_fp
        ),
        'num_players': len(pairs),
        'top_k': k,
        'selected_topk': [p for p, _ in chosen],
        'selected_scores': [float(s) for _, s in chosen],
        'ablation_selected_file_verified': False,
        'read_time_sec': float(time.perf_counter() - start),
    }

    return meta['selected_topk'], meta['selected_scores'], meta
