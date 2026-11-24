"""ACAS-like safety data and metrics, preserving the user's original safety scripts.

train: ../data/{model}/counterexample.h5, drawndown.h5
TEST : ../data/{model}/counterexample_test.h5, drawndown_test.h5

Counterexamples are assessed by unsafe argmin output classes; drawndown
retention is the rate of safe argmin classes on a distinct preservation set.
These are NOT supervised classification accuracy measures.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from read_CGARepair import model_name_checked


# Threshold 0.999 matches the original safety repair's retention guard.
CONFIG = {model: {'threshold_acc': .999, 'threshold_safety': 0.0}
          for model in ('a19', 'a29', 'a33')}


def initialize_tf(seed=42):
    import tensorflow as tf
    import tensorflow.python.keras.backend as K
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass
    np.random.seed(int(seed))
    if hasattr(tf.random, 'set_seed'):
        tf.random.set_seed(int(seed))
    else:
        tf.compat.v1.set_random_seed(int(seed))
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=cfg)
    if hasattr(K, 'set_session'):
        K.set_session(session)
    else:
        tf.compat.v1.keras.backend.set_session(session)
    return tf, K, session


def _find_model_root(override=None):
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError('--model-root missing: {}'.format(root))
        return root
    cwd, here = Path.cwd(), Path(__file__).resolve().parent
    for path in (cwd.parent/'model', here.parent/'model', cwd/'model', here/'model'):
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError('Could not find ../model; specify --model-root')


def model_file(model, model_root=None):
    model = model_name_checked(model)
    target = _find_model_root(model_root) / '{}.h5'.format(model)
    if not target.is_file():
        raise FileNotFoundError('Safety model is missing: {}'.format(target))
    return str(target)


def _find_data_folder(model, data_root=None):
    here, cwd = Path(__file__).resolve().parent, Path.cwd()
    roots = ([Path(data_root).expanduser().resolve()] if data_root else
             [cwd.parent/'data', here.parent/'data', cwd/'data', here/'data'])
    candidates = []
    for root in roots:
        candidates.extend([root / model, root / 'safety' / model, root,
                           root / 'data' / model, root / 'data' / 'safety' / model])
    valid = [p.resolve() for p in candidates if (p/'counterexample.h5').is_file()]
    if not valid:
        raise FileNotFoundError('Cannot locate ../data/{}/counterexample.h5; checked:\n  {}\n'
                                'Use --data-root.'.format(model, '\n  '.join(str(p) for p in candidates)))
    return valid[0]


def _read_h5(path, key):
    if not path.is_file():
        raise FileNotFoundError('Safety split missing: {}'.format(path))
    with h5py.File(str(path), 'r') as file:
        if key not in file:
            raise KeyError('{} missing dataset key {!r}; available={}'.format(
                path, key, list(file.keys())))
        values = np.asarray(file[key][:], dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim < 2 or len(values) < 1:
        raise ValueError('Invalid safety matrix {}: {}'.format(path, values.shape))
    if not np.all(np.isfinite(values)):
        raise ValueError('Non-finite values in {}'.format(path))
    return values


def _select(data, n, seed):
    if n <= 0 or len(data) <= n:
        return data, np.arange(len(data), dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    ids = rng.choice(len(data), size=int(n), replace=False)
    return data[ids], ids


def _ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def load_safety_data(model, split='train', data_root=None, samples=0, seed=42):
    model = model_name_checked(model)
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if samples < 0:
        raise ValueError('samples must be >=0; use 0 for all available')
    folder = _find_data_folder(model, data_root)
    suffix = '' if split == 'train' else '_test'
    counter_path = folder / ('counterexample{}.h5'.format(suffix))
    draw_path = folder / ('drawndown{}.h5'.format(suffix))
    ct = _read_h5(counter_path, 'counterexample{}'.format(suffix))
    dr = _read_h5(draw_path, 'drawndown{}'.format(suffix))
    ct_total, dr_total = len(ct), len(dr)
    # Select each pool independently. Never reuse indices from a different size.
    ct, ct_ids = _select(ct, int(samples), int(seed)+17011)
    dr, dr_ids = _select(dr, int(samples), int(seed)+39017)
    if ct.shape[1:] != dr.shape[1:]:
        raise ValueError('Counterexample/drawdown feature dimensions disagree: {} vs {}'.format(
            ct.shape, dr.shape))
    info = {
        'model': model, 'split': split,
        'counterexample_file': str(counter_path.resolve()),
        'drawndown_file': str(draw_path.resolve()),
        'counterexample_total': int(ct_total), 'drawndown_total': int(dr_total),
        'counterexample_selected': int(len(ct)), 'drawndown_selected': int(len(dr)),
        'counterexample_index_sha256': _ids_hash(ct_ids),
        'drawndown_index_sha256': _ids_hash(dr_ids),
        'independent_subsample_seed': int(seed),
        'counterexample_and_drawndown_are_distinct_unpaired_pools': True,
    }
    return ct, dr, info


def safe_mask(model, predicted_classes):
    model = model_name_checked(model)
    pred = np.asarray(predicted_classes, dtype=np.int64).reshape(-1)
    if model == 'a19':
        return (pred != 3) & (pred != 4)
    if model == 'a29':
        return (pred == 0) | (pred == 1)
    return pred == 0  # a33, as in the original CFRepair safety definition


def prediction(model, images, session, batch_size=512):
    outputs = []
    for start in range(0, len(images), int(batch_size)):
        raw = session.run(model.output, {model.input: images[start:start+int(batch_size)]})
        if np.asarray(raw).ndim < 2:
            raise ValueError('Safety network must emit multiple ACAS output classes')
        outputs.append(np.argmin(raw, axis=-1).reshape(-1))
    if not outputs:
        raise ValueError('Empty safety evaluation pool')
    return np.concatenate(outputs)


def calculate_metrics(network, model_name, counterexamples, drawdown, session, batch_size=512):
    pred_ct = prediction(network, counterexamples, session, batch_size)
    pred_dr = prediction(network, drawdown, session, batch_size)
    unsafe_count = int(np.sum(~safe_mask(model_name, pred_ct)))
    preserved_count = int(np.sum(safe_mask(model_name, pred_dr)))
    return {
        'safety_violation_rate': float(unsafe_count / len(pred_ct)),
        'drawdown_retention': float(preserved_count / len(pred_dr)),
        'counterexample_violations': unsafe_count,
        'counterexample_count': int(len(pred_ct)),
        'drawdown_retained': preserved_count,
        'drawdown_count': int(len(pred_dr)),
        'metric_note': 'argmin output: a19 unsafe={3,4}; a29 unsafe=others than 0,1; a33 unsafe=others than 0',
    }


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
