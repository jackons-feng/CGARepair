"""Shared fairness data/metrics/model-weight utilities for CGARepair + MOEA/D.

Faithful to the original fairness target: two-group *accuracy gap*, not EO.
The TEST loader reads only an explicit full TEST split; never a train/eval fallback.
Python 3.7 and TF graph mode compatible; TensorFlow is imported lazily.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

SUPPORTED_DATASETS = ('bank', 'adult', 'default')


def check_dataset(value):
    name = str(value).strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError('Fairness dataset must be bank, adult or default; got {}. '
                         'The original census/credit/lfw branch is not the verified '
                         'adult/default full-test protocol.'.format(value))
    return name


def root_dir(configured, relative):
    if configured:
        return Path(configured).expanduser().resolve()
    here = Path(__file__).resolve().parent
    # Designed to work after placing scripts under fairness/.
    return (here / relative).resolve()


def model_path_for(dataset, model_root=None):
    root = root_dir(model_root, '../model')
    filename = root / (check_dataset(dataset) + '.h5')
    if not filename.is_file():
        raise FileNotFoundError('Model not found: {}. Use --model-root.'.format(filename))
    return filename


def _first_hdf5_dataset(group):
    for key in sorted(group.keys()):
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            return obj[:]
        if isinstance(obj, h5py.Group):
            value = _first_hdf5_dataset(obj)
            if value is not None:
                return value
    return None


def read_h5(path, preferred):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing required FAIRNESS data file: {}. '
                                'Training cannot fall back to TEST data.'.format(path))
    with h5py.File(str(path), 'r') as file:
        for name in preferred:
            if name in file:
                value = file[name]
                if isinstance(value, h5py.Dataset):
                    return np.asarray(value[:])
                if isinstance(value, h5py.Group):
                    nested = _first_hdf5_dataset(value)
                    if nested is not None:
                        return np.asarray(nested)
        value = _first_hdf5_dataset(file)
        if value is None:
            raise ValueError('No HDF5 array found in {}'.format(path))
        return np.asarray(value)


def normalize_groups(groups):
    """Match CGARepair's official fairness group convention, not Y-as-G."""
    g = np.asarray(groups).reshape(-1).astype(np.int64).copy()
    if g.size == 0:
        raise ValueError('Sensitive group array is empty')
    if int(np.max(g)) > 1:
        g[g == 1] = 0
        g[g >= 2] = 1
    unique = np.unique(g)
    if not np.array_equal(unique, [0, 1]):
        raise ValueError('Need two nonempty sensitive groups {0,1}, got {}. '
                         'Check g_train/g_test encoding.'.format(unique.tolist()))
    return g


def _select_balanced_indices(y, g, count, seed):
    """Exactly count samples, equal group quotas, label proportions stratified in each group."""
    n = len(y)
    if count is None or int(count) <= 0 or int(count) >= n:
        return np.arange(n, dtype=np.int64)
    count = int(count)
    if count < 2:
        raise ValueError('Fairness sample count must be >=2')
    rng = np.random.RandomState(int(seed))
    # If a group is too small for exactly half the budget, include all of it
    # and assign the remaining slots to the other group, without replacement.
    cap0 = int(np.sum(g == 0))
    cap1 = int(np.sum(g == 1))
    q0 = min(count // 2, cap0)
    q1 = min(count - q0, cap1)
    if q1 < count - q0:
        q0 += count - q0 - q1
    if q0 > cap0 or q1 > cap1 or q0 + q1 != count:
        raise ValueError('Cannot select {} train samples from groups ({} and {})'.format(
            count, cap0, cap1))
    quotas = (q0, q1)
    sampled = []
    for gv, target in zip((0, 1), quotas):
        inds = np.flatnonzero(g == gv)
        if len(inds) < target:
            raise ValueError('Insufficient train group {} samples: have {}, need {}'.format(
                gv, len(inds), target))
        # Stratify by true labels inside each sensitive group.
        labels, cls_counts = np.unique(y[inds], return_counts=True)
        expected = target * cls_counts.astype(float) / float(len(inds))
        parts = np.floor(expected).astype(int)
        leftovers = target - int(np.sum(parts))
        order = np.argsort(-(expected - parts), kind='mergesort')
        for i in order:
            if leftovers <= 0:
                break
            if parts[i] < cls_counts[i]:
                parts[i] += 1
                leftovers -= 1
        if int(np.sum(parts)) != target:
            raise RuntimeError('Internal label-stratified allocation failed')
        for label, need in zip(labels, parts):
            if need:
                cls_inds = inds[y[inds] == label]
                sampled.append(rng.choice(cls_inds, size=int(need), replace=False))
    selected = np.concatenate(sampled).astype(np.int64)
    rng.shuffle(selected)
    if len(selected) != count:
        raise RuntimeError('Requested {} but selected {} train records'.format(count, len(selected)))
    return selected


def load_fairness_split(dataset, split, data_root=None, sample_count=0, sample_seed=42):
    ds = check_dataset(dataset)
    if split not in ('train', 'test'):
        raise ValueError('Split must be train or test; got {}'.format(split))
    root = root_dir(data_root, '../data/fairness') / ds
    paths = {'x': root / ('x_' + split + '.h5'),
             'y': root / ('y_' + split + '.h5'),
             'g': root / ('g_' + split + '.h5')}
    x = read_h5(paths['x'], ('x', 'X')).astype('float32')
    y = read_h5(paths['y'], ('y', 'Y'))
    if y.ndim > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    y = y.reshape(-1).astype(np.int64)
    g = normalize_groups(read_h5(paths['g'], ('g', 'G')))
    if not (len(x) == len(y) == len(g)) or len(x) == 0:
        raise ValueError('{} split lengths disagree: X={}, Y={}, G={}'.format(
            split, len(x), len(y), len(g)))
    original_count = int(len(x))
    selected = np.arange(original_count, dtype=np.int64)
    if split == 'train':
        selected = _select_balanced_indices(y, g, sample_count, sample_seed)
        x, y, g = x[selected], y[selected], g[selected]
    elif sample_count not in (None, 0):
        raise ValueError('TEST must use complete test split (sample_count=0); no test sampling.')
    groups, counts = np.unique(g, return_counts=True)
    if not np.array_equal(groups, [0, 1]):
        raise ValueError('Both groups required in sampled {} set'.format(split))
    selection_hash = hashlib.sha256(np.ascontiguousarray(selected).tobytes()).hexdigest()
    manifest = {'split': split, 'dataset': ds,
                'original_count': original_count, 'selected_count': int(len(x)),
                'selection_seed': int(sample_seed) if split == 'train' else None,
                'selection_index_sha256': selection_hash,
                'group_counts': {str(int(k)): int(v) for k, v in zip(groups, counts)},
                'paths': {k: str(v.resolve()) for k, v in paths.items()},
                'labels_used_as_groups': False}
    return x, y, g, manifest


def set_up_tf_graph():
    import tensorflow as tf
    import tensorflow.python.keras.backend as K
    tf.compat.v1.disable_eager_execution()
    options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=options))
    K.set_session(session)
    return tf, K, session


def model_predictions(model, x, session, batch_size=1024):
    x = np.asarray(x, dtype='float32')
    if len(x) == 0:
        raise ValueError('Cannot evaluate empty fairness dataset')
    result = []
    try:
        import tensorflow.python.keras.backend as K
        lp = K.learning_phase()
        feed_lp = not isinstance(lp, (int, np.integer))
    except Exception:
        feed_lp = False
        lp = None
    for start in range(0, len(x), int(batch_size)):
        stop = min(len(x), start + int(batch_size))
        feed = {model.input: x[start:stop]}
        if feed_lp:
            feed[lp] = 0
        result.append(session.run(model.output, feed_dict=feed))
    outputs = np.concatenate(result, axis=0)
    if outputs.ndim > 1 and outputs.shape[-1] > 1:
        return np.argmax(outputs, axis=-1).reshape(-1).astype(np.int64)
    return (outputs.reshape(-1) > 0.5).astype(np.int64)


def fairness_metrics(preds, y, g):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    g = normalize_groups(g)
    preds = np.asarray(preds).reshape(-1).astype(np.int64)
    if not (len(preds) == len(y) == len(g)):
        raise ValueError('Pred/Y/G lengths differ')
    acc = float(np.mean(preds == y))
    group_acc = [float(np.mean(preds[g == idx] == y[g == idx])) for idx in (0, 1)]
    return {'fairness_gap': float(abs(group_acc[1] - group_acc[0])),
            'accuracy': acc,
            'group_accuracies': {'0': group_acc[0], '1': group_acc[1]},
            'signed_accuracy_gap': float(group_acc[1] - group_acc[0]),
            'group_counts': {'0': int(np.sum(g == 0)), '1': int(np.sum(g == 1))},
            'sample_count': int(len(y))}


def macro_or_binary_f1(preds, y):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(preds).reshape(-1)
    classes = np.unique(np.concatenate([y, p]))
    labels = [1] if set(classes.tolist()).issubset({0, 1}) else classes
    per_class = []
    for cls in labels:
        tp = np.sum((p == cls) & (y == cls))
        fp = np.sum((p == cls) & (y != cls))
        fn = np.sum((p != cls) & (y == cls))
        denom = 2 * tp + fp + fn
        per_class.append(float(2.0 * tp / float(denom)) if denom else 0.0)
    return float(np.mean(per_class))


def evaluate_model(model, x, y, g, session, batch_size=1024, include_f1=False):
    preds = model_predictions(model, x, session, batch_size)
    metrics = fairness_metrics(preds, y, g)
    if include_f1:
        metrics['f1'] = macro_or_binary_f1(preds, y)
    return metrics


def build_neuron_specs(model, players):
    """Resolve actual model.layers indices, counting all BN arrays correctly."""
    base = model.get_weights()
    offsets, index = {}, 0
    for layer_idx, layer in enumerate(model.layers):
        local = layer.get_weights()
        if local:
            offsets[layer_idx] = index
            index += len(local)
    if index != len(base):
        raise RuntimeError('Model weight layout inconsistent')
    specs, vectors, offset = [], [], 0
    touched = set()
    import re
    pat = re.compile(r'^(Conv\w*|Dense)_(\d+)_(\d+)$', re.IGNORECASE)
    for name in players:
        m = pat.fullmatch(str(name))
        if not m:
            raise ValueError('Invalid CGARepair neuron {} (expect Dense_5_12)'.format(name))
        family, layer_idx, neuron_idx = m.group(1), int(m.group(2)), int(m.group(3))
        if layer_idx >= len(model.layers) or layer_idx not in offsets:
            raise ValueError('Neuron {} references invalid/unweighted model layer'.format(name))
        layer = model.layers[layer_idx]
        class_name = type(layer).__name__
        if class_name.lower() != family.lower():
            raise ValueError('Neuron {} layer mismatch: model.layers[{}] is {}; '
                             'do not renumber CGARepair layer IDs'.format(name, layer_idx, class_name))
        weight_idx = offsets[layer_idx]
        kernel = base[weight_idx]
        if kernel.ndim < 2 or neuron_idx >= kernel.shape[-1]:
            raise ValueError('Neuron {} invalid output-channel dimension {}'.format(name, kernel.shape))
        if (weight_idx, neuron_idx) in touched:
            raise ValueError('Duplicate kernel output channel: {}'.format(name))
        touched.add((weight_idx, neuron_idx))
        channel = np.array(kernel[..., neuron_idx], copy=True)
        n = int(channel.size)
        specs.append({'player': str(name), 'model_layer_idx': int(layer_idx),
                      'layer_name': layer.name, 'kernel_weight_idx': int(weight_idx),
                      'channel_idx': int(neuron_idx), 'shape': list(channel.shape),
                      'slice_start': offset, 'slice_end': offset + n})
        vectors.append(channel.reshape(-1))
        offset += n
    if not vectors:
        raise ValueError('No selected repair neurons')
    return specs, np.concatenate(vectors).astype(np.float64), base


def apply_vector(weights, specs, vector):
    for spec in specs:
        start, stop = spec['slice_start'], spec['slice_end']
        channel = np.asarray(vector[start:stop]).reshape(tuple(spec['shape']))
        weights[spec['kernel_weight_idx']][..., spec['channel_idx']] = channel


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
