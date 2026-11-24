"""Shared adversarial repair dataset reader and metrics (Python 3.7 + TF1 graph mode).

Source convention preserved from user-supplied adv_test_2.py:
- MNIST: ../data/adv/mnist/{train,test}/epsilon0.05.txt and label.txt
- CIFAR10: ../data/adv/cifar10/{train,test}/epsilon0.1.npy and label.npy
- CIFAR-10 preprocessing: x in [0,1] then (x - mean)/std.

Train and TEST are strictly separate; sample counts use reproducible independent
clean/adv selection because source adversarial arrays may not be paired to the
standard Keras clean split. Never invent a row-to-row alignment.
"""
import json
import os
from pathlib import Path
import numpy as np

CONFIG = {
    'mnist': {'input_shape': (28, 28, 1), 'threshold_acc': .97, 'threshold_adv_error': .05},
    'cifar10': {'input_shape': (32, 32, 3), 'threshold_acc': .75, 'threshold_adv_error': .20},
}
CIFAR_MEAN = np.array([.4914, .4822, .4465], dtype=np.float32)
CIFAR_STD = np.array([.247, .243, .261], dtype=np.float32)


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
    sess = tf.compat.v1.Session(config=cfg)
    if hasattr(K, 'set_session'):
        K.set_session(sess)
    else:
        tf.compat.v1.keras.backend.set_session(sess)
    return tf, K, sess


def _find_root(override, folder):
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError('{} folder does not exist: {}'.format(folder, p))
        return p
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    for p in (cwd.parent / folder, here.parent / folder, cwd / folder, here / folder):
        if p.is_dir():
            return p.resolve()
    raise FileNotFoundError('Cannot locate ../{}. Specify --{}-root'.format(folder, folder))


def model_file(model, model_root=None):
    root = _find_root(model_root, 'model')
    p = root / '{}_adv.h5'.format(model)
    if not p.is_file():
        raise FileNotFoundError('Adversarial pretrained model missing: {}'.format(p))
    return str(p)


def _clean_split(model, split):
    import tensorflow as tf
    if model == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif model == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError('Unsupported adversarial model: {}'.format(model))
    x, y = (x_train, y_train) if split == 'train' else (x_test, y_test)
    x = np.asarray(x, dtype=np.float32) / 255.0
    if model == 'mnist':
        x = x[..., None]
    else:
        x = normalize_cifar10(x)
    return x, np.asarray(y, dtype=np.int64).reshape(-1)


def normalize_cifar10(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - CIFAR_MEAN) / CIFAR_STD


def _read_labels(path):
    if path.suffix == '.npy':
        y = np.load(str(path), allow_pickle=False)
    else:
        y = np.loadtxt(str(path), delimiter=',', dtype=np.int64)
    return np.asarray(y, dtype=np.int64).reshape(-1)


def _choose(n_total, n_requested, seed):
    if n_total <= 0:
        raise ValueError('Dataset contains zero rows')
    if n_requested is None or int(n_requested) <= 0:
        return np.arange(n_total, dtype=np.int64)
    n = min(int(n_requested), n_total)
    rng = np.random.RandomState(int(seed))
    return rng.choice(n_total, size=n, replace=False)


def _load_mnist_txt(path, chosen, total_labels):
    """Read selected TXT rows without materializing a potentially enormous full file."""
    lookup = {int(row): pos for pos, row in enumerate(chosen)}
    result = np.empty((len(chosen), 784), dtype=np.float32)
    found = np.zeros(len(chosen), dtype=bool)
    line_count = 0
    with open(str(path), 'r', encoding='utf-8') as f:
        for row, line in enumerate(f):
            line_count += 1
            if row not in lookup:
                continue
            values = np.fromstring(line.strip(), sep=',', dtype=np.float32)
            if len(values) != 784:
                raise ValueError('{} line {}: expected 784 MNIST features, got {}'.format(
                    path, row+1, len(values)))
            result[lookup[row]] = values
            found[lookup[row]] = True
    if line_count != total_labels:
        raise ValueError('{} has {} rows; labels have {} rows'.format(path, line_count, total_labels))
    if not np.all(found):
        raise ValueError('Missing {} requested MNIST TXT rows'.format(int((~found).sum())))
    return result.reshape(-1, 28, 28, 1)


def _reshape_adv(model, raw):
    shape = CONFIG[model]['input_shape']
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 4 and raw.shape[1] == 3 and raw.shape[-1] != 3:
        raw = raw.transpose(0, 2, 3, 1)
    if raw.ndim in (2, 3, 4):
        expected_count = int(np.prod(shape))
        if np.prod(raw.shape[1:]) != expected_count:
            raise ValueError('ADV data feature size mismatch: got {}, expected {} for {}'.format(
                raw.shape, shape, model))
        return raw.reshape((-1,) + shape)
    raise ValueError('ADV data dimensions invalid: {}'.format(raw.shape))


def load_adv_and_clean(model, epsilon, split='train', data_root=None, samples=0, seed=42):
    """Return adv_X, adv_y, clean_X, clean_y, provenance.

    Both independent subsets are chosen with fixed seeds, and the same data are
    used by MOEA/D and MOPSO. Do NOT assume adv row i corresponds to clean row i.
    """
    from read_CGARepair import model_name_checked, epsilon_string
    model = model_name_checked(model)
    eps = epsilon_string(epsilon)
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    base = _find_root(data_root, 'data') / 'adv' / model / split
    name = 'epsilon{}'.format(eps)
    prefer_ext = ('.txt', '.npy') if model == 'mnist' else ('.npy', '.txt')
    candidates = [base / (name + ext) for ext in prefer_ext]
    present = [p for p in candidates if p.is_file()]
    if not present:
        raise FileNotFoundError('Adversarial {} file missing. Expected one of:\n  {}'.format(
            split, '\n  '.join(str(p) for p in candidates)))
    path = present[0]
    label_path = base / ('label.txt' if path.suffix == '.txt' else 'label.npy')
    if not label_path.is_file():
        raise FileNotFoundError('Adversarial {} labels missing: {}'.format(split, label_path))
    adv_labels_full = _read_labels(label_path)
    adv_ids = _choose(len(adv_labels_full), samples, seed + 27011)
    if path.suffix == '.txt':
        if model != 'mnist':
            raise ValueError('Only MNIST TXT reader supported. CIFAR-10 requires .npy as in original script.')
        adv_images = _load_mnist_txt(path, adv_ids, len(adv_labels_full))
    else:
        raw = np.load(str(path), allow_pickle=False, mmap_mode='r')
        if len(raw) != len(adv_labels_full):
            raise ValueError('{} has {} arrays but labels {} has {} rows'.format(
                path, len(raw), label_path, len(adv_labels_full)))
        adv_images = _reshape_adv(model, raw[adv_ids])
    raw_min, raw_max = float(np.min(adv_images)), float(np.max(adv_images))
    if raw_min < -.2 or raw_max > 1.2:
        print('[ADV data WARNING] {} raw range [{:.3f}, {:.3f}], but legacy scripts '
              'assume adversarial arrays already in [0,1]. Check normalization.'.format(
                  model, raw_min, raw_max), flush=True)
    if model == 'cifar10':
        # Exactly the old adv_test_2.py transformation; never normalize twice.
        adv_images = normalize_cifar10(adv_images)
    clean_images_all, clean_labels_all = _clean_split(model, split)
    clean_ids = _choose(len(clean_images_all), samples, seed + 49127)
    adv_y = adv_labels_full[adv_ids].astype(np.int64)
    clean_x = clean_images_all[clean_ids]
    clean_y = clean_labels_all[clean_ids]
    if not (len(adv_images) == len(adv_y) and len(clean_x) == len(clean_y)):
        raise RuntimeError('Adversarial/clean labels length mismatch')
    if not np.all(np.isfinite(adv_images)) or not np.all(np.isfinite(clean_x)):
        raise ValueError('Input contains nonfinite values')
    if np.any((adv_y < 0) | (adv_y > 9)) or np.any((clean_y < 0) | (clean_y > 9)):
        raise ValueError('Labels must be class indices 0..9')
    provenance = {
        'split': split, 'model': model, 'epsilon': eps,
        'adv_file': str(path.resolve()), 'adv_labels_file': str(label_path.resolve()),
        'adv_total': int(len(adv_labels_full)), 'adv_selected': int(len(adv_y)),
        'clean_total': int(len(clean_labels_all)), 'clean_selected': int(len(clean_y)),
        'raw_adv_range_before_cifar_standardization': [raw_min, raw_max],
        'adv_and_clean_rows_are_not_assumed_pairwise_aligned': True,
        'independent_subsample_seed': int(seed),
        'preprocessing': ('mnist_raw_as_in_original_adv_test' if model == 'mnist'
                          else 'cifar10_(x-[.4914,.4822,.4465])/[.247,.243,.261]'),
    }
    return adv_images, adv_y, clean_x, clean_y, provenance


def predict_labels(model, images, sess, batch_size=512):
    predictions = []
    for i in range(0, len(images), int(batch_size)):
        output = sess.run(model.output, {model.input: images[i:i+int(batch_size)]})
        predictions.append(np.argmax(np.asarray(output), axis=-1).reshape(-1))
    if not predictions:
        raise ValueError('No model input samples')
    return np.concatenate(predictions)


def calculate_metrics(model, adv_x, adv_y, clean_x, clean_y, session, batch_size=512):
    if not (len(adv_x) == len(adv_y) and len(clean_x) == len(clean_y)):
        raise ValueError('X/Y mismatch for ADV or clean subset')
    pred_adv = predict_labels(model, adv_x, session, batch_size)
    pred_clean = predict_labels(model, clean_x, session, batch_size)
    robust = float(np.mean(pred_adv == adv_y))
    acc = float(np.mean(pred_clean == clean_y))
    return {
        'adv_robustness': robust,
        'adv_error_rate': float(1.0-robust),
        'clean_accuracy': acc,
        'adv_correct': int(np.sum(pred_adv == adv_y)),
        'adv_samples': int(len(adv_y)),
        'clean_correct': int(np.sum(pred_clean == clean_y)),
        'clean_samples': int(len(clean_y)),
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
