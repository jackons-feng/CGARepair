"""Strict split resolver for the user's NSrepair adversarial benchmarks."""
from __future__ import print_function
import hashlib
import json
from pathlib import Path
import numpy as np


def root_from_script():
    return Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def ensure(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def epsilon_tag(eps):
    return 'epsilon{:g}'.format(float(eps))


def load_array(path):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(str(path))
    try:
        return np.loadtxt(str(path), dtype=np.float32, delimiter=',')
    except ValueError:
        return np.loadtxt(str(path), dtype=np.float32)


def clean_split(dataset, split, tf):
    if dataset == 'mnist':
        train, test = tf.keras.datasets.mnist.load_data()
        x, y = train if split == 'train' else test
        return x.astype('float32')[..., None] / 255.0, y.astype('int64').reshape(-1)
    if dataset == 'cifar10':
        train, test = tf.keras.datasets.cifar10.load_data()
        x, y = train if split == 'train' else test
        arr = x.astype('float32') / 255.0
        means = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        stds = np.array([0.247, 0.243, 0.261], dtype=np.float32)
        return (arr - means) / stds, y.astype('int64').reshape(-1)
    raise ValueError(dataset)


def adv_split(root, dataset, split, eps):
    if split not in ('train', 'test'):
        raise ValueError('Only explicit TRAIN/TEST splits supported')
    root = Path(root)
    folders = [root / 'data' / 'adv' / dataset / split,
               root / 'data' / dataset / split]
    filename = epsilon_tag(eps)
    attempts = [(d / (filename + ext), d / ('label' + label_ext))
                for d in folders
                for ext in ('.npy', '.txt')
                for label_ext in ('.npy', '.txt')]
    pairs = [(a, b) for a, b in attempts if a.is_file() and b.is_file()]
    if not pairs:
        raise FileNotFoundError('No adversarial {} input/label pair for {} eps={} under {}'.format(
            split, dataset, eps, folders))
    # Canonical selection is fixed; no silent comparison of different sources.
    source, label = pairs[0]
    x = load_array(source).astype(np.float32)
    y = load_array(label).astype(np.int64).reshape(-1)
    if dataset == 'mnist':
        x = x.reshape((-1, 28, 28, 1))
        if x.min() < -0.001 or x.max() > 1.001:
            raise ValueError('MNIST adv should be [0,1] already: min/max={}/{}'.format(x.min(),x.max()))
        transform = 'already_in_0_1'
    else:
        x = x.reshape((-1, 32, 32, 3))
        amin, amax = float(x.min()), float(x.max())
        if -0.001 <= amin and amax <= 1.001:
            x = (x - np.array([.4914,.4822,.4465],np.float32)) / np.array([.247,.243,.261],np.float32)
            transform = 'raw_0_1_standardized_using_CGAREPAIR_CIFAR'
        elif -3.5 <= amin <= -0.1 and 0.1 <= amax <= 3.5:
            transform = 'already_standardized'
        else:
            raise ValueError('CIFAR adversarial range unrecognized; inspect before proceeding: {}..{}'.format(amin,amax))
    if len(x) != len(y):
        raise ValueError('Adversarial x/y length mismatch: {} vs {}'.format(len(x), len(y)))
    if len(x) == 0 or np.any(~np.isfinite(x)):
        raise ValueError('Empty/nonfinite adversarial input')
    return x, y, {'source_path': str(source.resolve()), 'label_path': str(label.resolve()),
                  'source_sha256': sha256(source), 'label_sha256': sha256(label),
                  'split': split, 'epsilon': float(eps), 'preprocess': transform, 'n':len(x)}


def hash_model(path):
    p=ensure(path)
    return {'original_model':str(p.resolve()),'original_model_sha256':sha256(p)}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(str(path),'w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
