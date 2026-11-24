"""Shared backdoor data, trigger and metrics for CGARepair optimization and testing.

Data scale follows the user's original backdoor_moo_2.py/backdoor_test_2.py:
MNIST/Fashion in [0,1]; GTSRB train/test pickle multiplied by 255 after resize.
The clean-vs-triggered evaluation pair is kept aligned.
"""
import json
import os
import pickle
from pathlib import Path
import numpy as np

DATASET_CONFIG = {
    'mnist': {'input_shape': (28, 28, 1), 'num_classes': 10, 'target_label': 3,
              'threshold_acc': 0.9858, 'threshold_sr': 0.01},
    'fashion': {'input_shape': (28, 28, 1), 'num_classes': 10, 'target_label': 3,
                'threshold_acc': 0.923, 'threshold_sr': 0.01},
    'gtsrb': {'input_shape': (32, 32, 3), 'num_classes': 43, 'target_label': 33,
              'threshold_acc': 0.92, 'threshold_sr': 0.01},
}


def initialize_tf(seed=42):
    import tensorflow as tf
    import tensorflow.python.keras.backend as K
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass
    np.random.seed(int(seed))
    try:
        tf.random.set_seed(int(seed))
    except AttributeError:
        tf.compat.v1.set_random_seed(int(seed))
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=cfg)
    if hasattr(K, 'set_session'):
        K.set_session(sess)
    else:
        from tensorflow.compat.v1.keras import backend as K1
        K1.set_session(sess)
    return tf, K, sess


def _root(custom, folder):
    if custom:
        p = Path(custom).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError('{} root not found: {}'.format(folder, p))
        return p
    base = Path(__file__).resolve().parent
    for p in (Path.cwd().parent / folder, base.parent / folder, Path.cwd() / folder,
              base / folder):
        if p.is_dir():
            return p.resolve()
    raise FileNotFoundError('Cannot locate {} root. Specify --{}-root'.format(folder, folder))


def model_file(model_name, resnet=False, model_root=None):
    root = _root(model_root, 'model')
    filename = '{}{}.h5'.format(model_name, '_resnet' if resnet else '')
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError('Cannot load model checkpoint: {}'.format(path))
    return str(path)


def load_dataset(model_name, split='train', data_root=None, session=None):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if model_name in ('mnist', 'fashion'):
        import tensorflow as tf
        if model_name == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x, y = (x_train, y_train) if split == 'train' else (x_test, y_test)
        return np.expand_dims(x.astype(np.float32) / 255.0, -1), np.asarray(y, dtype=np.int64)
    if model_name == 'gtsrb':
        root = _root(data_root, 'data')
        file_path = root / 'gtsrb' / ('train_data.pkl' if split == 'train' else 'test_data.pkl')
        if not file_path.is_file():
            raise FileNotFoundError('Missing GTSRB {} pickle: {}'.format(split, file_path))
        with open(file_path, 'rb') as f:
            saved = pickle.load(f)
        x = np.asarray(saved['data'])
        y = np.asarray(saved['label'], dtype=np.int64).reshape(-1)
        if x.ndim != 4:
            raise ValueError('GTSRB pickle image shape must be NCHW/NHWC; got {}'.format(x.shape))
        if x.shape[1] == 3 and x.shape[-1] != 3:
            x = x.transpose((0, 2, 3, 1))
        elif x.shape[-1] != 3:
            raise ValueError('GTSRB channels could not be determined from {}'.format(x.shape))
        if session is None:
            raise ValueError('TF session must be supplied for GTSRB resize')
        import tensorflow as tf
        result = []
        for start in range(0, len(x), 2048):
            batch = x[start:start + 2048].astype(np.float32)
            if batch.shape[1:3] != (32, 32):
                # Legacy training/test scripts resized the 48x48 source to 32x32.
                resized = session.run(tf.image.resize(batch, (32, 32)))
            else:
                resized = batch
            result.append((resized * 255.0).astype(np.float32))
        return np.concatenate(result, axis=0), y
    raise ValueError('Unknown backdoor dataset: {}'.format(model_name))


def trigger_inputs(x, model_name, data_root=None):
    """Apply the original mask/pattern backdoor from ../data/img, keeping shape and scale."""
    from tensorflow.keras.preprocessing import image
    root = _root(data_root, 'data') / 'img'
    cfg = DATASET_CONFIG[model_name]
    target = cfg['target_label']
    h, w, channels = cfg['input_shape']
    mask_path = root / '{}_visualize_mask_label_{}.png'.format(model_name, target)
    pattern_path = root / '{}_visualize_pattern_label_{}.png'.format(model_name, target)
    for p in (mask_path, pattern_path):
        if not p.is_file():
            raise FileNotFoundError('Missing backdoor trigger file: {}'.format(p))
    mask = image.img_to_array(image.load_img(str(mask_path), color_mode='grayscale',
                                                 target_size=(h, w))).astype(np.float32)[..., 0] / 255.0
    pattern = image.img_to_array(image.load_img(str(pattern_path), color_mode='rgb',
                                                    target_size=(h, w))).astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4 or tuple(x.shape[1:]) != cfg['input_shape']:
        raise ValueError('Model input shape mismatch: expected {}, got {}'.format(cfg['input_shape'], x.shape))
    if channels == 1:
        # Matches original backdoor_moo_2.py's green-channel trigger for grayscale.
        patched = x * (1.0 - mask[None, ..., None])
        patched += (pattern[..., 1] / 255.0 * mask)[None, ..., None]
    else:
        patched = x * (1.0 - mask[None, ..., None]) + (pattern * mask[..., None])[None, ...]
    return patched.astype(np.float32)


def select_subset(clean, labels, poisoned, n, seed):
    n = min(int(n), len(clean))
    if n < 1:
        raise ValueError('Evaluation subset would be empty')
    rng = np.random.RandomState(int(seed))
    ids = rng.choice(len(clean), size=n, replace=False)
    return clean[ids], labels[ids], poisoned[ids]


def predict_labels(model, x, session, batch_size=512):
    preds = []
    for start in range(0, len(x), int(batch_size)):
        logits = session.run(model.output, feed_dict={model.input: x[start:start+batch_size]})
        logits = np.asarray(logits)
        preds.append(np.argmax(logits, axis=-1))
    return np.concatenate(preds).reshape(-1)


def calculate_metrics(model, clean, labels, poisoned, target, session, sr_definition='legacy', batch_size=512):
    labels = np.asarray(labels).reshape(-1)
    if not (len(clean) == len(poisoned) == len(labels)):
        raise ValueError('clean/label/triggered lengths differ')
    clean_preds = predict_labels(model, clean, session, batch_size)
    poisoned_preds = predict_labels(model, poisoned, session, batch_size)
    mask = labels != target
    target_count = int(np.sum((poisoned_preds == target) & mask))
    old_sr = target_count / float(len(labels))
    standard_asr = target_count / float(np.sum(mask)) if np.any(mask) else None
    if sr_definition == 'standard':
        if standard_asr is None:
            raise ValueError('No original non-target class examples in evaluation subset')
        sr = standard_asr
    elif sr_definition == 'legacy':
        sr = old_sr
    else:
        raise ValueError('Unknown SR definition: {}'.format(sr_definition))
    return {
        'sr': float(sr), 'clean_accuracy': float(np.mean(clean_preds == labels)),
        'legacy_sr': float(old_sr),
        'standard_asr': float(standard_asr) if standard_asr is not None else None,
        'non_target_examples': int(np.sum(mask)),
        'target_successes': target_count,
        'n_examples': int(len(labels)),
    }


def save_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
