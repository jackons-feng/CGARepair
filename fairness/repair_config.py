import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

# The project uses a single utils.py with layer/weight helpers; it does not
# provide a utils.data package. Fairness data are therefore loaded directly
# from the dataset H5/NPY files below.
from MTL import PCGrad, MGDA, CAGrad, NashMTL, GradNorm, RLW, DWA

BATCH_SIZE = 256

# TF1 graph mode — guard against multiple imports
try:
    tf.compat.v1.disable_eager_execution()
except RuntimeError:
    pass  # already disabled
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

# ============================================================
# Metric helpers
# ============================================================

DATASET_CONFIG = {
    'bank': {
        'input_shape': None,
        'num_classes': 2,
        'target_label': 1,
    },
    'adult': {
        'input_shape': None,
        'num_classes': 2,
        'target_label': 1,
    },
    'default': {
        'input_shape': None,
        'num_classes': 2,
        'target_label': 1,
    },
    # Backward-compatible alias for older experiments that used "census".
    'census': {
        'input_shape': None,
        'num_classes': 2,
        'target_label': 1,
    },
}

SUPPORTED_FAIRNESS_DATASETS = tuple(DATASET_CONFIG.keys())

# Compatibility alias: older experiments may call Adult as 'census'.
DATASET_DIR_ALIASES = {
    'bank': 'bank',
    'adult': 'adult',
    'default': 'default',
    'census': 'adult',
}


def labels_from_output(output):
    """Match the original fairness scripts' prediction rule.

    Bank/Adult/Default fairness models commonly use a single sigmoid output,
    where labels are
    produced by thresholding at 0.5. Multi-output classifiers keep argmax.
    """
    output = np.asarray(output)
    if output.ndim == 1 or (output.ndim >= 2 and output.shape[-1] == 1):
        return (output.reshape(-1) > 0.5).astype(int)
    return np.argmax(output, axis=-1).astype(int)


def normalize_binary_groups(groups):
    """Normalize sensitive group labels to the 0/1 convention used upstream."""
    groups = np.asarray(groups).reshape(-1).astype(int)
    if groups.size and np.max(groups) > 1:
        groups = groups.copy()
        groups[groups == 1] = 0
        groups[groups == 2] = 1
    return groups


def binary_group_accuracy_gap(pred_labels, labels, group_labels):
    labels = np.asarray(labels).reshape(-1).astype(int)
    pred_labels = np.asarray(pred_labels).reshape(-1).astype(int)
    groups = normalize_binary_groups(group_labels)
    mask0 = (groups == 0)
    mask1 = (groups == 1)
    if np.sum(mask0) == 0 or np.sum(mask1) == 0:
        return 0.0
    acc0 = np.mean(pred_labels[mask0] == labels[mask0])
    acc1 = np.mean(pred_labels[mask1] == labels[mask1])
    return float(abs(acc1 - acc0))

def value_accuracy(model, images, labels):
    output = sess_run(model, model.output, images)
    pred_labels = labels_from_output(output)
    labels = np.asarray(labels).reshape(-1).astype(int)
    return float(np.mean(pred_labels == labels))

def value_fairness_gap(model, x, y, group_labels=None, dataset=None):
    """Compute |Acc(G=1) - Acc(G=0)|, matching value_fairness in CFRepair."""
    output = sess_run(model, model.output, x)
    pred = labels_from_output(output)
    groups = y if group_labels is None else group_labels
    return binary_group_accuracy_gap(pred, y, groups)


def evaluate_model(model, test_data):
    return {
        'clean_acc': float(value_accuracy(model, test_data['x_test'], test_data['y_test'])),
        'fairness_gap': float(value_fairness_gap(
            model, test_data['x_test'], test_data['y_test'],
            test_data.get('g_test'), test_data.get('dataset'))),
    }

def _extract_xyg(split):
    """Normalize common fairness loader outputs to (x, y, group).

    Supported formats:
      - (x, y)
      - (x, y, group)
      - {'x'/'data'/'features': ..., 'y'/'label'/'labels': ...}
    """
    g = None
    if isinstance(split, dict):
        x = None
        y = None
        for key in ('x', 'data', 'features', 'X'):
            if key in split:
                x = split[key]
                break
        for key in ('y', 'label', 'labels', 'Y'):
            if key in split:
                y = split[key]
                break
        for key in ('g', 'group', 'groups', 'sensitive', 'sensitive_attr', 'protected', 'A'):
            if key in split:
                g = split[key]
                break
        if x is None or y is None:
            raise ValueError(f'Cannot parse fairness split keys: {list(split.keys())}')
    elif isinstance(split, (tuple, list)) and len(split) >= 2:
        x, y = split[0], split[1]
        if len(split) >= 3:
            g = split[2]
    else:
        raise ValueError(f'Unsupported fairness split format: {type(split)}')

    x = np.asarray(x).astype('float32')
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        y = np.argmax(y, axis=-1)
    y = y.reshape(-1).astype(int)
    if g is None:
        g = y
    g = normalize_binary_groups(g)
    return x, y, g


def _load_npy_split(base_dir, split):
    candidates = [
        (
            os.path.join(base_dir, f'{split}_x.npy'),
            os.path.join(base_dir, f'{split}_y.npy'),
            os.path.join(base_dir, f'{split}_g.npy'),
        ),
        (
            os.path.join(base_dir, split, 'data.npy'),
            os.path.join(base_dir, split, 'label.npy'),
            os.path.join(base_dir, split, 'group.npy'),
        ),
        (
            os.path.join(base_dir, split, 'x.npy'),
            os.path.join(base_dir, split, 'y.npy'),
            os.path.join(base_dir, split, 'g.npy'),
        ),
    ]
    group_names = ('group.npy', 'groups.npy', 'sensitive.npy', 'sensitive_attr.npy', 'protected.npy')
    for x_path, y_path, g_path in candidates:
        if os.path.exists(x_path) and os.path.exists(y_path):
            if not os.path.exists(g_path):
                split_dir = os.path.dirname(x_path)
                for name in group_names:
                    cand = os.path.join(split_dir, name)
                    if os.path.exists(cand):
                        g_path = cand
                        break
            x = np.load(x_path)
            y = np.load(y_path)
            g = np.load(g_path) if os.path.exists(g_path) else y
            return x, y, g
    return None


def _read_h5_array(path):
    import h5py
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f'Empty h5 file: {path}')
        # Most saved Keras/HDF5 arrays contain a single dataset. If groups are
        # present, descend once and take the first dataset found.
        obj = f[keys[0]]
        if hasattr(obj, 'shape'):
            return np.asarray(obj)
        for key in obj.keys():
            child = obj[key]
            if hasattr(child, 'shape'):
                return np.asarray(child)
    raise ValueError(f'Cannot find dataset in h5 file: {path}')


def _load_h5_split(base_dir, split):
    x_path = os.path.join(base_dir, f'x_{split}.h5')
    y_path = os.path.join(base_dir, f'y_{split}.h5')
    g_path = os.path.join(base_dir, f'g_{split}.h5')
    if os.path.exists(x_path) and os.path.exists(y_path):
        x = _read_h5_array(x_path)
        y = _read_h5_array(y_path)
        g = _read_h5_array(g_path) if os.path.exists(g_path) else y
        return x, y, g
    return None


def _call_fairness_loader(model_name):
    """Load complete train/test fairness splits directly from local files.

    The project-level utils.py only contains layer/weight helpers, so this
    function intentionally does not import ``utils.data``. Supported layouts:

      ../data/fairness/<dataset>/x_train.h5, y_train.h5, g_train.h5
      ../data/fairness/<dataset>/x_test.h5,  y_test.h5,  g_test.h5

    Equivalent paths under ./data and NPY layouts remain supported.
    """
    model_name = str(model_name).lower()
    if model_name not in SUPPORTED_FAIRNESS_DATASETS:
        supported = ', '.join(SUPPORTED_FAIRNESS_DATASETS)
        raise ValueError(
            f'Unsupported fairness dataset: {model_name}. Use one of: {supported}.'
        )

    dataset_dir = DATASET_DIR_ALIASES.get(model_name, model_name)
    searched = []
    for base in (
        os.path.join('data', 'fairness', dataset_dir),
        os.path.join('..', 'data', 'fairness', dataset_dir),
        os.path.join('data', dataset_dir),
        os.path.join('..', 'data', dataset_dir),
    ):
        searched.append(base)
        train = _load_h5_split(base, 'train')
        if train is None:
            train = _load_npy_split(base, 'train')
        test = _load_h5_split(base, 'test')
        if test is None:
            test = _load_npy_split(base, 'test')
        if train is not None and test is not None:
            print(f'[repair_config] Loaded fairness data for {model_name} from: {base}')
            return train, test

    raise FileNotFoundError(
        'Cannot find complete fairness train/test data for {}. Searched:\n  - {}\n'
        'Expected x_train/y_train/g_train and x_test/y_test/g_test H5 files.'.format(
            model_name, '\n  - '.join(searched)
        )
    )


def _sample_xyg(x, y, g, sample_size, seed):
    """Deterministically sample the same indices from x/y/g."""
    if sample_size is None or sample_size >= len(x):
        return x, y, g
    if sample_size <= 0:
        raise ValueError(f'sample_size must be positive or None, got {sample_size}')
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), int(sample_size), replace=False)
    return x[idx], y[idx], g[idx]


def load_fairness_data(model_name='bank', split=None, sample_size=1024, seed=42, return_groups=False):
    model_name = str(model_name).lower()
    train_raw, test_raw = _call_fairness_loader(model_name)
    train = _extract_xyg(train_raw)
    test = _extract_xyg(test_raw)

    if split in ('train', 'training'):
        x, y, g = _sample_xyg(*train, sample_size=sample_size, seed=seed)
        if return_groups:
            return x, y, g
        return x, y

    if split in ('test', 'testing'):
        x, y, g = _sample_xyg(*test, sample_size=sample_size, seed=seed)
        if return_groups:
            return x, y, g
        return x, y

    # Backward-compatible behavior when split is omitted:
    # keep the complete training split and optionally sample only the test split.
    sampled_test = _sample_xyg(*test, sample_size=sample_size, seed=seed)
    if return_groups:
        return train, sampled_test
    return (train[0], train[1]), (sampled_test[0], sampled_test[1])


def sess_run(model, variable, images, batch_size=BATCH_SIZE):
    output = []
    num_batches = int(np.ceil(len(images) / batch_size))
    for batch in range(num_batches):
        batch_idxs = np.arange(batch * batch_size, min((batch + 1) * batch_size, len(images)))
        input_dic = {model.input: images[batch_idxs]}
        output.append(K.get_session().run(variable, input_dic))
    try:
        return np.concatenate(output, 0)
    except Exception:
        return np.array(output)


def value(model, images, labels, group_labels=None, metric='accuracy', dataset=None):
    output = sess_run(model, model.output, images)
    pred_labels = labels_from_output(output)
    labels = np.asarray(labels).reshape(-1).astype(int)

    if metric == 'accuracy':
        return float(np.mean(pred_labels == labels))
    groups = labels if group_labels is None else group_labels
    return binary_group_accuracy_gap(pred_labels, labels, groups)


def get_layer_kernel_indices(model, trainable):
    var_index = {id(v): idx for idx, v in enumerate(trainable)}
    conv_meta, dense_meta = [], []
    for layer in model.layers:
        lname = type(layer).__name__
        if 'Conv' not in lname and 'Dense' not in lname:
            continue
        if len(layer.trainable_weights) == 0:
            continue
        kernel_var = None
        for w in layer.trainable_weights:
            if 'kernel' in w.name:
                kernel_var = w
                break
        if kernel_var is None:
            kernel_var = layer.trainable_weights[0]
        rec = {'name': layer.name, 'var_idx': var_index.get(id(kernel_var))}
        if rec['var_idx'] is None:
            continue
        if 'Conv' in lname:
            conv_meta.append(rec)
        else:
            dense_meta.append(rec)
    return conv_meta, dense_meta


def parse_neuron_player(name):
    parts = name.split('_')
    if len(parts) < 3:
        return None
    return {
        'layer_type': parts[0].lower(),
        'layer_name': '_'.join(parts[1:-1]),
        'neuron_idx': int(parts[-1]),
    }


def build_masked_grads(grads, masks):
    return [None if g is None else g * m for g, m in zip(grads, masks)]


def get_surrogate_losses(model, x, y, g=None, fairness_weight=1.0, acc_weight=1.0):
    logits = model(x)
    out_shape = logits.shape.as_list()
    if len(out_shape) >= 2 and out_shape[-1] == 1:
        y_float = tf.cast(tf.reshape(y, [-1, 1]), tf.float32)
        per_sample_ce = tf.reshape(
            tf.keras.backend.binary_crossentropy(y_float, logits), [-1])
    else:
        num_classes = model.output_shape[-1]
        y_onehot = tf.one_hot(y, num_classes)
        per_sample_ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_onehot, logits=logits)
    loss_acc = acc_weight * tf.reduce_mean(per_sample_ce)

    groups = y if g is None else g
    female_mask = tf.cast(tf.equal(groups, 0), tf.float32)
    male_mask = tf.cast(tf.equal(groups, 1), tf.float32)

    loss_female = tf.reduce_sum(per_sample_ce * female_mask) / (tf.reduce_sum(female_mask) + 1e-8)
    loss_male = tf.reduce_sum(per_sample_ce * male_mask) / (tf.reduce_sum(male_mask) + 1e-8)
    loss_fairness = fairness_weight * tf.square(loss_female - loss_male)

    return loss_fairness, loss_acc, loss_female, loss_male


def get_optimizer(task, data, method, mode):
    method = method.lower()

    defaults_by_dataset = {
        'bank': {'lr': 1e-4, 'steps': 2000, 'eval_every': 20},
        'adult': {'lr': 5e-4, 'steps': 2000, 'eval_every': 20},
        'default': {'lr': 1e-3, 'steps': 2000, 'eval_every': 20},
        # Backward-compatible alias for older Adult/Census runs.
        'census': {'lr': 1e-4, 'steps': 2000, 'eval_every': 20},
    }
    base = defaults_by_dataset.get(data, {'lr': 1e-3, 'steps': 100, 'eval_every': 20})

    method_cfg = {
        'pcgrad': {'cls': PCGrad, 'kwargs': {}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
        'mgda': {'cls': MGDA, 'kwargs': {'qp_steps': 50, 'qp_lr': 0.05}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
        'cagrad': {'cls': CAGrad, 'kwargs': {'c': 20, 'steps': 15, 'lr': 0.005}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
        'nashmtl': {'cls': NashMTL, 'kwargs': {'steps': 40, 'eps': 1e-4, 'tol': 1e-4}, 'lr': base['lr']/10, 'steps': base['steps'], 'eval_every': base['eval_every']},
        'gradnorm': {'cls': GradNorm, 'kwargs': {'alpha': 5}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
        'rlw': {'cls': RLW, 'kwargs': {}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
        'dwa': {'cls': DWA, 'kwargs': {'temp': 1.0}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
    }
    if method not in method_cfg:
        raise ValueError(f'Unknown method: {method}')

    cfg = method_cfg[method]
    lr = cfg['lr'] * 10 if mode == 'neuron' else cfg['lr']
    base_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    mtl_opt = cfg['cls'](base_optimizer, **cfg['kwargs'])
    return mtl_opt, base_optimizer, lr, cfg['steps'], cfg['eval_every']