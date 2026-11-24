import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from MTL import PCGrad, MGDA, CAGrad, NashMTL, GradNorm, RLW, DWA

BATCH_SIZE = 256
IMG_DIR = '../data/img'

DATASET_CONFIG = {
    'gtsrb': {
        'input_shape': (32, 32, 3),
        'num_classes': 43,
        'target_label': 33
    },
    'fashion': {
        'input_shape': (28, 28, 1),
        'num_classes': 10,
        'target_label': 3
    }
}


# ============================================================
# CAGrad sensitivity configurations for RQ5
# ============================================================
# Fashion-MNIST single-backdoor repair:
#
#   F1: default configuration
#   F2: smaller conflict-control coefficient
#   F3: larger conflict-control coefficient
#   F4: fewer CAGrad inner optimization steps
#
# When studying c:
#   S = 15 and eta_w = 0.005 are fixed.
#
# When studying S:
#   c = 0.4 and eta_w = 0.005 are fixed.
# ============================================================

CAGRAD_SENSITIVITY_CONFIGS = {
    'backdoor': {
        'fashion': {
            'F1': {
                'c': 0.4,
                'steps': 15,
                'lr': 0.005,
                'description': 'default'
            },
            'F2': {
                'c': 0.2,
                'steps': 15,
                'lr': 0.005,
                'description': 'smaller_c'
            },
            'F3': {
                'c': 0.8,
                'steps': 15,
                'lr': 0.005,
                'description': 'larger_c'
            },
            'F4': {
                'c': 0.4,
                'steps': 10,
                'lr': 0.005,
                'description': 'fewer_inner_steps'
            },
        }
    }
}

DEFAULT_CAGRAD_EXPERIMENT = {
    'backdoor': {
        'fashion': 'F1'
    }
}


def get_cagrad_experiment(task, data, experiment_id=None):
    """Return one predefined CAGrad sensitivity configuration.

    Parameters
    ----------
    task : str
        Repair task name, e.g. ``backdoor``.
    data : str
        Dataset name, e.g. ``fashion``.
    experiment_id : str or None
        Configuration identifier such as F1, F2, F3, or F4. If omitted,
        the default configuration for the task/dataset is used.

    Returns
    -------
    dict or None
        A copy of the selected configuration, including ``experiment_id``.
        Returns None when no predefined configuration exists and no explicit
        identifier is supplied.
    """
    task_key = str(task).lower()
    data_key = str(data).lower()

    task_configs = CAGRAD_SENSITIVITY_CONFIGS.get(task_key, {})
    dataset_configs = task_configs.get(data_key, {})

    if experiment_id is None or str(experiment_id).strip() == '':
        experiment_id = DEFAULT_CAGRAD_EXPERIMENT.get(task_key, {}).get(data_key)

    if experiment_id is None:
        return None

    exp_key = str(experiment_id).upper()
    if exp_key not in dataset_configs:
        available = ', '.join(sorted(dataset_configs)) or 'none'
        raise ValueError(
            f'Unknown CAGrad experiment {experiment_id!r} for '
            f'task={task_key}, data={data_key}. Available: {available}'
        )

    config = dict(dataset_configs[exp_key])
    config['experiment_id'] = exp_key
    return config


# TF1 graph mode
tf.compat.v1.disable_eager_execution()
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))


def load_clean_data(model_name):
    if model_name == 'gtsrb':
        import pickle
        data_file = os.path.join('..', 'data', 'gtsrb', 'train_data.pkl')
        with open(data_file, 'rb') as f:
            train_data = pickle.load(f)
            X_clean = np.transpose(train_data['data'], (0, 2, 3, 1))
            X_clean = K.get_session().run(tf.image.resize(X_clean.astype('float32'), (32, 32)))
            X_clean = X_clean * 255.0
            Y_clean = np.array(train_data['label'], dtype=int)
        return X_clean, Y_clean
    if model_name == 'fashion':
        (x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
        return x_train, y_train
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
    return x_train, y_train


def get_perturbed_input(x, model_name, target_label, input_shape):
    mask = []
    pattern = []

    img_filename_template = '{}_visualize_%s_label_%d.png'.format(model_name)
    mask_filename = img_filename_template % ('mask', target_label)
    if os.path.isfile('%s/%s' % (IMG_DIR, mask_filename)):
        img = image.load_img(
            '%s/%s' % (IMG_DIR, mask_filename),
            color_mode='grayscale',
            target_size=input_shape)
        mask = image.img_to_array(img)
        mask /= 255
        mask = mask[:, :, 0]

    pattern_filename = img_filename_template % ('pattern', target_label)
    if os.path.isfile('%s/%s' % (IMG_DIR, pattern_filename)):
        img = image.load_img(
            '%s/%s' % (IMG_DIR, pattern_filename),
            color_mode='rgb',
            target_size=input_shape)
        pattern = image.img_to_array(img)

    mask_shape = np.ceil(np.array(input_shape[0:2], dtype=float)).astype(int)
    filtered = np.multiply(x, np.expand_dims(np.subtract(np.ones(mask_shape), mask), axis=2))
    if model_name == 'gtsrb':
        fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
        x_out = np.add(filtered, fusion)
    else:
        pattern = pattern[:, :, 1] / 255.
        fusion = np.expand_dims(np.multiply(pattern, mask), axis=2)
        x_out = np.add(filtered, fusion)
    return x_out


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


def value_accuracy(model, images, labels):
    output = sess_run(model, model.output, images)
    pred_labels = np.argmax(output, axis=-1)
    corr_predictions = pred_labels == labels
    return float(np.sum(corr_predictions) / corr_predictions.shape[0])


def value_backdoor(model, images_p, labels_clean, target_label):
    pred_p = np.argmax(sess_run(model, model.output, images_p), axis=-1)
    return float((np.sum(pred_p == target_label) - np.sum((labels_clean == target_label) & (pred_p == target_label))) / len(images_p))


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


def parse_weight_player(p):
    # format: w|{layer_type}|{layer_name}|{neuron_idx}|{flat_idx}
    parts = p.split('|')
    if len(parts) != 5 or parts[0] != 'w':
        return None
    return {
        'layer_type': parts[1],
        'layer_name': parts[2],
        'neuron_idx': int(parts[3]),
        'flat_idx': int(parts[4])
    }


def build_masked_grads(grads, masks):
    masked = []
    for g, m in zip(grads, masks):
        masked.append(None if g is None else g * m)
    return masked


def compute_weight_change_stats(weights_before, weights_after, masks, atol=1e-12):
    total_changed = 0
    masked_changed = 0
    unmasked_changed = 0
    total_params = 0
    l1_total = 0.0
    l2_total = 0.0
    max_abs = 0.0
    changed_examples = []

    for vidx, (wb, wa, m) in enumerate(zip(weights_before, weights_after, masks)):
        diff = wa - wb
        abs_diff = np.abs(diff)
        changed = abs_diff > atol

        total_params += int(diff.size)
        total_changed += int(np.sum(changed))

        mask_bool = (m > 0.5)
        masked_changed += int(np.sum(changed & mask_bool))
        unmasked_changed += int(np.sum(changed & (~mask_bool)))

        l1_total += float(np.sum(abs_diff))
        l2_total += float(np.sum(diff * diff))
        layer_max = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
        max_abs = max(max_abs, layer_max)

        if len(changed_examples) < 10 and np.any(changed):
            idxs = np.argwhere(changed)
            for ind in idxs[:max(0, 10 - len(changed_examples))]:
                ind_t = tuple(int(i) for i in ind.tolist())
                changed_examples.append({
                    'var_index': int(vidx),
                    'index': ind_t,
                    'is_masked': bool(mask_bool[ind_t]),
                    'before': float(wb[ind_t]),
                    'after': float(wa[ind_t]),
                    'delta': float(diff[ind_t]),
                })
                if len(changed_examples) >= 10:
                    break

    return {
        'atol': float(atol),
        'total_params': int(total_params),
        'total_changed_l0': int(total_changed),
        'masked_changed_l0': int(masked_changed),
        'unmasked_changed_l0': int(unmasked_changed),
        'change_ratio': float(total_changed / max(1, total_params)),
        'l1_total': float(l1_total),
        'l2_total': float(np.sqrt(l2_total)),
        'max_abs_change': float(max_abs),
        'changed_examples': changed_examples,
    }


def get_surrogate_losses(model, x_clean, y_clean, x_bd, y_bd, backdoor_weight=1.0, acc_weight=1.0):
    logits_clean = model(x_clean)
    logits_bd = model(x_bd)
    num_classes = model.output_shape[-1]

    y_clean_onehot = tf.one_hot(y_clean, num_classes)
    y_bd_onehot = tf.one_hot(y_bd, num_classes)

    clean_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_clean_onehot, logits=logits_clean)
    bd_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_bd_onehot, logits=logits_bd)

    loss_acc = acc_weight * tf.reduce_mean(clean_ce)
    loss_backdoor = backdoor_weight * tf.reduce_mean(bd_ce)

    return loss_backdoor, loss_acc


def get_optimizer(task, data, method, mode, cagrad_kwargs=None):
    """Build the requested multi-task optimizer.

    ``cagrad_kwargs`` optionally overrides only CAGrad's internal parameters
    (``c``, ``steps``, and ``lr``). The outer Adam learning rate, outer
    training steps, and evaluation interval remain controlled by the dataset
    configuration.
    """
    method = method.lower()

    defaults_by_dataset = {
        'fashion': {'lr': 1e-3, 'steps': 400, 'eval_every': 40},
        'gtsrb': {'lr': 1e-4, 'steps': 300, 'eval_every': 30},
    }
    base = defaults_by_dataset.get(
        data,
        {'lr': 1e-5, 'steps': 200, 'eval_every': 20}
    )

    cagrad_internal = {
        # Default selected from the Fashion-MNIST sensitivity study (F3).
        'c': 0.4,
        'steps': 15,
        'lr': 0.005,
    }
    if cagrad_kwargs is not None:
        for key in ('c', 'steps', 'lr'):
            if key in cagrad_kwargs:
                cagrad_internal[key] = cagrad_kwargs[key]

    method_overrides = {
        'pcgrad': {
            'mtl_cls': PCGrad,
            'mtl_kwargs': {},
            'lr': base['lr'],
            'steps': base['steps'],
            'eval_every': base['eval_every']
        },
        'mgda': {
            'mtl_cls': MGDA,
            'mtl_kwargs': {'qp_steps': 50, 'qp_lr': 0.05},
            'lr': base['lr'],
            'steps': base['steps'],
            'eval_every': base['eval_every']
        },
        'cagrad': {
            'mtl_cls': CAGrad,
            'mtl_kwargs': cagrad_internal,
            'lr': base['lr'],
            'steps': base['steps'],
            'eval_every': base['eval_every']
        },
        'nashmtl': {
            'mtl_cls': NashMTL,
            'mtl_kwargs': {'steps': 40, 'eps': 1e-4, 'tol': 1e-4},
            'lr': base['lr'],
            'steps': base['steps'],
            'eval_every': base['eval_every']
        },
        'gradnorm': {
            'mtl_cls': GradNorm,
            'mtl_kwargs': {'alpha': 5},
            'lr': base['lr'],
            'steps': base['steps'],
            'eval_every': base['eval_every']
        },
        'rlw': {
            'mtl_cls': RLW,
            'mtl_kwargs': {},
            'lr': base['lr'],
            'steps': base['steps'],
            'eval_every': base['eval_every']
        },
        'dwa': {
            'mtl_cls': DWA,
            'mtl_kwargs': {'temp': 1.0},
            'lr': base['lr'],
            'steps': base['steps'],
            'eval_every': base['eval_every']
        },
    }

    if method not in method_overrides:
        raise ValueError(f"Unknown method: {method}")

    cfg = method_overrides[method]
    outer_lr = cfg['lr'] * 10 if mode == 'neuron' else cfg['lr']
    outer_steps = cfg['steps']
    eval_every = cfg['eval_every']

    base_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=outer_lr)
    mtl_opt = cfg['mtl_cls'](base_optimizer, **cfg['mtl_kwargs'])

    return mtl_opt, base_optimizer, outer_lr, outer_steps, eval_every
