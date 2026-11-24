import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from MTL import PCGrad, MGDA, CAGrad, NashMTL, GradNorm, RLW, DWA

BATCH_SIZE = 256


def load_safety_data(model_name):
    with h5py.File(f'../data/{model_name}/counterexample.h5', 'r') as hf:
        ct_x = hf['counterexample'][:]
    with h5py.File(f'../data/{model_name}/drawndown.h5', 'r') as hf:
        dr_x = hf['drawndown'][:]
    return ct_x, dr_x


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


def value(model, images, model_name, metric='safety'):
    output = sess_run(model, model.output, images)
    pred = np.argmin(output, axis=-1)
    if metric == 'safety':
        if model_name == 'a19':
            return float((np.sum(pred == 3) + np.sum(pred == 4)) / len(output))
        if model_name == 'a29':
            return float((len(images) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output))
        return float((np.sum(pred != 0)) / len(output))
    if model_name == 'a19':
        return float((len(output) - np.sum(pred == 3) - np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((np.sum(pred == 0) + np.sum(pred == 1)) / len(output))
    return float(np.sum(pred == 0) / len(output))


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


def get_surrogate_losses(model, model_name, x_ct, x_dr, safety_weight=1.0, drawndown_weight=1.0):
    logits_ct = model(x_ct)
    logits_dr = model(x_dr)

    if model_name == 'a19':
        min_unsafe_ct = tf.reduce_min(logits_ct[:, 3:5], axis=1)
        min_safe_ct = tf.reduce_min(logits_ct[:, :3], axis=1)
        min_safe_dr = tf.reduce_min(logits_dr[:, :3], axis=1)
    elif model_name == 'a29':
        min_unsafe_ct = tf.reduce_min(logits_ct[:, 2:5], axis=1)
        min_safe_ct = tf.reduce_min(logits_ct[:, :2], axis=1)
        min_safe_dr = tf.reduce_min(logits_dr[:, :2], axis=1)
    else:
        min_unsafe_ct = tf.reduce_min(logits_ct[:, 1:], axis=1)
        min_safe_ct = logits_ct[:, 0]
        min_safe_dr = logits_dr[:, 0]

    loss_safety = safety_weight * tf.reduce_mean(min_safe_ct - min_unsafe_ct)
    loss_drawndown = drawndown_weight * tf.reduce_mean(min_safe_dr)

    return loss_safety, loss_drawndown, tf.reduce_mean(min_unsafe_ct), tf.reduce_mean(min_safe_dr)


def get_optimizer(task, data, method, mode):
    method = method.lower()

    defaults_by_dataset = {
        'a19': {'lr': 1e-3, 'steps': 300, 'eval_every': 20},
        'a29': {'lr': 1e-3, 'steps': 300, 'eval_every': 20},
        'a33': {'lr': 1e-3, 'steps': 300, 'eval_every': 20},
    }
    base = defaults_by_dataset.get(data, {'lr': 1e-2, 'steps': 400, 'eval_every': 20})

    method_cfg = {
        'pcgrad': {'cls': PCGrad, 'kwargs': {}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
        'mgda': {'cls': MGDA, 'kwargs': {'qp_steps': 50, 'qp_lr': 0.05}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
        'cagrad': {'cls': CAGrad, 'kwargs': {'c': 10, 'steps': 20, 'lr': 0.005}, 'lr': base['lr'], 'steps': base['steps'], 'eval_every': base['eval_every']},
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
