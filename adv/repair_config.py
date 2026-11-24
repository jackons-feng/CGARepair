import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from MTL import PCGrad, MGDA, CAGrad, NashMTL, GradNorm, RLW, DWA

BATCH_SIZE = 256
try:
    tf.compat.v1.disable_eager_execution()
except RuntimeError:
    pass  # already disabled
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

def load_adv_data(model_name, epsilon):
    if model_name == 'mnist':
        adv_data_file = os.path.join('..', 'data', 'adv', 'mnist', 'train', f'epsilon{epsilon}.txt')
        adv_label_file = os.path.join('..', 'data', 'adv', 'mnist', 'train', 'label.txt')

        with open(adv_data_file, 'r') as f:
            data_lines = f.readlines()
        adv_x = [np.array(line.strip().split(','), dtype=float).reshape((28, 28, 1)) for line in data_lines]
        adv_x = np.array(adv_x, dtype='float32')

        with open(adv_label_file, 'r') as f:
            label_lines = f.readlines()
        adv_y = np.array([int(line.strip()) for line in label_lines], dtype=int)

        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        clean_x = np.expand_dims(x_train.astype('float32') / 255, -1)
        clean_y = y_train
        clean_x = clean_x[:len(adv_x)]
        clean_y = clean_y[:len(adv_x)]
    else:
        adv_data_file = os.path.join('..', 'data', 'adv', 'cifar10', 'train', f'epsilon{epsilon}.npy')
        adv_label_file = os.path.join('..', 'data', 'adv', 'cifar10', 'train', 'label.npy')

        def preprocess_cifar10(x):
            x = x.astype('float32')
            x -= np.array([0.4914, 0.4822, 0.4465])
            x /= np.array([0.247, 0.243, 0.261])
            return x

        x_adv_flat = np.load(adv_data_file)
        adv_x = x_adv_flat.reshape((-1, 32, 32, 3))
        adv_x = preprocess_cifar10(adv_x)
        adv_y = np.load(adv_label_file)

        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
        clean_x = preprocess_cifar10(x_train.astype('float32') / 255.0)
        clean_y = y_train.flatten()
        clean_x = clean_x[:len(adv_x)]
        clean_y = clean_y[:len(adv_x)]

    return adv_x, adv_y, clean_x, clean_y


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


def get_surrogate_losses(model, x_adv, y_adv, x_clean, y_clean, adv_weight=1.0, acc_weight=1.0):
    logits_adv = model(x_adv)
    logits_clean = model(x_clean)
    num_classes = model.output_shape[-1]

    y_adv_onehot = tf.one_hot(y_adv, num_classes)
    y_clean_onehot = tf.one_hot(y_clean, num_classes)

    adv_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_adv_onehot, logits=logits_adv)
    clean_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_clean_onehot, logits=logits_clean)

    loss_adv = adv_weight * tf.reduce_mean(adv_ce)
    loss_acc = acc_weight * tf.reduce_mean(clean_ce)
    return loss_adv, loss_acc


def get_optimizer(task, data, method, mode):
    method = method.lower()

    defaults_by_dataset = {
        'mnist': {'lr': 1e-3, 'steps': 500, 'eval_every': 20},
        'cifar10': {'lr': 1e-5, 'steps': 500, 'eval_every': 100},
    }
    base = defaults_by_dataset.get(data, {'lr': 1e-5, 'steps': 500, 'eval_every': 20})

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
    if mode != 'neuron':
        learning_rate = cfg['lr']
        base_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        learning_rate = cfg['lr']*10
        base_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    mtl_opt = cfg['cls'](base_optimizer, **cfg['kwargs'])
    return mtl_opt, base_optimizer, learning_rate, cfg['steps'], cfg['eval_every']
