"""CGARepair ACAS Xu safety local-Kernel FAMO repair (TF1 compatible).

Uses the EXACT original GradAct localization and selected Kernel output weights.
ACAS Xu predicts its action by ARGMIN, not argmax. Unsafe actions:
    a19: {3,4}, a29: {2,...,C-1}, a33: {1,...,C-1}.

FAMO differentiable tasks:
 (1) minimize negative log probability of predicting a SAFE action on TRAIN
     counterexamples under softmax(-network_output / temperature);
 (2) preserve original pre-trained ARGMIN actions on TRAIN drawdown samples
     by teacher-action cross-entropy under the same softmax(-output).

Discrete TRAIN safety violation rate and drawdown retention select the snapshot.
Independent TEST never selects a checkpoint. Five-seed mode starts separate
processes per seed to isolate the TensorFlow graph/GPU state.

Depends only on the original read_CGARepair.py and cgarepair_safety_utils.py.
The weight mapping and TRAIN selection functions are embedded unchanged from
safety_moo_2.py; no old optimizer module is imported or invoked.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from read_CGARepair import load_ranking, export_ranking, model_name_checked
from cgarepair_safety_utils import (
    CONFIG, initialize_tf, load_safety_data, model_file, calculate_metrics, save_json,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
SEEDS = (42, 123, 256, 1024, 2026)


# Unchanged helper logic from the original safety_moo_2.py; no MOEA/D dependency.
def run_directory(root, model, seed, optimizer, run_tag='', quick=False):
    suffix = ('_' + run_tag) if run_tag else ('_quick' if quick else '')
    return (Path(root).expanduser() / model / 'CGARepair' /
            ('seed_' + str(seed)) / (optimizer + suffix))


def map_players_to_weights(model, selected_players):
    """Use real model.layers indices and count *all* intervening weight arrays.

    Older ResNet repairs could unintentionally omit BatchNorm offsets. This
    mapping deliberately includes them while keeping biases/BN frozen.
    """
    flattened = model.get_weights()
    mapping = {}
    offset = 0
    for layer_idx, layer in enumerate(model.layers):
        local = layer.get_weights()
        if local and ('Conv' in type(layer).__name__ or 'Dense' in type(layer).__name__):
            mapping[layer_idx] = (offset, local[0], type(layer).__name__)
        offset += len(local)
    if offset != len(flattened):
        raise RuntimeError('Model weight offsets do not match model.get_weights; '
                           'nested Keras models require explicit support')
    result = []
    seen = set()
    start = 0
    for name in selected_players:
        m = re.fullmatch(r'(Conv\w*|Dense)_(\d+)_(\d+)', str(name), flags=re.IGNORECASE)
        if m is None:
            raise ValueError('Bad CGARepair player identifier: {}'.format(name))
        expected_type, layer_idx, neuron = m.group(1), int(m.group(2)), int(m.group(3))
        if layer_idx not in mapping:
            raise KeyError('Selected layer is missing or not Conv/Dense: {}'.format(name))
        weight_idx, kernel, true_type = mapping[layer_idx]
        if expected_type.lower() != true_type.lower():
            raise ValueError('CGARepair/model layer type mismatch: {} actual={}'.format(name, true_type))
        if kernel.ndim not in (2, 3, 4, 5):
            raise ValueError('Unsupported parameter tensor {} {}'.format(name, kernel.shape))
        if not 0 <= neuron < kernel.shape[-1]:
            raise ValueError('Neuron index outside kernel shape: {} {}'.format(name, kernel.shape))
        key = (weight_idx, neuron)
        if key in seen:
            raise ValueError('Duplicate parameter slice: {}'.format(name))
        seen.add(key)
        shape = tuple(kernel[..., neuron].shape)
        length = int(np.prod(shape))
        result.append({'player': name, 'layer_index': layer_idx,
                       'weight_index': weight_idx, 'neuron_index': neuron,
                       'shape': shape, 'start': start, 'stop': start+length,
                       'num_parameters': length})
        start += length
    if not result:
        raise ValueError('Empty repair mask')
    return result


def original_vector(weights, specs):
    return np.concatenate([weights[s['weight_index']][..., s['neuron_index']].reshape(-1)
                           for s in specs]).astype(np.float64)


def apply_vector(weights, specs, vector):
    vector = np.asarray(vector, dtype=np.float64)
    for s in specs:
        weights[s['weight_index']][..., s['neuron_index']] = (
            vector[s['start']:s['stop']].reshape(s['shape']))
    return weights


def dominates(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b + 1.e-12) and np.any(a < b - 1.e-12))


def non_dominated_indices(F):
    F = np.atleast_2d(np.asarray(F, dtype=float))
    result = []
    for i in range(len(F)):
        if any(dominates(F[j], F[i]) for j in range(len(F)) if j != i):
            continue
        if any(np.allclose(F[i], F[j], atol=1.e-12, rtol=0) for j in result):
            continue
        result.append(i)
    return np.asarray(result, dtype=int)


def crowding_distance(F):
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n = len(F)
    if n <= 2:
        return np.full(n, np.inf)
    crowd = np.zeros(n, dtype=float)
    for j in range(F.shape[1]):
        order = np.argsort(F[:, j], kind='mergesort')
        crowd[order[0]] = np.inf
        crowd[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span > 1.e-14:
            for k in range(1, n-1):
                crowd[order[k]] += (F[order[k+1], j] - F[order[k-1], j])/span
    return crowd


def reduce_archive(X, F, max_size):
    X, F = np.asarray(X, dtype=np.float64), np.asarray(F, dtype=np.float64)
    keep = non_dominated_indices(F)
    X, F = X[keep], F[keep]
    while len(X) > max_size:
        crowd = crowding_distance(F)
        finite = np.where(np.isfinite(crowd))[0]
        idx = finite[np.argmin(crowd[finite])] if len(finite) else len(X)-1
        X, F = np.delete(X, idx, axis=0), np.delete(F, idx, axis=0)
    return X, F


def choose_train_solution(F, min_retention, max_violation):
    F = np.asarray(F, dtype=float)
    error, acc = F[:, 0], -F[:, 1]
    good = (error <= max_violation) & (acc >= min_retention)
    utility = acc >= min_retention
    if np.any(good):
        ids = np.where(good)[0]
        policy = 'meets_both_min_error_then_max_acc'
        order = np.lexsort((-acc[ids], error[ids]))
    elif np.any(utility):
        ids = np.where(utility)[0]
        policy = 'utility_guard_min_error_then_max_acc'
        order = np.lexsort((-acc[ids], error[ids]))
    else:
        ids = np.arange(len(F))
        policy = 'no_utility_feasible_max_acc_then_min_error'
        order = np.lexsort((error[ids], -acc[ids]))
    return int(ids[order[0]]), policy


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z-np.max(z))
    return e/np.sum(e)


class FAMOTaskWeights:
    def __init__(self, lr=.025, gamma=.001, eps=1.e-8):
        if lr <= 0 or gamma < 0 or eps <= 0:
            raise ValueError('Invalid FAMO task optimizer configuration')
        self.w = np.zeros(2, dtype=np.float64)
        self.m = np.zeros(2, dtype=np.float64)
        self.v = np.zeros(2, dtype=np.float64)
        self.k = 0
        self.lr, self.gamma, self.eps = float(lr), float(gamma), float(eps)

    @property
    def z(self):
        return softmax(self.w)

    def update(self, before, after):
        b, a = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
        if b.shape != (2,) or a.shape != (2,) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
            raise FloatingPointError('Nonfinite FAMO task losses')
        if np.any(b < 0) or np.any(a < 0):
            raise ValueError('FAMO task losses must be nonnegative')
        delta = np.log(b+self.eps)-np.log(a+self.eps)
        z = self.z
        grad = z*(delta-np.dot(z, delta))+self.gamma*self.w
        self.k += 1
        self.m = .9*self.m + .1*grad
        self.v = .999*self.v + .001*(grad*grad)
        mh = self.m/(1.-.9**self.k)
        vh = self.v/(1.-.999**self.k)
        self.w = np.clip(self.w-self.lr*mh/(np.sqrt(vh)+1.e-8), -30., 30.)
        return delta


def create_masks_bounds(base_weights, specs, lower, upper):
    if not specs or len(lower) != len(upper) or len(lower) != specs[-1]['stop']:
        raise ValueError('Weight specs and selected parameter bounds mismatch')
    result = {}
    for s in specs:
        idx, channel = int(s['weight_index']), int(s['neuron_index'])
        if idx not in result:
            result[idx] = {
                'layer_index': int(s['layer_index']),
                'mask': np.zeros(base_weights[idx].shape, dtype=np.float32),
                'lower': np.zeros(base_weights[idx].shape, dtype=np.float32),
                'upper': np.zeros(base_weights[idx].shape, dtype=np.float32),
            }
        blk = result[idx]
        if np.any(blk['mask'][..., channel]):
            raise ValueError('Duplicate selected Kernel slice: {}'.format(s['player']))
        blk['mask'][..., channel] = 1.
        blk['lower'][..., channel] = lower[s['start']:s['stop']].reshape(s['shape'])
        blk['upper'][..., channel] = upper[s['start']:s['stop']].reshape(s['shape'])
        if np.any(blk['lower'][..., channel] >= blk['upper'][..., channel]):
            raise ValueError('Invalid selected weight bounds')
    return result


def check_weight_scope(before, after, blocks):
    if len(before) != len(after):
        raise RuntimeError('Changed model weight tensor count')
    changed = 0
    for i, (old, new) in enumerate(zip(before, after)):
        if old.shape != new.shape or not np.all(np.isfinite(new)):
            raise RuntimeError('Invalid repaired tensor {}'.format(i))
        if i in blocks:
            blk = blocks[i]
            mask = blk['mask'].astype(bool)
            if not np.array_equal(old[~mask], new[~mask]):
                raise RuntimeError('FAMO modified unselected Kernel weight {}'.format(i))
            if (np.any(new[mask] < blk['lower'][mask] - 1.e-5) or
                    np.any(new[mask] > blk['upper'][mask] + 1.e-5)):
                raise RuntimeError('FAMO modified selected weight outside bounds {}'.format(i))
            changed += int(np.count_nonzero(old[mask] != new[mask]))
        elif not np.array_equal(old, new):
            raise RuntimeError('FAMO modified unselected tensor {}'.format(i))
    return {
        'modified_parameter_count': changed,
        'selected_parameter_count': int(sum(np.count_nonzero(b['mask']) for b in blocks.values())),
        'outside_scope_modified_count': 0, 'other_tensor_untouched': True,
    }


def safe_actions_for(model_name, n_classes):
    # Mirrors original safety GradAct/VIOLATION definitions, ARGMIN action.
    if model_name == 'a19':
        unsafe = {3, 4}
    elif model_name == 'a29':
        unsafe = set(range(2, n_classes))
    elif model_name == 'a33':
        unsafe = set(range(1, n_classes))
    else:
        raise ValueError('Unknown safety model {}'.format(model_name))
    if n_classes < 5 or not unsafe or max(unsafe) >= n_classes:
        raise ValueError('Unexpected ACAS Xu action output count: {}'.format(n_classes))
    safe = sorted(set(range(n_classes)) - unsafe)
    if not safe:
        raise ValueError('No safe actions for model {}'.format(model_name))
    return safe, sorted(unsafe)


def formatted_input(X, model):
    arr = np.asarray(X, dtype=np.float32)
    expected = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if arr.ndim == 1:
        d = int(np.prod(expected[1:]))
        if d < 1 or arr.size % d:
            raise ValueError('Cannot reshape 1D ACAS inputs to model feature shape')
        arr = arr.reshape((-1,)+tuple(expected[1:]))
    if tuple(arr.shape[1:]) != tuple(expected[1:]):
        raise ValueError('ACAS sample shape {} != model input {}'.format(arr.shape, expected))
    return arr


def predict_original_actions(session, model, X, batch_size):
    actions = []
    for start in range(0, len(X), batch_size):
        out = session.run(model.output, feed_dict={model.input: X[start:start+batch_size]})
        if not np.all(np.isfinite(out)):
            raise ValueError('Original safety network output contains NaN/Inf')
        actions.append(np.argmin(out, axis=-1).astype(np.int32))
    return np.concatenate(actions)


class MaskedSafetyFAMOGraph:
    def __init__(self, tf, session, model, base_weights, blocks, safe_actions,
                 model_lr, clip_norm, loss_eps, temperature):
        if temperature <= 0:
            raise ValueError('Safety softmax temperature must be >0')
        v1 = tf.compat.v1
        self.session = session
        self.x = model.input
        self.n_counter = v1.placeholder(tf.int32, [], name='famo_safety_counter_count')
        self.ref_actions = v1.placeholder(tf.int32, [None], name='famo_drawdown_teacher_actions')
        self.z = v1.placeholder(tf.float32, [2], name='famo_safety_task_weights')
        self.blocks = blocks
        output = tf.cast(model.output, tf.float32)
        if output.shape.ndims != 2:
            raise ValueError('Safety model output must be [N,num_actions]')
        action_logits = -output/float(temperature)  # ACAS action rule = argmin output
        counter = action_logits[:self.n_counter]
        drawdown = action_logits[self.n_counter:]
        # -log(P(any safe action)), a smooth surrogate for argmin safety violation.
        log_all = tf.reduce_logsumexp(counter, axis=1)
        log_safe = tf.reduce_logsumexp(tf.gather(counter, safe_actions, axis=1), axis=1)
        safety_loss = tf.reduce_mean(tf.maximum(log_all - log_safe, 0.))
        # Keep original drawdown action (not arbitrary safe-class remapping).
        retention_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.ref_actions, logits=drawdown))
        self.task_losses = tf.stack([safety_loss, retention_loss],
                                    name='famo_safety_violation_proxy_and_teacher_action_CE')
        positive = tf.maximum(self.task_losses, 0.) + float(loss_eps)
        denom = tf.stop_gradient(tf.reduce_sum(self.z/positive))
        combined = tf.reduce_sum(self.z*tf.math.log(positive))/denom
        variables, masks, bounds = [], [], []
        for idx, blk in sorted(blocks.items()):
            kernel = model.layers[blk['layer_index']].weights[0]
            if tuple(kernel.shape.as_list()) != tuple(base_weights[idx].shape):
                raise ValueError('Selected ACAS Kernel mapping mismatch at {}'.format(idx))
            variables.append(kernel)
            dtype = kernel.dtype.base_dtype
            masks.append(tf.constant(blk['mask'], dtype=dtype))
            bounds.append((tf.constant(blk['lower'], dtype=dtype),
                           tf.constant(blk['upper'], dtype=dtype)))
        grads = tf.gradients(combined, variables)
        if any(g is None for g in grads):
            raise RuntimeError('Selected ACAS Kernel disconnected from model output')
        masked = []
        for grad, mask in zip(grads, masks):
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
            masked.append(grad*mask)
        self.gradnorm = v1.global_norm(masked)
        if clip_norm > 0:
            masked, _ = tf.clip_by_global_norm(masked, float(clip_norm), use_norm=self.gradnorm)
        original_var_names = {v.name for v in v1.global_variables()}
        optimizer = v1.train.AdamOptimizer(learning_rate=float(model_lr))
        train = optimizer.apply_gradients(list(zip(masked, variables)))
        projections = []
        with tf.control_dependencies([train]):
            for kernel, mask, (lo, hi) in zip(variables, masks, bounds):
                clipped = tf.clip_by_value(kernel, lo, hi)
                projections.append(v1.assign(kernel, tf.where(tf.greater(mask, 0), clipped, kernel)))
        self.train_op = tf.group(*projections, name='famo_safety_masked_adam')
        new_vars = [v for v in v1.global_variables() if v.name not in original_var_names]
        if new_vars:
            session.run(v1.variables_initializer(new_vars))

    def feed(self, counter_x, drawdown_x, teacher_actions, z):
        if not len(counter_x) or not len(drawdown_x):
            raise ValueError('Safety FAMO requires nonempty counterexample/drawdown batches')
        return {
            self.x: np.concatenate((counter_x, drawdown_x), axis=0),
            self.n_counter: int(len(counter_x)),
            self.ref_actions: np.asarray(teacher_actions, dtype=np.int32),
            self.z: np.asarray(z, dtype=np.float32),
        }

    def step(self, feed):
        before, _, norm = self.session.run(
            [self.task_losses, self.train_op, self.gradnorm], feed_dict=feed)
        after = self.session.run(self.task_losses, feed_dict=feed)
        if (not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)) or
                not np.isfinite(norm)):
            raise FloatingPointError('Nonfinite safety FAMO loss/gradient')
        return np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64), float(norm)


def write_csv(path, rows):
    if not rows:
        return
    with open(str(path), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def get_parser():
    p = argparse.ArgumentParser(description='CGARepair safety GradAct + masked FAMO')
    p.add_argument('model_name', choices=('a19','a29','a33'))
    p.add_argument('seed', type=int, nargs='?', help='Optimization seed; omit for --five-seeds')
    p.add_argument('--optimizer', '--algorithm', choices=('famo',), default='famo')
    p.add_argument('--five-seeds', action='store_true', help='Five-seed TRAIN + full independent TEST')
    p.add_argument('--loc-json')
    p.add_argument('--loc-root')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-tag', default='paper_gamma001_steps5000')
    p.add_argument('--quick', action='store_true', help='Five gradient steps with small TRAIN pools')
    p.add_argument('--overwrite', action='store_true', help='Back up pre-existing single-seed result before rerun')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--eval-samples', type=int, default=1024,
                   help='Original safety TRAIN pool cap per set; 0 means full TRAIN')
    p.add_argument('--grad-batch-size', type=int, default=64,
                   help='Examples PER TASK for each gradient update')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--eval-every', type=int, default=20)
    p.add_argument('--model-lr', type=float, default=.001)
    p.add_argument('--task-lr', type=float, default=.025)
    p.add_argument('--task-decay', type=float, default=.001)
    p.add_argument('--loss-eps', type=float, default=1.e-8)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--safety-temperature', type=float, default=1.)
    p.add_argument('--search-radius', type=float, default=10.0,
                   help='Original MOEA/D safety local weight radius, intersect [-5,+5]')
    p.add_argument('--threshold-acc', type=float, default=None,
                   help='Minimum TRAIN drawdown retention, old default from CONFIG')
    p.add_argument('--threshold-safety', type=float, default=None,
                   help='Maximum TRAIN violation rate, old default from CONFIG')
    p.add_argument('--archive-size', type=int, default=60)
    p.add_argument('--save-front-limit', type=int, default=10)
    p.add_argument('--log-every', type=int, default=100)
    return p


def check_args(args):
    if args.quick:
        args.run_tag = 'smoke_5steps'
        args.max_steps = 5
        args.eval_samples = min(int(args.eval_samples), 128) if args.eval_samples else 128
        args.grad_batch_size = min(int(args.grad_batch_size), 16)
        args.eval_every = 1
    if (args.max_steps < 1 or args.eval_samples < 0 or args.grad_batch_size < 1 or
            args.batch_size < 1 or args.eval_every < 1 or args.log_every < 1 or
            args.model_lr <= 0 or args.task_lr <= 0 or args.task_decay < 0 or
            args.loss_eps <= 0 or args.max_grad_norm < 0 or args.safety_temperature <= 0 or
            args.search_radius <= 0 or args.archive_size < 2 or args.save_front_limit < 0):
        raise ValueError('Invalid safety FAMO optimizer hyperparameters')


def run(args):
    t0 = time.perf_counter()
    name = model_name_checked(args.model_name)
    check_args(args)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    outdir = run_directory(args.output_root, name, args.seed, 'famo', args.run_tag, args.quick)
    if outdir.is_dir() and any(outdir.iterdir()):
        if not args.overwrite:
            raise FileExistsError('Existing or incomplete FAMO run: {}'.format(outdir))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup = outdir.with_name(outdir.name+'_backup_'+stamp)
        outdir.rename(backup)
        print('[BACKUP] {}'.format(backup), flush=True)
    outdir.mkdir(parents=True, exist_ok=True)

    stage = time.perf_counter()
    players, scores, loc = load_ranking(name, args.loc_seed, args.loc_samples,
                                         args.loc_json, args.loc_root)
    exported = export_ranking(players, scores, loc, outdir/'localization_input')
    loc_sec = time.perf_counter()-stage
    print('[Locator] {} Top-{} from {} (loc={} opt={})'.format(
        name, len(players), loc['source_json'], args.loc_seed, args.seed), flush=True)

    stage = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    try:
        K.set_learning_phase(0)
    except AttributeError:
        tf.compat.v1.keras.backend.set_learning_phase(0)
    from tensorflow.keras.models import load_model
    checkpoint = model_file(name, args.model_root)
    net = load_model(checkpoint, compile=False)
    signature = loc.get('source_model_signature')
    if (isinstance(signature, (list,tuple)) and signature and
            int(signature[0]) != os.stat(checkpoint).st_size):
        raise RuntimeError('Safety locator and pretrained checkpoint sizes differ')
    specs = map_players_to_weights(net, players)
    base_weights = [w.copy() for w in net.get_weights()]
    x0 = original_vector(base_weights, specs)
    low = np.maximum(-5., x0-args.search_radius)
    high = np.minimum(5., x0+args.search_radius)
    if np.any(low >= high) or np.any(x0 < low) or np.any(x0 > high):
        raise ValueError('Original weights not strictly inside selected safety bounds')
    blocks = create_masks_bounds(base_weights, specs, low, high)
    shape = net.output_shape[0] if isinstance(net.output_shape, list) else net.output_shape
    if len(shape) != 2 or shape[-1] is None:
        raise ValueError('ACAS Xu output shape is not [N,C]')
    safe, unsafe = safe_actions_for(name, int(shape[-1]))
    net_sec = time.perf_counter()-stage
    print('[Repair] {} TopK={} variables={} safe_actions={} unsafe_actions={}'.format(
        name, len(players), len(x0), safe, unsafe), flush=True)

    stage = time.perf_counter()
    ct_x, dr_x, provenance = load_safety_data(
        name, split='train', data_root=args.data_root,
        samples=args.eval_samples, seed=args.seed)
    ct_x = formatted_input(ct_x, net)
    dr_x = formatted_input(dr_x, net)
    if not len(ct_x) or not len(dr_x):
        raise ValueError('Empty ACAS counterexample/drawdown TRAIN pool')
    original_actions = predict_original_actions(session, net, dr_x, args.batch_size)
    baseline = calculate_metrics(net, name, ct_x, dr_x, session, args.batch_size)
    min_retention = (float(CONFIG[name]['threshold_acc']) if args.threshold_acc is None
                     else float(args.threshold_acc))
    max_violation = (float(CONFIG[name]['threshold_safety']) if args.threshold_safety is None
                     else float(args.threshold_safety))
    if not 0 <= min_retention <= 1 or not 0 <= max_violation <= 1:
        raise ValueError('TRAIN safety/retention thresholds must be in [0,1]')
    data_sec = time.perf_counter()-stage
    print('[Baseline] TRAIN VR={:.6f} Retention={:.6f} n_ct={} n_drawdown={} '
          'guard={:.6f}'.format(baseline['safety_violation_rate'],
          baseline['drawdown_retention'], len(ct_x), len(dr_x), min_retention), flush=True)

    stage = time.perf_counter()
    graph = MaskedSafetyFAMOGraph(
        tf, session, net, base_weights, blocks, safe, args.model_lr,
        args.max_grad_norm, args.loss_eps, args.safety_temperature)
    graph_sec = time.perf_counter()-stage
    weights = FAMOTaskWeights(args.task_lr, args.task_decay, args.loss_eps)
    rng = np.random.RandomState(int(args.seed))
    all_x, all_f, eval_steps, snapshots, trace = [], [], [], [], []

    def record(step, elapsed):
        met = calculate_metrics(net, name, ct_x, dr_x, session, args.batch_size)
        f = [float(met['safety_violation_rate']), -float(met['drawdown_retention'])]
        if not np.all(np.isfinite(f)):
            raise FloatingPointError('Nonfinite TRAIN safety metrics')
        all_x.append(original_vector(net.get_weights(), specs))
        all_f.append(f)
        eval_steps.append(int(step))
        snapshots.append({
            'step': int(step), 'train_safety_violation': f[0],
            'train_drawdown_retention': -f[1],
            'feasible': bool(f[0] <= max_violation and -f[1] >= min_retention),
            'z_safety': float(weights.z[0]), 'z_retention': float(weights.z[1]),
            'elapsed_opt_sec': float(elapsed),
        })
        return met

    record(0, 0.)
    opt_start = time.perf_counter()
    print('[FAMO] {} steps={} safe_probability_loss + drawdown_teacher_CE, '
          'grad_batch={} eval_every={} temperature={}'.format(
              name, args.max_steps, args.grad_batch_size, args.eval_every,
              args.safety_temperature), flush=True)
    for step in range(1, args.max_steps+1):
        ic = rng.randint(len(ct_x), size=args.grad_batch_size)
        ir = rng.randint(len(dr_x), size=args.grad_batch_size)
        feed = graph.feed(ct_x[ic], dr_x[ir], original_actions[ir], weights.z)
        before, after, norm = graph.step(feed)
        delta = weights.update(before, after)
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print('[FAMO step {:04d}/{}] safetyLoss={:.6f}->{:.6f} '
                  'retentionCE={:.6f}->{:.6f} z=({:.3f},{:.3f}) norm={:.5f}'.format(
                     step, args.max_steps, before[0], after[0], before[1], after[1],
                     weights.z[0], weights.z[1], norm), flush=True)
        trace.append({
            'step': int(step), 'safety_loss_pre': float(before[0]),
            'safety_loss_post': float(after[0]),
            'retention_ce_pre': float(before[1]),
            'retention_ce_post': float(after[1]),
            'safety_log_improvement': float(delta[0]),
            'retention_log_improvement': float(delta[1]),
            'z_safety': float(weights.z[0]), 'z_retention': float(weights.z[1]),
            'gradient_global_norm': float(norm),
            'elapsed_opt_sec': float(time.perf_counter()-opt_start),
        })
        if step % args.eval_every == 0 or step == args.max_steps:
            met = record(step, time.perf_counter()-opt_start)
            print('[TRAIN step={}] VR={:.6f} Retention={:.6f} feasible={}'.format(
                step, met['safety_violation_rate'], met['drawdown_retention'],
                snapshots[-1]['feasible']), flush=True)
    optimization_sec = time.perf_counter()-opt_start
    full_x, full_f = np.asarray(all_x, dtype=np.float64), np.asarray(all_f, dtype=np.float64)
    selected_idx, policy = choose_train_solution(full_f, min_retention, max_violation)
    selected_vector = full_x[selected_idx].copy()
    selected_step = int(eval_steps[selected_idx])
    front_x, front_f = reduce_archive(full_x, full_f, args.archive_size)

    stage = time.perf_counter()
    write_csv(outdir/'famo_step_history.csv', trace)
    write_csv(outdir/'famo_train_snapshots.csv', snapshots)
    chosen = apply_vector([w.copy() for w in base_weights], specs, selected_vector)
    scope = check_weight_scope(base_weights, chosen, blocks)
    net.set_weights(chosen)
    best_file = outdir/'best_weights.h5'
    net.save_weights(str(best_file))
    check_weight_scope(base_weights, net.get_weights(), blocks)
    selected_train = calculate_metrics(net, name, ct_x, dr_x, session, args.batch_size)
    if (abs(selected_train['safety_violation_rate']-float(full_f[selected_idx, 0])) > 1.e-9 or
            abs(selected_train['drawdown_retention']+float(full_f[selected_idx, 1])) > 1.e-9):
        raise RuntimeError('Restored safety FAMO checkpoint metrics differ from selection metrics')
    front_dir = outdir/'pareto_front'
    front_dir.mkdir(exist_ok=True)
    records = []
    for j in range(min(len(front_x), args.save_front_limit)):
        w = apply_vector([a.copy() for a in base_weights], specs, front_x[j])
        check_weight_scope(base_weights, w, blocks)
        net.set_weights(w)
        path = front_dir/('weights_{}.h5'.format(j))
        net.save_weights(str(path))
        records.append({'index': j, 'train_violation': float(front_f[j, 0]),
                        'train_retention': float(-front_f[j, 1]), 'file': str(path)})
    net.set_weights(base_weights)
    write_csv(outdir/'pareto_front.txt', records)
    with open(str(outdir/'pareto_front.pkl'), 'wb') as f:
        pickle.dump({'X': front_x, 'F': front_f, 'all_evaluated_X': full_x,
                     'all_evaluated_F': full_f, 'all_evaluated_steps': eval_steps,
                     'selected_step': selected_step, 'selected_index_history': selected_idx,
                     'source_locator': loc['source_json'],
                     'note': 'FAMO TRAIN snapshots, not an evolutionary population'},
                    f, protocol=4)
    save_sec = time.perf_counter()-stage
    timing = {
        'localization_previous_run_sec': loc.get('source_runtime_sec'),
        'localization_previous_run_not_included_in_repair_wall': True,
        'localization_read_export_sec': float(loc_sec),
        'model_load_and_weight_mapping_sec': float(net_sec),
        'train_data_load_sec': float(data_sec),
        'famo_graph_build_sec': float(graph_sec),
        'optimization_wall_sec': float(optimization_sec),
        'pareto_and_checkpoint_save_sec': float(save_sec),
        'total_repair_run_wall_sec': float(time.perf_counter()-t0),
        'notes': 'FAMO steps are not MOEA/D FEs. TRAIN snapshot eval included in optimization wall.',
    }
    summary = {
        'schema': 'CGARepair_safety_FAMO_masked_weight_repair_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'safety', 'model': name, 'optimizer': 'famo', 'method': 'CGARepair_FAMO',
        'optimization_seed': int(args.seed), 'localization_seed': int(args.loc_seed),
        'localization': loc, 'localization_exported_dir': exported,
        'model_file': str(checkpoint), 'model_sha256': sha256_file(checkpoint),
        'selected_players': players, 'top_k': int(loc['top_k']),
        'total_ranked_neurons': int(loc['num_players']),
        'repair_specs': [{k: list(v) if isinstance(v, tuple) else v for k, v in s.items()}
                         for s in specs],
        'decision_variables': int(len(x0)),
        'parameter_budget_percent': float(100.*len(x0)/max(net.count_params(), 1)),
        'parameter_scope': 'Selected Conv/Dense Kernel output channels only; bias and BN frozen',
        'mask_audit': scope, 'train_data': provenance,
        'original_train_metrics': baseline,
        'optimization_proxy_losses': ['counterexample_negative_log_safe_action_probability',
                                      'drawdown_pretrained_action_cross_entropy'],
        'safety_action_rule': 'argmin_output',
        'safe_actions': safe, 'unsafe_actions': unsafe,
        'objective_definitions': ['safety_violation_rate=minimize',
                                  '-drawdown_retention=minimize'],
        'threshold_drawdown_retention': float(min_retention),
        'threshold_safety_violation': float(max_violation),
        'gradient_steps_requested': int(args.max_steps),
        'gradient_steps_actual': int(len(trace)),
        'train_snapshot_evaluations_actual': int(len(snapshots)),
        'front_size': int(len(front_x)),
        'front_feasible_count': int(np.sum((full_f[:, 0] <= max_violation) &
                                           (-full_f[:, 1] >= min_retention))),
        'selected_train_step': int(selected_step),
        'selected_history_index': int(selected_idx), 'selection_policy': policy,
        'selected_train_safety_violation': float(full_f[selected_idx, 0]),
        'selected_train_drawdown_retention': float(-full_f[selected_idx, 1]),
        'best_weights_file': str(best_file.resolve()),
        'optimizer_parameters': {
            'name': 'FAMO', 'max_steps': int(args.max_steps),
            'grad_batch_size_per_task': int(args.grad_batch_size),
            'evaluation_batch_size': int(args.batch_size),
            'full_train_eval_every': int(args.eval_every),
            'train_eval_samples': int(args.eval_samples),
            'model_lr': float(args.model_lr), 'task_lr': float(args.task_lr),
            'task_decay': float(args.task_decay), 'loss_eps': float(args.loss_eps),
            'max_grad_norm': float(args.max_grad_norm),
            'safety_temperature': float(args.safety_temperature),
            'search_radius': float(args.search_radius),
            'archive_size': int(args.archive_size),
            'gradient_algorithm': 'FAMO log-loss task weights + masked model Adam',
        },
        'task_logit_final': weights.w.tolist(), 'task_weight_final': weights.z.tolist(),
        'optimizer_history': snapshots, 'timing_sec': timing,
        'total_time_sec': float(timing['total_repair_run_wall_sec']),
    }
    temporary = outdir/'summary.json.incomplete'
    save_json(temporary, summary)
    os.replace(str(temporary), str(outdir/'summary.json'))
    print('[Done] {} FAMO TRAIN VR={:.6f}->{:.6f} Retention={:.6f}->{:.6f} step={}'.format(
        name, baseline['safety_violation_rate'], selected_train['safety_violation_rate'],
        baseline['drawdown_retention'], selected_train['drawdown_retention'], selected_step), flush=True)
    print('[Mask] modified={} / allowed={} outside_scope=0'.format(
        scope['modified_parameter_count'], scope['selected_parameter_count']), flush=True)
    print('[Files] summary={} best={}'.format(outdir/'summary.json', best_file), flush=True)
    return summary


def run_five_seeds(args):
    if args.seed is not None:
        raise ValueError('Omit positional optimizer seed with --five-seeds')
    if args.quick or args.overwrite:
        raise ValueError('Do not use --quick or --overwrite for five-seed formal experiments')
    if args.loc_seed != 256:
        raise ValueError('Fixed locator seed 256 is required for official five-seed comparison')
    check_args(args)
    name = model_name_checked(args.model_name)
    _, _, loc = load_ranking(name, args.loc_seed, args.loc_samples,
                             args.loc_json, args.loc_root)
    source = str(Path(loc['source_json']).resolve())
    test_script = Path(__file__).resolve().with_name('safety_famo_test_2.py')
    if not test_script.is_file():
        raise FileNotFoundError('Partner independent TEST script missing: {}'.format(test_script))
    records = []
    for seed in SEEDS:
        directory = run_directory(args.output_root, name, seed, 'famo', args.run_tag, False)
        summary_file, best, test_file = directory/'summary.json', directory/'best_weights.h5', directory/'test_results.json'
        print('\n========== {} FAMO seed={} TRAIN+TEST =========='.format(name, seed), flush=True)
        if summary_file.is_file() and best.is_file():
            print('[SKIP TRAIN] {}'.format(directory), flush=True)
        elif directory.exists() and any(directory.iterdir()):
            raise RuntimeError('Existing incomplete TRAIN run: {}'.format(directory))
        else:
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), name, str(seed),
                   '--loc-json', source, '--loc-seed', str(args.loc_seed),
                   '--loc-samples', str(args.loc_samples), '--output-root', args.output_root,
                   '--run-tag', args.run_tag, '--max-steps', str(args.max_steps),
                   '--eval-samples', str(args.eval_samples),
                   '--grad-batch-size', str(args.grad_batch_size),
                   '--batch-size', str(args.batch_size),
                   '--eval-every', str(args.eval_every), '--model-lr', str(args.model_lr),
                   '--task-lr', str(args.task_lr), '--task-decay', str(args.task_decay),
                   '--loss-eps', str(args.loss_eps), '--max-grad-norm', str(args.max_grad_norm),
                   '--safety-temperature', str(args.safety_temperature),
                   '--search-radius', str(args.search_radius),
                   '--archive-size', str(args.archive_size),
                   '--save-front-limit', str(args.save_front_limit), '--log-every', str(args.log_every)]
            if args.model_root:
                cmd += ['--model-root', args.model_root]
            if args.data_root:
                cmd += ['--data-root', args.data_root]
            if args.threshold_acc is not None:
                cmd += ['--threshold-acc', str(args.threshold_acc)]
            if args.threshold_safety is not None:
                cmd += ['--threshold-safety', str(args.threshold_safety)]
            if args.cpu:
                cmd += ['--cpu']
            subprocess.run(cmd, check=True)
        with summary_file.open('r', encoding='utf-8') as f:
            summary = json.load(f)
        if (summary.get('task') != 'safety' or summary.get('optimizer') != 'famo' or
                summary.get('model') != name or
                int(summary.get('optimization_seed', -1)) != seed or
                int(summary.get('localization_seed', -1)) != 256 or
                int(summary.get('gradient_steps_actual', -1)) != args.max_steps or
                Path(summary.get('localization', {}).get('source_json', '')).resolve() != Path(source)):
            raise ValueError('Existing TRAIN run identity mismatch: {}'.format(directory))
        hp = summary.get('optimizer_parameters', {})
        expected = {
            'max_steps': args.max_steps, 'model_lr': args.model_lr,
            'task_lr': args.task_lr, 'task_decay': args.task_decay,
            'train_eval_samples': args.eval_samples,
            'grad_batch_size_per_task': args.grad_batch_size,
            'full_train_eval_every': args.eval_every,
            'evaluation_batch_size': args.batch_size,
            'loss_eps': args.loss_eps, 'max_grad_norm': args.max_grad_norm,
            'safety_temperature': args.safety_temperature,
            'search_radius': args.search_radius, 'archive_size': args.archive_size,
        }
        for k, v in expected.items():
            if k not in hp or abs(float(hp[k])-float(v)) > 1.e-9:
                raise ValueError('TRAIN hyperparameter mismatch {} in {}'.format(k, directory))
        if abs(summary.get('threshold_drawdown_retention', -1) - (
                CONFIG[name]['threshold_acc'] if args.threshold_acc is None else args.threshold_acc)) > 1.e-9:
            raise ValueError('TRAIN retention guard mismatch {}'.format(directory))
        if abs(summary.get('threshold_safety_violation', -1) - (
                CONFIG[name]['threshold_safety'] if args.threshold_safety is None else args.threshold_safety)) > 1.e-9:
            raise ValueError('TRAIN safety target mismatch {}'.format(directory))
        if not best.is_file():
            raise FileNotFoundError('TRAIN-selected weights missing: {}'.format(best))
        if not test_file.is_file():
            cmd = [sys.executable, '-u', str(test_script), name, str(seed),
                   '--optimizer', 'famo', '--run-dir', str(directory),
                   '--test-samples', '0', '--batch-size', str(args.batch_size)]
            if args.model_root:
                cmd += ['--model-root', args.model_root]
            if args.data_root:
                cmd += ['--data-root', args.data_root]
            subprocess.run(cmd, check=True)
        with test_file.open('r', encoding='utf-8') as f:
            result = json.load(f)
        if (result.get('optimizer') != 'famo' or result.get('model') != name or
                int(result.get('optimization_seed', -1)) != seed or
                int(result.get('test_samples_requested', -1)) != 0 or
                result.get('checkpoint_selection') != 'TRAIN only; TEST never selects checkpoints' or
                Path(result.get('selected_checkpoint', '')).resolve() != best.resolve()):
            raise ValueError('Existing TEST file identity/protocol mismatch: {}'.format(test_file))
        original, repaired = result['original_test_metrics'], result['repaired_test_metrics']
        records.append({
            'seed': seed, 'selected_train_step': int(summary['selected_train_step']),
            'original_violation_percent': 100.*float(original['safety_violation_rate']),
            'repaired_violation_percent': 100.*float(repaired['safety_violation_rate']),
            'original_retention_percent': 100.*float(original['drawdown_retention']),
            'repaired_retention_percent': 100.*float(repaired['drawdown_retention']),
            'test_file': str(test_file.resolve()),
        })
    def mean_sd(key):
        vals = [r[key] for r in records]
        return float(statistics.mean(vals)), float(statistics.stdev(vals))
    vr_mu, vr_sd = mean_sd('repaired_violation_percent')
    ret_mu, ret_sd = mean_sd('repaired_retention_percent')
    print('\n========== {} SAFETY FAMO FIVE-SEED TEST =========='.format(name.upper()), flush=True)
    print('Seed        TEST VR(%)     TEST Retention(%)', flush=True)
    for r in records:
        print('{:<8d} {:>12.6f} {:>18.6f}'.format(
            r['seed'], r['repaired_violation_percent'], r['repaired_retention_percent']), flush=True)
    print('TEST VR        MEAN +/- SD = {:.6f} +/- {:.6f}%'.format(vr_mu, vr_sd), flush=True)
    print('TEST Retention MEAN +/- SD = {:.6f} +/- {:.6f}%'.format(ret_mu, ret_sd), flush=True)
    dest = Path(args.output_root).expanduser()/name/'CGARepair'/(
        'famo_{}_five_seed_TEST_summary.json'.format(args.run_tag))
    save_json(dest, {
        'schema': 'CGARepair_safety_FAMO_5seed_TEST_v1',
        'model': name, 'optimizer': 'famo', 'seeds': list(SEEDS),
        'records': records, 'violation_mean_percent': vr_mu,
        'violation_sample_sd_percent': vr_sd,
        'retention_mean_percent': ret_mu,
        'retention_sample_sd_percent': ret_sd,
        'notes': 'All TEST checkpoints selected by TRAIN only. FAMO gradient steps != MOEA/D FEs.',
    })
    print('[FIVE-SEED SUMMARY] {}'.format(dest), flush=True)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.five_seeds:
        run_five_seeds(args)
    else:
        if args.seed is None:
            parser.error('Single seed required without --five-seeds')
        run(args)


if __name__ == '__main__':
    main()
