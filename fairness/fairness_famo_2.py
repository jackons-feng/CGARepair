"""CGARepair fairness local-Kernel FAMO repair (TF1 graph compatible).

Preserves GradAct localization and the original fairness TEST metric: the
absolute difference in 0/1 classification accuracy between sensitive groups.

FAMO optimization tasks:
  1) differentiable, temperature-smoothed group accuracy gap;
  2) normal binary classification cross-entropy on a population-sampled batch.

The *true* discrete TRAIN Gap and Accuracy select a saved snapshot, not TEST.
Only the original GradAct-selected output-channel Kernel weights can change.
Five-seed mode runs one independent process per seed and calls the separate
fairness_famo_test_2.py for full hold-out TEST after TRAIN selection.

Depends on original project modules read_CGARepair.py,
cgarepair_fairness_utils.py. No MOEA/D optimizer is invoked.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from read_CGARepair import load_ranking, export_ranking, verify_model_signature
from cgarepair_fairness_utils import (
    check_dataset, model_path_for, load_fairness_split, set_up_tf_graph,
    evaluate_model, build_neuron_specs, apply_vector, save_json, sha256_file,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
SEEDS = (42, 123, 256, 1024, 2026)


def softmax(weights):
    x = np.asarray(weights, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class FAMOTaskWeights:
    """Two-task log-loss FAMO weighting, with Adam task-logit updates."""
    def __init__(self, lr=0.025, gamma=0.001, eps=1.e-8):
        if lr <= 0 or gamma < 0 or eps <= 0:
            raise ValueError('Invalid FAMO task-weight parameters')
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
            raise FloatingPointError('Invalid/nonfinite FAMO task losses')
        if np.any(b < 0) or np.any(a < 0):
            raise ValueError('FAMO loss must be nonnegative')
        delta = np.log(b + self.eps) - np.log(a + self.eps)
        z = self.z
        g = z * (delta - np.dot(z, delta)) + self.gamma * self.w
        self.k += 1
        self.m = .9 * self.m + .1 * g
        self.v = .999 * self.v + .001 * (g * g)
        mhat = self.m / (1 - .9 ** self.k)
        vhat = self.v / (1 - .999 ** self.k)
        self.w = np.clip(self.w - self.lr * mhat / (np.sqrt(vhat) + 1.e-8), -30., 30.)
        return delta


def binary_labels(labels, name='labels'):
    a = np.asarray(labels)
    if a.ndim > 1 and a.shape[-1] == 2:
        a = np.argmax(a, axis=-1)
    a = np.asarray(a).reshape(-1)
    if not np.all(np.isin(a, (0, 1))):
        raise ValueError('{} must be binary 0/1; values={}'.format(name, np.unique(a)[:15]))
    return a.astype(np.int32)


def vector_from_weights(weights, specs):
    return np.concatenate([
        np.asarray(weights[int(s['kernel_weight_idx'])][..., int(s['channel_idx'])]).reshape(-1)
        for s in specs
    ]).astype(np.float64)


def make_blocks(model, base_weights, specs, lower, upper):
    if not specs or len(lower) != len(upper):
        raise ValueError('Empty specs or inconsistent bounds')
    mapping = {}
    cursor = 0
    for layer_index, layer in enumerate(model.layers):
        local = layer.get_weights()
        if local and ('Dense' in type(layer).__name__ or 'Conv' in type(layer).__name__):
            mapping[cursor] = (layer_index, layer)
        cursor += len(local)
    if cursor != len(base_weights):
        raise RuntimeError('Keras layer/global weight offsets mismatch')
    blocks = {}
    vector_cursor = 0
    for spec in specs:
        idx = int(spec['kernel_weight_idx'])
        channel = int(spec['channel_idx'])
        if idx not in mapping:
            raise ValueError('Kernel index {} not a Conv/Dense first Kernel'.format(idx))
        original = base_weights[idx]
        layer_idx, layer = mapping[idx]
        if tuple(layer.weights[0].shape.as_list()) != tuple(original.shape):
            raise ValueError('Keras Kernel mapping shape mismatch at {}'.format(idx))
        if not 0 <= channel < original.shape[-1]:
            raise ValueError('Invalid output channel index {} for {}'.format(channel, original.shape))
        item = blocks.setdefault(idx, {
            'layer_index': layer_idx,
            'mask': np.zeros(original.shape, dtype=np.float32),
            'lower': np.zeros(original.shape, dtype=np.float32),
            'upper': np.zeros(original.shape, dtype=np.float32),
        })
        if np.any(item['mask'][..., channel] != 0):
            raise ValueError('Duplicated selected Kernel output channel')
        n = int(original[..., channel].size)
        if vector_cursor + n > len(lower):
            raise ValueError('Selected parameter vector/bounds length mismatch')
        item['mask'][..., channel] = 1.
        item['lower'][..., channel] = lower[vector_cursor:vector_cursor+n].reshape(original[..., channel].shape)
        item['upper'][..., channel] = upper[vector_cursor:vector_cursor+n].reshape(original[..., channel].shape)
        if np.any(item['lower'][..., channel] >= item['upper'][..., channel]):
            raise ValueError('Degenerate selected parameter bounds')
        vector_cursor += n
    if vector_cursor != len(lower):
        raise ValueError('FAMO mask did not cover entire parameter vector')
    return blocks


def audit_scope(before, after, blocks):
    if len(before) != len(after):
        raise RuntimeError('Weight list length differs')
    modified = 0
    for idx, (a, b) in enumerate(zip(before, after)):
        if a.shape != b.shape or not np.all(np.isfinite(b)):
            raise RuntimeError('Invalid repaired tensor {}'.format(idx))
        if idx not in blocks:
            if not np.array_equal(a, b):
                raise RuntimeError('FAMO modified unselected tensor {}'.format(idx))
        else:
            blk = blocks[idx]
            allowed = blk['mask'].astype(bool)
            if not np.array_equal(a[~allowed], b[~allowed]):
                raise RuntimeError('FAMO modified an unselected Kernel weight {}'.format(idx))
            if (np.any(b[allowed] < blk['lower'][allowed] - 1.e-5) or
                    np.any(b[allowed] > blk['upper'][allowed] + 1.e-5)):
                raise RuntimeError('FAMO selected weight outside bounds {}'.format(idx))
            modified += int(np.count_nonzero(a[allowed] != b[allowed]))
    return {
        'modified_parameter_count': modified,
        'selected_parameter_count': int(sum(int(np.count_nonzero(x['mask'])) for x in blocks.values())),
        'outside_scope_modified_count': 0,
        'other_tensor_untouched': True,
    }


def detect_output_mode(session, model, X, requested):
    if requested != 'auto':
        return requested
    out = np.asarray(session.run(model.output, feed_dict={model.input: X[:min(len(X), 32)]}))
    if out.ndim != 2 or out.shape[1] not in (1, 2):
        raise ValueError('Fairness FAMO supports sigmoid [N,1] or softmax [N,2]; got {}'.format(out.shape))
    if out.shape[1] == 1:
        if np.all(out >= -1.e-6) and np.all(out <= 1 + 1.e-6):
            return 'probabilities'
    elif (np.all(out >= -1.e-6) and np.all(out <= 1 + 1.e-6) and
          np.allclose(np.sum(out, axis=1), 1., atol=.02, rtol=.02)):
        return 'probabilities'
    return 'logits'


class MaskedFairnessFAMOGraph:
    """Soft group-correctness gap + population classification CE, masked Adam."""
    def __init__(self, tf, session, network, base_weights, blocks, model_lr,
                 clip_norm, loss_eps, temperature, output_mode):
        if temperature <= 0:
            raise ValueError('Fairness smooth temperature must be positive')
        v1 = tf.compat.v1
        self.session = session
        self.x = network.input
        self.n_group = v1.placeholder(tf.int32, shape=[], name='famo_fair_per_group_count')
        self.y0 = v1.placeholder(tf.float32, shape=[None], name='famo_fair_labels_group0')
        self.y1 = v1.placeholder(tf.float32, shape=[None], name='famo_fair_labels_group1')
        self.yutil = v1.placeholder(tf.float32, shape=[None], name='famo_fair_labels_utility')
        self.z = v1.placeholder(tf.float32, shape=[2], name='famo_fair_task_z')
        self.blocks = blocks
        out = tf.cast(network.output, tf.float32)
        if out.shape.ndims != 2 or out.shape.as_list()[-1] not in (1, 2):
            raise ValueError('Fairness FAMO expects output shape [N,1] or [N,2]')
        k = out.shape.as_list()[-1]
        if output_mode not in ('logits', 'probabilities'):
            raise ValueError('Invalid output-mode: {}'.format(output_mode))
        if k == 1:
            raw = tf.reshape(out, (-1,))
            if output_mode == 'probabilities':
                probs = tf.clip_by_value(raw, 1.e-7, 1. - 1.e-7)
                margin = tf.math.log(probs) - tf.math.log1p(-probs)
            else:
                margin = raw
        else:
            if output_mode == 'probabilities':
                probs = tf.clip_by_value(out, 1.e-7, 1.)
                margin = tf.math.log(probs[:, 1]) - tf.math.log(probs[:, 0])
            else:
                margin = out[:, 1] - out[:, 0]
        a, b = margin[:self.n_group], margin[self.n_group:2*self.n_group]
        util = margin[2*self.n_group:]
        # Smooth correct-prediction indicator: sigmoid((2y-1)*logit_margin/tau).
        corr0 = tf.math.sigmoid((2. * self.y0 - 1.) * a / float(temperature))
        corr1 = tf.math.sigmoid((2. * self.y1 - 1.) * b / float(temperature))
        gap = tf.reduce_mean(corr0) - tf.reduce_mean(corr1)
        fair_loss = tf.sqrt(tf.square(gap) + float(loss_eps) ** 2)
        utility_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.yutil, logits=util))
        self.task_losses = tf.stack([fair_loss, utility_ce], name='famo_fairness_proxy_and_BCE')
        positive = tf.maximum(self.task_losses, 0.) + float(loss_eps)
        denom = tf.stop_gradient(tf.reduce_sum(self.z / positive))
        combined = tf.reduce_sum(self.z * tf.math.log(positive)) / denom
        selected_vars, masks, bounds = [], [], []
        for idx, blk in sorted(blocks.items()):
            kernel = network.layers[blk['layer_index']].weights[0]
            if tuple(kernel.shape.as_list()) != tuple(base_weights[idx].shape):
                raise RuntimeError('Kernel variable mismatch at {}'.format(idx))
            selected_vars.append(kernel)
            dtype = kernel.dtype.base_dtype
            masks.append(tf.constant(blk['mask'], dtype=dtype))
            bounds.append((tf.constant(blk['lower'], dtype=dtype),
                           tf.constant(blk['upper'], dtype=dtype)))
        gradients = tf.gradients(combined, selected_vars)
        if any(g is None for g in gradients):
            raise RuntimeError('Selected fairness Kernel has no output gradient')
        masked = []
        for grad, mask in zip(gradients, masks):
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
            masked.append(grad * mask)
        self.gradnorm = v1.global_norm(masked)
        if clip_norm > 0:
            masked, _ = tf.clip_by_global_norm(masked, float(clip_norm), use_norm=self.gradnorm)
        old = {v.name for v in v1.global_variables()}
        optimizer = v1.train.AdamOptimizer(learning_rate=float(model_lr))
        train = optimizer.apply_gradients(list(zip(masked, selected_vars)))
        project = []
        with tf.control_dependencies([train]):
            for kernel, mask, (lo, hi) in zip(selected_vars, masks, bounds):
                clipped = tf.clip_by_value(kernel, lo, hi)
                project.append(v1.assign(kernel, tf.where(tf.greater(mask, 0), clipped, kernel)))
        self.train_op = tf.group(*project, name='famo_fair_masked_adam')
        new_vars = [v for v in v1.global_variables() if v.name not in old]
        if new_vars:
            session.run(v1.variables_initializer(new_vars))

    def feed(self, x0, y0, x1, y1, xu, yu, z):
        n = len(x0)
        if n < 1 or len(x1) != n or len(xu) < 1:
            raise ValueError('Both sensitive groups and utility batch must be nonempty')
        return {
            self.x: np.concatenate((x0, x1, xu), axis=0),
            self.n_group: int(n),
            self.y0: np.asarray(y0, dtype=np.float32),
            self.y1: np.asarray(y1, dtype=np.float32),
            self.yutil: np.asarray(yu, dtype=np.float32),
            self.z: np.asarray(z, dtype=np.float32),
        }

    def step(self, feed):
        before, _, norm = self.session.run(
            [self.task_losses, self.train_op, self.gradnorm], feed_dict=feed)
        after = self.session.run(self.task_losses, feed_dict=feed)
        if (not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)) or
                not np.isfinite(norm)):
            raise FloatingPointError('Nonfinite FAMO fairness proxy/gradient')
        return np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64), float(norm)


def nondominated_archive(X, F, max_size):
    """Diagnostic only; final checkpoint selection always sees ALL snapshots."""
    F = np.asarray(F, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    chosen = []
    for i, f in enumerate(F):
        if any(np.all(F[j] <= f + 1.e-12) and np.any(F[j] < f - 1.e-12)
               for j in range(len(F)) if j != i):
            continue
        if any(np.allclose(F[i], F[j], atol=1.e-12, rtol=0.) for j in chosen):
            continue
        chosen.append(i)
    if len(chosen) > max_size:
        # Preserve minimum-Gap and maximum-Accuracy objective extremes.
        order = sorted(chosen, key=lambda j: (F[j, 0], F[j, 1], j))
        chosen = order[:max_size-1] + [max(chosen, key=lambda j: -F[j, 1])]
        chosen = list(dict.fromkeys(chosen))
    return X[chosen], F[chosen], chosen


def select_train(F, baseline_gap, min_acc, gap_target):
    gap, acc = F[:, 0], -F[:, 1]
    ids = np.flatnonzero((gap <= baseline_gap + 1.e-12) & (acc >= min_acc - 1.e-12))
    if len(ids) == 0:
        return None, 'no_train_candidate_meets_accuracy_and_no_fairness_regression'
    target = ids[gap[ids] <= gap_target + 1.e-12] if gap_target is not None else np.array([], dtype=int)
    policy = 'utility_guard_min_gap_then_max_acc'
    if len(target):
        ids = target
        policy = 'target_feasible_min_gap_then_max_acc'
    order = np.lexsort((-acc[ids], gap[ids]))
    return int(ids[order[0]]), policy


def write_csv(path, rows):
    if not rows:
        return
    with open(str(path), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def output_dir(args):
    suffix = ('_' + args.run_tag) if args.run_tag else ('_quick' if args.quick else '')
    if args.quick and args.run_tag and not suffix.endswith('_quick'):
        suffix += '_quick'
    return Path(args.output_root).expanduser() / args.dataset / 'CGARepair' / ('seed_' + str(args.seed)) / ('famo' + suffix)


def get_parser():
    p = argparse.ArgumentParser(description='CGARepair fairness GradAct + masked FAMO repair')
    p.add_argument('dataset', choices=('bank', 'adult', 'default'))
    p.add_argument('seed', nargs='?', type=int, help='Optimizer seed; omit for --five-seeds')
    p.add_argument('--optimizer', choices=('famo',), default='famo')
    p.add_argument('--five-seeds', action='store_true', help='Five independent TRAIN + full TEST runs')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=4096)
    p.add_argument('--loc-json')
    p.add_argument('--loc-root')
    p.add_argument('--top-k', type=int)
    p.add_argument('--skip-signature-check', action='store_true')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-tag', default='paper_gamma001_steps5000')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--quick', action='store_true', help='Five-step smoke TEST, separate result directory')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--eval-samples', type=int, default=10000,
                   help='TRAIN pool; 0 means full TRAIN (not TEST)')
    p.add_argument('--grad-batch-size', type=int, default=64,
                   help='Per sensitive group, plus same-size population utility batch')
    p.add_argument('--eval-batch-size', type=int, default=1024)
    p.add_argument('--eval-every', type=int, default=20)
    p.add_argument('--model-lr', type=float, default=.001)
    p.add_argument('--task-lr', type=float, default=.025)
    p.add_argument('--task-decay', type=float, default=.001)
    p.add_argument('--loss-eps', type=float, default=1.e-8)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--fair-temperature', type=float, default=.5)
    p.add_argument('--output-mode', choices=('auto', 'probabilities', 'logits'), default='auto')
    p.add_argument('--search-radius', type=float, default=10.,
                   help='Same original fairness MOEA/D bounds intersected with [-5,+5]')
    p.add_argument('--max-acc-drop', type=float, default=.02,
                   help='TRAIN accuracy guard as fraction, default original fairness 2pp')
    p.add_argument('--min-acc', type=float, default=None)
    p.add_argument('--gap-target', type=float, default=.05)
    p.add_argument('--archive-size', type=int, default=120)
    p.add_argument('--save-front-limit', type=int, default=10)
    p.add_argument('--log-every', type=int, default=100)
    return p


def check_args(args):
    if args.quick:
        args.run_tag = 'smoke_5steps'
        args.max_steps = 5
        args.eval_samples = min(int(args.eval_samples), 256) if args.eval_samples else 256
        args.grad_batch_size = min(int(args.grad_batch_size), 16)
        args.eval_every = 1
    if (args.max_steps < 1 or args.grad_batch_size < 1 or args.eval_samples < 0 or
            args.eval_batch_size < 1 or args.eval_every < 1 or args.log_every < 1 or
            args.task_lr <= 0 or args.model_lr <= 0 or args.task_decay < 0 or
            args.loss_eps <= 0 or args.max_grad_norm < 0 or args.fair_temperature <= 0 or
            args.search_radius <= 0 or args.archive_size < 2 or args.save_front_limit < 0):
        raise ValueError('Invalid FAMO fairness optimization parameters')
    if not 0 <= args.max_acc_drop <= 1:
        raise ValueError('--max-acc-drop must be in [0,1]')
    if args.min_acc is not None and not 0 <= args.min_acc <= 1:
        raise ValueError('--min-acc must be in [0,1]')
    if args.gap_target is not None and not 0 <= args.gap_target <= 1:
        raise ValueError('--gap-target must be in [0,1]')


def run(args):
    t0 = time.perf_counter()
    args.dataset = check_dataset(args.dataset)
    check_args(args)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    directory = output_dir(args)
    if directory.exists() and any(directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError('Existing/incomplete FAMO directory: {}'.format(directory))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup = directory.with_name(directory.name + '_backup_' + stamp)
        directory.rename(backup)
        print('[BACKUP] {} -> {}'.format(directory, backup), flush=True)
    directory.mkdir(parents=True, exist_ok=True)

    stage = time.perf_counter()
    checkpoint_path = model_path_for(args.dataset, args.model_root)
    players, scores, provenance = load_ranking(
        args.dataset, args.loc_seed, args.loc_samples, args.loc_json,
        args.loc_root, top_k=args.top_k)
    signature_check = ('skipped_by_user' if args.skip_signature_check else
                       verify_model_signature(provenance, checkpoint_path))
    exported = export_ranking(players, scores, provenance, directory/'localization_input')
    loc_sec = time.perf_counter()-stage
    print('[Locator] {} Top-{} from {} (loc={} opt={})'.format(
        args.dataset, len(players), provenance['source_json'], args.loc_seed, args.seed), flush=True)

    stage = time.perf_counter()
    np.random.seed(int(args.seed))
    tf, K, session = set_up_tf_graph()
    tf.compat.v1.set_random_seed(int(args.seed))
    try:
        K.set_learning_phase(0)
    except AttributeError:
        tf.compat.v1.keras.backend.set_learning_phase(0)
    from tensorflow.keras.models import load_model
    model = load_model(str(checkpoint_path), compile=False)
    specs, x0, original_weights = build_neuron_specs(model, players)
    original_weights = [w.copy() for w in original_weights]
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    reconstructed = vector_from_weights(original_weights, specs)
    if x0.shape != reconstructed.shape or not np.allclose(x0, reconstructed, rtol=0., atol=1.e-7):
        raise ValueError('Original fairness parameter vector does not match selected Kernel slices')
    lower = np.maximum(-5., x0 - args.search_radius)
    upper = np.minimum(5., x0 + args.search_radius)
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError('Original weight outside original MOEA/D search bounds')
    blocks = make_blocks(model, original_weights, specs, lower, upper)
    model_sec = time.perf_counter()-stage

    stage = time.perf_counter()
    X, Y, G, manifest = load_fairness_split(
        args.dataset, 'train', args.data_root,
        sample_count=args.eval_samples, sample_seed=args.seed)
    X = np.asarray(X, dtype=np.float32)
    y = binary_labels(Y, 'class labels')
    g = binary_labels(G, 'sensitive groups')
    if len(X) != len(y) or len(y) != len(g):
        raise ValueError('Fairness X/Y/G sample sizes disagree')
    group0, group1 = np.flatnonzero(g == 0), np.flatnonzero(g == 1)
    if len(group0) == 0 or len(group1) == 0:
        raise ValueError('Both sensitive groups must occur in TRAIN pool')
    declared = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if tuple(declared[1:]) != tuple(X.shape[1:]):
        raise ValueError('Model expected {}, actual TRAIN shape {}'.format(declared, X.shape))
    baseline = evaluate_model(model, X, Y, G, session,
                              args.eval_batch_size, include_f1=True)
    min_acc = (float(args.min_acc) if args.min_acc is not None else
               max(0., float(baseline['accuracy']) - args.max_acc_drop))
    train_sec = time.perf_counter()-stage
    print('[Baseline] {} TRAIN Gap={:.6f} Acc={:.6f} F1={:.6f} minAcc={:.6f} '
          'g0={} g1={} d={}'.format(args.dataset, baseline['fairness_gap'],
          baseline['accuracy'], baseline['f1'], min_acc,
          len(group0), len(group1), len(x0)), flush=True)

    stage = time.perf_counter()
    output_mode = detect_output_mode(session, model, X, args.output_mode)
    graph = MaskedFairnessFAMOGraph(
        tf, session, model, original_weights, blocks, args.model_lr,
        args.max_grad_norm, args.loss_eps, args.fair_temperature, output_mode)
    graph_sec = time.perf_counter()-stage
    task_weights = FAMOTaskWeights(args.task_lr, args.task_decay, args.loss_eps)
    rng = np.random.RandomState(int(args.seed))
    trace, snapshots, all_x, all_f, eval_steps = [], [], [], [], []

    def record(step, elapsed):
        met = evaluate_model(model, X, Y, G, session,
                             args.eval_batch_size, include_f1=True)
        f = [float(met['fairness_gap']), -float(met['accuracy'])]
        if not np.all(np.isfinite(f)):
            raise FloatingPointError('Nonfinite TRAIN fairness metrics')
        all_x.append(vector_from_weights(model.get_weights(), specs))
        all_f.append(f)
        eval_steps.append(int(step))
        snapshots.append({
            'step': int(step), 'train_gap': f[0], 'train_accuracy': -f[1],
            'train_f1': float(met['f1']),
            'train_eligible': bool(f[0] <= baseline['fairness_gap'] + 1.e-12 and
                                   -f[1] >= min_acc - 1.e-12),
            'z_fair': float(task_weights.z[0]), 'z_utility': float(task_weights.z[1]),
            'elapsed_opt_sec': float(elapsed),
        })
        return met

    record(0, 0.)
    start_opt = time.perf_counter()
    print('[FAMO] {} steps={} smooth_Gap(tau={}) + population_BCE, '
          'per_group_batch={} utility_batch={} eval_every={}'.format(
              args.dataset, args.max_steps, args.fair_temperature,
              args.grad_batch_size, args.grad_batch_size, args.eval_every), flush=True)
    for step in range(1, args.max_steps+1):
        i0 = group0[rng.randint(len(group0), size=args.grad_batch_size)]
        i1 = group1[rng.randint(len(group1), size=args.grad_batch_size)]
        iu = rng.randint(len(y), size=args.grad_batch_size)
        feed = graph.feed(X[i0], y[i0], X[i1], y[i1], X[iu], y[iu], task_weights.z)
        before, after, norm = graph.step(feed)
        delta = task_weights.update(before, after)
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print('[FAMO step {:04d}/{}] fairLoss={:.6f}->{:.6f} '
                  'utilityCE={:.6f}->{:.6f} z=({:.3f},{:.3f}) gradnorm={:.5f}'.format(
                      step, args.max_steps, before[0], after[0], before[1], after[1],
                      task_weights.z[0], task_weights.z[1], norm), flush=True)
        trace.append({
            'step': step, 'fair_loss_pre': float(before[0]),
            'fair_loss_post': float(after[0]),
            'utility_ce_pre': float(before[1]),
            'utility_ce_post': float(after[1]),
            'fair_log_improvement': float(delta[0]),
            'utility_log_improvement': float(delta[1]),
            'z_fair': float(task_weights.z[0]),
            'z_utility': float(task_weights.z[1]),
            'gradient_norm': float(norm),
            'elapsed_opt_sec': float(time.perf_counter()-start_opt),
        })
        if step % args.eval_every == 0 or step == args.max_steps:
            met = record(step, time.perf_counter()-start_opt)
            print('[TRAIN step={}] Gap={:.6f} Acc={:.6f} F1={:.6f} eligible={}'.format(
                step, met['fairness_gap'], met['accuracy'], met['f1'],
                snapshots[-1]['train_eligible']), flush=True)
    optimization_sec = time.perf_counter()-start_opt
    full_x, full_f = np.asarray(all_x, dtype=np.float64), np.asarray(all_f, dtype=np.float64)
    selected_idx, policy = select_train(
        full_f, float(baseline['fairness_gap']), min_acc, args.gap_target)
    if selected_idx is None:
        selected_vec = x0.copy()
        selected_step = 0
    else:
        selected_vec = full_x[selected_idx].copy()
        selected_step = int(eval_steps[selected_idx])
    candidate_x, candidate_f, diagnostic_indices = nondominated_archive(
        full_x, full_f, args.archive_size)

    stage = time.perf_counter()
    write_csv(directory/'famo_step_history.csv', trace)
    write_csv(directory/'famo_train_snapshots.csv', snapshots)
    selected_weights = [w.copy() for w in original_weights]
    apply_vector(selected_weights, specs, selected_vec)
    audit = audit_scope(original_weights, selected_weights, blocks)
    model.set_weights(selected_weights)
    best_weights = directory/'best_weights.h5'
    model.save_weights(str(best_weights))
    audit_scope(original_weights, model.get_weights(), blocks)
    best_train = evaluate_model(model, X, Y, G, session, args.eval_batch_size, include_f1=True)
    if (abs(best_train['fairness_gap'] - (full_f[selected_idx, 0] if selected_idx is not None
                                          else baseline['fairness_gap'])) > 1.e-8 or
            abs(best_train['accuracy'] - (float(-full_f[selected_idx, 1]) if selected_idx is not None
                                            else baseline['accuracy'])) > 1.e-8):
        raise RuntimeError('Restored TRAIN-selected FAMO snapshot does not reproduce its metrics')
    frontdir = directory/'pareto_front'
    frontdir.mkdir(exist_ok=True)
    save_ids = list(range(min(len(candidate_x), args.save_front_limit)))
    diagnostics = []
    for j in save_ids:
        w = [v.copy() for v in original_weights]
        apply_vector(w, specs, candidate_x[j])
        audit_scope(original_weights, w, blocks)
        model.set_weights(w)
        f = frontdir/('weights_{}.h5'.format(j))
        model.save_weights(str(f))
        diagnostics.append({'index': j, 'train_gap': float(candidate_f[j, 0]),
                            'train_acc': float(-candidate_f[j, 1]), 'file': str(f)})
    model.set_weights(original_weights)
    with open(str(directory/'pareto_front.pkl'), 'wb') as f:
        pickle.dump({'X': candidate_x, 'F': candidate_f, 'all_evaluated_X': full_x,
                     'all_evaluated_F': full_f, 'all_evaluated_steps': eval_steps,
                     'selected_step': selected_step, 'selected_history_index': selected_idx,
                     'source_locator': provenance['source_json'],
                     'note': 'Historical FAMO TRAIN snapshots, NOT an evolutionary population'},
                    f, protocol=4)
    write_csv(directory/'pareto_front'/'pareto_front.csv', diagnostics)
    save_sec = time.perf_counter()-stage
    total_wall = time.perf_counter()-t0
    timing = {
        'localization_previous_run_sec': provenance.get('source_runtime_sec'),
        'localization_previous_run_excluded_from_current_wall': True,
        'localization_read_sec': float(loc_sec),
        'train_data_model_baseline_sec': float(train_sec + model_sec),
        'famo_graph_build_sec': float(graph_sec),
        'optimization_wall_sec': float(optimization_sec),
        'save_results_sec': float(save_sec),
        'repair_run_total_wall_sec': float(total_wall),
        'notes': 'TRAIN snapshot evaluation is included in optimization time; FAMO steps != MOEA/D FEs',
    }
    summary = {
        'schema': 'CGARepair_fairness_FAMO_masked_weight_repair_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'fairness', 'method': 'CGARepair_FAMO', 'optimizer': 'famo',
        'dataset': args.dataset, 'seed': int(args.seed), 'loc_seed': int(args.loc_seed),
        'localization': provenance, 'model_signature_check': signature_check,
        'model_path': str(checkpoint_path), 'model_sha256': sha256_file(checkpoint_path),
        'train_data': manifest, 'evaluation_scope': 'train_only',
        'test_used_for_selection': False,
        'fairness_metric': 'absolute_gap_of_sensitive_group_classification_accuracies',
        'optimization_proxy_losses': ['temperature_smoothed_group_accuracy_gap',
                                      'population_sampled_binary_cross_entropy'],
        'selected_neurons': players, 'parameter_specs': specs,
        'decision_variables': int(len(x0)),
        'total_original_model_parameters': int(sum(int(w.size) for w in original_weights)),
        'mask_audit': audit,
        'baseline_train': baseline, 'repaired_train': best_train,
        'delta_fairness_gap': float(best_train['fairness_gap'] - baseline['fairness_gap']),
        'delta_accuracy': float(best_train['accuracy'] - baseline['accuracy']),
        'min_train_accuracy': float(min_acc), 'target_train_gap': float(args.gap_target),
        'selected_train_step': int(selected_step),
        'selected_history_index': selected_idx,
        'selection_policy': policy,
        'train_accuracy_guard_met': bool(best_train['accuracy'] >= min_acc - 1.e-12),
        'train_gap_nonregression_met': bool(best_train['fairness_gap'] <= baseline['fairness_gap']+1.e-12),
        'gradient_steps_requested': int(args.max_steps),
        'gradient_steps_actual': int(len(trace)),
        'train_snapshot_evaluations_actual': int(len(snapshots)),
        'historical_pareto_size': int(len(candidate_x)),
        'optimizer_params': {
            'name': 'FAMO', 'max_steps': int(args.max_steps),
            'train_eval_samples': int(args.eval_samples),
            'grad_batch_size_per_group_and_utility': int(args.grad_batch_size),
            'full_train_eval_every': int(args.eval_every),
            'evaluation_batch_size': int(args.eval_batch_size),
            'model_lr': float(args.model_lr), 'task_lr': float(args.task_lr),
            'task_decay': float(args.task_decay), 'loss_eps': float(args.loss_eps),
            'max_grad_norm': float(args.max_grad_norm),
            'fair_temperature': float(args.fair_temperature),
            'output_mode': output_mode, 'search_radius': float(args.search_radius),
            'max_acc_drop': float(args.max_acc_drop), 'gap_target': float(args.gap_target),
            'archive_size': int(args.archive_size),
            'gradient_algorithm': 'FAMO two-task log-loss reweighting; masked Adam and bounds',
        },
        'task_logit_final': task_weights.w.tolist(),
        'task_weights_final': task_weights.z.tolist(),
        'optimizer_history': snapshots,
        'timing_sec': timing,
        'best_weights_file': str(best_weights.resolve()),
        'files': {'best_weights': 'best_weights.h5',
                  'pareto_vectors': 'pareto_front.pkl',
                  'diagnostic_front': 'pareto_front/pareto_front.csv'},
    }
    temporary = directory/'summary.json.incomplete'
    save_json(temporary, summary)
    os.replace(str(temporary), str(directory/'summary.json'))
    save_json(directory/'time_summary.json', timing)
    print('[Done] {} FAMO steps={} TRAIN Gap={:.6f}->{:.6f} Acc={:.6f}->{:.6f} step={}'.format(
        args.dataset, args.max_steps, baseline['fairness_gap'], best_train['fairness_gap'],
        baseline['accuracy'], best_train['accuracy'], selected_step), flush=True)
    print('[Mask] modified={} / allowed={}, outside_scope=0'.format(
        audit['modified_parameter_count'], audit['selected_parameter_count']), flush=True)
    print('[Files] summary={} best={}'.format(directory/'summary.json', best_weights), flush=True)
    return summary


def run_five_seeds(args):
    if args.seed is not None:
        raise ValueError('Omit positional seed when using --five-seeds')
    if args.quick or args.overwrite:
        raise ValueError('No --quick / --overwrite in formal five-seed mode')
    if args.loc_seed != 256:
        raise ValueError('Formal five-seed comparison must fix locator seed=256')
    check_args(args)
    dataset = check_dataset(args.dataset)
    # Resolve ONCE to prevent mixing different rank files across optimizer seeds.
    _, _, loc = load_ranking(dataset, args.loc_seed, args.loc_samples,
                             args.loc_json, args.loc_root, top_k=args.top_k)
    source = str(Path(loc['source_json']).resolve())
    test_script = Path(__file__).resolve().with_name('fairness_famo_test_2.py')
    if not test_script.is_file():
        raise FileNotFoundError('Missing partner test script {}'.format(test_script))
    records = []
    for seed in SEEDS:
        sub_args = argparse.Namespace(**vars(args))
        sub_args.seed = seed
        directory = output_dir(sub_args)
        summary_file, best, test_file = (directory/'summary.json', directory/'best_weights.h5',
                                         directory/'test_results.json')
        print('\n========== {} FAMO seed={} TRAIN+TEST =========='.format(dataset, seed), flush=True)
        if summary_file.is_file() and best.is_file():
            print('[SKIP TRAIN] {}'.format(directory), flush=True)
        elif directory.exists() and any(directory.iterdir()):
            raise RuntimeError('Incomplete existing run; inspect before retry: {}'.format(directory))
        else:
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), dataset, str(seed),
                   '--loc-json', source, '--loc-seed', str(args.loc_seed),
                   '--loc-samples', str(args.loc_samples),
                   '--output-root', args.output_root, '--run-tag', args.run_tag,
                   '--max-steps', str(args.max_steps), '--eval-samples', str(args.eval_samples),
                   '--grad-batch-size', str(args.grad_batch_size),
                   '--eval-batch-size', str(args.eval_batch_size),
                   '--eval-every', str(args.eval_every), '--model-lr', str(args.model_lr),
                   '--task-lr', str(args.task_lr), '--task-decay', str(args.task_decay),
                   '--loss-eps', str(args.loss_eps), '--max-grad-norm', str(args.max_grad_norm),
                   '--fair-temperature', str(args.fair_temperature),
                   '--output-mode', args.output_mode, '--search-radius', str(args.search_radius),
                   '--max-acc-drop', str(args.max_acc_drop),
                   '--gap-target', str(args.gap_target),
                   '--archive-size', str(args.archive_size),
                   '--save-front-limit', str(args.save_front_limit),
                   '--log-every', str(args.log_every)]
            if args.top_k is not None:
                cmd += ['--top-k', str(args.top_k)]
            if args.min_acc is not None:
                cmd += ['--min-acc', str(args.min_acc)]
            if args.model_root:
                cmd += ['--model-root', args.model_root]
            if args.data_root:
                cmd += ['--data-root', args.data_root]
            if args.skip_signature_check:
                cmd += ['--skip-signature-check']
            if args.cpu:
                cmd += ['--cpu']
            subprocess.run(cmd, check=True)
        with summary_file.open('r', encoding='utf-8') as f:
            summary = json.load(f)
        if (summary.get('task') != 'fairness' or summary.get('optimizer') != 'famo' or
                summary.get('dataset') != dataset or int(summary.get('seed', -1)) != seed or
                int(summary.get('loc_seed', -1)) != args.loc_seed or
                int(summary.get('gradient_steps_actual', -1)) != args.max_steps or
                Path(summary.get('localization', {}).get('source_json', '')).resolve() != Path(source)):
            raise ValueError('TRAIN identity/config mismatch: {}'.format(directory))
        hp = summary.get('optimizer_params', {})
        expected = {
            'max_steps': args.max_steps, 'model_lr': args.model_lr,
            'task_lr': args.task_lr, 'task_decay': args.task_decay,
            'train_eval_samples': args.eval_samples,
            'grad_batch_size_per_group_and_utility': args.grad_batch_size,
            'full_train_eval_every': args.eval_every,
            'evaluation_batch_size': args.eval_batch_size,
            'loss_eps': args.loss_eps, 'max_grad_norm': args.max_grad_norm,
            'fair_temperature': args.fair_temperature,
            'search_radius': args.search_radius, 'max_acc_drop': args.max_acc_drop,
            'gap_target': args.gap_target, 'archive_size': args.archive_size,
        }
        for name, value in expected.items():
            if name not in hp or abs(float(hp[name])-float(value)) > 1.e-9:
                raise ValueError('Mismatched TRAIN hyperparameter {} in {}'.format(name, directory))
        if args.min_acc is not None and abs(float(summary.get('min_train_accuracy', -1))-
                                             float(args.min_acc)) > 1.e-9:
            raise ValueError('Existing TRAIN min-accuracy override mismatch: {}'.format(directory))
        if not best.is_file():
            raise FileNotFoundError('Missing best_weights.h5: {}'.format(directory))
        if not test_file.is_file():
            cmd = [sys.executable, '-u', str(test_script), dataset, str(seed),
                   '--optimizer', 'famo', '--run-dir', str(directory),
                   '--batch-size', str(args.eval_batch_size)]
            if args.model_root:
                cmd += ['--model-root', args.model_root]
            if args.data_root:
                cmd += ['--data-root', args.data_root]
            subprocess.run(cmd, check=True)
        with test_file.open('r', encoding='utf-8') as f:
            result = json.load(f)
        if (result.get('optimizer') != 'famo' or result.get('dataset') != dataset or
                int(result.get('seed', -1)) != seed or result.get('evaluation_scope') != 'full_test' or
                result.get('test_used_to_select_checkpoint') is not False or
                Path(result.get('repaired_checkpoint', '')).resolve() != best.resolve()):
            raise ValueError('TEST identity/protocol mismatch: {}'.format(test_file))
        repaired, original = result['repaired'], result['original']
        records.append({
            'seed': seed, 'selected_train_step': int(summary['selected_train_step']),
            'original_gap_percent': 100.*float(original['fairness_gap']),
            'repaired_gap_percent': 100.*float(repaired['fairness_gap']),
            'original_acc_percent': 100.*float(original['accuracy']),
            'repaired_acc_percent': 100.*float(repaired['accuracy']),
            'repaired_f1_percent': 100.*float(repaired['f1']),
            'test_file': str(test_file.resolve()),
        })
    def mean_sd(key):
        vals = [r[key] for r in records]
        return float(statistics.mean(vals)), float(statistics.stdev(vals))
    gap_mu, gap_sd = mean_sd('repaired_gap_percent')
    acc_mu, acc_sd = mean_sd('repaired_acc_percent')
    f1_mu, f1_sd = mean_sd('repaired_f1_percent')
    print('\n========== {} FAIRNESS FAMO FIVE-SEED TEST =========='.format(dataset.upper()), flush=True)
    print('Seed       TEST Gap(%)    TEST Acc(%)    TEST F1(%)', flush=True)
    for r in records:
        print('{:<8d} {:>12.4f} {:>13.4f} {:>12.4f}'.format(
            r['seed'], r['repaired_gap_percent'], r['repaired_acc_percent'], r['repaired_f1_percent']),
              flush=True)
    print('TEST Gap MEAN +/- SD = {:.4f} +/- {:.4f}%'.format(gap_mu, gap_sd), flush=True)
    print('TEST Acc MEAN +/- SD = {:.4f} +/- {:.4f}%'.format(acc_mu, acc_sd), flush=True)
    print('TEST F1  MEAN +/- SD = {:.4f} +/- {:.4f}%'.format(f1_mu, f1_sd), flush=True)
    output = Path(args.output_root).expanduser()/dataset/'CGARepair'/(
        'famo_{}_five_seed_TEST_summary.json'.format(args.run_tag))
    save_json(output, {
        'schema': 'CGARepair_fairness_FAMO_5seed_TEST_v1',
        'dataset': dataset, 'optimizer': 'famo', 'seeds': list(SEEDS),
        'records': records, 'gap_mean_percent': gap_mu, 'gap_sample_sd_percent': gap_sd,
        'acc_mean_percent': acc_mu, 'acc_sample_sd_percent': acc_sd,
        'f1_mean_percent': f1_mu, 'f1_sample_sd_percent': f1_sd,
        'notes': 'All independent TEST results evaluate TRAIN-selected checkpoints only.',
    })
    print('[FIVE-SEED SUMMARY] {}'.format(output), flush=True)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.five_seeds:
        run_five_seeds(args)
    else:
        if args.seed is None:
            parser.error('A seed is required without --five-seeds')
        run(args)


if __name__ == '__main__':
    main()
