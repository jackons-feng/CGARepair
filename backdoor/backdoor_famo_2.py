"""CGARepair backdoor local-weight repair using two-task masked FAMO.

Dependencies already used by the original project:
    read_CGARepair.py, cgarepair_backdoor_utils.py, backdoor_moo_2.py

The GradAct localization and Kernel weight selection are IDENTICAL to the
backdoor MOEA/D implementation. FAMO uses two differentiable task losses:
  (1) cross-entropy to true labels on triggered NON-TARGET examples;
  (2) clean cross-entropy on original training examples.

Discrete TRAIN SR/ASR and CleanAcc select the checkpoint; independent TEST
uses backdoor_famo_test_2.py and NEVER selects a checkpoint. Five-seed mode
starts isolated subprocesses to prevent TF1 graph and GPU-memory accumulation.

FAMO follows the log-loss task weighting in Liu et al., NeurIPS 2023; using
masked Adam, bounded Kernel projection, and TRAIN snapshot selection is our
repair-specific adaptation, not an assertion of byte-identical official FAMO.
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

from read_CGARepair import load_ranking, export_ranking, resolve_model_name, model_tag
from cgarepair_backdoor_utils import (
    DATASET_CONFIG, initialize_tf, load_dataset, model_file, trigger_inputs,
    select_subset, calculate_metrics, save_json,
)
from backdoor_moo_2 import (
    map_players_to_weights, original_vector, apply_vector, reduce_archive,
    pick_train_solution, run_directory,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
SEEDS = (42, 123, 256, 1024, 2026)


def softmax(w):
    x = np.asarray(w, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


class FAMOTaskWeights:
    """Two-task logit update via bias-corrected Adam with L2/Adam decay."""

    def __init__(self, lr=0.025, gamma=0.001, eps=1.e-8):
        if lr <= 0 or gamma < 0 or eps <= 0:
            raise ValueError('Invalid task-weight optimizer configuration')
        self.w = np.zeros(2, dtype=np.float64)
        self.m = np.zeros(2, dtype=np.float64)
        self.v = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.lr, self.gamma, self.eps = float(lr), float(gamma), float(eps)

    @property
    def z(self):
        return softmax(self.w)

    def update(self, before, after):
        before = np.asarray(before, dtype=np.float64)
        after = np.asarray(after, dtype=np.float64)
        if before.shape != (2,) or after.shape != (2,):
            raise ValueError('Two task losses required')
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise FloatingPointError('Nonfinite FAMO task loss')
        if np.any(before < 0) or np.any(after < 0):
            raise ValueError('Cross-entropy task losses must be nonnegative')
        delta = np.log(before + self.eps) - np.log(after + self.eps)
        z = self.z
        # Derivative of softmax task-logit objective, with Adam weight decay.
        gradient = z * (delta - np.dot(z, delta)) + self.gamma * self.w
        self.step_count += 1
        self.m = .9 * self.m + .1 * gradient
        self.v = .999 * self.v + .001 * (gradient * gradient)
        m_hat = self.m / (1 - .9 ** self.step_count)
        v_hat = self.v / (1 - .999 ** self.step_count)
        self.w -= self.lr * m_hat / (np.sqrt(v_hat) + 1.e-8)
        self.w = np.clip(self.w, -30., 30.)
        return delta


def labels_as_indices(labels):
    arr = np.asarray(labels)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        return np.argmax(arr, axis=-1).reshape(-1).astype(np.int32)
    return arr.reshape(-1).astype(np.int32)


def create_masks_and_bounds(base_weights, specs, lower, upper):
    if not specs or len(lower) != len(upper) or len(lower) != specs[-1]['stop']:
        raise ValueError('Weight specs and bounds disagree')
    blocks = {}
    for s in specs:
        idx, j = int(s['weight_index']), int(s['neuron_index'])
        if idx not in blocks:
            shape = base_weights[idx].shape
            blocks[idx] = dict(
                layer_index=int(s['layer_index']),
                mask=np.zeros(shape, dtype=np.float32),
                lower=np.zeros(shape, dtype=np.float32),
                upper=np.zeros(shape, dtype=np.float32),
            )
        entry = blocks[idx]
        if np.any(entry['mask'][..., j] != 0):
            raise ValueError('Duplicate selected weight block: {}'.format(s['player']))
        entry['mask'][..., j] = 1
        entry['lower'][..., j] = lower[s['start']:s['stop']].reshape(s['shape'])
        entry['upper'][..., j] = upper[s['start']:s['stop']].reshape(s['shape'])
        if np.any(entry['lower'][..., j] >= entry['upper'][..., j]):
            raise ValueError('Nonpositive search interval')
    return blocks


def check_weight_scope(before, after, specs, lower, upper):
    """Bit-exact check outside chosen Conv/Dense Kernel output slices."""
    if len(before) != len(after):
        raise RuntimeError('Model tensor count changed')
    blocks = create_masks_and_bounds(before, specs, lower, upper)
    changed = 0
    for i, (a, b) in enumerate(zip(before, after)):
        if a.shape != b.shape or not np.all(np.isfinite(b)):
            raise RuntimeError('Weight shape or finiteness invalid: {}'.format(i))
        if i not in blocks:
            if not np.array_equal(a, b):
                raise RuntimeError('Unselected tensor modified: {}'.format(i))
        else:
            mask = blocks[i]['mask'].astype(bool)
            if not np.array_equal(a[~mask], b[~mask]):
                raise RuntimeError('Unselected Kernel entries changed: {}'.format(i))
            if (np.any(b[mask] < blocks[i]['lower'][mask] - 1.e-5) or
                    np.any(b[mask] > blocks[i]['upper'][mask] + 1.e-5)):
                raise RuntimeError('Selected Kernel entry outside bounds: {}'.format(i))
            changed += int(np.count_nonzero(a[mask] != b[mask]))
    return {
        'modified_parameter_count': int(changed),
        'selected_parameter_count': int(sum(s['num_parameters'] for s in specs)),
        'outside_scope_modified_count': 0,
        'other_tensor_untouched': True,
    }


def detect_output_mode(session, model, x, requested):
    if requested != 'auto':
        return requested
    output = np.asarray(session.run(model.output, feed_dict={model.input: x[:min(len(x), 8)]}))
    if output.ndim != 2 or output.shape[1] < 2:
        raise ValueError('Expected network output [N, classes]; got {}'.format(output.shape))
    if (np.all(output >= -1.e-6) and np.all(output <= 1 + 1.e-6) and
            np.allclose(np.sum(output, axis=1), 1., atol=.02, rtol=.02)):
        return 'probabilities'
    return 'logits'


class MaskedBackdoorFAMOGraph:
    """Two CE tasks share graph and model; only GradAct-selected Kernels update."""

    def __init__(self, tf, session, model, base_weights, specs, lower, upper,
                 model_lr, grad_clip, loss_eps, output_mode):
        v1 = tf.compat.v1
        self.session = session
        self.x = model.input
        self.trigger_labels = v1.placeholder(tf.int32, [None], name='famo_trigger_true_labels')
        self.clean_labels = v1.placeholder(tf.int32, [None], name='famo_clean_true_labels')
        self.n_trigger = v1.placeholder(tf.int32, [], name='famo_trigger_count')
        self.z = v1.placeholder(tf.float32, [2], name='famo_task_z')
        self.blocks = create_masks_and_bounds(base_weights, specs, lower, upper)
        output = tf.cast(model.output, tf.float32)
        if output.shape.ndims != 2:
            raise ValueError('Multiclass model output [N, classes] required')
        if output_mode == 'probabilities':
            logits = tf.math.log(tf.clip_by_value(output, 1.e-7, 1.0))
        elif output_mode == 'logits':
            logits = output
        else:
            raise ValueError('Invalid output mode')
        trigger_logits = logits[:self.n_trigger]
        clean_logits = logits[self.n_trigger:]
        trigger_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.trigger_labels, logits=trigger_logits))
        clean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.clean_labels, logits=clean_logits))
        self.losses = tf.stack([trigger_ce, clean_ce], name='famo_trigger_clean_losses')
        positive = tf.maximum(self.losses, 0.) + float(loss_eps)
        normalization = tf.stop_gradient(tf.reduce_sum(self.z / positive))
        combined = tf.reduce_sum(self.z * tf.math.log(positive)) / normalization

        vars_, mask_tensors, bounds = [], [], []
        for idx, block in sorted(self.blocks.items()):
            layer = model.layers[block['layer_index']]
            var = layer.weights[0]
            if tuple(var.shape.as_list()) != tuple(base_weights[idx].shape):
                raise RuntimeError('Layer/Kernel mapping error at index {}'.format(idx))
            vars_.append(var)
            dtype = var.dtype.base_dtype
            mask_tensors.append(tf.constant(block['mask'], dtype=dtype))
            bounds.append((tf.constant(block['lower'], dtype=dtype),
                           tf.constant(block['upper'], dtype=dtype)))
        gradients = tf.gradients(combined, vars_)
        if any(g is None for g in gradients):
            raise RuntimeError('Selected Kernel disconnected from model output')
        masked = []
        for g, mask in zip(gradients, mask_tensors):
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            masked.append(g * mask)
        self.gradient_norm = v1.global_norm(masked)
        if grad_clip > 0:
            masked, _ = tf.clip_by_global_norm(masked, float(grad_clip),
                                               use_norm=self.gradient_norm)
        existing = {v.name for v in v1.global_variables()}
        opt = v1.train.AdamOptimizer(learning_rate=float(model_lr))
        update = opt.apply_gradients(list(zip(masked, vars_)))
        projections = []
        with tf.control_dependencies([update]):
            for var, mask, (lo, hi) in zip(vars_, mask_tensors, bounds):
                clamped = tf.clip_by_value(var, lo, hi)
                projections.append(v1.assign(var, tf.where(tf.greater(mask, 0), clamped, var)))
        self.train_op = tf.group(*projections, name='famo_masked_backdoor_adam')
        new_variables = [v for v in v1.global_variables() if v.name not in existing]
        if new_variables:
            session.run(v1.variables_initializer(new_variables))
        self.updated_kernel_tensors = len(vars_)

    def feed(self, x_trigger, y_trigger, x_clean, y_clean, z):
        return {
            self.x: np.concatenate([x_trigger, x_clean], axis=0),
            self.trigger_labels: labels_as_indices(y_trigger),
            self.clean_labels: labels_as_indices(y_clean),
            self.n_trigger: int(len(x_trigger)),
            self.z: np.asarray(z, dtype=np.float32),
        }

    def step(self, feed):
        before, _, norm = self.session.run(
            [self.losses, self.train_op, self.gradient_norm], feed_dict=feed)
        after = self.session.run(self.losses, feed_dict=feed)
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise FloatingPointError('Nonfinite FAMO loss; consider reducing --model-lr')
        if not np.isfinite(norm):
            raise FloatingPointError('Nonfinite masked FAMO gradient')
        return np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64), float(norm)


def append_train_candidate(model, session, specs, xs, fs, steps,
                           clean, labels, triggered, target, sr_definition, batch, step):
    metrics = calculate_metrics(model, clean, labels, triggered, target, session,
                                sr_definition=sr_definition, batch_size=batch)
    if not np.isfinite(metrics['sr']) or not np.isfinite(metrics['clean_accuracy']):
        raise FloatingPointError('Nonfinite full TRAIN SR/ACC')
    xs.append(original_vector(model.get_weights(), specs))
    fs.append([float(metrics['sr']), -float(metrics['clean_accuracy'])])
    steps.append(int(step))
    return metrics


def write_csv(path, records):
    if not records:
        return
    with open(str(path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parser():
    p = argparse.ArgumentParser(description='CGARepair backdoor GradAct + masked FAMO repair')
    p.add_argument('model_name', help='fashion, gtsrb, gtsrb_resnet, mnist')
    p.add_argument('seed', nargs='?', type=int, help='Optimizer seed; omit with --five-seeds')
    p.add_argument('legacy_resnet', nargs='?', default=None)
    p.add_argument('--resnet', action='store_true', help='Use GTSRB ResNet-18')
    p.add_argument('--optimizer', '--algorithm', choices=('famo',), default='famo')
    p.add_argument('--five-seeds', action='store_true',
                   help='Run seeds 42/123/256/1024/2026 and full independent TEST')
    p.add_argument('--loc-json', help='Existing GradAct ranking JSON (optional)')
    p.add_argument('--loc-root', help='Root for read_CGARepair localization lookup')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-tag', default='paper_gamma001_steps5000')
    p.add_argument('--quick', action='store_true', help='Five-step TRAIN-only smoke test')
    p.add_argument('--overwrite', action='store_true',
                   help='Back up previous result directory and run a fresh single seed')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--eval-samples', type=int, default=1024,
                   help='TRAIN pool size; preserves original backdoor MOEA/D default')
    p.add_argument('--grad-batch-size', type=int, default=64,
                   help='Examples per task per gradient update')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=20)
    p.add_argument('--model-lr', type=float, default=0.001)
    p.add_argument('--task-lr', type=float, default=0.025)
    p.add_argument('--task-decay', type=float, default=0.001)
    p.add_argument('--loss-eps', type=float, default=1.e-8)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--search-radius', type=float, default=10.0,
                   help='Default matches original backdoor MOEA/D radius; bounds [-5,5]')
    p.add_argument('--sr-definition', choices=('legacy', 'standard'), default='legacy',
                   help='TRAIN SR denominator; final TEST always additionally reports standard ASR')
    p.add_argument('--threshold-acc', type=float, default=None)
    p.add_argument('--threshold-sr', type=float, default=None)
    p.add_argument('--archive-size', type=int, default=60)
    p.add_argument('--save-front-limit', type=int, default=10)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--output-mode', choices=('auto', 'probabilities', 'logits'), default='auto')
    p.add_argument('--evasion', action='store_true',
                   help='In five-seed mode also run original evasion TEST diagnostics')
    p.add_argument('--evasion-samples', type=int, default=200)
    return p


def _check_args(args):
    if (min(args.max_steps, args.eval_samples, args.grad_batch_size, args.batch_size,
            args.eval_every, args.log_every) <= 0 or args.task_lr <= 0 or
            args.model_lr <= 0 or args.task_decay < 0 or args.loss_eps <= 0 or
            args.search_radius <= 0 or args.max_grad_norm < 0 or args.archive_size < 2 or
            args.save_front_limit < 0):
        raise ValueError('Invalid optimization budget, learning rates, or mask bounds')


def run(args):
    t0 = time.perf_counter()
    is_resnet = args.resnet or str(args.legacy_resnet).lower() in ('true', '1', 'yes', 'y')
    name, is_resnet = resolve_model_name(args.model_name, is_resnet)
    tag = model_tag(name, is_resnet)
    if args.quick:
        # Never let a smoke test occupy the five-seed production run directory.
        if args.run_tag == 'paper_gamma001_steps5000':
            args.run_tag = 'smoke_5steps'
        args.max_steps = 5
        args.eval_samples = min(args.eval_samples, 128)
        args.grad_batch_size = min(args.grad_batch_size, 16)
        args.eval_every = 1
    _check_args(args)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    outdir = run_directory(args.output_root, tag, args.seed, 'famo', args.run_tag, args.quick)
    if outdir.is_dir() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError('Existing or incomplete result directory: {}. Use a new --run-tag.'.format(outdir))
    if outdir.is_dir() and any(outdir.iterdir()) and args.overwrite:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup = outdir.with_name(outdir.name + '_backup_' + stamp)
        if backup.exists():
            raise FileExistsError('Backup directory already exists: {}'.format(backup))
        outdir.rename(backup)
        print('[BACKUP] {}'.format(backup), flush=True)
    outdir.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    players, scores, loc = load_ranking(name, is_resnet, args.loc_seed,
                                         args.loc_samples, args.loc_json, args.loc_root)
    loc_export = export_ranking(players, scores, loc, export_dir=outdir/'localization_input')
    loc_sec = time.perf_counter() - t
    print('[Locator] {} Top-{} from {} (loc_seed={} opt_seed={})'.format(
        tag, loc['top_k'], loc['source_json'], args.loc_seed, args.seed), flush=True)
    print('[Locator] selected={}'.format(players), flush=True)

    t = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    try:
        K.set_learning_phase(0)
    except AttributeError:
        tf.compat.v1.keras.backend.set_learning_phase(0)
    from tensorflow.keras.models import load_model
    pretrained_path = model_file(name, is_resnet, args.model_root)
    model = load_model(pretrained_path, compile=False)
    source_signature = loc.get('source_model_signature')
    if (isinstance(source_signature, (list, tuple)) and source_signature and
            int(source_signature[0]) != os.stat(pretrained_path).st_size):
        raise ValueError('Localization and pretrained checkpoint file-size mismatch')
    specs = map_players_to_weights(model, players)
    base_weights = [w.copy() for w in model.get_weights()]
    base_vec = original_vector(base_weights, specs)
    lower = np.maximum(-5., base_vec - args.search_radius)
    upper = np.minimum(5., base_vec + args.search_radius)
    if np.any(lower >= upper):
        raise ValueError('Degenerate repair bounds: check --search-radius')
    setup_sec = time.perf_counter() - t
    print('[Repair] {} selected_neurons={} decision_variables={} checkpoint={}'.format(
        tag, len(specs), len(base_vec), pretrained_path), flush=True)

    t = time.perf_counter()
    full_x, full_y = load_dataset(name, 'train', args.data_root, session)
    full_triggered = trigger_inputs(full_x, name, args.data_root)
    clean, raw_labels, triggered = select_subset(
        full_x, full_y, full_triggered, args.eval_samples, args.seed)
    del full_x, full_y, full_triggered
    clean = np.asarray(clean)
    triggered = np.asarray(triggered)
    labels = labels_as_indices(raw_labels)
    target = int(DATASET_CONFIG[name]['target_label'])
    if len(clean) != len(labels) or len(triggered) != len(labels):
        raise RuntimeError('Clean, labels, triggered sample counts disagree')
    non_target = labels != target
    n_non_target = int(np.sum(non_target))
    if n_non_target == 0:
        raise ValueError('No non-target examples in TRAIN pool; cannot optimize ASR')
    poison_x, poison_y = triggered[non_target], labels[non_target]
    train_data_sec = time.perf_counter() - t
    t = time.perf_counter()
    # Preserve the original evaluator's label representation. FAMO CE uses
    # integer indices, but the original SR/Acc metrics receive raw labels.
    baseline = calculate_metrics(model, clean, raw_labels, triggered, target, session,
                                 sr_definition=args.sr_definition, batch_size=args.batch_size)
    baseline_sec = time.perf_counter()-t
    cfg = DATASET_CONFIG[name]
    threshold_acc = (float(cfg['threshold_acc']) if args.threshold_acc is None
                     else float(args.threshold_acc))
    threshold_sr = (float(cfg['threshold_sr']) if args.threshold_sr is None
                    else float(args.threshold_sr))
    print('[TRAIN baseline] SR({})={:.6f} standard_ASR={} CleanAcc={:.6f} '
          'n={} non_target={}'.format(args.sr_definition, baseline['sr'],
                                    baseline.get('standard_asr'), baseline['clean_accuracy'],
                                    len(labels), n_non_target), flush=True)
    print('[TRAIN selection] SR <= {:.6f}, CleanAcc >= {:.6f} for full feasibility; '
          'fallback uses minimum SR under accuracy guard'.format(threshold_sr, threshold_acc), flush=True)

    t = time.perf_counter()
    mode = detect_output_mode(session, model, clean, args.output_mode)
    graph = MaskedBackdoorFAMOGraph(
        tf, session, model, base_weights, specs, lower, upper,
        args.model_lr, args.max_grad_norm, args.loss_eps, mode)
    graph_sec = time.perf_counter()-t
    task_weights = FAMOTaskWeights(args.task_lr, args.task_decay, args.loss_eps)
    rng = np.random.RandomState(int(args.seed))
    xs, fs, eval_steps = [], [], []
    trace, snapshots = [], []
    initial = append_train_candidate(model, session, specs, xs, fs, eval_steps,
                                     clean, raw_labels, triggered, target, args.sr_definition,
                                     args.batch_size, 0)
    snapshots.append({
        'step': 0, 'train_sr': float(initial['sr']),
        'train_standard_asr': initial.get('standard_asr'),
        'train_clean_accuracy': float(initial['clean_accuracy']),
        'feasible': bool(initial['sr'] <= threshold_sr and
                         initial['clean_accuracy'] >= threshold_acc),
        'z_trigger': .5, 'z_clean': .5, 'elapsed_opt_sec': 0.,
    })

    print('[FAMO] steps={} triggered_NON_TARGET_CE + clean_CE, '
          'batch={} per task, eval_every={}, task_decay={}, output_mode={}'.format(
              args.max_steps, args.grad_batch_size, args.eval_every,
              args.task_decay, mode), flush=True)
    opt_start = time.perf_counter()
    for step in range(1, int(args.max_steps)+1):
        i_trigger = rng.randint(0, len(poison_y), size=args.grad_batch_size)
        i_clean = rng.randint(0, len(labels), size=args.grad_batch_size)
        feed = graph.feed(poison_x[i_trigger], poison_y[i_trigger],
                          clean[i_clean], labels[i_clean], task_weights.z)
        before, after, norm = graph.step(feed)
        delta = task_weights.update(before, after)
        z = task_weights.z
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print('[FAMO step {:04d}/{}] triggerCE={:.5f}->{:.5f} '
                  'cleanCE={:.5f}->{:.5f} z=({:.3f},{:.3f}) gradnorm={:.4f}'.format(
                      step, args.max_steps, before[0], after[0], before[1], after[1],
                      z[0], z[1], norm), flush=True)
        trace.append({
            'step': step, 'trigger_ce_before': float(before[0]),
            'trigger_ce_after': float(after[0]), 'clean_ce_before': float(before[1]),
            'clean_ce_after': float(after[1]),
            'trigger_log_improvement': float(delta[0]),
            'clean_log_improvement': float(delta[1]),
            'z_trigger': float(z[0]), 'z_clean': float(z[1]),
            'gradient_global_norm': norm,
            'elapsed_opt_sec': float(time.perf_counter()-opt_start),
        })
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = append_train_candidate(
                model, session, specs, xs, fs, eval_steps, clean, raw_labels,
                triggered, target, args.sr_definition, args.batch_size, step)
            feasible = bool(metrics['sr'] <= threshold_sr and
                            metrics['clean_accuracy'] >= threshold_acc)
            snapshots.append({
                'step': int(step), 'train_sr': float(metrics['sr']),
                'train_standard_asr': metrics.get('standard_asr'),
                'train_clean_accuracy': float(metrics['clean_accuracy']),
                'feasible': feasible, 'z_trigger': float(z[0]),
                'z_clean': float(z[1]),
                'elapsed_opt_sec': float(time.perf_counter()-opt_start),
            })
            print('[TRAIN step={}] SR({})={:.6f} standardASR={} '
                  'CleanAcc={:.6f} feasible={}'.format(
                      step, args.sr_definition, metrics['sr'],
                      metrics.get('standard_asr'), metrics['clean_accuracy'], feasible),
                  flush=True)
    optimization_sec = time.perf_counter()-opt_start

    all_x = np.asarray(xs, dtype=np.float64)
    all_f = np.asarray(fs, dtype=np.float64)
    selected_global, policy = pick_train_solution(all_f, threshold_acc, threshold_sr)
    selected_vector = all_x[selected_global].copy()
    selected_f = all_f[selected_global].copy()
    candidate_x, candidate_f = reduce_archive(all_x, all_f, args.archive_size)
    match = np.where(np.all(candidate_x == selected_vector, axis=1))[0]
    if len(match):
        selected_index = int(match[0])
    else:
        # Keep TRAIN-selected fallback even if its point is not nondominated.
        selected_index = int(len(candidate_x))
        candidate_x = np.concatenate([candidate_x, selected_vector[None]], axis=0)
        candidate_f = np.concatenate([candidate_f, selected_f[None]], axis=0)

    t = time.perf_counter()
    write_csv(outdir / 'famo_step_history.csv', trace)
    write_csv(outdir / 'famo_train_snapshots.csv', snapshots)
    selected_weights = apply_vector([v.copy() for v in base_weights], specs, selected_vector)
    mask_audit = check_weight_scope(base_weights, selected_weights, specs, lower, upper)
    model.set_weights(selected_weights)
    best_file = outdir/'best_weights.h5'
    model.save_weights(str(best_file))
    check_weight_scope(base_weights, model.get_weights(), specs, lower, upper)

    front_dir = outdir/'pareto_front'
    front_dir.mkdir(exist_ok=True)
    chosen_ids = list(range(min(args.save_front_limit, len(candidate_x))))
    if selected_index not in chosen_ids and chosen_ids:
        chosen_ids[-1] = selected_index
    front_records = []
    for idx in sorted(set(chosen_ids)):
        w = apply_vector([v.copy() for v in base_weights], specs, candidate_x[idx])
        check_weight_scope(base_weights, w, specs, lower, upper)
        model.set_weights(w)
        path = front_dir/('weights_{}.h5'.format(idx))
        model.save_weights(str(path))
        front_records.append({
            'index': idx, 'train_sr': float(candidate_f[idx, 0]),
            'train_clean_accuracy': float(-candidate_f[idx, 1]),
            'path': str(path),
        })
    model.set_weights(base_weights)
    write_csv(outdir/'pareto_front.txt', front_records)
    with (outdir/'pareto_front.pkl').open('wb') as f:
        pickle.dump({
            'X': candidate_x, 'F': candidate_f,
            'selected_index': selected_index,
            'selected_all_snapshot_index': int(selected_global),
            'all_evaluated_X': all_x, 'all_evaluated_F': all_f,
            'all_evaluated_steps': eval_steps,
            'history': snapshots, 'total_evaluations': len(snapshots),
            'source_locator': loc['source_json'],
            'note': 'TRAIN FAMO snapshots, NOT evolutionary FE or a MOEA/D population',
        }, f, protocol=4)
    save_sec = time.perf_counter()-t
    wall_sec = time.perf_counter()-t0
    timing = {
        'localization_previous_run_sec': loc.get('source_runtime_sec'),
        'localization_previous_run_not_included_in_repair_wall': True,
        'localization_read_export_sec': float(loc_sec),
        'model_load_and_mapping_sec': float(setup_sec),
        'train_data_and_trigger_sec': float(train_data_sec),
        'baseline_eval_sec': float(baseline_sec),
        'famo_graph_build_sec': float(graph_sec),
        'optimization_wall_sec': float(optimization_sec),
        'checkpoint_save_sec': float(save_sec),
        'total_repair_run_wall_sec': float(wall_sec),
        'notes': 'Gradient steps are NOT MOEA/D fitness evaluations; snapshot eval is included in optimization wall.',
    }
    selected_step = int(eval_steps[selected_global])
    summary = {
        'schema': 'CGARepair_backdoor_FAMO_masked_weight_repair_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'backdoor', 'model': name, 'model_tag': tag,
        'optimizer': 'famo', 'method': 'CGARepair_FAMO',
        'optimization_seed': int(args.seed), 'localization_seed': int(args.loc_seed),
        'localization': loc, 'exported_locator_dir': str(loc_export),
        'model_path': str(pretrained_path),
        'selected_players': players, 'top_k': int(loc['top_k']),
        'num_players_total': int(loc['num_players']),
        'repair_specs': [{k: list(v) if isinstance(v, tuple) else v
                          for k, v in s.items()} for s in specs],
        'decision_variables': int(len(base_vec)),
        'parameter_budget_percent': float(100. * len(base_vec) / max(1, model.count_params())),
        'parameter_scope': 'Selected Conv/Dense Kernel output channels; BN and biases frozen',
        'mask_audit': mask_audit,
        'sr_definition': args.sr_definition,
        'sr_definition_note': ('legacy: non-target successes / all triggered images; '
                               'standard: non-target successes / non-target triggered images'),
        'train_eval_samples': int(len(labels)),
        'train_non_target_count': n_non_target,
        'original_train_metrics': baseline,
        'optimization_proxy_losses': ['triggered_non_target_true_label_CE', 'clean_CE'],
        'objective_definitions': ['TRAIN_SR=minimize', '-TRAIN_clean_accuracy=minimize'],
        'threshold_acc': float(threshold_acc),
        'threshold_sr': float(threshold_sr),
        'gradient_steps_requested': int(args.max_steps),
        'gradient_steps_actual': int(len(trace)),
        'train_snapshot_evaluations_actual': int(len(snapshots)),
        'front_size': int(len(candidate_x)),
        'front_feasible_count': int(np.sum((candidate_f[:, 0] <= threshold_sr) &
                                           (-candidate_f[:, 1] >= threshold_acc))),
        'selected_index': int(selected_index),
        'selected_global_snapshot_index': int(selected_global),
        'selected_train_step': selected_step,
        'selection_policy': policy,
        'selected_train_sr': float(selected_f[0]),
        'selected_train_acc': float(-selected_f[1]),
        'best_weights_file': str(best_file.resolve()),
        'optimizer_parameters': {
            'name': 'FAMO', 'max_steps': int(args.max_steps),
            'grad_batch_size_per_task': int(args.grad_batch_size),
            'evaluation_batch_size': int(args.batch_size),
            'full_train_eval_every': int(args.eval_every),
            'model_lr': float(args.model_lr),
            'task_lr': float(args.task_lr),
            'task_decay': float(args.task_decay),
            'loss_eps': float(args.loss_eps),
            'max_grad_norm': float(args.max_grad_norm),
            'search_radius': float(args.search_radius),
            'output_mode': mode,
            'sr_definition': args.sr_definition,
            'archive_size': int(args.archive_size),
            'train_eval_samples': int(args.eval_samples),
            'gradient_algorithm': 'FAMO log-loss task weights and masked Kernel Adam',
        },
        'task_logit_final': task_weights.w.tolist(),
        'task_weight_final': task_weights.z.tolist(),
        'optimizer_history': snapshots,
        'timing_sec': timing,
        'total_time_sec': float(wall_sec),
        'comparability_note': ('Same GradAct locator, mask, dataset and TRAIN SR definition as original backdoor repair; '
                               'FAMO uses differentiable triggered/non-target CE, not discrete SR.'),
    }
    incomplete = outdir/'summary.json.incomplete'
    save_json(incomplete, summary)
    os.replace(str(incomplete), str(outdir/'summary.json'))
    print('[Done] {} FAMO steps={} selected_STEP={} '
          'train_SR={:.6f} train_ACC={:.6f} baseline_SR={:.6f} baseline_ACC={:.6f}'.format(
              tag, args.max_steps, selected_step, selected_f[0], -selected_f[1],
              baseline['sr'], baseline['clean_accuracy']), flush=True)
    print('[Mask] modified={} / allowed={} outside_scope={}'.format(
        mask_audit['modified_parameter_count'],
        mask_audit['selected_parameter_count'],
        mask_audit['outside_scope_modified_count']), flush=True)
    print('[Timing] optimization={:.2f}s total={:.2f}s'.format(optimization_sec, wall_sec), flush=True)
    print('[Files] summary={} best={} archive={}'.format(
        outdir/'summary.json', best_file, outdir/'pareto_front.pkl'), flush=True)
    return summary


def run_five_seeds(args):
    """Idempotent five-seed TRAIN/independent TEST, each run isolated in a subprocess."""
    if args.quick or args.overwrite or args.cpu:
        raise ValueError('--five-seeds forbids --quick/--overwrite/--cpu')
    if args.seed is not None or args.legacy_resnet is not None:
        raise ValueError('Omit positional seed and legacy_resnet with --five-seeds')
    if args.loc_seed != 256:
        raise ValueError('Five-seed protocol requires fixed --loc-seed 256')
    if args.evasion and args.evasion_samples <= 0:
        raise ValueError('--evasion-samples must be positive')
    _check_args(args)
    name, is_resnet = resolve_model_name(args.model_name, args.resnet)
    tag = model_tag(name, is_resnet)
    test_script = Path(__file__).resolve().with_name('backdoor_famo_test_2.py')
    if not test_script.is_file():
        raise FileNotFoundError('Missing independent TEST script: {}'.format(test_script))
    # Resolve localization ONCE so that the five isolated processes all load
    # exactly the same GradAct ranking instead of searching per seed.
    _, _, loc = load_ranking(name, is_resnet, args.loc_seed, args.loc_samples,
                              args.loc_json, args.loc_root)
    loc_source = str(Path(loc['source_json']).expanduser().resolve())
    if not Path(loc_source).is_file():
        raise FileNotFoundError('GradAct localization JSON not found: {}'.format(loc_source))
    own_file = str(Path(__file__).resolve())
    records = []
    for seed in SEEDS:
        directory = run_directory(args.output_root, tag, seed, 'famo', args.run_tag, False)
        summary_file = directory/'summary.json'
        best = directory/'best_weights.h5'
        test_file = directory/'test_results.json'
        print('\n========== {} FAMO seed={} TRAIN+TEST =========='.format(tag, seed), flush=True)
        if summary_file.is_file() and best.is_file():
            print('[SKIP TRAIN] {}'.format(directory), flush=True)
        elif directory.exists() and any(directory.iterdir()):
            raise RuntimeError('Existing incomplete directory; inspect before retry: {}'.format(directory))
        else:
            cmd = [sys.executable, '-u', own_file, args.model_name, str(seed),
                   '--loc-json', loc_source, '--loc-seed', str(args.loc_seed),
                   '--loc-samples', str(args.loc_samples), '--output-root', args.output_root,
                   '--run-tag', args.run_tag, '--max-steps', str(args.max_steps),
                   '--eval-samples', str(args.eval_samples),
                   '--grad-batch-size', str(args.grad_batch_size),
                   '--batch-size', str(args.batch_size),
                   '--eval-every', str(args.eval_every),
                   '--model-lr', str(args.model_lr), '--task-lr', str(args.task_lr),
                   '--task-decay', str(args.task_decay), '--loss-eps', str(args.loss_eps),
                   '--max-grad-norm', str(args.max_grad_norm),
                   '--search-radius', str(args.search_radius),
                   '--sr-definition', args.sr_definition,
                   '--archive-size', str(args.archive_size),
                   '--save-front-limit', str(args.save_front_limit),
                   '--log-every', str(args.log_every), '--output-mode', args.output_mode]
            if is_resnet:
                cmd += ['--resnet']
            if args.model_root:
                cmd += ['--model-root', args.model_root]
            if args.data_root:
                cmd += ['--data-root', args.data_root]
            if args.threshold_acc is not None:
                cmd += ['--threshold-acc', str(args.threshold_acc)]
            if args.threshold_sr is not None:
                cmd += ['--threshold-sr', str(args.threshold_sr)]
            subprocess.run(cmd, check=True)
        with summary_file.open('r', encoding='utf-8') as f:
            summary = json.load(f)
        if (summary.get('task') != 'backdoor' or summary.get('optimizer') != 'famo' or
                summary.get('model_tag') != tag or
                int(summary.get('optimization_seed', -1)) != seed or
                int(summary.get('localization_seed', -1)) != 256 or
                int(summary.get('gradient_steps_actual', -1)) != args.max_steps or
                summary.get('sr_definition') != args.sr_definition):
            raise ValueError('TRAIN result identity/protocol mismatch: {}'.format(directory))
        if Path(summary['localization']['source_json']).resolve() != Path(loc_source):
            raise ValueError('TRAIN locator mismatch: {}'.format(directory))
        if (abs(float(summary.get('threshold_acc', -1)) - float(
                DATASET_CONFIG[name]['threshold_acc'] if args.threshold_acc is None else args.threshold_acc)) > 1.e-9 or
                abs(float(summary.get('threshold_sr', -1)) - float(
                    DATASET_CONFIG[name]['threshold_sr'] if args.threshold_sr is None else args.threshold_sr)) > 1.e-9):
            raise ValueError('TRAIN SR/ACC threshold mismatch: {}'.format(directory))
        hp = summary.get('optimizer_parameters', {})
        expected = {
            'max_steps': args.max_steps, 'task_decay': args.task_decay,
            'model_lr': args.model_lr, 'task_lr': args.task_lr,
            'grad_batch_size_per_task': args.grad_batch_size,
            'full_train_eval_every': args.eval_every,
            'search_radius': args.search_radius,
            'train_eval_samples': args.eval_samples,
            'evaluation_batch_size': args.batch_size,
            'loss_eps': args.loss_eps,
            'max_grad_norm': args.max_grad_norm,
            'archive_size': args.archive_size,
        }
        for k, v in expected.items():
            actual = hp.get(k)
            if actual is None or abs(float(actual)-float(v)) > 1.e-9:
                raise ValueError('Existing TRAIN config mismatch: {} {} vs {}'.format(k, actual, v))
        if not best.is_file():
            raise FileNotFoundError('TRAIN-selected weight file missing: {}'.format(best))
        if test_file.is_file():
            print('[SKIP TEST] {}'.format(test_file), flush=True)
        else:
            cmd = [sys.executable, '-u', str(test_script), args.model_name, str(seed),
                   '--optimizer', 'famo', '--run-dir', str(directory),
                   '--test-samples', '0', '--batch-size', str(args.batch_size)]
            if is_resnet:
                cmd += ['--resnet']
            if args.model_root:
                cmd += ['--model-root', args.model_root]
            if args.data_root:
                cmd += ['--data-root', args.data_root]
            if args.evasion:
                cmd += ['--evasion', '--evasion-samples', str(args.evasion_samples)]
            subprocess.run(cmd, check=True)
        with test_file.open('r', encoding='utf-8') as f:
            test = json.load(f)
        if (test.get('optimizer') != 'famo' or test.get('model') != tag or
                int(test.get('seed', -1)) != seed or
                int(test.get('test_sample_count', 0)) <= 0 or
                int(test.get('test_samples_requested', -1)) != 0 or
                test.get('sr_definition') != args.sr_definition or
                test.get('checkpoint_selection') != 'TRAIN only'):
            raise ValueError('TEST result identity/protocol mismatch: {}'.format(test_file))
        if args.evasion and not test.get('evasion_results'):
            raise ValueError('Expected evasion diagnostics but TEST file lacks them: {}'.format(test_file))
        if Path(test.get('selected_checkpoint', '')).resolve() != best.resolve():
            raise ValueError('TEST checkpoint mismatch: {}'.format(test_file))
        after = test['repaired_test_metrics']
        before = test['original_test_metrics']
        records.append({
            'seed': seed,
            'selected_train_step': int(summary['selected_train_step']),
            'modified_parameter_count': int(summary['mask_audit']['modified_parameter_count']),
            'before_standard_asr_percent': 100. * float(before['standard_asr']),
            'after_standard_asr_percent': 100. * float(after['standard_asr']),
            'before_sr_percent': 100. * float(before['sr']),
            'after_sr_percent': 100. * float(after['sr']),
            'before_clean_acc_percent': 100. * float(before['clean_accuracy']),
            'after_clean_acc_percent': 100. * float(after['clean_accuracy']),
            'test_file': str(test_file.resolve()),
        })
    def mean_sd(key):
        vals = [r[key] for r in records]
        return float(statistics.mean(vals)), float(statistics.stdev(vals))
    asr_mean, asr_sd = mean_sd('after_standard_asr_percent')
    acc_mean, acc_sd = mean_sd('after_clean_acc_percent')
    sr_mean, sr_sd = mean_sd('after_sr_percent')
    print('\n========== {} FAMO FIVE-SEED TEST =========='.format(tag.upper()), flush=True)
    print('Seed      Standard_ASR(%)   CleanAcc(%)   TRAIN-defined_SR(%)', flush=True)
    for r in records:
        print('{:<8d} {:>14.5f} {:>12.5f} {:>18.5f}'.format(
            r['seed'], r['after_standard_asr_percent'],
            r['after_clean_acc_percent'], r['after_sr_percent']), flush=True)
    print('TEST standard ASR MEAN +/- SD = {:.5f} +/- {:.5f}%'.format(asr_mean, asr_sd), flush=True)
    print('TEST CleanAcc     MEAN +/- SD = {:.5f} +/- {:.5f}%'.format(acc_mean, acc_sd), flush=True)
    print('TEST SR({}) MEAN +/- SD = {:.5f} +/- {:.5f}%'.format(
        args.sr_definition, sr_mean, sr_sd), flush=True)
    out = Path(args.output_root)/tag/'CGARepair'/(
        'famo_{}_five_seed_TEST_summary.json'.format(args.run_tag))
    save_json(out, {
        'schema': 'CGARepair_backdoor_FAMO_5seed_TEST_v1',
        'model_tag': tag, 'optimizer': 'famo', 'seeds': list(SEEDS),
        'train_sr_definition': args.sr_definition,
        'records': records,
        'standard_asr_mean_percent': asr_mean,
        'standard_asr_sample_sd_percent': asr_sd,
        'clean_acc_mean_percent': acc_mean,
        'clean_acc_sample_sd_percent': acc_sd,
        'sr_mean_percent': sr_mean,
        'sr_sample_sd_percent': sr_sd,
        'notes': 'All five TEST evaluations use TRAIN-selected weights; TEST never selects checkpoints.',
    })
    print('[FIVE-SEED SUMMARY] {}'.format(out), flush=True)


def main():
    p = parser()
    args = p.parse_args()
    if args.five_seeds:
        run_five_seeds(args)
    else:
        if args.seed is None:
            p.error('seed is required without --five-seeds')
        if args.evasion:
            p.error('--evasion is for --five-seeds TEST; use test CLI for a single seed')
        run(args)


if __name__ == '__main__':
    main()
