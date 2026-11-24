"""CGARepair adversarial local-weight FAMO optimizer (TF1 graph-mode compatible).

Place alongside adv_moo_2.py, read_CGARepair.py, cgarepair_adv_utils.py.

Original CGARepair MOEA/D: objectives are evaluated on discrete TRAIN error and
clean accuracy.  FAMO uses differentiable proxy losses (ADV/clean cross-entropy)
to update only the SAME selected Conv/Dense kernel output channels. Checkpoint
selection uses the original TRAIN discrete metric and guard. Independent TEST
uses adv_famo_test_2.py. No TEST data are accessed here.

FAMO model update and task-logit update follow Liu et al. NeurIPS 2023 and the
public FAMO implementation (Cranial-XIX/FAMO/famo.py). Gradient clipping, Adam,
finite-batch TRAIN selection, and mask/bounds are engineering adaptations.
"""
import argparse
import csv
from datetime import datetime, timezone
import os
import pickle
import time
from pathlib import Path

import numpy as np

from read_cgarepair_igd_ad import load_ranking, export_ranking, epsilon_string, model_name_checked
from cgarepair_adv_utils import (CONFIG, initialize_tf, load_adv_and_clean,
                                 model_file, calculate_metrics, save_json)
from adv_moo_2 import (map_players_to_weights, original_vector, apply_vector,
                       choose_train_solution, reduce_archive, run_directory)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max())
    return e / e.sum()


class FAMOTaskWeights:
    """Two-task FAMO logit optimizer, matching official Adam(weight_decay=gamma).

    Model gradient: sum_i [ (z_i / D_i) / sum_j(z_j / D_j) ] grad(L_i).
    Logit descent: gradient J_softmax(z)^T [log(D_before)-log(D_after)].
    D_i = L_i - L_i_min + eps; here CE_min=0 for both tasks.
    """
    def __init__(self, count=2, learning_rate=0.025, gamma=0.01, eps=1e-8):
        if count != 2:
            raise ValueError('This CGARepair adapter expects two tasks')
        if learning_rate <= 0 or gamma < 0 or eps <= 0:
            raise ValueError('Invalid FAMO task-weight hyperparameters')
        self.w = np.zeros(count, dtype=np.float64)
        self.m = np.zeros(count, dtype=np.float64)
        self.v = np.zeros(count, dtype=np.float64)
        self.steps = 0
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.eps = float(eps)

    @property
    def z(self):
        return softmax(self.w)

    def update(self, before, after):
        before = np.asarray(before, dtype=np.float64)
        after = np.asarray(after, dtype=np.float64)
        if before.shape != (2,) or after.shape != (2,):
            raise ValueError('Expected two task losses')
        if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
            raise FloatingPointError('Nonfinite FAMO loss')
        if np.any(before < 0) or np.any(after < 0):
            raise ValueError('Task CE losses must be nonnegative')
        delta = np.log(before + self.eps) - np.log(after + self.eps)
        z = self.z
        # J_softmax^T @ delta = z * (delta - dot(z,delta)).
        gradient = z * (delta - np.dot(z, delta)) + self.gamma * self.w
        self.steps += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * (gradient * gradient)
        m_hat = self.m / (1.0 - 0.9 ** self.steps)
        v_hat = self.v / (1.0 - 0.999 ** self.steps)
        self.w -= self.lr * m_hat / (np.sqrt(v_hat) + 1.e-8)
        self.w = np.clip(self.w, -30, 30)
        return delta


def create_masks_and_bounds(base_weights, specs, lower, upper):
    """Mask/select only named kernel[..., output_neuron] blocks.

    Every other weight entry receives mask=0 and is never projected.
    """
    if len(lower) != len(upper) or len(lower) != specs[-1]['stop']:
        raise ValueError('Flattened bounds/specs dimension mismatch')
    blocks = {}
    for s in specs:
        idx = s['weight_index']
        if idx not in blocks:
            kernel = base_weights[idx]
            blocks[idx] = {
                'mask': np.zeros(kernel.shape, dtype=np.float32),
                'lower': np.zeros(kernel.shape, dtype=np.float32),
                'upper': np.zeros(kernel.shape, dtype=np.float32),
                'layer_index': s['layer_index'],
            }
        entry = blocks[idx]
        j = s['neuron_index']
        if np.any(entry['mask'][..., j] > 0):
            raise ValueError('Duplicated selected neuron')
        entry['mask'][..., j] = 1.0
        entry['lower'][..., j] = lower[s['start']:s['stop']].reshape(s['shape'])
        entry['upper'][..., j] = upper[s['start']:s['stop']].reshape(s['shape'])
        if np.any(entry['lower'][..., j] >= entry['upper'][..., j]):
            raise ValueError('Invalid search bounds')
    return blocks


def check_weight_scope(before, after, specs, lower, upper):
    """Bit-exact audit of unselected weights, including BN and biases."""
    if len(before) != len(after):
        raise RuntimeError('Model weight structure changed')
    blocks = create_masks_and_bounds(before, specs, lower, upper)
    changed = 0
    for i, (a, b) in enumerate(zip(before, after)):
        if a.shape != b.shape or not np.all(np.isfinite(b)):
            raise RuntimeError('Shape or finite-value mismatch in tensor {}'.format(i))
        mask = blocks[i]['mask'].astype(bool) if i in blocks else None
        if mask is None:
            if not np.array_equal(a, b):
                raise RuntimeError('FAMO modified nonselected model tensor {}'.format(i))
        else:
            if not np.array_equal(a[~mask], b[~mask]):
                raise RuntimeError('FAMO modified entries outside selected mask in {}'.format(i))
            if np.any(b[mask] < blocks[i]['lower'][mask]-1.e-5) or np.any(
                    b[mask] > blocks[i]['upper'][mask]+1.e-5):
                raise RuntimeError('FAMO modified a weight beyond search bounds in {}'.format(i))
            changed += int(np.count_nonzero(a[mask] != b[mask]))
    return {'modified_parameter_count': int(changed),
            'selected_parameter_count': int(sum(s['num_parameters'] for s in specs)),
            'outside_scope_modified_count': 0,
            'other_tensor_untouched': True}


def detect_output_mode(session, network, x, requested):
    if requested != 'auto':
        return requested
    if len(x) < 1:
        raise ValueError('Cannot detect output mode without examples')
    output = np.asarray(session.run(network.output, feed_dict={network.input: x[:min(8, len(x))]}))
    if output.ndim != 2 or output.shape[1] < 2:
        raise ValueError('Expected multiclass network output of shape [N,C], got {}'.format(output.shape))
    if np.all(output >= -1.e-6) and np.all(output <= 1+1.e-6) and np.allclose(
            np.sum(output, axis=1), 1.0, rtol=0.02, atol=0.02):
        return 'probabilities'
    return 'logits'


class MaskedFAMOGraph:
    """One backpropagation for both losses, masked Adam update, bounded projection."""
    def __init__(self, tf, session, network, base_weights, specs, lower, upper,
                 model_lr, max_grad_norm, loss_eps, output_mode):
        v1 = tf.compat.v1
        self.tf, self.session = tf, session
        self.x = network.input
        self.adv_labels = v1.placeholder(tf.int32, shape=[None], name='famo_adv_labels')
        self.clean_labels = v1.placeholder(tf.int32, shape=[None], name='famo_clean_labels')
        self.num_adv = v1.placeholder(tf.int32, shape=[], name='famo_adv_count')
        self.z = v1.placeholder(tf.float32, shape=[2], name='famo_task_weights')
        self.blocks = create_masks_and_bounds(base_weights, specs, lower, upper)
        if output_mode not in ('probabilities', 'logits'):
            raise ValueError('Unsupported output mode')
        predictions = tf.cast(network.output, tf.float32)
        if predictions.shape.ndims != 2:
            raise ValueError('Expected [batch,classes] output')
        if output_mode == 'probabilities':
            # log(probs) are valid softmax logits, preserving the pretrained
            # model's original output convention (no accidental softmax twice).
            ce_logits = tf.math.log(tf.clip_by_value(predictions, 1.e-7, 1.0))
        else:
            ce_logits = predictions
        adv_logits = ce_logits[:self.num_adv]
        clean_logits = ce_logits[self.num_adv:]
        adv_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.adv_labels, logits=adv_logits))
        clean_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.clean_labels, logits=clean_logits))
        self.task_losses = tf.stack([adv_loss, clean_loss], name='famo_task_CE_losses')
        positive = tf.maximum(self.task_losses, 0.0) + float(loss_eps)
        # Official FAMO: grad(sum_i z_i log(D_i) / stop_grad(sum_i z_i/D_i)).
        c = tf.stop_gradient(tf.reduce_sum(self.z / positive))
        weighted = tf.reduce_sum(self.z * tf.math.log(positive)) / c
        selected_vars = []
        mask_tensors = []
        bound_tensors = []
        for idx, block in sorted(self.blocks.items()):
            layer_idx = int(block['layer_index'])
            layer = network.layers[layer_idx]
            if not layer.weights:
                raise RuntimeError('Mapped layer has no Keras weight variable')
            var = layer.weights[0]
            if tuple(var.shape.as_list()) != base_weights[idx].shape:
                raise RuntimeError('Keras kernel variable shape differs from mapped kernel')
            selected_vars.append(var)
            mask_tensors.append(tf.constant(block['mask'], dtype=var.dtype.base_dtype))
            bound_tensors.append((
                tf.constant(block['lower'], dtype=var.dtype.base_dtype),
                tf.constant(block['upper'], dtype=var.dtype.base_dtype)))
        raw_grads = tf.gradients(weighted, selected_vars)
        if any(g is None for g in raw_grads):
            disconnected = [int(i) for i, g in enumerate(raw_grads) if g is None]
            raise RuntimeError('Selected kernel is disconnected from output; indices {}'.format(disconnected))
        grads = []
        for grad, mask in zip(raw_grads, mask_tensors):
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
            grads.append(grad * mask)
        global_norm = tf.compat.v1.global_norm(grads)
        self.grad_norm = global_norm
        if max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, float(max_grad_norm), use_norm=global_norm)
        old_variables = {v.name for v in v1.global_variables()}
        opt = v1.train.AdamOptimizer(learning_rate=float(model_lr))
        update = opt.apply_gradients(list(zip(grads, selected_vars)))
        # Bound only selected kernel slices.  Untouched entries are bit-exact.
        projections = []
        with tf.control_dependencies([update]):
            for var, mask, bounds in zip(selected_vars, mask_tensors, bound_tensors):
                low_t, high_t = bounds
                clipped = tf.clip_by_value(var, low_t, high_t)
                projections.append(v1.assign(var, tf.where(tf.greater(mask, 0), clipped, var)))
        self.train_step = tf.group(*projections, name='famo_masked_adam_step')
        new_vars = [v for v in v1.global_variables() if v.name not in old_variables]
        if new_vars:
            session.run(v1.variables_initializer(new_vars))
        self.trainable_kernel_count = len(selected_vars)

    def feed(self, x_adv, y_adv, x_clean, y_clean, z):
        return {self.x: np.concatenate([x_adv, x_clean], axis=0),
                self.adv_labels: np.asarray(y_adv, dtype=np.int32),
                self.clean_labels: np.asarray(y_clean, dtype=np.int32),
                self.num_adv: int(len(x_adv)),
                self.z: np.asarray(z, dtype=np.float32)}

    def step(self, feed):
        # Train op needs gradient from same task losses; graph Session.run
        # shares the predecessor tensors so returned losses are pre-update.
        before, _, norm = self.session.run(
            [self.task_losses, self.train_step, self.grad_norm], feed_dict=feed)
        after = self.session.run(self.task_losses, feed_dict=feed)
        if not (np.all(np.isfinite(before)) and np.all(np.isfinite(after))):
            raise FloatingPointError('Nonfinite FAMO losses; reduce --model-lr')
        return np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64), float(norm)


def append_candidate(network, session, specs, xs, fs, steps,
                     adv_x, adv_y, clean_x, clean_y, batch_size, step):
    metrics = calculate_metrics(network, adv_x, adv_y, clean_x, clean_y, session, batch_size)
    if not all(np.isfinite(metrics[k]) for k in ('adv_error_rate', 'clean_accuracy')):
        raise FloatingPointError('Nonfinite full TRAIN metric')
    vec = original_vector(network.get_weights(), specs)
    # Save snapshot's *actual* model weights, not reconstructed predictions.
    xs.append(vec)
    fs.append(np.array([metrics['adv_error_rate'], -metrics['clean_accuracy']], dtype=np.float64))
    steps.append(int(step))
    return metrics


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    with open(str(path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_parser():
    p = argparse.ArgumentParser(description='CGARepair ADV masked FAMO local-weight repair')
    p.add_argument('model_name', choices=('mnist', 'cifar10'))
    p.add_argument('--optimizer', choices=('famo',), default='famo',
                   help='Only FAMO is supported; compatibility flag')
    p.add_argument('epsilon', help='MNIST 0.05, CIFAR10 0.1')
    p.add_argument('seed', type=int, nargs='?', help='Optimization seed; omit with --five-seeds')
    p.add_argument('--five-seeds', action='store_true',
                   help='Run five fixed optimization seeds and independent TEST automatically')
    p.add_argument('--loc-json')
    p.add_argument('--loc-root')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-tag', default='')
    p.add_argument('--quick', action='store_true', help='5-step smoke test (TRAIN only)')
    p.add_argument('--overwrite', action='store_true', help='Explicitly overwrite an existing FAMO summary')
    p.add_argument('--cpu', action='store_true', help='Disable CUDA before importing TensorFlow')
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--eval-samples', type=int, default=10000,
                   help='Use up to this number of TRAIN adversarial and clean examples each')
    p.add_argument('--grad-batch-size', type=int, default=64,
                   help='Examples PER TASK in each FAMO optimizer update')
    p.add_argument('--batch-size', type=int, default=256,
                   help='Batch size of full TRAIN checkpoint-selection evaluation')
    p.add_argument('--eval-every', type=int, default=20,
                   help='Evaluate full TRAIN metrics and save candidate every N steps')
    p.add_argument('--model-lr', type=float, default=0.001)
    p.add_argument('--task-lr', type=float, default=0.025)
    p.add_argument('--task-decay', type=float, default=0.01)
    p.add_argument('--loss-eps', type=float, default=1e-8)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--search-radius', type=float, default=10.0,
                   help='Same mask bounds as MOEAD: max(-5,x0-radius),min(5,x0+radius)')
    p.add_argument('--threshold-acc', type=float, default=None)
    p.add_argument('--threshold-adv-error', type=float, default=None)
    p.add_argument('--archive-size', type=int, default=60)
    p.add_argument('--save-front-limit', type=int, default=10)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--output-mode', choices=('auto', 'probabilities', 'logits'), default='auto')
    return p


def run(args):
    wall_start = time.perf_counter()
    ds, eps = model_name_checked(args.model_name), epsilon_string(args.epsilon)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.quick:
        args.max_steps = 5
        args.eval_samples = min(int(args.eval_samples), 128)
        args.grad_batch_size = min(int(args.grad_batch_size), 16)
        args.eval_every = 1
    if min(args.max_steps, args.eval_samples, args.grad_batch_size, args.batch_size,
           args.eval_every, args.log_every) <= 0:
        raise ValueError('steps, sample sizes, batch sizes and intervals must be >0')
    if (args.model_lr <= 0 or args.task_lr <= 0 or args.task_decay < 0 or
            args.search_radius <= 0 or args.loss_eps <= 0 or args.max_grad_norm < 0 or
            args.archive_size < 2 or args.save_front_limit < 0):
        raise ValueError('Invalid FAMO learning rates, bounds or archive setting')
    outdir = run_directory(args.output_root, ds, eps, args.seed, 'famo',
                           run_tag=args.run_tag, quick=args.quick)
    if (outdir / 'summary.json').exists() and not args.overwrite:
        raise FileExistsError('Existing FAMO result: {}'.format(outdir / 'summary.json'))
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError('Nonempty FAMO directory (possibly incomplete): {}. '
                              'Use a new --run-tag; do not delete archived experiments.'.format(outdir))
    if outdir.exists() and any(outdir.iterdir()) and args.overwrite:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup = outdir.with_name(outdir.name+'_backup_'+stamp)
        if backup.exists():
            raise FileExistsError('Backup exists; choose a different tag: {}'.format(backup))
        outdir.rename(backup)
        print('[BACKUP] {} -> {}'.format(outdir, backup), flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    players, scores, loc = load_ranking(ds, eps, args.loc_seed, args.loc_samples,
                                         args.loc_json, args.loc_root)
    exported = export_ranking(players, scores, loc, outdir/'localization_input')
    loc_sec = time.perf_counter() - t
    print('[Locator] {} eps={} Top-{} from {}'.format(ds, eps, loc['top_k'], loc['source_json']), flush=True)
    print('[Locator] selected={}'.format(players), flush=True)

    t = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    try:
        K.set_learning_phase(0)
    except AttributeError:
        tf.compat.v1.keras.backend.set_learning_phase(0)
    from tensorflow.keras.models import load_model
    path = model_file(ds, args.model_root)
    model = load_model(path, compile=False)
    signature = loc.get('source_model_signature')
    if isinstance(signature, (list, tuple)) and signature and int(signature[0]) != os.stat(path).st_size:
        raise ValueError('Locator/model file-size mismatch: {} {}'.format(signature[0], os.stat(path).st_size))
    specs = map_players_to_weights(model, players)
    base_weights = [w.copy() for w in model.get_weights()]
    base_vec = original_vector(base_weights, specs)
    low = np.maximum(-5.0, base_vec - args.search_radius)
    high = np.minimum(5.0, base_vec + args.search_radius)
    if np.any(low >= high):
        raise ValueError('Invalid selection bounds (pretrained weights outside [-5,5]?)')
    setup_sec = time.perf_counter() - t
    print('[Repair] {} eps={} neurons={} decision_variables={} checkpoint={}'.format(
        ds, eps, len(specs), len(base_vec), path), flush=True)

    t = time.perf_counter()
    adv_x, adv_y, clean_x, clean_y, data_info = load_adv_and_clean(
        ds, eps, split='train', data_root=args.data_root, samples=args.eval_samples, seed=args.seed)
    data_sec = time.perf_counter() - t
    t = time.perf_counter()
    baseline = calculate_metrics(model, adv_x, adv_y, clean_x, clean_y, session, args.batch_size)
    baseline_sec = time.perf_counter() - t
    print('[Baseline] ADV error={:.6f}, cleanAcc={:.6f} (adv_n={}, clean_n={})'.format(
        baseline['adv_error_rate'], baseline['clean_accuracy'], len(adv_y), len(clean_y)), flush=True)
    min_acc = CONFIG[ds]['threshold_acc'] if args.threshold_acc is None else float(args.threshold_acc)
    max_err = CONFIG[ds]['threshold_adv_error'] if args.threshold_adv_error is None else float(args.threshold_adv_error)

    output_mode = detect_output_mode(session, model, clean_x, args.output_mode)
    graph_start = time.perf_counter()
    graph = MaskedFAMOGraph(tf, session, model, base_weights, specs, low, high,
                            args.model_lr, args.max_grad_norm, args.loss_eps, output_mode)
    graph_sec = time.perf_counter() - graph_start
    weights = FAMOTaskWeights(2, args.task_lr, args.task_decay, args.loss_eps)
    rng = np.random.RandomState(int(args.seed))
    X, F, eval_steps = [], [], []
    trace, snapshots = [], []
    baseline_record = append_candidate(model, session, specs,
                                       X, F, eval_steps, adv_x, adv_y, clean_x,
                                       clean_y, args.batch_size, 0)
    snapshots.append({'step': 0, 'train_adv_error': baseline_record['adv_error_rate'],
                      'train_clean_accuracy': baseline_record['clean_accuracy'],
                      'feasible': bool(baseline_record['adv_error_rate'] <= max_err and
                                       baseline_record['clean_accuracy'] >= min_acc),
                      'task_z_adv': float(weights.z[0]), 'task_z_clean': float(weights.z[1]),
                      'elapsed_opt_sec': 0.0})
    optimization_start = time.perf_counter()
    print('[FAMO] steps={} grad_batch={} per task, eval_every={} output={} kernels={} lr={} '.format(
        args.max_steps, args.grad_batch_size, args.eval_every, output_mode, graph.trainable_kernel_count,
        args.model_lr) + 'losses=(ADV_CE,CLEAN_CE) task_lr={}'.format(args.task_lr), flush=True)
    for step in range(1, int(args.max_steps) + 1):
        idx_adv = rng.randint(0, len(adv_y), size=args.grad_batch_size)
        idx_clean = rng.randint(0, len(clean_y), size=args.grad_batch_size)
        z_old = weights.z.copy()
        fd = graph.feed(adv_x[idx_adv], adv_y[idx_adv], clean_x[idx_clean], clean_y[idx_clean], z_old)
        before, after, gradnorm = graph.step(fd)
        delta = weights.update(before, after)
        if not np.isfinite(gradnorm):
            raise FloatingPointError('Nonfinite gradient norm')
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print('[FAMO step {:04d}/{}] advCE={:.5f}->{:.5f} cleanCE={:.5f}->{:.5f} '
                  'z=({:.3f},{:.3f}) gradnorm={:.4f}'.format(
                    step, args.max_steps, before[0], after[0], before[1], after[1],
                    weights.z[0], weights.z[1], gradnorm), flush=True)
        trace.append({'step': step, 'adv_ce_pre': float(before[0]), 'adv_ce_post': float(after[0]),
                      'clean_ce_pre': float(before[1]), 'clean_ce_post': float(after[1]),
                      'adv_log_improvement': float(delta[0]), 'clean_log_improvement': float(delta[1]),
                      'z_adv': float(weights.z[0]), 'z_clean': float(weights.z[1]),
                      'gradient_global_norm': float(gradnorm),
                      'elapsed_opt_sec': float(time.perf_counter()-optimization_start)})
        if step % args.eval_every == 0 or step == args.max_steps:
            mt = append_candidate(model, session, specs,
                                  X, F, eval_steps, adv_x, adv_y, clean_x,
                                  clean_y, args.batch_size, step)
            valid = bool(mt['adv_error_rate'] <= max_err and mt['clean_accuracy'] >= min_acc)
            snapshots.append({'step': int(step), 'train_adv_error': mt['adv_error_rate'],
                              'train_clean_accuracy': mt['clean_accuracy'], 'feasible': valid,
                              'task_z_adv': float(weights.z[0]), 'task_z_clean': float(weights.z[1]),
                              'elapsed_opt_sec': float(time.perf_counter()-optimization_start)})
            print('[TRAIN step={}] ADVerr={:.6f} cleanAcc={:.6f} feasible={}'.format(
                step, mt['adv_error_rate'], mt['clean_accuracy'], valid), flush=True)
    optimization_sec = time.perf_counter() - optimization_start
    X, F = np.asarray(X, dtype=np.float64), np.asarray(F, dtype=np.float64)
    candidate_x, candidate_f = reduce_archive(X, F, args.archive_size)
    selected, policy = choose_train_solution(candidate_f, min_acc, max_err)
    feasible = (candidate_f[:, 0] <= max_err) & (-candidate_f[:, 1] >= min_acc)
    t = time.perf_counter()
    write_csv(outdir/'famo_step_history.csv', trace)
    write_csv(outdir/'famo_train_snapshots.csv', snapshots)
    front_dir = outdir/'pareto_front'
    front_dir.mkdir(parents=True, exist_ok=True)
    # FAMO historical TRAIN snapshots, not an evolutionary population front.
    front_txt = []
    selected_vector = candidate_x[selected].copy()
    best_file = outdir / 'best_weights.h5'
    select_weights = apply_vector([w.copy() for w in base_weights], specs, selected_vector)
    check = check_weight_scope(base_weights, select_weights, specs, low, high)
    model.set_weights(select_weights)
    model.save_weights(str(best_file))
    stored = model.get_weights()
    check_weight_scope(base_weights, stored, specs, low, high)
    max_front = min(len(candidate_x), args.save_front_limit)
    ids = list(range(max_front))
    if selected not in ids and ids:
        ids[-1] = selected
    for i in sorted(set(ids)):
        w = apply_vector([v.copy() for v in base_weights], specs, candidate_x[i])
        check_weight_scope(base_weights, w, specs, low, high)
        model.set_weights(w)
        filename = front_dir / 'weights_{}.h5'.format(i)
        model.save_weights(str(filename))
        front_txt.append({'index': i, 'train_adv_error': float(candidate_f[i, 0]),
                          'train_robustness': float(1 - candidate_f[i, 0]),
                          'train_accuracy': float(-candidate_f[i, 1]), 'file': str(filename)})
    model.set_weights(base_weights)
    write_csv(outdir/'pareto_front.txt', front_txt)
    with open(str(outdir/'pareto_front.pkl'), 'wb') as f:
        pickle.dump({'X': candidate_x, 'F': candidate_f,
                     'history': snapshots, 'selected_index': int(selected),
                     'total_evaluations': len(snapshots),
                     'all_evaluated_X': X, 'all_evaluated_F': F,
                     'all_evaluated_steps': eval_steps,
                     'source_locator': loc['source_json'],
                     'note': 'FAMO TRAIN evaluated snapshots; NOT a MOEA/D population'}, f, protocol=4)
    save_sec = time.perf_counter()-t
    duration = time.perf_counter()-wall_start
    timing = {
        'localization_previous_run_sec': loc.get('source_runtime_sec'),
        'localization_previous_run_not_included_in_repair_wall': True,
        'localization_read_export_sec': float(loc_sec),
        'model_load_and_weight_mapping_sec': float(setup_sec),
        'train_data_load_sec': float(data_sec),
        'train_baseline_eval_sec': float(baseline_sec),
        'famo_graph_build_sec': float(graph_sec),
        'optimization_wall_sec': float(optimization_sec),
        'pareto_and_checkpoint_save_sec': float(save_sec),
        'total_repair_run_wall_sec': float(duration),
        'notes': ('TRAIN snapshot evaluation is included within optimization wall. '
                  'FAMO gradient steps are NOT MOEA/D fitness evaluations.'),
    }
    summary = {
        'schema': 'CGARepair_adv_FAMO_masked_weight_repair_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'adv', 'model': ds, 'epsilon': eps, 'optimizer': 'famo',
        'method': 'CGARepair_FAMO',
        'optimization_seed': int(args.seed), 'localization_seed': int(args.loc_seed),
        'localization': loc, 'localization_exported_dir': exported,
        'model_file': path, 'selected_players': players, 'top_k': int(loc['top_k']),
        'total_ranked_neurons': int(loc['num_players']),
        'repair_specs': [{k: list(v) if isinstance(v, tuple) else v for k,v in s.items()} for s in specs],
        'decision_variables': int(len(base_vec)),
        'parameter_budget_percent': float(100.0*len(base_vec)/max(model.count_params(), 1)),
        'parameter_scope': 'Selected Conv/Dense kernel output channels only; BN and biases frozen',
        'mask_audit': check,
        'train_data': data_info, 'original_train_metrics': baseline,
        'optimization_proxy_losses': ['adv_sparse_cross_entropy', 'clean_sparse_cross_entropy'],
        'objective_definitions': ['adv_error_rate=minimize', '-clean_accuracy=minimize'],
        'threshold_accuracy': float(min_acc), 'threshold_adv_error': float(max_err),
        'gradient_steps_requested': int(args.max_steps),
        'gradient_steps_actual': int(len(trace)),
        'train_snapshot_evaluations_actual': int(len(snapshots)),
        'front_size': int(len(candidate_x)), 'front_feasible_count': int(feasible.sum()),
        'selected_index': int(selected), 'selection_policy': policy,
        'selected_train_adv_error': float(candidate_f[selected, 0]),
        'selected_train_robustness': float(1.0 - candidate_f[selected, 0]),
        'selected_train_clean_accuracy': float(-candidate_f[selected, 1]),
        'best_weights_file': str(best_file.resolve()),
        'optimizer_parameters': {'name': 'FAMO', 'max_steps': int(args.max_steps),
                                 'grad_batch_size_per_task': int(args.grad_batch_size),
                                 'evaluation_batch_size': int(args.batch_size),
                                 'full_train_eval_every': int(args.eval_every),
                                 'model_lr': float(args.model_lr), 'task_lr': float(args.task_lr),
                                 'task_decay': float(args.task_decay), 'loss_eps': float(args.loss_eps),
                                 'max_grad_norm': float(args.max_grad_norm),
                                 'search_radius': float(args.search_radius),
                                 'output_mode': output_mode,
                                 'archive_size': int(args.archive_size),
                                 'gradient_algorithm': 'Official FAMO log-loss weighting and task-logit Adam; masked model Adam'},
        'task_logit_final': weights.w.tolist(),
        'task_weight_final': weights.z.tolist(),
        'optimizer_history': snapshots, 'timing_sec': timing,
        'total_time_sec': float(duration),
        'comparability_note': ('Fixed locator, model, TRAIN pool, parameter mask and TRAIN decision guard; '
                               'FAMO optimizes surrogate CE while MOEA/D directly optimizes discrete metrics. '
                               'Gradient steps and MOEA/D FEs are not equivalent.'),
    }
    tmp_summary = outdir / 'summary.json.incomplete'
    save_json(tmp_summary, summary)
    os.replace(str(tmp_summary), str(outdir / 'summary.json'))
    print('[Done] {} eps={} famo steps={} train err={:.6f} cleanAcc={:.6f} '
          '(baseline err={:.6f} cleanAcc={:.6f})'.format(
            ds, eps, args.max_steps, summary['selected_train_adv_error'],
            summary['selected_train_clean_accuracy'], baseline['adv_error_rate'],
            baseline['clean_accuracy']), flush=True)
    print('[Mask] modified={} / allowed={}, outside_scope=0'.format(
        check['modified_parameter_count'], check['selected_parameter_count']), flush=True)
    print('[Timing] optimization={:.2f}s total={:.2f}s'.format(optimization_sec, duration), flush=True)
    print('[Files] summary={} best={} front={}'.format(outdir/'summary.json', best_file,
                                                       outdir/'pareto_front.pkl'), flush=True)
    return summary


def run_five_seeds(args):
    """Five isolated TRAIN/TEST processes, with safe resume and mean +/- sample SD."""
    import json
    import statistics
    import subprocess
    import sys

    seeds = (42, 123, 256, 1024, 2026)
    ds, eps = model_name_checked(args.model_name), epsilon_string(args.epsilon)
    if args.quick or args.overwrite:
        raise ValueError('--five-seeds forbids --quick and --overwrite to protect results')
    if args.seed is not None:
        raise ValueError('Omit positional seed when using --five-seeds')
    if args.loc_seed != 256 or not args.loc_json:
        raise ValueError('Five-seed protocol requires --loc-seed 256 and --loc-json')
    if not Path(args.loc_json).is_file():
        raise FileNotFoundError('Missing locator: {}'.format(args.loc_json))
    if args.max_steps != 5000 or abs(args.task_decay - 0.001) > 1e-10:
        raise ValueError('Official five-seed protocol: --max-steps 5000 --task-decay 0.001')
    if args.run_tag != 'paper_gamma001_steps5000':
        raise ValueError('Use --run-tag paper_gamma001_steps5000 to safely reuse seed 256')

    def check_identity(obj, seed, label):
        if (obj.get('model') != ds or epsilon_string(obj.get('epsilon')) != eps
                or obj.get('optimizer') != 'famo'
                or int(obj.get('optimization_seed', -1)) != seed):
            raise ValueError('{} identity mismatch, seed {}'.format(label, seed))

    # Child processes are isolated: no cumulative TF1 graph or GPU memory buildup.
    forward = [x for x in sys.argv[3:] if x != '--five-seeds']
    py_file = str(Path(__file__).resolve())
    test_file = str(Path(__file__).with_name('adv_famo_test_2.py').resolve())
    if not Path(test_file).is_file():
        raise FileNotFoundError('Missing independent TEST script: {}'.format(test_file))
    records = []
    for seed in seeds:
        directory = run_directory(args.output_root, ds, eps, seed, 'famo',
                                  run_tag=args.run_tag, quick=False)
        summary_file = directory / 'summary.json'
        weight_file = directory / 'best_weights.h5'
        test_result = directory / 'test_results.json'
        print('\n========== {} seed={} TRAIN+TEST =========='.format(ds, seed), flush=True)
        if summary_file.is_file() and weight_file.is_file():
            print('[SKIP TRAIN] {}'.format(directory), flush=True)
        elif directory.exists() and any(directory.iterdir()):
            raise RuntimeError('Incomplete existing result; inspect before retry: {}'.format(directory))
        else:
            subprocess.run([sys.executable, '-u', py_file, ds, eps, str(seed)] + forward,
                           check=True)
        if not summary_file.is_file() or not weight_file.is_file():
            raise RuntimeError('TRAIN did not produce summary and weights: {}'.format(directory))
        with summary_file.open('r', encoding='utf-8') as f:
            summary = json.load(f)
        check_identity(summary, seed, 'TRAIN')
        hp = summary.get('optimizer_parameters', {})
        checks = {'max_steps': args.max_steps, 'task_decay': args.task_decay,
                  'model_lr': args.model_lr, 'task_lr': args.task_lr,
                  'grad_batch_size_per_task': args.grad_batch_size,
                  'full_train_eval_every': args.eval_every,
                  'search_radius': args.search_radius}
        for key, wanted in checks.items():
            actual = hp.get(key)
            if actual is None or abs(float(actual) - float(wanted)) > 1e-8:
                raise ValueError('TRAIN {} mismatches requested {}: {} != {}'.format(
                    directory, key, actual, wanted))
        if (int(summary.get('gradient_steps_actual', -1)) != args.max_steps
                or int(summary.get('localization_seed', -1)) != 256):
            raise ValueError('Existing TRAIN steps / localization seed differ: {}'.format(directory))
        loc_source = summary.get('localization', {}).get('source_json')
        if not loc_source or Path(loc_source).resolve() != Path(args.loc_json).resolve():
            raise ValueError('Existing TRAIN localization differs: {}'.format(directory))

        if test_result.is_file():
            print('[SKIP TEST] {}'.format(test_result), flush=True)
        else:
            cmd = [sys.executable, '-u', test_file, ds, eps, str(seed),
                   '--optimizer', 'famo', '--run-dir', str(directory),
                   '--test-samples', '0', '--batch-size', str(args.batch_size)]
            if args.model_root:
                cmd += ['--model-root', args.model_root]
            if args.data_root:
                cmd += ['--data-root', args.data_root]
            subprocess.run(cmd, check=True)
        with test_result.open('r', encoding='utf-8') as f:
            test = json.load(f)
        check_identity(test, seed, 'TEST')
        if Path(test.get('selected_checkpoint', '')).resolve() != weight_file.resolve():
            raise ValueError('TEST used a different checkpoint: {}'.format(test_result))
        after = test['repaired_test_metrics']
        before = test['original_test_metrics']
        records.append({'seed': seed,
                        'before_aer_percent': float(before['adv_error_rate']) * 100,
                        'after_aer_percent': float(after['adv_error_rate']) * 100,
                        'before_clean_acc_percent': float(before['clean_accuracy']) * 100,
                        'after_clean_acc_percent': float(after['clean_accuracy']) * 100,
                        'test_file': str(test_result.resolve())})

    aer = [r['after_aer_percent'] for r in records]
    acc = [r['after_clean_acc_percent'] for r in records]
    print('\n========== {} FAMO FIVE-SEED TEST =========='.format(ds.upper()))
    print('Seed      TEST AER(%)     TEST CleanAcc(%)')
    for r in records:
        print('{:<8d}  {:>11.4f}       {:>11.4f}'.format(
            r['seed'], r['after_aer_percent'], r['after_clean_acc_percent']))
    print('MEAN +/- SD TEST AER     = {:.4f} +/- {:.4f}%'.format(
        statistics.mean(aer), statistics.stdev(aer)))
    print('MEAN +/- SD TEST CleanAcc= {:.4f} +/- {:.4f}%'.format(
        statistics.mean(acc), statistics.stdev(acc)))
    out = Path(args.output_root) / 'adv' / ds / ('epsilon_' + eps) / 'CGARepair' / (
        'famo_' + args.run_tag + '_five_seed_TEST_summary.json')
    save_json(out, {'model': ds, 'epsilon': eps, 'optimizer': 'famo',
                    'seeds': list(seeds), 'records': records,
                    'aer_mean_percent': statistics.mean(aer),
                    'aer_sample_sd_percent': statistics.stdev(aer),
                    'acc_mean_percent': statistics.mean(acc),
                    'acc_sample_sd_percent': statistics.stdev(acc),
                    'note': 'All scores from TRAIN-selected checkpoints on independent TEST.'})
    print('[FIVE-SEED SUMMARY] {}'.format(out), flush=True)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.five_seeds:
        run_five_seeds(args)
    else:
        if args.seed is None:
            parser.error('seed is required unless --five-seeds is specified')
        run(args)


if __name__ == '__main__':
    main()
