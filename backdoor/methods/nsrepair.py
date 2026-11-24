"""
NSRepair: Shapley-guided neural network repair.

GradientSHAP-based neuron localization + NSGA-III multi-objective repair.

GradientSHAP (Aumann-Shapley) replaces Monte Carlo Shapley for neuron-level
attribution — one backward pass computes SHAP values for all neurons via
Integrated Gradients approximation, reducing localization from ~24h to ~seconds.

Repair: NSGA-III multi-objective optimization on the kernel weights of the
top-k neurons. After optimization, Pareto front solutions are evaluated on the
test set, and the best solution is selected according to defect-specific
thresholds (with a fallback to the utopia-closest solution if none qualifies).

Reference:
  "Neural Network Repair With Shapley-Guided Search"
  https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70147
"""

import os
import json
import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from pymoo.optimize import minimize
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.util.ref_dirs import get_reference_directions

from repair_config import (
    DATASET_CONFIG,
    sess_run,
    value_accuracy,
    value_backdoor,
)



# ============================================================
# Constants
# ============================================================

# --- GradientSHAP localization ---
SHAP_N_SAMPLE = 256        # max defect samples for SHAP computation
SHAP_CLEAN_SAMPLE = 256    # max clean samples for baseline

# --- Neuron selection ---
REPAIR_RATIO = 0.01        # fraction of total neurons to repair (1%)
REPAIR_NUM_MAX = 11        # hard cap on neurons to repair

# --- NSGA-III ---
NSGA3_POP_SIZE = 100       # population size
NSGA3_N_GEN = 100          # max generations
NSGA3_PATIENCE = 10        # patience for early stopping
NSGA3_EVAL_SAMPLE = 10000  # max samples for fitness evaluation
NSGA3_PATIENCE_ACC = 0.9  # accuracy threshold for patience check
NSGA3_PATIENCE_DEFECT_BACKDOOR = 0.1
NSGA3_PATIENCE_DEFECT_ADV = 0.5
NSGA3_PATIENCE_DEFECT_FAIRNESS = 0.1

# --- Pareto selection thresholds ---
# Relative to initial (pre-repair) performance on repair data:
#  - acc_drop: max allowed accuracy drop
#  - defect_improve_ratio: defect must drop to at most this fraction of initial
SELECT_ACC_DROP = 0.05          # max allowed accuracy drop from baseline
SELECT_REL_DEFECT = 0.5         # defect must drop below initial * SELECT_REL_DEFECT
SELECT_HARD_THRESHOLD_BACKDOOR = 0.10  # absolute cap for backdoor
SELECT_HARD_THRESHOLD_ADV = 0.40       # absolute cap for adv error
SELECT_HARD_THRESHOLD_FAIRNESS = 0.10  # absolute cap for fairness


# ============================================================
# Evaluation helpers (mirror care_repair.py for self-containedness)
# ============================================================

def _value_adv_error(model, images_adv, labels_true):
    """Adversarial error rate: fraction misclassified."""
    pred_adv = np.argmax(sess_run(model, model.output, images_adv), axis=-1)
    return float(np.sum(pred_adv != labels_true) / len(labels_true))


def _value_fairness_gap(model, images, labels):
    """|female_acc - male_acc|. Female=0, Male=1."""
    outs = sess_run(model, model.output, images)
    pred = np.argmax(outs, axis=-1)
    correct = pred == labels
    female_mask = (labels == 0)
    male_mask = (labels == 1)
    female_acc = float(np.sum(female_mask & correct)
                       / max(1, np.sum(female_mask)))
    male_acc = float(np.sum(male_mask & correct)
                     / max(1, np.sum(male_mask)))
    return female_acc, male_acc, abs(female_acc - male_acc)


def _compute_defect_rate(model, x, y, defect_type, target_label):
    """Compute defect-specific metric (to be minimized)."""
    if defect_type == 'backdoor':
        return value_backdoor(model, x, y, target_label)
    elif defect_type == 'adv':
        return _value_adv_error(model, x, y)
    elif defect_type == 'fairness':
        _, _, gap = _value_fairness_gap(model, x, y)
        return gap
    else:
        raise ValueError(f'Unknown defect type: {defect_type}')


# ============================================================
# GradientSHAP localization
# ============================================================

def _build_defect_loss(logits, y_ph, defect_type, target_label):
    """Build a differentiable proxy loss for the defect metric.

    The loss approximates the direction we want to minimize:
    - backdoor:  penalize predicting target_label  → -log(1 - P[target])
    - adv:       penalize misclassification         → cross-entropy
    - fairness:  penalize gender accuracy gap       → (CE_f - CE_m)^2
    """
    if defect_type == 'backdoor':
        probs = tf.nn.softmax(logits)
        loss = -tf.reduce_mean(
            tf.math.log(tf.clip_by_value(1.0 - probs[:, target_label],
                                         1e-8, 1.0)))
    elif defect_type == 'adv':
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_ph, logits=logits))
    elif defect_type == 'fairness':
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_ph, logits=logits)
        female_mask = tf.cast(tf.equal(y_ph, 0), tf.float32)
        male_mask = tf.cast(tf.equal(y_ph, 1), tf.float32)
        n_f = tf.reduce_sum(female_mask) + 1e-8
        n_m = tf.reduce_sum(male_mask) + 1e-8
        ce_f = tf.reduce_sum(ce * female_mask) / n_f
        ce_m = tf.reduce_sum(ce * male_mask) / n_m
        loss = tf.square(ce_f - ce_m)
    else:
        raise ValueError(f'Unknown defect type: {defect_type}')
    return loss


def compute_gradientshap_localization(model, defect_data, clean_data,
                                       defect_type, target_label,
                                       n_sample=SHAP_N_SAMPLE,
                                       cache_path=None):
    """GradientSHAP-based neuron Shapley value computation.

    For each neuron i, the Aumann-Shapley value is approximated via
    two-point Integrated Gradients:
        SHAP_i = |0.5 * (∇_i(defect) + ∇_i(clean)) · (a_i^defect - a_i^clean)|

    where ∇_i is the gradient of the defect proxy loss w.r.t. the neuron's
    activation, and a_i is the neuron's activation value.

    This replaces Monte Carlo TMC-Shapley (~24h for 200 iterations) with
    a single forward + backward pass per data type (~seconds).

    Returns:
        dict with 'players' (names) and 'scores' (SHAP values, higher = more
        important for the defect).
    """
    # ---- Check cache ----
    if cache_path and os.path.exists(cache_path):
        print(f'  Loading GradientSHAP localization from cache: {cache_path}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if cached.get('scores') and cached.get('players'):
            return cached
        else:
            print(f'  Cache is empty/corrupt, re-running localization...')

    x_clean, y_clean = clean_data
    dx, dy = defect_data

    sess = K.get_session()

    # ---- Subsample data ----
    if n_sample is not None and n_sample < len(dx):
        idx_d = np.random.choice(len(dx), n_sample, replace=False)
        dx, dy = dx[idx_d], dy[idx_d]
    n_clean_sample = min(SHAP_CLEAN_SAMPLE, len(x_clean))
    idx_c = np.random.choice(len(x_clean), n_clean_sample, replace=False)
    x_clean_sub = x_clean[idx_c]
    y_clean_sub = y_clean[idx_c]

    # ---- Build computation graph in TF1 ----
    with sess.graph.as_default():
        y_ph = tf.compat.v1.placeholder(tf.int32, shape=[None],
                                        name='nsrepair_shap_y')

        # Collect intermediate layer outputs (Conv2D + Dense only)
        layer_outputs = []
        layer_metas = []
        for layer in model.layers[1:]:  # skip Input
            ltype = type(layer).__name__.lower()
            if ('conv' not in ltype) and ('dense' not in ltype):
                continue
            if not hasattr(layer, 'output') or layer.output is None:
                continue
            output_shape = layer.output_shape
            layer_outputs.append(layer.output)
            layer_metas.append({
                'name': layer.name,
                'type': ltype,
                'num_neurons': output_shape[-1],
                'rank': len(output_shape),  # 4 for Conv2D, 2 for Dense
            })

        # Build loss and gradients
        loss = _build_defect_loss(model.output, y_ph, defect_type, target_label)
        grads = tf.gradients(loss, layer_outputs)

        # Second forward pass for clean data gradients
        # (reuse the same loss node — gradients are recomputed per feed_dict)
        # Actually, grad from clean and defect use the SAME gradient op but
        # different feed_dict values → we need to run twice
        # No — tf.gradients returns a symbolic op; running it with different
        # feed_dicts gives different numerical values.  So one grad op is enough.

    print(f'  GradientSHAP: {len(layer_metas)} layers, '
          f'{len(dx)} defect samples, {len(x_clean_sub)} clean samples')

    # ---- Run defect data ----
    feed_defect = {model.input: dx, y_ph: dy}
    acts_d, grads_d = sess.run([layer_outputs, grads], feed_dict=feed_defect)

    # ---- Run clean data (baseline) ----
    feed_clean = {model.input: x_clean_sub, y_ph: y_clean_sub}
    acts_c, grads_c = sess.run([layer_outputs, grads], feed_dict=feed_clean)

    # ---- Aggregate SHAP scores per neuron ----
    players = []
    scores = []

    for li, meta in enumerate(layer_metas):
        act_d = acts_d[li]          # [batch, ...]
        act_c = acts_c[li]          # [batch, ...]
        grad_d = grads_d[li]
        grad_c = grads_c[li]

        if grad_d is None or grad_c is None:
            print(f'  Warning: grad is None for layer {meta["name"]}, skipping')
            continue

        num_neurons = meta['num_neurons']

        if meta['rank'] == 4:   # Conv2D: [batch, H, W, C]
            # Mean gradient over batch + spatial, per channel
            for c in range(num_neurons):
                g_d = grad_d[..., c]   # [batch, H, W]
                g_c = grad_c[..., c]
                a_d = act_d[..., c]
                a_c = act_c[..., c]
                # 2-pt Integrated Gradients:
                # (0.5*(g_d+g_c), (a_d - a_c)) → mean over batch & spatial
                g_avg = 0.5 * (g_d + g_c)
                a_diff = a_d - a_c
                shap = np.abs(np.mean(g_avg * a_diff))
                scores.append(float(shap))
                players.append(f'conv_{meta["name"]}_{c}')

        else:  # meta['rank'] == 2, Dense: [batch, D]
            for d in range(num_neurons):
                g_d = grad_d[:, d]
                g_c = grad_c[:, d]
                a_d = act_d[:, d]
                a_c = act_c[:, d]
                g_avg = 0.5 * (g_d + g_c)
                a_diff = a_d - a_c
                shap = np.abs(np.mean(g_avg * a_diff))
                scores.append(float(shap))
                players.append(f'dense_{meta["name"]}_{d}')

    # ---- Sort by SHAP score descending ----
    order = np.argsort(scores)[::-1]
    players = [players[i] for i in order]
    scores = [scores[i] for i in order]

    # ---- Cache ----
    result = {'players': players, 'scores': scores}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f'  GradientSHAP localization cached to: {cache_path}')

    return result


# ============================================================
# Neuron kernel weight info (OWN kernel, NOT next layer)
# ============================================================

def _get_own_kernel_weight_info(model, repair_neurons):
    """Get the OWN kernel weight information for each selected neuron.

    Unlike CARE (which modifies the next layer's kernel columns to scale the
    neuron's contribution), NSrepair directly modifies the neuron's OWN kernel
    weights — the parameters that compute the neuron's activation.

    For a Conv2D layer with kernel [H, W, C_in, C_out]:
        neuron c → kernel[:, :, :, c]   (shape [H, W, C_in])

    For a Dense layer with kernel [D_in, D_out]:
        neuron d → kernel[:, d]         (shape [D_in,])

    Returns:
        list of (weight_idx, neuron_idx, kernel_shape)
    """
    info = []
    for layer_type, layer_name, neuron_idx in repair_neurons:
        # Find the layer's weight index
        layer_idx_in_model = None
        for i, layer in enumerate(model.layers):
            if layer.name == layer_name:
                layer_idx_in_model = i
                break
        if layer_idx_in_model is None:
            print(f'  Warning: layer {layer_name} not found, skipping')
            continue

        ltype = type(model.layers[layer_idx_in_model]).__name__.lower()

        # Compute weight index in model.get_weights()
        weight_idx = 0
        for j in range(layer_idx_in_model):
            weight_idx += len(model.layers[j].get_weights())

        kernel_shape = model.get_weights()[weight_idx].shape

        if 'conv' in ltype:
            # kernel shape: [H, W, C_in, C_out]
            out_dim = kernel_shape[-1]
        else:
            # kernel shape: [D_in, D_out]
            out_dim = kernel_shape[-1]

        if neuron_idx >= out_dim:
            print(f'  Warning: neuron_idx {neuron_idx} >= out_dim={out_dim} '
                  f'(layer={layer_name}), skipping')
            continue

        info.append({
            'layer_name': layer_name,
            'neuron_idx': neuron_idx,
            'weight_idx': weight_idx,
            'kernel_shape': kernel_shape,
            'layer_type': ltype,
        })

    # Pre-compute weight slicing for NSGA-III decision variables
    # Each neuron's kernel is a contiguous block in the decision vector
    kernel_shapes = []
    begin = 0
    for item in info:
        if 'conv' in item['layer_type']:
            # Conv2D kernel for neuron c: [H, W, C_in]
            neuron_kernel_shape = (item['kernel_shape'][0],
                                   item['kernel_shape'][1],
                                   item['kernel_shape'][2])
        else:
            # Dense kernel for neuron d: [D_in]
            neuron_kernel_shape = (item['kernel_shape'][0],)

        size = int(np.prod(neuron_kernel_shape))
        kernel_shapes.append((begin, begin + size, neuron_kernel_shape))
        begin += size

    return info, kernel_shapes


# ============================================================
# NSGA-III repair
# ============================================================

class _NSRepairProblem(Problem):
    """Multi-objective repair problem for NSGA-III.

    Objectives (both minimized):
        F[0] = defect_rate
        F[1] = 1 - clean_accuracy

    Decision variables: flattened kernel weights of selected neurons.
    """

    def __init__(self, model, neuron_info, kernel_shapes, base_weights,
                 x_clean, y_clean, x_defect, y_defect,
                 defect_type, target_label,lb,ub):
        n_var = kernel_shapes[-1][1]  # total flattened params
        n_obj = 2
        n_ieq_constr = 0

        xl = np.full(n_var, lb)
        xu = np.full(n_var, ub)

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr,
                         xl=xl, xu=xu)

        self.model = model
        self.neuron_info = neuron_info
        self.kernel_shapes = kernel_shapes
        self.base_weights = base_weights
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.x_defect = x_defect
        self.y_defect = y_defect
        self.defect_type = defect_type
        self.target_label = target_label

        self.eval_count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            weights = copy.deepcopy(self.base_weights)
            for info, (begin, end, shape) in zip(self.neuron_info,
                                                  self.kernel_shapes):
                w_idx = info['weight_idx']
                n_idx = info['neuron_idx']
                weights[w_idx][..., n_idx] = x[begin:end].reshape(shape)

            self.model.set_weights(weights)

            defect_rate = _compute_defect_rate(
                self.model, self.x_defect, self.y_defect,
                self.defect_type, self.target_label)
            clean_acc = value_accuracy(self.model, self.x_clean, self.y_clean)

            F.append([defect_rate, 1.0 - clean_acc])

            self.eval_count += 1

        # Restore base weights
        self.model.set_weights(copy.deepcopy(self.base_weights))
        out["F"] = np.array(F)

        gen = self.eval_count // X.shape[0]
        if gen == 1 or gen % 10 == 0:
            best_defect = np.min(out["F"][:, 0])
            best_acc = 1.0 - np.min(out["F"][:, 1])
            print(f'  Gen {gen:3d}: best defect={best_defect:.6f}, '
                  f'best acc={best_acc:.6f}')


class PatienceTermination(Termination):
    """Stop when feasible solutions persist for 'patience' consecutive gens.

    A solution is 'feasible' if its accuracy and defect rate meet pre-set
    patience thresholds (more relaxed than final selection thresholds).
    """

    def __init__(self, patience, max_gen, acc_threshold, defect_threshold):
        super().__init__()
        self.patience = patience
        self.max_gen = max_gen
        self.acc_threshold = acc_threshold
        self.defect_threshold = defect_threshold
        self.counter = 0
        self._terminated = False

    def update(self, algorithm):
        if getattr(algorithm, 'n_gen', 0) >= self.max_gen:
            self._terminated = True
            return

        pop = getattr(algorithm, 'pop', None)
        if pop is None or len(pop) == 0:
            return

        F = pop.get("F")
        if F is None:
            return

        has_feasible = False
        for row in F:
            defect = row[0]
            acc = 1.0 - row[1]
            if defect <= self.defect_threshold and acc >= self.acc_threshold:
                has_feasible = True
                break

        if has_feasible:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self._terminated = True

    def has_terminated(self, algorithm=None):
        return self._terminated

    def do_continue(self, algorithm=None):
        # pymoo 0.6.1.1 的 minimize() 循环不会主动调用 update()，
        # 只在 do_continue() / has_next() 检查时调用 termination。
        # 因此把 update 逻辑内嵌到 do_continue 中，确保每代都检查 patience。
        if algorithm is not None:
            self.update(algorithm)
        return not self._terminated


# ============================================================
# Pareto front selection
# ============================================================

def _select_pareto_solution(model, pareto_X, pareto_F, neuron_info,
                             kernel_shapes, base_weights,
                             eval_data, defect_data, defect_type, target_label):
    """Select the best model from the Pareto front using ONLY repair-data metrics.

    Uses res.F directly (computed on repair/eval data during NSGA-III) —
    no re-evaluation, no test data involved. This preserves fairness:
    the test set is held out entirely for final evaluation.

    Selection logic:
      1. Compute initial (pre-repair) performance on repair data as baseline.
      2. Filter Pareto solutions that meet defect-specific thresholds.
         Thresholds are relative to initial performance:
           - acc >= init_acc - SELECT_ACC_DROP
           - defect <= max(init_defect * SELECT_REL_DEFECT, hard_threshold)
      3. Among qualifying solutions, pick the one with lowest defect rate.
      4. If none qualifies, fallback to utopia-point-closest solution.

    Args:
        model: Keras model (weights will be restored on it)
        pareto_X: Pareto-optimal decision vectors from NSGA-III
        pareto_F: Pareto-optimal objective vectors [defect_rate, 1-acc]
        neuron_info, kernel_shapes: weight repair metadata
        base_weights: original model weights (pre-repair)
        eval_data: (eval_x_clean, eval_y_clean) — repair eval data
        defect_data: (x_defect, y_defect) — full repair defect data
        defect_type: 'backdoor' | 'adv' | 'fairness'
        target_label: target label for backdoor evaluation

    Returns:
        repaired_model, selection_info dict
    """
    eval_x_clean, eval_y_clean = eval_data

    hard_threshold_map = {
        'backdoor': SELECT_HARD_THRESHOLD_BACKDOOR,
        'adv': SELECT_HARD_THRESHOLD_ADV,
        'fairness': SELECT_HARD_THRESHOLD_FAIRNESS,
    }
    hard_threshold = hard_threshold_map[defect_type]

    # ---- Compute initial (pre-repair) performance from repair data ----
    model.set_weights(copy.deepcopy(base_weights))
    init_acc = value_accuracy(model, eval_x_clean, eval_y_clean)

    if defect_type == 'backdoor':
        init_defect = value_backdoor(model, defect_data[0],
                                     defect_data[1], target_label)
    elif defect_type == 'adv':
        init_defect = _value_adv_error(model, defect_data[0],
                                       defect_data[1])
    elif defect_type == 'fairness':
        _, _, init_defect = _value_fairness_gap(model, eval_x_clean,
                                                eval_y_clean)
    else:
        raise ValueError(f'Unknown defect type: {defect_type}')

    print(f'  Initial (repair data): acc={init_acc:.6f}, defect={init_defect:.6f}')

    # ---- Build thresholds relative to initial performance ----
    acc_min = init_acc - SELECT_ACC_DROP
    defect_max = max(init_defect * SELECT_REL_DEFECT, hard_threshold)
    print(f'  Selection thresholds: acc >= {acc_min:.4f}, '
          f'defect <= {defect_max:.4f}')

    # ---- Select from Pareto front using res.F ----
    candidates = []
    print(f'  Pareto front ({len(pareto_F)} solutions):')
    for i in range(len(pareto_F)):
        f_defect = pareto_F[i][0]
        f_acc = 1.0 - pareto_F[i][1]
        ok = f_acc >= acc_min and f_defect <= defect_max
        flag = 'OK' if ok else '--'
        print(f'    [{i:3d}] defect={f_defect:.6f}, acc={f_acc:.6f}  {flag}')
        if ok:
            candidates.append((i, pareto_X[i], f_acc, f_defect))

    if candidates:
        best = min(candidates, key=lambda t: t[3])
        print(f'  -> Selected: index={best[0]}, acc={best[2]:.4f}, '
              f'defect={best[3]:.4f} ({len(candidates)}/{len(pareto_F)} qualified)')
        model.set_weights(copy.deepcopy(base_weights))
        weights = model.get_weights()
        for info, (begin, end, shape) in zip(neuron_info, kernel_shapes):
            w_idx = info['weight_idx']
            n_idx = info['neuron_idx']
            weights[w_idx][..., n_idx] = best[1][begin:end].reshape(shape)
        model.set_weights(weights)
        return model, {'selection': 'threshold_met',
                       'repair_acc': best[2],
                       'repair_defect': best[3]}

    # ---- Fallback: closest to utopia point [0, 0] ----
    F_range = pareto_F.max(axis=0) - pareto_F.min(axis=0)
    F_range[F_range < 1e-8] = 1.0
    F_norm = (pareto_F - pareto_F.min(axis=0)) / F_range
    distances = np.sqrt((F_norm ** 2).sum(axis=1))
    fallback_idx = int(np.argmin(distances))

    best_f = pareto_F[fallback_idx]
    fallback_acc = 1.0 - best_f[1]
    fallback_defect = best_f[0]
    print(f'  -> Fallback (utopia-closest): index={fallback_idx}, '
          f'acc={fallback_acc:.4f}, defect={fallback_defect:.4f}')

    model.set_weights(copy.deepcopy(base_weights))
    weights = model.get_weights()
    for info, (begin, end, shape) in zip(neuron_info, kernel_shapes):
        w_idx = info['weight_idx']
        n_idx = info['neuron_idx']
        weights[w_idx][..., n_idx] = pareto_X[fallback_idx][begin:end].reshape(shape)
    model.set_weights(weights)
    return model, {'selection': 'fallback_utopia',
                   'repair_acc': fallback_acc,
                   'repair_defect': fallback_defect}


# ============================================================
# Main entry point
# ============================================================

def nsrepair_single(model, defect_data, clean_data, defect_type,
                     loc_cache_path, seed, model_name, epsilon, **kwargs):
    """NSRepair single-defect repair.

    Compatible with sequentialrepair.py SINGLE_DEFECT_METHODS interface:
        fn(model, defect_data, clean_data, defect_type,
           loc_cache_path, seed, model_name, epsilon, **kwargs)
        -> (repaired_model, stats)

    Flow:
      1. GradientSHAP localization → neuron SHAP scores
      2. Select top-k neurons
      3. NSGA-III multi-objective repair on selected neuron weights
      4. Pareto front evaluation + selection on test set

    Args:
        model: Keras model
        defect_data: (x, y) for the specific defect type
        clean_data: (x_clean, y_clean)
        defect_type: 'backdoor', 'adv', or 'fairness'
        loc_cache_path: path to main loc cache (used to derive NSrepair cache)
        seed: random seed
        model_name: model identifier
        epsilon: adversarial epsilon
        **kwargs: overrides (repair_ratio, shap_n_sample, nsga3_pop_size,
                  nsga3_n_gen, nsga3_patience, nsga3_eval_sample, test_data)

    Returns:
        (repaired_model, stats)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg = DATASET_CONFIG.get(model_name)
    target_label = cfg.get('target_label')

    x_clean, y_clean = clean_data
    dx, dy = defect_data
    # ---- Overrides ----
    repair_ratio = kwargs.get('repair_ratio', REPAIR_RATIO)
    shap_n_sample = kwargs.get('shap_n_sample', SHAP_N_SAMPLE)
    nsga3_pop_size = kwargs.get('nsga3_pop_size', NSGA3_POP_SIZE)
    nsga3_n_gen = kwargs.get('nsga3_n_gen', NSGA3_N_GEN)
    nsga3_patience = kwargs.get('nsga3_patience', NSGA3_PATIENCE)
    nsga3_eval_sample = kwargs.get('nsga3_eval_sample', NSGA3_EVAL_SAMPLE)
    # test_data = kwargs.get('test_data', None)

    # ---- Cache path (includes defect_order to avoid collisions when the
    # same defect is repaired in different orders) ----
    loc_dir = os.path.dirname(loc_cache_path)
    defect_order = kwargs.get('defect_order', [defect_type])
    defect_order_str = '_'.join(defect_order)
    ns_loc_cache = os.path.join(
        loc_dir, f'ns_loc_{defect_order_str}_{defect_type}.json')

    print('=' * 60)
    print(f'NSRepair: defect_type={defect_type}')
    print('=' * 60)

    # ---- Compute baseline accuracy (before repair) ----
    baseline_acc = value_accuracy(model, x_clean, y_clean)
    print(f'  Baseline clean accuracy (train): {baseline_acc:.6f}')

    # ---- 1. GradientSHAP localization ----
    print(f'  GradientSHAP: n_sample={shap_n_sample}')
    loc_result = compute_gradientshap_localization(
        model,
        defect_data=(dx, dy),
        clean_data=clean_data,
        defect_type=defect_type,
        target_label=target_label,
        n_sample=shap_n_sample,
        cache_path=ns_loc_cache,
    )

    scores = loc_result['scores']
    players = loc_result['players']

    if not scores or not players:
        raise ValueError(f'GradientSHAP produced no scores for '
                         f'defect_type={defect_type}')

    # ---- 2. Select top-k neurons ----
    repair_num = max(REPAIR_NUM_MAX, int(len(players) * repair_ratio))
    selected = list(zip(players, scores))[:repair_num]
    repair_neurons = []
    for name, _ in selected:
        parts = name.split('_')
        if len(parts) < 3:
            continue
        layer_type = parts[0].lower()
        layer_name = '_'.join(parts[1:-1])
        neuron_idx = int(parts[-1])
        repair_neurons.append((layer_type, layer_name, neuron_idx))

    print(f'\nTotal neurons: {len(players)}, repair_ratio={repair_ratio:.0%} '
          f'-> repair_num={repair_num}')
    print(f'Selected {len(repair_neurons)} neurons for repair')

    # ---- 3. Get neuron OWN kernel weight info ----
    neuron_info, kernel_shapes = _get_own_kernel_weight_info(
        model, repair_neurons)
    if not neuron_info:
        raise ValueError('No valid neurons found for repair')
    print(f'Valid neurons for weight modification: {len(neuron_info)}')
    n_var = kernel_shapes[-1][1]
    print(f'Total decision variables: {n_var}')

    # ---- 4. Prepare evaluation data (subsampled) ----
    n_clean = min(nsga3_eval_sample, len(x_clean))
    idx_clean = np.random.choice(len(x_clean), n_clean, replace=False)
    eval_x_clean = x_clean[idx_clean]
    eval_y_clean = y_clean[idx_clean]

    n_defect = min(nsga3_eval_sample, len(dx))
    idx_defect = np.random.choice(len(dx), n_defect, replace=False)
    eval_x_defect = dx[idx_defect]
    eval_y_defect = dy[idx_defect]

    # ---- 5. NSGA-III optimization ----
    base_weights = copy.deepcopy(model.get_weights())

    # Patience thresholds (relaxed)
    patience_defect_map = {
        'backdoor': NSGA3_PATIENCE_DEFECT_BACKDOOR,
        'adv': NSGA3_PATIENCE_DEFECT_ADV,
        'fairness': NSGA3_PATIENCE_DEFECT_FAIRNESS,
    }

    print(f'\nNSGA-III: pop_size={nsga3_pop_size}, n_gen={nsga3_n_gen}, '
          f'patience={nsga3_patience}, n_var={n_var}, '
          f'sample_size={nsga3_eval_sample}')

    # Extract initial weights of selected neurons for population init
    init_weights = []
    max_value = []
    for info in neuron_info:
        w_idx = info['weight_idx']
        n_idx = info['neuron_idx']
        potienal_weights =base_weights[w_idx]
        weight = potienal_weights[..., n_idx]
        max_value.append(np.max(abs(potienal_weights)))
        init_weights.append(weight.flatten())
    init_weights = np.concatenate(init_weights)
    max_value = np.max(max_value)
    init_population = np.tile(init_weights, (nsga3_pop_size, 1))
    init_population += np.random.uniform(-max_value/2, max_value/2, init_population.shape)

    ref_dirs = get_reference_directions('das-dennis', 2, n_partitions=12)

    problem = _NSRepairProblem(
        model=model,
        neuron_info=neuron_info,
        kernel_shapes=kernel_shapes,
        base_weights=base_weights,
        x_clean=eval_x_clean,
        y_clean=eval_y_clean,
        x_defect=eval_x_defect,
        y_defect=eval_y_defect,
        defect_type=defect_type,
        target_label=target_label,
        lb=-2*max_value,
        ub=2*max_value
    )



    algorithm = NSGA3(
        pop_size=nsga3_pop_size,
        ref_dirs=ref_dirs,
        sampling=init_population,
        crossover=UniformCrossover(prob=0.9),
        mutation=PolynomialMutation(prob=0.9),
        restart_form_best="True",
        restart=5,
    )

    termination = PatienceTermination(
        patience=nsga3_patience,
        max_gen=nsga3_n_gen,
        acc_threshold=NSGA3_PATIENCE_ACC,
        defect_threshold=patience_defect_map.get(
            defect_type, NSGA3_PATIENCE_DEFECT_BACKDOOR),
    )

    start_time = time.time()
    res = minimize(problem, algorithm, termination=termination,
                   seed=seed, verbose=False)
    nsga3_time = time.time() - start_time

    n_evals = problem.eval_count
    print(f'NSGA-III completed in {nsga3_time:.2f}s '
          f'(n_evals={n_evals}, n_gen={res.algorithm.n_gen})')

    # ---- 6. Pareto front selection ----
    pareto_X = res.X
    pareto_F = res.F

    if pareto_X is None or len(pareto_X) == 0:
        raise RuntimeError('NSGA-III returned no Pareto solutions')

    print(f'Pareto front size: {len(pareto_X)}')
    print(f'Pareto front F (defect, 1-acc):')
    for i, f in enumerate(pareto_F[:min(5, len(pareto_F))]):
        print(f'  [{i}] defect={f[0]:.6f}, acc={1.0-f[1]:.6f}')
    eval_data = eval_x_clean, eval_y_clean
    # Select best model from Pareto front — uses repair-data res.F only
    repaired_model, selection_info = _select_pareto_solution(
        model, pareto_X, pareto_F, neuron_info, kernel_shapes,
        base_weights, eval_data, (dx, dy), defect_type, target_label)

    # ---- 7. Build stats ----
    repair_neuron_names = [f'{lt}_{ln}_{ni}'
                           for lt, ln, ni in repair_neurons]

    stats = {
        'method': 'nsrepair',
        'defect_type': defect_type,
        'localization': 'gradientshap',
        'shap_n_sample': shap_n_sample,
        'repair_ratio': repair_ratio,
        'repair_num': repair_num,
        'repair_neurons': repair_neuron_names,
        'selected_shap_scores': [float(s) for _, s in selected],
        'nsga3_pop_size': nsga3_pop_size,
        'nsga3_n_gen': nsga3_n_gen,
        'nsga3_patience': nsga3_patience,
        'nsga3_eval_sample': nsga3_eval_sample,
        'n_var': n_var,
        'n_evals': n_evals,
        'n_generations': int(res.algorithm.n_gen),
        'time_seconds': float(nsga3_time),
        'pareto_size': int(len(pareto_X)),
        'pareto_selection': selection_info,
    }

    print(f'NSRepair completed in {nsga3_time:.2f}s')
    return repaired_model, stats