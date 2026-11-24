"""
CCBR: Counterfactual Causal Behavior Repair.

Integrated Gradients-based neuron localization + NSGA-III multi-objective
scale factor repair.

Localization: Integrated Gradients of sum(model output) w.r.t. each neuron's
activation, computed for both clean and defect data. Neuron score =
|mean(IG_defect) - mean(IG_clean)|. The baseline is alpha * x (non-zero),
matching the original CCBR paper.

Repair: NSGA-III optimizes scale factors for neuron outputs (same paradigm
as CARE). Scale factors are applied by modifying the NEXT layer's kernel
columns (or the current layer's kernel for last-layer neurons), which is
mathematically equivalent to scaling the neuron's activation.

Reference:
  "CCBR: Counterfactual Causal Behavior Repair"
  https://www.sciencedirect.com/science/article/abs/pii/S0957417424019493
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
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from repair_config import (
    value_accuracy
)

# Reuse nsrepair's evaluation, pareto selection, and patience helpers
from methods.nsrepair import (
    _compute_defect_rate,
    _value_adv_error,
    _value_fairness_gap,
    _select_pareto_solution,
    PatienceTermination,
)

# Reuse CARE's weight info helper (modify next layer's kernel)
from methods.care_repair import _get_neuron_weight_info


# ============================================================
# Constants
# ============================================================

# --- Integrated Gradients localization ---
IG_ALPHA = 0.5              # non-zero baseline: baseline = alpha * x
IG_STEPS = 5                # interpolation steps
IG_N_SAMPLE = 256           # max defect samples for IG computation
IG_CLEAN_SAMPLE = 256       # max clean samples for baseline

# --- Neuron selection ---
REPAIR_RATIO = 0.1         # fraction of total neurons to repair (1%)
REPAIR_NUM_MAX = 11         # hard cap on neurons to repair

# --- NSGA-III ---
NSGA3_POP_SIZE = 40         # population size
NSGA3_N_GEN = 100           # max generations
NSGA3_EVAL_SAMPLE = 10000   # max samples for fitness evaluation
NSGA3_PATIENCE = 10          # patience for early stopping
NSGA3_PATIENCE_ACC = 0.9    # accuracy threshold for patience check
NSGA3_PATIENCE_DEFECT_BACKDOOR = 0.1
NSGA3_PATIENCE_DEFECT_ADV = 0.5
NSGA3_PATIENCE_DEFECT_FAIRNESS = 0.1

# --- Pareto selection thresholds (same as nsrepair) ---
SELECT_ACC_DROP = 0.05
SELECT_REL_DEFECT = 0.5
SELECT_HARD_THRESHOLD_BACKDOOR = 0.10
SELECT_HARD_THRESHOLD_ADV = 0.61
SELECT_HARD_THRESHOLD_FAIRNESS = 0.10


# ============================================================
# Integrated Gradients localization
# ============================================================

def _compute_ig_for_layers(sess, model, x_data, layer_outputs, grads,
                           alpha, steps, batch_size=32):
    """Compute Integrated Gradients for all layers.

    For each layer l, computes IG of sum(model_output) w.r.t. layer l's output.
    IG(x) = (f(x) - f(baseline)) * (1/(steps+1)) * sum_gradients
    where baseline = alpha * x.

    Args:
        sess: TF1 session
        model: Keras model
        x_data: input data [N, ...]
        layer_outputs: list of layer output tensors
        grads: list of gradient tensors (same order as layer_outputs)
        alpha: baseline coefficient (baseline = alpha * x)
        steps: number of interpolation steps
        batch_size: batch size for processing

    Returns:
        list of arrays, one per layer. Each array is the mean IG over all
        samples, with shape matching the layer output (without batch dim).
    """
    n_samples = len(x_data)
    n_layers = len(layer_outputs)

    ig_accum = [None] * n_layers

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        x_batch = x_data[start:end]
        baseline = alpha * x_batch

        total_grads = [[] for _ in range(n_layers)]
        total_neurons = [[] for _ in range(n_layers)]

        for step_idx in range(steps + 1):
            interpolated = baseline + (float(step_idx) / steps) * (x_batch - baseline)
            feed_dict = {model.input: interpolated}
            results = sess.run(layer_outputs + grads, feed_dict=feed_dict)

            for li in range(n_layers):
                total_neurons[li].append(results[li])
                total_grads[li].append(results[n_layers + li])

        for li in range(n_layers):
            avg_grad = np.mean(np.array(total_grads[li]), axis=0)
            diff = total_neurons[li][-1] - total_neurons[li][0]
            ig = diff * avg_grad

            # Normalize (same as original CCBR)
            norm = np.sqrt(np.mean(np.square(ig))) + K.epsilon()
            ig = ig / norm

            # Mean over batch
            ig_batch = np.mean(ig, axis=0)

            if ig_accum[li] is None:
                ig_accum[li] = np.zeros_like(ig_batch)
            ig_accum[li] += ig_batch

    n_batches = max(1, int(np.ceil(n_samples / batch_size)))
    for li in range(n_layers):
        ig_accum[li] /= n_batches

    return ig_accum


def compute_ccbr_localization(model, defect_data, clean_data,
                              defect_type, target_label,
                              alpha=IG_ALPHA, steps=IG_STEPS,
                              n_sample=IG_N_SAMPLE,
                              cache_path=None, target_layer=-1):
    """CCBR-style Integrated Gradients difference localization.

    For each neuron in selected layers:
      1. Compute IG of sum(model_output) w.r.t. the neuron's activation
         for both clean and defect data.
      2. Neuron score = |mean(IG_defect) - mean(IG_clean)|.
      3. Sort by score descending.

    Args:
        model: Keras model
        defect_data: (x, y) for the specific defect type
        clean_data: (x_clean, y_clean)
        defect_type: 'backdoor', 'adv', or 'fairness'
        target_label: backdoor target label
        alpha: baseline coefficient
        steps: IG interpolation steps
        n_sample: max defect samples
        cache_path: if provided, cache results to JSON
        target_layer: which layers to analyze.
            -1 or 'last_dense': last HIDDEN Dense layer (before classification,
                matching original CARE/CCBR paper which splits at classification layer)
            'dense': all Dense layers only
            'all': all Conv + Dense layers

    Returns:
        dict with 'players' (neuron names) and 'scores' (IG diff values)
    """
    # ---- Check cache ----
    if cache_path and os.path.exists(cache_path):
        print(f'  Loading CCBR IG localization from cache: {cache_path}')
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
    n_clean_sample = min(IG_CLEAN_SAMPLE, len(x_clean))
    idx_c = np.random.choice(len(x_clean), n_clean_sample, replace=False)
    x_clean_sub = x_clean[idx_c]

    # ---- Build computation graph ----
    with sess.graph.as_default():
        # Collect all conv/dense layer metas first
        all_layer_metas = []
        for i, layer in enumerate(model.layers[1:], start=1):
            ltype = type(layer).__name__.lower()
            if 'conv' not in ltype and 'dense' not in ltype:
                continue
            if not hasattr(layer, 'output') or layer.output is None:
                continue
            all_layer_metas.append({
                'layer': layer,
                'name': layer.name,
                'type': ltype,
                'num_neurons': layer.output_shape[-1],
                'rank': len(layer.output_shape),
            })

        # Filter based on target_layer
        if target_layer == -1 or (isinstance(target_layer, str) and target_layer == 'last_dense'):
            # Last HIDDEN Dense layer (second-to-last Dense), matching original CARE/CCBR paper.
            # The original code splits the model at the classification layer and analyzes
            # the hidden Dense layer before it.  Repair modifies the classification layer's
            # in-weights (next-layer kernel).
            dense_metas = [m for m in all_layer_metas if m['type'] == 'dense']
            if len(dense_metas) >= 2:
                selected_metas = [dense_metas[-2]]  # last hidden Dense
            elif len(dense_metas) == 1:
                selected_metas = [dense_metas[-1]]  # fallback: only one Dense layer
            else:
                print('  Warning: no Dense layers found')
                selected_metas = []
        elif isinstance(target_layer, str) and target_layer == 'dense':
            selected_metas = [m for m in all_layer_metas if m['type'] == 'dense']
        elif isinstance(target_layer, str) and target_layer == 'all':
            selected_metas = all_layer_metas
        else:
            print(f'  Warning: unknown target_layer={target_layer}, using last hidden Dense layer')
            dense_metas = [m for m in all_layer_metas if m['type'] == 'dense']
            if len(dense_metas) >= 2:
                selected_metas = [dense_metas[-2]]
            elif len(dense_metas) == 1:
                selected_metas = [dense_metas[-1]]
            else:
                selected_metas = []

        if not selected_metas:
            print('  Warning: no layers selected, returning empty result')
            return {'players': [], 'scores': []}

        layer_outputs = []
        layer_metas = []
        for m in selected_metas:
            layer_outputs.append(m['layer'].output)
            layer_metas.append({
                'name': m['name'],
                'type': m['type'],
                'num_neurons': m['num_neurons'],
                'rank': m['rank'],
            })

        # Gradient of sum(output) w.r.t. each layer output
        grads = tf.gradients(tf.reduce_sum(model.output), layer_outputs)

    print(f'  CCBR IG localization: {len(layer_metas)} layers, '
          f'alpha={alpha}, steps={steps}, '
          f'{len(dx)} defect samples, {len(x_clean_sub)} clean samples')

    # ---- Compute IG for clean data ----
    ig_clean = _compute_ig_for_layers(
        sess, model, x_clean_sub, layer_outputs, grads, alpha, steps)

    # ---- Compute IG for defect data ----
    ig_defect = _compute_ig_for_layers(
        sess, model, dx, layer_outputs, grads, alpha, steps)

    # ---- Compute per-neuron scores ----
    players = []
    scores = []

    for li, meta in enumerate(layer_metas):
        ig_c = ig_clean[li]
        ig_d = ig_defect[li]

        if meta['rank'] == 4:  # Conv2D: [H, W, C]
            # Average over spatial dimensions to get per-channel IG
            ig_c_flat = np.mean(ig_c, axis=(0, 1))  # (C,)
            ig_d_flat = np.mean(ig_d, axis=(0, 1))  # (C,)
            for c in range(meta['num_neurons']):
                score = float(abs(ig_d_flat[c] - ig_c_flat[c]))
                scores.append(score)
                players.append(f'conv_{meta["name"]}_{c}')
        else:  # Dense: [D]
            for d in range(meta['num_neurons']):
                score = float(abs(ig_d[d] - ig_c[d]))
                scores.append(score)
                players.append(f'dense_{meta["name"]}_{d}')

    # ---- Sort by score descending ----
    order = np.argsort(scores)[::-1]
    players = [players[i] for i in order]
    scores = [scores[i] for i in order]

    # ---- Cache ----
    result = {'players': players, 'scores': scores}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f'  CCBR IG localization cached to: {cache_path}')

    return result


# ============================================================
# NSGA-III repair problem (scale factors, CARE-style)
# ============================================================

class _CCBRRepairProblem(Problem):
    """Multi-objective repair problem for CCBR.

    Decision variables: scale factors for selected neuron outputs.
    Applied by modifying the next layer's kernel (same as CARE).

    Optimized: only copies/modifies the specific layer kernels that are
    affected by the repair, instead of deepcopying the entire model weights
    for each evaluation.

    Objectives (both minimized):
        F[0] = defect_rate
        F[1] = 1 - clean_accuracy
    """

    def __init__(self, model, neuron_info, base_weights,
                 x_clean, y_clean, x_defect, y_defect,
                 defect_type, target_label):
        n_var = len(neuron_info)
        n_obj = 2
        n_ieq_constr = 0

        # Bounds from original CCBR: xl=-3, xu=2
        xl = np.full(n_var, -3.0)
        xu = np.full(n_var, 2.0)

        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr,
                         xl=xl, xu=xu)

        self.model = model
        self.neuron_info = neuron_info
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.x_defect = x_defect
        self.y_defect = y_defect
        self.defect_type = defect_type
        self.target_label = target_label

        self.eval_count = 0

        # ---- Build per-layer weight cache (only the layers we modify) ----
        self._layer_kernels = {}
        for info in neuron_info:
            if info.get('repair_type') == 'last_layer':
                target_idx = info['layer_idx']
            else:
                target_idx = info['next_layer_idx']

            if target_idx not in self._layer_kernels:
                layer = model.layers[target_idx]
                layer_w = layer.get_weights()
                self._layer_kernels[target_idx] = {
                    'layer': layer,
                    'kernel': layer_w[0].copy(),
                    'bias': layer_w[1].copy() if len(layer_w) > 1 else None,
                }

    def _apply_scales(self, scales):
        """Apply scale factors to layer kernels (only copies needed kernels)."""
        kernels = {}
        for idx, lw in self._layer_kernels.items():
            kernels[idx] = {
                'kernel': lw['kernel'].copy(),
                'bias': lw['bias'].copy() if lw['bias'] is not None else None,
            }

        for info, scale in zip(self.neuron_info, scales):
            # scale is the NSGA-III variable; actual scale = 1 + scale
            # (matching original CCBR: do_hidden = (1+X[i]) * _p_prediction)
            # NSGA-III bounds [-3, 2] → actual scale [-2, 3], center at 1 = no change
            actual_scale = 1.0 + scale
            if info.get('repair_type') == 'last_layer':
                target_idx = info['layer_idx']
                kernel = kernels[target_idx]['kernel']
                layer_type = info['last_layer_type']
                if 'conv' in layer_type:
                    kernel[..., :, info['neuron_idx']] *= actual_scale
                else:
                    kernel[:, info['neuron_idx']] *= actual_scale
                if info.get('bias_weight_idx') is not None:
                    bias = kernels[target_idx]['bias']
                    if bias is not None:
                        bias[info['neuron_idx']] *= actual_scale
            else:
                target_idx = info['next_layer_idx']
                kernel = kernels[target_idx]['kernel']
                next_layer_type = info['next_layer_type']
                if 'conv' in next_layer_type:
                    kernel[..., info['neuron_idx'], :] *= actual_scale
                else:
                    kernel[info['neuron_idx'], :] *= actual_scale

        for idx, kinfo in kernels.items():
            layer = self._layer_kernels[idx]['layer']
            if kinfo['bias'] is not None:
                layer.set_weights([kinfo['kernel'], kinfo['bias']])
            else:
                layer.set_weights([kinfo['kernel']])

    def _restore_weights(self):
        """Restore modified layers to their original weights."""
        for idx, lw in self._layer_kernels.items():
            if lw['bias'] is not None:
                lw['layer'].set_weights([lw['kernel'].copy(), lw['bias'].copy()])
            else:
                lw['layer'].set_weights([lw['kernel'].copy()])

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            self._apply_scales(x)

            defect_rate = _compute_defect_rate(
                self.model, self.x_defect, self.y_defect,
                self.defect_type, self.target_label)
            clean_acc = value_accuracy(self.model, self.x_clean, self.y_clean)

            F.append([defect_rate, 1.0 - clean_acc])
            self.eval_count += 1

        # Restore original layer weights
        self._restore_weights()
        out["F"] = np.array(F)

        gen = self.eval_count // X.shape[0]
        if gen == 1 or gen % 10 == 0:
            best_defect = np.min(out["F"][:, 0])
            best_acc = 1.0 - np.min(out["F"][:, 1])
            print(f'  Gen {gen:3d}: best defect={best_defect:.6f}, '
                  f'best acc={best_acc:.6f}')


# ============================================================
# Main entry point
# ============================================================

def ccbr_single(model, defect_data, clean_data, defect_type,
                loc_cache_path, seed, model_name, epsilon, **kwargs):
    """CCBR single-defect repair.

    Compatible with sequentialrepair.py SINGLE_DEFECT_METHODS interface:
        fn(model, defect_data, clean_data, defect_type,
           loc_cache_path, seed, model_name, epsilon, **kwargs)
        -> (repaired_model, stats)

    Flow:
      1. Integrated Gradients difference localization
      2. Select top-k neurons
      3. NSGA-III multi-objective repair on scale factors
      4. Pareto front selection (same as nsrepair)

    Args:
        model: Keras model
        defect_data: (x, y) for the specific defect type
        clean_data: (x_clean, y_clean)
        defect_type: 'backdoor', 'adv', or 'fairness'
        loc_cache_path: path to main loc cache (used to derive CCBR cache)
        seed: random seed
        model_name: model identifier
        epsilon: adversarial epsilon
        **kwargs: overrides (repair_ratio, ig_alpha, ig_steps, ig_n_sample,
                  nsga3_pop_size, nsga3_n_gen, nsga3_patience,
                  nsga3_eval_sample, k_sample, target_layer)

    Returns:
        (repaired_model, stats)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    target_label = None

    x_clean, y_clean = clean_data
    dx, dy = defect_data

    # ---- Overrides ----
    repair_ratio = kwargs.get('repair_ratio', REPAIR_RATIO)
    ig_alpha = kwargs.get('ig_alpha', IG_ALPHA)
    ig_steps = kwargs.get('ig_steps', IG_STEPS)
    ig_n_sample = kwargs.get('ig_n_sample', IG_N_SAMPLE)
    k_sample = kwargs.get('k_sample', IG_N_SAMPLE)  # alias for ig_n_sample
    nsga3_pop_size = kwargs.get('nsga3_pop_size', NSGA3_POP_SIZE)
    nsga3_n_gen = kwargs.get('nsga3_n_gen', NSGA3_N_GEN)
    nsga3_patience = kwargs.get('nsga3_patience', NSGA3_PATIENCE)
    nsga3_eval_sample = kwargs.get('nsga3_eval_sample', NSGA3_EVAL_SAMPLE)
    target_layer = kwargs.get('target_layer', -1)  # default: last hidden Dense

    # k_sample overrides ig_n_sample if provided
    if 'k_sample' in kwargs:
        ig_n_sample = k_sample

    # ---- Cache path (includes defect_order to avoid collisions when the
    # same defect is repaired in different orders) ----
    loc_dir = os.path.dirname(loc_cache_path)
    defect_order = kwargs.get('defect_order', [defect_type])
    defect_order_str = '_'.join(defect_order)
    ccbr_loc_cache = os.path.join(
        loc_dir, f'ccbr_loc_{defect_order_str}_{defect_type}.json')

    print('=' * 60)
    print(f'CCBR Repair: defect_type={defect_type}')
    print('=' * 60)

    # ---- Compute baseline accuracy ----
    baseline_acc = value_accuracy(model, x_clean, y_clean)
    print(f'  Baseline clean accuracy (train): {baseline_acc:.6f}')

    # ---- 1. CCBR localization (Integrated Gradients difference) ----
    loc_result = compute_ccbr_localization(
        model,
        defect_data=(dx, dy),
        clean_data=clean_data,
        defect_type=defect_type,
        target_label=target_label,
        alpha=ig_alpha,
        steps=ig_steps,
        n_sample=ig_n_sample,
        cache_path=ccbr_loc_cache,
        target_layer=target_layer,
    )

    scores = loc_result['scores']
    players = loc_result['players']

    if not scores or not players:
        raise ValueError(f'CCBR localization produced no scores for '
                         f'defect_type={defect_type}')

    # ---- 2. Select top-k neurons ----
    repair_num =  max(REPAIR_NUM_MAX, int(len(players) * repair_ratio))
    # Select more candidates to account for invalid neurons
    candidate_num = min(repair_num * 3, len(players))
    selected = list(zip(players, scores))[:candidate_num]

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
    print(f'Selected {len(repair_neurons)} candidate neurons for repair')

    # ---- 3. Get neuron weight info (CARE-style: modify next layer) ----
    neuron_info = _get_neuron_weight_info(model, repair_neurons)
    if not neuron_info:
        raise ValueError('No valid neurons found for repair')
    # Take only repair_num valid neurons
    neuron_info = neuron_info[:repair_num]
    print(f'Valid neurons for weight modification: {len(neuron_info)}')
    n_var = len(neuron_info)
    print(f'Total decision variables: {n_var}')

    # ---- 4. Prepare evaluation data (subsampled, same as nsrepair) ----
    n_clean = min(nsga3_eval_sample, len(x_clean))
    idx_clean = np.random.choice(len(x_clean), n_clean, replace=False)
    eval_x_clean = x_clean[idx_clean]
    eval_y_clean = y_clean[idx_clean]

    n_defect = min(nsga3_eval_sample, len(dx))
    idx_defect = np.random.choice(len(dx), n_defect, replace=False)
    eval_x_defect = dx[idx_defect]
    eval_y_defect = dy[idx_defect]

    # ---- 5. NSGA-III optimization (scale factors) ----
    base_weights = copy.deepcopy(model.get_weights())

    print(f'\nNSGA-III: pop_size={nsga3_pop_size}, n_gen={nsga3_n_gen}, '
          f'patience={nsga3_patience}, n_var={n_var}, '
          f'sample_size={nsga3_eval_sample}')

    problem = _CCBRRepairProblem(
        model=model,
        neuron_info=neuron_info,
        base_weights=base_weights,
        x_clean=eval_x_clean,
        y_clean=eval_y_clean,
        x_defect=eval_x_defect,
        y_defect=eval_y_defect,
        defect_type=defect_type,
        target_label=target_label,
    )

    ref_dirs = get_reference_directions('energy', n_dim=2,
                                        n_points=nsga3_pop_size)

    # NSGA-III parameters from nsrepair
    algorithm = NSGA3(
        pop_size=nsga3_pop_size,
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / n_var if n_var > 0 else 0.1, eta=20),
        eliminate_duplicates=True,
    )

    # Patience thresholds (relaxed, same as nsrepair)
    patience_defect_map = {
        'backdoor': NSGA3_PATIENCE_DEFECT_BACKDOOR,
        'adv': NSGA3_PATIENCE_DEFECT_ADV,
        'fairness': NSGA3_PATIENCE_DEFECT_FAIRNESS,
    }

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

    # ---- 6. Pareto front selection (same as nsrepair) ----
    pareto_X = res.X
    pareto_F = res.F

    if pareto_X is None or len(pareto_X) == 0:
        raise RuntimeError('NSGA-III returned no Pareto solutions')

    print(f'Pareto front size: {len(pareto_X)}')
    print(f'Pareto front F (defect, 1-acc):')
    for i, f in enumerate(pareto_F[:min(5, len(pareto_F))]):
        print(f'  [{i}] defect={f[0]:.6f}, acc={1.0 - f[1]:.6f}')

    # Build kernel_shapes for _select_pareto_solution
    # (CCBR uses scale factors, not kernel weights, but _select_pareto_solution
    #  needs this info to apply the repair)
    # For CCBR, the "weights" are the next layer's kernel columns that get
    # multiplied by scale factors. We need to build a compatible interface.
    # We'll use a wrapper that applies scale factors instead of kernel weights.

    # Build a fake neuron_info/kernel_shapes for _select_pareto_solution
    # Actually, _select_pareto_solution applies weights by modifying
    # weights[w_idx][..., n_idx] = x[begin:end].reshape(shape).
    # For CCBR, we need to modify weights by multiplying by scale factor.
    # Simplest approach: apply the repair directly using our own logic.

    eval_data = eval_x_clean, eval_y_clean

    # Compute initial performance on repair data (model already restored by _restore_weights)
    init_acc = value_accuracy(model, eval_x_clean, eval_y_clean)

    if defect_type == 'backdoor':
        init_defect = value_backdoor(model, dx, dy, target_label)
    elif defect_type == 'adv':
        init_defect = _value_adv_error(model, dx, dy)
    elif defect_type == 'fairness':
        _, _, init_defect = _value_fairness_gap(model, eval_x_clean,
                                                eval_y_clean)
    else:
        raise ValueError(f'Unknown defect type: {defect_type}')

    hard_threshold_map = {
        'backdoor': SELECT_HARD_THRESHOLD_BACKDOOR,
        'adv': SELECT_HARD_THRESHOLD_ADV,
        'fairness': SELECT_HARD_THRESHOLD_FAIRNESS,
    }
    hard_threshold = hard_threshold_map[defect_type]

    print(f'  Initial (repair data): acc={init_acc:.6f}, defect={init_defect:.6f}')

    acc_min = NSGA3_PATIENCE_ACC
    defect_max = max(init_defect * SELECT_REL_DEFECT, hard_threshold)
    print(f'  Selection thresholds: acc >= {acc_min:.4f}, '
          f'defect <= {defect_max:.4f}')
    
    # Select from Pareto front
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
              f'defect={best[3]:.4f} '
              f'({len(candidates)}/{len(pareto_F)} qualified)')
        selected_x = best[1]
        selection_info = {'selection': 'threshold_met',
                          'repair_acc': best[2],
                          'repair_defect': best[3]}
    else:
        # Fallback: closest to utopia point [0, 0]
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
        selected_x = pareto_X[fallback_idx]
        selection_info = {'selection': 'fallback_utopia',
                          'repair_acc': fallback_acc,
                          'repair_defect': fallback_defect}

    # ---- Apply best repair to model: save coefficients (NOT modify model weights) ----
    # CCBR returns scale coefficients (r coefficients) for the last layer's
    # neuron outputs. The coefficients are applied during evaluation by
    # scaling the neuron outputs of the repair layer.
    # Restore original weights (model stays unmodified)
    problem._restore_weights()

    # Build coefficient data for evaluation
    coefficient_data = {
        'method': 'ccbr',
        'defect_type': defect_type,
        'repair_neurons': [],
        'scale_factors': selected_x.tolist(),
    }
    for info, scale in zip(neuron_info, selected_x):
        entry = {
            'layer_name': info['layer_name'],
            'layer_idx': int(info['layer_idx']),
            'neuron_idx': int(info['neuron_idx']),
            'scale': float(scale),
            'actual_scale': float(1.0 + scale),
            'repair_type': info.get('repair_type', 'next_layer'),
        }
        if info.get('repair_type') == 'last_layer':
            entry['last_layer_type'] = info.get('last_layer_type', '')
        else:
            entry['next_layer_idx'] = int(info.get('next_layer_idx', -1))
            entry['next_layer_type'] = info.get('next_layer_type', '')
        coefficient_data['repair_neurons'].append(entry)

    # Save coefficients to JSON in the result directory (same level as weights)
    loc_dir = os.path.dirname(loc_cache_path)
    result_dir = os.path.join(loc_dir, 'defect_repair')
    os.makedirs(result_dir, exist_ok=True)
    coeff_path = os.path.join(result_dir, 'coefficients_ccbr.json')
    with open(coeff_path, 'w', encoding='utf-8') as f:
        json.dump(coefficient_data, f, indent=2)
    print(f'  CCBR coefficients saved to: {coeff_path}')

    # ---- 7. Build stats ----
    repair_neuron_names = [f'{lt}_{ln}_{ni}'
                           for lt, ln, ni in repair_neurons[:len(neuron_info)]]

    stats = {
        'method': 'ccbr',
        'defect_type': defect_type,
        'target_layer': target_layer,
        'localization': 'integrated_gradients',
        'ig_alpha': ig_alpha,
        'ig_steps': ig_steps,
        'ig_n_sample': ig_n_sample,
        'repair_ratio': repair_ratio,
        'repair_num': repair_num,
        'repair_neurons': repair_neuron_names,
        'selected_ig_scores': [float(s) for _, s in selected[:len(neuron_info)]],
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
        'best_scale_factors': selected_x.tolist(),
        'coefficient_data': coefficient_data,
        'coefficient_path': coeff_path,
    }

    print(f'CCBR completed in {nsga3_time:.2f}s')
    return model, stats