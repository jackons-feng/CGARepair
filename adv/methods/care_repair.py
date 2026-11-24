"""
CARE-style repair method for single-defect repair.

Compatible with sequentialrepair.py's SINGLE_DEFECT_METHODS registry.

This is a COMPLETE localization + repair pipeline:
1. Localization: CARE's own ACE (Average Causal Effect) causal analysis.
   This is INDEPENDENT from loc_neuron_output.py — it runs CARE's original
   ACE algorithm (split-model + neuron output intervention at multiple
   values across the activation range) to produce its own localization
   results. Results are cached to care_loc_{defect_type}.json for reuse.

2. Repair: pymoo PSO (matching original CARE's pyswarms parameters:
   w=0.8, c1=0.41, c2=0.41, pop_size=20) optimizes scale factors for
   neuron outputs. Scale factors are applied by modifying the NEXT layer's
   weight kernel columns, which is mathematically equivalent to scaling
   the neuron's activation output.

Reference:
  casual_imp.py    - original CARE using pyswarms
  potienalways.py  - CARE adapted with pymoo/NSGA3
"""

import os
import json
import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from repair_config import (
    sess_run,
    value_accuracy
)
from loc_neuron_output import (
    split_model_at_layer,
    get_layer_weight_indices,
)

# ============================================================
# Constants (from original CARE / casual_imp.py)
# ============================================================
PSO_POP_SIZE = 30
PSO_N_GEN = 100
PSO_FTOL = 1e-5
PSO_PERIOD = 20
PSO_W = 0.8             # inertia weight (matching original CARE / pyswarms)
PSO_C1 = 0.5           # cognitive parameter (matching original CARE / pyswarms)
PSO_C2 = 0.5           # social parameter (matching original CARE / pyswarms)
PSO_BOUND_LO = -1.0
PSO_BOUND_HI = 1.0
ALPHA = 0.7            # importance of accuracy in PSO fitness
REPAIR_RATIO = 0.1      # fraction of total neurons to repair (1%)
CARE_STEPSIZE = 2       # number of intervention values for ACE
CARE_K_SAMPLE = 512    # max defect samples for ACE (None = use all); reduce to speed up
PSO_SAMPLE_SIZE = 10000   # max samples for PSO fitness evaluation


# ============================================================
# Evaluation helpers
# ============================================================

def value_adv_error(model, images_adv, labels_true):
    """Adversarial error rate: fraction of misclassified adversarial samples."""
    pred_adv = np.argmax(sess_run(model, model.output, images_adv), axis=-1)
    return float(np.sum(pred_adv != labels_true) / len(labels_true))


def _compute_defect_rate(logits, labels, defect_type, target_label=None):
    """Compute defect-specific metric from logits and labels."""
    pred = np.argmax(logits, axis=-1)
    if defect_type == 'backdoor' and target_label!=None:
        return float(
            (np.sum(pred == target_label)
             - np.sum((labels == target_label) & (pred == target_label)))
            / max(1, len(labels)))
    elif defect_type == 'adv':
        return float(np.sum(pred != labels) / max(1, len(labels)))
    elif defect_type == 'fairness':
        correct = pred == labels
        f_mask = labels == 0
        m_mask = labels == 1
        f_acc = np.sum(correct[f_mask]) / max(1, np.sum(f_mask))
        m_acc = np.sum(correct[m_mask]) / max(1, np.sum(m_mask))
        return float(abs(f_acc - m_acc))
    else:
        raise ValueError(f'Unknown defect type: {defect_type}')


# ============================================================
# CARE localization (ACE causal analysis)
# ============================================================

def compute_care_localization(model, defect_data, clean_data, defect_type,
                               target_label, stepsize=CARE_STEPSIZE,
                               cache_path=None, target_layer=-1):
    """
    CARE-style ACE (Average Causal Effect) localization.

    For each neuron in selected layers, computes the ACE by:
    1. Splitting the model at the neuron's layer
    2. Computing the activation range across defect samples
    3. For each intervention value in the range, measuring the defect metric
    4. ACE = average |delta_defect_metric| across intervention values

    The data preprocessing (k_sample, adv filtering) is handled externally
    before this function is called.

    Args:
        model: Keras model
        defect_data: (x, y) tuple for the specific defect type
        clean_data: (x_clean, y_clean) for accuracy baseline
        defect_type: 'backdoor', 'adv', or 'fairness'
        target_label: backdoor target label
        stepsize: number of intervention values in the activation range
        cache_path: if provided, cache results to this JSON file
        target_layer: which layers to analyze.
            -1 or 'last_dense': last HIDDEN Dense layer (before classification,
                matching original CARE paper which splits at classification layer)
            'dense': all Dense layers only
            'all': all Conv + Dense layers

    Returns:
        dict with 'players' (list of neuron names) and 'scores' (list of ACE scores)
    """
    # ---- Check cache ----
    if cache_path and os.path.exists(cache_path):
        print(f'  Loading CARE localization from cache: {cache_path}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        if cached.get('scores') and cached.get('players'):
            return cached
        else:
            print(f'  Cache is empty/corrupt, re-running localization...')

    x_clean, y_clean = clean_data
    dx, dy = defect_data

    # ---- Pre-process: filter adv data to successful attacks only ----
    if defect_type == 'adv':
        pred_adv = np.argmax(sess_run(model, model.output, dx), axis=-1)
        adv_success_mask = (pred_adv != dy)
        n_before = len(dx)
        dx, dy = dx[adv_success_mask], dy[adv_success_mask]
        print(f'  CARE loc: filtered adv data: {n_before} -> {len(dx)} '
              f'(kept {np.sum(adv_success_mask)} successful attacks)')

    # ---- Get layer metadata ----
    conv_meta, dense_meta = get_layer_weight_indices(model)
    
    # Filter layers based on target_layer setting
    if target_layer == -1 or (isinstance(target_layer, str) and target_layer == 'last_dense'):
        # Last HIDDEN Dense layer (second-to-last Dense), matching original CARE paper.
        # The original CARE splits the model at the classification layer and analyzes
        # the hidden Dense layer before it.  Repair modifies the classification layer's
        # in-weights (next-layer kernel).
        if len(dense_meta) >= 2:
            selected_meta = [dense_meta[-2]]  # last hidden Dense (before classification)
        elif len(dense_meta) == 1:
            # Only one Dense layer (classification layer, e.g. ResNet18 'fc').
            # Fall back to last Conv layer as the intervention target, since
            # CARE should intervene on intermediate representations, not logits.
            print('  Warning: only 1 Dense layer (classification), '
                  'falling back to last Conv layer')
            if len(conv_meta) > 0:
                selected_meta = [conv_meta[-1]]
            else:
                selected_meta = [dense_meta[-1]]
        else:
            # No Dense layers found (e.g. CIFAR-10 VGG uses only Conv layers).
            # Fall back to the last Conv layer (closest to output).
            print('  Warning: no Dense layers found, falling back to last Conv layer')
            if len(conv_meta) > 0:
                selected_meta = [conv_meta[-1]]
            else:
                selected_meta = []
    elif isinstance(target_layer, str) and target_layer == 'dense':
        # All Dense layers only
        selected_meta = dense_meta
    elif isinstance(target_layer, str) and target_layer == 'all':
        # All Conv + Dense layers
        selected_meta = conv_meta + dense_meta
    else:
        # Unknown setting, default to last hidden Dense
        print(f'  Warning: unknown target_layer={target_layer}, using last hidden Dense layer')
        if len(dense_meta) >= 2:
            selected_meta = [dense_meta[-2]]
        elif len(dense_meta) == 1:
            selected_meta = [dense_meta[-1]]
        else:
            selected_meta = []
    
    all_meta = selected_meta

    print(f'  CARE localization: target_layer={target_layer} -> {len(all_meta)} layers selected, '
          f'stepsize={stepsize}, defect_type={defect_type}')

    all_players = []
    all_scores = []
    sess = K.get_session()

    for meta in all_meta:
        layer_index = meta['index']
        layer_name = meta['name']
        layer_type = meta['type']
        n_neurons = meta.get('num_neurons', 0)
        if n_neurons == 0:
            continue

        print(f'    Layer {layer_index} ({layer_name}): {n_neurons} neurons')

        # Split model (capture skip_indices for ResNet skip connections)
        model1, model2, skip_indices = split_model_at_layer(model, layer_index)

        # Pre-compute model1 outputs (target layer output)
        layer_out_clean = sess.run(model1.output, {model1.input: x_clean})
        layer_out_defect = sess.run(model1.output, {model1.input: dx})

        # Pre-compute skip connection source outputs (for ResNet)
        skip_outs_clean = {}
        skip_outs_defect = {}
        for skip_idx in skip_indices:
            skip_model = tf.keras.Model(
                inputs=model.input,
                outputs=model.layers[skip_idx].output)
            skip_outs_clean[skip_idx] = sess.run(
                skip_model.output, {skip_model.input: x_clean})
            skip_outs_defect[skip_idx] = sess.run(
                skip_model.output, {skip_model.input: dx})

        # Helper: build feed dict for model2 (handles multi-input skip connections)
        def _make_model2_feed(layer_out, skip_outs):
            if isinstance(model2.input, list):
                feed = {model2.input[0]: layer_out}
                for i, skip_idx in enumerate(skip_indices):
                    feed[model2.input[i + 1]] = skip_outs[skip_idx]
                return feed
            else:
                return {model2.input: layer_out}

        # Original model2 outputs (baseline, no intervention)
        orig_clean_out = sess.run(model2.output,
                                  _make_model2_feed(layer_out_clean, skip_outs_clean))
        orig_clean_acc = float(np.mean(
            np.argmax(orig_clean_out, axis=-1) == y_clean))

        orig_defect_out = sess.run(model2.output,
                                   _make_model2_feed(layer_out_defect, skip_outs_defect))
        orig_defect_rate = _compute_defect_rate(
            orig_defect_out, dy, defect_type, target_label)

        # For each neuron, compute ACE
        for neuron_idx in range(n_neurons):
            # Activation range for this neuron
            activations = layer_out_defect[..., neuron_idx].ravel()
            act_min = float(np.min(activations))
            act_max = float(np.max(activations))

            if act_min == act_max:
                # Constant activation: no causal effect possible
                all_players.append(f'{layer_type}_{layer_name}_{neuron_idx}')
                all_scores.append(0.0)
                continue

            # Compute ACE across intervention values
            delta_rates = []
            for h_val in np.linspace(act_min, act_max, stepsize):
                modified = layer_out_defect.copy()
                modified[..., neuron_idx] = h_val
                z_out = sess.run(model2.output,
                                 _make_model2_feed(modified, skip_outs_defect))
                z_rate = _compute_defect_rate(
                    z_out, dy, defect_type, target_label)
                delta_rates.append(abs(z_rate - orig_defect_rate))

            ace_score = float(np.mean(delta_rates))
            all_players.append(f'{layer_type}_{layer_name}_{neuron_idx}')
            all_scores.append(ace_score)

    # Sort by score descending
    items = sorted(zip(all_players, all_scores),
                   key=lambda x: x[1], reverse=True)
    all_players = [p for p, _ in items]
    all_scores = [s for _, s in items]

    result = {
        'players': all_players,
        'scores': all_scores,
        'defect_type': defect_type,
        'stepsize': stepsize,
    }

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f'  CARE localization saved to: {cache_path}')

    return result


# ============================================================
# Neuron weight info helpers
# ============================================================

def _get_neuron_weight_info(model, repair_neurons):
    """For each (layer_type, layer_name, neuron_idx), find the weight index
    to modify.

    For neurons in non-last conv/dense layers:
        Scaling neuron i's output at layer L by s is equivalent to modifying
        the NEXT layer's kernel along the input channel/row i:
        - Conv2D next layer: K_{L+1}[:, :, i, :] *= s
        - Dense next layer:    W_{L+1}[i, :] *= s

    For neurons in the LAST conv/dense layer (no next layer exists):
        Scaling logit/output channel i by s is equivalent to modifying the
        CURRENT layer's kernel (output channel/column) AND bias:
        - Conv2D last layer: K_L[:, :, :, i] *= s  AND  b_L[i] *= s
        - Dense last layer:  W_L[:, i] *= s       AND  b_L[i] *= s
    """
    info = []
    for layer_type, layer_name, neuron_idx in repair_neurons:
        layer_idx = None
        for i, layer in enumerate(model.layers):
            if layer.name == layer_name:
                layer_idx = i
                break
        if layer_idx is None:
            continue

        # Find the next conv/dense layer
        next_layer_idx = None
        for j in range(layer_idx + 1, len(model.layers)):
            ltype = type(model.layers[j]).__name__.lower()
            if 'conv' in ltype or 'dense' in ltype:
                next_layer_idx = j
                break

        if next_layer_idx is not None:
            # ---- Standard case: modify NEXT layer's kernel ----
            weight_idx = 0
            for j in range(next_layer_idx):
                weight_idx += len(model.layers[j].get_weights())

            kernel_shape = model.get_weights()[weight_idx].shape
            next_layer_type = type(model.layers[next_layer_idx]).__name__.lower()

            if 'conv' in next_layer_type:
                in_dim = kernel_shape[-2]
            else:
                in_dim = kernel_shape[0]

            if neuron_idx >= in_dim:
                print(f'  Warning: neuron_idx {neuron_idx} >= in_dim={in_dim} '
                      f'(layer={layer_name}, next={model.layers[next_layer_idx].name}), '
                      f'skipping')
                continue

            info.append({
                'layer_name': layer_name,
                'layer_idx': layer_idx,
                'neuron_idx': neuron_idx,
                'repair_type': 'next_layer',
                'kernel_weight_idx': weight_idx,
                'next_layer_type': next_layer_type,
                'next_layer_idx': next_layer_idx,
            })
        else:
            # ---- Last layer: modify CURRENT layer's kernel + bias ----
            current_layer = model.layers[layer_idx]
            current_layer_type = type(current_layer).__name__.lower()
            weights = current_layer.get_weights()
            if len(weights) == 0:
                continue

            kernel_shape = weights[0].shape
            out_dim = kernel_shape[-1]  # output channels / output dim

            if neuron_idx >= out_dim:
                print(f'  Warning: neuron_idx {neuron_idx} >= out_dim={out_dim} '
                      f'(last layer={layer_name}), skipping')
                continue

            # Compute weight index in model.get_weights()
            weight_idx = 0
            for j in range(layer_idx):
                weight_idx += len(model.layers[j].get_weights())
            has_bias = len(weights) > 1

            info.append({
                'layer_name': layer_name,
                'layer_idx': layer_idx,
                'neuron_idx': neuron_idx,
                'repair_type': 'last_layer',
                'kernel_weight_idx': weight_idx,
                'bias_weight_idx': weight_idx + 1 if has_bias else None,
                'last_layer_type': current_layer_type,
            })
    return info


# ============================================================
# PSO repair via pymoo
# ============================================================

class _RepairProblem(Problem):
    """pymoo Problem wrapper for CARE PSO fitness evaluation.

    Optimized: only copies/modifies the specific layer kernels that are
    affected by the repair, instead of deepcopying the entire model weights
    for each evaluation. This mirrors the original CARE approach of using
    split models (model1/model2) where only the small model2 needs to be
    modified, but adapted to work with full-model weight modification.
    """

    def __init__(self, n_var, xl, xu, model, neuron_info, base_weights,
                 pso_x_clean, pso_y_clean, pso_x_defect, pso_y_defect,
                 defect_type, target_label, alpha):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.model = model
        self.neuron_info = neuron_info
        self.pso_x_clean = pso_x_clean
        self.pso_y_clean = pso_y_clean
        self.pso_x_defect = pso_x_defect
        self.pso_y_defect = pso_y_defect
        self.defect_type = defect_type
        self.target_label = target_label
        self.alpha = alpha

        self.eval_count = 0

        # ---- Build per-layer weight cache (only the layers we modify) ----
        # Instead of deepcopying ALL model weights each evaluation,
        # we only copy the specific kernel(s) that need modification.
        # This is equivalent to the original CARE's split-model approach
        # where only model2 (classification layer) weights are manipulated.
        self._layer_kernels = {}  # layer_idx -> {'layer': layer, 'kernel': np.array, 'bias': np.array|None}
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
        # Copy only the original kernels (not entire model weights)
        kernels = {}
        for idx, lw in self._layer_kernels.items():
            kernels[idx] = {
                'kernel': lw['kernel'].copy(),
                'bias': lw['bias'].copy() if lw['bias'] is not None else None,
            }

        for info, scale in zip(self.neuron_info, scales):
            # scale is the PSO variable; actual scale = 1 + scale
            # (matching original CARE: do_hidden = (1+X[i]) * _p_prediction)
            # PSO bounds [-1, 1] → actual scale [0, 2], center at 1 = no change
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

        # Set only the modified layers' weights (not entire model)
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

            clean_acc = value_accuracy(self.model, self.pso_x_clean,
                                       self.pso_y_clean)

            if self.defect_type == 'backdoor':
                defect_rate = value_backdoor(self.model, self.pso_x_defect,
                                             self.pso_y_defect,
                                             self.target_label)
            elif self.defect_type == 'adv':
                defect_rate = value_adv_error(self.model, self.pso_x_defect,
                                              self.pso_y_defect)
            elif self.defect_type == 'fairness':
                _, _, defect_rate = value_fairness_gap(
                    self.model, self.pso_x_defect, self.pso_y_defect)
            else:
                raise ValueError(f'Unknown defect type: {self.defect_type}')

            fitness = ((1.0 - self.alpha) * defect_rate
                       + self.alpha * (1.0 - clean_acc))
            F.append(fitness)
            self.eval_count += 1
        # Restore original layer weights after evaluation
        self._restore_weights()
        out["F"] = np.array(F)
        gen = self.eval_count // X.shape[0]
        if gen == 1 or gen % 10 == 0:
            best_defect = np.min(out["F"])
            print(f'  Gen {gen:3d}: best fitness={best_defect:.6f}, ')

# ============================================================
# Main CARE repair entry point
# ============================================================

def care_repair_single(model, defect_data, clean_data, defect_type,
                       loc_cache_path, seed, model_name, epsilon, **kwargs):
    """
    CARE-style single-defect repair.

    Compatible with sequentialrepair.py SINGLE_DEFECT_METHODS interface:
        fn(model, defect_data, clean_data, defect_type,
           loc_cache_path, seed, model_name, epsilon, **kwargs)
        -> (repaired_model, stats)

    Internal params (fixed, from original CARE):
        repair_ratio=0.1 (10% of all neurons), alpha=0.3,
        pso_pop_size=20, pso_n_gen=100,
        pso_bounds=(-1,1), pso_w=0.8, pso_c1=0.41, pso_c2=0.41,
        stepsize=2 (ACE intervention points), k_sample=None (all data)

    IMPORTANT: CARE uses its OWN independent localization (ACE causal
    analysis). The loc_cache_path parameter is used ONLY to derive the
    directory for CARE's own cache file (care_loc_{defect_type}.json).
    CARE does NOT load or depend on loc_neuron_output.py's localization data.

    Args:
        model: pre-loaded Keras model
        defect_data: (x, y) tuple for the specific defect type
        clean_data: (x_clean, y_clean) for accuracy preservation
        defect_type: 'backdoor', 'adv', or 'fairness'
        loc_cache_path: path to the main localization cache file
            (only used to derive the directory for CARE's own cache)
        seed: random seed
        model_name: model identifier
        epsilon: adversarial epsilon
        **kwargs: optional overrides (repair_ratio, alpha, pso_n_gen,
                  pso_sample_size, stepsize, k_sample, target_layer)

    Returns:
        repaired_model: model with CARE repair applied (weights modified)
        stats: dict with repair statistics
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    target_label = None

    x_clean, y_clean = clean_data
    dx, dy = defect_data

    repair_ratio = kwargs.get('repair_ratio', REPAIR_RATIO)
    alpha = kwargs.get('alpha', ALPHA)
    pso_n_gen = kwargs.get('pso_n_gen', PSO_N_GEN)
    pso_sample_size = kwargs.get('pso_sample_size', PSO_SAMPLE_SIZE)
    stepsize = kwargs.get('stepsize', CARE_STEPSIZE)
    k_sample = kwargs.get('k_sample', CARE_K_SAMPLE)
    target_layer = kwargs.get('target_layer', 'dense')  # default: last hidden Dense

    # ---- Derive CARE-specific cache path (includes defect_order to avoid
    # collisions when the same defect is repaired in different orders) ----
    loc_dir = os.path.dirname(loc_cache_path)
    defect_order = kwargs.get('defect_order', [defect_type])
    defect_order_str = '_'.join(defect_order)
    care_loc_cache = os.path.join(loc_dir, f'care_loc_{defect_order_str}_{defect_type}.json')

    print('=' * 60)
    print(f'CARE Repair: defect_type={defect_type}')
    print('=' * 60)

    # ---- 1. CARE localization (ACE) ----
    # Apply k_sample to the defect data for localization
    if k_sample is not None and k_sample < len(dx):
        idx = np.random.choice(len(dx), k_sample, replace=False)
        loc_dx, loc_dy = dx[idx], dy[idx]
    else:
        loc_dx, loc_dy = dx, dy

    print(f'  CARE localization data: {len(loc_dx)} defect samples, '
          f'{len(x_clean)} clean samples')

    loc_result = compute_care_localization(
        model,
        defect_data=(loc_dx, loc_dy),
        clean_data=clean_data,
        defect_type=defect_type,
        target_label=target_label,
        stepsize=stepsize,
        cache_path=care_loc_cache,
        target_layer=target_layer,
    )

    scores = loc_result['scores']
    players = loc_result['players']

    if not scores or not players:
        raise ValueError(f'CARE localization produced no scores for '
                         f'defect_type={defect_type}')

    # ---- 2. Select top-k neurons (k = repair_ratio * total_neurons) ----
    repair_num = max(30, int(len(players) * repair_ratio))
    selected = list(zip(players, scores))[:repair_num]
    repair_neurons = []
    for name, score in selected:
        parts = name.split('_')
        if len(parts) < 3:
            continue
        layer_type = parts[0].lower()
        layer_name = '_'.join(parts[1:-1])
        neuron_idx = int(parts[-1])
        repair_neurons.append((layer_type, layer_name, neuron_idx))

    print(f'\nTotal neurons: {len(players)}, repair_ratio={repair_ratio:.0%} '
          f'-> repair_num={repair_num}')
    print(f'Selected {len(repair_neurons)} neurons for repair:')
    # for (name, score), _ in zip(selected, repair_neurons):
    #     print(f'  {name}  (ACE={score:.6f})')

    # ---- 3. Build neuron weight info ----
    neuron_info = _get_neuron_weight_info(model, repair_neurons)
    if not neuron_info:
        raise ValueError('No valid neurons found for repair (all neurons '
                         'may be in the last layer with no next layer)')
    print(f'Valid neurons for weight modification: {len(neuron_info)}')

    # ---- 4. Prepare PSO evaluation data (subsampled) ----
    n_clean = min(pso_sample_size, len(x_clean))
    idx_clean = np.random.choice(len(x_clean), n_clean, replace=False)
    pso_x_clean = x_clean[idx_clean]
    pso_y_clean = y_clean[idx_clean]

    n_defect = min(pso_sample_size, len(dx))
    idx_defect = np.random.choice(len(dx), n_defect, replace=False)
    pso_x_defect = dx[idx_defect]
    pso_y_defect = dy[idx_defect]

    # ---- 5. Run pymoo PSO ----
    base_weights = copy.deepcopy(model.get_weights())
    n_var = len(neuron_info)

    print(f'\nPSO: pop_size={PSO_POP_SIZE}, n_gen={pso_n_gen}, '
          f'alpha={alpha}, n_var={n_var}, sample_size={pso_sample_size}')

    problem = _RepairProblem(
        n_var=n_var,
        xl=np.full(n_var, PSO_BOUND_LO),
        xu=np.full(n_var, PSO_BOUND_HI),
        model=model,
        neuron_info=neuron_info,
        base_weights=base_weights,
        pso_x_clean=pso_x_clean,
        pso_y_clean=pso_y_clean,
        pso_x_defect=pso_x_defect,
        pso_y_defect=pso_y_defect,
        defect_type=defect_type,
        target_label=target_label,
        alpha=alpha,
    )

    algorithm = PSO(
        pop_size=PSO_POP_SIZE,
        w=PSO_W,
        c1=PSO_C1,
        c2=PSO_C2,
        adaptive=False,  # disable adaptive w/c1/c2 to match original CARE's pyswarms params
    )

    termination = DefaultSingleObjectiveTermination(
        ftol=PSO_FTOL,
        period=PSO_PERIOD,
        n_max_gen=pso_n_gen,
    )

    start_time = time.time()
    res = minimize(problem, algorithm, termination=termination,
                   seed=seed, verbose=False)
    pso_time = time.time() - start_time

    best_pos = res.X
    best_cost = float(res.F[0])
    n_evals = int(res.algorithm.evaluator.n_eval)

    print(f'PSO completed in {pso_time:.2f}s (n_evals={n_evals})')
    print(f'Best fitness: {best_cost:.6f}')
    print(f'Best scale factors: {np.array2string(best_pos, precision=6)}')

    # ---- 6. Save coefficients (NOT modify model weights) ----
    # CARE returns scale coefficients (r coefficients) for the last layer's
    # neuron outputs. The coefficients are applied during evaluation by
    # scaling the neuron outputs of the repair layer.
    # Restore original weights (model stays unmodified)
    problem._restore_weights()

    # Build coefficient data for evaluation
    coefficient_data = {
        'method': 'care',
        'defect_type': defect_type,
        'repair_neurons': [],
        'scale_factors': best_pos.tolist(),
    }
    for info, scale in zip(neuron_info, best_pos):
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
    coeff_path = os.path.join(result_dir, 'coefficients_care.json')
    with open(coeff_path, 'w', encoding='utf-8') as f:
        json.dump(coefficient_data, f, indent=2)
    print(f'  CARE coefficients saved to: {coeff_path}')

    # ---- 7. Build stats ----
    repair_neuron_names = [f'{lt}_{ln}_{ni}'
                           for lt, ln, ni in repair_neurons]
    stats = {
        'method': 'care',
        'defect_type': defect_type,
        'target_layer': target_layer,
        'repair_ratio': repair_ratio,
        'repair_num': repair_num,
        'alpha': alpha,
        'pso_pop_size': PSO_POP_SIZE,
        'pso_n_gen': pso_n_gen,
        'pso_w': PSO_W,
        'pso_c1': PSO_C1,
        'pso_c2': PSO_C2,
        'pso_ftol': PSO_FTOL,
        'pso_period': PSO_PERIOD,
        'pso_time': float(pso_time),
        'pso_n_evals': n_evals,
        'best_fitness': best_cost,
        'best_scale_factors': best_pos.tolist(),
        'repair_neurons': repair_neuron_names,
        'neuron_weight_info': [
            {'layer_name': ni['layer_name'],
             'neuron_idx': int(ni['neuron_idx']),
             'repair_type': ni.get('repair_type', 'next_layer'),
             'kernel_weight_idx': int(ni['kernel_weight_idx']),
             **({'next_layer_idx': int(ni['next_layer_idx'])}
                if ni.get('repair_type') != 'last_layer' else {}),
             **({'bias_weight_idx': int(ni['bias_weight_idx'])}
                if ni.get('bias_weight_idx') is not None else {}),
             }
            for ni in neuron_info
        ],
        'pso_sample_size': pso_sample_size,
        'care_stepsize': stepsize,
        'care_loc_cache': care_loc_cache,
        'coefficient_data': coefficient_data,
        'coefficient_path': coeff_path,
    }

    return model, stats