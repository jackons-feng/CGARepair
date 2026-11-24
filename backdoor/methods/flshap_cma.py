"""
FLShap-CMA: Fault Localization with Hierarchical Shapley Values
           + CMA-ES Weight Optimization.

Single-defect repair method compatible with sequentialrepair.py.

Localization: For each Dense layer, compute the gradient of the
defect-related loss w.r.t. each neuron's output (pre-activation).
The importance is |mean(grad_defect) - mean(grad_clean)|, capturing
the Shapley-value concept of marginal contribution to defect behavior.

Repair: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
optimizes weight deltas for the incoming connections of selected
neurons. The objective combines activation distance, accuracy
preservation, and defect reduction.

Reference: Original implementation in RepairBackdoor.py / FL_Mul_VGG13.py
"""
import os
import json
import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K


import cma
HAS_CMA = True


from repair_config import (
    DATASET_CONFIG,
    get_layer_kernel_indices,
    sess_run,
    BATCH_SIZE,
)
from loc_neuron_output import split_model_at_layer

# ============================================================
# Constants
# ============================================================

# --- Localization ---
SHAP_N_SAMPLE = 256           # number of samples for gradient computation
REPAIR_RATIO = 0.01           # fraction of neurons to repair
REPAIR_NUM_MAX = 10            # max neurons to repair

# --- CMA-ES ---
CMA_SIGMA0 = 2.0              # initial step size
CMA_MAX_FEVALS = 1000         # max function evaluations
CMA_DELTA_BOUND = 6.0         # bound for weight deltas [-bound, bound]

# --- Objective weights ---
W_ACT = 1.0                   # activation distance weight
W_ACC = 20.0                  # accuracy penalty weight
W_REG = 0.001                 # regularization weight

# --- Thresholds (same spirit as original RepairBackdoor.py) ---
CA_TARGET_ABS = 0.88          # clean accuracy should stay above this
ASR_TARGET_ABS = 0.15         # defect rate should drop below this


# ============================================================
# Localization: Hierarchical Shapley-style importance
# ============================================================

def compute_flshap_localization(model, defect_data, clean_data,
                                defect_type, target_label,
                                n_sample=SHAP_N_SAMPLE,
                                cache_path=None):
    """
    Compute neuron importance using hierarchical Shapley-style analysis.

    For each selected Dense layer, computes the gradient of the defect-related
    loss w.r.t. each neuron's pre-activation output. The importance score is
    the absolute difference between the mean gradient on clean vs. defect data.

    This approximates the Shapley value concept: how much each neuron's
    marginal contribution differs between clean and defect behavior.

    Args:
        model: Keras model
        defect_data: (x, y) for defect samples
        clean_data: (x, y) for clean samples
        defect_type: 'backdoor', 'adv', or 'fairness'
        target_label: target label for backdoor
        n_sample: max samples to use
        cache_path: optional path to cache results

    Returns:
        list of (layer_name, layer_idx, neuron_idx, importance_score)
    """
    if cache_path and os.path.exists(cache_path):
        print(f'  Loading FLShap localization from cache: {cache_path}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f'  FLShap localization: n_sample={n_sample}')

    x_clean, y_clean = clean_data
    x_defect, y_defect = defect_data

    n_sample = min(n_sample, len(x_clean), len(x_defect))
    clean_idx = np.random.choice(len(x_clean), n_sample, replace=False)
    defect_idx = np.random.choice(len(x_defect), n_sample, replace=False)

    x_clean_sub = x_clean[clean_idx]
    y_clean_sub = y_clean[clean_idx]
    x_defect_sub = x_defect[defect_idx]
    y_defect_sub = y_defect[defect_idx]

    num_classes = model.output_shape[-1]

    # All Dense and Conv2D layers
    target_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            target_layers.append((i, layer))

    if not target_layers:
        print('  Warning: No target layers found, skipping localization')
        return []

    all_scores = []

    for layer_idx, layer in target_layers:
        layer_name = layer.name
        is_conv = isinstance(layer, tf.keras.layers.Conv2D)
        if is_conv:
            out_dim = layer.filters
        else:
            out_dim = layer.units

        # Use split_model_at_layer to handle ResNet skip connections
        _, tail_model, skip_indices = split_model_at_layer(model, layer_idx)

        # Pre-compute skip connection outputs for clean data
        skip_outs_clean = {}
        if skip_indices:
            for skip_idx in skip_indices:
                skip_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.layers[skip_idx].output)
                skip_outs_clean[skip_idx] = K.get_session().run(
                    skip_model.output, {skip_model.input: x_clean_sub})

        # Pre-compute skip connection outputs for defect data
        skip_outs_defect = {}
        if skip_indices:
            for skip_idx in skip_indices:
                skip_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.layers[skip_idx].output)
                skip_outs_defect[skip_idx] = K.get_session().run(
                    skip_model.output, {skip_model.input: x_defect_sub})

        # Original sub-model output tensor (for gradient computation)
        sub_output = model.layers[layer_idx].output

        # Compute gradients on clean data
        grads_clean = _compute_neuron_gradients(
            model, tail_model, skip_indices, skip_outs_clean,
            layer_idx, sub_output,
            x_clean_sub, y_clean_sub, defect_type, target_label,
            num_classes, out_dim, is_conv=is_conv)

        # Compute gradients on defect data
        grads_defect = _compute_neuron_gradients(
            model, tail_model, skip_indices, skip_outs_defect,
            layer_idx, sub_output,
            x_defect_sub, y_defect_sub, defect_type, target_label,
            num_classes, out_dim, is_conv=is_conv)

        if grads_clean is None or grads_defect is None:
            continue

        # Importance = |mean(grad_defect) - mean(grad_clean)|
        mean_clean = np.mean(np.abs(grads_clean), axis=0)   # (out_dim,)
        mean_defect = np.mean(np.abs(grads_defect), axis=0)  # (out_dim,)
        importance = np.abs(mean_defect - mean_clean)        # (out_dim,)

        for neuron_idx in range(out_dim):
            all_scores.append({
                'layer_name': layer_name,
                'layer_idx': layer_idx,
                'neuron_idx': int(neuron_idx),
                'importance': float(importance[neuron_idx]),
            })

    # Sort by importance descending
    all_scores.sort(key=lambda x: x['importance'], reverse=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(all_scores, f, indent=2)
        print(f'  FLShap localization saved to: {cache_path}')

    print(f'  Total neurons scored: {len(all_scores)}')
    # if all_scores:
    #     print(f'  Top-5 importance: {[f"{s["importance"]:.4f}" for s in all_scores[:5]]}')

    return all_scores


def _compute_neuron_gradients(model, tail_model, skip_indices, skip_outs,
                              layer_idx, layer_output,
                              x_data, y_data, defect_type, target_label,
                              num_classes, out_dim, is_conv=False):
    """
    Compute gradients of the defect-related loss w.r.t. each neuron's
    pre-activation output.

    Uses TF1 graph-mode (tf.gradients + sess.run) since eager execution
    is disabled in this project.

    For Conv2D layers (is_conv=True), gradients are averaged over spatial
    dimensions to produce per-channel (per-filter) importance scores.

    Args:
        skip_indices: list of layer indices for skip connections (ResNet)
        skip_outs: dict mapping skip_idx -> pre-computed numpy output arrays

    Returns:
        numpy array of shape (n_samples, out_dim) or None on error.
    """
    try:
        sess = K.get_session()

        # Build a sub-model that outputs intermediate layer output
        sub_model = tf.keras.Model(
            inputs=model.input,
            outputs=layer_output)

        # Get intermediate outputs
        intermediate = sess_run(sub_model, sub_model.output, x_data)

        if is_conv:
            # Conv2D output: shape (batch, h, w, filters)
            # Keep spatial dims for gradient computation
            if intermediate.ndim != 4:
                print(f'  Warning: expected Conv2D output to be 4D, '
                      f'got {intermediate.ndim}D for layer {layer_idx}')
                return None
        else:
            # Dense output: may need to flatten if from Conv
            if intermediate.ndim > 2:
                intermediate = intermediate.reshape(intermediate.shape[0], -1)
                if intermediate.shape[1] != out_dim:
                    return None

        # ---- TF1 graph-mode gradient computation ----
        # Use placeholders + tf.gradients instead of eager tf.GradientTape
        with sess.graph.as_default():
            intermediate_ph = tf.compat.v1.placeholder(
                tf.float32, shape=[None] + list(intermediate.shape[1:]))
            y_ph = tf.compat.v1.placeholder(tf.int32, shape=[None])

            if skip_indices:
                skip_phs = []
                for skip_idx in skip_indices:
                    skip_shape = [None] + list(skip_outs[skip_idx].shape[1:])
                    skip_phs.append(tf.compat.v1.placeholder(
                        tf.float32, shape=skip_shape))
                tail_out = tail_model([intermediate_ph] + skip_phs)
            else:
                tail_out = tail_model(intermediate_ph)

            loss = _defect_loss_tf(tail_out, y_ph, defect_type,
                                   target_label, num_classes)
            grads = tf.gradients(loss, intermediate_ph)[0]

        # Compute gradients in batches via sess.run()
        all_grads = []
        batch_size = 32
        for start in range(0, len(intermediate), batch_size):
            end = min(start + batch_size, len(intermediate))
            feed_dict = {
                intermediate_ph: intermediate[start:end],
                y_ph: y_data[start:end],
                K.learning_phase(): 0,
            }
            if skip_indices:
                for i, skip_idx in enumerate(skip_indices):
                    feed_dict[skip_phs[i]] = skip_outs[skip_idx][start:end]
            grad_vals = sess.run(grads, feed_dict)
            all_grads.append(grad_vals)

        grads_array = np.concatenate(all_grads, axis=0)

        if is_conv:
            # grads_array shape: (n_samples, h, w, filters)
            # Average over spatial dims to get per-channel importance
            grads_array = np.mean(grads_array, axis=(1, 2))  # (n_samples, filters)

        return grads_array

    except Exception as e:
        print(f'  Warning: gradient computation failed for layer {layer_idx}: {e}')
        return None


def _defect_loss_tf(logits, y_true, defect_type, target_label, num_classes):
    """
    Compute defect-specific loss for gradient computation.
    Mirrors the logic in repair_config.py.
    """
    y_true = tf.convert_to_tensor(y_true, dtype=tf.int32)
    y_oh = tf.one_hot(y_true, num_classes)

    if defect_type == 'backdoor':
        # Backdoor: maximize probability of target_label
        target_logit = logits[:, target_label]
        return tf.reduce_mean(target_logit)

    elif defect_type == 'adv':
        # Adversarial: cross-entropy loss on wrong predictions
        ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_oh, logits=logits)
        return -tf.reduce_mean(ce)  # negative = maximize error

    elif defect_type == 'fairness':
        # Fairness: squared difference of female vs male loss
        per_sample_ce = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_oh, logits=logits)
        female_mask = tf.cast(tf.equal(y_true, 0), tf.float32)
        male_mask = tf.cast(tf.equal(y_true, 1), tf.float32)
        loss_female = tf.reduce_sum(per_sample_ce * female_mask) / \
            (tf.reduce_sum(female_mask) + 1e-8)
        loss_male = tf.reduce_sum(per_sample_ce * male_mask) / \
            (tf.reduce_sum(male_mask) + 1e-8)
        return tf.square(loss_female - loss_male)

    else:
        raise ValueError(f'Unknown defect type: {defect_type}')


# ============================================================
# Helper: get neuron weight info (reused from CARE)
# ============================================================

def _get_neuron_weight_info(model, neuron_scores):
    """
    Convert neuron scores to weight modification info.

    For each neuron, identifies the weight index in model.get_weights()
    that corresponds to the INCOMING connections of the current layer's kernel.
    This matches the original RepairAdversarialAttack.py / RepairBackdoor.py:
    modify the current layer's kernel[:, neuron_idx] for Dense, or
    kernel[0, 0, 0, neuron_idx] for Conv2D (single scalar at fixed position).

    Args:
        model: Keras model
        neuron_scores: list of dicts with layer_name, layer_idx, neuron_idx

    Returns:
        list of dicts with weight modification info, filtered to valid neurons only
    """
    info_list = []

    for ns in neuron_scores:
        layer_name = ns['layer_name']
        layer_idx = ns['layer_idx']
        neuron_idx = ns['neuron_idx']

        current_layer = model.layers[layer_idx]
        weights = current_layer.get_weights()
        if len(weights) == 0:
            continue

        kernel_shape = weights[0].shape

        if isinstance(current_layer, tf.keras.layers.Conv2D):
            # kernel shape: (h, w, in_ch, out_ch)
            if neuron_idx >= kernel_shape[-1]:
                continue
            in_features = 1  # single scalar at fixed position (0,0,0,neuron_idx)
        elif isinstance(current_layer, tf.keras.layers.Dense):
            # kernel shape: (in_features, out_features)
            if neuron_idx >= kernel_shape[-1]:
                continue
            in_features = kernel_shape[0]  # incoming dimension
        else:
            continue

        # Compute weight index in model.get_weights()
        weight_idx = 0
        for j in range(layer_idx):
            weight_idx += len(model.layers[j].get_weights())

        has_bias = len(weights) > 1
        info_list.append({
            'layer_name': layer_name,
            'layer_idx': layer_idx,
            'neuron_idx': neuron_idx,
            'repair_type': 'in_weights',
            'kernel_weight_idx': weight_idx,
            'bias_weight_idx': weight_idx + 1 if has_bias else None,
            'in_features': in_features,
        })

    return info_list


# ============================================================
# CMA-ES Evaluator
# ============================================================

class _FLShapEvaluator:
    """
    CMA-ES fitness evaluator for weight delta optimization.

    Combines:
    - Activation distance: MSE between current and golden activations
    - Accuracy penalty: how far clean acc is below target
    - Defect penalty: how far defect rate is above target
    - Regularization: L2 norm of delta vector
    """

    def __init__(self, model, neuron_info, base_weights,
                 x_clean, y_clean, x_defect, y_defect,
                 defect_type, target_label):
        self.model = model
        self.neuron_info = neuron_info
        self.base_weights = copy.deepcopy(base_weights)
        self.x_clean = x_clean
        self.y_clean = y_clean
        self.x_defect = x_defect
        self.y_defect = y_defect
        self.defect_type = defect_type
        self.target_label = target_label

        # Pre-compute golden activations (only for backdoor repair)
        if self.defect_type == 'backdoor':
            self.golden_activations = self._compute_golden_activations()
        else:
            self.golden_activations = {}

        # Baseline metrics
        self.acc_clean_raw = self._compute_clean_acc(self.model)
        self.defect_rate_raw = self._compute_defect_rate(self.model)

        # Initial normalization factors
        initial_delta = np.zeros(self.total_dim)
        self.initial_act_dist = self._activation_distance()
        self.initial_acc_pen = self._accuracy_penalty(
            self.acc_clean_raw, self.defect_rate_raw)
        if self.initial_act_dist < 1e-8:
            self.initial_act_dist = 1.0
        if self.initial_acc_pen < 1e-8:
            self.initial_acc_pen = 1.0

        self.step_count = 0
        self.best_score = float('inf')
        self.best_solution = None
        self.best_acc_clean = 0.0
        self.best_defect_rate = 0.0

        print(f'  Baseline: clean_acc={self.acc_clean_raw:.4f}, '
              f'defect_rate={self.defect_rate_raw:.4f}')
        print(f'  Initial act_dist={self.initial_act_dist:.4f}, '
              f'acc_pen={self.initial_acc_pen:.4f}')

    @property
    def total_dim(self):
        return sum(info['in_features'] for info in self.neuron_info)

    def _compute_golden_activations(self):
        """Compute golden activations for the target neurons on clean data."""
        golden = {}
        sub_models = {}

        for info in self.neuron_info:
            layer_idx = info['layer_idx']
            if layer_idx not in sub_models:
                sub_models[layer_idx] = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[layer_idx].output)

        for layer_idx, sub_model in sub_models.items():
            acts = sess_run(sub_model, sub_model.output, self.x_clean)
            golden[layer_idx] = acts

        return golden

    def _activation_distance(self):
        """Compute MSE between golden and current activations on defect data.
        Only active for backdoor defect type (matches RepairBackdoor.py).
        Assumes self.model already has the modified weights applied."""
        if self.defect_type != 'backdoor':
            return 0.0

        total_dist = 0.0
        count = 0

        for info in self.neuron_info:
            layer_idx = info['layer_idx']
            if layer_idx not in self.golden_activations:
                continue

            sub_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[layer_idx].output)

            current_acts = sess_run(sub_model, sub_model.output, self.x_defect)
            golden_acts = self.golden_activations[layer_idx]

            # Match shapes
            if current_acts.shape != golden_acts.shape:
                min_shape = [min(c, g) for c, g in
                             zip(current_acts.shape, golden_acts.shape)]
                current_acts = current_acts[tuple([slice(0, s) for s in min_shape])]
                golden_acts = golden_acts[tuple([slice(0, s) for s in min_shape])]

            total_dist += np.mean((current_acts - golden_acts) ** 2)
            count += 1

        return total_dist / max(1, count)

    def _apply_delta(self, delta_vector):
        """Apply delta vector to base weights; return modified weights list.
        Modifies INCOMING weights of the current layer's kernel:
        - Dense: kernel[:, neuron_idx] += delta
        - Conv2D: kernel[0, 0, 0, neuron_idx] += delta[0]"""
        weights = copy.deepcopy(self.base_weights)
        ptr = 0

        for info in self.neuron_info:
            dim = info['in_features']
            delta = delta_vector[ptr:ptr + dim]
            ptr += dim

            w_idx = info['kernel_weight_idx']
            neuron_idx = info['neuron_idx']

            current_layer = self.model.layers[info['layer_idx']]
            if isinstance(current_layer, tf.keras.layers.Conv2D):
                # Conv: kernel shape (h, w, in_ch, out_ch)
                # Fix single scalar at (h=0, w=0, in_ch=0, out_ch=neuron_idx)
                if neuron_idx < weights[w_idx].shape[-1]:
                    weights[w_idx][0, 0, 0, neuron_idx] += delta[0]
            else:
                # Dense: kernel shape (in_features, out_features)
                # Modify kernel[:, neuron_idx] (incoming weights)
                weights[w_idx][:, neuron_idx] += delta

            # Also modify bias if present
            if info.get('bias_weight_idx') is not None:
                weights[info['bias_weight_idx']][neuron_idx] += np.mean(delta)

        return weights

    def _compute_clean_acc(self, model_instance):
        """Compute clean accuracy."""
        output = sess_run(model_instance, model_instance.output, self.x_clean)
        pred = np.argmax(output, axis=-1)
        return np.mean(pred == self.y_clean)

    def _compute_defect_rate(self, model_instance):
        """Compute defect rate (ASR for backdoor, error rate for adv, gap for fairness)."""
        if self.defect_type == 'backdoor':
            output = sess_run(model_instance, model_instance.output,
                              self.x_defect)
            pred = np.argmax(output, axis=-1)
            return np.mean(pred == self.target_label)

        elif self.defect_type == 'adv':
            output = sess_run(model_instance, model_instance.output,
                              self.x_defect)
            pred = np.argmax(output, axis=-1)
            return 1.0 - np.mean(pred == self.y_defect)

        elif self.defect_type == 'fairness':
            output = sess_run(model_instance, model_instance.output,
                              self.x_defect)
            pred = np.argmax(output, axis=-1)
            corr = pred == self.y_defect
            female_mask = (self.y_defect == 0)
            male_mask = (self.y_defect == 1)
            female_acc = np.sum(female_mask & corr) / max(1, np.sum(female_mask))
            male_acc = np.sum(male_mask & corr) / max(1, np.sum(male_mask))
            return float(abs(female_acc - male_acc))

        return 0.0

    def _accuracy_penalty(self, acc_clean, defect_rate):
        """Compute accuracy penalty: how far from targets."""
        ca_penalty = max(0, CA_TARGET_ABS - acc_clean) ** 2
        asr_penalty = max(0, defect_rate - ASR_TARGET_ABS) ** 2
        return 40.0 * ca_penalty + 5.0 * asr_penalty

    def evaluate(self, solution_vector):
        """CMA-ES objective function. Lower is better."""
        self.step_count += 1

        # Apply solution to model (avoid clone_model to save GPU memory)
        weights = self._apply_delta(solution_vector)
        self.model.set_weights(weights)

        act_dist = self._activation_distance()
        norm_act_dist = act_dist / (self.initial_act_dist + 1e-8)

        acc_clean = self._compute_clean_acc(self.model)
        defect_rate = self._compute_defect_rate(self.model)

        acc_pen = self._accuracy_penalty(acc_clean, defect_rate)
        norm_acc_pen = acc_pen / (self.initial_acc_pen + 1e-8)

        reg_pen = np.mean(solution_vector ** 2)

        score = (W_ACT * norm_act_dist +
                 W_ACC * norm_acc_pen +
                 W_REG * reg_pen)

        # if self.step_count % 100 == 1 or score < self.best_score:
        if self.step_count % 100 == 1 or self.step_count == 1:
            print(f'  CMA-ES step {self.step_count:4d}: score={score:.4f} '
                  f'| act={norm_act_dist:.3f} acc_pen={norm_acc_pen:.3f} '
                  f'reg={reg_pen:.3f} | CA={acc_clean:.4f} DR={defect_rate:.4f}')

        if score < self.best_score:
            self.best_score = score
            self.best_solution = solution_vector.copy()
            self.best_acc_clean = acc_clean
            self.best_defect_rate = defect_rate

        # Restore original weights
        self.model.set_weights(self.base_weights)

        return score


# ============================================================
# Main entry point
# ============================================================

def flshap_cma_single(model, defect_data, clean_data, defect_type,
                      loc_cache_path, seed, model_name, epsilon, **kwargs):
    """
    FLShap-CMA single-defect repair.

    Localization: hierarchical Shapley-style gradient analysis.
    Repair: CMA-ES optimization of weight deltas for selected neurons.

    Args:
        model: Keras model
        defect_data: (x, y) for defect samples
        clean_data: (x, y) for clean samples
        defect_type: 'backdoor', 'adv', or 'fairness'
        loc_cache_path: base path for localization cache
        seed: random seed
        model_name: model identifier
        epsilon: adversarial epsilon (unused, kept for interface compatibility)
        **kwargs: overrides (repair_ratio, shap_n_sample, cma_max_fevals,
                  cma_sigma0, cma_delta_bound, k_sample)

    Returns:
        repaired_model, stats
    """
    if not HAS_CMA:
        raise ImportError(
            "CMA-ES package 'cma' is required for FLShap-CMA. "
            "Install with: pip install cma")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    cfg = DATASET_CONFIG.get('lfw', DATASET_CONFIG.get(model_name, {}))
    target_label = cfg.get('target_label', 1)

    # ---- Resolve kwargs ----
    repair_ratio = kwargs.get('repair_ratio', REPAIR_RATIO)
    repair_num_max = kwargs.get('repair_num_max', REPAIR_NUM_MAX)
    shap_n_sample = kwargs.get('shap_n_sample', SHAP_N_SAMPLE)
    k_sample = kwargs.get('k_sample', SHAP_N_SAMPLE)  # alias
    if 'k_sample' in kwargs:
        shap_n_sample = k_sample
    cma_max_fevals = kwargs.get('cma_max_fevals', CMA_MAX_FEVALS)
    cma_sigma0 = kwargs.get('cma_sigma0', CMA_SIGMA0)
    cma_delta_bound = kwargs.get('cma_delta_bound', CMA_DELTA_BOUND)

    print('=' * 60)
    print(f'FLShap-CMA: defect_type={defect_type}')
    print('=' * 60)

    # ---- Cache path (includes defect_order) ----
    loc_dir = os.path.dirname(loc_cache_path)
    defect_order = kwargs.get('defect_order', [defect_type])
    defect_order_str = '_'.join(defect_order)
    flshap_loc_cache = os.path.join(
        loc_dir, f'flshap_loc_{defect_order_str}_{defect_type}.json')

    # ---- 1. Localization ----
    print(f'  FLShap localization: shap_n_sample={shap_n_sample}')
    neuron_scores = compute_flshap_localization(
        model, defect_data, clean_data, defect_type, target_label,
        n_sample=shap_n_sample, cache_path=flshap_loc_cache)

    if not neuron_scores:
        print('  Warning: No neurons scored, returning original model')
        return model, {'error': 'No neurons scored'}

    # ---- 2. Select neurons ----
    repair_num = max(1, min(repair_num_max,
                      int(np.ceil(len(neuron_scores) * repair_ratio))))

    # Try selecting repair_num * 3 candidates, filter, then take top repair_num
    candidate_scores = neuron_scores[:repair_num * 3]
    valid_info = _get_neuron_weight_info(model, candidate_scores)
    valid_info = valid_info[:repair_num]

    if len(valid_info) < repair_num:
        print(f'  Warning: only {len(valid_info)}/{repair_num} valid neurons '
              f'(from {len(candidate_scores)} candidates)')
        if len(valid_info) == 0:
            print('  No valid neurons for weight modification')
            return model, {'error': 'No valid neurons'}

    selected_neurons = valid_info
    total_dim = sum(info['in_features'] for info in selected_neurons)

    print(f'  Total neurons: {len(neuron_scores)}, '
          f'repair_ratio={repair_ratio} -> repair_num={repair_num}')
    print(f'  Valid neurons for weight modification: {len(selected_neurons)}')
    print(f'  Total decision variables: {total_dim}')

    for info in selected_neurons:
        print(f'    - {info["layer_name"]} neuron {info["neuron_idx"]} '
              f'({info["repair_type"]}, dim={info["in_features"]})')

    # ---- 3. CMA-ES Optimization ----
    base_weights = copy.deepcopy(model.get_weights())
    x_clean, y_clean = clean_data
    x_defect, y_defect = defect_data

    evaluator = _FLShapEvaluator(
        model, selected_neurons, base_weights,
        x_clean, y_clean, x_defect, y_defect,
        defect_type, target_label)

    if total_dim == 0:
        print('  Warning: total_dim=0, returning original model')
        return model, {'error': 'Zero dimension'}

    initial_solution = np.zeros(total_dim)
    sigma0 = cma_sigma0
    options = {
        'bounds': [-cma_delta_bound, cma_delta_bound],
        'maxfevals': cma_max_fevals,
        'verbose': -9,
        'popsize': 4 + int(3 * np.log(total_dim)),
    }

    print(f'\nCMA-ES: max_fevals={cma_max_fevals}, sigma0={sigma0}, '
          f'n_var={total_dim}, delta_bound={cma_delta_bound}')

    start_time = time.time()
    try:
        best_solution, es = cma.fmin2(
            evaluator.evaluate, initial_solution, sigma0, options)
    except Exception as e:
        print(f'  CMA-ES failed: {e}')
        # Fall back to best found solution
        if evaluator.best_solution is not None:
            best_solution = evaluator.best_solution
        else:
            best_solution = initial_solution

    elapsed = time.time() - start_time
    print(f'CMA-ES completed in {elapsed:.2f}s')

    # ---- 4. Apply best solution ----
    # Apply directly to the original model (no clone_model to avoid OOM)
    final_weights = evaluator._apply_delta(best_solution)
    model.set_weights(final_weights)
    repaired_model = model

    final_acc = evaluator._compute_clean_acc(repaired_model)
    final_defect = evaluator._compute_defect_rate(repaired_model)

    print(f'\nFinal: clean_acc={final_acc:.4f}, defect_rate={final_defect:.4f}')
    print(f'  Baseline: clean_acc={evaluator.acc_clean_raw:.4f}, '
          f'defect_rate={evaluator.defect_rate_raw:.4f}')

    # ---- 5. Build stats ----
    stats = {
        'method': 'flshap_cma',
        'defect_type': defect_type,
        'seed': int(seed),
        'repair_ratio': repair_ratio,
        'repair_num': repair_num,
        'selected_neurons': len(selected_neurons),
        'total_dim': total_dim,
        'shap_n_sample': shap_n_sample,
        'cma_max_fevals': cma_max_fevals,
        'cma_sigma0': cma_sigma0,
        'cma_delta_bound': cma_delta_bound,
        'cma_time': float(elapsed),
        'baseline_clean_acc': float(evaluator.acc_clean_raw),
        'baseline_defect_rate': float(evaluator.defect_rate_raw),
        'final_clean_acc': float(final_acc),
        'final_defect_rate': float(final_defect),
        'best_score': float(evaluator.best_score),
        'neuron_info': [{
            'layer_name': info['layer_name'],
            'neuron_idx': info['neuron_idx'],
            'repair_type': info['repair_type'],
            'in_features': info['in_features'],
        } for info in selected_neurons],
    }

    return repaired_model, stats