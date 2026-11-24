"""
DeepProv-style repair method for single-defect repair (adversarial).

Based on DeepProv's approach, adapted from PyTorch to Keras classification
models.

Original approach (DeepProv):
  1. Build Inference Provenance Graph (IPG) from neuron activations
  2. Select nodes based on activation frequency, attribution values, and
     activation differences between benign and adversarial samples
  3. Extract benign activation distribution via KDE
  4. Apply repair by constraining selected neurons toward benign distribution:
     x = x - alpha * (x - ben_distr)  (when |Delta| < beta)

Adaptation for Keras:
  1. Extract intermediate layer activations for benign and adversarial data
  2. Compute per-neuron activation differences
  3. Select top-k neurons with largest differences
  4. Compute scale factors based on benign_mean / adversarial_mean ratio
  5. Apply repair by scaling next-layer kernel weights

Reference:
  DeepProv-main/artifact/main.py           - main repair pipeline
  DeepProv-main/artifact/library/mod_models.py - act_on constraint mechanism
  DeepProv-main/artifact/library/actions_selections.py - node selection
"""

import os
import json
import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

# ============================================================
# Constants
# ============================================================
DEFAULT_NEURON_NUM = 50        # number of neurons to constrain
DEFAULT_ALPHA = 0.5            # interpolation strength (matching DeepProv search range)
DEFAULT_BETA = 10.0            # max Delta threshold for constraint


# ============================================================
# Activation extraction
# ============================================================

def get_layer_outputs(model, x_data, batch_size=256):
    """Extract intermediate layer outputs for a batch of data.

    Args:
        model: Keras model
        x_data: input data array
        batch_size: batch size for inference

    Returns:
        dict: layer_name -> numpy array of activations (samples, ..., neurons)
    """
    layer_names = []
    for layer in model.layers:
        ltype = type(layer).__name__.lower()
        if 'dense' in ltype or 'conv' in ltype:
            layer_names.append(layer.name)

    if not layer_names:
        return {}

    outputs = [model.get_layer(name).output for name in layer_names]
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=outputs)

    all_outputs = intermediate_model.predict(x_data, batch_size=batch_size, verbose=0)

    result = {}
    # Handle case where only one layer
    if len(layer_names) == 1:
        result[layer_names[0]] = all_outputs
    else:
        for name, out in zip(layer_names, all_outputs):
            result[name] = out

    return result


# ============================================================
# Neuron selection (adapted from DeepProv's select_nodes)
# ============================================================

def compute_activation_stats(activations):
    """Compute per-neuron mean and std from activation samples.

    Args:
        activations: dict of layer_name -> numpy array (samples, ..., neurons)

    Returns:
        dict: layer_name -> {'mean': array, 'std': array}
    """
    stats = {}
    for name, act in activations.items():
        mean_val = np.mean(act, axis=0)
        std_val = np.std(act, axis=0)
        stats[name] = {'mean': mean_val, 'std': std_val}
    return stats


def select_neurons_deepprov(acts_benign, acts_adv, neuron_num):
    """Select neurons based on activation differences between benign and
    adversarial samples.

    This adapts DeepProv's three-criterion selection (activation frequency,
    attribution, activation difference) into a simplified version using
    activation differences only, since we don't have the IPG attribution
    infrastructure.

    For each neuron, computes the normalized activation difference:
        diff = |mean_benign - mean_adv| / (std_benign + std_adv + eps)

    Selects top neuron_num neurons globally across all layers.

    Args:
        acts_benign: dict of layer_name -> numpy array
        acts_adv: dict of layer_name -> numpy array
        neuron_num: number of neurons to select

    Returns:
        dict: layer_name -> {
            'selected_neurons': list of neuron indices,
            'benign_means': list of benign mean values,
            'adv_means': list of adversarial mean values,
            'diff_scores': list of difference scores,
        }
    """
    eps = 1e-8

    # Compute per-neuron normalized differences
    all_diffs = []
    for name in acts_benign:
        ben_mean = np.mean(acts_benign[name], axis=0)
        adv_mean = np.mean(acts_adv[name], axis=0)
        ben_std = np.std(acts_benign[name], axis=0)
        adv_std = np.std(acts_adv[name], axis=0)

        l1_diff = np.abs(ben_mean - adv_mean)

        # Flatten spatial dims to get per-neuron differences
        if l1_diff.ndim > 1:
            spatial_axes = tuple(range(len(l1_diff.shape) - 1))
            neuron_diffs = np.mean(l1_diff, axis=spatial_axes)
            neuron_ben_means = np.mean(ben_mean, axis=spatial_axes)
            neuron_adv_means = np.mean(adv_mean, axis=spatial_axes)
            neuron_ben_stds = np.mean(ben_std, axis=spatial_axes)
            neuron_adv_stds = np.mean(adv_std, axis=spatial_axes)
        else:
            neuron_diffs = l1_diff
            neuron_ben_means = ben_mean
            neuron_adv_means = adv_mean
            neuron_ben_stds = ben_std
            neuron_adv_stds = adv_std

        for ni in range(len(neuron_diffs)):
            # Normalized difference (like DeepProv's z-score approach)
            combined_std = neuron_ben_stds[ni] + neuron_adv_stds[ni] + eps
            norm_diff = float(neuron_diffs[ni] / combined_std)

            all_diffs.append({
                'layer': name,
                'neuron_idx': ni,
                'norm_diff': norm_diff,
                'raw_diff': float(neuron_diffs[ni]),
                'ben_mean': float(neuron_ben_means[ni]),
                'adv_mean': float(neuron_adv_means[ni]),
            })

    # Sort by normalized difference descending
    all_diffs.sort(key=lambda x: x['norm_diff'], reverse=True)
    selected = all_diffs[:min(neuron_num, len(all_diffs))]

    # Build result
    result = {}
    for name in acts_benign:
        result[name] = {
            'selected_neurons': [],
            'benign_means': [],
            'adv_means': [],
            'diff_scores': [],
        }
    for item in selected:
        name = item['layer']
        result[name]['selected_neurons'].append(item['neuron_idx'])
        result[name]['benign_means'].append(item['ben_mean'])
        result[name]['adv_means'].append(item['adv_mean'])
        result[name]['diff_scores'].append(item['norm_diff'])

    return result


# ============================================================
# Weight modification (Keras adaptation of act_on constraints)
# ============================================================

def _get_neuron_weight_info(model, selection_info):
    """For each selected neuron, find the weight to modify.

    Args:
        model: Keras model
        selection_info: dict from select_neurons_deepprov

    Returns:
        list of dicts with weight modification info
    """
    info = []
    for layer_name, sinfo in selection_info.items():
        selected_neurons = sinfo['selected_neurons']
        if not selected_neurons:
            continue

        # Find layer index
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
            weight_idx = 0
            for j in range(next_layer_idx):
                weight_idx += len(model.layers[j].get_weights())

            kernel_shape = model.get_weights()[weight_idx].shape
            next_layer_type = type(model.layers[next_layer_idx]).__name__.lower()

            if 'conv' in next_layer_type:
                in_dim = kernel_shape[-2]
            else:
                in_dim = kernel_shape[0]

            for ni, ben_mean, adv_mean in zip(
                    selected_neurons, sinfo['benign_means'], sinfo['adv_means']):
                if ni >= in_dim:
                    continue
                info.append({
                    'layer_name': layer_name,
                    'neuron_idx': ni,
                    'benign_mean': ben_mean,
                    'adv_mean': adv_mean,
                    'repair_type': 'next_layer',
                    'kernel_weight_idx': weight_idx,
                    'next_layer_type': next_layer_type,
                })
        else:
            # Last layer
            current_layer = model.layers[layer_idx]
            current_layer_type = type(current_layer).__name__.lower()
            weights = current_layer.get_weights()
            if len(weights) == 0:
                continue

            kernel_shape = weights[0].shape
            out_dim = kernel_shape[-1]

            weight_idx = 0
            for j in range(layer_idx):
                weight_idx += len(model.layers[j].get_weights())
            has_bias = len(weights) > 1

            for ni, ben_mean, adv_mean in zip(
                    selected_neurons, sinfo['benign_means'], sinfo['adv_means']):
                if ni >= out_dim:
                    continue
                info.append({
                    'layer_name': layer_name,
                    'neuron_idx': ni,
                    'benign_mean': ben_mean,
                    'adv_mean': adv_mean,
                    'repair_type': 'last_layer',
                    'kernel_weight_idx': weight_idx,
                    'bias_weight_idx': weight_idx + 1 if has_bias else None,
                    'last_layer_type': current_layer_type,
                })

    return info


def apply_deepprov_repair(model, selection_info, alpha=0.5, beta=10.0):
    """Apply DeepProv-style repair by constraining neuron activations toward
    benign distribution.

    This adapts the act_on() mechanism:
        x_new = x - alpha * (x - ben_distr)  when |x - ben_distr| < beta

    In weight space, this is approximated by scaling the next-layer kernel
    weights for selected neurons. The scale factor is computed as:
        scale = 1.0 - alpha * (1.0 - ben_mean / adv_mean)

    This effectively pulls the neuron's contribution toward the benign level.

    Args:
        model: Keras model
        selection_info: dict from select_neurons_deepprov
        alpha: interpolation strength (0 = no change, 1 = fully benign)
        beta: max Delta threshold (unused in weight-space approximation)

    Returns:
        model: modified model
        stats: dict with repair statistics
    """
    eps = 1e-8
    neuron_info = _get_neuron_weight_info(model, selection_info)
    if not neuron_info:
        print('  DeepProv: no valid neurons found for weight modification')
        return model, {'constrained_neurons': 0, 'neuron_info': []}

    weights = model.get_weights()
    total_constrained = 0
    scale_factors = []

    for info in neuron_info:
        ben_mean = info['benign_mean']
        adv_mean = info['adv_mean']

        # Compute scale factor: pull toward benign
        # If adv > ben (over-activated): scale down
        # If adv < ben (under-activated): scale up
        if abs(adv_mean) > eps:
            ratio = ben_mean / adv_mean
        else:
            ratio = 1.0

        # Clamp scale factor to reasonable range
        scale = 1.0 - alpha * (1.0 - ratio)
        scale = np.clip(scale, 0.1, 2.0)  # don't go too extreme
        scale_factors.append(float(scale))

        if info.get('repair_type') == 'last_layer':
            kernel = weights[info['kernel_weight_idx']]
            layer_type = info['last_layer_type']
            ni = info['neuron_idx']

            if 'conv' in layer_type:
                kernel[..., ni] *= scale
            else:
                kernel[:, ni] *= scale

            if info.get('bias_weight_idx') is not None:
                bias = weights[info['bias_weight_idx']]
                bias[ni] *= scale
        else:
            kernel = weights[info['kernel_weight_idx']]
            next_layer_type = info['next_layer_type']
            ni = info['neuron_idx']

            if 'conv' in next_layer_type:
                kernel[..., ni, :] *= scale
            else:
                kernel[ni, :] *= scale

        total_constrained += 1

    model.set_weights(weights)

    stats = {
        'constrained_neurons': total_constrained,
        'alpha': alpha,
        'beta': beta,
        'scale_factors': scale_factors,
        'scale_stats': {
            'mean': float(np.mean(scale_factors)),
            'std': float(np.std(scale_factors)),
            'min': float(np.min(scale_factors)),
            'max': float(np.max(scale_factors)),
        },
        'neuron_info': [
            {
                'layer_name': ni['layer_name'],
                'neuron_idx': int(ni['neuron_idx']),
                'benign_mean': float(ni['benign_mean']),
                'adv_mean': float(ni['adv_mean']),
                'scale_factor': float(sf),
                'repair_type': ni.get('repair_type', 'next_layer'),
            }
            for ni, sf in zip(neuron_info, scale_factors)
        ],
    }

    return model, stats


# ============================================================
# Main DeepProv entry point
# ============================================================

def deepprov_single(model, defect_data, clean_data, defect_type,
                    loc_cache_path, seed, model_name, epsilon, **kwargs):
    """
    DeepProv-style single-defect repair.

    Compatible with defect_repair.py interface:
        fn(model, defect_data, clean_data, defect_type,
           loc_cache_path, seed, model_name, epsilon, **kwargs)
        -> (repaired_model, stats)

    Core steps:
      1. Extract intermediate layer activations for benign and adversarial data
      2. Compute per-neuron activation differences (normalized by std)
      3. Select top-k neurons with largest differences
      4. Compute scale factors based on benign/adv ratio
      5. Apply repair by scaling next-layer kernel weights

    Args:
        model: pre-loaded Keras model
        defect_data: (x_adv, y_adv) tuple for adversarial data
        clean_data: (x_clean, y_clean) for benign reference
        defect_type: 'adv' (or 'fairness', 'safety')
        loc_cache_path: path to localization cache
        seed: random seed
        model_name: model identifier
        epsilon: adversarial epsilon
        **kwargs: optional overrides
            - neuron_num: number of neurons to constrain (default 50)
            - alpha: interpolation strength (default 0.5)
            - beta: max Delta threshold (default 10.0)

    Returns:
        repaired_model: model with DeepProv repair applied
        stats: dict with repair statistics
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_clean, y_clean = clean_data
    x_adv, y_adv = defect_data

    neuron_num = kwargs.get('neuron_num', DEFAULT_NEURON_NUM)
    alpha = kwargs.get('alpha', DEFAULT_ALPHA)
    beta = kwargs.get('beta', DEFAULT_BETA)

    print('=' * 60)
    print(f'DeepProv Repair: defect_type={defect_type}')
    print(f'  neuron_num={neuron_num}, alpha={alpha}, beta={beta}')
    print('=' * 60)

    # ---- 1. Extract activations ----
    start_time = time.time()
    print(f'  Extracting activations: {len(x_clean)} benign, {len(x_adv)} adversarial samples')

    acts_benign = get_layer_outputs(model, x_clean)
    acts_adv = get_layer_outputs(model, x_adv)

    extract_time = time.time() - start_time
    print(f'  Activation extraction: {extract_time:.2f}s')

    # ---- 2. Select neurons ----
    selection = select_neurons_deepprov(acts_benign, acts_adv, neuron_num)

    total_selected = sum(len(v['selected_neurons']) for v in selection.values())
    print(f'  Neurons selected for constraint: {total_selected}')

    for name, sinfo in selection.items():
        if sinfo['selected_neurons']:
            avg_score = np.mean(sinfo['diff_scores']) if sinfo['diff_scores'] else 0
            print(f'    {name}: {len(sinfo["selected_neurons"])} neurons'
                  f' (avg score={avg_score:.4f})')

    # ---- 3. Apply repair ----
    model, repair_stats = apply_deepprov_repair(model, selection, alpha, beta)

    repair_time = time.time() - start_time
    print(f'  Total repair time: {repair_time:.2f}s')
    if 'scale_stats' in repair_stats:
        ss = repair_stats['scale_stats']
        print(f'  Scale factors: mean={ss["mean"]:.4f}, std={ss["std"]:.4f}, '
              f'range=[{ss["min"]:.4f}, {ss["max"]:.4f}]')

    # ---- 4. Build stats ----
    stats = {
        'method': 'deepprov',
        'defect_type': defect_type,
        'neuron_num': neuron_num,
        'alpha': alpha,
        'beta': beta,
        'total_constrained': total_selected,
        'extract_time': float(extract_time),
        'repair_time': float(repair_time),
        'selection_info': {
            name: {
                'selected_neurons': [int(n) for n in sinfo['selected_neurons']],
                'selected_count': len(sinfo['selected_neurons']),
                'avg_diff_score': float(np.mean(sinfo['diff_scores'])) if sinfo['diff_scores'] else 0.0,
            }
            for name, sinfo in selection.items()
            if sinfo['selected_neurons']
        },
        **repair_stats,
    }

    return model, stats