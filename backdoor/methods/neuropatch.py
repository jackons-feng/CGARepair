"""
Neuropatch-style repair method for single-defect repair (backdoor).

Based on BadDiffusion's Neuropatch approach, adapted from PyTorch diffusion
models to Keras classification models.

Original approach:
  1. Register hooks on SiLU layers to capture activations
  2. Compare clean vs backdoor neuron activations
  3. Generate mask (L1 distance or MAD-based outlier detection)
  4. Apply mask via hooks (flip or zero out targeted neurons)

Adaptation for Keras:
  1. Extract intermediate layer outputs via Keras functional API
  2. Compute per-neuron activation differences (L1 between clean and backdoor means)
  3. Select top-k neurons with largest differences
  4. Apply repair by modifying next-layer kernel weights (zero-out or flip)

Reference:
  BadDiffusion/baddiffusion.py  - original Neuropatch implementation
  BadDiffusion/hook.py          - hook mechanism for activation capture
"""

import os
import json
import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from repair_config import (
    DATASET_CONFIG,
    sess_run,
    value_accuracy,
    value_backdoor,
)
from loc_neuron_output import (
    get_layer_weight_indices,
)

# ============================================================
# Constants
# ============================================================
DEFAULT_NEURON_NUM = 50        # number of neurons to mask (matching original default)
DEFAULT_OPERATION_TYPE = 0     # 0 = flip, 1 = zero (matching original default)
DEFAULT_IS_MAD = True          # use MAD-based outlier detection


# ============================================================
# Activation extraction
# ============================================================

def get_layer_outputs(model, x_data, batch_size=256):
    """Extract intermediate layer outputs for a batch of data.

    Creates a multi-output Keras model that returns all Dense/Conv layer outputs,
    then runs inference on the provided data.

    Args:
        model: Keras model
        x_data: input data array
        batch_size: batch size for inference

    Returns:
        dict: layer_name -> numpy array of activations (samples, ..., neurons)
    """
    # Find all Dense and Conv2D layers
    layer_outputs = {}
    layer_names = []
    for layer in model.layers:
        ltype = type(layer).__name__.lower()
        if 'dense' in ltype or 'conv' in ltype:
            layer_names.append(layer.name)

    # Build multi-output model
    outputs = [model.get_layer(name).output for name in layer_names]
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=outputs)

    # Run inference
    all_outputs = intermediate_model.predict(x_data, batch_size=batch_size, verbose=0)

    # Organize into dict
    for name, out in zip(layer_names, all_outputs):
        layer_outputs[name] = out

    return layer_outputs


# ============================================================
# Mask computation (adapted from BadDiffusion)
# ============================================================

def compute_mask_l1(activations_clean, activations_backdoor, neuron_num):
    """Compute mask using L1 distance between clean and backdoor activation means.

    This corresponds to cauculate_mask_whole_neural_network in the original
    BadDiffusion code.

    For each layer, compute the L1 distance between mean clean and backdoor
    activations per neuron. Then select the top neuron_num neurons globally
    across all layers.

    Args:
        activations_clean: dict of layer_name -> numpy array (samples, ..., neurons)
        activations_backdoor: dict of layer_name -> numpy array
        neuron_num: number of neurons to mask

    Returns:
        dict: layer_name -> dict with 'mask' (bool array, True=keep) and
              'neuron_indices' (list of (neuron_idx,) tuples to mask)
    """
    # Compute per-neuron L1 distances
    all_diffs = []  # list of (layer_name, neuron_idx, diff_value)
    for name in activations_clean:
        clean_mean = np.mean(activations_clean[name], axis=0)
        bd_mean = np.mean(activations_backdoor[name], axis=0)
        l1_diff = np.abs(clean_mean - bd_mean)

        # Flatten spatial dims to get per-neuron differences
        if l1_diff.ndim > 1:
            # For Conv layers: average over spatial dimensions
            spatial_axes = tuple(range(len(l1_diff.shape) - 1))
            neuron_diffs = np.mean(l1_diff, axis=spatial_axes)
        else:
            neuron_diffs = l1_diff

        for ni in range(len(neuron_diffs)):
            all_diffs.append((name, ni, float(neuron_diffs[ni])))

    # Sort by difference descending and select top neuron_num
    all_diffs.sort(key=lambda x: x[2], reverse=True)
    selected = all_diffs[:min(neuron_num, len(all_diffs))]

    # Build result
    result = {}
    for name in activations_clean:
        result[name] = {
            'masked_neurons': [],
            'masked_values': [],
        }
    for name, ni, val in selected:
        result[name]['masked_neurons'].append(ni)
        result[name]['masked_values'].append(val)

    return result


def compute_mask_mad(activations_clean, activations_backdoor, neuron_num):
    """Compute mask using MAD-based outlier detection.

    This corresponds to cauculate_mask_whole_neural_network_mad in the original
    BadDiffusion code.

    Computes per-neuron activation differences, normalizes by max, then applies
    MAD (Median Absolute Deviation) to identify outlier neurons. Selects top
    neuron_num neurons with largest normalized differences.

    Args:
        activations_clean: dict of layer_name -> numpy array
        activations_backdoor: dict of layer_name -> numpy array
        neuron_num: number of neurons to mask

    Returns:
        dict: same format as compute_mask_l1
    """
    consistency_constant = 1.4826

    # Compute per-neuron activation differences
    all_diffs = []
    for name in activations_clean:
        clean_mean = np.mean(activations_clean[name], axis=0)
        bd_mean = np.mean(activations_backdoor[name], axis=0)
        l1_diff = np.abs(clean_mean - bd_mean)

        if l1_diff.ndim > 1:
            spatial_axes = tuple(range(len(l1_diff.shape) - 1))
            neuron_diffs = np.mean(l1_diff, axis=spatial_axes)
        else:
            neuron_diffs = l1_diff

        for ni in range(len(neuron_diffs)):
            all_diffs.append((name, ni, float(neuron_diffs[ni])))

    if not all_diffs:
        return {name: {'masked_neurons': [], 'masked_values': []}
                for name in activations_clean}

    # Normalize
    diff_values = np.array([d[2] for d in all_diffs])
    max_val = np.max(diff_values)
    if max_val > 0:
        diff_values = diff_values / max_val

    # MAD computation
    median = np.median(diff_values)
    abs_deviation = np.abs(diff_values - median)
    mad = consistency_constant * np.median(abs_deviation)

    # Select top neuron_num by normalized difference
    sorted_indices = np.argsort(diff_values)[::-1]
    selected_indices = sorted_indices[:min(neuron_num, len(sorted_indices))]

    # Build result
    result = {}
    for name in activations_clean:
        result[name] = {
            'masked_neurons': [],
            'masked_values': [],
        }
    for si in selected_indices:
        name, ni, val = all_diffs[si]
        result[name]['masked_neurons'].append(ni)
        result[name]['masked_values'].append(val)

    return result


# ============================================================
# Weight modification (Keras adaptation of mask hooks)
# ============================================================

def _get_neuron_weight_info(model, mask_info):
    """For each masked neuron, find the weight to modify.

    For neurons in non-last layers:
        Scaling neuron i's output at layer L to 0 is equivalent to setting
        the NEXT layer's kernel to 0 along input channel/row i:
        - Conv2D next layer: K_{L+1}[:, :, i, :] = 0
        - Dense next layer:    W_{L+1}[i, :] = 0

    For flip (operation_type=0):
        Negate the same weights.

    For neurons in the last layer:
        Modify the current layer's kernel.

    Args:
        model: Keras model
        mask_info: dict from compute_mask_* functions

    Returns:
        list of dicts with weight modification info
    """
    info = []
    for layer_name, minfo in mask_info.items():
        masked_neurons = minfo['masked_neurons']
        if not masked_neurons:
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
            # Standard case: modify NEXT layer's kernel
            weight_idx = 0
            for j in range(next_layer_idx):
                weight_idx += len(model.layers[j].get_weights())

            kernel_shape = model.get_weights()[weight_idx].shape
            next_layer_type = type(model.layers[next_layer_idx]).__name__.lower()

            if 'conv' in next_layer_type:
                in_dim = kernel_shape[-2]
            else:
                in_dim = kernel_shape[0]

            valid_neurons = [n for n in masked_neurons if n < in_dim]
            if len(valid_neurons) < len(masked_neurons):
                skipped = len(masked_neurons) - len(valid_neurons)
                print(f'  Warning: {skipped} neurons skipped (idx >= in_dim={in_dim}) '
                      f'for layer {layer_name} -> {model.layers[next_layer_idx].name}')

            for ni in valid_neurons:
                info.append({
                    'layer_name': layer_name,
                    'neuron_idx': ni,
                    'repair_type': 'next_layer',
                    'kernel_weight_idx': weight_idx,
                    'next_layer_type': next_layer_type,
                })
        else:
            # Last layer: modify CURRENT layer's kernel
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

            valid_neurons = [n for n in masked_neurons if n < out_dim]
            for ni in valid_neurons:
                info.append({
                    'layer_name': layer_name,
                    'neuron_idx': ni,
                    'repair_type': 'last_layer',
                    'kernel_weight_idx': weight_idx,
                    'bias_weight_idx': weight_idx + 1 if has_bias else None,
                    'last_layer_type': current_layer_type,
                })

    return info


def apply_neuropatch_repair(model, mask_info, operation_type=0):
    """Apply Neuropatch repair by modifying model weights.

    operation_type:
        0 = flip (negate weights for masked neurons)
        1 = zero (set weights to 0 for masked neurons)

    Args:
        model: Keras model
        mask_info: dict from compute_mask_*
        operation_type: 0 (flip) or 1 (zero)

    Returns:
        model: modified model
        stats: dict with repair statistics
    """
    neuron_info = _get_neuron_weight_info(model, mask_info)
    if not neuron_info:
        print('  Neuropatch: no valid neurons found for weight modification')
        return model, {'masked_neurons': 0, 'neuron_info': []}

    weights = model.get_weights()
    total_masked = 0

    for info in neuron_info:
        if info.get('repair_type') == 'last_layer':
            kernel = weights[info['kernel_weight_idx']]
            layer_type = info['last_layer_type']
            ni = info['neuron_idx']

            if operation_type == 0:
                # Flip: negate
                if 'conv' in layer_type:
                    kernel[..., ni] *= -1.0
                else:
                    kernel[:, ni] *= -1.0
            else:
                # Zero
                if 'conv' in layer_type:
                    kernel[..., ni] = 0.0
                else:
                    kernel[:, ni] = 0.0

            if info.get('bias_weight_idx') is not None:
                bias = weights[info['bias_weight_idx']]
                if operation_type == 0:
                    bias[ni] *= -1.0
                else:
                    bias[ni] = 0.0
        else:
            kernel = weights[info['kernel_weight_idx']]
            next_layer_type = info['next_layer_type']
            ni = info['neuron_idx']

            if operation_type == 0:
                if 'conv' in next_layer_type:
                    kernel[..., ni, :] *= -1.0
                else:
                    kernel[ni, :] *= -1.0
            else:
                if 'conv' in next_layer_type:
                    kernel[..., ni, :] = 0.0
                else:
                    kernel[ni, :] = 0.0

        total_masked += 1

    model.set_weights(weights)

    stats = {
        'masked_neurons': total_masked,
        'operation_type': operation_type,
        'neuron_info': [
            {
                'layer_name': ni['layer_name'],
                'neuron_idx': int(ni['neuron_idx']),
                'repair_type': ni.get('repair_type', 'next_layer'),
            }
            for ni in neuron_info
        ],
    }

    return model, stats


# ============================================================
# Main Neuropatch entry point
# ============================================================

def neuropatch_single(model, defect_data, clean_data, defect_type,
                      loc_cache_path, seed, model_name, epsilon, **kwargs):
    """
    Neuropatch-style single-defect repair.

    Compatible with defect_repair.py interface:
        fn(model, defect_data, clean_data, defect_type,
           loc_cache_path, seed, model_name, epsilon, **kwargs)
        -> (repaired_model, stats)

    Core steps:
      1. Extract intermediate layer activations for clean and backdoor data
      2. Compute activation differences (L1 or MAD)
      3. Select top-k neurons with largest differences
      4. Apply repair by modifying weights (flip or zero)

    Args:
        model: pre-loaded Keras model
        defect_data: (x, y) tuple for backdoor data
        clean_data: (x_clean, y_clean) for accuracy preservation
        defect_type: 'backdoor'
        loc_cache_path: path to localization cache (for stats/cache)
        seed: random seed
        model_name: model identifier
        epsilon: unused (kept for interface compatibility)
        **kwargs: optional overrides
            - neuron_num: number of neurons to mask (default 50)
            - operation_type: 0=flip, 1=zero (default 0)
            - is_mad: use MAD-based detection (default True)

    Returns:
        repaired_model: model with Neuropatch repair applied
        stats: dict with repair statistics
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_clean, y_clean = clean_data
    x_bd, y_bd = defect_data

    neuron_num = kwargs.get('neuron_num', DEFAULT_NEURON_NUM)
    operation_type = kwargs.get('operation_type', DEFAULT_OPERATION_TYPE)
    is_mad = kwargs.get('is_mad', DEFAULT_IS_MAD)

    print('=' * 60)
    print(f'Neuropatch Repair: defect_type={defect_type}')
    print(f'  neuron_num={neuron_num}, operation_type={operation_type}'
          f' ({"flip" if operation_type == 0 else "zero"}), is_mad={is_mad}')
    print('=' * 60)

    # ---- 1. Extract activations ----
    start_time = time.time()
    print(f'  Extracting activations: {len(x_clean)} clean, {len(x_bd)} backdoor samples')

    acts_clean = get_layer_outputs(model, x_clean)
    acts_bd = get_layer_outputs(model, x_bd)

    extract_time = time.time() - start_time
    print(f'  Activation extraction: {extract_time:.2f}s')

    # ---- 2. Compute mask ----
    if is_mad:
        mask_info = compute_mask_mad(acts_clean, acts_bd, neuron_num)
    else:
        mask_info = compute_mask_l1(acts_clean, acts_bd, neuron_num)

    total_masked = sum(len(v['masked_neurons']) for v in mask_info.values())
    print(f'  Neurons selected for masking: {total_masked}')

    for name, minfo in mask_info.items():
        if minfo['masked_neurons']:
            print(f'    {name}: {len(minfo["masked_neurons"])} neurons')

    # ---- 3. Apply repair ----
    model, repair_stats = apply_neuropatch_repair(model, mask_info, operation_type)

    repair_time = time.time() - start_time
    print(f'  Total repair time: {repair_time:.2f}s')

    # ---- 4. Build stats ----
    stats = {
        'method': 'neuropatch',
        'defect_type': defect_type,
        'neuron_num': neuron_num,
        'operation_type': operation_type,
        'is_mad': is_mad,
        'total_masked': total_masked,
        'extract_time': float(extract_time),
        'repair_time': float(repair_time),
        'mask_info': {
            name: {
                'masked_neurons': [int(n) for n in minfo['masked_neurons']],
                'masked_count': len(minfo['masked_neurons']),
            }
            for name, minfo in mask_info.items()
            if minfo['masked_neurons']
        },
        **repair_stats,
    }

    return model, stats