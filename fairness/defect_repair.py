"""
Unified fairness repair evaluation framework.

Design:
- Loads model and fairness data once
- Dispatches to different repair methods via a registry
- Each method returns a repaired model + stats
- Evaluates all methods on the test set with consistent metrics
- Saves results (weights, stats) organized by method name

Single defect type: fairness
Supported models: bank, census
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input

from repair_config import (
    load_fairness_data,
    sess_run,
    value_accuracy,
    value_fairness_gap,
    labels_from_output,
    binary_group_accuracy_gap,
)


# ============================================================
# Method implementations
# ============================================================

def run_multirepair(model, clean_data, defect_data, seed, model_name,
                    method_kwargs):
    """My method: neuron localization + multi-task gradient repair."""
    from multirepair import repair_defects

    return repair_defects(
        model, clean_data, defect_data,
        loc_cache_path=method_kwargs['loc_cache_path'],
        model_name=model_name, seed=seed,
        method=method_kwargs.get('method', 'pcgrad'),
        mode=method_kwargs.get('mode', 'auto'),
        repair_ratio=method_kwargs.get('repair_ratio', 0.01),
        epsilon=method_kwargs.get('epsilon', 0.0),
        lr=method_kwargs.get('lr'),
        steps=method_kwargs.get('steps'),
        eval_every=method_kwargs.get('eval_every'),
    )


def run_care(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """CARE: causal analysis + PSO repair."""
    from methods.care_repair import care_repair_single
    return care_repair_single(
        model, defect_data, clean_data, 'fairness',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_ccbr(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """CCBR: Integrated Gradients + NSGA-III scale factor repair."""
    from methods.ccbr import ccbr_single
    return ccbr_single(
        model, defect_data, clean_data, 'fairness',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_flshap_cma(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """FLShap-CMA: hierarchical Shapley + CMA-ES weight optimization."""
    from methods.flshap_cma import flshap_cma_single
    return flshap_cma_single(
        model, defect_data, clean_data, 'fairness',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_nsrepair(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """NSRepair: GradientSHAP + NSGA-III multi-objective repair."""
    from methods.nsrepair import nsrepair_single
    return nsrepair_single(
        model, defect_data, clean_data, 'fairness',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_fairflrep(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """FairFLRep: fairness-aware gradient localization + PSO weight repair."""
    from methods.fairflrep import fairflrep_single
    return fairflrep_single(
        model, defect_data, clean_data, 'fairness',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


REPAIR_METHODS = {
    'multirepair': {
        'fn': run_multirepair,
        'description': 'Neuron localization + multi-task gradient repair',
        'default_kwargs': {'method': 'pcgrad', 'mode': 'auto', 'repair_ratio': 0.01},
    },
    'care': {
        'fn': run_care,
        'description': 'CARE: causal analysis + PSO repair',
        'default_kwargs': {},
    },
    'ccbr': {
        'fn': run_ccbr,
        'description': 'CCBR: Integrated Gradients + NSGA-III scale factor repair',
        'default_kwargs': {},
    },
    'flshap_cma': {
        'fn': run_flshap_cma,
        'description': 'FLShap-CMA: hierarchical Shapley + CMA-ES',
        'default_kwargs': {},
    },
    'nsrepair': {
        'fn': run_nsrepair,
        'description': 'NSRepair: GradientSHAP + NSGA-III repair',
        'default_kwargs': {},
    },
    'fairflrep': {
        'fn': run_fairflrep,
        'description': 'FairFLRep: fairness-aware gradient localization + PSO direct weight repair',
        'default_kwargs': {},
    },
}


# ============================================================
# Metric helpers
# ============================================================

def evaluate_with_coefficients(model, coefficient_data, x_data, batch_size=1024):
    """Evaluate model with CARE/CCBR coefficients applied via weight modification.

    Scale factors are applied to the NEXT layer's kernel weights (matching the
    PSO training approach), which bypasses BatchNorm layers that would otherwise
    normalize away the scaled intermediate activations.
    """
    neurons = coefficient_data.get('repair_neurons', [])
    if not neurons:
        return sess_run(model, model.output, x_data, batch_size)

    # Group neurons by their repair layer
    layer_groups = {}
    for n in neurons:
        layer_idx = n['layer_idx']
        if layer_idx not in layer_groups:
            layer_groups[layer_idx] = []
        layer_groups[layer_idx].append(n)

    # Find the next conv/dense layer for each repair layer and save original weights
    layer_mods = {}  # next_layer_idx -> {'layer': layer, 'original_weights': [...], 'kernel': array}
    for repair_layer_idx, group_neurons in layer_groups.items():
        repair_layer = model.layers[repair_layer_idx]
        print(f'  [DEBUG] Repair layer: idx={repair_layer_idx}, '
              f'name={repair_layer.name}, type={type(repair_layer).__name__}, '
              f'n_neurons={len(group_neurons)}')

        # Find the next conv/dense layer
        next_layer_idx = None
        for j in range(repair_layer_idx + 1, len(model.layers)):
            ltype = type(model.layers[j]).__name__.lower()
            if 'conv' in ltype or 'dense' in ltype:
                next_layer_idx = j
                break

        if next_layer_idx is None:
            print(f'  [DEBUG] WARNING: no next conv/dense layer found!')
            continue

        next_layer = model.layers[next_layer_idx]
        print(f'  [DEBUG] Next layer: idx={next_layer_idx}, '
              f'name={next_layer.name}, type={type(next_layer).__name__}')

        if next_layer_idx not in layer_mods:
            layer = model.layers[next_layer_idx]
            orig_w = layer.get_weights()
            layer_mods[next_layer_idx] = {
                'layer': layer,
                'original_weights': orig_w,
                'kernel': orig_w[0].copy(),
                'layer_type': type(layer).__name__.lower(),
                'neurons': [],
            }
        layer_mods[next_layer_idx]['neurons'].extend(group_neurons)

    if not layer_mods:
        print(f'  [DEBUG] WARNING: layer_mods is empty, returning unscaled output')
        return sess_run(model, model.output, x_data, batch_size)

    # Apply scale factors to the next layer's kernel
    for next_idx, mod_info in layer_mods.items():
        kernel = mod_info['kernel']
        ltype = mod_info['layer_type']
        orig_kernel = mod_info['original_weights'][0]
        print(f'  [DEBUG] Modifying layer {next_idx} ({mod_info["layer"].name}), '
              f'kernel shape={kernel.shape}, type={ltype}')
        for n in mod_info['neurons']:
            actual_scale = n['actual_scale']
            neuron_idx = n['neuron_idx']
            if 'conv' in ltype:
                kernel[..., neuron_idx, :] *= actual_scale
            else:
                kernel[neuron_idx, :] *= actual_scale

        # Check if kernel actually changed
        max_diff = np.max(np.abs(kernel - orig_kernel))
        print(f'  [DEBUG] Kernel max diff after scaling: {max_diff:.6f}')

        # Set modified weights
        orig_w = mod_info['original_weights']
        if len(orig_w) > 1:
            mod_info['layer'].set_weights([kernel, orig_w[1]])
        else:
            mod_info['layer'].set_weights([kernel])

    # Verify weights were set by reading them back
    for next_idx, mod_info in layer_mods.items():
        readback = mod_info['layer'].get_weights()[0]
        max_diff = np.max(np.abs(readback - mod_info['kernel']))
        print(f'  [DEBUG] Weight set verification (layer {next_idx}): max diff={max_diff:.6f}')

    # Evaluate with full model (no splitting needed)
    result = sess_run(model, model.output, x_data, batch_size)

    # Restore original weights
    for next_idx, mod_info in layer_mods.items():
        orig_w = mod_info['original_weights']
        if len(orig_w) > 1:
            mod_info['layer'].set_weights([orig_w[0], orig_w[1]])
        else:
            mod_info['layer'].set_weights([orig_w[0]])

    return result


def evaluate_model(model, test_data, coefficient_data=None):
    """Evaluate model on test metrics. If coefficient_data is provided,
    uses split-model evaluation with coefficient scaling (for CARE/CCBR)."""
    if coefficient_data is not None and coefficient_data.get('repair_neurons'):
        clean_out = evaluate_with_coefficients(model, coefficient_data,
                                               test_data['x_test'])
        clean_pred = labels_from_output(clean_out)
        clean_acc = float(np.mean(clean_pred == test_data['y_test']))

        groups = test_data.get('g_test', test_data['y_test'])
        fairness_gap = binary_group_accuracy_gap(
            clean_pred, test_data['y_test'], groups)

        return {'clean_acc': clean_acc, 'fairness_gap': fairness_gap}

    return {
        'clean_acc': float(value_accuracy(model, test_data['x_test'], test_data['y_test'])),
        'fairness_gap': float(value_fairness_gap(
            model, test_data['x_test'], test_data['y_test'],
            test_data.get('g_test'), test_data.get('dataset'))),
    }


# ============================================================
# Data preparation
# ============================================================

def prepare_data(model_name, seed):
    """Prepare fairness data for training and test."""
    print('=' * 60)
    print(f'Preparing {model_name} fairness data ...')
    print('=' * 60)

    train_full, sample_test_full = load_fairness_data(
        model_name, sample_size=512, seed=seed, return_groups=True)
    train = train_full
    sample_test = sample_test_full

    # For fairness, clean_data and defect_data use the same data
    # but with different loss functions
    clean_data = train
    defect_data = sample_test  # fairness loss uses same data

    # Test data
    _, test_full = load_fairness_data(
        model_name, sample_size=None, seed=seed + 999, return_groups=True)

    test_data = {
        'x_test': test_full[0],
        'y_test': test_full[1],
        'g_test': test_full[2],
        'dataset': model_name,
    }

    print(f'Training data size: {len(train[0])}')
    print(f'Test data size:     {len(test_full[0])}')
    print()

    return clean_data, defect_data, test_data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified fairness repair evaluation framework.')
    parser.add_argument('model_name', type=str, help='Model name: bank or census')
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('methods', type=str,
                        help='Method name, comma-separated list, or "all"')
    parser.add_argument('mode', type=str, nargs='?', default='auto',
                        help='Repair mode (auto, neuron, full)')
    args = parser.parse_args()

    model_name = args.model_name.lower()
    if model_name not in ('bank', 'census'):
        print('Unsupported fairness dataset. Use bank or census.')
        sys.exit(1)
    seed = args.seed
    methods_arg = args.methods
    mode = args.mode

    if methods_arg.lower() == 'all':
        method_names = list(REPAIR_METHODS.keys())
    else:
        method_names = [m.strip() for m in methods_arg.split(',')]
    for m in method_names:
        if m not in REPAIR_METHODS:
            print(f'Unknown method: {m}. Available: {list(REPAIR_METHODS.keys())}')
            sys.exit(1)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ---- Load model ----
    model_path = f'../model/{model_name}.h5'
    model = load_model(model_path, compile=False)
    original_weights = copy.deepcopy(model.get_weights())

    # ---- Prepare data ----
    clean_data, defect_data, test_data = prepare_data(model_name, seed)

    # ---- Evaluate BEFORE repair ----
    print('=' * 60)
    print('Initial model evaluation (before repair)')
    print('=' * 60)
    init_metrics = evaluate_model(model, test_data)
    for k, v in init_metrics.items():
        print(f'  {k}: {v:.6f}')
    print()

    all_results = {}

    for method_name in method_names:
        print('=' * 60)
        print(f'Method: {method_name}')
        print(f'  {REPAIR_METHODS[method_name]["description"]}')
        print('=' * 60)

        model.set_weights(copy.deepcopy(original_weights))

        method_info = REPAIR_METHODS[method_name]
        method_kwargs = dict(method_info['default_kwargs'])
        method_kwargs['mode'] = mode
        method_kwargs['loc_cache_path'] = os.path.join(
            'loc_result', model_name, str(seed), 'loc_neuron_output.json')

        repaired_model, repair_stats = method_info['fn'](
            model, clean_data, defect_data, seed, model_name, method_kwargs)

        # ---- Load coefficients for CARE/CCBR ----
        coeff_data = None
        if method_name in ('care', 'ccbr'):
            coeff_dir = os.path.join('loc_result', model_name, str(seed),
                                     'defect_repair')
            coeff_path = os.path.join(coeff_dir, f'coefficients_{method_name}.json')
            if os.path.exists(coeff_path):
                with open(coeff_path, 'r', encoding='utf-8') as f:
                    coeff_data = json.load(f)
                print(f'  Loaded coefficients from: {coeff_path}')

        # ---- Evaluate AFTER repair ----
        print('=' * 60)
        print(f'Evaluation: {method_name}')
        print('=' * 60)
        final_metrics = evaluate_model(repaired_model, test_data, coefficient_data=coeff_data)
        for k, v in final_metrics.items():
            init_v = init_metrics[k]
            delta = v - init_v
            print(f'  {k}: {v:.6f} (delta: {delta:+.6f})')
        print()

        # ---- Save results ----
        out_dir = os.path.join('loc_result', model_name, str(seed),
                               'defect_repair', method_name)
        os.makedirs(out_dir, exist_ok=True)

        if method_name in ('care', 'ccbr'):
            # Coefficients are saved by the repair function itself
            print(f'  Coefficients saved by repair function')
        else:
            weights_filename = f'weights_final_{method_name}.h5'
            repaired_model.save_weights(os.path.join(out_dir, weights_filename))

        run_stats = {
            'seed': int(seed),
            'method': method_name,
            'mode': mode,
            'model_name': model_name,
            'initial_metrics': init_metrics,
            'final_metrics': final_metrics,
            'repair_stats': repair_stats,
        }
        with open(os.path.join(out_dir, 'run_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(run_stats, f, indent=2)

        print(f'Saved to: {out_dir}')
        all_results[method_name] = final_metrics

    # ---- Summary ----
    if len(method_names) > 1:
        print()
        print('=' * 80)
        print('Comparison Summary')
        print('=' * 80)
        header = f'{"Metric":<20} {"Initial":>12}'
        for mn in method_names:
            header += f' {mn:>14}'
        print(header)
        print('-' * 80)
        for metric in ['clean_acc', 'fairness_gap']:
            row = f'{metric:<20} {init_metrics[metric]:>12.6f}'
            for mn in method_names:
                row += f' {all_results[mn][metric]:>14.6f}'
            print(row)
        print('=' * 80)


if __name__ == '__main__':
    main()

