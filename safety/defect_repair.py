"""
Unified safety repair evaluation framework.

Design:
- Loads model and safety data once
- Dispatches to different repair methods via a registry
- Each method returns a repaired model + stats
- Evaluates all methods on the test set with consistent metrics
- Saves results (weights, stats) organized by method name

Single defect type: safety
Supported models: a19, a29, a33
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
    load_safety_data,
    sess_run,
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
        model, defect_data, clean_data, 'safety',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_ccbr(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """CCBR: Integrated Gradients + NSGA-III scale factor repair."""
    from methods.ccbr import ccbr_single
    return ccbr_single(
        model, defect_data, clean_data, 'safety',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_flshap_cma(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """FLShap-CMA: hierarchical Shapley + CMA-ES weight optimization."""
    from methods.flshap_cma import flshap_cma_single
    return flshap_cma_single(
        model, defect_data, clean_data, 'safety',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_nsrepair(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """NSRepair: GradientSHAP + NSGA-III multi-objective repair."""
    from methods.nsrepair import nsrepair_single
    return nsrepair_single(
        model, defect_data, clean_data, 'safety',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_birdnn(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """BIRDNN: Behavior-Informed Repair via dual-task retraining (global)."""
    from methods.birdnn import birdnn_single
    return birdnn_single(
        model, defect_data, clean_data, 'safety',
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
    'birdnn': {
        'fn': run_birdnn,
        'description': 'BIRDNN: behavior-informed dual-task retraining (global)',
        'default_kwargs': {},
    },
}


# ============================================================
# Metric helpers
# ============================================================

def evaluate_with_coefficients(model, coefficient_data, x_data, batch_size=1024):
    """Evaluate model with CARE/CCBR coefficients applied via weight modification.

    Scale factors are applied by modifying the NEXT layer's kernel weights
    (or the current layer's kernel for last-layer neurons). This bypasses
    BatchNorm layers and supports neurons from multiple layers.
    """
    neurons = coefficient_data.get('repair_neurons', [])
    if not neurons:
        return sess_run(model, model.output, x_data, batch_size)

    # ---- Group neurons by their repair layer and find modification targets ----
    layer_mods = {}

    for n in neurons:
        repair_layer_idx = n['layer_idx']
        neuron_idx = n['neuron_idx']
        actual_scale = n['actual_scale']

        repair_layer = model.layers[repair_layer_idx]
        repair_ltype = type(repair_layer).__name__.lower()

        # Find the next conv/dense layer after the repair layer
        next_layer_idx = None
        for j in range(repair_layer_idx + 1, len(model.layers)):
            ltype = type(model.layers[j]).__name__.lower()
            if 'conv' in ltype or 'dense' in ltype:
                next_layer_idx = j
                break

        if next_layer_idx is not None:
            # Standard case: modify NEXT layer's kernel
            next_layer = model.layers[next_layer_idx]
            next_ltype = type(next_layer).__name__.lower()
            next_kernel_shape = next_layer.get_weights()[0].shape
            if 'conv' in next_ltype:
                in_dim = next_kernel_shape[-2]
            else:
                in_dim = next_kernel_shape[0]
            if neuron_idx >= in_dim:
                continue
            target_idx = next_layer_idx
            modify_type = 'next_layer'
            layer_type = next_ltype
        else:
            # Last layer case: modify CURRENT layer's kernel
            kernel_shape = repair_layer.get_weights()[0].shape
            out_dim = kernel_shape[-1]
            if neuron_idx >= out_dim:
                continue
            target_idx = repair_layer_idx
            modify_type = 'last_layer'
            layer_type = repair_ltype

        if target_idx not in layer_mods:
            layer = model.layers[target_idx]
            orig_w = layer.get_weights()
            layer_mods[target_idx] = {
                'layer': layer,
                'original_weights': orig_w,
                'kernel': orig_w[0].copy(),
                'bias': orig_w[1].copy() if len(orig_w) > 1 else None,
                'layer_type': layer_type,
                'neurons': [],
            }
        layer_mods[target_idx]['neurons'].append(
            (neuron_idx, actual_scale, modify_type))

    if not layer_mods:
        return sess_run(model, model.output, x_data, batch_size)

    # Apply scale factors to kernels
    for target_idx, mod_info in layer_mods.items():
        kernel = mod_info['kernel']
        ltype = mod_info['layer_type']
        for neuron_idx, actual_scale, modify_type in mod_info['neurons']:
            if modify_type == 'last_layer':
                if 'conv' in ltype:
                    kernel[..., :, neuron_idx] *= actual_scale
                else:
                    kernel[:, neuron_idx] *= actual_scale
                if mod_info['bias'] is not None:
                    mod_info['bias'][neuron_idx] *= actual_scale
            else:
                if 'conv' in ltype:
                    kernel[..., neuron_idx, :] *= actual_scale
                else:
                    kernel[neuron_idx, :] *= actual_scale

        if mod_info['bias'] is not None:
            mod_info['layer'].set_weights([kernel, mod_info['bias']])
        else:
            mod_info['layer'].set_weights([kernel])

    result = sess_run(model, model.output, x_data, batch_size)

    # Restore original weights
    for target_idx, mod_info in layer_mods.items():
        mod_info['layer'].set_weights(mod_info['original_weights'])

    return result


def value_safety(model, images, model_name):
    output = sess_run(model, model.output, images)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((np.sum(pred == 3) + np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((np.sum(pred == 0) + np.sum(pred == 1)) / len(output))
    return float((np.sum(pred != 0)) / len(output))


def value_safety_with_coeff(model, coefficient_data, images, model_name):
    output = evaluate_with_coefficients(model, coefficient_data, images)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((np.sum(pred == 3) + np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((np.sum(pred == 0) + np.sum(pred == 1)) / len(output))
    return float((np.sum(pred != 0)) / len(output))


def value_drawndown(model, images, model_name):
    output = sess_run(model, model.output, images)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((len(output) - np.sum(pred == 3) - np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((len(output) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output))
    return float(np.sum(pred == 0) / len(output))


def value_drawndown_with_coeff(model, coefficient_data, images, model_name):
    output = evaluate_with_coefficients(model, coefficient_data, images)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((len(output) - np.sum(pred == 3) - np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((len(output) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output))
    return float(np.sum(pred == 0) / len(output))


def evaluate_model(model, test_data, model_name, coefficient_data=None):
    """Evaluate model on test metrics. If coefficient_data is provided,
    uses split-model evaluation with coefficient scaling (for CARE/CCBR)."""
    if coefficient_data is not None and coefficient_data.get('repair_neurons'):
        return {
            'safety': float(value_safety_with_coeff(model, coefficient_data,
                                                    test_data['ct_x'], model_name)),
            'drawndown': float(value_drawndown_with_coeff(model, coefficient_data,
                                                          test_data['dr_x'], model_name)),
        }

    return {
        'safety': float(value_safety(model, test_data['ct_x'], model_name)),
        'drawndown': float(value_drawndown(model, test_data['dr_x'], model_name)),
    }


# ============================================================
# Data preparation
# ============================================================

def prepare_data(model_name):
    """Prepare safety data for training and test."""
    print('=' * 60)
    print(f'Preparing {model_name} safety data ...')
    print('=' * 60)

    ct_x, dr_x = load_safety_data(model_name)

    clean_data = dr_x           # drawndown data
    defect_data = ct_x          # counterexample data

    test_data = {
        'ct_x': ct_x,
        'dr_x': dr_x,
    }

    print(f'Training data sizes: ct={len(ct_x)}, dr={len(dr_x)}')
    print()

    return clean_data, defect_data, test_data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified safety repair evaluation framework.')
    parser.add_argument('model_name', type=str, help='Model name: a19, a29, a33')
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('methods', type=str,
                        help='Method name, comma-separated list, or "all"')
    parser.add_argument('mode', type=str, nargs='?', default='auto',
                        help='Repair mode (auto, neuron, full)')
    args = parser.parse_args()

    model_name = args.model_name
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
    clean_data, defect_data, test_data = prepare_data(model_name)

    # ---- Evaluate BEFORE repair ----
    print('=' * 60)
    print('Initial model evaluation (before repair)')
    print('=' * 60)
    init_metrics = evaluate_model(model, test_data, model_name)
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
        final_metrics = evaluate_model(repaired_model, test_data, model_name, coefficient_data=coeff_data)
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
        for metric in ['safety', 'drawndown']:
            row = f'{metric:<20} {init_metrics[metric]:>12.6f}'
            for mn in method_names:
                row += f' {all_results[mn][metric]:>14.6f}'
            print(row)
        print('=' * 80)


if __name__ == '__main__':
    main()