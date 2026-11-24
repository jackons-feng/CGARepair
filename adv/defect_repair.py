"""
Unified adversarial repair evaluation framework.

Design:
- Loads model and adversarial data once
- Dispatches to different repair methods via a registry
- Each method returns a repaired model + stats
- Evaluates all methods on the test set with consistent metrics
- Saves results (weights, stats) organized by method name

Single defect type: adversarial
Supported models: mnist, cifar10
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from repair_config import (
    load_adv_data,
    value_accuracy,
    sess_run,
)


# ============================================================
# Method implementations
# ============================================================

def run_multirepair(model, clean_data, defect_data, seed, model_name,
                    method_kwargs):
    """My method: neuron localization + multi-task gradient repair."""
    from multirepair import repair_defects

    repaired_model, stats = repair_defects(
        model, clean_data, defect_data,
        loc_cache_path=method_kwargs['loc_cache_path'],
        model_name=model_name, seed=seed,
        method=method_kwargs.get('method', 'pcgrad'),
        mode=method_kwargs.get('mode', 'auto'),
        repair_ratio=method_kwargs.get('repair_ratio', 0.01),
        epsilon=method_kwargs.get('epsilon', 0.1),
        lr=method_kwargs.get('lr'),
        steps=method_kwargs.get('steps'),
        eval_every=method_kwargs.get('eval_every'),
    )
    return repaired_model, stats


def run_care(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """CARE: causal analysis + PSO repair."""
    from methods.care_repair import care_repair_single
    return care_repair_single(
        model, defect_data, clean_data, 'adv',
        method_kwargs['loc_cache_path'], seed, model_name,
        method_kwargs.get('epsilon', 0.1))


def run_ccbr(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """CCBR: Integrated Gradients + NSGA-III scale factor repair."""
    from methods.ccbr import ccbr_single
    return ccbr_single(
        model, defect_data, clean_data, 'adv',
        method_kwargs['loc_cache_path'], seed, model_name,
        method_kwargs.get('epsilon', 0.1))


def run_flshap_cma(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """FLShap-CMA: hierarchical Shapley + CMA-ES weight optimization."""
    from methods.flshap_cma import flshap_cma_single
    return flshap_cma_single(
        model, defect_data, clean_data, 'adv',
        method_kwargs['loc_cache_path'], seed, model_name,
        method_kwargs.get('epsilon', 0.1))


def run_nsrepair(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """NSRepair: GradientSHAP + NSGA-III multi-objective repair."""
    from methods.nsrepair import nsrepair_single
    return nsrepair_single(
        model, defect_data, clean_data, 'adv',
        method_kwargs['loc_cache_path'], seed, model_name,
        method_kwargs.get('epsilon', 0.1))


def run_deepprov(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """DeepProv: activation constraint + benign distribution repair."""
    from methods.deepprov import deepprov_single
    return deepprov_single(
        model, defect_data, clean_data, 'adv',
        method_kwargs['loc_cache_path'], seed, model_name,
        method_kwargs.get('epsilon', 0.1))


def run_birdnn(model, clean_data, defect_data, seed, model_name, method_kwargs):
    """BIRDNN: Behavior-Informed Repair via dual-task retraining (global)."""
    from methods.birdnn import birdnn_single
    return birdnn_single(
        model, defect_data, clean_data, 'adv',
        method_kwargs['loc_cache_path'], seed, model_name,
        method_kwargs.get('epsilon', 0.1))


# ============================================================
# Method registry
# ============================================================

REPAIR_METHODS = {
    'multirepair': {
        'fn': run_multirepair,
        'description': 'Neuron localization + multi-task gradient repair',
        'default_kwargs': {'method': 'pcgrad', 'mode': 'auto', 'repair_ratio': 0.01},
    },
    'care': {
        'fn': run_care,
        'description': 'CARE: causal analysis + PSO repair of intermediate outputs',
        'default_kwargs': {},
    },
    'ccbr': {
        'fn': run_ccbr,
        'description': 'CCBR: Integrated Gradients + NSGA-III scale factor repair',
        'default_kwargs': {},
    },
    'flshap_cma': {
        'fn': run_flshap_cma,
        'description': 'FLShap-CMA: hierarchical Shapley + CMA-ES weight optimization',
        'default_kwargs': {},
    },
    'nsrepair': {
        'fn': run_nsrepair,
        'description': 'NSRepair: GradientSHAP + NSGA-III multi-objective repair',
        'default_kwargs': {},
    },
    'deepprov': {
        'fn': run_deepprov,
        'description': 'DeepProv: activation constraint toward benign distribution',
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

    Two repair types are supported:
      - next_layer: Scale the NEXT layer's kernel rows/input-channels.
        (Standard case: neuron at layer L → modify kernel of layer L+1)
      - last_layer: Scale the CURRENT layer's kernel columns/output-channels
        AND bias. (Last-layer case: neuron is the classification layer itself,
        no next layer exists)

    This bypasses BatchNorm layers that would otherwise normalize away the
    scaled intermediate activations.
    """
    neurons = coefficient_data.get('repair_neurons', [])
    if not neurons:
        return sess_run(model, model.output, x_data, batch_size)

    # Separate neurons by repair_type
    next_layer_neurons = [n for n in neurons if n.get('repair_type') != 'last_layer']
    last_layer_neurons = [n for n in neurons if n.get('repair_type') == 'last_layer']

    # Collect all layers that need modification
    # layer_mods: target_layer_idx -> {layer, original_weights, kernel, bias, layer_type, neurons}
    layer_mods = {}

    # ---- Handle next_layer neurons: modify NEXT layer's kernel ----
    for n in next_layer_neurons:
        target_idx = n['next_layer_idx']
        if target_idx not in layer_mods:
            layer = model.layers[target_idx]
            orig_w = layer.get_weights()
            layer_mods[target_idx] = {
                'layer': layer,
                'original_weights': orig_w,
                'kernel': orig_w[0].copy(),
                'bias': orig_w[1].copy() if len(orig_w) > 1 else None,
                'layer_type': n.get('next_layer_type', type(layer).__name__.lower()),
                'neurons': [],
            }
        layer_mods[target_idx]['neurons'].append(n)

    # ---- Handle last_layer neurons: modify CURRENT layer's kernel + bias ----
    for n in last_layer_neurons:
        target_idx = n['layer_idx']
        if target_idx not in layer_mods:
            layer = model.layers[target_idx]
            orig_w = layer.get_weights()
            layer_mods[target_idx] = {
                'layer': layer,
                'original_weights': orig_w,
                'kernel': orig_w[0].copy(),
                'bias': orig_w[1].copy() if len(orig_w) > 1 else None,
                'layer_type': n.get('last_layer_type', type(layer).__name__.lower()),
                'neurons': [],
            }
        layer_mods[target_idx]['neurons'].append(n)

    if not layer_mods:
        return sess_run(model, model.output, x_data, batch_size)

    # ---- Apply scale factors to the target layer's kernel ----
    for target_idx, mod_info in layer_mods.items():
        kernel = mod_info['kernel']
        ltype = mod_info['layer_type']
        for n in mod_info['neurons']:
            actual_scale = n['actual_scale']
            neuron_idx = n['neuron_idx']
            if n.get('repair_type') == 'last_layer':
                # Scale current layer's output channel/column + bias
                if 'conv' in ltype:
                    kernel[..., :, neuron_idx] *= actual_scale
                else:
                    kernel[:, neuron_idx] *= actual_scale
                if mod_info['bias'] is not None:
                    mod_info['bias'][neuron_idx] *= actual_scale
            else:
                # Scale next layer's input channel/row
                if 'conv' in ltype:
                    kernel[..., neuron_idx, :] *= actual_scale
                else:
                    kernel[neuron_idx, :] *= actual_scale

        # Set modified weights
        if mod_info['bias'] is not None:
            mod_info['layer'].set_weights([kernel, mod_info['bias']])
        else:
            mod_info['layer'].set_weights([kernel])

    # Evaluate with full model (no splitting needed)
    result = sess_run(model, model.output, x_data, batch_size)

    # Restore original weights for all modified layers
    for target_idx, mod_info in layer_mods.items():
        mod_info['layer'].set_weights(mod_info['original_weights'])

    return result


def evaluate_model(model, test_data, coefficient_data=None):
    """Evaluate model on test metrics. If coefficient_data is provided,
    uses split-model evaluation with coefficient scaling (for CARE/CCBR)."""
    if coefficient_data is not None and coefficient_data.get('repair_neurons'):
        clean_out = evaluate_with_coefficients(model, coefficient_data,
                                               test_data['x_test'])
        clean_acc = float(np.mean(np.argmax(clean_out, axis=-1) == test_data['y_test']))
        adv_out = evaluate_with_coefficients(model, coefficient_data,
                                             test_data['adv_x_test'])
        adv_acc = float(np.mean(np.argmax(adv_out, axis=-1) == test_data['adv_y_test']))
        return {'clean_acc': clean_acc, 'adv_acc': adv_acc}

    return {
        'clean_acc': float(value_accuracy(model, test_data['x_test'], test_data['y_test'])),
        'adv_acc': float(value_accuracy(model, test_data['adv_x_test'], test_data['adv_y_test'])),
    }


# ============================================================
# Data preparation
# ============================================================

def prepare_data(model_name, epsilon):
    """Prepare adversarial data for training and test."""
    print('=' * 60)
    print(f'Preparing {model_name} adversarial data (eps={epsilon}) ...')
    print('=' * 60)

    adv_x, adv_y, clean_x, clean_y = load_adv_data(model_name, epsilon)

    # Test data
    if model_name == 'mnist':
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = np.expand_dims(x_test.astype('float32') / 255, -1)
    else:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_test = y_test.flatten()
        x_test = x_test.astype('float32') / 255.0
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
        x_test = (x_test - mean) / std

    # Load test adversarial data
    if model_name == 'mnist':
        adv_test_path = os.path.join('..', 'data', 'adv', 'mnist', 'test', f'epsilon{epsilon}.txt')
        adv_label_test_path = os.path.join('..', 'data', 'adv', 'mnist', 'test', 'label.txt')
        with open(adv_test_path, 'r') as f:
            adv_test_x = np.array([np.array(line.strip().split(','), dtype=float).reshape(28, 28, 1)
                                   for line in f.readlines()], dtype='float32')
        with open(adv_label_test_path, 'r') as f:
            adv_test_y = np.array([int(line.strip()) for line in f.readlines()], dtype=int)
    else:
        adv_test_path = os.path.join('..', 'data', 'adv', 'cifar10', 'test', f'epsilon{epsilon}.npy')
        adv_label_test_path = os.path.join('..', 'data', 'adv', 'cifar10', 'test', 'label.npy')
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
        adv_test_x = np.load(adv_test_path).reshape((-1, 32, 32, 3))
        adv_test_x = (adv_test_x.astype('float32') - mean) / std
        adv_test_y = np.load(adv_label_test_path)

    clean_data = (clean_x, clean_y)
    defect_data = (adv_x, adv_y)

    test_data = {
        'x_test': x_test,
        'y_test': y_test,
        'adv_x_test': adv_test_x,
        'adv_y_test': adv_test_y,
    }

    print(f'Training data sizes: clean={len(clean_x)}, adv={len(adv_x)}')
    print(f'Test data sizes:     clean={len(x_test)}, adv={len(adv_test_x)}')
    print()

    return clean_data, defect_data, test_data


# ============================================================
# Main evaluation framework
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified adversarial repair evaluation framework.')
    parser.add_argument('model_name', type=str, help='Model name: mnist, cifar10')
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('methods', type=str,
                        help='Method name, comma-separated list, or "all"')
    parser.add_argument('epsilon', type=float, nargs='?', default=0.1,
                        help='Adversarial epsilon (default: 0.1)')
    parser.add_argument('mode', type=str, nargs='?', default='auto',
                        help='Repair mode (auto, neuron, full)')
    args = parser.parse_args()

    model_name = args.model_name
    seed = args.seed
    methods_arg = args.methods
    epsilon = args.epsilon
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
    if model_name == 'mnist':
        model_path = f'../model/{model_name}_adv.h5'
    else:
        model_path = f'../model/{model_name}_adv.h5'
    model = load_model(model_path, compile=False)
    # print(model.summary())
    # ---- Save original weights ----
    original_weights = copy.deepcopy(model.get_weights())

    # ---- Prepare data ----
    clean_data, defect_data, test_data = prepare_data(model_name, epsilon)

    # ---- Evaluate BEFORE repair ----
    print('=' * 60)
    print('Initial model evaluation (before repair)')
    print('=' * 60)
    init_metrics = evaluate_model(model, test_data)
    for k, v in init_metrics.items():
        print(f'  {k}: {v:.6f}')
    print()

    # ---- Run each method ----
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
        method_kwargs['epsilon'] = epsilon
        method_kwargs['loc_cache_path'] = os.path.join(
            'loc_result', model_name, str(seed), f'eps{epsilon}',
            'loc_neuron_output.json')

        repaired_model, repair_stats = method_info['fn'](
            model, clean_data, defect_data, seed, model_name, method_kwargs)

        # ---- Load coefficients for CARE/CCBR ----
        coeff_data = None
        if method_name in ('care', 'ccbr'):
            coeff_dir = os.path.join('loc_result', model_name, str(seed),
                                     f'eps{epsilon}', 'defect_repair')
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
                               f'eps{epsilon}', 'defect_repair', method_name)
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
            'epsilon': epsilon,
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
        for metric in ['clean_acc', 'adv_acc']:
            row = f'{metric:<20} {init_metrics[metric]:>12.6f}'
            for mn in method_names:
                row += f' {all_results[mn][metric]:>14.6f}'
            print(row)
        print('=' * 80)


if __name__ == '__main__':
    main()