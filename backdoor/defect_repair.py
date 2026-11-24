"""
Unified backdoor repair evaluation framework.

Design:
- Loads model and backdoor data once
- Dispatches to different repair methods via a registry
- Evaluates all methods on the test set with consistent metrics
- Saves results (weights, stats) organized by method name

Single defect type: backdoor
Supported models: gtsrb, fashion, gtsrb_resnet
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
    DATASET_CONFIG,
    load_clean_data,
    get_perturbed_input,
    value_accuracy,
    value_backdoor,
    sess_run,
)


# ============================================================
# Method implementations
# ============================================================

def run_multirepair(model, clean_data, defect_data, seed, model_name,
                    method_kwargs):
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
    from methods.care_repair import care_repair_single
    return care_repair_single(
        model, defect_data, clean_data, 'backdoor',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_ccbr(model, clean_data, defect_data, seed, model_name, method_kwargs):
    from methods.ccbr import ccbr_single
    return ccbr_single(
        model, defect_data, clean_data, 'backdoor',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_flshap_cma(model, clean_data, defect_data, seed, model_name, method_kwargs):
    from methods.flshap_cma import flshap_cma_single
    return flshap_cma_single(
        model, defect_data, clean_data, 'backdoor',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_nsrepair(model, clean_data, defect_data, seed, model_name, method_kwargs):
    from methods.nsrepair import nsrepair_single
    return nsrepair_single(
        model, defect_data, clean_data, 'backdoor',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_neuropatch(model, clean_data, defect_data, seed, model_name, method_kwargs):
    from methods.neuropatch import neuropatch_single
    return neuropatch_single(
        model, defect_data, clean_data, 'backdoor',
        method_kwargs['loc_cache_path'], seed, model_name, 0.0)


def run_birdnn(model, clean_data, defect_data, seed, model_name, method_kwargs):
    from methods.birdnn import birdnn_single
    return birdnn_single(
        model, defect_data, clean_data, 'backdoor',
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
    'neuropatch': {
        'fn': run_neuropatch,
        'description': 'Neuropatch: activation difference + neuron masking (flip/zero)',
        'default_kwargs': {},
    },
    'birdnn': {
        'fn': run_birdnn,
        'description': 'BIRDNN: behavior-informed neuron localization + PSO (local)',
        'default_kwargs': {},
    },
}


# ============================================================
# Metric helpers
# ============================================================

def evaluate_with_coefficients(model, coefficient_data, x_data, batch_size=1024):
    """Evaluate model with CARE/CCBR coefficients applied via split-model."""
    from loc_neuron_output import split_model_at_layer

    neurons = coefficient_data.get('repair_neurons', [])
    if not neurons:
        return sess_run(model, model.output, x_data, batch_size)

    repair_layer_idx = neurons[0]['layer_idx']
    model1, model2, skip_indices = split_model_at_layer(model, repair_layer_idx)

    outputs = []
    n_samples = len(x_data)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        x_batch = x_data[start:end]
        hidden = model1.predict(x_batch, verbose=0)
        for n in neurons:
            hidden[..., n['neuron_idx']] *= n['actual_scale']

        # Build input list for model2 (handles ResNet skip connections)
        if isinstance(model2.input, list):
            model2_inputs = [hidden]
            for skip_idx in skip_indices:
                skip_model = KerasModel(
                    inputs=model.input,
                    outputs=model.layers[skip_idx].output)
                skip_out = skip_model.predict(x_batch, verbose=0)
                model2_inputs.append(skip_out)
            output = model2.predict(model2_inputs, verbose=0)
        else:
            output = model2.predict(hidden, verbose=0)
        outputs.append(output)
    return np.concatenate(outputs, axis=0)


def evaluate_model(model, test_data, coefficient_data=None):
    """Evaluate model on test metrics. If coefficient_data is provided,
    uses split-model evaluation with coefficient scaling (for CARE/CCBR)."""
    if coefficient_data is not None and coefficient_data.get('repair_neurons'):
        clean_out = evaluate_with_coefficients(model, coefficient_data,
                                               test_data['x_test'])
        clean_acc = float(np.mean(np.argmax(clean_out, axis=-1) == test_data['y_test']))

        bd_out = evaluate_with_coefficients(model, coefficient_data,
                                            test_data['backdoor_x_test'])
        bd_pred = np.argmax(bd_out, axis=-1)
        target_label = test_data['target_label']
        backdoor_rate = float(
            (np.sum(bd_pred == target_label)
             - np.sum((test_data['backdoor_y_test'] == target_label)
                      & (bd_pred == target_label)))
            / len(test_data['backdoor_x_test']))

        return {'clean_acc': clean_acc, 'backdoor_rate': backdoor_rate}

    return {
        'clean_acc': float(value_accuracy(model, test_data['x_test'], test_data['y_test'])),
        'backdoor_rate': float(value_backdoor(model, test_data['backdoor_x_test'],
                                              test_data['backdoor_y_test'],
                                              test_data['target_label'])),
    }


# ============================================================
# Data preparation
# ============================================================

def prepare_data(model_name, resnet_flag=False):
    print('=' * 60)
    print(f'Preparing {model_name} backdoor data ...')
    print('=' * 60)

    cfg = DATASET_CONFIG[model_name]
    target_label = cfg['target_label']
    input_shape = cfg['input_shape']

    x_clean, y_clean = load_clean_data(model_name)

    # Generate backdoor data
    x_bd = get_perturbed_input(x_clean, model_name, target_label, input_shape)
    y_bd = y_clean
    bd_mask = (y_bd != target_label)
    n_bd_before = len(x_bd)
    x_bd, y_bd = x_bd[bd_mask], y_bd[bd_mask]
    print(f'Backdoor filtering: {np.sum(bd_mask)}/{n_bd_before} samples kept')

    clean_data = (x_clean, y_clean)
    defect_data = (x_bd, y_bd)

    # Test data
    if model_name in ('gtsrb', 'gtsrb_resnet'):
        import pickle
        data_file = os.path.join('..', 'data', 'gtsrb', 'test_data.pkl')
        with open(data_file, 'rb') as f:
            test_data_pkl = pickle.load(f)
            x_test = np.transpose(test_data_pkl['data'], (0, 2, 3, 1))
            import tensorflow.python.keras.backend as K
            x_test = K.get_session().run(tf.image.resize(x_test.astype('float32'), (32, 32)))
            x_test = x_test * 255.0
            y_test = np.array(test_data_pkl['label'], dtype=int)
    else:
        (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_test = np.expand_dims(x_test.astype('float32') / 255, -1)

    backdoor_x_test = get_perturbed_input(x_test, model_name, target_label, input_shape)
    backdoor_y_test = y_test

    test_data = {
        'x_test': x_test,
        'y_test': y_test,
        'backdoor_x_test': backdoor_x_test,
        'backdoor_y_test': backdoor_y_test,
        'target_label': target_label,
    }

    print(f'Training data sizes: clean={len(x_clean)}, bd={len(x_bd)}')
    print(f'Test data sizes:     clean={len(x_test)}, bd={len(backdoor_x_test)}')
    print()

    return clean_data, defect_data, test_data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified backdoor repair evaluation framework.')
    parser.add_argument('model_name', type=str, help='Model name: fashion, gtsrb, gtsrb_resnet')
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('methods', type=str, help='Method name, comma-separated, or "all"')
    parser.add_argument('mode', type=str, nargs='?', default='auto',
                        help='Repair mode (auto, neuron, full)')
    parser.add_argument('--resnet', action='store_true', help='Use ResNet model for gtsrb')
    args = parser.parse_args()

    model_name = args.model_name
    seed = args.seed
    methods_arg = args.methods
    mode = args.mode
    resnet_flag = args.resnet

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

    # ---- Resolve model ----
    if resnet_flag and model_name == 'gtsrb':
        model_path = f'../model/{model_name}_resnet.h5'
        name_tag = 'gtsrb_resnet'
    else:
        model_path = f'../model/{model_name}.h5'
        name_tag = model_name

    model = load_model(model_path, compile=False)
    original_weights = copy.deepcopy(model.get_weights())

    clean_data, defect_data, test_data = prepare_data(model_name, resnet_flag)

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
            'loc_result', name_tag, str(seed), 'loc_neuron_output.json')

        repaired_model, repair_stats = method_info['fn'](
            model, clean_data, defect_data, seed, name_tag, method_kwargs)

        # ---- Load coefficients for CARE/CCBR ----
        coeff_data = None
        if method_name in ('care', 'ccbr', 'birdnn'):
            coeff_dir = os.path.join('loc_result', name_tag, str(seed),
                                     'defect_repair')
            coeff_path = os.path.join(coeff_dir, f'coefficients_{method_name}.json')
            if os.path.exists(coeff_path):
                with open(coeff_path, 'r', encoding='utf-8') as f:
                    coeff_data = json.load(f)
                print(f'  Loaded coefficients from: {coeff_path}')

        print('=' * 60)
        print(f'Evaluation: {method_name}')
        print('=' * 60)
        final_metrics = evaluate_model(repaired_model, test_data, coefficient_data=coeff_data)
        for k, v in final_metrics.items():
            init_v = init_metrics[k]
            delta = v - init_v
            print(f'  {k}: {v:.6f} (delta: {delta:+.6f})')
        print()

        out_dir = os.path.join('loc_result', name_tag, str(seed),
                               'defect_repair', method_name)
        os.makedirs(out_dir, exist_ok=True)

        if method_name in ('care', 'ccbr', 'birdnn'):
            # Coefficients are saved by the repair function itself
            print(f'  Coefficients saved by repair function')
        else:
            weights_filename = f'weights_final_{method_name}.h5'
            repaired_model.save_weights(os.path.join(out_dir, weights_filename))

        run_stats = {
            'seed': int(seed),
            'method': method_name,
            'mode': mode,
            'model_name': name_tag,
            'initial_metrics': init_metrics,
            'final_metrics': final_metrics,
            'repair_stats': repair_stats,
        }
        with open(os.path.join(out_dir, 'run_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(run_stats, f, indent=2)

        print(f'Saved to: {out_dir}')
        all_results[method_name] = final_metrics

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
        for metric in ['clean_acc', 'backdoor_rate']:
            row = f'{metric:<20} {init_metrics[metric]:>12.6f}'
            for mn in method_names:
                row += f' {all_results[mn][metric]:>14.6f}'
            print(row)
        print('=' * 80)


if __name__ == '__main__':
    main()