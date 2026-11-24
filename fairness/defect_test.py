"""
Test fairness repair results on test set.

Auto-discovers weights_final_*.h5 and coefficients_*.json files under
the result directory.

For CARE and CCBR methods, loads coefficients and evaluates using
split-model approach (scaling neuron outputs of the repair layer).

Usage:
  python defect_test.py <model_name> <seed>
  python defect_test.py bank 42
  python defect_test.py census 42
"""

import os
import sys
import re
import json
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

from repair_config import (
    load_fairness_data,
    sess_run,
    labels_from_output,
    binary_group_accuracy_gap,
)

BATCH_SIZE_TEST = 1024


def value_accuracy(model, images, labels):
    output = sess_run(model, model.output, images, batch_size=BATCH_SIZE_TEST)
    pred_labels = labels_from_output(output)
    labels = np.asarray(labels).reshape(-1).astype(int)
    return float(np.mean(pred_labels == labels))


def value_fairness_gap(model, x, y, group_labels=None):
    output = sess_run(model, model.output, x, batch_size=BATCH_SIZE_TEST)
    pred = labels_from_output(output)
    groups = y if group_labels is None else group_labels
    return binary_group_accuracy_gap(pred, y, groups)


def parse_sub_method(weights_filename):
    base = os.path.basename(weights_filename)
    m = re.match(r'weights_final_(.+)\.h5$', base)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(weights_filename))
    return parent if parent else None


def parse_coeff_method(coeff_filename):
    base = os.path.basename(coeff_filename)
    m = re.match(r'coefficients_(.+)\.json$', base)
    if m:
        return m.group(1)
    parent = os.path.basename(os.path.dirname(coeff_filename))
    return parent if parent else None


def evaluate_with_coefficients(model, coefficient_data, x_data, batch_size=BATCH_SIZE_TEST):
    """
    Evaluate a model with CARE/CCBR coefficients applied.

    Splits the model at the repair layer, scales the neuron outputs
    by (1 + r), and passes through the rest of the model.
    """
    neurons = coefficient_data['repair_neurons']
    if not neurons:
        return sess_run(model, model.output, x_data, batch_size)

    repair_layer_idx = neurons[0]['layer_idx']

    # Split model at repair layer
    model1 = Model(inputs=model.input,
                   outputs=model.layers[repair_layer_idx].output)
    model2_input = Input(model.layers[repair_layer_idx + 1].input_shape[1:])
    m2 = model2_input
    for layer in model.layers[repair_layer_idx + 1:]:
        m2 = layer(m2)
    model2 = Model(inputs=model2_input, outputs=m2)

    # Evaluate in batches
    outputs = []
    n_samples = len(x_data)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        x_batch = x_data[start:end]

        hidden = model1.predict(x_batch, verbose=0)
        for n in neurons:
            hidden[..., n['neuron_idx']] *= n['actual_scale']
        output = model2.predict(hidden, verbose=0)
        outputs.append(output)

    return np.concatenate(outputs, axis=0)


def value_accuracy_with_coeff(model, coefficient_data, images, labels,
                               batch_size=BATCH_SIZE_TEST):
    output = evaluate_with_coefficients(
        model, coefficient_data, images, batch_size=batch_size)
    pred_labels = labels_from_output(output)
    labels = np.asarray(labels).reshape(-1).astype(int)
    return float(np.mean(pred_labels == labels))


def value_fairness_gap_with_coeff(model, coefficient_data, x, y, group_labels=None,
                                   batch_size=BATCH_SIZE_TEST):
    output = evaluate_with_coefficients(
        model, coefficient_data, x, batch_size=batch_size)
    pred = labels_from_output(output)
    groups = y if group_labels is None else group_labels
    return binary_group_accuracy_gap(pred, y, groups)


def main():
    if len(sys.argv) < 3:
        print('Usage: python defect_test.py <model_name> <seed>')
        print('Example:')
        print('  python defect_test.py bank 42')
        print('  python defect_test.py census 42')
        sys.exit(1)

    model_name = sys.argv[1].lower()
    if model_name not in ('bank', 'census'):
        print('Unsupported fairness dataset. Use bank or census.')
        sys.exit(1)
    seed = int(sys.argv[2])

    t0 = time.time()

    model_path = f'../model/{model_name}.h5'

    result_dir = os.path.join('loc_result', model_name, str(seed), 'defect_repair')

    if not os.path.exists(result_dir):
        print(f'Error: result directory not found: {result_dir}')
        sys.exit(1)

    # Auto-discover weight files and coefficient files recursively.
    weight_files = []
    coeff_files = []
    for root, _, files in os.walk(result_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.startswith('weights_final_') and f.endswith('.h5'):
                weight_files.append(full_path)
            elif f == 'weights_final.h5':
                weight_files.append(full_path)
            elif f.startswith('coefficients_') and f.endswith('.json'):
                coeff_files.append(full_path)
            elif f == 'coefficients.json':
                coeff_files.append(full_path)
    weight_files = sorted(weight_files)
    coeff_files = sorted(coeff_files)

    if len(weight_files) == 0 and len(coeff_files) == 0:
        print(f'Error: no weights_final_*.h5 or coefficients_*.json '
              f'files found in: {result_dir}')
        sys.exit(1)

    print(f'Testing Fairness Repair Results on TEST set')
    print(f'  Model: {model_name}')
    print(f'  Result dir: {result_dir}')
    print(f'  Found {len(weight_files)} weight file(s), '
          f'{len(coeff_files)} coefficient file(s)')

    model = load_model(model_path, compile=False)
    original_weights = [w.copy() for w in model.get_weights()]

    _, test_full = load_fairness_data(
        model_name, sample_size=None, seed=seed + 999, return_groups=True)
    test_x, test_y, test_g = test_full

    # Evaluate original
    clean_acc_o = value_accuracy(model, test_x, test_y)
    fairness_gap_o = value_fairness_gap(model, test_x, test_y, test_g)

    print(f'\n{"="*60}')
    print(f'{"Sub-Method":<20} {"clean_acc":>14} {"fairness_gap":>14}')
    print(f'{"-"*60}')
    print(f'{"(original)":<20} {clean_acc_o:>14.6f} {fairness_gap_o:>14.6f}')

    all_results = []

    # ---- Evaluate weight-based methods ----
    for wf in weight_files:
        sub_method = parse_sub_method(wf) or 'final'
        weights_path = wf
        model.set_weights(original_weights)
        model.load_weights(weights_path)

        ca = value_accuracy(model, test_x, test_y)
        fg = value_fairness_gap(model, test_x, test_y, test_g)

        print(f'{sub_method:<20} {ca:>14.6f} {fg:>14.6f}')

        all_results.append({
            'sub_method': sub_method,
            'artifact_path': weights_path,
            'artifact_type': 'weights',
            'clean_acc': ca,
            'fairness_gap': fg,
        })

        # Save per-sub-method result
        result_filename = f'test_results_{sub_method}.txt'
        result_path = os.path.join(result_dir, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'sub_method={sub_method}\n')
            f.write(f'seed={seed}\n')
            f.write(f'model_name={model_name}\n')
            f.write(f'weights_file={weights_path}\n')
            f.write('\n')
            f.write(f'clean_acc_o={clean_acc_o:.10f}\n')
            f.write(f'fairness_gap_o={fairness_gap_o:.10f}\n')
            f.write('\n')
            f.write(f'clean_acc={ca:.10f}\n')
            f.write(f'fairness_gap={fg:.10f}\n')

        print(f'  -> Saved: {result_filename}')

    # ---- Evaluate coefficient-based methods (CARE / CCBR) ----
    for cf in coeff_files:
        sub_method = parse_coeff_method(cf) or 'coeff'
        coeff_path = cf
        with open(coeff_path, 'r', encoding='utf-8') as f:
            coefficient_data = json.load(f)

        model.set_weights(original_weights)
        ca = value_accuracy_with_coeff(
            model, coefficient_data, test_x, test_y)
        fg = value_fairness_gap_with_coeff(
            model, coefficient_data, test_x, test_y, test_g)

        print(f'{sub_method:<20} {ca:>14.6f} {fg:>14.6f}')

        all_results.append({
            'sub_method': sub_method,
            'artifact_path': coeff_path,
            'artifact_type': 'coefficients',
            'clean_acc': ca,
            'fairness_gap': fg,
        })

        # Save per-sub-method result
        result_filename = f'test_results_{sub_method}.txt'
        result_path = os.path.join(result_dir, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'sub_method={sub_method}\n')
            f.write(f'seed={seed}\n')
            f.write(f'model_name={model_name}\n')
            f.write(f'coeff_file={coeff_path}\n')
            f.write('\n')
            f.write(f'clean_acc_o={clean_acc_o:.10f}\n')
            f.write(f'fairness_gap_o={fairness_gap_o:.10f}\n')
            f.write('\n')
            f.write(f'clean_acc={ca:.10f}\n')
            f.write(f'fairness_gap={fg:.10f}\n')

        print(f'  -> Saved: {result_filename}')

    print(f'{"="*60}')

    # Save summary CSV
    summary_path = os.path.join(result_dir, 'test_summary.csv')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('sub_method,clean_acc,fairness_gap\n')
        f.write(f'original,{clean_acc_o:.10f},{fairness_gap_o:.10f}\n')
        for r in all_results:
            f.write(f'{r["sub_method"]},{r["clean_acc"]:.10f},{r["fairness_gap"]:.10f}\n')

    print(f'Summary saved: {summary_path}')

    if all_results:
        best = min(all_results, key=lambda r: (r['fairness_gap'], -r['clean_acc']))
        best_path = os.path.join(result_dir, 'best_solution_min_rate.txt')
        with open(best_path, 'w', encoding='utf-8') as f:
            f.write(f'model_name={model_name}\n')
            f.write(f'seed={seed}\n')
            f.write(f'original_clean_acc={clean_acc_o:.10f}\n')
            f.write(f'original_fairness_gap={fairness_gap_o:.10f}\n')
            f.write(f'best_method={best["sub_method"]}\n')
            f.write(f'best_artifact_type={best["artifact_type"]}\n')
            f.write(f'best_artifact_path={best["artifact_path"]}\n')
            f.write(f'best_test_clean_acc={best["clean_acc"]:.10f}\n')
            f.write(f'best_test_fairness_gap={best["fairness_gap"]:.10f}\n')
            f.write('\n')
            f.write('all_results:\n')
            for r in all_results:
                f.write(
                    f'{r["sub_method"]}: '
                    f'clean_acc={r["clean_acc"]:.10f}, '
                    f'fairness_gap={r["fairness_gap"]:.10f}, '
                    f'artifact={r["artifact_path"]}\n'
                )
        print(f'Best result saved: {best_path}')
    print(f'Done in {time.time() - t0:.2f}s')


if __name__ == '__main__':
    main()

