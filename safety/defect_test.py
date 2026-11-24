"""
Test safety repair results on test set.

Auto-discovers weights_final_*.h5 and coefficients_*.json files under
the result directory.

For CARE and CCBR methods, loads coefficients and evaluates using
split-model approach (scaling neuron outputs of the repair layer).

Usage:
  python defect_test.py <model_name> <seed>
  python defect_test.py a29 42
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

from repair_config import load_safety_data, sess_run

BATCH_SIZE_TEST = 1024


def value_safety(model, images, model_name):
    output = sess_run(model, model.output, images, batch_size=BATCH_SIZE_TEST)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((np.sum(pred == 3) + np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((len(output) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output))
    return float((np.sum(pred != 0)) / len(output))


def value_drawndown(model, images, model_name):
    output = sess_run(model, model.output, images, batch_size=BATCH_SIZE_TEST)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((len(output) - np.sum(pred == 3) - np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((np.sum(pred == 0) + np.sum(pred == 1)) / len(output))
    return float(np.sum(pred == 0) / len(output))


def parse_sub_method(weights_filename):
    m = re.match(r'weights_final_(.+)\.h5$', weights_filename)
    return m.group(1) if m else None


def parse_coeff_method(coeff_filename):
    m = re.match(r'coefficients_(.+)\.json$', coeff_filename)
    return m.group(1) if m else None


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


def value_safety_with_coeff(model, coefficient_data, images, model_name,
                             batch_size=BATCH_SIZE_TEST):
    output = evaluate_with_coefficients(
        model, coefficient_data, images, batch_size=batch_size)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((np.sum(pred == 3) + np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((len(output) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output))
    return float((np.sum(pred != 0)) / len(output))


def value_drawndown_with_coeff(model, coefficient_data, images, model_name,
                                batch_size=BATCH_SIZE_TEST):
    output = evaluate_with_coefficients(
        model, coefficient_data, images, batch_size=batch_size)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        return float((len(output) - np.sum(pred == 3) - np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        return float((np.sum(pred == 0) + np.sum(pred == 1)) / len(output))
    return float(np.sum(pred == 0) / len(output))


def main():
    if len(sys.argv) < 3:
        print('Usage: python defect_test.py <model_name> <seed>')
        print('Example:')
        print('  python defect_test.py a29 42')
        sys.exit(1)

    model_name = sys.argv[1]
    seed = int(sys.argv[2])

    t0 = time.time()

    model_path = f'../model/{model_name}.h5'

    result_dir = os.path.join('loc_result', model_name, str(seed), 'defect_repair')

    if not os.path.exists(result_dir):
        print(f'Error: result directory not found: {result_dir}')
        sys.exit(1)

    # Auto-discover weight files and coefficient files
    weight_files = sorted(
        f for f in os.listdir(result_dir)
        if f.startswith('weights_final_') and f.endswith('.h5')
    )
    coeff_files = sorted(
        f for f in os.listdir(result_dir)
        if f.startswith('coefficients_') and f.endswith('.json')
    )

    if len(weight_files) == 0 and len(coeff_files) == 0:
        print(f'Error: no weights_final_*.h5 or coefficients_*.json '
              f'files found in: {result_dir}')
        sys.exit(1)

    print(f'Testing Safety Repair Results on TEST set')
    print(f'  Model: {model_name}')
    print(f'  Result dir: {result_dir}')
    print(f'  Found {len(weight_files)} weight file(s), '
          f'{len(coeff_files)} coefficient file(s)')

    model = load_model(model_path, compile=False)

    ct_x, dr_x = load_safety_data(model_name)

    # Evaluate original
    safety_o = value_safety(model, ct_x, model_name)
    drawndown_o = value_drawndown(model, dr_x, model_name)

    print(f'\n{"="*60}')
    print(f'{"Sub-Method":<20} {"safety":>14} {"drawndown":>14}')
    print(f'{"-"*60}')
    print(f'{"(original)":<20} {safety_o:>14.6f} {drawndown_o:>14.6f}')

    all_results = []

    # ---- Evaluate weight-based methods ----
    for wf in weight_files:
        sub_method = parse_sub_method(wf) or 'final'
        weights_path = os.path.join(result_dir, wf)
        model.load_weights(weights_path)

        s = value_safety(model, ct_x, model_name)
        d = value_drawndown(model, dr_x, model_name)

        print(f'{sub_method:<20} {s:>14.6f} {d:>14.6f}')

        all_results.append((sub_method, wf, s, d))

        # Save per-sub-method result
        result_filename = f'test_results_{sub_method}.txt'
        result_path = os.path.join(result_dir, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'sub_method={sub_method}\n')
            f.write(f'seed={seed}\n')
            f.write(f'model_name={model_name}\n')
            f.write(f'weights_file={wf}\n')
            f.write('\n')
            f.write(f'safety_o={safety_o:.10f}\n')
            f.write(f'drawndown_o={drawndown_o:.10f}\n')
            f.write('\n')
            f.write(f'safety={s:.10f}\n')
            f.write(f'drawndown={d:.10f}\n')

        print(f'  -> Saved: {result_filename}')

    # ---- Evaluate coefficient-based methods (CARE / CCBR) ----
    for cf in coeff_files:
        sub_method = parse_coeff_method(cf) or 'coeff'
        coeff_path = os.path.join(result_dir, cf)
        with open(coeff_path, 'r', encoding='utf-8') as f:
            coefficient_data = json.load(f)

        s = value_safety_with_coeff(
            model, coefficient_data, ct_x, model_name)
        d = value_drawndown_with_coeff(
            model, coefficient_data, dr_x, model_name)

        print(f'{sub_method:<20} {s:>14.6f} {d:>14.6f}')

        all_results.append((sub_method, cf, s, d))

        # Save per-sub-method result
        result_filename = f'test_results_{sub_method}.txt'
        result_path = os.path.join(result_dir, result_filename)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'sub_method={sub_method}\n')
            f.write(f'seed={seed}\n')
            f.write(f'model_name={model_name}\n')
            f.write(f'coeff_file={cf}\n')
            f.write('\n')
            f.write(f'safety_o={safety_o:.10f}\n')
            f.write(f'drawndown_o={drawndown_o:.10f}\n')
            f.write('\n')
            f.write(f'safety={s:.10f}\n')
            f.write(f'drawndown={d:.10f}\n')

        print(f'  -> Saved: {result_filename}')

    print(f'{"="*60}')

    # Save summary CSV
    summary_path = os.path.join(result_dir, 'test_summary.csv')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('sub_method,safety,drawndown\n')
        for sub_method, _wf, s, d in all_results:
            f.write(f'{sub_method},{s:.10f},{d:.10f}\n')
        f.write(f'original,,{safety_o:.10f},{drawndown_o:.10f}\n')

    print(f'Summary saved: {summary_path}')
    print(f'Done in {time.time() - t0:.2f}s')


if __name__ == '__main__':
    main()