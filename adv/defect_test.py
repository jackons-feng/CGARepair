"""
Test adversarial repair results on test set.

Auto-discovers weights_final_*.h5 and coefficients_*.json files
under the result directory.

For CARE and CCBR methods, loads coefficients and evaluates using
split-model approach (scaling neuron outputs of the repair layer).

Usage:
  python defect_test.py <model_name> <seed> [epsilon]
  python defect_test.py cifar10 42 0.1
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

from repair_config import sess_run

BATCH_SIZE_TEST = 1024


def value_accuracy(model, images, labels, batch_size=BATCH_SIZE_TEST):
    output = sess_run(model, model.output, images, batch_size=batch_size)
    pred_labels = np.argmax(output, axis=-1)
    return float(np.mean(pred_labels == labels))


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

    Args:
        model: Keras model
        coefficient_data: dict with 'repair_neurons' list
        x_data: input data
        batch_size: batch size for evaluation

    Returns:
        model predictions (logits)
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
    pred_labels = np.argmax(output, axis=-1)
    return float(np.mean(pred_labels == labels))


def main():
    if len(sys.argv) < 3:
        print('Usage: python defect_test.py <model_name> <seed> [epsilon]')
        sys.exit(1)

    model_name = sys.argv[1]
    seed = int(sys.argv[2])
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    t0 = time.time()

    model_path = f'../model/{model_name}_adv.h5'

    result_dir = os.path.join('loc_result', model_name, str(seed),
                              f'eps{epsilon}', 'defect_repair')

    if not os.path.exists(result_dir):
        print(f'Error: result directory not found: {result_dir}')
        sys.exit(1)

    # Collect weight files and coefficient files
    weight_files = []
    for root, _, files in os.walk(result_dir):
        for f in files:
            if f.startswith('weights_final_') and f.endswith('.h5'):
                weight_files.append(os.path.join(root, f))
    weight_files = sorted(weight_files)
    coeff_files = []
    for root, _, files in os.walk(result_dir):
        for f in files:
            if f.startswith('coefficients_') and f.endswith('.json'):
                coeff_files.append(os.path.join(root, f))
    coeff_files = sorted(coeff_files)

    if len(weight_files) == 0 and len(coeff_files) == 0:
        print(f'Error: no weights_final_*.h5 or coefficients_*.json '
              f'files found in: {result_dir}')
        sys.exit(1)

    print(f'Testing Adversarial Repair Results on TEST set')
    print(f'  Model: {model_name}')
    print(f'  Epsilon: {epsilon}')
    print(f'  Result dir: {result_dir}')
    print(f'  Found {len(weight_files)} weight file(s), '
          f'{len(coeff_files)} coefficient file(s)')

    model = load_model(model_path, compile=False)
    original_weights = [w.copy() for w in model.get_weights()]

    # Load test data
    if model_name == 'mnist':
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = np.expand_dims(x_test.astype('float32') / 255, -1)
        adv_test_path = os.path.join('..', 'data', 'adv', 'mnist', 'test',
                                     f'epsilon{epsilon}.txt')
        adv_label_test_path = os.path.join('..', 'data', 'adv', 'mnist',
                                           'test', 'label.txt')
        with open(adv_test_path, 'r') as f:
            adv_test_x = np.array(
                [np.array(line.strip().split(','), dtype=float).reshape(
                    28, 28, 1) for line in f.readlines()], dtype='float32')
        with open(adv_label_test_path, 'r') as f:
            adv_test_y = np.array(
                [int(line.strip()) for line in f.readlines()], dtype=int)
    else:
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_test = y_test.flatten()
        x_test = x_test.astype('float32') / 255.0
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
        x_test = (x_test - mean) / std
        adv_test_path = os.path.join('..', 'data', 'adv', 'cifar10', 'test',
                                     f'epsilon{epsilon}.npy')
        adv_label_test_path = os.path.join('..', 'data', 'adv', 'cifar10',
                                           'test', 'label.npy')
        adv_test_x = np.load(adv_test_path).reshape((-1, 32, 32, 3))
        adv_test_x = (adv_test_x.astype('float32') - mean) / std
        adv_test_y = np.load(adv_label_test_path)

    # Evaluate original
    clean_acc_o = value_accuracy(model, x_test, y_test)
    adv_acc_o = value_accuracy(model, adv_test_x, adv_test_y)

    print(f'\n{"="*60}')
    print(f'{"Sub-Method":<20} {"clean_acc":>14} {"adv_acc":>14}')
    print(f'{"-"*60}')
    print(f'{"(original)":<20} {clean_acc_o:>14.6f} {adv_acc_o:>14.6f}')

    results = []

    # ---- Evaluate weight-based methods ----
    for wf in weight_files:
        sub_method = parse_sub_method(wf) or 'final'
        weights_path = wf
        model.set_weights(original_weights)
        model.load_weights(weights_path)

        ca = value_accuracy(model, x_test, y_test)
        aa = value_accuracy(model, adv_test_x, adv_test_y)

        print(f'{sub_method:<20} {ca:>14.6f} {aa:>14.6f}')
        results.append({
            'sub_method': sub_method,
            'clean_acc': ca,
            'adv_acc': aa,
            'adv_error': 1.0 - aa,
            'artifact_path': weights_path,
            'artifact_type': 'weights',
        })

    # ---- Evaluate coefficient-based methods (CARE / CCBR) ----
    for cf in coeff_files:
        sub_method = parse_coeff_method(cf) or 'coeff'
        coeff_path = cf
        with open(coeff_path, 'r', encoding='utf-8') as f:
            coefficient_data = json.load(f)

        model.set_weights(original_weights)
        ca = value_accuracy_with_coeff(model, coefficient_data, x_test, y_test)
        aa = value_accuracy_with_coeff(
            model, coefficient_data, adv_test_x, adv_test_y)

        print(f'{sub_method:<20} {ca:>14.6f} {aa:>14.6f}')
        results.append({
            'sub_method': sub_method,
            'clean_acc': ca,
            'adv_acc': aa,
            'adv_error': 1.0 - aa,
            'artifact_path': coeff_path,
            'artifact_type': 'coefficients',
        })

    print(f'{"="*60}')
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.2f}s')

    # ---- Save result summary in the same style as repair result text files ----
    if results:
        best = min(results, key=lambda r: r['adv_error'])
        out_path = os.path.join(result_dir, 'best_solution_min_rate.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f'model_name={model_name}\n')
            f.write(f'seed={seed}\n')
            f.write(f'epsilon={epsilon}\n')
            f.write(f'original_clean_acc={clean_acc_o:.10f}\n')
            f.write(f'original_adv_acc={adv_acc_o:.10f}\n')
            f.write(f'original_adv_error={1.0 - adv_acc_o:.10f}\n')
            f.write(f'best_method={best["sub_method"]}\n')
            f.write(f'best_artifact_type={best["artifact_type"]}\n')
            f.write(f'best_artifact_path={best["artifact_path"]}\n')
            f.write(f'best_test_clean_acc={best["clean_acc"]:.10f}\n')
            f.write(f'best_test_adv_acc={best["adv_acc"]:.10f}\n')
            f.write(f'best_test_adv_error={best["adv_error"]:.10f}\n')
            f.write('\n')
            f.write('all_results:\n')
            for r in results:
                f.write(
                    f'{r["sub_method"]}: '
                    f'clean_acc={r["clean_acc"]:.10f}, '
                    f'adv_acc={r["adv_acc"]:.10f}, '
                    f'adv_error={r["adv_error"]:.10f}, '
                    f'artifact={r["artifact_path"]}\n'
                )
        print(f'Saved best result summary to: {out_path}')


if __name__ == '__main__':
    main()
