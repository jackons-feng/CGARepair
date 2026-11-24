"""
Test backdoor repair results on test set.

Auto-discovers all weights_final_*.h5 and coefficients_*.json files under
the result directory and evaluates each sub-method.

For CARE and CCBR methods, loads coefficients and evaluates using
split-model approach (scaling neuron outputs of the repair layer).

Metrics:
- backdoor_rate (lower is better)
- clean_acc (higher is better)

Usage:
  python defect_test.py <model_name> <seed> [resnet]
  python defect_test.py fashion 42
  python defect_test.py gtsrb 42 --resnet
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
    DATASET_CONFIG,
    get_perturbed_input,
    sess_run,
)

BATCH_SIZE_TEST = 1024


def value_accuracy(model, images, labels):
    output = sess_run(model, model.output, images, batch_size=BATCH_SIZE_TEST)
    pred_labels = np.argmax(output, axis=-1)
    return float(np.mean(pred_labels == labels))


def value_backdoor(model, images_p, labels_clean, target_label):
    pred_p = np.argmax(sess_run(model, model.output, images_p, batch_size=BATCH_SIZE_TEST), axis=-1)
    return float((np.sum(pred_p == target_label) - np.sum((labels_clean == target_label) & (pred_p == target_label))) / len(images_p))


def load_test_data(model_name):
    """Load test split data for the given model."""
    if model_name in ('gtsrb', 'gtsrb_resnet'):
        import pickle
        data_file = os.path.join('..', 'data', 'gtsrb', 'test_data.pkl')
        with open(data_file, 'rb') as f:
            test_data = pickle.load(f)
            x_test = np.transpose(test_data['data'], (0, 2, 3, 1))
            x_test = K.get_session().run(tf.image.resize(x_test.astype('float32'), (32, 32)))
            x_test = x_test * 255.0
            y_test = np.array(test_data['label'], dtype=int)
        return x_test, y_test

    if model_name == 'fashion':
        (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_test = np.expand_dims(x_test.astype('float32') / 255, -1)
        return x_test, y_test

    raise ValueError(f'Unsupported model_name: {model_name}')


def parse_sub_method(weights_filename):
    """Extract sub-method name from 'weights_final_<name>.h5'."""
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
    pred_labels = np.argmax(output, axis=-1)
    return float(np.mean(pred_labels == labels))


def value_backdoor_with_coeff(model, coefficient_data, images_p,
                               labels_clean, target_label,
                               batch_size=BATCH_SIZE_TEST):
    output = evaluate_with_coefficients(
        model, coefficient_data, images_p, batch_size=batch_size)
    pred_p = np.argmax(output, axis=-1)
    return float((np.sum(pred_p == target_label) - np.sum((labels_clean == target_label) & (pred_p == target_label))) / len(images_p))


def print_sep(ch='=', n=88):
    print(ch * n)


def main():
    if len(sys.argv) < 3:
        print('Usage: python defect_test.py <model_name> <seed> [--resnet]')
        print('  model_name: gtsrb, fashion, gtsrb_resnet')
        print('Example:')
        print('  python defect_test.py fashion 42')
        print('  python defect_test.py gtsrb 42 --resnet')
        sys.exit(1)

    model_name = sys.argv[1]
    seed = int(sys.argv[2])
    resnet_flag = '--resnet' in sys.argv

    t0 = time.time()

    # Resolve model path and tag
    if resnet_flag and model_name == 'gtsrb':
        model_path = f'../model/{model_name}_resnet.h5'
        name_tag = 'gtsrb_resnet'
        cfg_key = 'gtsrb_resnet'
    elif model_name == 'gtsrb_resnet':
        model_path = '../model/gtsrb_resnet.h5'
        name_tag = 'gtsrb_resnet'
        cfg_key = 'gtsrb_resnet'
    else:
        model_path = f'../model/{model_name}.h5'
        name_tag = model_name
        cfg_key = model_name

    if cfg_key not in DATASET_CONFIG:
        print(f'Unknown model: {cfg_key}. Available: {list(DATASET_CONFIG.keys())}')
        sys.exit(1)

    result_dir = os.path.join('loc_result', name_tag, str(seed), 'defect_repair')

    if not os.path.exists(result_dir):
        print(f'Error: result directory not found: {result_dir}')
        sys.exit(1)

    # Auto-discover weight files and coefficient files recursively.
    # Repair scripts may save results under method subdirectories, e.g.
    # defect_repair/flshap_cma/weights_final_flshap_cma.h5.
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
        print(f'  Available files: {os.listdir(result_dir)}')
        sys.exit(1)

    print_sep()
    print(f'Testing Backdoor Repair Results on TEST set')
    print(f'  Model:           {model_name} ({name_tag})')
    print(f'  Result dir:      {result_dir}')
    print(f'  Found {len(weight_files)} weight file(s), '
          f'{len(coeff_files)} coefficient file(s)')
    for wf in weight_files:
        sub = parse_sub_method(wf) or 'final'
        print(f'    [weight] -> {sub}')
    for cf in coeff_files:
        sub = parse_coeff_method(cf) or 'coeff'
        print(f'    [coeff]  -> {sub}')
    print_sep()

    model = load_model(model_path, compile=False)
    original_weights = [w.copy() for w in model.get_weights()]

    test_x, test_y = load_test_data(model_name)

    cfg = DATASET_CONFIG[cfg_key]
    target_label = cfg['target_label']
    input_shape = cfg['input_shape']

    test_x_p = get_perturbed_input(test_x, model_name, target_label, input_shape)

    # Evaluate original model (once)
    backdoor_o = value_backdoor(model, test_x_p, test_y, target_label)
    clean_acc_o = value_accuracy(model, test_x, test_y)

    # ---- Summary table header ----
    print(f'\n{"="*60}')
    print(f'{"Sub-Method":<20} {"backdoor_rate":>14} {"clean_acc":>14}')
    print(f'{"-"*60}')
    print(f'{"(original)":<20} {backdoor_o:>14.6f} {clean_acc_o:>14.6f}')

    all_results = []

    # ---- Evaluate weight-based methods ----
    for wf in weight_files:
        sub_method = parse_sub_method(wf) or 'final'
        weights_path = wf

        model.set_weights(original_weights)
        model.load_weights(weights_path)

        br = value_backdoor(model, test_x_p, test_y, target_label)
        ca = value_accuracy(model, test_x, test_y)

        delta_br = br - backdoor_o
        delta_ca = ca - clean_acc_o

        print(f'{sub_method:<20} {br:>14.6f} {ca:>14.6f}')

        all_results.append({
            'sub_method': sub_method,
            'artifact_path': weights_path,
            'artifact_type': 'weights',
            'backdoor_rate': br,
            'clean_acc': ca,
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
            f.write(f'backdoor_rate_o={backdoor_o:.10f}\n')
            f.write(f'clean_acc_o={clean_acc_o:.10f}\n')
            f.write('\n')
            f.write(f'backdoor_rate={br:.10f}  (delta: {delta_br:+.10f})\n')
            f.write(f'clean_acc={ca:.10f}  (delta: {delta_ca:+.10f})\n')

        print(f'  -> Saved: {result_filename}')

    # ---- Evaluate coefficient-based methods (CARE / CCBR) ----
    for cf in coeff_files:
        sub_method = parse_coeff_method(cf) or 'coeff'
        coeff_path = cf
        with open(coeff_path, 'r', encoding='utf-8') as f:
            coefficient_data = json.load(f)

        model.set_weights(original_weights)
        br = value_backdoor_with_coeff(
            model, coefficient_data, test_x_p, test_y, target_label)
        ca = value_accuracy_with_coeff(
            model, coefficient_data, test_x, test_y)

        delta_br = br - backdoor_o
        delta_ca = ca - clean_acc_o

        print(f'{sub_method:<20} {br:>14.6f} {ca:>14.6f}')

        all_results.append({
            'sub_method': sub_method,
            'artifact_path': coeff_path,
            'artifact_type': 'coefficients',
            'backdoor_rate': br,
            'clean_acc': ca,
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
            f.write(f'backdoor_rate_o={backdoor_o:.10f}\n')
            f.write(f'clean_acc_o={clean_acc_o:.10f}\n')
            f.write('\n')
            f.write(f'backdoor_rate={br:.10f}  (delta: {delta_br:+.10f})\n')
            f.write(f'clean_acc={ca:.10f}  (delta: {delta_ca:+.10f})\n')

        print(f'  -> Saved: {result_filename}')

    print(f'{"="*60}')

    # Save summary CSV
    summary_path = os.path.join(result_dir, 'test_summary.csv')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('sub_method,backdoor_rate,clean_acc\n')
        f.write(f'original,{backdoor_o:.10f},{clean_acc_o:.10f}\n')
        for r in all_results:
            f.write(f'{r["sub_method"]},{r["backdoor_rate"]:.10f},{r["clean_acc"]:.10f}\n')

    print(f'Summary saved: {summary_path}')

    # Save best result by minimum backdoor rate. If two methods tie, keep the
    # one with higher clean accuracy.
    if all_results:
        best = min(all_results, key=lambda r: (r['backdoor_rate'], -r['clean_acc']))
        best_path = os.path.join(result_dir, 'best_solution_min_rate.txt')
        with open(best_path, 'w', encoding='utf-8') as f:
            f.write(f'model_name={model_name}\n')
            f.write(f'name_tag={name_tag}\n')
            f.write(f'seed={seed}\n')
            f.write(f'target_label={target_label}\n')
            f.write(f'original_backdoor_rate={backdoor_o:.10f}\n')
            f.write(f'original_clean_acc={clean_acc_o:.10f}\n')
            f.write(f'best_method={best["sub_method"]}\n')
            f.write(f'best_artifact_type={best["artifact_type"]}\n')
            f.write(f'best_artifact_path={best["artifact_path"]}\n')
            f.write(f'best_test_backdoor_rate={best["backdoor_rate"]:.10f}\n')
            f.write(f'best_test_clean_acc={best["clean_acc"]:.10f}\n')
            f.write('\n')
            f.write('all_results:\n')
            for r in all_results:
                f.write(
                    f'{r["sub_method"]}: '
                    f'backdoor_rate={r["backdoor_rate"]:.10f}, '
                    f'clean_acc={r["clean_acc"]:.10f}, '
                    f'artifact={r["artifact_path"]}\n'
                )
        print(f'Best result saved: {best_path}')
    print(f'Done in {time.time() - t0:.2f}s')


if __name__ == '__main__':
    main()
