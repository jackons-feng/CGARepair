#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepProv-style FINAL-checkpoint evaluation (no TEST-based selection).

This standalone TEST evaluator intentionally NEVER imports adv_test_2.py,
adv_weight_test.py or defect_repair.py. Local projects have multiple divergent
versions of those modules. It reproduces their documented clean/FGSM TEST
preprocessing and top-1 classification accuracy directly.

Run from NSrepair/cfrepair_adv:
  python -u deepprov_strict_test.py mnist 123 --epsilon 0.05 --dry-run
  python -u deepprov_strict_test.py mnist 123 --epsilon 0.05

Requires the already-existing *one FINAL checkpoint* for the named seed.
Does not change repair weights, select any candidate on TEST, or train.
"""
from __future__ import print_function
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = (42, 123, 256, 1024, 2026)
MODELS = ('mnist', 'cifar10')
EPS = '0.05'
BATCH_SIZE = 1024
CIFAR_MEAN = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.asarray([0.247, 0.243, 0.261], dtype=np.float32)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=MODELS)
    p.add_argument('seed', type=int, choices=EXPECTED_SEEDS)
    p.add_argument('--epsilon', default=EPS)
    p.add_argument('--model-path', default=None)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--data-root', default='../data',
                   help='Data directory relative to cfrepair_adv (default ../data)')
    p.add_argument('--test-source', choices=('auto', 'legacy', 'canonical'),
                   default='auto', help='Select exact TEST file family; auto prefers legacy adv_test_2.py')
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def test_file_candidates(model, data_root):
    root = Path(data_root).expanduser().resolve()
    suffix = 'txt' if model == 'mnist' else 'npy'
    return {
        # Mirrors the historically supplied adv_test_2.py loader.
        'legacy': (root / 'adv' / model / 'test' / ('epsilon%s.%s' % (EPS, suffix)),
                   root / 'adv' / model / 'test' / ('label.%s' % suffix)),
        # Supported by the user's adv_weight_test.py in addition to legacy.
        'canonical': (root / model / 'test' / ('epsilon%s.npy' % EPS),
                      root / model / 'test' / 'label.npy'),
    }


def resolve_test_files(model, data_root, source):
    pairs = test_file_candidates(model, data_root)
    selected = []
    for label, (data_file, label_file) in pairs.items():
        if data_file.is_file() and label_file.is_file():
            selected.append(label)
    if source == 'auto':
        if not selected:
            searched = '\n'.join('  %s: %s | %s' % (k, x, y)
                                 for k, (x, y) in pairs.items())
            raise FileNotFoundError('No complete adversarial TEST input/label pair found:\n' + searched)
        chosen = 'legacy' if 'legacy' in selected else 'canonical'
        if len(selected) > 1:
            print('[WARNING] Both canonical and legacy TEST sets exist. Auto chooses %s.' % chosen)
            print('[WARNING] Verify that this matches CGARepair formal TEST; otherwise pass --test-source canonical.')
    else:
        chosen = source
        if chosen not in selected:
            a, b = pairs[chosen]
            raise FileNotFoundError('%s TEST files missing: %s / %s' % (chosen, a, b))
    x_path, y_path = pairs[chosen]
    return chosen, x_path, y_path


def _read_labels(path):
    if path.suffix == '.txt':
        ys = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ys.append(int(float(line.strip())))
        values = np.asarray(ys, dtype=np.int64)
    else:
        values = np.load(str(path), allow_pickle=False)
        if values.ndim == 2 and values.shape[1] > 1:
            values = np.argmax(values, axis=1)
        values = np.asarray(values).reshape(-1).astype(np.int64)
    return values


def _read_adv_images(path, model):
    if path.suffix.lower() == '.txt':
        if model != 'mnist':
            raise ValueError('Text adversarial samples are only supported for MNIST')
        rows = []
        with path.open('r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                v = np.fromstring(line.strip(), sep=',', dtype=np.float32)
                if v.size != 784:
                    raise ValueError('MNIST TEST line %d has %d pixels, expected 784' % (line_number, v.size))
                rows.append(v)
        if not rows:
            raise ValueError('Empty adversarial TEST file: %s' % path)
        array = np.stack(rows)
    else:
        array = np.load(str(path), allow_pickle=False)
        array = np.asarray(array, dtype=np.float32)
    shape = (28, 28, 1) if model == 'mnist' else (32, 32, 3)
    if array.size == 0 or array.size % int(np.prod(shape)) != 0:
        raise ValueError('Adversarial TEST array shape is invalid: %s' % (array.shape,))
    array = array.reshape((-1,) + shape)
    if not np.all(np.isfinite(array)):
        raise ValueError('Non-finite adversarial TEST pixels')
    # Formal project data loaders expect [0,1] images BEFORE CIFAR normalization.
    lo, hi = float(array.min()), float(array.max())
    print('[TEST RAW PIXELS] min=%.6f max=%.6f' % (lo, hi))
    if lo < -0.20 or hi > 1.20:
        raise ValueError('Adversarial TEST pixels seem not to be in [0,1]. '
                         'Check data provenance/preprocessing rather than silently rescaling: min=%g max=%g' % (lo, hi))
    if lo < -1e-3 or hi > 1.001:
        print('[WARNING] Adversarial pixels slightly outside [0,1]; preserving values as in original loader.')
    if model == 'cifar10':
        array = (array - CIFAR_MEAN) / CIFAR_STD
    return array


def load_test_data(model, source, x_file, y_file, tf):
    adv_x = _read_adv_images(x_file, model)
    adv_y = _read_labels(y_file)
    if len(adv_x) != len(adv_y):
        raise ValueError('Adversarial inputs/labels count mismatch: %d / %d' % (len(adv_x), len(adv_y)))
    if model == 'mnist':
        (_, _), (clean_x, clean_y) = tf.keras.datasets.mnist.load_data()
        clean_x = clean_x.astype(np.float32)[..., None] / 255.0
        clean_y = np.asarray(clean_y).reshape(-1).astype(np.int64)
    else:
        (_, _), (clean_x, clean_y) = tf.keras.datasets.cifar10.load_data()
        clean_x = clean_x.astype(np.float32) / 255.0
        clean_x = (clean_x - CIFAR_MEAN) / CIFAR_STD
        clean_y = np.asarray(clean_y).reshape(-1).astype(np.int64)
    for name, y in [('adversarial', adv_y), ('clean', clean_y)]:
        if len(y) == 0 or np.any(y < 0) or np.any(y >= 10):
            raise ValueError('%s labels invalid (expected 0..9)' % name)
    if not np.all(np.isfinite(clean_x)):
        raise ValueError('Non-finite clean TEST pixels')
    print('[TEST SOURCE] %s' % source)
    print('[TEST INPUT]  %s' % x_file)
    print('[TEST LABEL]  %s' % y_file)
    print('[TEST SHAPES] adv=%s labels=%s clean=%s labels=%s' %
          (adv_x.shape, adv_y.shape, clean_x.shape, clean_y.shape))
    return adv_x, adv_y, clean_x, clean_y


def evaluate_accuracy(model, images, labels, session, keras_backend, batch_size):
    if len(images) != len(labels):
        raise ValueError('Input/label length mismatch in evaluation')
    phase = keras_backend.learning_phase()
    phase_placeholder = not isinstance(phase, (int, np.integer))
    total = 0
    for start in range(0, len(labels), batch_size):
        end = min(len(labels), start + batch_size)
        feed = {model.input: images[start:end]}
        if phase_placeholder:
            feed[phase] = 0
        output = session.run(model.output, feed_dict=feed)
        output = np.asarray(output)
        if output.ndim != 2 or output.shape[1] < 2:
            raise ValueError('Expected multiclass logits/probabilities, got shape %s' % (output.shape,))
        predictions = output.argmax(axis=1)
        total += int(np.count_nonzero(predictions == labels[start:end]))
    return float(total) / float(len(labels))


def main(argv=None):
    args = get_parser().parse_args(argv)
    if abs(float(args.epsilon) - 0.05) > 1e-12:
        raise SystemExit('Only epsilon=0.05 allowed; do not mix epsilon=0.1 baselines')
    if args.batch_size <= 0:
        raise SystemExit('batch-size must be positive')
    model_path = Path(args.model_path or '../model/%s_adv.h5' % args.model).expanduser().resolve()
    checkpoint = Path(args.checkpoint or
                      ('loc_result/%s/%d/eps0.05/defect_repair/deepprov/weights_final_deepprov.h5'
                       % (args.model, args.seed))).expanduser().resolve()
    output = Path(args.output or (checkpoint.parent / 'deepprov_strict_test_results.json')).expanduser().resolve()
    print('[METHOD] DeepProv-style adapter; fixed FINAL checkpoint')
    print('[MODEL] ', model_path)
    print('[WEIGHTS]', checkpoint)
    print('[OUTPUT]', output)
    print('[SEED] %d [EPSILON] %s' % (args.seed, EPS))
    print('[SELECTION] Exactly one existing checkpoint; TEST never selects a winner')
    if not model_path.is_file():
        raise SystemExit('Original model missing: %s' % model_path)
    if not checkpoint.is_file():
        raise SystemExit('Fixed checkpoint missing: %s' % checkpoint)
    if output.exists() and not args.overwrite:
        print('[SKIP] Existing output: %s (pass --overwrite to re-evaluate)' % output)
        return 0
    source, test_x, test_y = resolve_test_files(args.model, args.data_root, args.test_source)
    print('[TEST SOURCE] %s' % source)
    print('[TEST INPUT] %s' % test_x)
    print('[TEST LABEL] %s' % test_y)
    if args.dry_run:
        print('[DRY RUN OK] Original model, fixed checkpoint and TEST files all exist.')
        print('[IMPORTANT] Verify selected TEST source matches CGARepair official TEST.')
        return 0

    start = time.time()
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    import tensorflow.python.keras.backend as K
    from tensorflow.keras.models import load_model
    tf.compat.v1.disable_eager_execution()
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    try:
        adv_x, adv_y, clean_x, clean_y = load_test_data(args.model, source, test_x, test_y, tf)
        model = load_model(str(model_path), compile=False)
        if len(model.input_shape) < 4 or tuple(model.input_shape[1:]) != tuple(clean_x.shape[1:]):
            raise ValueError('Original model input shape %s vs TEST shape %s' %
                             (model.input_shape, clean_x.shape))
        baseline_adv = evaluate_accuracy(model, adv_x, adv_y, sess, K, args.batch_size)
        baseline_clean = evaluate_accuracy(model, clean_x, clean_y, sess, K, args.batch_size)
        print('[BEFORE] CleanAcc=%.8f AdvAcc=%.8f AER=%.8f' %
              (baseline_clean, baseline_adv, 1.0 - baseline_adv))
        # Only one preselected weight checkpoint. Its SHA is captured for provenance.
        model.load_weights(str(checkpoint))
        repaired_adv = evaluate_accuracy(model, adv_x, adv_y, sess, K, args.batch_size)
        repaired_clean = evaluate_accuracy(model, clean_x, clean_y, sess, K, args.batch_size)
        print('[AFTER]  CleanAcc=%.8f AdvAcc=%.8f AER=%.8f' %
              (repaired_clean, repaired_adv, 1.0 - repaired_adv))
    finally:
        sess.close()
    metrics = [baseline_clean, baseline_adv, repaired_clean, repaired_adv]
    if any((not np.isfinite(x)) or x < 0 or x > 1 for x in metrics):
        raise ValueError('Metric outside [0,1]')
    record = {
        'task': 'adversarial',
        'method': 'DeepProv',
        'implementation': 'DeepProv-style weight adapter via defect_repair.py/deepprov.py',
        'model_name': args.model,
        'seed': args.seed,
        'epsilon': 0.05,
        'evaluation_scope': 'complete_available_adversarial_TEST_file_and_clean_Keras_TEST',
        'test_only_reporting': True,
        'test_used_for_selection': False,
        'test_used_for_model_selection': False,
        'checkpoint_selection': 'One pre-existing final weights checkpoint',
        'training_data_provenance_verified': False,
        'test_source': {
            'kind': source,
            'adversarial_inputs': str(test_x),
            'adversarial_inputs_sha256': sha256_file(test_x),
            'adversarial_labels': str(test_y),
            'adversarial_labels_sha256': sha256_file(test_y),
            'clean_inputs': 'tf.keras.datasets.%s.load_data()[1]' % args.model,
            'preprocessing': 'MNIST [0,1]; CIFAR-10 channel mean/std from adv_test_2',
            'evaluator': 'self-contained TensorFlow TF1 Session top-1 classification accuracy',
        },
        'test_count': len(adv_y),
        'clean_test_count': len(clean_y),
        'original_model': str(model_path),
        'original_model_sha256': sha256_file(model_path),
        'fixed_checkpoint': str(checkpoint),
        'fixed_checkpoint_sha256': sha256_file(checkpoint),
        'original_test_metrics': {
            'clean_accuracy': baseline_clean,
            'adv_accuracy': baseline_adv,
            'adversarial_error_rate': 1.0 - baseline_adv,
        },
        'repaired_test_metrics': {
            'clean_accuracy': repaired_clean,
            'adv_accuracy': repaired_adv,
            'adversarial_error_rate': 1.0 - repaired_adv,
        },
        'elapsed_sec': time.time() - start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tempfile = output.with_name(output.name + '.tmp')
    with tempfile.open('w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(str(tempfile), str(output))
    print('[SAVED]', output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
