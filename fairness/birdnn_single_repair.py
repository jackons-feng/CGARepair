# -*- coding: utf-8 -*-
"""BIRDNN-FT adapted to tabular single-defect repair.

This file contains both BIRDNN behavior-difference localization and PSO repair.
It is intentionally TRAIN-ONLY: no test split is loaded or used for checkpoint
selection. The repaired artifact is a normal Keras model with the original
architecture, so existing test code can load it without custom objects.

Supported datasets
------------------
    bank, adult, default

Supported single defects
------------------------
    fairness, adv, backdoor

BIRDNN mechanisms preserved
----------------------------
1. Construct positive/negative behavior samples.
2. Score each hidden Dense neuron by the absolute difference between the
   aggregated positive and negative activations.
3. Select the global Top-K neurons.
4. Search one delta in [-1, 1] for every selected neuron with global-best PSO.
5. Use scale = 1 + delta, exactly matching the source activation scaling.
6. Minimize
       (1 - alpha) * defect_loss + alpha * (1 - clean_accuracy).

For a chain-like Dense network, multiplying hidden activation h_j by a positive
scale s is exactly equivalent to multiplying row j of the immediately following
downstream Dense-kernel row by s. This script uses that exact materialization so the saved
model keeps the original architecture and remains compatible with load_weights.

Examples
--------
Fairness:
    python -B birdnn_single_repair.py fairness bank 256 \
        --model-root ../model --combined-root ../data/combined_defects

Adversarial robustness:
    python -B birdnn_single_repair.py adv adult 256 \
        --model-root ../model --combined-root ../data/combined_defects

Backdoor:
    python -B birdnn_single_repair.py backdoor default 256 \
        --model-root ../model --combined-root ../data/combined_defects
"""

from __future__ import print_function

import argparse
import copy
import inspect
import json
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model


SUPPORTED_DATASETS = ('bank', 'adult', 'default')
TASK_ALIASES = {
    'fairness': 'fairness',
    'fair': 'fairness',
    'f': 'fairness',
    'adv': 'adv',
    'adversarial': 'adv',
    'a': 'adv',
    'backdoor': 'backdoor',
    'bd': 'backdoor',
    'b': 'backdoor',
}
BATCH_SIZE = 1024
SOURCE_DEFAULT_REPAIR_NEURONS = 20


class NoQualifyingBehaviorSamples(RuntimeError):
    """Raised when the current TRAIN model has no BIRDNN negative behavior."""


def canonical_task(value):
    key = str(value).strip().lower()
    if key not in TASK_ALIASES:
        raise ValueError(
            'Unsupported task %r. Choose fairness, adv, or backdoor.' % value
        )
    return TASK_ALIASES[key]


def validate_dataset(value):
    dataset = str(value).strip().lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            'Unsupported dataset %r. Choose from: %s'
            % (value, ', '.join(SUPPORTED_DATASETS))
        )
    return dataset


def configure_tensorflow(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.compat.v1.disable_eager_execution()
    except RuntimeError:
        pass
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    )
    K.set_session(session)
    return session


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def normalize_labels(values):
    values = np.asarray(values)
    if values.ndim > 1 and values.shape[-1] > 1:
        values = np.argmax(values, axis=-1)
    return values.reshape(-1).astype(np.int64)


def normalize_groups(values):
    groups = np.asarray(values).reshape(-1).astype(np.int64)
    unique = np.unique(groups)
    if len(unique) != 2:
        raise ValueError(
            'BIRDNN fairness repair requires exactly two groups; got %s.'
            % unique.tolist()
        )
    if set(unique.tolist()) != {0, 1}:
        mapping = {int(unique[0]): 0, int(unique[1]): 1}
        groups = np.asarray(
            [mapping[int(item)] for item in groups], dtype=np.int64
        )
    return groups


def _call_supported(function, args, kwargs):
    signature = inspect.signature(function)
    accepted = {
        key: value for key, value in kwargs.items()
        if key in signature.parameters
    }
    return function(*args, **accepted)


def _load_fairness_train_from_project(dataset, seed):
    errors = []
    try:
        import repair_config
    except Exception as exc:
        raise ImportError('repair_config.py is required.') from exc

    loader = getattr(repair_config, 'load_fairness_data', None)
    if loader is not None:
        attempts = [
            ((dataset,), {
                'split': 'train', 'sample_size': None,
                'seed': seed, 'return_groups': True,
            }),
            ((), {
                'model_name': dataset, 'split': 'train',
                'sample_size': None, 'seed': seed,
                'return_groups': True,
            }),
        ]
        for positional, keyword in attempts:
            try:
                result = _call_supported(loader, positional, keyword)
                if isinstance(result, dict):
                    x = result.get('x_clean', result.get('x'))
                    y = result.get('y_clean', result.get('y'))
                    g = result.get('g_clean', result.get('g'))
                else:
                    x, y, g = result[:3]
                return {
                    'dataset': dataset,
                    'task': 'fairness',
                    'x_clean': np.asarray(x, dtype=np.float32),
                    'y_clean': normalize_labels(y),
                    'g_clean': normalize_groups(g),
                    'source': 'repair_config.load_fairness_data(train)',
                    'target_label': 1,
                }
            except Exception as exc:
                errors.append(repr(exc))

    # Compatibility fallback: the clean branch of prepared FA/FB TRAIN assets
    # is also a valid fairness training split.
    dual_loader = getattr(repair_config, 'load_twodefect_repair_data', None)
    if dual_loader is not None:
        for task_mode in ('fa', 'fb'):
            try:
                bundle = _call_supported(
                    dual_loader,
                    (dataset, task_mode),
                    {'combined_root': '../data/combined_defects'},
                )
                return {
                    'dataset': dataset,
                    'task': 'fairness',
                    'x_clean': np.asarray(bundle['x_clean'], dtype=np.float32),
                    'y_clean': normalize_labels(bundle['y_clean']),
                    'g_clean': normalize_groups(bundle['g_clean']),
                    'source': (
                        'repair_config.load_twodefect_repair_data(%s);clean_branch'
                        % task_mode
                    ),
                    'target_label': int(bundle.get('target_label') or 1),
                }
            except Exception as exc:
                errors.append(repr(exc))

    raise RuntimeError(
        'Unable to load TRAIN-only fairness data for %s. Errors: %s'
        % (dataset, errors)
    )


def load_dual_train_bundle(dataset, task_mode, combined_root):
    try:
        import repair_config
    except Exception as exc:
        raise ImportError('repair_config.py is required.') from exc
    loader = getattr(repair_config, 'load_twodefect_repair_data', None)
    if loader is None:
        raise ImportError(
            'repair_config.load_twodefect_repair_data is required for '
            'adversarial/backdoor BIRDNN repair.'
        )
    bundle = _call_supported(
        loader,
        (dataset, task_mode),
        {'combined_root': combined_root},
    )
    required = {'x_clean', 'y_clean', 'g_clean', 'x_defect', 'y_defect'}
    missing = sorted(required.difference(bundle))
    if missing:
        raise KeyError(
            'Dual-defect TRAIN bundle is missing keys: %s' % missing
        )
    result = dict(bundle)
    result['x_clean'] = np.asarray(result['x_clean'], dtype=np.float32)
    result['y_clean'] = normalize_labels(result['y_clean'])
    result['g_clean'] = normalize_groups(result['g_clean'])
    result['x_defect'] = np.asarray(result['x_defect'], dtype=np.float32)
    result['y_defect'] = normalize_labels(result['y_defect'])
    result['target_label'] = int(result.get('target_label') or 1)
    result['source'] = result.get(
        'root', 'repair_config.load_twodefect_repair_data(%s)' % task_mode
    )
    return result


def load_single_task_bundle(dataset, task, seed, combined_root,
                            external_bundle=None):
    task = canonical_task(task)
    if external_bundle is not None:
        bundle = dict(external_bundle)
        result = {
            'dataset': dataset,
            'task': task,
            'x_clean': np.asarray(bundle['x_clean'], dtype=np.float32),
            'y_clean': normalize_labels(bundle['y_clean']),
            'g_clean': normalize_groups(bundle['g_clean']),
            'source': bundle.get('root', bundle.get('source', 'external_bundle')),
            'target_label': int(bundle.get('target_label') or 1),
        }
        if task in ('adv', 'backdoor'):
            result['x_defect'] = np.asarray(
                bundle['x_defect'], dtype=np.float32
            )
            result['y_defect'] = normalize_labels(bundle['y_defect'])
        return result

    if task == 'fairness':
        # First use the caller-selected combined root for the fallback path.
        try:
            import repair_config
            loader = getattr(repair_config, 'load_fairness_data', None)
        except Exception:
            loader = None
        if loader is not None:
            attempts = [
                ((dataset,), {
                    'split': 'train', 'sample_size': None,
                    'seed': seed, 'return_groups': True,
                }),
                ((), {
                    'model_name': dataset, 'split': 'train',
                    'sample_size': None, 'seed': seed,
                    'return_groups': True,
                }),
            ]
            for positional, keyword in attempts:
                try:
                    values = _call_supported(loader, positional, keyword)
                    if isinstance(values, dict):
                        x = values.get('x_clean', values.get('x'))
                        y = values.get('y_clean', values.get('y'))
                        g = values.get('g_clean', values.get('g'))
                    else:
                        x, y, g = values[:3]
                    return {
                        'dataset': dataset,
                        'task': task,
                        'x_clean': np.asarray(x, dtype=np.float32),
                        'y_clean': normalize_labels(y),
                        'g_clean': normalize_groups(g),
                        'source': 'repair_config.load_fairness_data(train)',
                        'target_label': 1,
                    }
                except Exception:
                    pass
        for mode in ('fa', 'fb'):
            try:
                dual = load_dual_train_bundle(dataset, mode, combined_root)
                return {
                    'dataset': dataset,
                    'task': task,
                    'x_clean': dual['x_clean'],
                    'y_clean': dual['y_clean'],
                    'g_clean': dual['g_clean'],
                    'source': '%s;clean_branch' % dual['source'],
                    'target_label': dual['target_label'],
                }
            except Exception:
                pass
        return _load_fairness_train_from_project(dataset, seed)

    mode = 'fa' if task == 'adv' else 'fb'
    dual = load_dual_train_bundle(dataset, mode, combined_root)
    return {
        'dataset': dataset,
        'task': task,
        'x_clean': dual['x_clean'],
        'y_clean': dual['y_clean'],
        'g_clean': dual['g_clean'],
        'x_defect': dual['x_defect'],
        'y_defect': dual['y_defect'],
        'target_label': dual['target_label'],
        'source': dual['source'],
    }


def resolve_default_model_path(dataset, task, model_root='../model'):
    dataset = validate_dataset(dataset)
    task = canonical_task(task)
    filename = (
        '%s_backdoor.h5' % dataset
        if task == 'backdoor'
        else '%s.h5' % dataset
    )
    candidates = [
        os.path.join(model_root, filename),
        os.path.join('model', filename),
        os.path.join('..', 'model', filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        'Model not found for %s/%s. Checked:\n  %s'
        % (dataset, task, '\n  '.join(candidates))
    )


def session_run(model, tensors, images, batch_size=BATCH_SIZE):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
        single = True
    else:
        single = False
    all_outputs = [[] for _ in tensors]
    learning_phase = None
    try:
        phase = K.learning_phase()
        if not isinstance(phase, (int, np.integer)):
            learning_phase = phase
    except Exception:
        learning_phase = None

    for start in range(0, len(images), int(batch_size)):
        end = min(start + int(batch_size), len(images))
        feed = {model.input: images[start:end]}
        if learning_phase is not None:
            feed[learning_phase] = 0
        values = K.get_session().run(list(tensors), feed_dict=feed)
        for index, value in enumerate(values):
            all_outputs[index].append(np.asarray(value))
    merged = [
        np.concatenate(parts, axis=0) if parts else np.asarray([])
        for parts in all_outputs
    ]
    return merged[0] if single else merged


def output_to_labels(output):
    output = np.asarray(output)
    if output.ndim == 1 or (output.ndim >= 2 and output.shape[-1] == 1):
        return (output.reshape(-1) > 0.5).astype(np.int64)
    return np.argmax(output, axis=-1).reshape(-1).astype(np.int64)


def predict_labels(model, images, batch_size=BATCH_SIZE):
    return output_to_labels(
        session_run(model, model.output, images, batch_size=batch_size)
    )


def accuracy_from_predictions(predictions, labels):
    labels = normalize_labels(labels)
    predictions = np.asarray(predictions).reshape(-1).astype(np.int64)
    if len(predictions) != len(labels):
        raise ValueError(
            'Prediction/label mismatch: %d vs %d'
            % (len(predictions), len(labels))
        )
    return float(np.mean(predictions == labels))


def fairness_metrics_from_predictions(predictions, labels, groups):
    predictions = np.asarray(predictions).reshape(-1).astype(np.int64)
    labels = normalize_labels(labels)
    groups = normalize_groups(groups)
    if not (len(predictions) == len(labels) == len(groups)):
        raise ValueError('Fairness arrays are not aligned.')
    acc0 = float(np.mean(predictions[groups == 0] == labels[groups == 0]))
    acc1 = float(np.mean(predictions[groups == 1] == labels[groups == 1]))
    return {
        'fairness_gap': float(abs(acc1 - acc0)),
        'group_0_accuracy': acc0,
        'group_1_accuracy': acc1,
    }


def backdoor_asr_from_predictions(predictions, labels, target_label):
    predictions = np.asarray(predictions).reshape(-1).astype(np.int64)
    labels = normalize_labels(labels)
    eligible = labels != int(target_label)
    if np.any(eligible):
        return float(np.mean(predictions[eligible] == int(target_label)))
    return float(np.mean(predictions == int(target_label)))


def evaluate_task(model, bundle, task, clean_indices=None,
                  defect_indices=None, batch_size=BATCH_SIZE):
    task = canonical_task(task)
    clean_x = bundle['x_clean']
    clean_y = bundle['y_clean']
    clean_g = bundle['g_clean']
    if clean_indices is not None:
        clean_x = clean_x[clean_indices]
        clean_y = clean_y[clean_indices]
        clean_g = clean_g[clean_indices]

    clean_pred = predict_labels(model, clean_x, batch_size=batch_size)
    clean_acc = accuracy_from_predictions(clean_pred, clean_y)
    fairness = fairness_metrics_from_predictions(clean_pred, clean_y, clean_g)
    result = {
        'clean_accuracy': clean_acc,
        'fairness_gap': fairness['fairness_gap'],
        'group_0_accuracy': fairness['group_0_accuracy'],
        'group_1_accuracy': fairness['group_1_accuracy'],
    }

    if task == 'fairness':
        result['defect_loss'] = fairness['fairness_gap']
    else:
        defect_x = bundle['x_defect']
        defect_y = bundle['y_defect']
        if defect_indices is not None:
            defect_x = defect_x[defect_indices]
            defect_y = defect_y[defect_indices]
        defect_pred = predict_labels(model, defect_x, batch_size=batch_size)
        if task == 'adv':
            adv_acc = accuracy_from_predictions(defect_pred, defect_y)
            result['adversarial_accuracy'] = adv_acc
            result['adversarial_asr'] = 1.0 - adv_acc
            result['defect_loss'] = 1.0 - adv_acc
        else:
            asr = backdoor_asr_from_predictions(
                defect_pred, defect_y, bundle['target_label']
            )
            result['backdoor_asr'] = asr
            result['defect_loss'] = asr
    return result


def unwrap_layer(layer):
    current = layer
    while isinstance(current, tf.keras.layers.Wrapper):
        current = current.layer
    return current


def _keras_tensor_source_layer(tensor):
    """Best-effort source-layer resolution for TF/Keras 1.x and 2.x tensors."""
    history = getattr(tensor, '_keras_history', None)
    if history is None:
        return None
    if isinstance(history, (tuple, list)):
        return history[0] if history else None
    layer = getattr(history, 'layer', None)
    if layer is not None:
        return layer
    return getattr(history, 'operation', None)


def discover_repairable_dense_layers(model):
    """Find hidden feature tensors that feed one or more later Dense layers.

    BIRDNN scales a selected hidden activation h_j by s_j.  For a Dense consumer
    y = hW + b, this can be materialized exactly by multiplying row j of W by
    s_j.  Therefore the correct repair site is the *input tensor of a downstream
    Dense layer*, not necessarily the output of the previous Dense layer.

    This supports common tabular blocks such as
        Dense -> BatchNormalization -> Activation -> Dropout -> Dense
    because localization is performed on the final feature tensor immediately
    before the downstream Dense layer.  Tensors shared by multiple Dense
    consumers are grouped and every matching outgoing kernel is scaled.
    """
    dense_layers = []
    for index, layer in enumerate(model.layers):
        base = unwrap_layer(layer)
        if not isinstance(base, tf.keras.layers.Dense):
            continue
        weights = base.get_weights()
        if not weights or np.asarray(weights[0]).ndim != 2:
            continue
        dense_layers.append((index, layer, base))

    if len(dense_layers) < 2:
        raise RuntimeError(
            'BIRDNN tabular repair requires at least two Dense layers '
            '(a hidden representation and a downstream Dense consumer).'
        )

    layer_identity = {id(layer): index for index, layer in enumerate(model.layers)}
    grouped = {}
    rejected = []

    # Skip the first Dense consumer: its input is normally the raw feature vector,
    # not a hidden neural representation.  Every later Dense input is a valid
    # hidden repair site when its width matches the consumer kernel input width.
    for dense_position, (next_idx, next_layer, next_base) in enumerate(dense_layers):
        if dense_position == 0:
            continue

        input_tensor = getattr(next_layer, 'input', None)
        if isinstance(input_tensor, (tuple, list)):
            if len(input_tensor) != 1:
                rejected.append({
                    'next_dense_idx': int(next_idx),
                    'next_dense_name': next_layer.name,
                    'reason': 'multiple_input_tensors',
                })
                continue
            input_tensor = input_tensor[0]
        if input_tensor is None:
            rejected.append({
                'next_dense_idx': int(next_idx),
                'next_dense_name': next_layer.name,
                'reason': 'missing_input_tensor',
            })
            continue

        shape = K.int_shape(input_tensor)
        weights = next_base.get_weights()
        kernel = np.asarray(weights[0]) if weights else None
        if shape is None or len(shape) != 2 or shape[-1] is None:
            rejected.append({
                'next_dense_idx': int(next_idx),
                'next_dense_name': next_layer.name,
                'reason': 'dense_input_is_not_known_rank2',
                'input_shape': None if shape is None else list(shape),
            })
            continue
        units = int(shape[-1])
        if kernel is None or kernel.ndim != 2 or int(kernel.shape[0]) != units:
            rejected.append({
                'next_dense_idx': int(next_idx),
                'next_dense_name': next_layer.name,
                'reason': 'input_width_kernel_row_mismatch',
                'input_width': units,
                'kernel_shape': None if kernel is None else list(kernel.shape),
            })
            continue

        source_layer = _keras_tensor_source_layer(input_tensor)
        source_idx = layer_identity.get(id(source_layer), -1)
        source_name = getattr(source_layer, 'name', 'pre_%s' % next_layer.name)
        source_type = type(source_layer).__name__ if source_layer is not None else 'KerasTensor'
        if isinstance(source_layer, tf.keras.layers.InputLayer):
            rejected.append({
                'next_dense_idx': int(next_idx),
                'next_dense_name': next_layer.name,
                'reason': 'raw_model_input_not_hidden_neuron',
            })
            continue

        tensor_name = getattr(input_tensor, 'name', None)
        key = tensor_name if tensor_name is not None else 'tensor_id_%d' % id(input_tensor)
        if key not in grouped:
            grouped[key] = {
                'layer_idx': int(source_idx if source_idx >= 0 else next_idx),
                'layer_name': str(source_name),
                'layer_type': str(source_type),
                'units': int(units),
                'activation_tensor_name': str(key),
                'next_dense_idx': int(next_idx),
                'next_dense_name': next_layer.name,
                'next_dense_indices': [int(next_idx)],
                'next_dense_names': [next_layer.name],
                '_activation_tensor': input_tensor,
            }
        else:
            item = grouped[key]
            if int(item['units']) != units:
                raise RuntimeError(
                    'Shared BIRDNN activation tensor has inconsistent widths: %s'
                    % key
                )
            item['next_dense_indices'].append(int(next_idx))
            item['next_dense_names'].append(next_layer.name)

    repairable = list(grouped.values())
    if not repairable:
        architecture = [
            '%d:%s(%s)' % (i, getattr(layer, 'name', '?'), type(unwrap_layer(layer)).__name__)
            for i, layer in enumerate(model.layers)
        ]
        raise RuntimeError(
            'No hidden rank-2 feature tensor feeding a Dense layer was found for '
            'exact BIRDNN activation scaling. Model layers=%s; rejected=%s'
            % (architecture, rejected)
        )
    return repairable

def _source_faithful_behavior_sample(pos_x, neg_x, rng, max_samples):
    """Keep BIRDNN's independent positive/negative sets.

    The source implementation aggregates each set with SUM and does not force
    the two sets to have the same cardinality.  A positive max_samples value is
    only an explicit resource cap; the default (<= 0) uses every qualifying
    training sample on each side.
    """
    if len(pos_x) <= 0 or len(neg_x) <= 0:
        raise RuntimeError('BIRDNN positive/negative behavior set is empty.')

    def take(values):
        if max_samples is None or int(max_samples) <= 0 or len(values) <= int(max_samples):
            return values
        index = rng.choice(len(values), int(max_samples), replace=False)
        return values[index]

    return take(pos_x), take(neg_x)


def construct_behavior_sets(model, bundle, task, seed,
                            max_localization_samples):
    """Construct positive and negative behavior samples for BIRDNN.

    fairness:
        positive = correctly classified samples in the lower-accuracy group
        negative = misclassified samples in the same lower-accuracy group
    adv:
        positive = clean-correct samples
        negative = adversarially misclassified samples
        paired when clean/adv arrays are aligned
    backdoor:
        positive = clean-correct non-target samples
        negative = successful triggered non-target samples
    """
    task = canonical_task(task)
    rng = np.random.RandomState(seed)
    clean_x = bundle['x_clean']
    clean_y = bundle['y_clean']
    clean_g = bundle['g_clean']
    clean_pred = predict_labels(model, clean_x)
    clean_correct = clean_pred == clean_y

    metadata = {
        'construction': None,
        'positive_pool_size': None,
        'negative_pool_size': None,
    }

    if task == 'fairness':
        fair = fairness_metrics_from_predictions(clean_pred, clean_y, clean_g)
        low_group = (
            0 if fair['group_0_accuracy'] <= fair['group_1_accuracy'] else 1
        )
        low_mask = clean_g == low_group
        pos_x = clean_x[low_mask & clean_correct]
        neg_x = clean_x[low_mask & (~clean_correct)]
        metadata.update({
            'construction': 'low_group_correct_vs_low_group_incorrect',
            'lower_accuracy_group': int(low_group),
            'group_0_accuracy_before_localization': fair['group_0_accuracy'],
            'group_1_accuracy_before_localization': fair['group_1_accuracy'],
        })
    elif task == 'adv':
        defect_x = bundle['x_defect']
        defect_y = bundle['y_defect']
        defect_pred = predict_labels(model, defect_x)
        defect_wrong = defect_pred != defect_y
        if len(clean_x) == len(defect_x):
            n = min(len(clean_x), len(defect_x), len(clean_y), len(defect_y))
            paired_mask = clean_correct[:n] & defect_wrong[:n]
            pos_x = clean_x[:n][paired_mask]
            neg_x = defect_x[:n][paired_mask]
            metadata['construction'] = (
                'paired_clean_correct_vs_successful_adversarial'
            )
        else:
            pos_x = clean_x[clean_correct]
            neg_x = defect_x[defect_wrong]
            metadata['construction'] = (
                'balanced_clean_correct_vs_successful_adversarial_unpaired'
            )
    else:
        defect_x = bundle['x_defect']
        defect_y = bundle['y_defect']
        target = int(bundle['target_label'])
        defect_pred = predict_labels(model, defect_x)
        pos_mask = clean_correct & (clean_y != target)
        neg_mask = (defect_y != target) & (defect_pred == target)
        pos_x = clean_x[pos_mask]
        neg_x = defect_x[neg_mask]
        metadata.update({
            'construction': (
                'clean_correct_non_target_vs_successful_triggered_non_target'
            ),
            'target_label': target,
        })

    metadata['positive_pool_size'] = int(len(pos_x))
    metadata['negative_pool_size'] = int(len(neg_x))
    if len(pos_x) == 0 or len(neg_x) == 0:
        raise NoQualifyingBehaviorSamples(
            'Cannot construct BIRDNN %s behavior sets: positive=%d, negative=%d. '
            'The current TRAIN model has no qualifying negative behavior or no '
            'matching positive reference samples.'
            % (task, len(pos_x), len(neg_x))
        )

    pos_x, neg_x = _source_faithful_behavior_sample(
        pos_x, neg_x, rng, max_localization_samples
    )
    metadata['positive_localization_sample_count'] = int(len(pos_x))
    metadata['negative_localization_sample_count'] = int(len(neg_x))
    metadata['positive_negative_equalized'] = False
    metadata['aggregation'] = 'independent_sum_as_in_source'
    metadata['localization_sample_cap_per_side'] = (
        int(max_localization_samples)
        if max_localization_samples is not None and int(max_localization_samples) > 0
        else None
    )
    return pos_x, neg_x, metadata


def _aggregate_layer_activations(model, layer_infos, images,
                                 batch_size=BATCH_SIZE):
    tensors = [item['_activation_tensor'] for item in layer_infos]
    sums = [np.zeros((item['units'],), dtype=np.float64) for item in layer_infos]
    learning_phase = None
    try:
        phase = K.learning_phase()
        if not isinstance(phase, (int, np.integer)):
            learning_phase = phase
    except Exception:
        learning_phase = None

    for start in range(0, len(images), int(batch_size)):
        end = min(start + int(batch_size), len(images))
        feed = {model.input: images[start:end]}
        if learning_phase is not None:
            feed[learning_phase] = 0
        outputs = K.get_session().run(tensors, feed_dict=feed)
        for index, output in enumerate(outputs):
            output = np.asarray(output)
            if output.ndim != 2:
                raise RuntimeError(
                    'Tabular BIRDNN expects 2D Dense activations, got %s at %s.'
                    % (output.shape, layer_infos[index]['layer_name'])
                )
            sums[index] += np.sum(output, axis=0, dtype=np.float64)
    return sums


def localize_birdnn(model, bundle, task, seed,
                    repair_neurons=SOURCE_DEFAULT_REPAIR_NEURONS,
                    max_localization_samples=0):
    layer_infos = discover_repairable_dense_layers(model)
    pos_x, neg_x, behavior_meta = construct_behavior_sets(
        model, bundle, task, seed, max_localization_samples
    )
    pos_sums = _aggregate_layer_activations(model, layer_infos, pos_x)
    neg_sums = _aggregate_layer_activations(model, layer_infos, neg_x)

    candidates = []
    for info, pos_sum, neg_sum in zip(layer_infos, pos_sums, neg_sums):
        scores = np.abs(pos_sum - neg_sum)
        for neuron_idx, score in enumerate(scores):
            if not np.isfinite(score):
                continue
            candidates.append({
                'player': 'Dense_%d_%d' % (info['layer_idx'], neuron_idx),
                'score': float(score),
                'layer_idx': int(info['layer_idx']),
                'layer_name': info['layer_name'],
                'neuron_idx': int(neuron_idx),
                'next_dense_idx': int(info['next_dense_idx']),
                'next_dense_name': info['next_dense_name'],
                'next_dense_indices': [int(v) for v in info.get('next_dense_indices', [info['next_dense_idx']])],
                'next_dense_names': list(info.get('next_dense_names', [info['next_dense_name']])),
            })
    candidates.sort(key=lambda item: item['score'], reverse=True)
    total = len(candidates)
    if total == 0:
        raise RuntimeError('BIRDNN localization produced no valid neurons.')

    repair_neurons = int(repair_neurons)
    if repair_neurons <= 0:
        raise ValueError(
            'BIRDNN source-faithful mode requires repair_neurons > 0; '
            'the source default is 20.'
        )
    top_k = min(repair_neurons, total)
    policy = 'source_repair_num_fixed_or_explicit'
    selected = candidates[:top_k]
    return {
        'method': 'BIRDNN-FT-source-faithful-task-adapted',
        'task': canonical_task(task),
        'score_formula': 'abs(sum_positive_activation-sum_negative_activation)',
        'positive_negative_equalized': False,
        'candidate_layer_policy': 'hidden_pre_dense_feature_tensor_with_exact_outgoing_kernel_materialization',
        'top_k_policy': policy,
        'top_k': int(top_k),
        'total_candidates': int(total),
        'players': [item['player'] for item in candidates],
        'scores': [float(item['score']) for item in candidates],
        'selected_players': [item['player'] for item in selected],
        'selected_scores': [float(item['score']) for item in selected],
        'selected_candidates': selected,
        'behavior_sets': behavior_meta,
        'repairable_layers': [
            {key: value for key, value in info.items() if not key.startswith('_')}
            for info in layer_infos
        ],
    }


class OutgoingKernelScaleMaterializer(object):
    """Materialize BIRDNN feature scales in all downstream Dense kernels."""

    def __init__(self, model, selected_candidates):
        self.model = model
        self.selected = list(selected_candidates)
        self.baseline = {}
        for item in self.selected:
            next_indices = item.get('next_dense_indices')
            if not next_indices:
                next_indices = [item['next_dense_idx']]
            for raw_idx in next_indices:
                next_idx = int(raw_idx)
                if next_idx in self.baseline:
                    continue
                weights = model.layers[next_idx].get_weights()
                if not weights or np.asarray(weights[0]).ndim != 2:
                    raise RuntimeError(
                        'Selected BIRDNN candidate does not map to a Dense kernel: %s'
                        % item
                    )
                self.baseline[next_idx] = [np.array(w, copy=True) for w in weights]

    def apply_delta(self, delta):
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        if len(delta) != len(self.selected):
            raise ValueError(
                'BIRDNN delta length mismatch: %d vs %d'
                % (len(delta), len(self.selected))
            )
        working = {
            index: [np.array(w, copy=True) for w in weights]
            for index, weights in self.baseline.items()
        }
        for value, item in zip(delta, self.selected):
            scale = 1.0 + float(value)
            if not np.isfinite(scale) or scale < -1e-12:
                raise ValueError('Invalid BIRDNN scale: %r' % scale)
            neuron_idx = int(item['neuron_idx'])
            next_indices = item.get('next_dense_indices')
            if not next_indices:
                next_indices = [item['next_dense_idx']]
            for raw_idx in next_indices:
                next_idx = int(raw_idx)
                kernel = working[next_idx][0]
                if neuron_idx < 0 or neuron_idx >= kernel.shape[0]:
                    raise IndexError(
                        'Neuron %d outside outgoing kernel input width %d.'
                        % (neuron_idx, kernel.shape[0])
                    )
                kernel[neuron_idx, :] *= scale
        for index, weights in working.items():
            self.model.layers[index].set_weights(weights)

    def restore(self):
        for index, weights in self.baseline.items():
            self.model.layers[index].set_weights(
                [np.array(w, copy=True) for w in weights]
            )

def _group_stratified_indices(groups, limit, rng):
    groups = normalize_groups(groups)
    n = len(groups)
    if limit is None or int(limit) <= 0 or int(limit) >= n:
        return np.arange(n, dtype=int)
    limit = int(limit)
    unique = np.unique(groups)
    chosen = []
    per_group = max(1, limit // len(unique))
    for group in unique:
        pool = np.where(groups == group)[0]
        take = min(per_group, len(pool))
        if take:
            chosen.extend(rng.choice(pool, take, replace=False).tolist())
    if len(chosen) < limit:
        remaining = np.setdiff1d(np.arange(n), np.asarray(chosen, dtype=int))
        take = min(limit - len(chosen), len(remaining))
        if take:
            chosen.extend(rng.choice(remaining, take, replace=False).tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen[:limit], dtype=int)


def _random_indices(length, limit, rng):
    if limit is None or int(limit) <= 0 or int(limit) >= length:
        return np.arange(length, dtype=int)
    return np.asarray(
        rng.choice(length, int(limit), replace=False), dtype=int
    )


def _internal_global_best_pso(objective, n_particles, dimensions, bounds,
                              iterations, seed, c1, c2, inertia):
    """Small fallback used only when pyswarms is unavailable."""
    rng = np.random.RandomState(seed)
    lower, upper = bounds
    position = rng.uniform(lower, upper, size=(n_particles, dimensions))
    position[0] = 0.0
    velocity = np.zeros_like(position)
    cost = np.asarray(objective(position), dtype=float)
    pbest_position = position.copy()
    pbest_cost = cost.copy()
    best_index = int(np.argmin(cost))
    gbest_position = position[best_index].copy()
    gbest_cost = float(cost[best_index])
    history = [gbest_cost]

    for _ in range(int(iterations)):
        r1 = rng.random_sample(position.shape)
        r2 = rng.random_sample(position.shape)
        velocity = (
            inertia * velocity
            + c1 * r1 * (pbest_position - position)
            + c2 * r2 * (gbest_position.reshape(1, -1) - position)
        )
        position = np.clip(position + velocity, lower, upper)
        cost = np.asarray(objective(position), dtype=float)
        improved = cost < pbest_cost
        pbest_position[improved] = position[improved]
        pbest_cost[improved] = cost[improved]
        index = int(np.argmin(pbest_cost))
        if float(pbest_cost[index]) < gbest_cost:
            gbest_cost = float(pbest_cost[index])
            gbest_position = pbest_position[index].copy()
        history.append(gbest_cost)
    return gbest_cost, gbest_position, history, 'internal_global_best_pso'


def run_pso(objective, dimensions, particles, iterations, seed,
            c1=0.41, c2=0.41, inertia=0.8):
    lower = np.full((dimensions,), -1.0, dtype=float)
    upper = np.full((dimensions,), 1.0, dtype=float)
    # Source code uses an all-zero initial swarm.
    init_pos = np.zeros((int(particles), int(dimensions)), dtype=float)
    try:
        import pyswarms as ps
        optimizer = ps.single.GlobalBestPSO(
            n_particles=int(particles),
            dimensions=int(dimensions),
            options={'c1': float(c1), 'c2': float(c2), 'w': float(inertia)},
            bounds=(lower, upper),
            init_pos=init_pos,
            ftol=1e-3,
            ftol_iter=10,
        )
        best_cost, best_position = optimizer.optimize(
            objective, iters=int(iterations), verbose=True
        )
        return (
            float(best_cost),
            np.asarray(best_position, dtype=float),
            [float(value) for value in optimizer.cost_history],
            'pyswarms.single.GlobalBestPSO',
        )
    except ImportError as exc:
        raise ModuleNotFoundError(
            'Source-faithful BIRDNN requires pyswarms, exactly as the original '
            'implementation. Install it with: pip install pyswarms'
        ) from exc


def _objective_value(metrics, alpha):
    return float(
        (1.0 - float(alpha)) * float(metrics['defect_loss'])
        + float(alpha) * (1.0 - float(metrics['clean_accuracy']))
    )


def run_birdnn_repair(task, dataset, seed, model_path, bundle, output_dir,
                      repair_neurons=SOURCE_DEFAULT_REPAIR_NEURONS,
                      alpha=0.5, particles=20, iterations=100, eval_sample=0,
                      max_localization_samples=0, batch_size=BATCH_SIZE,
                      c1=0.41, c2=0.41, inertia=0.8):
    """Run one complete TRAIN-only BIRDNN localization+repair stage."""
    task = canonical_task(task)
    dataset = validate_dataset(dataset)
    started = time.time()
    if not os.path.isfile(model_path):
        raise FileNotFoundError('Input model not found: %s' % model_path)
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != tuple(bundle['x_clean'].shape[1:]):
        raise ValueError(
            'Model/data shape mismatch: model=%s, clean=%s'
            % (model.input_shape[1:], bundle['x_clean'].shape[1:])
        )

    print('=' * 100)
    print('BIRDNN-FT source-faithful mechanics, task-adapted TRAIN-only repair')
    print('task=%s, dataset=%s, seed=%d' % (task, dataset, seed))
    print('input_model=%s' % model_path)
    print('train_source=%s' % bundle.get('source', 'unknown'))
    print('=' * 100)

    try:
        localization = localize_birdnn(
            model=model,
            bundle=bundle,
            task=task,
            seed=seed,
            repair_neurons=repair_neurons,
            max_localization_samples=max_localization_samples,
        )
        no_behavior_reason = None
    except NoQualifyingBehaviorSamples as exc:
        # A sequential stage may arrive after the previous stage has already
        # removed every qualifying attack/error sample. In that case the only
        # faithful action is an identity repair, not fabricated localization.
        no_behavior_reason = str(exc)
        localization = {
            'method': 'BIRDNN-FT-source-faithful-task-adapted',
            'task': task,
            'score_formula': 'abs(sum_positive_activation-sum_negative_activation)',
            'positive_negative_equalized': False,
            'top_k_policy': 'identity_no_qualifying_behavior_samples',
            'top_k': 0,
            'total_candidates': 0,
            'players': [],
            'scores': [],
            'selected_players': [],
            'selected_scores': [],
            'selected_candidates': [],
            'behavior_sets': {'error': no_behavior_reason},
            'repairable_layers': discover_repairable_dense_layers(model),
        }
    localization.update({
        'dataset': dataset,
        'seed': int(seed),
        'input_model': os.path.abspath(model_path),
        'train_source': str(bundle.get('source', 'unknown')),
        'test_split_accessed': False,
    })
    localization_path = os.path.join(output_dir, 'localization.json')
    write_json(localization_path, localization)

    selected = localization['selected_candidates']
    if not selected:
        identity_metrics = evaluate_task(model, bundle, task, batch_size=batch_size)
        weights_path = os.path.join(output_dir, 'weights_final.h5')
        model_path_out = os.path.join(output_dir, 'repaired_model.h5')
        model.save_weights(weights_path)
        model.save(model_path_out, include_optimizer=False)
        np.save(os.path.join(output_dir, 'repair_delta_final.npy'), np.asarray([], dtype=float))
        np.save(os.path.join(output_dir, 'repair_scale_final.npy'), np.asarray([], dtype=float))
        np.save(os.path.join(output_dir, 'pso_cost_history.npy'), np.asarray([], dtype=float))
        stats = {
            'protocol_version': 'birdnn_source_faithful_task_adapted_v2',
            'method': 'birdnn',
            'method_variant': 'BIRDNN-FT-source-faithful-task-adapted',
            'task': task,
            'dataset': dataset,
            'seed': int(seed),
            'input_model': os.path.abspath(model_path),
            'output_model': os.path.abspath(model_path_out),
            'weights_final': os.path.abspath(weights_path),
            'train_source': str(bundle.get('source', 'unknown')),
            'repair_scope': 'train_only',
            'checkpoint_selection_scope': 'complete_train',
            'test_split_accessed': False,
            'localization_file': os.path.abspath(localization_path),
            'selected_neuron_count': 0,
            'selected_players': [],
            'repair_items': [],
            'selected_checkpoint_source': 'identity_no_qualifying_behavior_samples',
            'no_behavior_reason': no_behavior_reason,
            'identity_full_train_metrics': identity_metrics,
            'selected_full_train_metrics': identity_metrics,
            'selected_full_train_objective': _objective_value(identity_metrics, alpha),
            'elapsed_sec': float(time.time() - started),
        }
        stats_path = os.path.join(output_dir, 'run_stats.json')
        write_json(stats_path, stats)
        with open(os.path.join(output_dir, 'repair_summary.txt'), 'w', encoding='utf-8') as handle:
            handle.write('method=BIRDNN-FT-source-faithful-task-adapted\n')
            handle.write('task=%s\n' % task)
            handle.write('selected_neuron_count=0\n')
            handle.write('selected_checkpoint_source=identity_no_qualifying_behavior_samples\n')
            handle.write('reason=%s\n' % no_behavior_reason)
            handle.write('repaired_model=%s\n' % model_path_out)
        print('[BIRDNN] No qualifying behavior samples; saved the unchanged model.')
        print('Reason: %s' % no_behavior_reason)
        print('Saved repaired model: %s' % model_path_out)
        return {
            'model': model_path_out,
            'weights': weights_path,
            'localization': localization_path,
            'stats': stats_path,
            'output_dir': output_dir,
        }
    print('Localized Top-%d neurons:' % len(selected))
    for index, item in enumerate(selected):
        print(
            '  %2d. %-24s score=%.10f -> %s'
            % (
                index + 1,
                item['player'],
                item['score'],
                item['next_dense_name'],
            )
        )

    materializer = OutgoingKernelScaleMaterializer(model, selected)
    rng = np.random.RandomState(seed + 104729)
    clean_indices = _group_stratified_indices(
        bundle['g_clean'], eval_sample, rng
    )
    defect_indices = None
    if task in ('adv', 'backdoor'):
        defect_indices = _random_indices(
            len(bundle['x_defect']), eval_sample, rng
        )

    evaluation_counter = {'count': 0}

    def objective(positions):
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        costs = []
        for position in positions:
            materializer.apply_delta(position)
            metrics = evaluate_task(
                model,
                bundle,
                task,
                clean_indices=clean_indices,
                defect_indices=defect_indices,
                batch_size=batch_size,
            )
            costs.append(_objective_value(metrics, alpha))
            evaluation_counter['count'] += 1
        materializer.restore()
        return np.asarray(costs, dtype=float)

    identity_metrics_search = evaluate_task(
        model,
        bundle,
        task,
        clean_indices=clean_indices,
        defect_indices=defect_indices,
        batch_size=batch_size,
    )
    identity_cost_search = _objective_value(identity_metrics_search, alpha)

    best_cost_search, best_delta, history, pso_backend = run_pso(
        objective=objective,
        dimensions=len(selected),
        particles=particles,
        iterations=iterations,
        seed=seed,
        c1=c1,
        c2=c2,
        inertia=inertia,
    )
    materializer.restore()

    # Source-faithful policy: use the PSO global-best position directly.
    # Complete TRAIN metrics are reported, but are not used to replace the PSO
    # solution with the identity model.
    identity_metrics_full = evaluate_task(
        model, bundle, task, batch_size=batch_size
    )
    identity_cost_full = _objective_value(identity_metrics_full, alpha)
    materializer.apply_delta(best_delta)
    best_metrics_full = evaluate_task(
        model, bundle, task, batch_size=batch_size
    )
    best_cost_full = _objective_value(best_metrics_full, alpha)
    materializer.restore()

    selected_delta = np.asarray(best_delta, dtype=float)
    selected_source = 'pso_global_best_source_faithful'
    selected_metrics = best_metrics_full
    selected_cost = best_cost_full

    selected_scale = 1.0 + selected_delta
    materializer.apply_delta(selected_delta)

    weights_path = os.path.join(output_dir, 'weights_final.h5')
    model_path_out = os.path.join(output_dir, 'repaired_model.h5')
    model.save_weights(weights_path)
    model.save(model_path_out, include_optimizer=False)
    np.save(os.path.join(output_dir, 'repair_delta_final.npy'), selected_delta)
    np.save(os.path.join(output_dir, 'repair_scale_final.npy'), selected_scale)
    np.save(os.path.join(output_dir, 'pso_cost_history.npy'), np.asarray(history))

    repair_items = []
    for item, delta, scale in zip(selected, selected_delta, selected_scale):
        record = dict(item)
        record['delta'] = float(delta)
        record['scale'] = float(scale)
        repair_items.append(record)

    stats = {
        'protocol_version': 'birdnn_source_faithful_task_adapted_v2',
        'method': 'birdnn',
        'method_variant': 'BIRDNN-FT-source-faithful-task-adapted',
        'task': task,
        'dataset': dataset,
        'seed': int(seed),
        'input_model': os.path.abspath(model_path),
        'output_model': os.path.abspath(model_path_out),
        'weights_final': os.path.abspath(weights_path),
        'train_source': str(bundle.get('source', 'unknown')),
        'repair_scope': 'train_only',
        'checkpoint_selection_scope': 'complete_train',
        'test_split_accessed': False,
        'localization_file': os.path.abspath(localization_path),
        'localization_formula': (
            'abs(sum_positive_activation-sum_negative_activation)'
        ),
        'selected_neuron_count': int(len(selected)),
        'selected_players': localization['selected_players'],
        'repair_items': repair_items,
        'repair_parameterization': 'activation_scale_equals_1_plus_delta',
        'delta_bounds': [-1.0, 1.0],
        'scale_bounds': [0.0, 2.0],
        'materialization': (
            'equivalent_outgoing_dense_kernel_row_scaling'
        ),
        'source_faithful_settings': {
            'repair_num': int(repair_neurons),
            'positive_negative_equalized': False,
            'localization_aggregation': 'independent_sum',
            'all_zero_initial_swarm': True,
            'use_pso_best_without_identity_fallback': True,
            'full_training_fitness_by_default': bool(eval_sample is None or int(eval_sample) <= 0),
            'full_localization_sets_by_default': bool(max_localization_samples is None or int(max_localization_samples) <= 0),
            'ftol': 1e-3,
            'ftol_iter': 10,
        },
        'alpha': float(alpha),
        'fitness': (
            '(1-alpha)*defect_loss+alpha*(1-clean_accuracy)'
        ),
        'particles': int(particles),
        'iterations': int(iterations),
        'pso_c1': float(c1),
        'pso_c2': float(c2),
        'pso_inertia': float(inertia),
        'pso_backend': pso_backend,
        'pso_evaluations': int(evaluation_counter['count']),
        'pso_best_cost_on_search_subset': float(best_cost_search),
        'identity_cost_on_search_subset': float(identity_cost_search),
        'search_clean_sample_count': int(len(clean_indices)),
        'search_defect_sample_count': (
            int(len(defect_indices)) if defect_indices is not None else 0
        ),
        'identity_full_train_metrics': identity_metrics_full,
        'pso_best_full_train_metrics': best_metrics_full,
        'selected_checkpoint_source': selected_source,
        'selected_full_train_metrics': selected_metrics,
        'selected_full_train_objective': float(selected_cost),
        'elapsed_sec': float(time.time() - started),
    }
    stats_path = os.path.join(output_dir, 'run_stats.json')
    write_json(stats_path, stats)

    with open(
        os.path.join(output_dir, 'repair_summary.txt'),
        'w', encoding='utf-8'
    ) as handle:
        handle.write('method=BIRDNN-FT-source-faithful-task-adapted\n')
        handle.write('task=%s\n' % task)
        handle.write('dataset=%s\n' % dataset)
        handle.write('seed=%d\n' % seed)
        handle.write('test_split_accessed=false\n')
        handle.write('selected_neuron_count=%d\n' % len(selected))
        handle.write('selected_checkpoint_source=%s\n' % selected_source)
        handle.write('identity_full_train=%s\n' % identity_metrics_full)
        handle.write('pso_best_full_train=%s\n' % best_metrics_full)
        handle.write('selected_full_train=%s\n' % selected_metrics)
        handle.write('repaired_model=%s\n' % model_path_out)

    print('-' * 100)
    print('Original complete-TRAIN metrics: %s' % identity_metrics_full)
    print('PSO-best complete-TRAIN metrics: %s' % best_metrics_full)
    print('Selected checkpoint: %s' % selected_source)
    print('Selected scales: %s' % selected_scale.tolist())
    print('Saved localization: %s' % localization_path)
    print('Saved repaired model: %s' % model_path_out)
    print('Saved final weights: %s' % weights_path)
    print('Saved stats: %s' % stats_path)
    print('Repair completed without loading a test split.')
    print('=' * 100)

    return {
        'model': model_path_out,
        'weights': weights_path,
        'localization': localization_path,
        'stats': stats_path,
        'output_dir': output_dir,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'BIRDNN-FT behavior-difference localization and PSO repair for '
            'tabular single defects. TRAIN split only.'
        )
    )
    parser.add_argument('task', choices=sorted(TASK_ALIASES))
    parser.add_argument('dataset', choices=SUPPORTED_DATASETS)
    parser.add_argument('seed', type=int)
    parser.add_argument('--model-root', default='../model')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--combined-root', default='../data/combined_defects')
    parser.add_argument('--output-root', default='loc_result')
    parser.add_argument('--result-suffix', default='source20')
    parser.add_argument(
        '--repair-neurons', '--repair-neurons-num',
        dest='repair_neurons', type=int,
        default=SOURCE_DEFAULT_REPAIR_NEURONS,
        help='BIRDNN repair_num; source full-network default is 20.',
    )
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--particles', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--eval-sample', type=int, default=0, help='<=0 uses the complete TRAIN set, matching the source workflow.')
    parser.add_argument('--localization-sample', type=int, default=0, help='<=0 uses all qualifying positive/negative TRAIN samples independently.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--pso-c1', type=float, default=0.41)
    parser.add_argument('--pso-c2', type=float, default=0.41)
    parser.add_argument('--pso-w', type=float, default=0.8)
    return parser


def main():
    args = build_parser().parse_args()
    task = canonical_task(args.task)
    dataset = validate_dataset(args.dataset)
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError('--alpha must be in [0, 1].')
    if args.repair_neurons <= 0:
        raise ValueError('--repair-neurons-num must be positive; source default is 20.')
    if args.particles < 2 or args.iterations < 1:
        raise ValueError('--particles >= 2 and --iterations >= 1 are required.')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive.')

    configure_tensorflow(args.seed)
    bundle = load_single_task_bundle(
        dataset=dataset,
        task=task,
        seed=args.seed,
        combined_root=args.combined_root,
    )
    model_path = args.model_path or resolve_default_model_path(
        dataset, task, args.model_root
    )
    suffix = ('_%s' % args.result_suffix) if args.result_suffix else ''
    output_dir = os.path.join(
        args.output_root,
        dataset,
        str(args.seed),
        'singledefect_birdnn_%s%s' % (task, suffix),
    )
    run_birdnn_repair(
        task=task,
        dataset=dataset,
        seed=args.seed,
        model_path=model_path,
        bundle=bundle,
        output_dir=output_dir,
        repair_neurons=args.repair_neurons,
        alpha=args.alpha,
        particles=args.particles,
        iterations=args.iterations,
        eval_sample=args.eval_sample,
        max_localization_samples=args.localization_sample,
        batch_size=args.batch_size,
        c1=args.pso_c1,
        c2=args.pso_c2,
        inertia=args.pso_w,
    )


if __name__ == '__main__':
    main()