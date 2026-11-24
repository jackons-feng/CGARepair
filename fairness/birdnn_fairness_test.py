"""BIRDNN-FT fairness evaluation: strict TEST-only stage.

This script evaluates exactly two models on the complete fairness TEST split:
1) the untouched original model; and
2) the BIRDNN weights_final.h5 fixed during the TRAIN-only repair stage.

It never loads the training split and never selects a checkpoint using test data.

Usage:
    python -B birdnn_fairness_test.py <model_name> <seed> [result_suffix]

Examples:
    python -B birdnn_fairness_test.py bank 256
    python -B birdnn_fairness_test.py adult 256 source20
    python -B birdnn_fairness_test.py default 256 source20

Default repair directory:
    loc_result/<model>/<seed>/singledefect_birdnn_fairness_source20/
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model

from repair_config import load_fairness_data, value


SUPPORTED_DATASETS = ('bank', 'adult', 'default')
DEFAULT_RESULT_SUFFIX = 'source20'
BATCH_SIZE = 1024


def print_sep(ch='=', n=100):
    print(ch * n)


def configure_tensorflow(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.disable_eager_execution()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    )
    K.set_session(session)
    return session


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
            'Fairness testing requires exactly two sensitive groups; got %s.'
            % unique.tolist()
        )
    if set(unique.tolist()) != {0, 1}:
        mapping = {int(unique[0]): 0, int(unique[1]): 1}
        groups = np.asarray(
            [mapping[int(item)] for item in groups], dtype=np.int64
        )
    return groups


def load_complete_test_split(model_name, seed):
    """Load only the complete TEST split; sample_size=None is mandatory."""
    attempts = [
        lambda: load_fairness_data(
            model_name,
            split='test',
            sample_size=None,
            seed=seed,
            return_groups=True,
        ),
        lambda: load_fairness_data(
            model_name=model_name,
            split='test',
            sample_size=None,
            seed=seed,
            return_groups=True,
        ),
    ]

    errors = []
    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict):
                x = result.get('x_clean', result.get('x'))
                y = result.get('y_clean', result.get('y'))
                g = result.get('g_clean', result.get('g'))
            else:
                x, y, g = result[:3]

            x = np.asarray(x, dtype=np.float32)
            y = normalize_labels(y)
            g = normalize_groups(g)
            if not (len(x) == len(y) == len(g)):
                raise ValueError(
                    'TEST split length mismatch: x=%d, y=%d, g=%d'
                    % (len(x), len(y), len(g))
                )
            if len(x) == 0:
                raise ValueError('The complete TEST split is empty.')
            return x, y, g
        except Exception as exc:
            errors.append(repr(exc))

    raise RuntimeError(
        'Failed to load the complete fairness TEST split for %s. Errors: %s'
        % (model_name, errors)
    )


def build_result_dir(output_root, model_name, seed, result_suffix):
    suffix = ('_%s' % result_suffix) if result_suffix else ''
    return os.path.join(
        output_root,
        model_name,
        str(seed),
        'singledefect_birdnn_fairness%s' % suffix,
    )


def predict_labels(model, x, batch_size=BATCH_SIZE):
    outputs = []
    session = K.get_session()
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        outputs.append(session.run(model.output, {model.input: batch}))
    output = np.concatenate(outputs, axis=0)
    if output.ndim == 1:
        return (output >= 0.5).astype(np.int64)
    if output.ndim == 2 and output.shape[-1] == 1:
        return (output[:, 0] >= 0.5).astype(np.int64)
    return np.argmax(output, axis=-1).astype(np.int64)


def binary_f1(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def evaluate(model, x, y, g, model_name):
    """Use the project's official fairness/accuracy metrics plus diagnostics."""
    fairness_gap = float(value(
        model,
        x,
        y,
        group_labels=g,
        metric='fairness',
        dataset=model_name,
    ))
    accuracy = float(value(
        model,
        x,
        y,
        group_labels=g,
        metric='accuracy',
        dataset=model_name,
    ))

    pred = predict_labels(model, x)
    group_metrics = {}
    for group in sorted(np.unique(g).tolist()):
        mask = g == group
        group_metrics[str(int(group))] = {
            'count': int(np.sum(mask)),
            'accuracy': float(np.mean(pred[mask] == y[mask])),
        }

    return {
        'fairness_gap': fairness_gap,
        'clean_accuracy': accuracy,
        'group_metrics': group_metrics,
        'binary_classification': binary_f1(y, pred),
    }


def metric_change(original, repaired):
    return {
        'fairness_gap_change_repaired_minus_original': float(
            repaired['fairness_gap'] - original['fairness_gap']
        ),
        'fairness_gap_improvement_original_minus_repaired': float(
            original['fairness_gap'] - repaired['fairness_gap']
        ),
        'accuracy_change_repaired_minus_original': float(
            repaired['clean_accuracy'] - original['clean_accuracy']
        ),
        'f1_change_repaired_minus_original': float(
            repaired['binary_classification']['f1']
            - original['binary_classification']['f1']
        ),
    }


def load_repair_metadata(result_dir, model_name, seed):
    path = os.path.join(result_dir, 'run_stats.json')
    if not os.path.isfile(path):
        return path, {}
    with open(path, 'r', encoding='utf-8') as handle:
        metadata = json.load(handle)

    if str(metadata.get('method', '')).lower() not in ('', 'birdnn'):
        raise ValueError('Unexpected repair method in %s: %r' % (
            path, metadata.get('method')
        ))
    if str(metadata.get('task', '')).lower() not in ('', 'fairness'):
        raise ValueError('Expected fairness BIRDNN result, got task=%r.' % (
            metadata.get('task'),
        ))
    if metadata.get('dataset') not in (None, model_name):
        raise ValueError(
            'Repair metadata dataset mismatch: requested=%s, metadata=%r.'
            % (model_name, metadata.get('dataset'))
        )
    if metadata.get('seed') not in (None, seed):
        raise ValueError(
            'Repair metadata seed mismatch: requested=%d, metadata=%r.'
            % (seed, metadata.get('seed'))
        )
    if metadata.get('test_split_accessed') is True:
        raise RuntimeError(
            'Repair metadata says the TEST split was accessed during repair.'
        )
    return path, metadata


def write_text_result(path, result):
    original = result['original_test_metrics']
    repaired = result['repaired_test_metrics']
    changes = result['changes']
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('protocol_version=%s\n' % result['protocol_version'])
        handle.write('method=birdnn\n')
        handle.write('task=fairness\n')
        handle.write('model=%s\n' % result['model'])
        handle.write('seed=%d\n' % result['seed'])
        handle.write('result_suffix=%s\n' % result['result_suffix'])
        handle.write('repair_scope=%s\n' % result['repair_scope'])
        handle.write('evaluation_scope=complete_test\n')
        handle.write('training_split_accessed=false\n')
        handle.write('test_used_for_model_selection=false\n')
        handle.write('evaluated_weight_file=weights_final.h5\n')
        handle.write('test_sample_count=%d\n' % result['test_sample_count'])
        handle.write('original_fairness_gap=%.10f\n' % original['fairness_gap'])
        handle.write('original_accuracy=%.10f\n' % original['clean_accuracy'])
        handle.write('original_f1=%.10f\n' % original['binary_classification']['f1'])
        handle.write('repaired_fairness_gap=%.10f\n' % repaired['fairness_gap'])
        handle.write('repaired_accuracy=%.10f\n' % repaired['clean_accuracy'])
        handle.write('repaired_f1=%.10f\n' % repaired['binary_classification']['f1'])
        handle.write(
            'fairness_gap_change=%+.10f\n'
            % changes['fairness_gap_change_repaired_minus_original']
        )
        handle.write(
            'fairness_gap_improvement=%+.10f\n'
            % changes['fairness_gap_improvement_original_minus_repaired']
        )
        handle.write(
            'accuracy_change=%+.10f\n'
            % changes['accuracy_change_repaired_minus_original']
        )
        handle.write(
            'f1_change=%+.10f\n'
            % changes['f1_change_repaired_minus_original']
        )
        for prefix, metrics in (('original', original), ('repaired', repaired)):
            for group, group_info in sorted(metrics['group_metrics'].items()):
                handle.write(
                    '%s_group_%s_accuracy=%.10f\n'
                    % (prefix, group, group_info['accuracy'])
                )


def build_parser():
    parser = argparse.ArgumentParser(
        description='Strict TEST-only evaluation for BIRDNN fairness repair.'
    )
    parser.add_argument('model_name', choices=SUPPORTED_DATASETS)
    parser.add_argument('seed', type=int)
    parser.add_argument(
        'result_suffix', nargs='?', default=DEFAULT_RESULT_SUFFIX,
        help='Repair suffix; default: source20.',
    )
    parser.add_argument('--model-root', default='../model')
    parser.add_argument('--output-root', default='loc_result')
    parser.add_argument(
        '--result-dir', default=None,
        help='Explicit BIRDNN result directory; overrides output-root/suffix.',
    )
    return parser


def main():
    args = build_parser().parse_args()
    started = time.time()
    configure_tensorflow(args.seed)

    model_path = os.path.join(args.model_root, '%s.h5' % args.model_name)
    result_dir = args.result_dir or build_result_dir(
        args.output_root,
        args.model_name,
        args.seed,
        args.result_suffix,
    )
    final_weight_path = os.path.join(result_dir, 'weights_final.h5')

    if not os.path.isfile(model_path):
        raise FileNotFoundError('Original model not found: %s' % model_path)
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(
            'BIRDNN result directory not found: %s\n'
            'Use the same seed and result suffix as birdnn_single_repair.py.'
            % result_dir
        )
    if not os.path.isfile(final_weight_path):
        raise FileNotFoundError(
            'Preselected BIRDNN weights_final.h5 not found: %s'
            % final_weight_path
        )

    metadata_path, repair_metadata = load_repair_metadata(
        result_dir, args.model_name, args.seed
    )

    # This is the only dataset read performed by the test script.
    test_x, test_y, test_g = load_complete_test_split(
        args.model_name, args.seed
    )
    unique_groups, group_counts = np.unique(test_g, return_counts=True)
    test_group_counts = {
        str(int(group)): int(count)
        for group, count in zip(unique_groups, group_counts)
    }

    model = load_model(model_path, compile=False)
    original_metrics = evaluate(
        model, test_x, test_y, test_g, args.model_name
    )

    # Evaluate exactly the checkpoint fixed before TEST evaluation.
    model.load_weights(final_weight_path)
    repaired_metrics = evaluate(
        model, test_x, test_y, test_g, args.model_name
    )
    changes = metric_change(original_metrics, repaired_metrics)

    result = {
        'protocol_version': 'birdnn_strict_test_evaluation_v1',
        'method': 'birdnn',
        'method_variant': repair_metadata.get(
            'method_variant', 'BIRDNN-FT-source-faithful-task-adapted'
        ),
        'task': 'fairness',
        'model': args.model_name,
        'seed': int(args.seed),
        'result_suffix': args.result_suffix,
        'result_dir': os.path.abspath(result_dir),
        'repair_scope': repair_metadata.get('repair_scope', 'unknown'),
        'repair_protocol_version': repair_metadata.get('protocol_version'),
        'selected_neuron_count': repair_metadata.get('selected_neuron_count'),
        'selected_checkpoint_source': repair_metadata.get(
            'selected_checkpoint_source'
        ),
        'evaluation_scope': 'complete_test',
        'training_split_accessed': False,
        'test_used_for_model_selection': False,
        'evaluated_weight_file': 'weights_final.h5',
        'repair_metadata_file': (
            os.path.abspath(metadata_path)
            if os.path.isfile(metadata_path) else None
        ),
        'test_sample_count': int(len(test_x)),
        'test_group_counts': test_group_counts,
        'original_test_metrics': original_metrics,
        'repaired_test_metrics': repaired_metrics,
        'changes': changes,
        'elapsed_sec': float(time.time() - started),
    }

    json_path = os.path.join(result_dir, 'full_test_results.json')
    txt_path = os.path.join(result_dir, 'full_test_results.txt')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    write_text_result(txt_path, result)

    print_sep()
    print('BIRDNN strict TEST-only fairness evaluation')
    print_sep()
    print('Model: %s' % args.model_name)
    print('Seed: %d' % args.seed)
    print('Result suffix: %s' % (args.result_suffix or '<none>'))
    print('Result directory: %s' % result_dir)
    print('Complete TEST samples: %d' % len(test_x))
    print('TEST group counts: %s' % test_group_counts)
    print('Evaluated repaired checkpoint: weights_final.h5')
    print('No TEST checkpoint selection was performed.')
    print_sep('-')
    print(
        'Original full-test: fairness_gap=%.10f, accuracy=%.10f, f1=%.10f'
        % (
            original_metrics['fairness_gap'],
            original_metrics['clean_accuracy'],
            original_metrics['binary_classification']['f1'],
        )
    )
    print(
        'Repaired full-test: fairness_gap=%.10f, accuracy=%.10f, f1=%.10f'
        % (
            repaired_metrics['fairness_gap'],
            repaired_metrics['clean_accuracy'],
            repaired_metrics['binary_classification']['f1'],
        )
    )
    for group in sorted(original_metrics['group_metrics']):
        print(
            'Group %s accuracy: original=%.10f, repaired=%.10f'
            % (
                group,
                original_metrics['group_metrics'][group]['accuracy'],
                repaired_metrics['group_metrics'][group]['accuracy'],
            )
        )
    print(
        'Fairness gap change (repaired-original): %+.10f'
        % changes['fairness_gap_change_repaired_minus_original']
    )
    print(
        'Fairness improvement (original-repaired): %+.10f'
        % changes['fairness_gap_improvement_original_minus_repaired']
    )
    print(
        'Accuracy change: %+.10f'
        % changes['accuracy_change_repaired_minus_original']
    )
    print(
        'F1 change: %+.10f'
        % changes['f1_change_repaired_minus_original']
    )
    print_sep('-')
    print('Saved: %s' % txt_path)
    print('Saved: %s' % json_path)
    print('Done in %.2fs' % (time.time() - started))
    print_sep()


if __name__ == '__main__':
    main()
