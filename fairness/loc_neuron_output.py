"""
Two-defect neuron localization: backdoor + adversarial (output zero-out).

Defect types: backdoor, adv (auto-detected from data keys).

Step1: gradient-based importance per defect type (CE gradient x activation)
Step2: output zero-out causal validation (model splitting), per defect type
Decision & union: adaptive based on which defect types are present

This is a rewrite based on threedefect/loc_neuron_output.py, using model
splitting for step2 instead of random weight perturbation. Much faster.

NOTE: Does NOT create a new TF session. The session is managed by
repair_config.py. This avoids the session-conflict bug that caused
"Resource variable ... does not exist" errors.
"""

import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input

from repair_config import (
    DATASET_CONFIG
)

BATCH_SIZE_LOC = 128
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# Evaluation helpers
# ============================================================

def sess_run(model, variable, images, batch_size=BATCH_SIZE_LOC):
    output = []
    num_batches = int(np.ceil(len(images) / batch_size))
    for batch in range(num_batches):
        batch_idxs = np.arange(batch * batch_size, min((batch + 1) * batch_size, len(images)))
        input_dic = {model.input: images[batch_idxs]}
        output.append(K.get_session().run(variable, input_dic))
    try:
        return np.concatenate(output, 0)
    except Exception:
        return np.array(output)


def value_accuracy(model, images, labels):
    outs = sess_run(model, model.output, images)
    pred = np.argmax(outs, axis=-1)
    return float(np.sum(pred == labels) / len(labels))


def value_backdoor(model, images_p, labels_clean, target_label):
    pred_p = np.argmax(sess_run(model, model.output, images_p), axis=-1)
    return float((np.sum(pred_p == target_label) - np.sum((labels_clean == target_label) & (pred_p == target_label))) / len(images_p))


def value_adv_error(model, images_adv, labels_true):
    pred_adv = np.argmax(sess_run(model, model.output, images_adv), axis=-1)
    return float(np.sum(pred_adv != labels_true) / len(labels_true))


# ============================================================
# Layer / neuron utilities
# ============================================================

def get_target_layers(model):
    target_layers = []
    for layer in model.layers:
        layer_type = type(layer).__name__.lower()
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        if 'conv' not in layer_type and 'dense' not in layer_type:
            continue
        target_layers.append({'layer': layer, 'name': layer.name, 'type': type(layer).__name__})
    return target_layers


def get_layer_weight_indices(model):
    conv_meta = []
    dense_meta = []
    j = 0
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        layer_type_lower = layer_type.lower()
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        is_conv = isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer_type_lower
        is_dense = isinstance(layer, tf.keras.layers.Dense) or 'dense' in layer_type_lower
        if is_conv or is_dense:
            meta = {'j': j, 'index': i, 'name': layer.name, 'type': layer_type, 'num_neurons': weights[0].shape[-1]}
            if is_conv:
                conv_meta.append(meta)
            else:
                dense_meta.append(meta)
        j += len(weights)
    return conv_meta, dense_meta


def get_neuron_players(model, conv_meta, dense_meta):
    players = []
    weights = model.get_weights()
    for meta in conv_meta:
        players.append([f'conv_{meta["name"]}_{i}' for i in range(weights[meta['j']].shape[-1])])
    for meta in dense_meta:
        players.append([f'dense_{meta["name"]}_{i}' for i in range(weights[meta['j']].shape[-1])])
    return np.sort(np.concatenate(players)) if players else np.array([])


def parse_neuron_player(neuron_name):
    parts = neuron_name.split('_')
    return parts[0].lower(), '_'.join(parts[1:-1]), int(parts[-1])


# ============================================================
# Model splitting for output perturbation
# ============================================================

def split_model_at_layer(model, layer_index):
    """
    Split the model at layer_index for output perturbation.

    model1: input -> layers[layer_index].output  (includes the target layer)
    model2: layers[layer_index].output -> final output

    Returns (model1, model2, skip_indices) where:
      - model1: input -> target layer output
      - model2: takes (target_output, *skip_connection_outputs) -> final output
      - skip_indices: list of layer indices whose outputs are needed as
        extra inputs to model2 (for ResNet skip connections). Empty if
        the model is purely sequential.
    """
    model1 = Model(inputs=model.inputs, outputs=model.layers[layer_index].output)

    output_shape = model.layers[layer_index].output_shape[1:]
    main_input = Input(shape=output_shape)

    # Map from original layer index -> tensor in model2
    tensor_map = {layer_index: main_input}
    skip_inputs = []       # list of Input tensors
    skip_indices = []      # list of original layer indices

    x = main_input
    for i in range(layer_index + 1, len(model.layers)):
        layer = model.layers[i]
        if not layer.inbound_nodes:
            continue

        node = layer.inbound_nodes[0]
        inbound_layers = node.inbound_layers
        # Normalize: inbound_layers can be a single layer or a list
        if not isinstance(inbound_layers, (list, tuple)):
            inbound_layers = [inbound_layers]

        if len(inbound_layers) == 1:
            # Single-input layer: look up the correct input from tensor_map,
            # NOT assume sequential flow (ResNet has branches).
            src_idx = _find_layer_index(model.layers, inbound_layers[0])
            if src_idx in tensor_map:
                inp = tensor_map[src_idx]
            elif src_idx < layer_index:
                # Skip connection from before the split point
                if src_idx not in skip_indices:
                    src_shape = inbound_layers[0].output_shape[1:]
                    skip_inp = Input(shape=src_shape, name=f'skip_from_l{src_idx}')
                    skip_inputs.append(skip_inp)
                    skip_indices.append(src_idx)
                    tensor_map[src_idx] = skip_inp
                inp = tensor_map[src_idx]
            else:
                raise RuntimeError(
                    f'Layer {i} depends on unprocessed layer {src_idx}')
            x = layer(inp)
            tensor_map[i] = x
        else:
            # Multi-input layer (e.g. Add in ResNet skip connections)
            layer_inputs = []
            for inb in inbound_layers:
                src_idx = _find_layer_index(model.layers, inb)
                if src_idx in tensor_map:
                    layer_inputs.append(tensor_map[src_idx])
                elif src_idx < layer_index:
                    # Skip connection from before the split point
                    if src_idx not in skip_indices:
                        src_shape = inb.output_shape[1:]
                        skip_inp = Input(shape=src_shape, name=f'skip_from_l{src_idx}')
                        skip_inputs.append(skip_inp)
                        skip_indices.append(src_idx)
                        tensor_map[src_idx] = skip_inp
                    layer_inputs.append(tensor_map[src_idx])
                else:
                    raise RuntimeError(
                        f'Layer {i} depends on unprocessed layer {src_idx}')
            x = layer(layer_inputs)
            tensor_map[i] = x

    all_inputs = [main_input] + skip_inputs
    model2 = Model(inputs=all_inputs, outputs=x)

    return model1, model2, skip_indices


def _find_layer_index(layers, target):
    """Find the index of a layer in the layers list."""
    for j, l in enumerate(layers):
        if l is target:
            return j
    return -1


# ============================================================
# Defect-type → metric function for step2 causal validation
# ============================================================

def _metric_backdoor_rate(logits, labels, target_label):
    pred = np.argmax(logits, axis=-1)
    return float((np.sum(pred == target_label)
                  - np.sum((labels == target_label) & (pred == target_label)))
                 / max(1, len(labels)))


def _metric_adv_error(logits, labels, _target_label=None):
    pred = np.argmax(logits, axis=-1)
    return float(np.sum(pred != labels) / max(1, len(labels)))


DEFECT_METRIC_FN = {
    'backdoor': _metric_backdoor_rate,
    'adv': _metric_adv_error,
}


# ============================================================
# Step 1: Gradient-based neuron importance
# ============================================================

def compute_step1_scores(model, defect_data, target_layers):
    """Compute step1 neuron importance for backdoor and adv."""
    y_ph = tf.compat.v1.placeholder(tf.int32, shape=(None,))

    logits = model.output
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits))

    layer_outputs = [info['layer'].output for info in target_layers]
    grads = tf.gradients(loss, layer_outputs)

    sess = K.get_session()

    def _neuron_names(info, n_neurons):
        prefix = 'conv' if 'conv' in info['type'].lower() else 'dense'
        lname = info['layer'].name
        return [f"{prefix}_{lname}_{j}" for j in range(n_neurons)]

    s1_scores = {}
    for dtype, (x_data, y_data) in defect_data.items():
        print(f'  Step1: {dtype} (n={len(x_data)})')

        vals, acts = sess.run(
            [grads, layer_outputs],
            {model.input: x_data, y_ph: y_data})

        s1_scores[dtype] = {}
        for i, info in enumerate(target_layers):
            imp = np.abs(vals[i] * acts[i])
            if 'conv' in info['type'].lower():
                imp = np.mean(imp, axis=(0, 1, 2))
            else:
                imp = np.mean(imp, axis=0)
            for name, val in zip(_neuron_names(info, len(imp)), imp):
                s1_scores[dtype][name] = float(val)

    all_vals = []
    for d_scores in s1_scores.values():
        all_vals.extend(d_scores.values())
    step1_stats = {
        'total_neurons_step1': int(len(all_vals)),
        'nonzero_count_step1': int(np.count_nonzero(np.array(all_vals) > 0.0)) if all_vals else 0,
        'nonzero_ratio_step1': float(np.count_nonzero(np.array(all_vals) > 0.0) / max(1, len(all_vals))) if all_vals else 0.0,
        'defect_types': list(s1_scores.keys()),
    }
    return s1_scores, step1_stats


# ============================================================
# Decision helper
# ============================================================

def build_single_task_decision(scores_top, ratio_threshold=10.0, zero_eps=1e-5):
    vals = np.array(scores_top, dtype=np.float64)
    top1 = float(vals[0]) if len(vals) >= 1 else 0.0
    top2 = float(vals[1]) if len(vals) >= 2 else 0.0

    if len(vals) < 2:
        return {'recommended_mode': 'full', 'top1': top1, 'top2': top2, 'ratio': None,
                'reason': 'insufficient_top_scores(<2)'}
    if top2 <= zero_eps:
        return {'recommended_mode': 'full', 'top1': top1, 'top2': top2, 'ratio': None,
                'reason': 'top2<=zero_eps'}

    ratio = float(top1 / top2)
    if ratio > ratio_threshold:
        return {'recommended_mode': 'neuron', 'top1': top1, 'top2': top2, 'ratio': ratio,
                'reason': f'top1/top2={ratio:.6f}>{ratio_threshold}'}
    return {'recommended_mode': 'full', 'top1': top1, 'top2': top2, 'ratio': ratio,
            'reason': f'top1/top2={ratio:.6f}<={ratio_threshold}'}


# ============================================================
# Step 2: Causal validation via OUTPUT zero-out perturbation
# ============================================================

def validate_step2_output_zeroout(model, candidates_sorted, conv_meta, dense_meta,
                                   defect_data, target_label, n_top=200):
    """Step2: output zero-out intervention. Much faster than random weight perturbation."""
    top_candidates = candidates_sorted[:n_top]
    sess = K.get_session()

    # Group candidates by layer index
    layer_candidates = {}
    for neuron_name, _ in top_candidates:
        layer_type, layer_name, neuron_idx = parse_neuron_player(neuron_name)
        meta = next((m for m in (conv_meta if layer_type == 'conv' else dense_meta)
                     if m['name'] == layer_name), None)
        if meta is None:
            continue
        layer_index = meta['index']
        layer_candidates.setdefault(layer_index, []).append(
            (neuron_name, neuron_idx, layer_type, layer_name))

    clean_data = defect_data.get('clean')
    if clean_data is None:
        raise ValueError('defect_data must contain "clean" key for step2 validation')

    x_clean, y_clean = clean_data
    defect_types = [k for k in defect_data if k != 'clean']

    s2_scores = {dtype: {} for dtype in defect_types}

    for layer_index, candidates in layer_candidates.items():
        layer_name = model.layers[layer_index].name
        print(f'  Step2: layer {layer_index} ({layer_name}), {len(candidates)} neurons')

        model1, model2, skip_indices = split_model_at_layer(model, layer_index)

        # ---- Build feed dict helpers for model2 ----
        if skip_indices:
            # Build extended model1 that also returns skip connection outputs
            model1_outputs = [model.layers[layer_index].output] + [
                model.layers[idx].output for idx in skip_indices]
            model1_ext = Model(inputs=model.inputs, outputs=model1_outputs)

            def _make_feed(layer_out, skip_outs):
                feed = {model2.inputs[0]: layer_out}
                for i, so in enumerate(skip_outs):
                    feed[model2.inputs[i + 1]] = so
                return feed

            def _run_model2(layer_out, skip_outs):
                return sess.run(model2.output, _make_feed(layer_out, skip_outs))

            # Compute model1 outputs once for all data types
            outs_clean = sess.run(model1_ext.outputs, {model1_ext.input: x_clean})
            layer_out_clean = outs_clean[0]
            skip_outs_clean = outs_clean[1:]

            layer_out_by_defect = {}
            skip_outs_by_defect = {}
            for dtype in defect_types:
                outs_d = sess.run(model1_ext.outputs, {model1_ext.input: defect_data[dtype][0]})
                layer_out_by_defect[dtype] = outs_d[0]
                skip_outs_by_defect[dtype] = outs_d[1:]
        else:
            # Simple sequential model: single-input model2
            layer_out_clean = sess.run(model1.output, {model1.input: x_clean})
            layer_out_by_defect = {}
            for dtype in defect_types:
                layer_out_by_defect[dtype] = sess.run(
                    model1.output, {model1.input: defect_data[dtype][0]})

        # ---- Original model2 outputs (no intervention) ----
        if skip_indices:
            orig_clean = _run_model2(layer_out_clean, skip_outs_clean)
        else:
            orig_clean = sess.run(model2.output, {model2.input: layer_out_clean})
        orig_clean_acc = float(np.mean(np.argmax(orig_clean, axis=-1) == y_clean))

        orig_metric = {}
        for dtype in defect_types:
            if dtype in DEFECT_METRIC_FN:
                if skip_indices:
                    orig_out = _run_model2(layer_out_by_defect[dtype],
                                           skip_outs_by_defect[dtype])
                else:
                    orig_out = sess.run(model2.output,
                                        {model2.input: layer_out_by_defect[dtype]})
                orig_metric[dtype] = DEFECT_METRIC_FN[dtype](
                    orig_out, defect_data[dtype][1], target_label)
            else:
                orig_metric[dtype] = 0.0

        # ---- Evaluate each neuron ----
        for neuron_name, neuron_idx, layer_type, _ in candidates:
            # Zero out this neuron's channel in clean activations
            z_clean = layer_out_clean.copy()
            z_clean[..., neuron_idx] = 0
            if skip_indices:
                z_pred_clean = _run_model2(z_clean, skip_outs_clean)
            else:
                z_pred_clean = sess.run(model2.output, {model2.input: z_clean})
            z_clean_acc = float(np.mean(np.argmax(z_pred_clean, axis=-1) == y_clean))
            delta_acc = abs(z_clean_acc - orig_clean_acc)

            # Zero out this neuron's channel for each defect type
            for dtype in defect_types:
                if dtype not in DEFECT_METRIC_FN:
                    continue
                z_defect = layer_out_by_defect[dtype].copy()
                z_defect[..., neuron_idx] = 0
                if skip_indices:
                    z_pred_defect = _run_model2(z_defect, skip_outs_by_defect[dtype])
                else:
                    z_pred_defect = sess.run(model2.output,
                                             {model2.input: z_defect})
                z_metric = DEFECT_METRIC_FN[dtype](
                    z_pred_defect, defect_data[dtype][1], target_label)
                delta_defect = abs(z_metric - orig_metric[dtype])
                s2_scores[dtype][neuron_name] = float(
                    delta_defect / (delta_acc + 1e-5))

    return s2_scores


# ============================================================
# Main computation: two-defect neuron importance with union strategy
# ============================================================

def compute_neurons_importance(model_name, k_sample=512, epsilon=0.05,
                                topk_per_defect=11, neuron_ratio=0.1,
                                model_override=None, defect_data_override=None):
    """
    Compute neuron importance for two defect types (backdoor + adv).

    Strategy:
    - Step1: gradient-based importance separately for each defect
    - Step2: output zero-out causal validation (model splitting)
    - Per-defect final score = normalized_s1 * s2
    - Decision: if any defect recommends full -> full repair; else neuron repair
    - Union: top-k neurons from each defect are unioned

    Args:
        model_name: model identifier (e.g. 'fashion', 'gtsrb', 'gtsrb_resnet')
        k_sample: sample size (None for all)
        epsilon: adversarial epsilon
        topk_per_defect: top-k neurons per defect for union
        neuron_ratio: ratio of neurons for step2 validation
        model_override: pre-loaded Keras model. If provided, skip loading from disk.
        defect_data_override: dict with pre-loaded data, keys:
            {'clean': (x, y), 'backdoor': (x, y), 'adv': (x, y)}
            If provided, skip all internal data loading.
    """
    # ---- Model loading ----
    if model_override is not None:
        model = model_override
    else:
        if model_name == 'gtsrb_resnet':
            model_path = '../model/gtsrb_resnet.h5'
        else:
            model_path = f'../model/{model_name}.h5'
        model = load_model(model_path, compile=False)

    cfg_key = model_name
    cfg = DATASET_CONFIG[cfg_key]
    target_label = cfg['target_label']

    # ---- Data loading ----
    if defect_data_override is not None:
        x_clean, y_clean = defect_data_override['clean']
        defect_types = sorted([k for k in defect_data_override if k != 'clean'])
        # Only backdoor and adv
        defect_types = [d for d in defect_types if d in ('backdoor', 'adv')]

        # Filter adversarial samples: keep only successful attacks
        if 'adv' in defect_data_override:
            x_adv, y_adv = defect_data_override['adv']
            pred_adv_all = np.argmax(sess_run(model, model.output, x_adv), axis=-1)
            adv_success_mask = (pred_adv_all != y_adv)
            n_adv_before = len(x_adv)
            x_adv = x_adv[adv_success_mask]
            y_adv = y_adv[adv_success_mask]
            defect_data_override['adv'] = (x_adv, y_adv)
            print(f'Adversarial filtering: {np.sum(adv_success_mask)}/{n_adv_before} '
                  f'samples kept (attack success rate: {np.sum(adv_success_mask)/n_adv_before:.4f})')

        if k_sample is not None:
            for key in list(defect_data_override.keys()):
                x_d, y_d = defect_data_override[key]
                if k_sample < len(x_d):
                    idx = np.random.choice(len(x_d), k_sample, replace=False)
                    x_d, y_d = x_d[idx], y_d[idx]
                    defect_data_override[key] = (x_d, y_d)
                    if key == 'clean':
                        x_clean, y_clean = x_d, y_d

        sizes_info = {d: len(defect_data_override[d][0]) for d in defect_types}
        print(f'Using override data: clean={len(x_clean)}, '
              f'defect_types={defect_types}, sizes={sizes_info}')
    else:
        # Internal data loading
        x_clean, y_clean = load_clean_data(model_name)
        x_adv, y_adv = load_adv_data(model_name, epsilon, test=False)

        # Filter adversarial samples
        pred_adv_all = np.argmax(sess_run(model, model.output, x_adv), axis=-1)
        adv_success_mask = (pred_adv_all != y_adv)
        n_adv_before = len(x_adv)
        x_adv = x_adv[adv_success_mask]
        y_adv = y_adv[adv_success_mask]
        print(f'Adversarial filtering: {np.sum(adv_success_mask)}/{n_adv_before} '
              f'samples kept (attack success rate: {np.sum(adv_success_mask)/n_adv_before:.4f})')

        n_align = min(len(x_clean), len(x_adv))
        x_clean, y_clean = x_clean[:n_align], y_clean[:n_align]
        x_adv, y_adv = x_adv[:n_align], y_adv[:n_align]

        if k_sample is not None and k_sample < len(x_clean):
            idx = np.random.choice(len(x_clean), k_sample, replace=False)
            x_clean, y_clean = x_clean[idx], y_clean[idx]
            x_adv, y_adv = x_adv[idx], y_adv[idx]

        # Backdoor data
        x_bd = get_perturbed_input(x_clean, model_name, target_label, cfg['input_shape'])
        y_bd = y_clean
        bd_mask = (y_bd != target_label)
        n_bd_before = len(x_bd)
        x_bd, y_bd = x_bd[bd_mask], y_bd[bd_mask]
        print(f'Backdoor filtering: {np.sum(bd_mask)}/{n_bd_before} samples kept '
              f'(removed {n_bd_before - np.sum(bd_mask)} target-class samples)')

        defect_types = ['backdoor', 'adv']
        defect_data_override = {
            'clean': (x_clean, y_clean),
            'backdoor': (x_bd, y_bd),
            'adv': (x_adv, y_adv),
        }

    # ---- Layer / neuron metadata ----
    conv_meta, dense_meta = get_layer_weight_indices(model)
    neuron_players = get_neuron_players(model, conv_meta, dense_meta)
    total_neurons = len(neuron_players)
    neuron_step2_top = max(10, int(np.ceil(total_neurons * neuron_ratio)))

    print(f'Total neurons: {total_neurons}, step2_top: {neuron_step2_top}, '
          f'topk_per_defect: {topk_per_defect}, defect_types: {defect_types}')

    # ===== Step 1: Gradient-based importance =====
    print('Step 1: Computing gradient-based neuron importance ...')
    target_layers = get_target_layers(model)
    s1_scores, step1_stats = compute_step1_scores(
        model, {dt: defect_data_override[dt] for dt in defect_types}, target_layers)
    print('Step 1 done.')

    # Combined step1 score across all defects
    s1_combined = {}
    for p in neuron_players:
        s1_combined[p] = float(sum(s1_scores.get(dt, {}).get(p, 0.0) for dt in defect_types))
    s1_sorted = sorted(s1_combined.items(), key=lambda x: x[1], reverse=True)

    # ===== Step 2: Output zero-out causal validation =====
    print(f'Step 2: Output zero-out validation on top {neuron_step2_top} neurons ...')
    s2_scores = validate_step2_output_zeroout(
        model, s1_sorted, conv_meta, dense_meta,
        defect_data_override, target_label, n_top=neuron_step2_top)
    print('Step 2 done.')

    # ===== Per-defect final scores =====
    def compute_final(s1_dict, s2_dict, players):
        max_s1 = max(s1_dict.values()) if s1_dict else 1.0
        return {p: float((s1_dict.get(p, 0.0) / max_s1) * s2_dict.get(p, 0.0)) for p in players}

    final_scores = {}
    for dt in defect_types:
        final_scores[dt] = compute_final(
            s1_scores.get(dt, {}), s2_scores.get(dt, {}), neuron_players)

    # ===== Per-defect decisions =====
    task_decisions = {}
    any_full = False
    for dt in defect_types:
        sorted_vals = sorted(final_scores[dt].values(), reverse=True)
        task_decisions[dt] = build_single_task_decision(sorted_vals)
        print(f'  {dt:10s} decision: {task_decisions[dt]["recommended_mode"]} '
              f'({task_decisions[dt]["reason"]})')
        if task_decisions[dt]['recommended_mode'] == 'full':
            any_full = True

    recommended_mode = 'full' if any_full else 'neuron'
    overall_reason = 'at least one task recommends full repair' if any_full \
        else 'all tasks recommend neuron-level repair'

    # ===== Union of top-k per defect =====
    union_neurons = set()
    for dt in defect_types:
        top_k = set(sorted(final_scores[dt], key=final_scores[dt].get, reverse=True)[:topk_per_defect])
        union_neurons |= top_k
    union_neurons = sorted(union_neurons)

    print(f'  Union size: {len(union_neurons)} '
          f'({", ".join(f"{dt}:{topk_per_defect}" for dt in defect_types)})')

    # ===== Combined scores =====
    neuron_scores = [float(sum(final_scores.get(dt, {}).get(p, 0.0) for dt in defect_types))
                     for p in neuron_players]
    per_defect_scores = {}
    for dt in defect_types:
        per_defect_scores[dt] = [float(final_scores[dt].get(p, 0.0)) for p in neuron_players]

    decision = {
        'recommended_mode': recommended_mode,
        'decision_reason': overall_reason,
        'task_decisions': task_decisions,
        'step2_top_k': int(neuron_step2_top),
        'total_neurons': int(total_neurons),
        'neuron_ratio': float(neuron_ratio),
        'topk_per_defect': int(topk_per_defect),
        'union_size': int(len(union_neurons)),
        'defect_types': defect_types,
        'score_source': 'union of per-defect (s1_norm * s2_output_zeroout)',
        'step2_method': 'output_zeroout',
    }

    result = {
        'players': neuron_players.tolist(),
        'scores': neuron_scores,
        'union_neurons': union_neurons,
        'total_neurons': int(total_neurons),
        'decision': decision,
        'step1_stats': step1_stats,
    }
    if 'backdoor' in per_defect_scores:
        result['scores_backdoor'] = per_defect_scores['backdoor']
    if 'adv' in per_defect_scores:
        result['scores_adv'] = per_defect_scores['adv']

    return result