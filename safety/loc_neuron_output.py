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

try:
    tf.compat.v1.disable_eager_execution()
except RuntimeError:
    pass  # already disabled

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