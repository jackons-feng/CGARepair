"""
FairFLRep: Fairness-aware Fault Localization and Repair.

Based on the FairFLRep paper, adapted for single-defect fairness repair.

Two phases:
1. Localization:
   - Build positive/negative sample sets from fairness data
     (_parse_results_sample_negative_sensitive)
   - Compute gradient-based importance on the classification layer's kernel
     (fairness-aware gradient aggregation: |grad_unfav| / (1 + |grad_fav|))
   - Compute forward impact: |activation × weight| for each (i, j) weight
   - Pareto front selection of key weights (grad_loss, fwd_imp)

2. Repair:
   - PSO optimization on individual weight values of the classification layer
   - Initialize particles from sibling weight distribution N(mu, std)
   - Velocity update: v = chi * (v + ro1*(p-x) + ro2*(g-x)), phi=4.1
   - Early stopping: stop if no improvement for 10 consecutive iterations
   - Only modifies the classification layer's kernel weights (not entire model)

Reference:
  fairflrep.py / validate_utils_cost.py from FairFLRep-main/src/
"""

import os
import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from repair_config import sess_run, value_accuracy


# ============================================================
# Constants (from original FairFLRep)
# ============================================================
PSO_NUM_PARTICLES = 100      # population size
PSO_NUM_ITERATIONS = 100     # max iterations
PSO_PHI = 4.1                # velocity constriction phi
PSO_MIN_ITER_RANGE = 10      # early stopping patience
PSO_EVAL_SAMPLE = 10000      # max samples for PSO fitness evaluation


# ============================================================
# Data preparation: build pos/neg sample sets
# ============================================================

def _reshape_labels(labels):
    """Convert one-hot or (N,1) labels to 1D integer labels."""
    if len(labels.shape) > 1:
        if labels.shape[1] != 1:
            labels = np.argmax(labels, axis=1)
        else:
            labels = labels.ravel()
    return labels.astype(np.int64)


def _get_unfav(y_true, y_pred):
    """Determine which class is the unfavored (disadvantaged) group.

    Original: data_util.py _get_unfav (lines 1001-1043 in validate_utils_cost.py)

    For each class i:
      P_exp = (n_i / N) * (n_correct_i / n_i)
      P_obs = n_correct_i / N
    If P_exp > P_obs, class i is unfavored.
    """
    y_true = _reshape_labels(y_true)
    y_pred = _reshape_labels(y_pred)
    n = len(y_true)
    for i in range(2):
        n_i = np.sum(y_true == i)
        if n_i == 0:
            continue
        n_correct_i = np.sum((y_true == i) & (y_pred == i))
        p_exp = (n_i / n) * (n_correct_i / max(1, n_i))
        p_obs = n_correct_i / n
        if p_exp > p_obs:
            return i
    return 0


def _parse_results_sample_negative_sensitive(test_images, test_labels,
                                              pred_labels, sensitive_labels,
                                              unfav):
    """Build positive/negative sample sets for fairness repair.

    Original: data_util.py _parse_results_sample_negative_sensitive (lines 1201-1224)

    Negative samples: mispredicted AND true label == unfavored group.
    Positive samples: correctly predicted (randomly sampled to match size).

    Args:
        test_images: np.ndarray [N, ...]
        test_labels: np.ndarray [N, num_classes] or [N,]
        pred_labels: np.ndarray [N, num_classes] (model predictions)
        sensitive_labels: np.ndarray [N,] (sensitive attribute, e.g. gender)
        unfav: int, the unfavored class label

    Returns:
        input_neg: (images, labels, sensitive_labels) for negative samples
        input_pos: (images, labels, sensitive_labels) for positive samples
        input_both: concatenated neg + pos
    """
    y_true_argmax = _reshape_labels(test_labels)
    y_pred_argmax = _reshape_labels(pred_labels)

    # All mispredicted
    neg_indices_all = np.where(y_true_argmax != y_pred_argmax)[0]
    # Correctly predicted
    pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    # Negative: mispredicted AND unfavored group
    neg_indices = np.where(
        (y_true_argmax != y_pred_argmax) & (y_true_argmax == unfav))[0]

    # Sample positives to match total negatives size
    sample_pos = np.random.choice(pos_indices,
                                   size=len(neg_indices_all),
                                   replace=False)

    sensitive_labels = _reshape_labels(sensitive_labels)

    input_neg = (test_images[neg_indices],
                 test_labels[neg_indices],
                 sensitive_labels[neg_indices])
    input_pos = (test_images[sample_pos],
                 test_labels[sample_pos],
                 sensitive_labels[sample_pos])

    # Use np.concatenate for labels to handle both 1D and 2D formats
    # (np.vstack fails on 1D arrays of different lengths)
    input_both = (np.vstack((input_neg[0], input_pos[0])),
                  np.concatenate([input_neg[1], input_pos[1]], axis=0),
                  np.concatenate([sensitive_labels[neg_indices],
                                  sensitive_labels[sample_pos]], axis=0))

    return input_neg, input_pos, input_both


def _select_subgroups_from_input(input_data):
    """Split input_data into two subgroups by class label.

    Original: data_util.py select_subgroups_from_input (lines 958-974)

    Args:
        input_data: (images, labels, [sensitive_labels])

    Returns:
        input_0: samples where class == 0
        input_1: samples where class == 1
    """
    y = _reshape_labels(input_data[1])
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]

    input_0 = (input_data[0][indices_0], input_data[1][indices_0])
    if len(input_data) > 2:
        input_0 = input_0 + (input_data[2][indices_0],)

    input_1 = (input_data[0][indices_1], input_data[1][indices_1])
    if len(input_data) > 2:
        input_1 = input_1 + (input_data[2][indices_1],)

    return input_0, input_1


# ============================================================
# Model layer helpers
# ============================================================

def _find_classification_layer(model):
    """Find the classification layer (last Dense layer).

    Returns:
        layer_index: index of the classification Dense layer
        prev_dense_index: index of the preceding Dense layer (or None)
    """
    dense_indices = []
    for i, layer in enumerate(model.layers):
        ltype = type(layer).__name__.lower()
        if 'dense' in ltype:
            dense_indices.append(i)

    if len(dense_indices) < 1:
        raise ValueError('No Dense layers found in model')

    classify_idx = dense_indices[-1]
    prev_dense_idx = dense_indices[-2] if len(dense_indices) >= 2 else None

    return classify_idx, prev_dense_idx


# ============================================================
# Fairness-aware gradient computation
# ============================================================

def _compute_each_gradient(model, x0, x1, layer_index):
    """Compute gradient of loss w.r.t. kernel for two subgroups.

    Original: validate_utils_cost.py _compute_each_gradient (lines 42-60)

    Adapted for TF1 graph mode: uses tf.gradients + sess.run instead of
    tf.GradientTape (which requires eager execution).

    Args:
        model: Keras model
        x0: (images, labels) for subgroup 0 (labels can be 1D or 2D one-hot)
        x1: (images, labels) for subgroup 1
        layer_index: index of target layer

    Returns:
        grad_kernel_0: numpy array, gradient for subgroup 0
        grad_kernel_1: numpy array, gradient for subgroup 1
    """
    layer = model.get_layer(index=layer_index)
    sess = K.get_session()

    use_sparse = (len(x0[1].shape) == 1)
    num_classes = model.output_shape[-1]

    with sess.graph.as_default():
        if use_sparse:
            y_ph = tf.compat.v1.placeholder(tf.int32, shape=[None])
        else:
            y_ph = tf.compat.v1.placeholder(
                tf.float32, shape=[None, num_classes])

        logits = model.output
        if use_sparse:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_ph, logits=logits)
        else:
            ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_ph, logits=logits)
        loss = tf.reduce_mean(ce)

        grads = tf.gradients(loss, [layer.kernel])[0]

    feed_0 = {model.input: x0[0], y_ph: x0[1], K.learning_phase(): 0}
    feed_1 = {model.input: x1[0], y_ph: x1[1], K.learning_phase(): 0}

    grad_kernel_0 = sess.run(grads, feed_0)
    grad_kernel_1 = sess.run(grads, feed_1)

    return grad_kernel_0, grad_kernel_1


def _agregate_based_on_fairness_notion(val_0, val_1, unfav):
    """Fairness-aware aggregation of gradient/impact values.

    Original: validate_utils_cost.py _agregate_based_on_fairness_notion (lines 62-69)

    If unfav == 0: return |val_0| / (1 + |val_1|)
    If unfav == 1: return |val_1| / (1 + |val_0|)

    This amplifies the signal from the unfavored group.
    """
    if unfav == 0:
        return abs(val_0) / (1.0 + abs(val_1))
    else:
        return abs(val_1) / (1.0 + abs(val_0))


def _compute_gradient(model, input_data, unfav, classify_layer_idx):
    """Compute fairness-aware gradient scores for all weights.

    Original: validate_utils_cost.py _compute_gradient (lines 71-135)

    For each weight (i, j) of the classification layer:
      - Compute gradient |dl_dw| for each subgroup
      - Aggregate: dl_dw = |dl_dw_unfav| / (1 + |dl_dw_fav|)
      - Sort by aggregated gradient (descending)

    Returns:
        list_imp: full list of [layer_index, i, j, |dl_dw|, dl_dw0, dl_dw1]
        candidates: sorted list of [layer_index, i, j, |dl_dw|]
        data_candidate_dict: {i: {j: |dl_dw|}} for fast lookup
    """
    input_0, input_1 = _select_subgroups_from_input(input_data)

    grad_kernel_0, grad_kernel_1 = _compute_each_gradient(
        model, input_0, input_1, classify_layer_idx)

    list_imp = []
    candidates = []
    data_candidate_dict = {}

    for j in range(grad_kernel_0.shape[1]):      # output dimension
        for i in range(grad_kernel_0.shape[0]):  # input dimension
            dl_dw0 = grad_kernel_0[i][j]
            dl_dw1 = grad_kernel_1[i][j]
            dl_dw = _agregate_based_on_fairness_notion(dl_dw0, dl_dw1, unfav)

            list_imp.append([classify_layer_idx, i, j,
                             abs(dl_dw), dl_dw0, dl_dw1])
            candidates.append([classify_layer_idx, i, j, abs(dl_dw)])

            if i not in data_candidate_dict:
                data_candidate_dict[i] = {}
            data_candidate_dict[i][j] = abs(dl_dw)

    candidates.sort(key=lambda tup: tup[3], reverse=True)

    return list_imp, candidates, data_candidate_dict


# ============================================================
# Forward impact computation
# ============================================================

def _compute_each_forward_impact(weight, activations, kernel):
    """Compute forward impact for a single weight.

    Original: validate_utils_cost.py _compute_each_forward_impact (lines 137-146)

    forward_impact = |activation[i] * kernel[i][j]|
    """
    layer_index = weight[0]
    nw_i = weight[1]   # input neuron index
    nw_j = weight[2]   # output neuron index
    o_i = activations[0][nw_i]   # activation of input neuron i
    w_ij = kernel[nw_i][nw_j]    # weight value
    return abs(o_i * w_ij)


def _forward_impact(model, input_data, classify_layer_idx, list_candidate,
                    prev_layer_idx):
    """Compute forward impact for all candidate weights.

    Original: validate_utils_cost.py _forward_impact (lines 148-166)

    Uses the activation of the previous layer and the kernel of the
    classification layer to compute |activation[i] * kernel[i][j]|.
    """
    prev_layer = model.get_layer(index=prev_layer_idx)
    classify_layer = model.get_layer(index=classify_layer_idx)

    # Get activations of the previous layer
    get_activations = K.function([model.input], prev_layer.output)
    activations = get_activations(input_data[0])

    # Get kernel of classification layer
    kernel = K.eval(classify_layer.kernel)

    list_fwd_imp = []
    for num in range(len(list_candidate)):
        layer_index, i, j, grad_loss, _, _ = list_candidate[num]
        fwd_imp = _compute_each_forward_impact(
            [layer_index, i, j], activations, kernel)
        list_fwd_imp.append(fwd_imp)

    return list_fwd_imp


def _compute_forward_impact_single(model, input_data, candidates,
                                    list_candidate, data_candidate_dict,
                                    num_grad, unfav, classify_layer_idx,
                                    prev_layer_idx):
    """Compute fairness-aware forward impact and build pool.

    Original: validate_utils_cost.py _compute_forward_impact_single (lines 215-263)

    For each candidate weight:
      - Compute forward impact for both subgroups
      - Aggregate with fairness notion
      - Build pool: {index: [layer_index, i, j, grad_loss, fwd_imp]}
      - Filter: keep only weights where both grad_loss >= mean and fwd_imp >= mean
    """
    layer_index = candidates[0][0]
    _num_grad = min(num_grad, len(candidates))

    input_0, input_1 = _select_subgroups_from_input(input_data)

    fwd_imp_0_list = _forward_impact(
        model, input_0, classify_layer_idx, list_candidate, prev_layer_idx)
    fwd_imp_1_list = _forward_impact(
        model, input_1, classify_layer_idx, list_candidate, prev_layer_idx)

    # Build per-neuron forward impact lookup
    pl_data = {}
    for num in range(len(list_candidate)):
        _, i, j, _, _, _ = list_candidate[num]
        if i not in pl_data:
            pl_data[i] = {}
        pl_data[i][j] = (fwd_imp_0_list[num], fwd_imp_1_list[num])

    list_candidate.sort(key=lambda tup: tup[3], reverse=True)

    grad_lst = []
    fwd_imp_lst = []
    pool = {}

    for num in range(_num_grad):
        _, i, j, grad_loss, grad_dw0, grad_dw1 = list_candidate[num]
        fwd_imp_0, fwd_imp_1 = pl_data[i][j]

        fwd_imp = _agregate_based_on_fairness_notion(fwd_imp_0, fwd_imp_1, unfav)

        pool[num] = [layer_index, i, j, grad_loss, fwd_imp]
        grad_lst.append(grad_loss)
        fwd_imp_lst.append(fwd_imp)

    mean_grad_loss = np.mean(grad_lst)
    mean_fwd_imp = np.mean(fwd_imp_lst)

    list_for_sample = []
    for num in range(_num_grad):
        _, i, j, grad_loss, grad_dw0, grad_dw1 = list_candidate[num]
        if grad_lst[num] >= mean_grad_loss and fwd_imp_lst[num] >= mean_fwd_imp:
            list_for_sample.append(
                [layer_index, i, j, grad_loss, grad_dw0, grad_dw1])

    print(f'  Forward impact: {len(list_for_sample)} weights pass '
          f'mean threshold (out of {_num_grad})')

    return list_for_sample, pool


# ============================================================
# Pareto front extraction
# ============================================================

def _identify_pareto(scores):
    """Identify Pareto-optimal solutions (non-dominated).

    Original: validate_utils_cost.py _identify_pareto (lines 426-454)

    For objectives (grad_loss, fwd_imp), both are maximized.
    Solution i is dominated by j if scores[j] >= scores[i] in all
    dimensions and > in at least one.

    Returns indices of Pareto-optimal solutions.
    """
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    pareto_front = np.ones(population_size, dtype=bool)

    for i in range(population_size):
        for j in range(population_size):
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                pareto_front[i] = 0
                break

    return population_ids[pareto_front]


def _extract_pareto_front(pool):
    """Extract weights on the Pareto front from the pool.

    Original: validate_utils_cost.py _extract_pareto_front (lines 473-504)

    Returns:
        List of [layer_index, i, j] for weights on the Pareto front.
    """
    objectives = []
    keys = list(pool.keys())
    for key in keys:
        weight = pool[key]
        objectives.append([weight[3], weight[4]])  # [grad_loss, fwd_imp]

    scores = np.array(objectives)
    pareto = _identify_pareto(scores)
    pareto_front = scores[pareto]

    results = []
    for key in keys:
        weight = pool[key]
        layer_index = weight[0]
        nw_i = weight[1]
        nw_j = weight[2]
        grad_loss = weight[3]
        fwd_imp = weight[4]

        for pf in pareto_front:
            if grad_loss == pf[0] and fwd_imp == pf[1]:
                results.append([layer_index, nw_i, nw_j])
                break

    print(f'  Pareto front: {len(results)} weights')
    return results


# ============================================================
# PSO repair (FairFLRep-style)
# ============================================================

def _get_initial_particle_positions(weights_t, model, classify_layer_idx,
                                     num_particles=PSO_NUM_PARTICLES):
    """Initialize particle positions from sibling weight distribution.

    Original: fairflrep.py _get_initial_particle_positions (lines 1176-1208)

    For each target weight (i, j):
      - Collect all other weights in the same layer (sibling weights)
      - Compute mu = mean(sibling_weights), std = std(sibling_weights)
      - Sample num_particles values from N(mu, std)

    Returns:
        locations: list of num_particles particles,
                   each particle is a list of (val, layer_idx, i, j) tuples
    """
    layer = model.get_layer(index=classify_layer_idx)
    target_weights = layer.get_weights()[0]
    n_out = target_weights.shape[1]
    n_in = target_weights.shape[0]

    # Collect all candidate (i, j) pairs
    candidate_pairs = set()
    for w in weights_t:
        candidate_pairs.add((int(w[1]), int(w[2])))

    locations = [[None for _ in range(len(weights_t))]
                 for _ in range(num_particles)]

    for n_w, weight in enumerate(weights_t):
        layer_index = int(weight[0])
        nw_i = int(weight[1])
        nw_j = int(weight[2])

        # Collect sibling weights (all other weights in same layer)
        sibling_weights = []
        for j in range(n_out):
            for i in range(n_in):
                if (i, j) not in candidate_pairs:
                    sibling_weights.append(target_weights[i][j])

        mu = np.mean(sibling_weights)
        std = np.std(sibling_weights)

        samples = np.random.default_rng().normal(
            loc=mu, scale=std, size=num_particles)

        for n_p in range(num_particles):
            locations[n_p][n_w] = (samples[n_p], layer_index, nw_i, nw_j)

    return locations


def _get_velocity_bounds(model, classify_layer_idx):
    """Compute velocity bounds for the classification layer.

    Original: fairflrep.py _get_velocity_bounds (lines 1210-1247)

    wb = max(all_weights) - min(all_weights)
    vb = (wb / 5, wb * 5)
    """
    velocity_bounds = {}

    layer = model.get_layer(index=classify_layer_idx)
    target_weights = layer.get_weights()[0]
    all_weights = [
        target_weights[i][j]
        for j in range(target_weights.shape[1])
        for i in range(target_weights.shape[0])
    ]

    wb = np.max(all_weights) - np.min(all_weights)
    vb = (wb / 5.0, wb * 5.0)
    velocity_bounds[classify_layer_idx] = vb

    return velocity_bounds


def _get_weight_values(weights):
    """Extract weight values from particle tuples."""
    return np.array([w[0] for w in weights])


def _update_position(x, v):
    """PSO position update: x_new = x + v.

    Original: fairflrep.py _update_position (lines 1560-1561)
    """
    return x + v


def _update_velocity(x, v, p, g, vb, layer_indices):
    """PSO velocity update with constriction factor.

    Original: fairflrep.py _update_velocity (lines 1563-1584)

    phi = 4.1
    chi = 2 / (phi - 2 + sqrt(phi^2 - 4*phi))
    v_new = chi * (v + ro1*(p-x) + ro2*(g-x))
    with velocity bounds enforced.
    """
    phi = PSO_PHI
    chi = 2.0 / (phi - 2.0 + np.sqrt(phi * phi - 4.0 * phi))

    ro1 = np.random.uniform(0, phi, size=len(x))
    ro2 = np.random.uniform(0, phi, size=len(x))

    new_v = chi * (v + ro1 * (p - x) + ro2 * (g - x))

    # Apply velocity bounds
    for n in range(len(new_v)):
        _vb = vb[layer_indices[n]]
        _new_v_abs = abs(new_v[n])
        _sign = 1.0 if new_v[n] > 0 else -1.0
        if _new_v_abs < _vb[0]:
            new_v[n] = _vb[0] * _sign
        if _vb[1] < _new_v_abs:
            new_v[n] = _vb[1] * _sign

    return new_v


def _copy_location_to_weights(location, model):
    """Apply particle weight values to the model.

    Original: fairflrep.py _copy_location_to_weights (lines 1479-1493)

    For each weight in the location, sets the model's kernel value.
    """
    for w in location:
        val = w[0]
        layer_index = int(np.round(w[1]))
        nw_i = int(np.round(w[2]))
        nw_j = int(np.round(w[3]))

        layer = model.get_layer(index=layer_index)
        weights = layer.get_weights()
        weights[0][nw_i][nw_j] = val
        layer.set_weights(weights)

    return model


def _copy_weights_to_location(model, location):
    """Save original weight values to the location.

    Original: fairflrep.py _copy_weights_to_location (lines 1495-1506)

    Converts tuple elements to lists for mutable access.
    """
    for i in range(len(location)):
        w = location[i]
        if isinstance(w, tuple):
            w = list(w)
            location[i] = w
        layer_index = int(np.round(w[1]))
        nw_i = int(np.round(w[2]))
        nw_j = int(np.round(w[3]))

        layer = model.get_layer(index=layer_index)
        weights = layer.get_weights()
        w[0] = weights[0][nw_i][nw_j]

    return location


def _compute_fairness_gap(y_true, y_pred):
    """Compute |female_acc - male_acc|. Female=0, Male=1."""
    y_true = _reshape_labels(y_true)
    y_pred = _reshape_labels(y_pred)
    correct = y_pred == y_true
    female_mask = (y_true == 0)
    male_mask = (y_true == 1)
    female_acc = float(np.sum(female_mask & correct)
                       / max(1, np.sum(female_mask)))
    male_acc = float(np.sum(male_mask & correct)
                     / max(1, np.sum(male_mask)))
    return abs(female_acc - male_acc)


def _evaluate_fitness(model, location, XA, classify_layer_idx):
    """Evaluate PSO fitness for a particle.

    Original: fairflrep.py _criterion (lines 1404-1477)

    Fitness = fairness_gap (to be minimized).

    Returns:
        fairness_gap, accuracy, 'n_intact'
    """
    # Save original weights and apply particle weights
    orig_location = copy.deepcopy(location)
    orig_location = _copy_weights_to_location(model, orig_location)
    model = _copy_location_to_weights(location, model)

    test_images, test_labels = XA[0], XA[1]
    y_pred = model.predict(test_images, verbose=0)

    y_pred = _reshape_labels(y_pred)
    y_true = _reshape_labels(test_labels)

    fairness_val = _compute_fairness_gap(y_true, y_pred)
    acc = float(np.mean(y_pred == y_true))

    # Restore original weights
    model = _copy_location_to_weights(orig_location, model)

    return fairness_val, acc, 0


def _fail_to_find_better_patch_fairness(t, history, patience=PSO_MIN_ITER_RANGE):
    """Early stopping: stop if no improvement for `patience` consecutive iters.

    Original: fairflrep.py _fail_to_find_better_patch_fairness (lines 1605-1622)

    For fairness, we minimize the fitness score.
    Stop if min(last patience scores) > score from (patience+1) iterations ago.
    """
    if t < patience:
        return False

    scores_in_history = np.array(history, dtype=float)[:, 0]
    best_x_before = scores_in_history[-patience - 1]
    best_last_x = np.min(scores_in_history[-patience:])

    if best_last_x > best_x_before:
        return True   # no improvement, stop
    return False


def _pso_repair(model, weights_t, classify_layer_idx, XA,
                num_particles=PSO_NUM_PARTICLES,
                num_iterations=PSO_NUM_ITERATIONS):
    """PSO optimization for fairness repair.

    Original: fairflrep.py optimize (lines 320-476)

    Flow:
      1. Initialize particle positions from sibling weight distribution
      2. Initialize velocities to zero
      3. Compute velocity bounds
      4. For each iteration:
         a. Update each particle's position and velocity
         b. Evaluate fitness (fairness gap)
         c. Update personal best and global best
         d. Check early stopping
      5. Apply global best weights to model

    Args:
        model: Keras model
        weights_t: list of [layer_index, i, j] target weights
        classify_layer_idx: index of classification layer
        XA: (images, labels) for fitness evaluation
        num_particles: population size
        num_iterations: max iterations

    Returns:
        model with best weights applied
    """
    print(f'\n  PSO: num_particles={num_particles}, '
          f'num_iterations={num_iterations}, '
          f'num_weights={len(weights_t)}')

    # 1. Initialize particle positions
    locations = _get_initial_particle_positions(
        weights_t, model, classify_layer_idx, num_particles)

    # 2. Initialize velocities to zero
    velocities = np.zeros((num_particles, len(weights_t)))

    # 3. Compute velocity bounds
    velocity_bounds = _get_velocity_bounds(model, classify_layer_idx)

    # 4. Initialize personal best scores
    personal_best_scores = []
    for n_p in range(num_particles):
        score, acc, n_intact = _evaluate_fitness(
            model, locations[n_p], XA, classify_layer_idx)
        personal_best_scores.append([score, acc, n_intact])

    personal_best_positions = list(locations)

    # 5. Find global best (minimize fairness gap)
    best_particle = np.argmin(np.array(personal_best_scores)[:, 0])
    global_best_position = copy.deepcopy(personal_best_positions[best_particle])

    history = []

    # Pre-compute layer indices for velocity bounds
    layer_indices = [int(w[0]) for w in weights_t]

    start_time = time.time()

    for t in range(num_iterations):
        g = _get_weight_values(global_best_position)

        for n in range(num_particles):
            x = _get_weight_values(locations[n])
            v = velocities[n]

            # Update position
            new_x = _update_position(x, v)
            new_weights = [
                (new_x[i], int(weights_t[i][0]),
                 int(weights_t[i][1]), int(weights_t[i][2]))
                for i in range(len(new_x))
            ]

            # Update velocity
            p = _get_weight_values(personal_best_positions[n])
            new_v = _update_velocity(new_x, v, p, g,
                                     velocity_bounds, layer_indices)

            # Evaluate fitness
            new_score, new_acc, new_n_intact = _evaluate_fitness(
                model, new_weights, XA, classify_layer_idx)

            # Update particle
            locations[n] = new_weights
            velocities[n] = new_v

            # Update personal best (minimize)
            if new_score < personal_best_scores[n][0]:
                personal_best_scores[n] = [new_score, new_acc, new_n_intact]
                personal_best_positions[n] = copy.deepcopy(new_weights)

        # Update global best
        best_particle = np.argmin(np.array(personal_best_scores)[:, 0])
        global_best_position = copy.deepcopy(
            personal_best_positions[best_particle])

        best_score = personal_best_scores[best_particle]
        history.append(best_score)

        if t % 10 == 0 or t == num_iterations - 1:
            print(f'  Iter {t:3d}: best fairness_gap={best_score[0]:.6f}, '
                  f'acc={best_score[1]:.6f}')

        # Early stopping
        if _fail_to_find_better_patch_fairness(t, history):
            print(f'  Early stopping at iteration {t}')
            break

    pso_time = time.time() - start_time
    print(f'  PSO completed in {pso_time:.2f}s')

    # Apply global best to model
    model = _copy_location_to_weights(global_best_position, model)

    best_score_final = history[-1]
    print(f'  Final: fairness_gap={best_score_final[0]:.6f}, '
          f'acc={best_score_final[1]:.6f}')

    return model, {
        'best_fairness_gap': float(best_score_final[0]),
        'best_acc': float(best_score_final[1]),
        'n_iterations': len(history),
        'pso_time': float(pso_time),
    }


# ============================================================
# Main entry point
# ============================================================

def fairflrep_single(model, defect_data, clean_data, defect_type,
                     loc_cache_path, seed, model_name, epsilon, **kwargs):
    """FairFLRep single-defect fairness repair.

    Compatible with defect_repair.py SINGLE_DEFECT_METHODS interface:
        fn(model, defect_data, clean_data, defect_type,
           loc_cache_path, seed, model_name, epsilon, **kwargs)
        -> (repaired_model, stats)

    Flow:
      1. Build pos/neg sample sets from fairness data
      2. Find classification layer
      3. Compute gradient-based localization (fairness-aware)
      4. Compute forward impact
      5. Extract Pareto front weights
      6. PSO optimization on individual weight values
      7. Early stopping when no improvement for 10 consecutive iterations

    Args:
        model: Keras model
        defect_data: (x, y) where y is gender (0=female, 1=male)
        clean_data: (x, y) same as defect_data for fairness
        defect_type: 'fairness'
        loc_cache_path: path to loc cache (used for output dir)
        seed: random seed
        model_name: model identifier
        epsilon: unused
        **kwargs: overrides (pso_num_particles, pso_num_iterations,
                  pso_eval_sample)

    Returns:
        (repaired_model, stats)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x, y = defect_data
    y = _reshape_labels(y)

    # ---- Overrides ----
    num_particles = kwargs.get('pso_num_particles', PSO_NUM_PARTICLES)
    num_iterations = kwargs.get('pso_num_iterations', PSO_NUM_ITERATIONS)
    eval_sample = kwargs.get('pso_eval_sample', PSO_EVAL_SAMPLE)

    print('=' * 60)
    print('FairFLRep Repair: defect_type=fairness')
    print('=' * 60)

    # ---- 1. Get model predictions and determine unfavored group ----
    print('\n[Step 1] Building pos/neg sample sets...')
    pred = sess_run(model, model.output, x)
    unfav = _get_unfav(y, pred)
    print(f'  Unfavored group: class {unfav}')

    # ---- 2. Build pos/neg sample sets ----
    input_neg, input_pos, input_both = _parse_results_sample_negative_sensitive(
        x, y, pred, y, unfav)
    print(f'  Negative samples: {len(input_neg[0])}')
    print(f'  Positive samples: {len(input_pos[0])}')
    print(f'  Combined samples: {len(input_both[0])}')

    # ---- 3. Find classification layer ----
    print('\n[Step 2] Finding classification layer...')
    classify_idx, prev_dense_idx = _find_classification_layer(model)
    print(f'  Classification layer: index={classify_idx} '
          f'({model.layers[classify_idx].name})')
    if prev_dense_idx is not None:
        print(f'  Previous Dense layer: index={prev_dense_idx} '
              f'({model.layers[prev_dense_idx].name})')
    else:
        raise ValueError('Need at least 2 Dense layers for FairFLRep')

    # ---- 4. Gradient-based localization ----
    print('\n[Step 3] Computing gradient-based localization...')
    num_grad = len(input_neg[0]) * 20  # "N_g = |neg| * 20"
    print(f'  num_grad = {len(input_neg[0])} * 20 = {num_grad}')

    list_candidates, candidates, data_candidate_dict = _compute_gradient(
        model, input_both, unfav, classify_idx)
    print(f'  Total gradient candidates: {len(candidates)}')

    # ---- 5. Forward impact computation ----
    print('\n[Step 4] Computing forward impact...')
    list_for_sample, pool = _compute_forward_impact_single(
        model, input_both, candidates, list_candidates,
        data_candidate_dict, num_grad, unfav,
        classify_idx, prev_dense_idx)

    # ---- 6. Pareto front extraction ----
    print('\n[Step 5] Extracting Pareto front...')
    weights_t = _extract_pareto_front(pool)

    if len(weights_t) == 0:
        # Fallback: use top-10 gradient candidates
        print('  Warning: empty Pareto front, using top-10 gradient candidates')
        weights_t = [[c[0], c[1], c[2]] for c in candidates[:10]]

    # ---- 7. Subsample data for PSO evaluation ----
    print('\n[Step 6] Preparing PSO evaluation data...')
    n_eval = min(eval_sample, len(input_both[0]))
    idx_eval = np.random.choice(len(input_both[0]), n_eval, replace=False)
    XA = (input_both[0][idx_eval], input_both[1][idx_eval])
    print(f'  PSO evaluation samples: {n_eval}')

    # ---- 8. PSO repair ----
    print('\n[Step 7] PSO optimization...')
    model, pso_stats = _pso_repair(
        model, weights_t, classify_idx, XA,
        num_particles=num_particles,
        num_iterations=num_iterations)

    # ---- 9. Build stats ----
    stats = {
        'method': 'fairflrep',
        'defect_type': defect_type,
        'unfavored_group': int(unfav),
        'num_neg_samples': int(len(input_neg[0])),
        'num_pos_samples': int(len(input_pos[0])),
        'num_both_samples': int(len(input_both[0])),
        'classify_layer_idx': int(classify_idx),
        'prev_dense_idx': int(prev_dense_idx),
        'num_grad': num_grad,
        'num_gradient_candidates': len(candidates),
        'num_forward_impact_candidates': len(list_for_sample),
        'num_pareto_weights': len(weights_t),
        'pareto_weights': [[int(w[0]), int(w[1]), int(w[2])]
                           for w in weights_t],
        'num_particles': num_particles,
        'num_iterations': num_iterations,
        'pso_eval_sample': n_eval,
        **pso_stats,
    }

    print(f'\nFairFLRep completed.')
    return model, stats