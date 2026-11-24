"""
BIRDNN: Behavior-Informed Repair via Dual-task Neural Network retraining.

Global retraining version for safety repair.

Core idea (from original BIRDNN paper, matching repair_retraining.py):
- Positive behavior = drawndown data (safe behavior, should be preserved)
- Negative behavior = counterexample data (unsafe behavior, should be fixed)
- Counterexample outputs are CORRECTED by matching to the nearest safe
  neighbor in the output space (correct_negative_data).
- Train with MSE loss on corrected targets:
    loss_ct = MSE(corrected_output, pred_on_counterexample)
    loss_dr = MSE(original_output, pred_on_drawndown)
    total_loss = alpha * loss_dr + beta * loss_ct
- Iterative fine-tuning with accuracy monitoring and LR decay.

Metrics:
  - safety = defect_rate = fraction of counterexamples predicted as unsafe
    (lower is better, 0 = all safe)
  - drawndown = clean_acc = fraction of drawndown predicted as safe
    (higher is better, 1 = all correct)

This is the GLOBAL version: all model weights are updated.
"""

import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from scipy.spatial.distance import cdist

from repair_config import (
    sess_run,
    BATCH_SIZE,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_ALPHA = 0.5       # weight for positive behavior preservation (dr)
DEFAULT_BETA = 0.5        # weight for negative behavior correction (ct)
DEFAULT_LR = 1e-3         # learning rate
DEFAULT_EPOCHS = 50       # training epochs per outer iteration
DEFAULT_ITERS = 10        # outer iterations (re-evaluate + re-train)
DEFAULT_PATIENCE = 10     # early stopping patience
DEFAULT_ACC_DROP = 0.05   # max allowed accuracy drop before restoring
DEFAULT_TRAIN_POS = 10000  # number of positive samples for training (use all)
DEFAULT_TRAIN_NEG = 10000  # number of corrected negative samples for training (use all)


# ============================================================
# Safety metrics
#   - safety = defect_rate = fraction of counterexamples predicted
#     as UNSAFE (lower is better, 0 = fully repaired)
#   - drawndown = clean_acc = fraction of drawndown data predicted
#     as SAFE (higher is better, 1 = fully correct)
# ============================================================

def _value_safety(model, images, model_name):
    """Defect rate: fraction of counterexamples predicted as UNSAFE."""
    output = sess_run(model, model.output, images)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        # unsafe classes: 3, 4
        return float((np.sum(pred == 3) + np.sum(pred == 4)) / len(output))
    if model_name == 'a29':
        # unsafe classes: 0, 1
        return float((np.sum(pred == 0) + np.sum(pred == 1)) / len(output))
    # default: unsafe class = 0
    return float(np.sum(pred == 0) / len(output))


def _value_drawndown(model, images, model_name):
    """Clean accuracy: fraction of drawndown data predicted as SAFE."""
    output = sess_run(model, model.output, images)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        # safe classes: 0, 1, 2
        return float((len(output) - np.sum(pred == 3) - np.sum(pred == 4))
                     / len(output))
    if model_name == 'a29':
        # safe classes: 2, 3, 4
        return float((len(output) - np.sum(pred == 0) - np.sum(pred == 1))
                     / len(output))
    # default: safe = not class 0
    return float(np.sum(pred != 0) / len(output))


# ============================================================
# Helper: safe/unsafe class masks per model
# ============================================================

def _get_class_info(model_name):
    """Return (safe_classes, unsafe_classes) for the given model."""
    if model_name == 'a19':
        return [0, 1, 2], [3, 4]
    if model_name == 'a29':
        return [2, 3, 4], [0, 1]
    return [1, 2, 3, 4], [0]


def _get_safe_mask(predictions, model_name):
    """Boolean mask: True for samples predicted as SAFE."""
    pred = np.argmin(predictions, axis=-1)
    _, unsafe_classes = _get_class_info(model_name)
    mask = np.ones(len(predictions), dtype=bool)
    for c in unsafe_classes:
        mask &= (pred != c)
    return mask


def _make_safe_target(output, safe_classes, unsafe_classes, margin=0.1):
    """Minimally adjust an output vector so the argmin falls in a safe class.

    Reduces all safe-class values so the minimum safe value is below
    all unsafe values. Preserves the relative ordering among safe classes.
    """
    out = output.copy()
    min_unsafe = np.min(out[unsafe_classes])
    min_safe = np.min(out[safe_classes])
    if min_safe < min_unsafe:
        return out  # already safe
    shift = min_unsafe - min_safe + margin
    out[safe_classes] -= shift
    return out


# ============================================================
# Counterexample correction (matching original BIRDNN)
# ============================================================

def correct_negative_data(neg_x, neg_y, pos_x, pos_y, model_name):
    """Correct counterexample outputs by matching to nearest safe neighbor.

    For each counterexample (neg), find the drawndown (pos) sample whose
    model output is closest in Euclidean distance, and use that drawndown's
    output as the corrected target.

    If no drawndown sample is predicted as safe, generates synthetic safe
    targets by minimally adjusting the nearest drawndown output.

    Args:
        neg_x: counterexample inputs  [N_neg, input_dim]
        neg_y: counterexample outputs [N_neg, output_dim]
        pos_x: drawndown inputs       [N_pos, input_dim]
        pos_y: drawndown outputs      [N_pos, output_dim]
        model_name: model identifier (a19, a29, a33)

    Returns:
        cor_x: counterexample inputs (unchanged)  [N_neg, input_dim]
        cor_y: corrected outputs (nearest safe)   [N_neg, output_dim]
    """
    safe_classes, unsafe_classes = _get_class_info(model_name)

    # Filter pos_y to only safe predictions
    safe_mask = _get_safe_mask(pos_y, model_name)
    safe_pos_y = pos_y[safe_mask]
    safe_pos_x = pos_x[safe_mask]
    n_safe = len(safe_pos_y)

    if n_safe > 0:
        # Use the filtered safe pool for correction
        dist = cdist(neg_y, safe_pos_y)  # [N_neg, N_safe]
        nearest_idx = np.argmin(dist, axis=1)
        cor_y = safe_pos_y[nearest_idx]
        print(f'  Correction pool: {n_safe}/{len(pos_y)} safe drawndown samples')
    else:
        # Fallback: no safe predictions on drawndown.
        # Use nearest drawndown output and adjust it to be safe.
        print(f'  WARNING: 0/{len(pos_y)} safe drawndown predictions. '
              f'Using synthetic safe targets.')
        dist = cdist(neg_y, pos_y)  # [N_neg, N_pos]
        nearest_idx = np.argmin(dist, axis=1)
        nearest_y = pos_y[nearest_idx]  # [N_neg, output_dim]
        cor_y = np.array([_make_safe_target(y, safe_classes, unsafe_classes)
                          for y in nearest_y])

    return neg_x.copy(), cor_y


# ============================================================
# Main BIRDNN repair function (global retraining for safety)
# ============================================================

def birdnn_single(model, defect_data, clean_data, defect_type,
                   loc_cache_path, seed, model_name, epsilon, **kwargs):
    """BIRDNN global retraining repair for safety defects.

    Implements the original BIRDNN approach:
    1. Compute model outputs on counterexample and drawndown data
    2. Correct counterexample outputs to nearest safe neighbor
    3. Split data: 800 pos + 7200 corrected neg for training (1:9 ratio)
    4. Train with MSE loss to match corrected targets
    5. Iterate with accuracy monitoring and LR decay

    Args:
        model: Keras model
        defect_data: counterexample inputs (x_ct,)
        clean_data: drawndown inputs (x_dr,)
        defect_type: 'safety'
        loc_cache_path: unused (kept for interface compatibility)
        seed: random seed
        model_name: model identifier (a19, a29, a33)
        epsilon: unused
        **kwargs: overrides (alpha, beta, lr, epochs, iters, patience, acc_drop)

    Returns:
        (repaired_model, stats)
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    alpha = kwargs.get('alpha', DEFAULT_ALPHA)
    beta = kwargs.get('beta', DEFAULT_BETA)
    lr = kwargs.get('lr', DEFAULT_LR)
    epochs = kwargs.get('epochs', DEFAULT_EPOCHS)
    iters = kwargs.get('iters', DEFAULT_ITERS)
    patience = kwargs.get('patience', DEFAULT_PATIENCE)
    acc_drop = kwargs.get('acc_drop', DEFAULT_ACC_DROP)
    train_pos_n = kwargs.get('train_pos_n', DEFAULT_TRAIN_POS)
    train_neg_n = kwargs.get('train_neg_n', DEFAULT_TRAIN_NEG)

    # For safety: clean_data = drawndown, defect_data = counterexample
    x_dr = clean_data  # drawndown (safe) inputs
    x_ct = defect_data  # counterexample (unsafe) inputs

    print('=' * 60)
    print(f'BIRDNN Global Repair (Safety): model={model_name}')
    print(f'  alpha={alpha}, beta={beta}, lr={lr}, epochs={epochs}, iters={iters}')
    print('=' * 60)

    # ---- Compute model outputs and correct counterexamples ----
    # This matches the original BIRDNN: neg_out = model(neg_in),
    # then correct_negative_data([neg_in, neg_out], [pos_in, pos_out])
    y_dr = sess_run(model, model.output, x_dr)
    y_ct = sess_run(model, model.output, x_ct)

    # Filter drawndown to only safe predictions for correction pool
    dr_safe_mask = _get_safe_mask(y_dr, model_name)
    n_dr_safe = int(np.sum(dr_safe_mask))
    print(f'  Drawndown safe predictions: {n_dr_safe}/{len(y_dr)}')

    # Correct counterexample outputs: each ct gets the output of the
    # nearest safe drawndown sample in output space
    cor_x, cor_y = correct_negative_data(x_ct, y_ct, x_dr, y_dr, model_name)

    # ---- Split data: 1:9 pos:neg ratio (matching original BIRDNN) ----
    # Use only safe drawndown predictions for training (preserve good behavior)
    if n_dr_safe > 0:
        safe_dr_x = x_dr[dr_safe_mask]
        safe_dr_y = y_dr[dr_safe_mask]
    else:
        # Fallback: no safe predictions, use all drawndown (model will be
        # pushed toward safe via the correction loss on counterexamples)
        safe_dr_x = x_dr
        safe_dr_y = y_dr

    n_train_pos = min(train_pos_n, len(safe_dr_x))
    n_train_neg = min(train_neg_n, len(cor_x))

    train_dr_x = safe_dr_x[:n_train_pos]
    train_dr_y = safe_dr_y[:n_train_pos]
    train_ct_x = cor_x[:n_train_neg]
    train_ct_y = cor_y[:n_train_neg]

    # ---- Baseline metrics ----
    baseline_dr = _value_drawndown(model, x_dr, model_name)
    baseline_safety = _value_safety(model, x_ct, model_name)
    print(f'  Baseline: drawndown={baseline_dr:.6f}, safety={baseline_safety:.6f}')
    print(f'  Training: pos={n_train_pos}, corrected_neg={n_train_neg}')

    # ---- Build training graph ----
    # loss_dr: safety hinge loss (push toward safe predictions on drawndown)
    # loss_ct: MSE on corrected targets (BIRDNN core: match nearest safe neighbor)
    sess = K.get_session()
    output_dim = model.output_shape[-1]

    safe_classes, unsafe_classes = _get_class_info(model_name)

    with sess.graph.as_default():
        x_dr_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None,) + model.input_shape[1:])
        x_ct_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None,) + model.input_shape[1:])
        y_ct_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, output_dim))

        logits_dr = model(x_dr_ph)
        logits_ct = model(x_ct_ph)

        # Safety hinge loss on drawndown: encourage safe predictions
        min_unsafe_dr = tf.reduce_min(
            tf.gather(logits_dr, unsafe_classes, axis=1), axis=1)
        min_safe_dr = tf.reduce_min(
            tf.gather(logits_dr, safe_classes, axis=1), axis=1)
        loss_dr = tf.reduce_mean(
            tf.maximum(0.0, min_safe_dr - min_unsafe_dr))

        # MSE loss on counterexamples: match corrected (safe) targets
        loss_ct = tf.reduce_mean(tf.square(y_ct_ph - logits_ct))
        total_loss = alpha * loss_dr + beta * loss_ct

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(total_loss,
                                       var_list=model.trainable_weights)

        try:
            sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))
        except Exception:
            pass

    # ---- Iterative training ----
    start_time = time.time()
    best_model_weights = copy.deepcopy(model.get_weights())
    best_dr = baseline_dr
    best_safety = baseline_safety
    # Fitness: lower is better (0 = perfect: safety=0, dr=1)
    best_fitness = beta * baseline_safety + alpha * (1.0 - baseline_dr)
    all_dr = [baseline_dr]

    terminal_count = 0
    total_epochs_run = 0

    for iteration in range(iters):
        print(f'\n--- Iteration {iteration + 1}/{iters} ---')

        current_dr = _value_drawndown(model, x_dr, model_name)
        current_safety = _value_safety(model, x_ct, model_name)
        current_fitness = beta * current_safety + alpha * (1.0 - current_dr)
        print(f'  Current: drawndown={current_dr:.6f}, '
              f'safety={current_safety:.6f}, fitness={current_fitness:.6f}')

        # Check termination (matching original BIRDNN logic):
        # safety=defect_rate (should go DOWN), drawndown=clean_acc (should go UP)
        if current_safety < 0.001 and current_dr > 0.999:
            terminal_count += 10
        elif current_dr <= max(all_dr):
            terminal_count += 1
        else:
            terminal_count = 0
        all_dr.append(current_dr)

        # Restore if drawndown drops too much (matching original BIRDNN)
        if best_dr - current_dr > acc_drop:
            print(f'  Large drawndown drop ({best_dr:.4f} -> '
                  f'{current_dr:.4f}), restoring...')
            model.set_weights(copy.deepcopy(best_model_weights))
            lr = lr * 0.8
            print(f'  Reduced LR to {lr}')
            with sess.graph.as_default():
                optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=lr)
                train_op = optimizer.minimize(
                    total_loss, var_list=model.trainable_weights)
                try:
                    sess.run(tf.compat.v1.variables_initializer(
                        optimizer.variables()))
                except Exception:
                    pass
            continue

        # Update best using combined fitness (lower is better)
        if current_fitness <= best_fitness:
            best_model_weights = copy.deepcopy(model.get_weights())
            best_dr = current_dr
            best_safety = current_safety
            best_fitness = current_fitness
            print(f'  New best: dr={best_dr:.6f}, safety={best_safety:.6f}, '
                  f'fitness={best_fitness:.6f}')

        # ---- Training loop ----
        # Use the smaller of the two training sets for batch count
        n_batches = min(len(train_dr_x), len(train_ct_x)) // BATCH_SIZE
        if n_batches < 1:
            n_batches = 1

        for e in range(epochs):
            idx_dr = np.random.permutation(len(train_dr_x))
            idx_ct = np.random.permutation(len(train_ct_x))

            epoch_loss_dr = 0.0
            epoch_loss_ct = 0.0

            for b in range(n_batches):
                # Drawndown batch (safety hinge loss, no target needed)
                start_dr = (b * BATCH_SIZE) % max(
                    1, len(train_dr_x) - BATCH_SIZE + 1)
                xb_dr = train_dr_x[idx_dr[start_dr:start_dr + BATCH_SIZE]]

                # Corrected counterexample batch (fix unsafe behavior)
                start_ct = (b * BATCH_SIZE) % max(
                    1, len(train_ct_x) - BATCH_SIZE + 1)
                xb_ct = train_ct_x[idx_ct[start_ct:start_ct + BATCH_SIZE]]
                yb_ct = train_ct_y[idx_ct[start_ct:start_ct + BATCH_SIZE]]

                _, ld, lc = sess.run(
                    [train_op, loss_dr, loss_ct],
                    feed_dict={x_dr_ph: xb_dr,
                               x_ct_ph: xb_ct, y_ct_ph: yb_ct})
                epoch_loss_dr += ld
                epoch_loss_ct += lc

            total_epochs_run += 1

            if (e + 1) % 10 == 0:
                avg_dr = epoch_loss_dr / max(1, n_batches)
                avg_ct = epoch_loss_ct / max(1, n_batches)
                print(f'    Epoch {e+1:3d}/{epochs}: '
                      f'loss_dr={avg_dr:.6f}, loss_ct={avg_ct:.6f}')

        # ---- Re-correct training data with updated model outputs ----
        # The model has changed, so re-compute correction targets
        y_dr = sess_run(model, model.output, x_dr)
        y_ct = sess_run(model, model.output, x_ct)
        dr_safe_mask = _get_safe_mask(y_dr, model_name)
        n_dr_safe = int(np.sum(dr_safe_mask))

        cor_x, cor_y = correct_negative_data(x_ct, y_ct, x_dr, y_dr, model_name)

        if n_dr_safe > 0:
            safe_dr_x = x_dr[dr_safe_mask]
            safe_dr_y = y_dr[dr_safe_mask]
        else:
            safe_dr_x = x_dr
            safe_dr_y = y_dr

        n_train_pos = min(train_pos_n, len(safe_dr_x))
        n_train_neg = min(train_neg_n, len(cor_x))
        train_dr_x = safe_dr_x[:n_train_pos]
        train_dr_y = safe_dr_y[:n_train_pos]
        train_ct_x = cor_x[:n_train_neg]
        train_ct_y = cor_y[:n_train_neg]

        if terminal_count >= patience:
            print(f'  Early stopping: terminal_count={terminal_count}')
            break

    # ---- Restore best model ----
    model.set_weights(best_model_weights)

    # ---- Final evaluation ----
    final_dr = _value_drawndown(model, x_dr, model_name)
    final_safety = _value_safety(model, x_ct, model_name)
    elapsed = time.time() - start_time

    print(f'\nBIRDNN completed in {elapsed:.2f}s')
    print(f'  Final: drawndown={final_dr:.6f}, safety={final_safety:.6f}')
    print(f'  Delta:  dr={final_dr - baseline_dr:+.6f}, '
          f'safety={final_safety - baseline_safety:+.6f}')

    stats = {
        'method': 'birdnn',
        'version': 'global',
        'defect_type': defect_type,
        'alpha': alpha,
        'beta': beta,
        'lr': float(lr),
        'epochs': epochs,
        'iters': iters,
        'total_epochs_run': total_epochs_run,
        'baseline_drawndown': float(baseline_dr),
        'baseline_safety': float(baseline_safety),
        'final_drawndown': float(final_dr),
        'final_safety': float(final_safety),
        'time_seconds': float(elapsed),
    }

    return model, stats