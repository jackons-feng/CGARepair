"""
BIRDNN: Behavior-Informed Repair via Dual-task Neural Network retraining.

Global retraining version for adversarial repair.

Core idea (from BIRDNN paper):
- Positive behavior = clean data (correct classification)
- Negative behavior = adversarial examples (misclassified)
- Train with combined loss:
    loss = alpha * CE(adv_x, true_labels) + beta * CE(clean_x, clean_labels)
- Iterative fine-tuning with accuracy monitoring and LR decay.

This is the GLOBAL version: all model weights are updated.
"""

import copy
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from repair_config import (
    sess_run,
    value_accuracy,
    BATCH_SIZE,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_ALPHA = 0.5       # weight for defect correction loss
DEFAULT_BETA = 0.5        # weight for clean accuracy preservation
DEFAULT_LR = 1e-4         # learning rate
DEFAULT_EPOCHS = 50       # training epochs
DEFAULT_ITERS = 10        # outer iterations (re-evaluate + re-train)
DEFAULT_PATIENCE = 3      # early stopping patience
DEFAULT_ACC_DROP = 0.05   # max allowed accuracy drop before restoring
BIRDNN_SAMPLE_SIZE = 10000

# ============================================================
# Helper: adversarial error rate
# ============================================================

def _value_adv_error(model, images_adv, labels_true):
    pred_adv = np.argmax(sess_run(model, model.output, images_adv), axis=-1)
    return float(np.sum(pred_adv != labels_true) / len(labels_true))


# ============================================================
# Main BIRDNN repair function (global retraining)
# ============================================================

def birdnn_single(model, defect_data, clean_data, defect_type,
                   loc_cache_path, seed, model_name, epsilon, **kwargs):
    """BIRDNN global retraining repair for adversarial defects.

    Args:
        model: Keras model
        defect_data: (x_adv, y_adv) - adversarial examples with true labels
        clean_data: (x_clean, y_clean) - clean training data
        defect_type: 'adv'
        loc_cache_path: unused (kept for interface compatibility)
        seed: random seed
        model_name: model identifier
        epsilon: adversarial epsilon
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

    x_clean, y_clean = clean_data
    dx, dy = defect_data

    print('=' * 60)
    print(f'BIRDNN Global Repair: defect_type={defect_type}')
    print(f'  alpha={alpha}, beta={beta}, lr={lr}, epochs={epochs}, iters={iters}')
    print('=' * 60)

    # ---- Baseline accuracy ----
    baseline_acc = value_accuracy(model, x_clean, y_clean)
    baseline_defect = _value_adv_error(model, dx, dy)
    print(f'  Baseline: clean_acc={baseline_acc:.6f}, adv_err={baseline_defect:.6f}')

    # ---- 4. Prepare birdnn evaluation data (subsampled) ----
    n_clean = min(BIRDNN_SAMPLE_SIZE, len(x_clean))
    idx_clean = np.random.choice(len(x_clean), n_clean, replace=False)
    x_clean = x_clean[idx_clean]
    y_clean = y_clean[idx_clean]

    n_defect = min(BIRDNN_SAMPLE_SIZE, len(dx))
    idx_defect = np.random.choice(len(dx), n_defect, replace=False)
    dx = dx[idx_defect]
    dy = dy[idx_defect]

    # ---- Build training graph (combined loss, matching original BIRDNN) ----
    # Original: loss = alpha*loss_adv + beta*loss_train, one backward pass
    # Here: loss = beta*CE_clean + alpha*CE_adv, one optimizer step
    sess = K.get_session()

    def _build_graph(lr_val):
        """Build/re-build training graph with given learning rate.
        Also builds accuracy ops from training graph logits to avoid
        model.output corruption in TF1 graph mode."""
        x_clean_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None,) + model.input_shape[1:])
        y_clean_ph = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        x_adv_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None,) + model.input_shape[1:])
        y_adv_ph = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        logits_clean = model(x_clean_ph)
        logits_adv = model(x_adv_ph)

        ce_clean = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_clean_ph, logits=logits_clean))
        ce_adv = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_adv_ph, logits=logits_adv))

        total_loss = beta * ce_clean + alpha * ce_adv

        # accuracy / error rate from training graph logits
        acc_clean = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(logits_clean, axis=1), tf.cast(y_clean_ph, tf.int64)),
            tf.float32))
        adv_err = tf.reduce_mean(tf.cast(tf.not_equal(
            tf.argmax(logits_adv, axis=1), tf.cast(y_adv_ph, tf.int64)),
            tf.float32))

        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_val)
        train = opt.minimize(total_loss, var_list=model.trainable_weights)
        try:
            sess.run(tf.compat.v1.variables_initializer(opt.variables()))
        except Exception:
            pass
        return x_clean_ph, y_clean_ph, x_adv_ph, y_adv_ph, \
            ce_clean, ce_adv, train, total_loss, acc_clean, adv_err

    with sess.graph.as_default():
        (x_clean_ph, y_clean_ph, x_adv_ph, y_adv_ph,
         ce_clean, ce_adv, train_op, total_loss,
         acc_clean, adv_err) = _build_graph(lr)

    # ---- Iterative training ----
    start_time = time.time()
    best_model_weights = copy.deepcopy(model.get_weights())
    best_acc = baseline_acc
    best_defect = baseline_defect


    terminal_count = 0
    total_epochs_run = 0

    for iteration in range(iters):
        print(f'\n--- Iteration {iteration + 1}/{iters} ---')

        # ---- Evaluate using TRAINING GRAPH ops (avoids model.output corruption) ----
        current_acc = sess.run(acc_clean, {x_clean_ph: x_clean, y_clean_ph: y_clean})
        current_defect = sess.run(adv_err, {x_adv_ph: dx, y_adv_ph: dy})
        print(f'  Current: clean_acc={current_acc:.6f}, adv_err={current_defect:.6f}')

        # Check termination (matching original BIRDNN logic)
        # Only stop when accuracy is consistently high
        if model_name =='mnist':
            acc_threshold = 0.99
        else:
            acc_threshold = 0.9
        if current_acc > acc_threshold:
            terminal_count += 1
        else:
            terminal_count = 0

        # Restore if accuracy drops too much
        if best_acc - current_acc > acc_drop:
            print(f'  Large accuracy drop ({best_acc:.4f} -> {current_acc:.4f}), '
                  f'restoring...')
            model.set_weights(copy.deepcopy(best_model_weights))
            lr = lr * 0.8
            print(f'  Reduced LR to {lr}')
            with sess.graph.as_default():
                (x_clean_ph, y_clean_ph, x_adv_ph, y_adv_ph,
                 ce_clean, ce_adv, train_op, total_loss,
                 acc_clean, adv_err) = _build_graph(lr)
            continue

        # Update best
        if current_acc >= best_acc:
            best_model_weights = copy.deepcopy(model.get_weights())
            best_acc = current_acc
            best_defect = current_defect

        # ---- Training loop ----
        n_batches = min(len(x_clean), len(dx)) // BATCH_SIZE
        if n_batches < 1:
            n_batches = 1

        for e in range(epochs):
            # Shuffle indices
            idx_clean = np.random.permutation(len(x_clean))
            idx_defect = np.random.permutation(len(dx))

            epoch_loss_clean = 0.0
            epoch_loss_defect = 0.0

            for b in range(n_batches):
                # Clean batch
                start_c = (b * BATCH_SIZE) % max(1, len(x_clean) - BATCH_SIZE + 1)
                xc = x_clean[idx_clean[start_c:start_c + BATCH_SIZE]]
                yc = y_clean[idx_clean[start_c:start_c + BATCH_SIZE]]

                # Defect batch
                start_d = (b * BATCH_SIZE) % max(1, len(dx) - BATCH_SIZE + 1)
                xd = dx[idx_defect[start_d:start_d + BATCH_SIZE]]
                yd = dy[idx_defect[start_d:start_d + BATCH_SIZE]]

                # Combined loss in ONE optimizer step (matching original BIRDNN)
                _, lc, ld = sess.run(
                    [train_op, ce_clean, ce_adv],
                    feed_dict={x_clean_ph: xc, y_clean_ph: yc,
                               x_adv_ph: xd, y_adv_ph: yd})
                epoch_loss_clean += lc
                epoch_loss_defect += ld

            total_epochs_run += 1

            if (e + 1) % 10 == 0:
                avg_lc = epoch_loss_clean / max(1, n_batches)
                avg_ld = epoch_loss_defect / max(1, n_batches)
                print(f'    Epoch {e+1:3d}/{epochs}: '
                      f'loss_clean={avg_lc:.4f}, loss_defect={avg_ld:.4f}')

        # Check termination
        if terminal_count >= patience:
            print(f'  Early stopping: terminal_count={terminal_count}')
            break

    # ---- Restore best model ----
    model.set_weights(best_model_weights)

    # ---- Final evaluation (using training graph ops) ----
    final_acc = sess.run(acc_clean, {x_clean_ph: x_clean, y_clean_ph: y_clean})
    final_defect = sess.run(adv_err, {x_adv_ph: dx, y_adv_ph: dy})
    elapsed = time.time() - start_time

    print(f'\nBIRDNN completed in {elapsed:.2f}s')
    print(f'  Final: clean_acc={final_acc:.6f}, adv_err={final_defect:.6f}')
    print(f'  Delta:  acc={final_acc - baseline_acc:+.6f}, '
          f'adv_err={final_defect - baseline_defect:+.6f}')

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
        'baseline_acc': float(baseline_acc),
        'baseline_defect': float(baseline_defect),
        'final_acc': float(final_acc),
        'final_defect': float(final_defect),
        'time_seconds': float(elapsed),
    }

    return model, stats