# -*- coding: utf-8 -*-
"""Train large 5-layer Keras MLPs for Bank, Adult and Default from H5 data.

This script does not recreate or modify the dataset split logic. It reads the
existing files under:

    data/fairness/<dataset>/
        x_train.h5, y_train.h5, g_train.h5,
        x_test.h5,  y_test.h5,  g_test.h5,
        metadata.json

The expected x_train order produced by the previous preparation script is:
    original fit partition, followed by original validation partition.

The script reconstructs that fit/validation split from metadata.json, selects
the epoch count using validation F1 only, rebuilds the model, trains it on the
complete non-test training set, and saves a complete Keras .h5 model.

Unified architecture:
    input -> 256 -> 256 -> 128 -> 64 -> 1

Each hidden block:
    Dense -> BatchNormalization -> ReLU -> Dropout

Stable repair layer names:
    layer1, layer2, layer3, layer4
Output layer:
    layer5
"""

from __future__ import print_function

import argparse
import json
import os
import random
import shutil
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K


DATASET_INPUT_DIMS = {
    "bank": 32,
    "adult": 34,
    "default": 30,
}

HIDDEN_SIZES = [256, 256, 128, 64]
REPAIRABLE_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
OUTPUT_LAYER = "layer5"


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_h5(path, preferred_key):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing H5 file: %s" % path)

    with h5py.File(path, "r") as handle:
        if preferred_key in handle:
            return np.asarray(handle[preferred_key])

        keys = list(handle.keys())
        if not keys:
            raise ValueError("Empty H5 file: %s" % path)

        obj = handle[keys[0]]
        if hasattr(obj, "shape"):
            return np.asarray(obj)

        for key in obj.keys():
            child = obj[key]
            if hasattr(child, "shape"):
                return np.asarray(child)

    raise ValueError("Cannot locate an array in H5 file: %s" % path)


def load_dataset(data_root, dataset):
    dataset_dir = os.path.join(data_root, dataset)
    arrays = {
        "x_train": read_h5(os.path.join(dataset_dir, "x_train.h5"), "x"),
        "y_train": read_h5(os.path.join(dataset_dir, "y_train.h5"), "y"),
        "g_train": read_h5(os.path.join(dataset_dir, "g_train.h5"), "g"),
        "x_test": read_h5(os.path.join(dataset_dir, "x_test.h5"), "x"),
        "y_test": read_h5(os.path.join(dataset_dir, "y_test.h5"), "y"),
        "g_test": read_h5(os.path.join(dataset_dir, "g_test.h5"), "g"),
    }

    metadata_path = os.path.join(dataset_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    arrays["x_train"] = np.asarray(arrays["x_train"], dtype=np.float32)
    arrays["x_test"] = np.asarray(arrays["x_test"], dtype=np.float32)
    arrays["y_train"] = np.asarray(arrays["y_train"]).reshape(-1).astype(np.int64)
    arrays["y_test"] = np.asarray(arrays["y_test"]).reshape(-1).astype(np.int64)
    arrays["g_train"] = np.asarray(arrays["g_train"]).reshape(-1).astype(np.int64)
    arrays["g_test"] = np.asarray(arrays["g_test"]).reshape(-1).astype(np.int64)

    if not (
        len(arrays["x_train"]) == len(arrays["y_train"]) == len(arrays["g_train"])
    ):
        raise ValueError("Train X/Y/G lengths do not match for %s" % dataset)
    if not (
        len(arrays["x_test"]) == len(arrays["y_test"]) == len(arrays["g_test"])
    ):
        raise ValueError("Test X/Y/G lengths do not match for %s" % dataset)

    expected_dim = DATASET_INPUT_DIMS[dataset]
    if arrays["x_train"].ndim != 2 or arrays["x_train"].shape[1] != expected_dim:
        raise ValueError(
            "%s expected input dimension %d, got %s"
            % (dataset, expected_dim, arrays["x_train"].shape)
        )

    return dataset_dir, arrays, metadata


def reconstruct_fit_validation(arrays, metadata, seed, fallback_validation_ratio):
    x_train = arrays["x_train"]
    y_train = arrays["y_train"]
    g_train = arrays["g_train"]

    fit_count = metadata.get("fit_sample_count")
    val_count = metadata.get("validation_sample_count")

    if fit_count is not None and val_count is not None:
        fit_count = int(fit_count)
        val_count = int(val_count)
        if fit_count > 0 and val_count > 0 and fit_count + val_count == len(x_train):
            return (
                x_train[:fit_count], y_train[:fit_count], g_train[:fit_count],
                x_train[fit_count:], y_train[fit_count:], g_train[fit_count:],
                "metadata_contiguous_fit_then_validation",
            )

    indices = np.arange(len(x_train))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=float(fallback_validation_ratio),
        random_state=int(seed) + 1,
        stratify=y_train,
    )
    return (
        x_train[fit_idx], y_train[fit_idx], g_train[fit_idx],
        x_train[val_idx], y_train[val_idx], g_train[val_idx],
        "fallback_stratified_split",
    )


def build_model(dataset, dropout_rate):
    input_dim = DATASET_INPUT_DIMS[dataset]
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="features")
    x = inputs

    for index, units in enumerate(HIDDEN_SIZES, start=1):
        x = tf.keras.layers.Dense(
            units,
            use_bias=True,
            activation=None,
            name="layer%d" % index,
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn%d" % index)(x)
        x = tf.keras.layers.Activation("relu", name="relu%d" % index)(x)
        x = tf.keras.layers.Dropout(
            float(dropout_rate),
            name="dropout%d" % index,
        )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name=OUTPUT_LAYER,
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="%s_large_5layer_mlp" % dataset,
    )


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def parameter_counts(model):
    trainable = int(np.sum([K.count_params(weight) for weight in model.trainable_weights]))
    non_trainable = int(
        np.sum([K.count_params(weight) for weight in model.non_trainable_weights])
    )
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }


def balanced_class_weights(y):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    classes = np.asarray([0, 1], dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {0: float(weights[0]), 1: float(weights[1])}


def group_accuracy_gap(y_true, y_pred, groups):
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    groups = np.asarray(groups).reshape(-1).astype(np.int64)

    group_accuracies = {}
    for group in [0, 1]:
        mask = groups == group
        if not np.any(mask):
            raise ValueError("Group %d has no samples" % group)
        group_accuracies[group] = float(np.mean(y_true[mask] == y_pred[mask]))

    gap = abs(group_accuracies[1] - group_accuracies[0])
    return float(gap), group_accuracies


def evaluate(model, x, y, groups, batch_size):
    probabilities = model.predict(x, batch_size=int(batch_size), verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int64)
    gap, group_acc = group_accuracy_gap(y, predictions, groups)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "fairness_gap": float(gap),
        "group_0_accuracy": float(group_acc[0]),
        "group_1_accuracy": float(group_acc[1]),
        "worst_group_accuracy": float(min(group_acc.values())),
    }


class BestValidationF1(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, batch_size):
        super(BestValidationF1, self).__init__()
        self.x_val = x_val
        self.y_val = np.asarray(y_val).reshape(-1).astype(np.int64)
        self.batch_size = int(batch_size)
        self.best_f1 = -1.0
        self.best_epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(
            self.x_val,
            batch_size=self.batch_size,
            verbose=0,
        ).reshape(-1)
        predictions = (probabilities >= 0.5).astype(np.int64)
        val_f1 = float(f1_score(self.y_val, predictions, zero_division=0))

        if logs is not None:
            logs["val_f1"] = val_f1

        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_epoch = int(epoch) + 1

        print(
            " - val_f1: %.6f - best_val_f1: %.6f - best_epoch: %d"
            % (val_f1, self.best_f1, self.best_epoch)
        )


def backup_existing_model(model_path, enabled):
    if not enabled or not os.path.exists(model_path):
        return None

    backup_dir = os.path.join(os.path.dirname(model_path), "model_backups")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    stem = os.path.splitext(os.path.basename(model_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, "%s_%s.h5" % (stem, timestamp))
    shutil.copy2(model_path, backup_path)
    return backup_path


def update_metadata(
    metadata_path,
    old_metadata,
    dataset,
    seed,
    model_path,
    split_policy,
    fit_count,
    val_count,
    full_train_count,
    test_count,
    best_epoch,
    best_val_f1,
    dropout,
    learning_rate,
    batch_size,
    max_epochs,
    model,
    train_metrics,
    test_metrics,
):
    metadata = dict(old_metadata)
    metadata.update({
        "dataset": dataset,
        "seed": int(seed),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "framework": "tensorflow.keras",
        "model_format": "full_h5",
        "model_path": os.path.abspath(model_path),
        "architecture": "5-layer MLP: input-256-256-128-64-1",
        "input_dim": int(DATASET_INPUT_DIMS[dataset]),
        "hidden_sizes": list(HIDDEN_SIZES),
        "hidden_layer_count": 4,
        "dense_layer_count_including_output": 5,
        "batch_normalization": True,
        "dropout": float(dropout),
        "optimizer": "Adam",
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "hidden_neuron_count": int(sum(HIDDEN_SIZES)),
        "total_neuron_count_including_output": int(sum(HIDDEN_SIZES) + 1),
        "repairable_layers": list(REPAIRABLE_LAYERS),
        "output_layer": OUTPUT_LAYER,
        "parameter_count": parameter_counts(model),
        "validation_reconstruction": split_policy,
        "fit_sample_count": int(fit_count),
        "validation_sample_count": int(val_count),
        "full_train_sample_count": int(full_train_count),
        "test_sample_count": int(test_count),
        "best_epoch_selected_on_validation": int(best_epoch),
        "best_validation_f1": float(best_val_f1),
        "final_training_scope": "complete_train_plus_validation",
        "test_used_for_selection": False,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    })

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def train_one_dataset(args, dataset):
    tf.keras.backend.clear_session()
    set_seed(args.seed)

    dataset_dir, arrays, metadata = load_dataset(args.data_root, dataset)
    (
        x_fit,
        y_fit,
        g_fit,
        x_val,
        y_val,
        g_val,
        split_policy,
    ) = reconstruct_fit_validation(
        arrays,
        metadata,
        seed=args.seed,
        fallback_validation_ratio=args.fallback_validation_ratio,
    )

    print("\n" + "=" * 88)
    print("Dataset: %s" % dataset)
    print("Architecture: %d -> 256 -> 256 -> 128 -> 64 -> 1" % DATASET_INPUT_DIMS[dataset])
    print("Fit/validation policy: %s" % split_policy)
    print("Fit shape: %s" % (x_fit.shape,))
    print("Validation shape: %s" % (x_val.shape,))
    print("Full train shape: %s" % (arrays["x_train"].shape,))
    print("Test shape: %s" % (arrays["x_test"].shape,))
    print("=" * 88)

    selector_model = compile_model(
        build_model(dataset, args.dropout),
        args.learning_rate,
    )
    selector_model.summary()
    print("Selector parameter counts: %s" % parameter_counts(selector_model))
    print(
        "Neuron counts: hidden=%d, total_including_output=%d"
        % (sum(HIDDEN_SIZES), sum(HIDDEN_SIZES) + 1)
    )

    f1_callback = BestValidationF1(x_val, y_val, args.batch_size)
    callbacks = [
        f1_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(args.early_stop_patience),
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=int(args.lr_patience),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    selector_model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=int(args.max_epochs),
        batch_size=int(args.batch_size),
        class_weight=balanced_class_weights(y_fit),
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    best_epoch = max(1, int(f1_callback.best_epoch))
    best_val_f1 = float(f1_callback.best_f1)
    print(
        "Selected epoch count using validation F1: %d (best F1=%.8f)"
        % (best_epoch, best_val_f1)
    )

    # Rebuild from the same seed and train on the complete non-test split.
    tf.keras.backend.clear_session()
    set_seed(args.seed)
    final_model = compile_model(
        build_model(dataset, args.dropout),
        args.learning_rate,
    )

    final_model.fit(
        arrays["x_train"],
        arrays["y_train"],
        epochs=best_epoch,
        batch_size=int(args.batch_size),
        class_weight=balanced_class_weights(arrays["y_train"]),
        shuffle=True,
        verbose=2,
    )

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, "%s.h5" % dataset)
    backup_path = backup_existing_model(model_path, args.backup_existing)
    if backup_path is not None:
        print("Backed up previous model to: %s" % backup_path)

    final_model.save(model_path, include_optimizer=False)

    train_metrics = evaluate(
        final_model,
        arrays["x_train"],
        arrays["y_train"],
        arrays["g_train"],
        args.batch_size,
    )
    test_metrics = evaluate(
        final_model,
        arrays["x_test"],
        arrays["y_test"],
        arrays["g_test"],
        args.batch_size,
    )

    metadata_path = os.path.join(dataset_dir, "metadata.json")
    update_metadata(
        metadata_path=metadata_path,
        old_metadata=metadata,
        dataset=dataset,
        seed=args.seed,
        model_path=model_path,
        split_policy=split_policy,
        fit_count=len(x_fit),
        val_count=len(x_val),
        full_train_count=len(arrays["x_train"]),
        test_count=len(arrays["x_test"]),
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        model=final_model,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    print("\nSaved full Keras model: %s" % model_path)
    print("Parameter counts: %s" % parameter_counts(final_model))
    print("Train metrics: %s" % train_metrics)
    print("Test metrics: %s" % test_metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Train large 5-layer Bank/Adult/Default Keras models from existing H5 data."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "bank", "adult", "default"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="../data/fairness")
    parser.add_argument("--model-dir", default="../model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=40)
    parser.add_argument("--lr-patience", type=int, default=12)
    parser.add_argument("--fallback-validation-ratio", type=float, default=0.25)
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Copy an existing <dataset>.h5 into model_backups before overwriting it.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 < args.fallback_validation_ratio < 1.0:
        raise ValueError("--fallback-validation-ratio must be in (0, 1)")

    datasets = ["bank", "adult", "default"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        train_one_dataset(args, dataset)


if __name__ == "__main__":
    main()