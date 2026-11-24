# -*- coding: utf-8 -*-
"""Configuration and Keras model builders for Bank, Adult and Default."""

from __future__ import print_function

import tensorflow as tf


DATASET_CONFIG = {
    "bank": {
        "expected_input_dim": 32,
        "hidden_sizes": [32, 32],
        "dropout": 0.2,
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "batch_size": 16,
        "max_epochs": 500,
        "early_stop_patience": 40,
        "lr_patience": 15,
        "group_name": "age_group",
        "group_policy": "age_minmax_ge_0.5",
    },
    "adult": {
        "expected_input_dim": 34,
        "hidden_sizes": [64, 128, 64],
        "dropout": 0.1,
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "batch_size": 256,
        "max_epochs": 2000,
        "early_stop_patience": 100,
        "lr_patience": 30,
        "group_name": "sex",
        "group_policy": "binary_sensitive_column",
    },
    "default": {
        "expected_input_dim": 30,
        "hidden_sizes": [16, 16, 16],
        "dropout": 0.2,
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "batch_size": 32,
        "max_epochs": 500,
        "early_stop_patience": 40,
        "lr_patience": 15,
        "group_name": "sex",
        "group_policy": "binary_sensitive_column",
    },
}


def build_optimizer(config):
    name = str(config.get("optimizer", "sgd")).lower()
    learning_rate = float(config.get("learning_rate", 0.01))
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)


def build_tabular_model(dataset, input_dim=None):
    """Build the Keras equivalent of the original PyTorch MLP.

    Layer names intentionally remain layer1, layer2, ... so localization and
    neuron-repair code can use stable names.
    """
    dataset = str(dataset).lower()
    if dataset not in DATASET_CONFIG:
        raise ValueError("Unsupported dataset: %s" % dataset)

    config = DATASET_CONFIG[dataset]
    expected_dim = int(config["expected_input_dim"])
    if input_dim is None:
        input_dim = expected_dim
    input_dim = int(input_dim)
    if input_dim != expected_dim:
        raise ValueError(
            "%s input dimension mismatch: expected %d, got %d"
            % (dataset, expected_dim, input_dim)
        )

    inputs = tf.keras.layers.Input(shape=(input_dim,), name="features")
    x = inputs
    for index, units in enumerate(config["hidden_sizes"], start=1):
        x = tf.keras.layers.Dense(
            int(units),
            activation="relu",
            name="layer%d" % index,
        )(x)
        x = tf.keras.layers.Dropout(
            float(config["dropout"]),
            name="dropout%d" % index,
        )(x)

    output_index = len(config["hidden_sizes"]) + 1
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="layer%d" % output_index,
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="%s_model" % dataset)


def compile_tabular_model(model, dataset):
    dataset = str(dataset).lower()
    config = DATASET_CONFIG[dataset]
    model.compile(
        optimizer=build_optimizer(config),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
