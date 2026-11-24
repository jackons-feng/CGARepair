#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FairFLRep adapter for the CFRepair Keras/H5 single-fairness-defect benchmark.

Protocol aligned to CFRepair's strict fairness repair stage:
  1. exactly one fixed pretrained model is used for each dataset: bank/adult/default;
  2. the complete TRAIN split is used for FairFL localization, FairRep PSO search,
     and best-solution selection;
  3. fairness is evaluated only by the CFRepair group-accuracy Gap metric;
  4. the repair script never opens TEST;
  5. final TEST evaluation is performed separately by fairflrep_test_strict.py;
  6. PSO uses a fixed implementation RNG seed only for reproducibility. This is
     not a model seed and is not an experimental dimension.

The FairFLRep algorithmic core is preserved: SettDiff repair-set construction,
last-layer FairFL weight localization (bias strength, gradient loss, forward
impact, Pareto front), and FairRep particle-swarm optimization. The only metric
adaptation is replacing FairFLRep's native fairness objective with the Gap metric
used by the CFRepair fairness experiment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.models import load_model


SUPPORTED_DATASETS = ("bank", "adult", "default")
NATIVE_FAIRFLREP_METRICS = ("eod", "spd", "fpr", "di")
SUPPORTED_FAIRNESS_METRICS = ("gap",) + NATIVE_FAIRFLREP_METRICS
EPS = 1e-12


@dataclass(frozen=True)
class TargetLayerMeta:
    model_layer_index: int
    layer_name: str
    layer_type: str
    kernel_weight_index: int
    bias_weight_index: Optional[int]
    kernel_shape: Tuple[int, int]


@dataclass(frozen=True)
class LocalizedWeight:
    flat_index: int
    row: int
    col: int
    base_value: float
    gradient_score: float
    forward_score: float
    combined_rank_score: float


# ---------------------------------------------------------------------------
# Runtime / IO helpers
# ---------------------------------------------------------------------------


def configure_tensorflow() -> None:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)


def _first_h5_dataset(obj):
    if isinstance(obj, h5py.Dataset):
        return np.asarray(obj)
    if isinstance(obj, h5py.Group):
        for key in obj.keys():
            result = _first_h5_dataset(obj[key])
            if result is not None:
                return result
    return None


def read_h5_array(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required H5 file not found: {path}")
    with h5py.File(str(path), "r") as handle:
        for key in preferred_keys:
            if key in handle:
                result = _first_h5_dataset(handle[key])
                if result is not None:
                    return np.asarray(result)
        result = _first_h5_dataset(handle)
        if result is not None:
            return np.asarray(result)
    raise ValueError(f"No readable dataset found in: {path}")


def normalize_labels(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim > 1 and values.shape[-1] > 1:
        values = np.argmax(values, axis=-1)
    values = values.reshape(-1).astype(np.int64)
    unique = np.unique(values)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"FairFLRep adapter expects binary labels; found {unique.tolist()}")
    return values


def normalize_binary_groups(values: np.ndarray) -> np.ndarray:
    groups = np.asarray(values).reshape(-1)
    unique = np.unique(groups)
    if len(unique) != 2:
        raise ValueError(
            "FairFLRep SettDiff adapter requires exactly two sensitive groups; "
            f"found {unique.tolist()}"
        )
    if np.array_equal(unique, np.asarray([0, 1])):
        return groups.astype(np.int64)
    mapping = {unique[0]: 0, unique[1]: 1}
    return np.asarray([mapping[item] for item in groups], dtype=np.int64)


def load_fairness_split(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = split.lower()
    if split not in ("train", "test"):
        raise ValueError(f"split must be train/test, got {split!r}")

    x = read_h5_array(data_dir / f"x_{split}.h5", ("x", "X")).astype(np.float32)
    y = normalize_labels(read_h5_array(data_dir / f"y_{split}.h5", ("y", "Y")))
    g = normalize_binary_groups(read_h5_array(data_dir / f"g_{split}.h5", ("g", "G")))

    n = len(x)
    if not (n == len(y) == len(g)):
        raise ValueError(
            f"Length mismatch in {data_dir}/{split}: x={len(x)}, y={len(y)}, g={len(g)}"
        )
    if n == 0:
        raise ValueError(f"Empty fairness split: {data_dir}/{split}")
    return x, y, g


def validate_model_data_shape(model, x: np.ndarray, dataset: str) -> None:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError("Only single-input models are supported.")
        input_shape = input_shape[0]
    expected = tuple(dim for dim in input_shape[1:] if dim is not None)
    actual = tuple(x.shape[1:])
    if expected and expected != actual:
        raise ValueError(
            f"Model/data input mismatch for {dataset}: model expects {expected}, data provides {actual}"
        )


def fingerprint_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Prediction and fairness metrics
# ---------------------------------------------------------------------------


def predict_outputs(model, x: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: List[np.ndarray] = []
    session = K.get_session()
    learning_phase = K.learning_phase()
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        feed = {model.input: x[start:end]}
        if not isinstance(learning_phase, (int, np.integer)):
            feed[learning_phase] = 0
        outputs.append(np.asarray(session.run(model.output, feed_dict=feed)))
    if not outputs:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def infer_output_prediction(outputs: np.ndarray) -> np.ndarray:
    outputs = np.asarray(outputs)
    if outputs.ndim > 1 and outputs.shape[-1] > 1:
        return np.argmax(outputs, axis=-1).reshape(-1).astype(np.int64)
    return (outputs.reshape(-1) > 0.5).astype(np.int64)


def predict_classes(model, x: np.ndarray, batch_size: int) -> np.ndarray:
    return infer_output_prediction(predict_outputs(model, x, batch_size=batch_size))


def _group_rate(mask: np.ndarray, event: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(event[mask]))


def group_accuracy_gap(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    accs = []
    for group in (0, 1):
        mask = groups == group
        if not np.any(mask):
            raise ValueError("Both sensitive groups must be present when computing fairness.")
        accs.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return abs(accs[1] - accs[0])


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    tprs = []
    for group in (0, 1):
        mask = (groups == group) & (y_true == 1)
        tprs.append(_group_rate(mask, y_pred == 1))
    return abs(tprs[1] - tprs[0])


def false_positive_rate_difference(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    fprs = []
    for group in (0, 1):
        mask = (groups == group) & (y_true == 0)
        fprs.append(_group_rate(mask, y_pred == 1))
    return abs(fprs[1] - fprs[0])


def statistical_parity_difference(y_pred: np.ndarray, groups: np.ndarray) -> float:
    rates = []
    for group in (0, 1):
        mask = groups == group
        rates.append(_group_rate(mask, y_pred == 1))
    return abs(rates[1] - rates[0])


def disparate_impact_ratio(y_pred: np.ndarray, groups: np.ndarray) -> float:
    rates = []
    for group in (0, 1):
        mask = groups == group
        rates.append(_group_rate(mask, y_pred == 1))
    lo = min(rates)
    hi = max(rates)
    if hi <= EPS:
        return 1.0
    return float(lo / hi)


def disparate_impact_distance(y_pred: np.ndarray, groups: np.ndarray) -> float:
    return abs(1.0 - disparate_impact_ratio(y_pred, groups))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    fairness_metric: str,
) -> Dict[str, object]:
    gap = group_accuracy_gap(y_true, y_pred, groups)
    eod = equal_opportunity_difference(y_true, y_pred, groups)
    fpr = false_positive_rate_difference(y_true, y_pred, groups)
    spd = statistical_parity_difference(y_pred, groups)
    di_ratio = disparate_impact_ratio(y_pred, groups)
    di_distance = abs(1.0 - di_ratio)

    group_acc = {}
    positive_rate = {}
    for group in (0, 1):
        mask = groups == group
        group_acc[str(group)] = float(np.mean(y_pred[mask] == y_true[mask]))
        positive_rate[str(group)] = float(np.mean(y_pred[mask] == 1))

    fairness_lookup = {
        "gap": gap,
        "eod": eod,
        "fpr": fpr,
        "spd": spd,
        "di": di_distance,
    }
    fairness = float(fairness_lookup[fairness_metric])
    return {
        "fairness": fairness,
        "fairness_metric": fairness_metric,
        "group_accuracy_gap": float(gap),
        "equal_opportunity_difference": float(eod),
        "false_positive_rate_difference": float(fpr),
        "statistical_parity_difference": float(spd),
        "disparate_impact_ratio": float(di_ratio),
        "disparate_impact_distance": float(di_distance),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "group_accuracy": group_acc,
        "group_positive_rate": positive_rate,
    }


def evaluate_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    fairness_metric: str,
    batch_size: int,
) -> Dict[str, object]:
    pred = predict_classes(model, x, batch_size=batch_size)
    return evaluate_predictions(y, pred, g, fairness_metric=fairness_metric)


def print_metrics(title: str, metrics: Dict[str, object]) -> None:
    print(f"\n{title}")
    print(f"  fairness objective ({metrics['fairness_metric']}): {float(metrics['fairness']):.8f}")
    print(f"  group accuracy gap: {float(metrics['group_accuracy_gap']):.8f}")
    print(f"  equal opportunity difference: {float(metrics['equal_opportunity_difference']):.8f}")
    print(f"  false positive rate difference: {float(metrics['false_positive_rate_difference']):.8f}")
    print(f"  statistical parity difference: {float(metrics['statistical_parity_difference']):.8f}")
    print(f"  disparate impact ratio: {float(metrics['disparate_impact_ratio']):.8f}")
    print(f"  F1: {float(metrics['f1']):.8f}")
    print(f"  accuracy: {float(metrics['accuracy']):.8f}")
    print(f"  group accuracy: {metrics['group_accuracy']}")


# ---------------------------------------------------------------------------
# FairFL localization (SettDiff for tabular tasks)
# ---------------------------------------------------------------------------


def find_target_dense_layer(model) -> TargetLayerMeta:
    """Use the last Dense layer kernel, matching FairFLRep's default last-layer repair."""
    weight_index = 0
    candidates: List[TargetLayerMeta] = []
    for model_layer_index, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue
        kernel = np.asarray(layer_weights[0])
        layer_type = type(layer).__name__
        bias_weight_index: Optional[int] = None
        if len(layer_weights) > 1:
            maybe_bias = np.asarray(layer_weights[1])
            if maybe_bias.ndim == 1 and kernel.ndim == 2 and maybe_bias.shape[0] == kernel.shape[-1]:
                bias_weight_index = weight_index + 1
        if "Dense" in layer_type and kernel.ndim == 2:
            candidates.append(
                TargetLayerMeta(
                    model_layer_index=model_layer_index,
                    layer_name=getattr(layer, "name", f"layer_{model_layer_index}"),
                    layer_type=layer_type,
                    kernel_weight_index=weight_index,
                    bias_weight_index=bias_weight_index,
                    kernel_shape=(int(kernel.shape[0]), int(kernel.shape[1])),
                )
            )
        weight_index += len(layer_weights)

    if not candidates:
        raise RuntimeError(
            "FairFLRep adapter currently requires a Dense target layer. "
            "No Dense layer with a 2-D kernel was found."
        )
    return candidates[-1]


def balanced_repair_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Balance correct vs. misclassified repair samples.

    This is the default because it follows Algorithm 2's D_rep_pos / D_rep_neg
    sampling step in the FairFLRep paper.  The surrounding prose can also be
    read as class balancing, so the adapter exposes that alternative explicitly
    through ``--repair-balance class`` rather than silently choosing one.
    """
    rng = np.random.RandomState(seed)
    wrong = np.flatnonzero(y_pred != y_true)
    correct = np.flatnonzero(y_pred == y_true)
    if len(wrong) == 0:
        # No misclassified examples: use all examples so localization remains well-defined.
        selected = np.arange(len(y_true), dtype=np.int64)
        return selected, {
            "balance_mode": "correctness",
            "misclassified_count": 0,
            "correct_count": int(len(correct)),
            "sampled_correct_count": int(len(correct)),
            "repair_subset_count": int(len(selected)),
            "balanced_fallback": True,
        }

    n_pos = min(len(correct), len(wrong))
    if n_pos > 0:
        sampled_correct = rng.choice(correct, size=n_pos, replace=False)
        selected = np.concatenate([wrong, sampled_correct]).astype(np.int64)
    else:
        selected = wrong.astype(np.int64)
    rng.shuffle(selected)
    return selected, {
        "balance_mode": "correctness",
        "misclassified_count": int(len(wrong)),
        "correct_count": int(len(correct)),
        "sampled_correct_count": int(n_pos),
        "repair_subset_count": int(len(selected)),
        "balanced_fallback": False,
    }


def class_balanced_repair_indices(
    y_true: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Alternative paper-text interpretation: equal samples from classes 0 and 1."""
    rng = np.random.RandomState(seed)
    idx0 = np.flatnonzero(y_true == 0)
    idx1 = np.flatnonzero(y_true == 1)
    n_each = min(len(idx0), len(idx1))
    if n_each == 0:
        selected = np.arange(len(y_true), dtype=np.int64)
        return selected, {
            "balance_mode": "class",
            "class_0_count": int(len(idx0)),
            "class_1_count": int(len(idx1)),
            "sampled_per_class": int(n_each),
            "repair_subset_count": int(len(selected)),
            "balanced_fallback": True,
        }
    s0 = rng.choice(idx0, size=n_each, replace=False)
    s1 = rng.choice(idx1, size=n_each, replace=False)
    selected = np.concatenate([s0, s1]).astype(np.int64)
    rng.shuffle(selected)
    return selected, {
        "balance_mode": "class",
        "class_0_count": int(len(idx0)),
        "class_1_count": int(len(idx1)),
        "sampled_per_class": int(n_each),
        "repair_subset_count": int(len(selected)),
        "balanced_fallback": False,
    }


def select_repair_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Select the FairFL repair subset under an explicit balancing interpretation."""
    if mode == "correctness":
        return balanced_repair_indices(y_true, y_pred, seed=seed)
    if mode == "class":
        return class_balanced_repair_indices(y_true, seed=seed)
    if mode == "none":
        selected = np.arange(len(y_true), dtype=np.int64)
        return selected, {
            "balance_mode": "none",
            "repair_subset_count": int(len(selected)),
            "balanced_fallback": False,
        }
    raise ValueError(f"Unknown repair balance mode: {mode!r}")


def _safe_cost(expected: float, observed: float) -> float:
    if observed <= EPS:
        if expected <= EPS:
            return 1.0
        return float(expected / EPS)
    return float(expected / observed)


def compute_settdiff_bias_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    deprived_group: Optional[int] = 0,
) -> Dict[str, object]:
    """
    Compute FairFLRep SettDiff bias-strength terms.

    The implementation follows Eqs. (9)--(18): class costs compare expected
    versus observed *correct* predictions; class-sensitive subgroup costs use
    P(N=g)P(Y=c) as the bias-free expectation and the correctly classified
    joint frequency P(correct,Y=c,N=g) as the observation.

    FairFLRep denotes O0 as deprived and O1 as favored unless the roles are
    reversed for the application. ``deprived_group=None`` enables a documented
    adapter-only automatic orientation based on overall correct-prediction cost.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot compute FairFL bias analysis on an empty repair subset.")
    correct = y_pred == y_true
    p_correct = float(np.mean(correct))

    if deprived_group is None:
        overall_group_costs: Dict[int, float] = {}
        for group in (0, 1):
            expected = float(np.mean(groups == group)) * p_correct
            observed = float(np.mean(correct & (groups == group)))
            overall_group_costs[group] = _safe_cost(expected, observed)
        dep_global = 0 if overall_group_costs[0] >= overall_group_costs[1] else 1
        orientation_source = "auto_overall_correctness_cost"
    else:
        dep_global = int(deprived_group)
        if dep_global not in (0, 1):
            raise ValueError(f"deprived_group must be 0, 1, or None; got {deprived_group!r}")
        overall_group_costs = {}
        orientation_source = "configured"
    fav_global = 1 - dep_global

    class_costs: Dict[int, float] = {}
    per_class: Dict[int, Dict[str, object]] = {}

    for label in (0, 1):
        p_label = float(np.mean(y_true == label))
        p_obs_correct_label = float(np.mean(correct & (y_true == label)))
        expected_correct_label = p_label * p_correct
        class_cost = _safe_cost(expected_correct_label, p_obs_correct_label)
        class_costs[label] = class_cost

        group_costs: Dict[int, float] = {}
        group_expected: Dict[int, float] = {}
        group_observed: Dict[int, float] = {}
        for group in (0, 1):
            p_group = float(np.mean(groups == group))
            expected = p_group * p_label
            observed = float(np.mean(correct & (y_true == label) & (groups == group)))
            group_expected[group] = expected
            group_observed[group] = observed
            group_costs[group] = _safe_cost(expected, observed)

        bias_strength = float(
            group_costs[dep_global] / max(group_costs[fav_global], EPS)
        )
        per_class[label] = {
            "class_cost": float(class_cost),
            "group_costs": {str(k): float(v) for k, v in group_costs.items()},
            "group_expected_prob": {str(k): float(v) for k, v in group_expected.items()},
            "group_observed_correct_prob": {
                str(k): float(v) for k, v in group_observed.items()
            },
            "deprived_group": int(dep_global),
            "favored_group": int(fav_global),
            "bias_strength": float(bias_strength),
            "support": int(np.sum(y_true == label)),
            "correct_support": int(np.sum(correct & (y_true == label))),
        }

    underrepresented_class = 0 if class_costs[0] >= class_costs[1] else 1
    reference_class = 1 - underrepresented_class
    return {
        "setting": "SettDiff",
        "class_costs": {str(k): float(v) for k, v in class_costs.items()},
        "underrepresented_class": int(underrepresented_class),
        "reference_class": int(reference_class),
        "deprived_group": int(dep_global),
        "favored_group": int(fav_global),
        "group_orientation_source": orientation_source,
        "overall_group_costs_for_auto_orientation": {
            str(k): float(v) for k, v in overall_group_costs.items()
        },
        "per_class": {str(k): v for k, v in per_class.items()},
        "repair_subset_accuracy": float(np.mean(correct)),
    }


def make_loss_gradient_function(model, target_layer) -> Tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], int]:
    output = model.output
    if isinstance(output, list):
        raise ValueError("Only single-output models are supported.")
    output_shape = K.int_shape(output)
    if not output_shape or output_shape[-1] is None:
        raise ValueError(f"Cannot infer model output dimension: {output_shape}")
    out_dim = int(output_shape[-1])
    kernel_var = target_layer.weights[0]
    learning_phase = K.learning_phase()

    if out_dim == 1:
        y_ph = K.placeholder(shape=(None, 1), dtype=K.floatx(), name="fairflrep_y_binary")
        p = K.clip(output, K.epsilon(), 1.0 - K.epsilon())
        loss = K.mean(-(y_ph * K.log(p) + (1.0 - y_ph) * K.log(1.0 - p)))
    else:
        y_ph = K.placeholder(shape=(None, out_dim), dtype=K.floatx(), name="fairflrep_y_categorical")
        p = K.clip(output, K.epsilon(), 1.0 - K.epsilon())
        loss = K.mean(-K.sum(y_ph * K.log(p), axis=-1))

    grad = K.gradients(loss, kernel_var)[0]
    if grad is None:
        raise RuntimeError("Could not construct gradient of loss with respect to target-layer kernel.")

    fn_inputs = [model.input, y_ph]
    use_learning_phase = not isinstance(learning_phase, (int, np.integer))
    if use_learning_phase:
        fn_inputs.append(learning_phase)
    fn = K.function(fn_inputs, [grad])

    def run(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(tuple(int(v) for v in K.int_shape(kernel_var)), dtype=np.float64)
        if out_dim == 1:
            encoded = y.reshape(-1, 1).astype(np.float32)
        else:
            encoded = np.zeros((len(y), out_dim), dtype=np.float32)
            encoded[np.arange(len(y)), y.astype(np.int64)] = 1.0
        args: List[object] = [x, encoded]
        if use_learning_phase:
            args.append(0)
        return np.asarray(fn(args)[0], dtype=np.float64)

    return run, out_dim


def mean_absolute_gradient(
    grad_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    kernel_shape: Tuple[int, int],
) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(kernel_shape, dtype=np.float64)
    total = np.zeros(kernel_shape, dtype=np.float64)
    count = 0
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        batch_grad = grad_fn(x[start:end], y[start:end])
        total += np.abs(batch_grad) * float(end - start)
        count += end - start
    return total / max(count, 1)


def make_prev_activation_function(model, target_layer):
    prev_tensor = target_layer.input
    if isinstance(prev_tensor, list):
        raise ValueError("FairFLRep adapter expects a single tensor feeding the target Dense layer.")
    learning_phase = K.learning_phase()
    fn_inputs = [model.input]
    use_learning_phase = not isinstance(learning_phase, (int, np.integer))
    if use_learning_phase:
        fn_inputs.append(learning_phase)
    fn = K.function(fn_inputs, [prev_tensor, model.output])

    def run(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        args: List[object] = [x]
        if use_learning_phase:
            args.append(0)
        prev, out = fn(args)
        return np.asarray(prev), np.asarray(out)

    return run


def forward_impact_matrix(
    activation_fn,
    target_layer,
    x: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    kernel = np.asarray(target_layer.get_weights()[0], dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("Forward-impact adapter currently supports a Dense target layer only.")
    in_dim, out_dim = kernel.shape
    if len(x) == 0:
        return np.zeros_like(kernel, dtype=np.float64)

    activation_sum = np.zeros(in_dim, dtype=np.float64)
    sensitivity_sum = np.zeros(out_dim, dtype=np.float64)
    count = 0

    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        prev, out = activation_fn(x[start:end])
        prev = np.asarray(prev)
        if prev.ndim != 2 or prev.shape[1] != in_dim:
            raise ValueError(
                f"Target Dense input shape mismatch: expected (*,{in_dim}), got {prev.shape}."
            )
        activation_sum += np.sum(prev, axis=0)
        out = np.asarray(out)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        if out.shape[1] == 1 and out_dim == 1:
            p = np.clip(out[:, 0], 0.0, 1.0)
            sensitivity_sum[0] += float(np.sum(p * (1.0 - p)))
        elif out.shape[1] == out_dim:
            p = np.clip(out, 0.0, 1.0)
            sensitivity_sum += np.sum(p * (1.0 - p), axis=0)
        else:
            # If an activation layer follows the target Dense layer, keep a neutral
            # unit sensitivity rather than silently changing the target layer.
            sensitivity_sum += float(end - start)
        count += end - start

    mean_activation = activation_sum / max(count, 1)
    mean_sensitivity = sensitivity_sum / max(count, 1)

    layer_weights = target_layer.get_weights()
    bias = np.zeros(out_dim, dtype=np.float64)
    if len(layer_weights) > 1 and np.asarray(layer_weights[1]).shape == (out_dim,):
        bias = np.asarray(layer_weights[1], dtype=np.float64)

    # FairFLRep Eq. (19): each incoming weight contributes through
    # w * mean(previous_activation) + output_bias, normalized over all
    # incoming weights of the same output neuron. We use the magnitude of
    # that influence for ranking so that large negative and positive effects
    # are both treated as strong forward impact.
    influence = np.abs(kernel * mean_activation[:, None] + bias[None, :])
    denom = np.sum(influence, axis=0, keepdims=True)
    normalized = influence / np.maximum(denom, EPS)
    return normalized * mean_sensitivity[None, :]


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= EPS:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def extract_pareto_front(
    gradient_score: np.ndarray,
    forward_score: np.ndarray,
    base_kernel: np.ndarray,
    max_weights: int = 0,
) -> Tuple[List[LocalizedWeight], List[Dict[str, object]]]:
    grad = np.asarray(gradient_score, dtype=np.float64).reshape(-1)
    fwd = np.asarray(forward_score, dtype=np.float64).reshape(-1)
    base = np.asarray(base_kernel, dtype=np.float64).reshape(-1)
    if not (len(grad) == len(fwd) == len(base)):
        raise ValueError("Pareto score arrays must have the same size.")

    finite = np.isfinite(grad) & np.isfinite(fwd)
    grad = np.where(finite, grad, -np.inf)
    fwd = np.where(finite, fwd, -np.inf)

    pareto: List[int] = []
    for i in range(len(grad)):
        if not finite[i]:
            continue
        dominated = False
        for j in range(len(grad)):
            if i == j or not finite[j]:
                continue
            if grad[j] >= grad[i] and fwd[j] >= fwd[i] and (grad[j] > grad[i] or fwd[j] > fwd[i]):
                dominated = True
                break
        if not dominated:
            pareto.append(i)

    if not pareto:
        finite_idx = np.flatnonzero(finite)
        if len(finite_idx) == 0:
            raise RuntimeError("All FairFL localization scores are non-finite.")
        combined_all = _minmax(grad[finite_idx]) + _minmax(fwd[finite_idx])
        pareto = [int(finite_idx[int(np.argmax(combined_all))])]

    pareto_arr = np.asarray(pareto, dtype=np.int64)
    grad_norm = _minmax(grad[pareto_arr])
    fwd_norm = _minmax(fwd[pareto_arr])
    combined = grad_norm + fwd_norm
    order = np.argsort(-combined)
    if max_weights > 0:
        order = order[:max_weights]
    selected = pareto_arr[order]

    rows, cols = base_kernel.shape
    localized: List[LocalizedWeight] = []
    for rank_pos, flat_index in enumerate(selected.tolist()):
        row = int(flat_index // cols)
        col = int(flat_index % cols)
        localized.append(
            LocalizedWeight(
                flat_index=int(flat_index),
                row=row,
                col=col,
                base_value=float(base[flat_index]),
                gradient_score=float(grad[flat_index]),
                forward_score=float(fwd[flat_index]),
                combined_rank_score=float(combined[order[rank_pos]]),
            )
        )

    all_scores = []
    pareto_set = set(int(v) for v in pareto)
    selected_set = set(int(v) for v in selected.tolist())
    for flat_index in range(len(base)):
        row = int(flat_index // cols)
        col = int(flat_index % cols)
        all_scores.append(
            {
                "flat_index": int(flat_index),
                "row": row,
                "col": col,
                "base_value": float(base[flat_index]),
                "gradient_score": float(grad[flat_index]) if np.isfinite(grad[flat_index]) else None,
                "forward_score": float(fwd[flat_index]) if np.isfinite(fwd[flat_index]) else None,
                "on_pareto_front": flat_index in pareto_set,
                "selected_for_repair": flat_index in selected_set,
            }
        )
    return localized, all_scores


def localize_fairfl_weights(
    model,
    x_rep: np.ndarray,
    y_rep: np.ndarray,
    g_rep: np.ndarray,
    seed: int,
    batch_size: int,
    localization_batch_size: int,
    max_localized_weights: int,
    deprived_group: Optional[int],
    repair_balance: str,
) -> Tuple[TargetLayerMeta, List[LocalizedWeight], Dict[str, object]]:
    baseline_pred = predict_classes(model, x_rep, batch_size=batch_size)
    repair_indices, balance_info = select_repair_indices(
        y_rep, baseline_pred, seed=seed, mode=repair_balance
    )
    x_work = x_rep[repair_indices]
    y_work = y_rep[repair_indices]
    g_work = g_rep[repair_indices]
    pred_work = baseline_pred[repair_indices]

    bias_analysis = compute_settdiff_bias_analysis(
        y_work, pred_work, g_work, deprived_group=deprived_group
    )
    target_meta = find_target_dense_layer(model)
    target_layer = model.layers[target_meta.model_layer_index]
    base_kernel = np.asarray(target_layer.get_weights()[0], dtype=np.float64)

    grad_fn, _ = make_loss_gradient_function(model, target_layer)
    activation_fn = make_prev_activation_function(model, target_layer)

    grad_by_class_group: Dict[Tuple[int, int], np.ndarray] = {}
    fwd_by_class_group: Dict[Tuple[int, int], np.ndarray] = {}
    subgroup_support: Dict[str, int] = {}

    for label in (0, 1):
        for group in (0, 1):
            mask = (y_work == label) & (g_work == group)
            key = f"y{label}_g{group}"
            subgroup_support[key] = int(np.sum(mask))
            grad_by_class_group[(label, group)] = mean_absolute_gradient(
                grad_fn,
                x_work[mask],
                y_work[mask],
                batch_size=localization_batch_size,
                kernel_shape=target_meta.kernel_shape,
            )
            fwd_by_class_group[(label, group)] = forward_impact_matrix(
                activation_fn,
                target_layer,
                x_work[mask],
                batch_size=localization_batch_size,
            )

    under = int(bias_analysis["underrepresented_class"])
    ref = int(bias_analysis["reference_class"])

    def class_weighted_ratio(source: Dict[Tuple[int, int], np.ndarray], label: int) -> np.ndarray:
        info = bias_analysis["per_class"][str(label)]
        dep = int(info["deprived_group"])
        fav = int(info["favored_group"])
        strength = float(info["bias_strength"])
        return strength * source[(label, dep)] / (1.0 + source[(label, fav)])

    grad_score = class_weighted_ratio(grad_by_class_group, under) - class_weighted_ratio(
        grad_by_class_group, ref
    )
    fwd_score = class_weighted_ratio(fwd_by_class_group, under) - class_weighted_ratio(
        fwd_by_class_group, ref
    )

    localized, all_scores = extract_pareto_front(
        grad_score,
        fwd_score,
        base_kernel,
        max_weights=max_localized_weights,
    )

    localization = {
        "method": "FairFLRep_FairFL_SettDiff_adapter",
        "target_layer": asdict(target_meta),
        "repair_subset": balance_info,
        "bias_analysis": bias_analysis,
        "subgroup_support": subgroup_support,
        "candidate_weight_count": int(base_kernel.size),
        "pareto_front_count_before_cap": int(sum(1 for item in all_scores if item["on_pareto_front"])),
        "selected_weight_count": int(len(localized)),
        "max_localized_weights": int(max_localized_weights),
        "selected_weights": [asdict(item) for item in localized],
        "all_weight_scores": all_scores,
    }
    return target_meta, localized, localization


# ---------------------------------------------------------------------------
# FairRep PSO
# ---------------------------------------------------------------------------


class FairRepPSO:
    def __init__(
        self,
        model,
        target_meta: TargetLayerMeta,
        localized: Sequence[LocalizedWeight],
        validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        fairness_metric: str,
        batch_size: int,
        seed: int,
        swarm_size: int,
        generations: int,
        patience: int,
        inertia: float,
        cognitive: float,
        social: float,
        init_sigma_scale: float,
        clip_sigma: float,
        target_fairness: Optional[float],
        log_every: int,
    ) -> None:
        if not localized:
            raise ValueError("FairRep PSO requires at least one localized weight.")
        self.model = model
        self.target_meta = target_meta
        self.localized = list(localized)
        self.x_val, self.y_val, self.g_val = validation_data
        self.fairness_metric = fairness_metric
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.swarm_size = int(swarm_size)
        self.generations = int(generations)
        self.patience = int(patience)
        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)
        self.init_sigma_scale = float(init_sigma_scale)
        self.clip_sigma = float(clip_sigma)
        self.target_fairness = target_fairness
        self.log_every = max(1, int(log_every))

        self.base_weights = [np.array(weight, copy=True) for weight in model.get_weights()]
        kernel = np.asarray(self.base_weights[target_meta.kernel_weight_index], dtype=np.float64)
        self.kernel_shape = kernel.shape
        flat = kernel.reshape(-1)
        self.indices = np.asarray([item.flat_index for item in localized], dtype=np.int64)
        self.base_position = flat[self.indices].astype(np.float64)
        self.layer_mean = float(np.mean(flat))
        self.layer_std = float(np.std(flat))
        if self.layer_std <= EPS:
            self.layer_std = max(float(np.mean(np.abs(flat))) * 0.1, 1e-3)
        self.lower = self.layer_mean - self.clip_sigma * self.layer_std
        self.upper = self.layer_mean + self.clip_sigma * self.layer_std
        self.evaluations = 0

    def apply_position(self, position: np.ndarray) -> None:
        weights = [np.array(weight, copy=True) for weight in self.base_weights]
        kernel = np.asarray(weights[self.target_meta.kernel_weight_index])
        flat = kernel.reshape(-1)
        flat[self.indices] = np.asarray(position, dtype=flat.dtype)
        weights[self.target_meta.kernel_weight_index] = flat.reshape(kernel.shape)
        self.model.set_weights(weights)

    def evaluate_position(self, position: np.ndarray) -> Tuple[float, Dict[str, object]]:
        self.apply_position(position)
        metrics = evaluate_model(
            self.model,
            self.x_val,
            self.y_val,
            self.g_val,
            fairness_metric=self.fairness_metric,
            batch_size=self.batch_size,
        )
        self.evaluations += 1
        return float(metrics["fairness"]), metrics

    def initialize_swarm(self) -> Tuple[np.ndarray, np.ndarray]:
        dim = len(self.base_position)
        sigma = max(self.layer_std * self.init_sigma_scale, 1e-6)
        # FairFLRep initializes candidate repaired weights from a Gaussian
        # whose mean and standard deviation are estimated from the existing
        # weights in the same layer. Particle 0 is kept as the original
        # localized-weight vector so the unrepaired model remains an explicit
        # candidate in the swarm.
        positions = self.rng.normal(
            loc=self.layer_mean,
            scale=sigma,
            size=(self.swarm_size, dim),
        )
        positions = np.clip(positions, self.lower, self.upper)
        positions[0] = self.base_position.copy()
        velocities = self.rng.normal(loc=0.0, scale=sigma * 0.1, size=(self.swarm_size, dim))
        velocities[0] = 0.0
        return positions, velocities

    def run(self) -> Dict[str, object]:
        positions, velocities = self.initialize_swarm()
        pbest = positions.copy()
        pbest_fitness = np.full(self.swarm_size, np.inf, dtype=np.float64)
        pbest_metrics: List[Optional[Dict[str, object]]] = [None] * self.swarm_size

        baseline_fitness, baseline_metrics = self.evaluate_position(self.base_position)
        global_best = self.base_position.copy()
        global_best_fitness = float(baseline_fitness)
        global_best_metrics = baseline_metrics

        # Initial population evaluation. Particle 0 is the original model.
        for i in range(self.swarm_size):
            if i == 0:
                fit, metrics = baseline_fitness, baseline_metrics
            else:
                fit, metrics = self.evaluate_position(positions[i])
            pbest_fitness[i] = fit
            pbest_metrics[i] = metrics
            if fit < global_best_fitness:
                global_best_fitness = float(fit)
                global_best = positions[i].copy()
                global_best_metrics = metrics

        history: List[Dict[str, object]] = []
        stagnant = 0
        start = time.time()
        completed_generations = 0

        for generation in range(1, self.generations + 1):
            previous_best = global_best_fitness
            r1 = self.rng.rand(*positions.shape)
            r2 = self.rng.rand(*positions.shape)
            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (pbest - positions)
                + self.social * r2 * (global_best[None, :] - positions)
            )
            positions = np.clip(positions + velocities, self.lower, self.upper)

            generation_fitness = np.empty(self.swarm_size, dtype=np.float64)
            for i in range(self.swarm_size):
                fit, metrics = self.evaluate_position(positions[i])
                generation_fitness[i] = fit
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest[i] = positions[i].copy()
                    pbest_metrics[i] = metrics
                if fit < global_best_fitness:
                    global_best_fitness = float(fit)
                    global_best = positions[i].copy()
                    global_best_metrics = metrics

            improved = global_best_fitness < previous_best - 1e-12
            stagnant = 0 if improved else stagnant + 1
            completed_generations = generation
            record = {
                "generation": generation,
                "best_fairness": float(global_best_fitness),
                "swarm_min": float(np.min(generation_fitness)),
                "swarm_mean": float(np.mean(generation_fitness)),
                "swarm_max": float(np.max(generation_fitness)),
                "best_accuracy": float(global_best_metrics["accuracy"]),
                "best_f1": float(global_best_metrics["f1"]),
                "improved": bool(improved),
                "stagnant_generations": int(stagnant),
                "evaluations": int(self.evaluations),
                "elapsed_seconds": float(time.time() - start),
            }
            history.append(record)

            if generation == 1 or generation % self.log_every == 0 or improved:
                print(
                    f"[FairRep PSO] gen={generation:03d} "
                    f"best_{self.fairness_metric}={global_best_fitness:.8f} "
                    f"acc={float(global_best_metrics['accuracy']):.6f} "
                    f"stagnant={stagnant}/{self.patience}"
                )

            if self.target_fairness is not None and global_best_fitness <= self.target_fairness:
                print(
                    f"[FairRep PSO] target fairness reached: {global_best_fitness:.8f} "
                    f"<= {self.target_fairness:.8f}"
                )
                break
            if self.patience > 0 and stagnant >= self.patience:
                print(f"[FairRep PSO] early stop after {stagnant} stagnant generations")
                break

        self.apply_position(global_best)
        return {
            "baseline_fitness": float(baseline_fitness),
            "best_fitness": float(global_best_fitness),
            "best_position": global_best.tolist(),
            "best_metrics": global_best_metrics,
            "generations": int(completed_generations),
            "evaluations": int(self.evaluations),
            "elapsed_seconds": float(time.time() - start),
            "history": history,
            "swarm_size": int(self.swarm_size),
            "max_generations": int(self.generations),
            "patience": int(self.patience),
            "inertia": float(self.inertia),
            "cognitive": float(self.cognitive),
            "social": float(self.social),
            "init_sigma_scale": float(self.init_sigma_scale),
            "clip_sigma": float(self.clip_sigma),
            "search_lower": float(self.lower),
            "search_upper": float(self.upper),
        }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_history_csv(path: Path, history: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def flatten_metric(prefix: str, metrics: Optional[Dict[str, object]], row: Dict[str, object]) -> None:
    keys = (
        "fairness",
        "accuracy",
        "f1",
        "group_accuracy_gap",
        "equal_opportunity_difference",
        "false_positive_rate_difference",
        "statistical_parity_difference",
        "disparate_impact_ratio",
        "disparate_impact_distance",
    )
    for key in keys:
        row[f"{prefix}_{key}"] = "" if metrics is None else metrics.get(key, "")


def write_summary_csv(
    path: Path,
    args: argparse.Namespace,
    target_meta: TargetLayerMeta,
    localization: Dict[str, object],
    pso: Dict[str, object],
    baseline_train: Dict[str, object],
    repaired_train: Dict[str, object],
) -> None:
    row: Dict[str, object] = {
        "dataset": args.dataset,
        "method": "FairFLRep_SettDiff_PSO_Gap_fulltrain_adapter",
        "fairness_metric": "gap",
        "repair_scope": "full_train",
        "checkpoint_selection_scope": "full_train",
        "test_split_accessed": False,
        "rng_seed": args.rng_seed,
        "target_layer": target_meta.layer_name,
        "target_layer_index": target_meta.model_layer_index,
        "candidate_weight_count": localization["candidate_weight_count"],
        "pareto_front_count": localization["pareto_front_count_before_cap"],
        "localized_weight_count": localization["selected_weight_count"],
        "repair_subset_count": localization["repair_subset"]["repair_subset_count"],
        "repair_balance": localization["repair_subset"].get("balance_mode", args.repair_balance),
        "underrepresented_class": localization["bias_analysis"]["underrepresented_class"],
        "pso_swarm_size": pso["swarm_size"],
        "pso_generations": pso["generations"],
        "pso_evaluations": pso["evaluations"],
        "pso_best_fitness": pso["best_fitness"],
        "pso_elapsed_seconds": pso["elapsed_seconds"],
        "before_train_gap": baseline_train["group_accuracy_gap"],
        "before_train_accuracy": baseline_train["accuracy"],
        "before_train_f1": baseline_train["f1"],
        "after_train_gap": repaired_train["group_accuracy_gap"],
        "after_train_accuracy": repaired_train["accuracy"],
        "after_train_f1": repaired_train["f1"],
        "delta_train_gap": float(repaired_train["group_accuracy_gap"]) - float(baseline_train["group_accuracy_gap"]),
        "delta_train_accuracy": float(repaired_train["accuracy"]) - float(baseline_train["accuracy"]),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FairFLRep SettDiff Gap adapter for the CFRepair Bank/Adult/Default benchmark. "
            "Uses complete TRAIN for localization, PSO repair, and selection; TEST is never opened."
        )
    )
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--model-root", default="../model")
    parser.add_argument("--data-root", default="../data/fairness")
    parser.add_argument("--output-root", default="./loc_result")
    parser.add_argument(
        "--run-name",
        default="fairflrep_settdiff_gap_fulltrain",
        help="Subdirectory name below <output-root>/<dataset>/.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help=(
            "Fixed RNG seed for repair-set sampling and PSO reproducibility only. "
            "It is not a model seed and is not an experimental dimension."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--localization-batch-size", type=int, default=256)
    parser.add_argument(
        "--repair-balance",
        choices=("correctness", "class", "none"),
        default="correctness",
        help=(
            "Repair-set balancing used by FairFL localization. correctness follows "
            "SettDiff correct-vs-misclassified sampling."
        ),
    )
    parser.add_argument(
        "--deprived-group",
        choices=("0", "1", "auto"),
        default="0",
        help="Sensitive-group orientation for FairFL localization.",
    )
    parser.add_argument(
        "--max-localized-weights",
        type=int,
        default=0,
        help="0 keeps the full FairFL Pareto front; positive values cap by combined Pareto rank.",
    )
    parser.add_argument("--swarm-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--pso-inertia", type=float, default=0.7298)
    parser.add_argument("--pso-cognitive", type=float, default=1.49618)
    parser.add_argument("--pso-social", type=float, default=1.49618)
    parser.add_argument("--init-sigma-scale", type=float, default=1.0)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument(
        "--target-gap",
        type=float,
        default=None,
        help="Optional Gap threshold for PSO early stopping; disabled by default.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Fixed RNG controls only stochastic internals of FairFLRep (sampling + PSO).
    # The CFRepair benchmark itself has one pretrained model per dataset.
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    tf.set_random_seed(args.rng_seed)
    configure_tensorflow()

    dataset = args.dataset.lower()
    fairness_metric = "gap"
    model_path = Path(args.model_root) / f"{dataset}.h5"
    data_dir = Path(args.data_root) / dataset
    output_dir = Path(args.output_root) / dataset / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Fairness data directory not found: {data_dir}")

    print("[FairFLRep-CFRepair] strict full-TRAIN repair; Gap only; TEST is never opened")
    print(f"Dataset: {dataset}")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print("Fairness objective: gap")
    print(f"Internal RNG seed (reproducibility only): {args.rng_seed}")

    model = load_model(str(model_path), compile=False)
    x_train, y_train, g_train = load_fairness_split(data_dir, "train")
    validate_model_data_shape(model, x_train, dataset)

    print(f"Complete TRAIN samples used for localization/repair/selection: {len(x_train)}")
    print("TEST split: unopened by this script")

    baseline_train = evaluate_model(
        model, x_train, y_train, g_train, fairness_metric, args.batch_size
    )
    print_metrics("Original COMPLETE TRAIN metrics", baseline_train)
    print("\n[TEST] not evaluated. Use fairflrep_test_strict.py only after configuration freeze.")

    deprived_group = None if args.deprived_group == "auto" else int(args.deprived_group)
    target_meta, localized, localization = localize_fairfl_weights(
        model,
        x_train,
        y_train,
        g_train,
        seed=args.rng_seed,
        batch_size=args.batch_size,
        localization_batch_size=args.localization_batch_size,
        max_localized_weights=args.max_localized_weights,
        deprived_group=deprived_group,
        repair_balance=args.repair_balance,
    )
    (output_dir / "localization.json").write_text(
        json.dumps(localization, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nFairFL localization on COMPLETE TRAIN")
    print(
        f"  target layer: layer[{target_meta.model_layer_index}] {target_meta.layer_name} "
        f"kernel_shape={target_meta.kernel_shape}"
    )
    print(f"  candidate weights: {localization['candidate_weight_count']}")
    print(f"  Pareto front: {localization['pareto_front_count_before_cap']}")
    print(f"  localized for repair: {localization['selected_weight_count']}")

    pso = FairRepPSO(
        model=model,
        target_meta=target_meta,
        localized=localized,
        validation_data=(x_train, y_train, g_train),
        fairness_metric=fairness_metric,
        batch_size=args.batch_size,
        seed=args.rng_seed,
        swarm_size=args.swarm_size,
        generations=args.generations,
        patience=args.patience,
        inertia=args.pso_inertia,
        cognitive=args.pso_cognitive,
        social=args.pso_social,
        init_sigma_scale=args.init_sigma_scale,
        clip_sigma=args.clip_sigma,
        target_fairness=args.target_gap,
        log_every=args.log_every,
    ).run()

    # FairRepPSO leaves the model at its global-best particle selected on full TRAIN Gap.
    repaired_train = evaluate_model(
        model, x_train, y_train, g_train, fairness_metric, args.batch_size
    )
    print_metrics("Selected repaired COMPLETE TRAIN metrics", repaired_train)

    weights_path = output_dir / "weights_final.h5"
    model.save_weights(str(weights_path))
    write_history_csv(output_dir / "history.csv", pso["history"])

    stats = {
        "success": True,
        "protocol_version": "fairflrep_cfrepair_strict_fulltrain_gap_v1",
        "method": "fairflrep_settdiff_pso_gap_fulltrain_adapter",
        "method_family": "FairFLRep",
        "setting": "SettDiff",
        "dataset": dataset,
        "model_seed": None,
        "model_count_for_dataset": 1,
        "rng_seed": args.rng_seed,
        "rng_seed_role": "repair_algorithm_reproducibility_only",
        "fairness_metric": "gap",
        "repair_scope": "full_train",
        "checkpoint_selection_scope": "full_train",
        "test_split_accessed": False,
        "model_path": str(model_path.resolve()),
        "data_dir": str(data_dir.resolve()),
        "train_size": len(x_train),
        "training_fingerprint": fingerprint_arrays(x_train, y_train, g_train),
        "baseline": {"train": baseline_train},
        "repaired": {"train": repaired_train},
        "localization": localization,
        "pso": pso,
        "weights_file": str(weights_path),
        "arguments": vars(args),
        "paper_alignment": {
            "one_pretrained_model_per_dataset": True,
            "datasets": list(SUPPORTED_DATASETS),
            "full_train_localization": True,
            "full_train_repair": True,
            "full_train_selection": True,
            "test_never_loaded_by_repair_script": True,
            "gap_only": True,
            "last_layer_weight_localization": True,
            "gradient_loss": True,
            "forward_impact": True,
            "bias_strength": True,
            "pareto_front": True,
            "particle_swarm_optimization": True,
            "fitness_is_gap_only": True,
            "paper_default_swarm_size": 100,
            "paper_default_max_generations": 100,
            "paper_default_stagnation_patience": 10,
        },
        "created_at": _dt.datetime.now().isoformat(),
    }
    (output_dir / "run_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_summary_csv(
        output_dir / "summary.csv",
        args,
        target_meta,
        localization,
        pso,
        baseline_train,
        repaired_train,
    )

    train_summary = [
        f"dataset={dataset}",
        "method=fairflrep_settdiff_pso_gap_fulltrain_adapter",
        "fairness_metric=gap",
        "repair_scope=full_train",
        "checkpoint_selection_scope=full_train",
        "test_split_accessed=False",
        f"rng_seed={args.rng_seed}",
        f"localized_weight_count={len(localized)}",
        f"original_train_gap={float(baseline_train['group_accuracy_gap']):.10f}",
        f"repaired_train_gap={float(repaired_train['group_accuracy_gap']):.10f}",
        f"original_train_accuracy={float(baseline_train['accuracy']):.10f}",
        f"repaired_train_accuracy={float(repaired_train['accuracy']):.10f}",
        f"weights_file={weights_path}",
    ]
    (output_dir / "best_solution_train.txt").write_text(
        "\n".join(train_summary) + "\n", encoding="utf-8"
    )

    print(f"\nSaved repaired weights: {weights_path}")
    print(f"Saved FairFL localization: {output_dir / 'localization.json'}")
    print(f"Saved PSO history: {output_dir / 'history.csv'}")
    print(f"Saved one-row TRAIN summary: {output_dir / 'summary.csv'}")
    print(f"Saved full metadata: {output_dir / 'run_stats.json'}")
    print("[TEST] not evaluated. Run fairflrep_test_strict.py only after configuration freeze.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
