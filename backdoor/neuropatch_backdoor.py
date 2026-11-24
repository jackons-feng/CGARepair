#!/usr/bin/env python3
"""
Neuropatch-style backdoor repair baseline under the CFRepair protocol.

This script is intended as a single-defect BACKDOOR repair comparison method.
It keeps the core idea of Neuropatch/BadDiffusion-style repair:

  1) compare clean and triggered/backdoor activations;
  2) select neurons/channels with the largest activation deviation;
  3) mask these neurons by zeroing or flipping their contribution;
  4) keep only changes that reduce backdoor rate without excessive clean-accuracy drop.

Unlike the original diffusion-model implementation, this file targets the current
CFRepair TensorFlow/Keras classifier setting and saves weights_final.h5 so that
single_defect_test.py can evaluate it with the same protocol as CARE/CCBR/NSRepair.

Typical usage:
  python neuropatch_backdoor.py backdoor mnist 0 0.05
  python neuropatch_backdoor.py backdoor fashion 256 0.1
  python neuropatch_backdoor.py backdoor gtsrb 256 1.0 1

Output:
  loc_result/<dataset_tag>/<seed>/singledefect_neuropatch_backdoor_bdonly_neuron_eps<epsilon>/
      weights_final.h5
      weights_last.h5
      run_stats.json
      selected_actions.json
      singledefect_backdoor_best_solution.txt
"""

import os
# Set before importing TensorFlow. Helpful on TF1-style graph execution.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import sys
import json
import time
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.preprocessing import image


# -----------------------------------------------------------------------------
# Import CFRepair helpers when available.
# -----------------------------------------------------------------------------
def _extend_import_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (
        here,
        os.getcwd(),
        os.path.dirname(here),
        os.path.join(here, ".cache"),
        os.path.join(os.getcwd(), ".cache"),
        "/workspace/.cache",
    ):
        if path and path not in sys.path:
            sys.path.append(path)


_extend_import_path()

try:
    from repair_config import DATASET_CONFIG as REPAIR_DATASET_CONFIG
    from repair_config import load_clean_data as repair_load_clean_data
    from repair_config import get_perturbed_input as repair_get_perturbed_input
    from repair_config import value_accuracy, value_backdoor, sess_run
except Exception as exc:  # pragma: no cover - used only in user's environment
    REPAIR_DATASET_CONFIG = {}
    repair_load_clean_data = None
    repair_get_perturbed_input = None
    value_accuracy = None
    value_backdoor = None
    sess_run = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# TF1 session setup
# -----------------------------------------------------------------------------
def configure_tf_session() -> None:
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    K.set_session(tf.compat.v1.Session(config=config))


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
def as_bool(v: Any) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "t", "on")


def resolve_first_existing(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


def name_tag(model_name: str, resnet_flag: bool) -> str:
    return "gtsrb_resnet" if (model_name == "gtsrb" and resnet_flag) else model_name


def default_cfg(model_name: str, resnet_flag: bool = False) -> Dict[str, Any]:
    tag = name_tag(model_name, resnet_flag)
    if model_name in REPAIR_DATASET_CONFIG:
        cfg = dict(REPAIR_DATASET_CONFIG[model_name])
    elif tag in REPAIR_DATASET_CONFIG:
        cfg = dict(REPAIR_DATASET_CONFIG[tag])
    else:
        fallback = {
            "mnist": {"input_shape": (28, 28, 1), "num_classes": 10, "target_label": 3},
            "fashion": {"input_shape": (28, 28, 1), "num_classes": 10, "target_label": 3},
            "cifar10": {"input_shape": (32, 32, 3), "num_classes": 10, "target_label": 0},
            "cifar": {"input_shape": (32, 32, 3), "num_classes": 10, "target_label": 0},
            "gtsrb": {"input_shape": (32, 32, 3), "num_classes": 43, "target_label": 33},
            "gtsrb_resnet": {"input_shape": (32, 32, 3), "num_classes": 43, "target_label": 33},
        }
        if tag not in fallback and model_name not in fallback:
            raise ValueError(f"Unsupported model_name={model_name}. Add it to repair_config.DATASET_CONFIG.")
        cfg = dict(fallback.get(tag, fallback[model_name]))
    return cfg


def model_path_for(model_name: str, resnet_flag: bool) -> str:
    tag = name_tag(model_name, resnet_flag)
    filename = f"{tag}.h5" if tag != model_name else f"{model_name}.h5"
    if model_name == "gtsrb" and resnet_flag:
        filename = "gtsrb_resnet.h5"
    return resolve_first_existing(
        os.path.join("model", filename),
        os.path.join("..", "model", filename),
        os.path.join("model", f"{model_name}.h5"),
        os.path.join("..", "model", f"{model_name}.h5"),
    )


def load_gtsrb_train_data(train_file: str = "../data/gtsrb/train_data.pkl") -> Tuple[np.ndarray, np.ndarray]:
    """Load GTSRB TRAIN exactly as CCBR does (RGB 32x32 in [0,255]).

    Do not use repair_config.load_clean_data() for GTSRB: some configurations
    silently fall through to MNIST/Fashion. This must never use TEST images.
    """
    path = Path(train_file).expanduser().resolve()
    if "test" in path.name.lower():
        raise ValueError(f"Refusing to use a TEST file for Neuropatch training: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"GTSRB TRAIN data missing: {path}. Set --gtsrb-train-file to the train_data.pkl "
            "used by CCBR; do NOT use test_data.pkl."
        )
    with path.open("rb") as f:
        data = pickle.load(f)  # Must be a trusted local dataset file.
    if not isinstance(data, dict) or "data" not in data or "label" not in data:
        raise ValueError(f"GTSRB TRAIN file must contain data and label keys: {path}")
    x = np.asarray(data["data"])
    y = np.asarray(data["label"])
    if y.ndim == 2 and y.shape[-1] > 1:
        y = np.argmax(y, axis=1)
    y = y.reshape(-1).astype(np.int64)
    if x.ndim != 4 or len(x) != len(y) or not len(x):
        raise ValueError(f"GTSRB TRAIN invalid shapes: X={x.shape}, y={y.shape}")
    if x.shape[1] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    if x.shape[-1] != 3:
        raise ValueError(f"GTSRB requires RGB images with 3 channels, got X={x.shape}")
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo < -1e-3 or hi > 1.001:
        raise ValueError(
            f"GTSRB TRAIN data must be in [0,1] before scaling (as in CCBR), "
            f"got min={lo:.6f}, max={hi:.6f}; check the input pickle file."
        )
    # CCBR: transpose CHW -> HWC, resize to 32x32, then multiply by 255.
    # Use a TF placeholder so a full data array is not embedded in the graph.
    ph = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, 3))
    resized_op = tf.image.resize(ph, (32, 32))
    batches = []
    session = K.get_session()
    for i in range(0, len(x), 1000):
        batch = np.asarray(x[i:i+1000], dtype=np.float32)
        resized = session.run(resized_op, feed_dict={ph: batch})
        batches.append((resized * 255.0).astype(np.float32))
    x = np.concatenate(batches, axis=0)
    if x.shape != (len(y), 32, 32, 3):
        raise AssertionError(f"Internal GTSRB conversion produced wrong shape: {x.shape}")
    if y.min() < 0 or y.max() >= 43:
        raise ValueError(f"GTSRB labels must be in [0,42], got {y.min()}..{y.max()}")
    print(f"[DATA] GTSRB TRAIN only: {path}, X={x.shape}, labels={y.shape}, "
          f"pixel_range=[{x.min():.3f},{x.max():.3f}]")
    return x, y


def load_clean_data(model_name: str, resnet_flag: bool = False,
                    gtsrb_train_file: str = "../data/gtsrb/train_data.pkl") -> Tuple[np.ndarray, np.ndarray]:
    # Explicit GTSRB loader BEFORE generic repair_config fallback.
    if model_name in ("gtsrb", "gtsrb_resnet"):
        return load_gtsrb_train_data(gtsrb_train_file)

    if repair_load_clean_data is not None:
        for name in (name_tag(model_name, resnet_flag), model_name):
            try:
                result = repair_load_clean_data(name)
                if isinstance(result, (tuple, list)) and len(result) == 2:
                    return result
            except Exception:
                pass

    # Fallback loaders for common image datasets.
    if model_name == "mnist":
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        return np.expand_dims(x_train.astype("float32") / 255.0, -1), y_train.astype(int)
    if model_name == "fashion":
        (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
        return np.expand_dims(x_train.astype("float32") / 255.0, -1), y_train.astype(int)
    if model_name in ("cifar10", "cifar"):
        (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
        return x_train.astype("float32") / 255.0, y_train.reshape(-1).astype(int)

    if _IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Could not import repair_config.py and no fallback loader is available."
        ) from _IMPORT_ERROR
    raise ValueError(f"No clean-data loader for model_name={model_name}")


def _load_trigger_img(path: str, color_mode: str, target_size: Tuple[int, int]) -> np.ndarray:
    img = image.load_img(path, color_mode=color_mode, target_size=target_size)
    return image.img_to_array(img).astype("float32")


def get_backdoor_data(x: np.ndarray, model_name: str, cfg: Dict[str, Any], resnet_flag: bool = False) -> np.ndarray:
    target_label = int(cfg["target_label"])
    input_shape = tuple(cfg["input_shape"])
    if x.ndim != 4 or tuple(x.shape[1:]) != input_shape:
        raise ValueError(
            f"Wrong clean dataset for {name_tag(model_name, resnet_flag)}: "
            f"got X={x.shape}, expected (N, {input_shape}). "
            "Check GTSRB TRAIN path and preprocessing, do not resize MNIST into GTSRB."
        )

    # GTSRB must use the exact CCBR/Neural-Cleanse PNG trigger convention.
    # Generic repair_config helpers may silently default to a wrong dataset.
    if model_name not in ("gtsrb", "gtsrb_resnet") and repair_get_perturbed_input is not None:
        for config_name in (name_tag(model_name, resnet_flag), model_name):
            try:
                triggered = np.asarray(repair_get_perturbed_input(
                    x, config_name, target_label, input_shape))
                if triggered.shape == x.shape:
                    return triggered.astype(np.float32, copy=False)
                print(f"[WARN] repair_config trigger returned {triggered.shape}, "
                      f"expected {x.shape}; using PNG trigger instead")
            except Exception:
                pass

    # Match CCBR PNG pattern and mask, using dataset name 'gtsrb' even for resnet.
    img_model_name = "gtsrb" if model_name in ("gtsrb", "gtsrb_resnet") else model_name
    mask_name = f"{img_model_name}_visualize_mask_label_{target_label}.png"
    pattern_name = f"{img_model_name}_visualize_pattern_label_{target_label}.png"
    mask_path = resolve_first_existing(
        os.path.join("data", "img", mask_name),
        os.path.join("..", "data", "img", mask_name),
    )
    pattern_path = resolve_first_existing(
        os.path.join("data", "img", pattern_name),
        os.path.join("..", "data", "img", pattern_name),
    )
    if not os.path.isfile(mask_path) or not os.path.isfile(pattern_path):
        raise FileNotFoundError(f"Trigger mask/pattern not found: {mask_path} / {pattern_path}")

    h, w = input_shape[:2]
    mask = _load_trigger_img(mask_path, "grayscale", (h, w))[:, :, 0] / 255.0
    pattern = _load_trigger_img(pattern_path, "rgb", (h, w))
    keep = (1.0 - mask)[None, :, :, None]
    m3 = mask[None, :, :, None]
    if input_shape[-1] == 1:
        pat = pattern[:, :, 1:2] / 255.0
    else:
        # GTSRB CCBR training uses [0,255] pixels and 0..255 trigger RGB.
        pat = pattern if model_name in ("gtsrb", "gtsrb_resnet") else (pattern / 255.0 if np.max(x) <= 1.5 else pattern)
    out = x.astype(np.float32) * keep + pat[None, :, :, :] * m3
    if out.shape != x.shape:
        raise RuntimeError(f"Trigger output shape mismatch: {out.shape} vs {x.shape}")
    # CCBR trigger generation does not clip the resized input.
    print(f"[TRIGGER] mask={mask_path}, pattern={pattern_path}, X_p={out.shape}")
    return out.astype(np.float32)


def ensure_value_accuracy(model, x, y) -> float:
    if value_accuracy is not None:
        return value_accuracy(model, x, y)
    pred = np.argmax(model.predict(x, batch_size=256, verbose=0), axis=-1)
    return float(np.mean(pred == y))


def ensure_value_backdoor(model, x_bd, y_clean, target_label: int) -> float:
    if value_backdoor is not None:
        return value_backdoor(model, x_bd, y_clean, target_label)
    pred = np.argmax(model.predict(x_bd, batch_size=256, verbose=0), axis=-1)
    return float((np.sum(pred == target_label) - np.sum((y_clean == target_label) & (pred == target_label))) / len(x_bd))


def run_outputs(model, outputs, x: np.ndarray, batch_size: int = 256):
    """TF1-compatible batched session.run for one or more output tensors."""
    session = K.get_session()
    all_res = None
    n_batch = int(np.ceil(len(x) / batch_size))
    for b in range(n_batch):
        idx = np.arange(b * batch_size, min((b + 1) * batch_size, len(x)))
        res = session.run(outputs, feed_dict={model.input: x[idx]})
        if not isinstance(res, list):
            res = [res]
        if all_res is None:
            all_res = [[] for _ in res]
        for i, r in enumerate(res):
            all_res[i].append(r)
    cat = [np.concatenate(parts, axis=0) for parts in all_res]
    return cat if len(cat) > 1 else cat[0]


# -----------------------------------------------------------------------------
# Neuropatch localization
# -----------------------------------------------------------------------------
def candidate_layers(model, include_output_layer: bool = False) -> List[Any]:
    layers = []
    for layer in model.layers:
        if isinstance(layer, (Conv2D, Dense)):
            layers.append(layer)
    if not include_output_layer and len(layers) > 1:
        # Usually do not mask the classifier output layer directly.
        layers = layers[:-1]
    return layers


def neuron_diff_scores(model, x_clean: np.ndarray, x_bd: np.ndarray,
                       batch_size: int = 128,
                       include_output_layer: bool = False,
                       per_layer_top: int = 50,
                       candidate_pool: int = 300) -> List[Dict[str, Any]]:
    layers = candidate_layers(model, include_output_layer=include_output_layer)
    if not layers:
        raise RuntimeError("No Conv2D/Dense layers found for Neuropatch localization.")

    outputs = [layer.output for layer in layers]
    clean_outs = run_outputs(model, outputs, x_clean, batch_size=batch_size)
    bd_outs = run_outputs(model, outputs, x_bd, batch_size=batch_size)
    if len(layers) == 1:
        clean_outs = [clean_outs]
        bd_outs = [bd_outs]

    candidates = []
    for layer, c_act, b_act in zip(layers, clean_outs, bd_outs):
        c_mean = np.mean(c_act, axis=0)
        b_mean = np.mean(b_act, axis=0)
        diff = np.abs(c_mean - b_mean)
        if diff.ndim > 1:
            # Conv2D: average over spatial dimensions, keep channel dimension.
            axes = tuple(range(diff.ndim - 1))
            per_neuron = np.mean(diff, axis=axes)
        else:
            per_neuron = diff

        local = []
        for i, score in enumerate(per_neuron.reshape(-1)):
            local.append({
                "layer_name": layer.name,
                "layer_index": int(model.layers.index(layer)),
                "layer_class": type(layer).__name__,
                "neuron_idx": int(i),
                "score": float(score),
            })
        local.sort(key=lambda z: z["score"], reverse=True)
        candidates.extend(local[:min(per_layer_top, len(local))])

    candidates.sort(key=lambda z: z["score"], reverse=True)
    return candidates[:min(candidate_pool, len(candidates))]


# -----------------------------------------------------------------------------
# Weight masking / flipping
# -----------------------------------------------------------------------------
def _conv_or_dense(layer) -> bool:
    return isinstance(layer, (Conv2D, Dense))


def _next_conv_or_dense(model, start_idx: int):
    for j in range(start_idx + 1, len(model.layers)):
        if _conv_or_dense(model.layers[j]):
            return j, model.layers[j]
    return None, None


def _layer_by_name(model, layer_name: str):
    for i, layer in enumerate(model.layers):
        if layer.name == layer_name:
            return i, layer
    raise KeyError(f"Layer not found: {layer_name}")


def apply_action_to_weights(model, weights: List[np.ndarray], action: Dict[str, Any], operation: str) -> bool:
    """Apply one Neuropatch action to a copied Keras weight list.

    operation: 'zero' or 'flip'.
    Returns True if the action maps to a valid kernel slice.
    """
    layer_idx, layer = _layer_by_name(model, action["layer_name"])
    neuron = int(action["neuron_idx"])
    next_idx, next_layer = _next_conv_or_dense(model, layer_idx)

    # Map layer index -> starting index in model.get_weights().
    starts = {}
    cursor = 0
    for i, ly in enumerate(model.layers):
        starts[i] = cursor
        cursor += len(ly.get_weights())

    def mutate(arr, slc):
        if operation == "zero":
            arr[slc] = 0.0
        elif operation == "flip":
            arr[slc] *= -1.0
        else:
            raise ValueError(f"Unsupported operation={operation}")

    if next_layer is not None and len(next_layer.get_weights()) > 0:
        w_idx = starts[next_idx]
        kernel = weights[w_idx]
        if isinstance(next_layer, Conv2D):
            # Kernel: (kh, kw, in_channels, out_channels)
            if neuron >= kernel.shape[-2]:
                return False
            mutate(kernel, (slice(None), slice(None), neuron, slice(None)))
            return True

        if isinstance(next_layer, Dense):
            if isinstance(layer, Conv2D):
                # Conv -> Flatten -> Dense. Keras Flatten with channels_last maps
                # channel c to rows whose flattened index % C == c.
                out_shape = layer.output_shape
                if isinstance(out_shape, list):
                    out_shape = out_shape[0]
                if len(out_shape) != 4 or out_shape[-1] is None:
                    return False
                channels = int(out_shape[-1])
                if neuron >= channels:
                    return False
                rows = np.arange(kernel.shape[0])[np.arange(kernel.shape[0]) % channels == neuron]
                if len(rows) == 0:
                    return False
                mutate(kernel, (rows, slice(None)))
                return True
            # Dense -> Dense.
            if neuron >= kernel.shape[0]:
                return False
            mutate(kernel, (neuron, slice(None)))
            return True

    # Last trainable Conv/Dense: modify its output column/filter.
    if len(layer.get_weights()) == 0:
        return False
    w_idx = starts[layer_idx]
    kernel = weights[w_idx]
    if isinstance(layer, Conv2D):
        if neuron >= kernel.shape[-1]:
            return False
        mutate(kernel, (slice(None), slice(None), slice(None), neuron))
        if len(layer.get_weights()) > 1:
            b = weights[w_idx + 1]
            mutate(b, neuron)
        return True
    if isinstance(layer, Dense):
        if neuron >= kernel.shape[-1]:
            return False
        mutate(kernel, (slice(None), neuron))
        if len(layer.get_weights()) > 1:
            b = weights[w_idx + 1]
            mutate(b, neuron)
        return True
    return False


def apply_actions(model, base_weights: List[np.ndarray], actions: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], int]:
    new_weights = [w.copy() for w in base_weights]
    valid = 0
    for action in actions:
        op = action.get("operation", "zero")
        if apply_action_to_weights(model, new_weights, action, op):
            valid += 1
    return new_weights, valid


# -----------------------------------------------------------------------------
# Greedy selection
# -----------------------------------------------------------------------------
def evaluate_model_pair(model, x_clean, y_clean, x_bd, target_label) -> Tuple[float, float]:
    clean_acc = ensure_value_accuracy(model, x_clean, y_clean)
    bd_rate = ensure_value_backdoor(model, x_bd, y_clean, target_label)
    return clean_acc, bd_rate


def greedy_neuropatch(model, candidates: List[Dict[str, Any]],
                      x_eval_clean: np.ndarray, y_eval_clean: np.ndarray,
                      x_eval_bd: np.ndarray, target_label: int,
                      max_actions: int = 50,
                      max_clean_drop: float = 0.03,
                      min_asr_improve: float = 1e-5,
                      operations: Tuple[str, ...] = ("zero", "flip")) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_weights = [w.copy() for w in model.get_weights()]
    base_clean, base_bd = evaluate_model_pair(model, x_eval_clean, y_eval_clean, x_eval_bd, target_label)
    clean_floor = max(0.0, base_clean - max_clean_drop)

    selected: List[Dict[str, Any]] = []
    best_clean, best_bd = base_clean, base_bd
    best_weights = [w.copy() for w in base_weights]
    history = []

    print(f"  Greedy baseline: clean_acc={base_clean:.6f}, backdoor_rate={base_bd:.6f}, clean_floor={clean_floor:.6f}")

    tried = set()
    for cand in candidates:
        if len(selected) >= max_actions:
            break
        key_base = (cand["layer_name"], int(cand["neuron_idx"]))
        if key_base in tried:
            continue
        tried.add(key_base)

        best_trial = None
        for op in operations:
            action = dict(cand)
            action["operation"] = op
            trial_actions = selected + [action]
            trial_weights, valid = apply_actions(model, base_weights, trial_actions)
            if valid != len(trial_actions):
                continue
            model.set_weights(trial_weights)
            clean_acc, bd_rate = evaluate_model_pair(model, x_eval_clean, y_eval_clean, x_eval_bd, target_label)
            trial = {
                "action": action,
                "clean_acc": float(clean_acc),
                "backdoor_rate": float(bd_rate),
                "valid": int(valid),
            }
            if best_trial is None:
                best_trial = trial
            else:
                # Prefer lower ASR; break ties by clean accuracy.
                if (bd_rate < best_trial["backdoor_rate"] - 1e-12) or (
                    abs(bd_rate - best_trial["backdoor_rate"]) <= 1e-12 and clean_acc > best_trial["clean_acc"]
                ):
                    best_trial = trial

        model.set_weights(best_weights)
        if best_trial is None:
            continue

        improved = (
            best_trial["clean_acc"] >= clean_floor and
            best_trial["backdoor_rate"] <= best_bd - min_asr_improve
        )
        rec = dict(best_trial)
        rec["accepted"] = bool(improved)
        history.append(rec)

        if improved:
            selected.append(best_trial["action"])
            best_clean = best_trial["clean_acc"]
            best_bd = best_trial["backdoor_rate"]
            best_weights, _ = apply_actions(model, base_weights, selected)
            model.set_weights(best_weights)
            print(
                f"    accept #{len(selected):02d}: "
                f"{best_trial['action']['operation']} "
                f"{best_trial['action']['layer_name']}[{best_trial['action']['neuron_idx']}] "
                f"clean={best_clean:.6f}, backdoor={best_bd:.6f}"
            )

    if not selected:
        print("  Warning: no single Neuropatch action improved ASR under the clean constraint.")
        model.set_weights(base_weights)
    else:
        model.set_weights(best_weights)

    stats = {
        "baseline_clean_acc": float(base_clean),
        "baseline_backdoor_rate": float(base_bd),
        "clean_floor": float(clean_floor),
        "final_eval_clean_acc": float(best_clean if selected else base_clean),
        "final_eval_backdoor_rate": float(best_bd if selected else base_bd),
        "selected_count": int(len(selected)),
        "history": history,
    }
    return selected, stats


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Neuropatch-style backdoor repair under CFRepair protocol")
    parser.add_argument("repair_task", choices=["backdoor", "bd"])
    parser.add_argument("model_name")
    parser.add_argument("seed", type=int)
    parser.add_argument("epsilon", help="Kept for output naming / dual-defect test alignment")
    parser.add_argument("resnet", nargs="?", default="0")
    parser.add_argument("mode", nargs="?", default="neuron")
    parser.add_argument("suffix", nargs="?", default="bdonly")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--gtsrb-train-file", default="../data/gtsrb/train_data.pkl",
                        help="GTSRB training pickle; must NOT be a TEST split")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--neuron-num", type=int, default=50)
    parser.add_argument("--loc-sample", type=int, default=1000)
    parser.add_argument("--eval-sample", type=int, default=2000)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--per-layer-top", type=int, default=50)
    parser.add_argument("--max-clean-drop", type=float, default=0.03)
    parser.add_argument("--operation", choices=["zero", "flip", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--include-output-layer", action="store_true")
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--force-topk-fallback", action="store_true",
                        help="If greedy selects no action, apply top-k zero/flip anyway. Usually keep off for fair reporting.")
    args = parser.parse_args()

    configure_tf_session()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    resnet_flag = as_bool(args.resnet)
    tag = name_tag(args.model_name, resnet_flag)
    cfg = default_cfg(args.model_name, resnet_flag)
    if args.target_label is not None:
        cfg["target_label"] = int(args.target_label)
    target_label = int(cfg["target_label"])

    model_path = args.model_path or model_path_for(args.model_name, resnet_flag)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = args.out_dir or os.path.join(
        "loc_result", tag, str(args.seed),
        f"singledefect_neuropatch_backdoor_{args.suffix}_{args.mode}_eps{args.epsilon}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 96)
    print("Neuropatch-style BACKDOOR repair under CFRepair protocol")
    print("=" * 96)
    print(f"Model path:   {model_path}")
    print(f"Dataset tag:  {tag}")
    print(f"Target label: {target_label}")
    print(f"Output dir:   {out_dir}")
    print("=" * 96)

    model = load_model(model_path, compile=False)
    clean_x, clean_y = load_clean_data(
        args.model_name, resnet_flag=resnet_flag, gtsrb_train_file=args.gtsrb_train_file)
    clean_x = np.asarray(clean_x, dtype=np.float32)
    expected = tuple(int(v) for v in cfg["input_shape"])
    if clean_x.ndim != 4 or tuple(clean_x.shape[1:]) != expected:
        raise ValueError(
            f"Wrong TRAIN dataset for {tag}: got X={clean_x.shape}, expected (N,{expected}). "
            "The loader returned a different dataset; refusing to repair."
        )
    if len(clean_x) != len(clean_y):
        raise ValueError(f"TRAIN images/labels count mismatch: {len(clean_x)} vs {len(clean_y)}")
    model_shape = model.input_shape
    if isinstance(model_shape, tuple) and all(v is not None for v in model_shape[1:]):
        if tuple(model_shape[1:]) != expected:
            raise ValueError(f"Model expects {model_shape[1:]}, dataset config expects {expected}")
    print(f"[DATA CHECK] {tag}: N={len(clean_x)}, X={clean_x.shape}, y={np.asarray(clean_y).shape}")
    clean_y = np.asarray(clean_y).reshape(-1).astype(int)
    backdoor_x = get_backdoor_data(clean_x, args.model_name, cfg, resnet_flag=resnet_flag)

    n = min(len(clean_x), len(clean_y), len(backdoor_x))
    clean_x, clean_y, backdoor_x = clean_x[:n], clean_y[:n], backdoor_x[:n]

    # Prefer non-target samples; ASR excludes original target-class samples anyway.
    non_target = np.where(clean_y != target_label)[0]
    if len(non_target) == 0:
        raise RuntimeError("No non-target samples available for backdoor repair.")

    # Focus localization on successful backdoor-triggered samples when possible.
    pred_bd = np.argmax(run_outputs(model, model.output, backdoor_x[non_target], batch_size=args.batch_size), axis=-1)
    success_idx = non_target[pred_bd == target_label]
    if len(success_idx) < max(10, min(100, args.loc_sample // 4)):
        print(f"  Warning: only {len(success_idx)} successful backdoor samples; using non-target samples for localization.")
        success_idx = non_target
    else:
        print(f"  Using successful backdoor samples: {len(success_idx)}/{len(non_target)} non-target samples")

    rng = np.random.RandomState(args.seed)
    loc_idx = success_idx.copy()
    if args.loc_sample > 0 and len(loc_idx) > args.loc_sample:
        loc_idx = rng.choice(loc_idx, args.loc_sample, replace=False)

    eval_idx = non_target.copy()
    if args.eval_sample > 0 and len(eval_idx) > args.eval_sample:
        eval_idx = rng.choice(eval_idx, args.eval_sample, replace=False)

    t0 = time.time()
    candidates = neuron_diff_scores(
        model,
        clean_x[loc_idx],
        backdoor_x[loc_idx],
        batch_size=args.batch_size,
        include_output_layer=args.include_output_layer,
        per_layer_top=args.per_layer_top,
        candidate_pool=args.candidate_pool,
    )
    print(f"  Candidate actions: {len(candidates)}")
    top_preview = candidates[:10]
    for i, c in enumerate(top_preview):
        print(f"    cand #{i+1:02d}: {c['layer_name']}[{c['neuron_idx']}] score={c['score']:.6g}")

    if args.operation == "both":
        operations = ("zero", "flip")
    else:
        operations = (args.operation,)

    selected, greedy_stats = greedy_neuropatch(
        model,
        candidates,
        clean_x[eval_idx],
        clean_y[eval_idx],
        backdoor_x[eval_idx],
        target_label=target_label,
        max_actions=args.neuron_num,
        max_clean_drop=args.max_clean_drop,
        operations=operations,
    )

    if not selected and args.force_topk_fallback:
        print("  Applying force-topk fallback because --force-topk-fallback was set.")
        fallback_op = "zero" if args.operation in ("zero", "both") else "flip"
        selected = []
        for c in candidates[:args.neuron_num]:
            a = dict(c)
            a["operation"] = fallback_op
            selected.append(a)
        weights, valid = apply_actions(model, model.get_weights(), selected)
        selected = selected[:valid]
        model.set_weights(weights)

    train_clean = ensure_value_accuracy(model, clean_x[eval_idx], clean_y[eval_idx])
    train_bd = ensure_value_backdoor(model, backdoor_x[eval_idx], clean_y[eval_idx], target_label)

    model.save_weights(os.path.join(out_dir, "weights_final.h5"))
    model.save_weights(os.path.join(out_dir, "weights_last.h5"))

    selected_json = [
        {
            "layer_name": a["layer_name"],
            "layer_index": int(a["layer_index"]),
            "layer_class": a["layer_class"],
            "neuron_idx": int(a["neuron_idx"]),
            "score": float(a["score"]),
            "operation": a.get("operation", "zero"),
        }
        for a in selected
    ]
    with open(os.path.join(out_dir, "selected_actions.json"), "w", encoding="utf-8") as f:
        json.dump(selected_json, f, indent=2)

    stats = {
        "method": "neuropatch",
        "repair_task": "backdoor",
        "selection_policy": "activation_diff_greedy_zero_flip",
        "model_name": args.model_name,
        "name_tag": tag,
        "seed": int(args.seed),
        "epsilon": str(args.epsilon),
        "model_path": model_path,
        "target_label": int(target_label),
        "neuron_num": int(args.neuron_num),
        "loc_sample": int(len(loc_idx)),
        "eval_sample": int(len(eval_idx)),
        "candidate_pool": int(args.candidate_pool),
        "per_layer_top": int(args.per_layer_top),
        "operation": args.operation,
        "max_clean_drop": float(args.max_clean_drop),
        "selected_count": int(len(selected)),
        "eval_clean_acc": float(train_clean),
        "eval_backdoor_rate": float(train_bd),
        "elapsed_sec": float(time.time() - t0),
        "greedy_stats": greedy_stats,
        "top_candidates": top_preview,
    }
    with open(os.path.join(out_dir, "run_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    summary_path = os.path.join(out_dir, "singledefect_backdoor_best_solution.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("repair_task=backdoor\n")
        f.write("method=neuropatch\n")
        f.write("selection_policy=activation_diff_greedy_zero_flip\n")
        f.write("best_weight_file=weights_final.h5\n")
        f.write(f"target_label={target_label}\n")
        f.write(f"selected_count={len(selected)}\n")
        f.write(f"eval_clean_acc={train_clean:.10f}\n")
        f.write(f"eval_backdoor_rate={train_bd:.10f}\n")

    print("=" * 96)
    print(f"Selected actions: {len(selected)}")
    print(f"Eval clean_acc={train_clean:.10f}, backdoor_rate={train_bd:.10f}")
    print(f"Saved weights: {os.path.join(out_dir, 'weights_final.h5')}")
    print(f"Saved stats:   {os.path.join(out_dir, 'run_stats.json')}")
    print("=" * 96)


if __name__ == "__main__":
    main()
