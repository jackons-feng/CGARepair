# === 必要依赖（TF1.x 兼容）===
import os
import sys
import json
import time
import gc
import random
import hashlib
import h5py
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.models import load_model, Model
from tensorflow.compat.v1.keras.layers import Input
from tensorflow.compat.v1.keras import backend as K


# 全局配置
BATCH_SIZE = 256
CALSAL_STEP = 4
seed = 256
RQ1_BASELINE_PROTOCOL = 'rq1_baseline_multiseed_v2'
np.random.seed(seed)
tf.set_random_seed(seed)
num_examples = 512

SUPPORTED_FAIRNESS_DATASETS = ('bank', 'adult', 'default')


def _normalize_binary_groups(groups):
    """Normalize sensitive-group labels to the binary 0/1 convention."""
    groups = np.asarray(groups).reshape(-1).astype(np.int64)
    if groups.size and np.max(groups) > 1:
        groups = groups.copy()
        groups[groups == 1] = 0
        groups[groups >= 2] = 1
    return groups


def _fairness_bundle_fingerprint(fairness_data):
    """Return a deterministic SHA-256 fingerprint for an in-memory X/Y/G bundle."""
    if fairness_data is None:
        return None
    digest = hashlib.sha256()
    for arr in fairness_data:
        value = np.ascontiguousarray(np.asarray(arr))
        digest.update(str(value.dtype).encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('utf-8'))
        digest.update(value.tobytes())
    return digest.hexdigest()




def _rq1_read_h5_first(h5_path, prefer_key=None):
    if not os.path.exists(h5_path):
        raise FileNotFoundError('Expected H5 not found: {}'.format(h5_path))
    with h5py.File(h5_path, 'r') as hf:
        if prefer_key is not None and prefer_key in hf:
            return hf[prefer_key][:]
        for key in hf.keys():
            try:
                return hf[key][:]
            except Exception:
                continue
    raise RuntimeError('No readable dataset inside: {}'.format(h5_path))


def _rq1_allocate_exact_stratified(indices, labels, quota, rng):
    """Select exactly quota rows inside one sensitive group, stratified by Y."""
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels).reshape(-1)
    quota = int(quota)
    if quota < 0:
        raise ValueError('quota must be non-negative')
    if quota > len(indices):
        raise ValueError(
            'Requested {} rows from a sensitive group containing only {} rows'.format(
                quota, len(indices)
            )
        )
    if quota == 0:
        return np.asarray([], dtype=np.int64)

    group_labels = labels[indices]
    strata = []
    for label in np.unique(group_labels):
        stratum = indices[group_labels == label].copy()
        rng.shuffle(stratum)
        raw = quota * len(stratum) / float(max(len(indices), 1))
        take = min(len(stratum), int(np.floor(raw)))
        strata.append({
            'indices': stratum,
            'take': take,
            'fraction': float(raw - np.floor(raw)),
        })
    remaining = quota - sum(item['take'] for item in strata)
    order = sorted(
        range(len(strata)),
        key=lambda i: (-strata[i]['fraction'], -len(strata[i]['indices']))
    )
    while remaining > 0:
        progressed = False
        for i in order:
            item = strata[i]
            if item['take'] < len(item['indices']):
                item['take'] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    selected_parts = [
        item['indices'][:item['take']] for item in strata if item['take'] > 0
    ]
    selected = (
        np.concatenate(selected_parts).astype(np.int64)
        if selected_parts else np.asarray([], dtype=np.int64)
    )
    if len(selected) != quota:
        raise RuntimeError(
            'Internal fairness stratified-sampling error: expected {}, got {}'.format(
                quota, len(selected)
            )
        )
    rng.shuffle(selected)
    return selected


def _rq1_sample_fairness_exact(X, Y, G, total_count, seed_value=256):
    """Return exactly total_count rows, balanced across two sensitive groups."""
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)
    G = np.asarray(G).reshape(-1)
    if not (len(X) == len(Y) == len(G)):
        raise RuntimeError(
            'Fairness X/Y/G length mismatch: X={}, Y={}, G={}'.format(
                len(X), len(Y), len(G)
            )
        )
    total_count = int(total_count)
    if total_count <= 0:
        raise ValueError('k_sample must be positive for fairness, got {}'.format(total_count))
    if total_count > len(X):
        raise ValueError(
            'Requested {} fairness samples, but dataset contains only {}'.format(
                total_count, len(X)
            )
        )
    groups = np.unique(G)
    if len(groups) != 2:
        raise ValueError(
            'RQ1 fairness exact sampling requires exactly two sensitive groups; got {}'.format(
                groups.tolist()
            )
        )
    quotas = [total_count // 2, total_count - total_count // 2]
    rng = np.random.default_rng(int(seed_value))
    selected_parts = []
    for group, quota in zip(groups, quotas):
        group_indices = np.where(G == group)[0]
        selected_parts.append(
            _rq1_allocate_exact_stratified(group_indices, Y, quota, rng)
        )
    selected_idx = np.concatenate(selected_parts).astype(np.int64)
    rng.shuffle(selected_idx)
    if len(selected_idx) != total_count:
        raise RuntimeError(
            'Fairness exact-sampling mismatch: requested={}, actual={}'.format(
                total_count, len(selected_idx)
            )
        )
    return X[selected_idx], Y[selected_idx], G[selected_idx]


def _rq1_load_fairness_exact(model_name, k_sample, sample_seed=256):
    """RQ1-local exact fairness loader; external load_data.py is not modified."""
    fr_dir = os.path.join('..', 'data', 'fairness', model_name)
    X = _rq1_read_h5_first(os.path.join(fr_dir, 'x_train.h5'), 'x').astype('float32')
    Y = np.asarray(_rq1_read_h5_first(os.path.join(fr_dir, 'y_train.h5'), 'y')).reshape(-1)
    g_path = os.path.join(fr_dir, 'g_train.h5')
    if os.path.exists(g_path):
        G = np.asarray(_rq1_read_h5_first(g_path, 'g')).reshape(-1)
    else:
        raise FileNotFoundError('Fairness sensitive groups missing: ' + g_path + '. Y cannot be used as G.')

    print('=== Loading FAIRNESS data for {} (RQ1 unified exact loader) ==='.format(model_name))
    print('Fairness data loaded from {}'.format(fr_dir))
    print('Shapes before sampling -> X:{}, Y:{}, G:{}'.format(X.shape, Y.shape, G.shape))
    X, Y, G = _rq1_sample_fairness_exact(
        X, Y, G, total_count=int(k_sample), seed_value=int(sample_seed)
    )
    print('After exact sampling -> X:{}, Y:{}, G:{}'.format(X.shape, Y.shape, G.shape))
    return X, Y, G


def load_fairness_shared_bundle(model_name, requested_count):
    """Load one exact fairness X/Y/G bundle shared by all four RQ1 methods.

    The first method run creates a disk snapshot under result/fairness_shared_samples.
    Later method processes load that same snapshot, which guarantees identical
    localization/validation samples and therefore an identical baseline Gap.
    """
    dataset = str(model_name).lower()
    if dataset not in SUPPORTED_FAIRNESS_DATASETS:
        raise ValueError(
            'Unsupported fairness dataset: {}. Use one of {}.'.format(
                model_name, ', '.join(SUPPORTED_FAIRNESS_DATASETS)
            )
        )
    requested_count = int(requested_count)
    if requested_count <= 0:
        raise ValueError('num_examples must be positive, got {}'.format(requested_count))

    shared_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'result', 'fairness_shared_samples'
    )
    os.makedirs(shared_dir, exist_ok=True)
    shared_path = os.path.join(
        shared_dir, '{}_samples{}_seed{}.npz'.format(dataset, requested_count, int(seed))
    )

    g_source = os.path.join('..', 'data', 'fairness', dataset, 'g_train.h5')
    if not os.path.exists(g_source):
        raise FileNotFoundError('Required sensitive-group labels missing: ' + g_source)
    source = 'new_sample'
    if os.path.exists(shared_path):
        with np.load(shared_path, allow_pickle=False) as data:
            X = np.asarray(data['X'], dtype=np.float32)
            Y = np.asarray(data['Y']).reshape(-1).astype(np.int64)
            G = _normalize_binary_groups(data['G'])
        source = 'shared_snapshot'
    else:
        # Reset both RNGs immediately before sampling.  This also makes creation of
        # the first snapshot deterministic when load_data uses NumPy/Python random.
        np.random.seed(seed)
        random.seed(seed)
        X, Y, G = _rq1_load_fairness_exact(
            dataset, k_sample=requested_count, sample_seed=seed
        )
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y).reshape(-1).astype(np.int64)
        G = _normalize_binary_groups(G)

    if not (len(X) == len(Y) == len(G)):
        raise RuntimeError(
            'Fairness X/Y/G length mismatch: X={}, Y={}, G={}'.format(
                len(X), len(Y), len(G)
            )
        )
    if len(X) != requested_count:
        raise RuntimeError(
            'Fairness loader/snapshot contains {} rows, but {} were requested. '
            'Do not run comparison experiments with a silently reduced sample set.'.format(
                len(X), requested_count
            )
        )

    if np.array_equal(G, Y):
        raise RuntimeError('Fairness group labels equal class labels for every sample; audit shared snapshot G vs Y.')
    unique_groups, counts = np.unique(G, return_counts=True)
    if len(unique_groups) != 2:
        raise RuntimeError(
            'Fairness comparison requires exactly two groups; got {}'.format(
                unique_groups.tolist()
            )
        )
    group_counts = {str(int(g)): int(c) for g, c in zip(unique_groups, counts)}
    bundle = (X, Y, G)
    fingerprint = _fairness_bundle_fingerprint(bundle)

    if source == 'new_sample':
        tmp_path = shared_path + '.pid{}.tmp.npz'.format(os.getpid())
        np.savez_compressed(tmp_path, X=X, Y=Y, G=G)
        os.replace(tmp_path, shared_path)

    print(
        '[Fairness:shared-data] dataset={}, requested={}, actual={}, groups={}, seed={}, '
        'fingerprint={}, source={}, localization_and_validation_same=True'.format(
            dataset, requested_count, len(X), group_counts, int(seed), fingerprint, source
        )
    )
    print('[Fairness:shared-data] snapshot={}'.format(shared_path))
    return bundle


def _require_fairness_bundle(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(
            'For task=fairness, pass the shared (X, Y, G) bundle as the X argument.'
        )
    X, Y, G = value
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y).reshape(-1).astype(np.int64)
    G = _normalize_binary_groups(G)
    if not (len(X) == len(Y) == len(G)):
        raise ValueError('Shared fairness bundle has inconsistent X/Y/G lengths.')
    return X, Y, G



def parse_localization_arguments(argv):
    """CLI shared with CGARepair. The seed is a flag, never a positional TOP_K."""
    import argparse
    parser = argparse.ArgumentParser(description='RQ1 localization with reproducible seeds')
    parser.add_argument('--seed', type=int, default=256)
    parser.add_argument('--check-args', action='store_true')
    parser.add_argument('task', choices=['backdoor', 'adv', 'safety', 'fairness'])
    parser.add_argument('model_name')
    parser.add_argument('num_examples', type=int)
    parser.add_argument('extras', nargs='*')
    args = parser.parse_args(argv)
    if args.num_examples <= 0 or not 0 <= args.seed < 2**32:
        parser.error('num_examples must be positive; seed must be a uint32')
    extras = list(args.extras)
    epsilon = None
    if args.task == 'adv':
        if not extras:
            parser.error('adv requires EPSILON after SAMPLES')
        epsilon = extras.pop(0)
        try:
            float(epsilon)
        except ValueError:
            parser.error('Invalid epsilon: {}'.format(epsilon))
    resnet_18_flag = False
    if extras and extras[0].lower() in ('true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 't', 'f'):
        resnet_18_flag = extras.pop(0).lower() in ('true', '1', 'yes', 'y', 't')
    top_k = None
    if extras:
        try:
            top_k = int(extras.pop(0))
        except ValueError:
            parser.error('TOP_K must be an integer')
    if extras or (top_k is not None and top_k <= 0):
        parser.error('Unexpected positional arguments or invalid TOP_K')
    return args, epsilon, resnet_18_flag, top_k


def rq1_auto_topk(total):
    total = int(total)
    if total <= 0:
        raise RuntimeError('No candidate neurons for localization')
    return min(total, max(4, min(11, int(total * 0.01))))


def rq1_save_rankings(result_path, players, scores, selected_k):
    import csv
    base = result_path[:-4]
    ranking_path = base + '_ranked_players.txt'
    selected_path = base + '_selected_topk.txt'
    scores_path = base + '_scores.csv'
    with open(ranking_path, 'w', encoding='utf-8') as f:
        f.write(','.join(players))
    with open(selected_path, 'w', encoding='utf-8') as f:
        f.write(','.join(players[:selected_k]))
    with open(scores_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'neuron', 'score', 'in_topk'])
        for i, (name, score) in enumerate(zip(players, scores), 1):
            writer.writerow([i, name, '{:.16g}'.format(score), int(i <= selected_k)])
    return ranking_path, selected_path, scores_path


def setup_tf():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)



# Optional: LOC_FORCE_RECOMPUTE=1 bypasses existing localization-score cache.
# Set this for true timing comparisons, or a cache hit will be reported as such.
def _localization_force_recompute():
    return os.environ.get('LOC_FORCE_RECOMPUTE', '').strip().lower() in ('1', 'true', 'yes', 'on')


# 复用的工具函数（与 loc_cfrepair_test.py 对齐）
def get_layer_indices(model):
    conv, dense, batch = [], [], []
    j = 0
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        lname = type(layer).__name__
        if 'Batch' in lname:
            batch.append(j)
        elif 'Conv' in lname:
            conv.append(j)
        elif 'Dense' in lname:
            dense.append(j)
        j += len(weights)
    return conv, dense, batch


def get_layer_indices_with_meta(model):
    conv_meta, dense_meta, batch_meta = [], [], []
    j = 0
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        lname = type(layer).__name__
        rec = {'j': j, 'idx': i, 'name': getattr(layer, 'name', f'layer_{i}'), 'type': lname}
        if 'Batch' in lname:
            batch_meta.append(rec)
        elif 'Conv' in lname:
            conv_meta.append(rec)
        elif 'Dense' in lname:
            dense_meta.append(rec)
        j += len(weights)
    return conv_meta, dense_meta, batch_meta


def get_players(model, convs, denses, c_flag=True, d_flag=True, conv_meta=None, dense_meta=None):
    players = []
    weights = model.get_weights()
    if c_flag:
        if conv_meta is not None:
            for conv_j in convs:
                layer_idx = None
                for meta in conv_meta:
                    if meta['j'] == conv_j:
                        layer_idx = meta['idx']
                        break
                if layer_idx is not None:
                    players.append([f"conv_{layer_idx}_neuron_{i}" for i in range(weights[conv_j].shape[-1])])
        else:
            for index, conv in enumerate(convs):
                players.append([f"conv_{index}_neuron_{i}" for i in range(weights[conv].shape[-1])])
    if d_flag:
        if dense_meta is not None:
            for dense_j in denses:
                layer_idx = None
                for meta in dense_meta:
                    if meta['j'] == dense_j:
                        layer_idx = meta['idx']
                        break
                if layer_idx is not None:
                    players.append([f"dense_{layer_idx}_neuron_{i}" for i in range(weights[dense_j].shape[-1])])
        else:
            for index, dense in enumerate(denses):
                players.append([f"dense_{index}_neuron_{i}" for i in range(weights[dense].shape[-1])])
    players = np.sort(np.concatenate(players))
    return players


# 数据加载（与现有脚本一致）
from load_data import load_data


def sess_run(model, tensor, images, batch_size=BATCH_SIZE):
    outputs = []
    num_batches = int(np.ceil(len(images) / batch_size))
    for b in range(num_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, len(images))
        batch = images[s:e]
        outputs.append(K.get_session().run(tensor, {model.input: batch}))
    try:
        return np.concatenate(outputs, axis=0)
    except Exception:
        return np.array(outputs)


# 验证 top-k 神经元（复用 loc_cfrepair_test.py 的逻辑）
def value_safety(model, images, model_name, batch_size=BATCH_SIZE):
    if len(images.shape) == 1:
        images = images.reshape(-1, 1)
    elif len(images.shape) > 2:
        images = images.reshape(len(images), -1)
    output = sess_run(model, model.output, images, batch_size)
    pred = np.argmin(output, axis=-1)
    if model_name == 'a19':
        val = (np.sum(pred == 3) + np.sum(pred == 4)) / len(output)
    elif model_name == 'a29':
        val = (len(images) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output)
    else:
        val = (np.sum(pred != 0)) / len(output)
    return val


def value_fairness(model, X, Y_true, G, batch_size=256):
    """
    Evaluate group accuracy gap |Acc(group=1) - Acc(group=0)| for fairness tasks.
    X: 输入特征
    Y_true: 样本真实标签
    G: 群体标签 (0/1)
    """
    if X is None or Y_true is None or G is None or len(X) == 0:
        return 0.0

    # =======================
    # Credit 数据集特殊处理
    # =======================
    G = _normalize_binary_groups(G)

    output = []
    num_batches = int(np.ceil(len(X) / batch_size))
    for batch in range(num_batches):
        batch_idxs = np.arange(batch * batch_size, min((batch + 1) * batch_size, len(X)))
        input_dic = {model.input: X[batch_idxs]}
        output.append(K.get_session().run(model.output, input_dic))

    output = np.concatenate(output, axis=0)
    if output.ndim > 1 and output.shape[-1] > 1:
        pred = np.argmax(output, axis=-1).astype(int)
    else:
        pred = (output.reshape(-1) > 0.5).astype(int)

    # 分组准确率
    mask0 = (G == 0)
    mask1 = (G == 1)
    acc0 = np.mean(pred[mask0] == Y_true[mask0])
    acc1 = np.mean(pred[mask1] == Y_true[mask1])

    return float(abs(acc1 - acc0))


def validate_topk_neurons(model, X, X_perturbed, ranked_players, model_name, top_k=None, task='backdoor'):
    if top_k is None:
        top_k = max(4, min(11, int(len(ranked_players) * 0.01)))

    def calculate_metric(model, X_data, task='backdoor', model_name=None):
        if task == 'backdoor':
            out_perturbed = []
            bs = 256
            for b in range(int(np.ceil(len(X_data)/bs))):
                batch = X_data[b*bs:(b+1)*bs]
                out_perturbed.append(K.get_session().run(model.output, {model.input: batch}))
            pred_perturbed = np.argmax(np.concatenate(out_perturbed, axis=0), axis=-1)
            dic = {'gtsrb': [(32, 32, 3), (43, 33)], 'mnist': [(28, 28, 1), (10, 3)], 'fashion': [(28, 28, 1), (10, 3)]}
            target_label = dic[model_name][1][1]
            return np.sum(pred_perturbed == target_label) / len(pred_perturbed)
        elif task == 'safety':
            return value_safety(model, X_data, model_name)
        elif task == 'adv':
            labels_true, images_adv = X_data
            pred_adv = []
            bs = 256
            for b in range(int(np.ceil(len(images_adv)/bs))):
                batch = images_adv[b*bs:(b+1)*bs]
                pred_adv.append(K.get_session().run(model.output, {model.input: batch}))
            pred_adv = np.argmax(np.concatenate(pred_adv, axis=0), axis=-1)
            return np.sum(pred_adv != labels_true) / len(labels_true)
        elif task == 'fairness':
            X, Y_true, G = X_data
            return value_fairness(model, X, Y_true, G)
        else:
            raise ValueError(f"Unknown task: {task}")

    import re

    def disable_neuron_by_name(model, neuron_name):
        """
        Support both legacy names like "conv_3_neuron_10" and new names such as
        "Dense_23_4" generated by the causal ranking pipeline.
        """
        target_layer = None
        neuron_idx = None

        # Case 1: New style "<LayerType>_<LayerIndex>_<NeuronIndex>"
        match_new = re.match(r'([A-Za-z0-9]+)_(\d+)_(\d+)$', neuron_name)
        if match_new:
            _, layer_idx_str, neuron_idx_str = match_new.groups()
            layer_idx = int(layer_idx_str)
            neuron_idx = int(neuron_idx_str)
            if 0 <= layer_idx < len(model.layers):
                target_layer = model.layers[layer_idx]
        else:
            # Case 2: Legacy style "<type>_(\d+)?_neuron_<neuron>"
            match_old = re.match(r'([A-Za-z0-9]+)(?:_(\d+))?_neuron_(\d+)', neuron_name)
            if not match_old:
                return False
            layer_type, layer_idx_str, neuron_idx_str = match_old.groups()
            neuron_idx = int(neuron_idx_str)
            if layer_idx_str is not None:
                layer_idx = int(layer_idx_str)
                if 0 <= layer_idx < len(model.layers):
                    target_layer = model.layers[layer_idx]
            if target_layer is None:
                # Fallback to fuzzy name matching
                for layer in model.layers:
                    if layer_type.lower() in layer.name.lower():
                        target_layer = layer
                        break

        if target_layer is None or neuron_idx is None:
            return False

        weights = target_layer.get_weights()
        if not weights:
            return False

        kernel = weights[0]
        if kernel.ndim == 4:
            # Convolutional layer: zero out the selected output channel
            if neuron_idx >= kernel.shape[-1]:
                return False
            kernel[:, :, :, neuron_idx] = 0.0
        elif kernel.ndim == 2:
            # Dense-like layer: zero out the selected column
            if neuron_idx >= kernel.shape[-1]:
                return False
            kernel[:, neuron_idx] = 0.0
        else:
            return False

        weights[0] = kernel
        target_layer.set_weights(weights)
        return True

    # 基线与对抗样本缓存（为 1-PCR 做准备）
    labels_true_adv = None
    adv_samples = None
    pred_before_adv = None
    successful_attack_mask = None
    baseline_one_minus_pcr = None

    if task == 'backdoor':
        baseline = calculate_metric(model, X_perturbed, task, model_name=model_name)
    elif task == 'safety':
        baseline = calculate_metric(model, X, task, model_name=model_name)
    elif task == 'adv':
        labels_true_adv = np.asarray(X)
        adv_samples = np.asarray(X_perturbed)
        if len(adv_samples) == 0 or len(labels_true_adv) == 0:
            baseline = 0.0
        else:
            eval_len = min(len(adv_samples), len(labels_true_adv))
            adv_samples = adv_samples[:eval_len]
            labels_true_adv = labels_true_adv[:eval_len]
            pred_before_batches = []
            bs = 256
            for b in range(int(np.ceil(len(adv_samples) / bs))):
                batch = adv_samples[b * bs:(b + 1) * bs]
                pred_before_batches.append(K.get_session().run(model.output, {model.input: batch}))
            pred_before_adv = np.argmax(np.concatenate(pred_before_batches, axis=0), axis=-1)
            successful_attack_mask = (pred_before_adv != labels_true_adv)
            baseline = float(np.mean(successful_attack_mask))
            # 基线 1-PCR：成功攻击样本上，初始前后预测完全相同，因此为 1.0（若没有成功攻击则记为 0.0）
            baseline_one_minus_pcr = 1.0 if np.sum(successful_attack_mask) > 0 else 0.0
        print(f"Baseline ASR (adv_pred != y_clean): {baseline:.4f}")
        if baseline_one_minus_pcr is not None:
            print(f"Baseline 1-PCR: {baseline_one_minus_pcr:.4f}")
    elif task == 'fairness':
        # Exactly reuse the localization (X, Y, G) shared sample bundle.
        X, Y_true, G = _require_fairness_bundle(X)
        baseline = value_fairness(model, X, Y_true, G)
        print(f"Baseline Fairness Gap |Acc(1)-Acc(0)|: {baseline:.4f}")

    original_weights = [np.copy(w) for w in model.get_weights()]

    metrics = []
    one_minus_pcr_results = []  # 仅在 adv 任务下有意义
    disabled = 0
    k = 0
    if top_k is None:
        top_k = rq1_auto_topk(len(ranked_players))
    while disabled < top_k and k < len(ranked_players):
        neuron_name = ranked_players[k]
        print(f"Trying to disable neuron {k+1}: {neuron_name}")
        if disable_neuron_by_name(model, ranked_players[k]):
            disabled += 1
            if task == 'backdoor':
                cur = calculate_metric(model, X_perturbed, task, model_name=model_name)
            elif task == 'safety':
                cur = calculate_metric(model, X, task, model_name=model_name)
            elif task == 'adv':
                # 重新评估 ASR 与 1-PCR
                if adv_samples is not None and labels_true_adv is not None and len(adv_samples) > 0:
                    pred_after_batches = []
                    bs = 256
                    for b in range(int(np.ceil(len(adv_samples) / bs))):
                        batch = adv_samples[b * bs:(b + 1) * bs]
                        pred_after_batches.append(K.get_session().run(model.output, {model.input: batch}))
                    pred_after = np.argmax(np.concatenate(pred_after_batches, axis=0), axis=-1)
                    cur = float(np.mean(pred_after != labels_true_adv))

                    if successful_attack_mask is not None and pred_before_adv is not None:
                        if np.sum(successful_attack_mask) == 0:
                            one_minus_pcr = 0.0
                        else:
                            prediction_changed_on_attacked = (
                                pred_after[successful_attack_mask] == pred_before_adv[successful_attack_mask]
                            )
                            one_minus_pcr = float(np.mean(prediction_changed_on_attacked))
                    else:
                        one_minus_pcr = None
                else:
                    cur = 0.0
                    one_minus_pcr = None

                print(f"  Successfully disabled top-{disabled} neuron: {neuron_name}")
                print(f"  Current ASR (adv): {cur:.4f}")
                if one_minus_pcr is not None:
                    print(f"  1-PCR: {one_minus_pcr:.4f}")
                else:
                    print("  1-PCR: N/A (insufficient successful adversarial samples)")
                one_minus_pcr_results.append(one_minus_pcr)
            elif task == 'fairness':
                # X 是输入特征，G 是群体标签
                cur = value_fairness(model, X, Y_true, G)
                print(f"  Successfully disabled top-{disabled} neuron: {neuron_name}")
                print(f"  Current Fairness Gap |Acc(1)-Acc(0)|: {cur:.4f}")
            metrics.append(cur)
        k += 1

    model.set_weights(original_weights)

    # 为保持一致：adv 返回 (metrics, one_minus_pcr_results, baseline_one_minus_pcr)，其它任务 baseline_one_minus_pcr 为 None
    if task != 'adv':
        return metrics, [], None, float(baseline)
    if baseline_one_minus_pcr is None:
        baseline_one_minus_pcr = 0.0
    return metrics, one_minus_pcr_results, baseline_one_minus_pcr, float(baseline)


# ================= 因果干预定位（原实现保留但不再使用）=================
def split_keras_model(lmodel, split_index):
    model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[split_index].output)
    model2_input = Input(lmodel.layers[split_index].output_shape[1:])
    x = model2_input
    for layer in lmodel.layers[split_index + 1:]:
        x = layer(x)
    model2 = Model(inputs=model2_input, outputs=x)
    return model1, model2


def estimate_activation_range(layer_outputs, per_channel=True):
    if per_channel and layer_outputs.ndim >= 3:
        # NHWC: 取每个通道的最小/最大（跨 batch 和空间）
        mins = layer_outputs.min(axis=(0, 1, 2))
        maxs = layer_outputs.max(axis=(0, 1, 2))
        return mins, maxs
    else:
        # Dense 或 2D: 逐维度
        mins = layer_outputs.min(axis=0)
        maxs = layer_outputs.max(axis=0)
        return mins, maxs


def causal_effect_scores_for_layer(model, split_idx, X, steps=CALSAL_STEP, target_class=None):
    # 直接对中间层张量做 feed 覆盖，避免含 Merge/残差结构时的模型重建报错
    layer_tensor = model.layers[split_idx].output
    activations = sess_run(model, layer_tensor, X, batch_size=min(64, len(X)))

    # 维度信息
    is_conv = (activations.ndim >= 3)
    if is_conv:
        units = activations.shape[-1]
        mins, maxs = estimate_activation_range(activations, per_channel=True)
    else:
        units = activations.shape[-1]
        mins, maxs = estimate_activation_range(activations, per_channel=False)

    scores = np.zeros((units,), dtype=np.float32)

    # 遍历每个神经元（通道/单元）
    for c in range(units):
        vals = np.linspace(float(mins[c]), float(maxs[c]), int(max(2, steps)))
        out_seq = []
        for v in vals:
            mod = np.array(activations, copy=True)
            if is_conv:
                mod[..., c] = v
            else:
                mod[:, c] = v
            # 通过 feed_dict 覆盖中间层输出，直接获取模型最终输出
            y = K.get_session().run(model.output, {model.input: X, layer_tensor: mod})
            out_seq.append(np.mean(y, axis=0))
            del mod, y
        out_seq = np.array(out_seq)
        if target_class is not None:
            sc = float(np.max(out_seq[:, target_class]) - np.min(out_seq[:, target_class]))
        else:
            sc = float(np.sum(np.max(out_seq, axis=0) - np.min(out_seq, axis=0)))
        scores[c] = sc
        del out_seq
        gc.collect()

    return scores


# ================= 新的统一“综合梯度差异定位方法” =================
def get_integradient_alpha_on_layer(model, layer_idx, x, alpha=0.8, steps=7):
    """
    在给定中间层 layer_idx 上，对输入 x 计算综合梯度（类似 ACAS 代码 get_integradient_alpha）。
    返回张量形状与该层输出相同：(batch, H, W, C) 或 (batch, units)。
    """
    if x is None or len(x) == 0:
        return None

    x = np.asarray(x, dtype=np.float32)

    input_tensor = model.input
    output_tensor = model.output
    layer_tensor = model.layers[layer_idx].output

    # 输出对中间层激活的梯度
    grads = K.gradients(output_tensor, layer_tensor)[0]
    iterate = K.function([input_tensor], [grads, layer_tensor])

    baseline = alpha * x
    x_diff = x - baseline

    total_grads = []
    total_layers = []

    for i in range(steps + 1):
        inputs = baseline + (float(i) / steps) * x_diff
        grads_value, layer_value = iterate([inputs])
        total_grads.append(grads_value)
        total_layers.append(layer_value)

    avg_grads = np.mean(total_grads, axis=0)
    diff_layer = total_layers[-1] - total_layers[0]
    grads_value = diff_layer * avg_grads

    # 归一化
    grads_value /= (np.sqrt(np.mean(np.square(grads_value))) + K.epsilon())
    return grads_value


def ig_diff_scores_for_layer(model, layer_idx, X_a, X_b, alpha=0.8, steps=7):
    """
    通用的“定位方法”：
    - X_a: 第一批样本（可理解为“原始/正常”）
    - X_b: 第二批样本（可理解为“反事实/异常”）
    - 对两批样本在 layer_idx 上分别算综合梯度，做差并取绝对值，得到每个神经元一个分数。
    """
    if X_a is None or X_b is None or len(X_a) == 0 or len(X_b) == 0:
        # 返回与该层神经元数目相同的全 0 分数
        layer_tensor = model.layers[layer_idx].output
        dummy = sess_run(model, layer_tensor, np.zeros_like(X_a[:1]))
        if dummy.ndim >= 3:
            units = dummy.shape[-1]
        else:
            units = dummy.shape[-1]
        return np.zeros((units,), dtype=np.float32)

    X_a = np.asarray(X_a, dtype=np.float32)
    X_b = np.asarray(X_b, dtype=np.float32)

    ig_a = get_integradient_alpha_on_layer(model, layer_idx, X_a, alpha=alpha, steps=steps)
    ig_b = get_integradient_alpha_on_layer(model, layer_idx, X_b, alpha=alpha, steps=steps)

    # 压缩到每个神经元一个值
    if ig_a.ndim >= 4:
        ig_a_mean = np.mean(ig_a, axis=(0, 1, 2))
        ig_b_mean = np.mean(ig_b, axis=(0, 1, 2))
    else:
        ig_a_mean = np.mean(ig_a, axis=0)
        ig_b_mean = np.mean(ig_b, axis=0)

    diff = ig_b_mean - ig_a_mean
    scores = np.abs(diff).astype(np.float32)
    return scores


def compute_causal_intervention_values(task, model_name, num_examples=1024, top_k=None,
                                       resnet_18_flag=False, epsilon=None, fairness_data=None):
    """
    统一的“定位方法”版本：
    - 不再区分任务使用不同的定位思想；
    - 所有任务都走：构造两批样本 X_a / X_b -> 在每一层上用 IG 做两批的差值 -> 排序得到神经元重要性。
    """
    global LOCALIZATION_CACHE_HIT
    LOCALIZATION_CACHE_HIT = False
    setup_tf()

    # 缓存目录
    if resnet_18_flag:
        cache_dir = os.path.join('result', 'loc_result', f'{model_name}_resnet')
    else:
        cache_dir = os.path.join('result', 'loc_result', model_name)
    os.makedirs(cache_dir, exist_ok=True)
    fairness_sample_fingerprint = (_fairness_bundle_fingerprint(fairness_data) if task == 'fairness' else None)
    if task == 'adv' and epsilon is not None:
        cache_file = os.path.join(cache_dir, f'ccbr_{task}_{epsilon}_{num_examples}_seed{seed}_top{top_k}.json')
    else:
        cache_file = os.path.join(cache_dir, f'ccbr_{task}_{num_examples}_seed{seed}_top{top_k}.json')
    if os.path.exists(cache_file) and not _localization_force_recompute():
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_cache = (
            data.get('rq1_protocol_version') == RQ1_BASELINE_PROTOCOL
            and int(data.get('seed', -1)) == int(seed)
            and data.get('task') == task
            and data.get('model') == model_name
            and len(data.get('players', [])) == len(data.get('scores', [])) > 0
            and (task != 'fairness' or data.get('fairness_sample_fingerprint') == fairness_sample_fingerprint)
        )
        if valid_cache:
            print('[CCBR:cache] Valid ranking cache: ' + cache_file)
            LOCALIZATION_CACHE_HIT = True
            return data['players'], data['scores']
        print('[CCBR:cache] Legacy/incompatible cache ignored: ' + cache_file)

    # 加载模型
    if resnet_18_flag:
        model_path = os.path.join('..', 'model', f'{model_name}_resnet.h5')
    else:
        if task == 'adv':
            model_path = os.path.join('..', 'model', f'{model_name}_adv.h5')
        else:
            model_path = os.path.join('..', 'model', f'{model_name}.h5')
    model = load_model(model_path, compile=False)

    # ========= 加载数据，并构造“两批样本” X_a / X_b =========
    X = None
    X_a = None
    X_b = None

    if task == 'backdoor':
        # X：干净样本，X_perturbed：带后门样本
        X, Y, X_perturbed = load_data(model_name, task='backdoor', k_sample=num_examples)
        X_a = X
        X_b = X_perturbed

    elif task == 'safety':
        # ct_x：满足安全约束的样本，dr_x：违反（或相反）样本
        ct_x, dr_x = load_data(model_name, task='safety', k_sample=num_examples)
        X = ct_x
        X_a = ct_x
        X_b = dr_x

    elif task == 'adv':
        # X_clean：原始干净样本，X_adv：对抗样本
        X_adv, X_clean, Y_clean = load_data(model_name, task='adv', epsilon=epsilon, k_sample=num_examples)
        X = X_clean
        X_a = X_clean
        X_b = X_adv

    elif task == 'fairness':
        if fairness_data is None:
            fairness_data = load_fairness_shared_bundle(model_name, num_examples)
        X, Y, G = _require_fairness_bundle(fairness_data)
        X = np.asarray(X, dtype=np.float32)
        G = np.asarray(G)
        # 把多值 G 压成 0 / 1（兼容 Credit）
        if G.max() > 1:
            G = G.copy()
            G[G == 1] = 0
            G[G == 2] = 1
        mask0 = (G == 0)
        mask1 = (G == 1)
        if np.sum(mask0) > 0 and np.sum(mask1) > 0:
            X_a = X[mask0]
            X_b = X[mask1]
        else:
            # 没有两组时，退化为随机划分两半
            X_a = X[:len(X)//2]
            X_b = X[len(X)//2:]

    else:
        raise ValueError(f"Unknown task: {task}")

    # 如果某些任务下 X_b 为空，再做一次兜底切分
    if X_a is None or X_b is None or len(X_a) == 0 or len(X_b) == 0:
        X_all = X if X is not None else X_a
        X_all = np.asarray(X_all, dtype=np.float32)
        half = len(X_all) // 2
        if half == 0:
            raise ValueError("Not enough data to construct two groups for IG-based localization.")
        X_a = X_all[:half]
        X_b = X_all[half:]

    # ========= 选择有权重的层 =========
    weighted_layers = []
    for i, layer in enumerate(model.layers):
        try:
            weights = layer.get_weights()
            if len(weights) > 0:
                lname = type(layer).__name__
                if any(c in lname for c in ['Conv', 'Dense']):
                    weighted_layers.append(i)
        except Exception:
            continue

    # ========= 新的统一“定位方法”：IG 差值评分 =========
    layer_to_scores = {}
    for li in weighted_layers:
        try:
            print(f"Computing IG-diff scores for layer {li}: {model.layers[li].name}")
            scores = ig_diff_scores_for_layer(
                model, li, X_a, X_b, alpha=0.8, steps=CALSAL_STEP
            )
            layer_to_scores[li] = scores
        except Exception as e:
            raise RuntimeError('CCBR IG-diff scoring failed for layer {}: {}'.format(li, e)) from e

    # ========= 汇总为全局排名 =========
    conv_meta, dense_meta, _ = get_layer_indices_with_meta(model)
    players_all = []
    scores_all = []
    for li, scores in layer_to_scores.items():
        layer_obj = model.layers[li]
        layer_type = type(layer_obj).__name__
        for idx, sc in enumerate(scores):
            # {LayerType}_{layerIndex}_{neuronIndex}
            players_all.append(f"{layer_type}_{li}_{idx}")
            scores_all.append(float(sc))

    items = list(zip(players_all, scores_all))
    items.sort(key=lambda x: x[1], reverse=True)
    ranked_players = [p for p, _ in items]
    ranked_scores = [s for _, s in items]

    # 写缓存
    out = {
        'task': task,
        'model': model_name,
        'num_examples': num_examples,
        'players': ranked_players,
        'scores': ranked_scores,
        'method': 'ccbr',
        'seed': int(seed),
        'rq1_protocol_version': RQ1_BASELINE_PROTOCOL,
        'fairness_sample_fingerprint': fairness_sample_fingerprint,
        'time': int(time.time())
    }
    if task == 'adv' and epsilon is not None:
        out['epsilon'] = epsilon
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    return ranked_players, ranked_scores


def main():
    global seed
    args, epsilon, resnet_18_flag, top_k = parse_localization_arguments(sys.argv[1:])
    task = args.task
    model_name = args.model_name
    num_examples = args.num_examples
    seed = int(args.seed)
    print('[CCBR] task={} model={} samples={} epsilon={} resnet={} seed={} top_k={}'.format(
        task, model_name, num_examples, epsilon, resnet_18_flag, seed,
        'auto' if top_k is None else top_k), flush=True)
    if args.check_args:
        return

    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    start_time = time.time()
    localization_start = time.perf_counter()
    fairness_shared_data = None
    if task == 'fairness':
        fairness_shared_data = load_fairness_shared_bundle(model_name, num_examples)

    players, scores = compute_causal_intervention_values(
        task, model_name, num_examples, top_k, resnet_18_flag, epsilon,
        fairness_data=fairness_shared_data)
    if not players or len(players) != len(scores):
        raise RuntimeError('No valid CCBR ranking')
    if len(set(players)) != len(players):
        raise RuntimeError('Duplicate neuron identifiers in CCBR ranking')
    actual_top_k = rq1_auto_topk(len(players)) if top_k is None else min(len(players), top_k)
    print('[CCBR] Total neurons={} selected Top-{}'.format(len(players), actual_top_k))
    print('=== Validating Top-{} Neurons ==='.format(actual_top_k))

    localization_wall_time_sec = float(time.perf_counter() - localization_start)
    cache_hit = bool(LOCALIZATION_CACHE_HIT)
    localization_time_sec = None if cache_hit else localization_wall_time_sec

    if resnet_18_flag:
        model_path = os.path.join('..', 'model', '{}_resnet.h5'.format(model_name))
    elif task == 'adv':
        model_path = os.path.join('..', 'model', '{}_adv.h5'.format(model_name))
    else:
        model_path = os.path.join('..', 'model', '{}.h5'.format(model_name))
    model = load_model(model_path, compile=False)

    if task != 'fairness':
        np.random.seed(seed)
        random.seed(seed)
    if task == 'backdoor':
        X, Y, X_perturbed = load_data(model_name, task='backdoor', k_sample=num_examples)
        results = validate_topk_neurons(model, X, X_perturbed, players, model_name, top_k=actual_top_k, task=task)
    elif task == 'safety':
        ct_x, dr_x = load_data(model_name, task='safety', k_sample=num_examples)
        results = validate_topk_neurons(model, ct_x, None, players, model_name, top_k=actual_top_k, task=task)
    elif task == 'adv':
        X_adv, X_clean, Y_clean = load_data(model_name, task='adv', epsilon=epsilon, k_sample=num_examples)
        results = validate_topk_neurons(model, Y_clean, X_adv, players, model_name, top_k=actual_top_k, task=task)
    else:
        results = validate_topk_neurons(
            model, fairness_shared_data, None, players, model_name, top_k=actual_top_k, task=task)

    metrics, one_minus_pcr_results, baseline_one_minus_pcr, baseline_metric = results
    if len(metrics) != actual_top_k or any(not np.isfinite(float(v)) for v in metrics):
        raise RuntimeError('CCBR cumulative ablation incomplete: {}/{}'.format(len(metrics), actual_top_k))
    if resnet_18_flag:
        out_dir = os.path.join('result', task, '{}_resnet'.format(model_name))
    else:
        out_dir = os.path.join('result', task, model_name)
    if task == 'adv' and epsilon is not None:
        out_dir = os.path.join(out_dir, 'epsilon_{}'.format(epsilon))
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(
        out_dir, 'top{}_samples{}_ccbr_seed{}.txt'.format(actual_top_k, num_examples, seed))
    ranking_path, selected_path, scores_path = rq1_save_rankings(
        result_path, players, scores, actual_top_k)
    meta = {
        'task': task, 'model': model_name, 'num_examples': int(num_examples),
        'requested_num_examples': int(num_examples),
        'actual_sample_count': int(num_examples),
        'seed': int(seed), 'top_k': int(actual_top_k), 'num_players': len(players),
        'method': 'ccbr', 'rq1_protocol_version': RQ1_BASELINE_PROTOCOL,
        'time_sec': float(time.time() - start_time),
        'localization_time_sec': localization_time_sec,
        'localization_wall_time_sec': localization_wall_time_sec,
        'localization_cache_hit': cache_hit,
        'localization_validation_same_samples': bool(task == 'fairness'),
        'fairness_sample_fingerprint': (
            _fairness_bundle_fingerprint(fairness_shared_data) if task == 'fairness' else None),
        'baseline_metric_before_disable': float(baseline_metric),
        'ranked_players_path': ranking_path,
        'selected_topk_path': selected_path,
        'scores_csv_path': scores_path,
        'top{}_metric_results'.format(actual_top_k): [float(v) for v in metrics],
    }
    if task == 'adv':
        meta['baseline_one_minus_pcr'] = baseline_one_minus_pcr
        meta['top{}_one_minus_pcr_results'.format(actual_top_k)] = one_minus_pcr_results
        meta['epsilon'] = epsilon
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('[CCBR] Saved: {}'.format(result_path), flush=True)


if __name__ == '__main__':
    main()
