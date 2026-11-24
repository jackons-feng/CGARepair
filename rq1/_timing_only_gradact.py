# === 必要依赖 ===
import os
import sys
import json
import time
import numpy as np
import h5py
import threading
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os, json, time, gc
import hashlib
# ✅ 只使用 TensorFlow 1.x 兼容模式
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ✅ 使用 TF1.x 版本的 Keras backend
from tensorflow.compat.v1.keras.models import load_model, clone_model
from tensorflow.compat.v1.keras import backend as K

# ✅ TensorFlow GPU 内存按需分配
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

num_examples = 512
BATCH_SIZE = 256
SAVE_FREQ = 10
PARALLEL_WORKERS = 10  # 并行worker数（>1启用并行）
RECORD_MOD = 10       # 记录频率：按 PARALLEL_WORKERS * counter % RECORD_MOD == 0 触发
# 全局时间限制：整个程序不超过24小时
GLOBAL_TIME_LIMIT = 43200  # 24小时86400

seed=256
np.random.seed(seed)
tf.set_random_seed(seed)


import csv

GRADACT_VERSION = 'CGARepair_GradAct_20261020_v3'
GRADACT_BATCH_SIZE = 32
BACKDOOR_TARGETS = {'gtsrb': 33, 'mnist': 3, 'fashion': 3}

import random
random.seed(seed)

from load_data import load_data

SUPPORTED_FAIRNESS_DATASETS = ('bank', 'adult', 'default')

def _fairness_bundle_fingerprint(bundle):
    """Deterministic SHA-256 fingerprint for the exact in-memory X/Y/G bundle."""
    if bundle is None or not isinstance(bundle, (tuple, list)) or len(bundle) != 3:
        return None
    digest = hashlib.sha256()
    for value in bundle:
        arr = np.ascontiguousarray(np.asarray(value))
        digest.update(str(arr.dtype).encode('utf-8'))
        digest.update(str(tuple(arr.shape)).encode('utf-8'))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def setup_tf():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


def get_layer_indices(model):
    """获取卷积层和全连接层的索引"""
    conv, dense, batch = [], [], []
    j = 0  # 全局权重列表下标
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        lname = type(layer).__name__
        # 记录当前层在全局weights中的起始索引 j
        if 'Batch' in lname:
            batch.append(j)
        elif 'Conv' in lname:
            # 卷积核通常在第一个权重位置
            conv.append(j)
        elif 'Dense' in lname:
            # 全连接的kernel在第一个权重位置
            dense.append(j)
        # 用该层实际权重数量推进全局下标，避免固定步长导致越界
        j += len(weights)
    return conv, dense, batch


def get_layer_indices_with_meta(model):
    """Return convs/denses with mapping to model layer index and name.
    Each entry: {'j': weight_start_index, 'idx': model_layer_index, 'name': layer.name, 'type': type(layer).__name__}
    """
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
    """生成神经元玩家列表"""
    players = []
    weights = model.get_weights()
    
    if c_flag:
        if conv_meta is not None:
            # 使用实际的层索引
            for conv_j in convs:
                # 找到对应的层索引
                layer_idx = None
                for meta in conv_meta:
                    if meta['j'] == conv_j:
                        layer_idx = meta['idx']
                        break
                if layer_idx is not None:
                    players.append([f'conv_{layer_idx}_{i}' for i in range(weights[conv_j].shape[-1])])

        else:
            # 回退到原始逻辑
            for index, conv in enumerate(convs):
                players.append([f'conv_{layer_idx}_{i}' for i in range(weights[conv_j].shape[-1])])

    
    if d_flag:
        if dense_meta is not None:
            # 使用实际的层索引
            for dense_j in denses:
                # 找到对应的层索引
                layer_idx = None
                for meta in dense_meta:
                    if meta['j'] == dense_j:
                        layer_idx = meta['idx']
                        break
                if layer_idx is not None:
                    players.append([f'dense_{layer_idx}_{i}' for i in range(weights[dense_j].shape[-1])])

        else:
            # 回退到原始逻辑
            for index, dense in enumerate(denses):
                players.append([f'dense_{layer_idx}_{i}' for i in range(weights[dense_j].shape[-1])])

    
    players = np.sort(np.concatenate(players))
    return players


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
        print('[Fairness:RQ1] No g_train.h5 found; using Y as group tags.')
        G = Y.copy()

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
        tmp_path = shared_path + '.tmp.npz'
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
        raise ValueError('For task=fairness, pass the shared (X, Y, G) bundle as X.')
    X, Y, G = value
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y).reshape(-1).astype(np.int64)
    G = _normalize_binary_groups(G)
    if not (len(X) == len(Y) == len(G)):
        raise ValueError('Shared fairness bundle has inconsistent X/Y/G lengths.')
    return X, Y, G


def sess_run(model, variable, images, batch_size=BATCH_SIZE):
    """分批运行模型"""
    output = []
    num_batches = int(np.ceil(len(images) / batch_size))
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(images))
        batch_idxs = np.arange(start_idx, end_idx)
        
        # 确保batch_idxs是整数数组
        batch_idxs = batch_idxs.astype(int)
        
        # 处理不同的数据格式
        if len(images.shape) == 1:
            # 1D数据（如safety任务）
            batch_images = images[batch_idxs]
        else:
            # 多维数据（如图像数据）
            batch_images = images[batch_idxs]
        
        input_dic = {model.input: batch_images}
        output.append(K.get_session().run(variable, input_dic))
    try:
        return np.concatenate(output, 0)
    except:
        return np.array(output)


def value_backdoor(model, images, images_p, labels, target_label, batch_size=BATCH_SIZE):
    """计算后门攻击成功率"""
    pred_p = np.argmax(sess_run(model, model.output, images_p, batch_size=batch_size), axis=-1)
    val = (np.sum(pred_p == target_label) - np.sum((labels == target_label) & (pred_p == target_label))) / len(images)
    return val


def value_safety(model, images, model_name, batch_size=BATCH_SIZE):
    """计算安全违规率"""
    # 确保images是正确的格式
    if len(images.shape) == 1:
        # 如果是1D数据，需要reshape为2D
        images = images.reshape(-1, 1)
    elif len(images.shape) == 2:
        # 如果已经是2D，保持不变
        pass
    else:
        # 其他情况，尝试reshape
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


def value_adv(model, images_clean, images_adv, labels_true, batch_size=BATCH_SIZE):
    """计算对抗攻击成功率：adv 预测 != 真实标签"""
    pred_adv = np.argmax(sess_run(model, model.output, images_adv, batch_size=batch_size), axis=-1)
    val = np.sum(pred_adv != labels_true) / len(labels_true)
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
    
    # Unified sensitive-group normalization used by every RQ1 localization method.
    G = _normalize_binary_groups(G)
    
    output = []
    num_batches = int(np.ceil(len(X) / batch_size))
    for batch in range(num_batches):
        batch_idxs = np.arange(batch * batch_size, min((batch + 1) * batch_size, len(X)))
        input_dic = {model.input: X[batch_idxs]}
        output.append(K.get_session().run(model.output, input_dic))
        
    # =======================
    # 修复后的模型输出处理 (方案A)
    # =======================
    output_array = np.concatenate(output, 0)
    
    if len(output_array.shape) > 1 and output_array.shape[1] == 2:
        # 针对输出维度为 (N, 2) 的情况（如 Softmax），取概率最大类的索引
        pred = np.argmax(output_array, axis=1)
    else:
        # 针对输出维度为 (N, 1) 或一维的情况（如 Sigmoid），按阈值划分
        pred = (output_array.flatten() > 0.5).astype(int)

    mask0 = (G == 0)
    mask1 = (G == 1)
    acc0 = np.mean(pred[mask0] == Y_true[mask0])
    acc1 = np.mean(pred[mask1] == Y_true[mask1])

    return float(abs(acc1 - acc0))


def _normalize_binary_groups(groups):
    """Normalize supported fairness sensitive-group labels to integer {0, 1}."""
    groups = np.asarray(groups).reshape(-1).astype(np.int64)
    if groups.size and np.max(groups) > 1:
        groups = groups.copy()
        groups[groups == 1] = 0
        groups[groups >= 2] = 1
    unique = np.unique(groups)
    if len(unique) != 2:
        raise ValueError(
            'Fairness localization requires exactly two sensitive groups; got {}'.format(
                unique.tolist()
            )
        )
    return groups.astype(np.int64)


def _predictions_from_output(output_array):
    """Support sigmoid (N,), sigmoid (N,1), and softmax (N,C) outputs."""
    output_array = np.asarray(output_array)
    if output_array.ndim == 1:
        return (output_array > 0.5).astype(np.int64)
    if output_array.shape[-1] == 1:
        return (output_array.reshape(-1) > 0.5).astype(np.int64)
    return np.argmax(output_array, axis=-1).astype(np.int64)


def _fairness_stats_from_logits(logits, y_true, groups):
    """Return overall accuracy and the signed/absolute group-accuracy gap."""
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    groups = _normalize_binary_groups(groups)
    pred = _predictions_from_output(logits)
    n = min(len(pred), len(y_true), len(groups))
    if n == 0:
        raise ValueError('Empty fairness evaluation data')
    pred = pred[:n]
    y_true = y_true[:n]
    groups = groups[:n]
    mask0 = groups == 0
    mask1 = groups == 1
    if not np.any(mask0) or not np.any(mask1):
        raise ValueError('Both sensitive groups must be present')
    accuracy = float(np.mean(pred == y_true))
    acc0 = float(np.mean(pred[mask0] == y_true[mask0]))
    acc1 = float(np.mean(pred[mask1] == y_true[mask1]))
    signed_gap = float(acc1 - acc0)
    direction = 1.0 if signed_gap >= 0.0 else -1.0
    return {
        'accuracy': accuracy,
        'acc_group0': acc0,
        'acc_group1': acc1,
        'signed_gap': signed_gap,
        'gap': float(abs(signed_gap)),
        'direction': float(direction),
    }


def _fairness_accuracy_stats(model, X, Y_true, G, batch_size=BATCH_SIZE):
    X = np.asarray(X)
    Y_true = np.asarray(Y_true).reshape(-1).astype(np.int64)
    G = _normalize_binary_groups(G)
    n = min(len(X), len(Y_true), len(G))
    if n == 0:
        raise ValueError('Empty fairness dataset')
    logits = sess_run(model, model.output, X[:n], batch_size=batch_size)
    return _fairness_stats_from_logits(logits, Y_true[:n], G[:n])


def _parse_player_layer_and_neuron(model, player_name):
    """Resolve names generated by this file, e.g. Dense_1_231."""
    if '_neuron_' in player_name:
        layer_name, neuron_text = player_name.rsplit('_neuron_', 1)
        try:
            neuron_idx = int(neuron_text)
        except Exception:
            return None, None
        for layer_idx, layer in enumerate(model.layers):
            if getattr(layer, 'name', '').lower() == layer_name.lower():
                return layer_idx, neuron_idx
        return None, None
    try:
        base, neuron_text = player_name.rsplit('_', 1)
        layer_idx = int(base.rsplit('_', 1)[-1])
        return layer_idx, int(neuron_text)
    except Exception:
        return None, None


def remove_players(model, players, convs, denses, b_flag=False):
    """移除指定的神经元"""
    if isinstance(players, str):
        players = [players]
    for player in players:
        if player.split('_')[0] == "conv":
            weight_index = convs[int(player.split('_')[1])]
            var_vals = model.get_weights()
            var_vals[weight_index][..., int(player.split('_')[-1])] = 0.  # conv weight
            if b_flag:
                var_vals[weight_index + 4][int(player.split('_')[-1])] = 1.  # batch norm variance
                var_vals[weight_index + 3][int(player.split('_')[-1])] = 0.  # batch norm mean
            model.set_weights(var_vals)
        else:
            weight_index = denses[int(player.split('_')[1])]
            var_vals = model.get_weights()
            neuron_idx = int(player.split('_')[-1])
            # 置零该输出神经元对应的kernel列
            var_vals[weight_index][..., neuron_idx] = 0.
            # 若存在bias，强行设为一个极小负值，避免仍被argmax选中
            """if weight_index + 1 < len(var_vals):
                try:
                    if var_vals[weight_index + 1].shape[0] > neuron_idx:
                        var_vals[weight_index + 1][neuron_idx] = -1e6
                except Exception:
                    pass
            """
            model.set_weights(var_vals)


def validate_topk_neurons(model, X, X_perturbed, ranked_players, model_name, top_k=None, task='backdoor'):
    """
    验证topk神经元的效果：逐步置0计算current_metric
    """
    # 计算自适应top_k：模型1%的神经元，最少4个，最多11个
    if top_k is None:
        top_k = max(4, min(11, int(len(ranked_players) * 0.01)))
    print(f"Using adaptive top_k: {top_k} ({top_k/len(ranked_players)*100:.1f}% of total neurons)")
    def calculate_metric(model, X_data, task='backdoor', model_name=None):
        """Calculate task-specific metric"""
        if task == 'backdoor':
            out_perturbed = []
            bs = 256
            for b in range(int(np.ceil(len(X_data)/bs))):
                batch = X_data[b*bs:(b+1)*bs]
                out_perturbed.append(K.get_session().run(model.output, {model.input: batch}))
            pred_perturbed = np.argmax(np.concatenate(out_perturbed, axis=0), axis=-1)
            
            dic = {'gtsrb': [(32, 32, 3), (43, 33)], 'mnist': [(28, 28, 1), (10, 3)], 'fashion': [(28, 28, 1), (10, 3)]}
            target_label = dic[model_name][1][1]
            asr = np.sum(pred_perturbed == target_label) / len(pred_perturbed)
            return asr
        elif task == 'safety':
            # 确保X_data是正确的格式
            if len(X_data.shape) == 1:
                X_data = X_data.reshape(-1, 1)
            elif len(X_data.shape) > 2:
                X_data = X_data.reshape(len(X_data), -1)
            return value_safety(model, X_data, model_name)
        elif task == 'adv':
            # X_data 在 adv 场景下传入的是 (labels_true, images_adv) 的二元组
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
    labels_true = None
    adv_samples = None
    pred_before = None
    successful_attack_mask = None
    baseline_one_minus_pcr = None
    bs = 256

    def disable_neuron_by_name(model, neuron_name, convs, denses):
        """
        根据名称禁用指定神经元/输出通道。
        支持：
        - 合成名 "{LayerType}_{layerIndex}_{neuron}"，如 "Dense_8_3"
        - 兼容旧格式 "<layer_name>_neuron_<idx>"
        """
        target_layer = None
        neuron_idx = None
        if '_neuron_' in neuron_name:
            try:
                layer_name_part, neuron_idx_str = neuron_name.rsplit('_neuron_', 1)
                neuron_idx = int(neuron_idx_str)
            except Exception:
                return False
            for l in model.layers:
                if getattr(l, 'name', '').lower() == layer_name_part.lower():
                    target_layer = l
                    break
            if target_layer is None:
                for l in model.layers:
                    if layer_name_part.lower() in getattr(l, 'name', '').lower():
                        target_layer = l
                        break
            if target_layer is None:
                return False
        else:
            try:
                base, neuron_idx_str = neuron_name.rsplit('_', 1)
                layer_idx_str = base.split('_')[-1]
                layer_idx = int(layer_idx_str)
                neuron_idx = int(neuron_idx_str)
            except Exception:
                return False
            if layer_idx < 0 or layer_idx >= len(model.layers):
                return False
            target_layer = model.layers[layer_idx]

        weights = target_layer.get_weights()
        if not weights:
            return False
        kernel = weights[0]
        if kernel.ndim == 4:
            out_c = int(kernel.shape[-1])
            if neuron_idx >= out_c:
                return False
            kernel[:, :, :, neuron_idx] = 0.0
            weights[0] = kernel
            target_layer.set_weights(weights)
            return True
        elif kernel.ndim == 2:
            out_dim = int(kernel.shape[-1])
            if neuron_idx >= out_dim:
                return False
            kernel[:, neuron_idx] = 0.0
            weights[0] = kernel
            target_layer.set_weights(weights)
            return True
        else:
            return False

    
    # 计算基线指标
    if task == 'backdoor':
        baseline_metric = calculate_metric(model, X_perturbed, task, model_name=model_name)
        print(f"Baseline ASR: {baseline_metric:.4f}")
    elif task == 'safety':
        baseline_metric = calculate_metric(model, X, task, model_name=model_name)
        print(f"Baseline Safety Violation Rate: {baseline_metric:.4f}")
    elif task == 'adv':
        # 这里 X 传入的是 y_clean（真实标签），X_perturbed 传入的是 X_adv
        labels_true = np.asarray(X) if X is not None else None
        adv_samples = np.asarray(X_perturbed) if X_perturbed is not None else None
        pred_before = None
        successful_attack_mask = None
        if adv_samples is None or labels_true is None or len(adv_samples) == 0 or len(labels_true) == 0:
            baseline_metric = 0.0
            print("⚠️ No adversarial samples or labels available; Initial ASR set to 0.0")
        else:
            eval_len = min(len(adv_samples), len(labels_true))
            if eval_len < len(adv_samples) or eval_len < len(labels_true):
                print(f"⚠️ Mismatched adv/label lengths (adv: {len(adv_samples)}, labels: {len(labels_true)}); using first {eval_len} pairs.")
            adv_samples = adv_samples[:eval_len]
            labels_true = labels_true[:eval_len]
            pred_before_batches = []
            for b in range(int(np.ceil(len(adv_samples)/bs))):
                batch = adv_samples[b*bs:(b+1)*bs]
                pred_before_batches.append(K.get_session().run(model.output, {model.input: batch}))
            pred_before = np.argmax(np.concatenate(pred_before_batches, axis=0), axis=-1)
            successful_attack_mask = (pred_before != labels_true)
            baseline_metric = float(np.mean(successful_attack_mask))
            # 基线 1-PCR：在成功攻击样本上，初始前后预测完全相同，因此为 1.0（若没有成功攻击则记为 0.0）
            baseline_one_minus_pcr = 1.0 if np.sum(successful_attack_mask) > 0 else 0.0
            print(f"Initial ASR (adv_pred != y_clean): {baseline_metric:.4f}")
            print(f"Baseline 1-PCR: {baseline_one_minus_pcr:.4f}")
    elif task == 'fairness':
        # X must be the exact (X, Y_true, G) bundle used by localization.
        # A fallback reload is kept only for backward compatibility with external callers.
        if isinstance(X, (tuple, list)) and len(X) == 3:
            X, Y_true, G = X
            print('[Fairness:validation] Reusing the exact localization sample bundle.')
        else:
            print('[Fairness:validation][warning] No shared bundle supplied; loading a fallback sample.')
            X, Y_true, G = load_fairness_shared_bundle(model_name, num_examples)
        X = np.asarray(X, dtype=np.float32)
        Y_true = np.asarray(Y_true).reshape(-1).astype(np.int64)
        G = _normalize_binary_groups(G)
        n_fair = min(len(X), len(Y_true), len(G))
        X, Y_true, G = X[:n_fair], Y_true[:n_fair], G[:n_fair]
        print(f'[Fairness:validation] sample_count={n_fair}')
        baseline_metric = value_fairness(model, X, Y_true, G)
        print(f"Baseline Fairness Gap |Acc(1)-Acc(0)|: {baseline_metric:.4f}")
    
    # 保存原始权重
    original_weights = [np.copy(w) for w in model.get_weights()]
    
    # 获取层索引
    convs, denses, _ = get_layer_indices(model)
    
    # 逐步禁用top-k神经元（只处理可训练层的神经元）
    metric_results = []
    one_minus_pcr_results = []  # 仅在 adv 任务下有意义
    disabled_count = 0
    k = 0
    
    while disabled_count < top_k and k < len(ranked_players):
        neuron_name = ranked_players[k]
        print(f"Trying to disable neuron {k+1}: {neuron_name}")
        
        # 禁用神经元
        success = disable_neuron_by_name(model, neuron_name, convs, denses)
        if success:
            disabled_count += 1
            # 计算当前指标
            if task == 'backdoor':
                current_metric = calculate_metric(model, X_perturbed, task, model_name=model_name)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current ASR: {current_metric:.4f}")
            elif task == 'safety':
                current_metric = calculate_metric(model, X, task, model_name=model_name)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current Safety Violation Rate: {current_metric:.4f}")
            elif task == 'adv':
                if adv_samples is not None and labels_true is not None and len(adv_samples) > 0:
                    pred_after_batches = []
                    for b in range(int(np.ceil(len(adv_samples)/bs))):
                        batch = adv_samples[b*bs:(b+1)*bs]
                        pred_after_batches.append(K.get_session().run(model.output, {model.input: batch}))
                    pred_after = np.argmax(np.concatenate(pred_after_batches, axis=0), axis=-1)
                    current_metric = float(np.mean(pred_after != labels_true))
                    if successful_attack_mask is not None:
                        if np.sum(successful_attack_mask) == 0:
                            one_minus_pcr = 0.0
                        else:
                            prediction_changed_on_attacked = (pred_after[successful_attack_mask] == pred_before[successful_attack_mask])
                            one_minus_pcr = float(np.mean(prediction_changed_on_attacked))
                    else:
                        one_minus_pcr = None
                else:
                    current_metric = 0.0
                    one_minus_pcr = None
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current ASR: {current_metric:.4f}")
                if one_minus_pcr is not None:
                    print(f"  1-PCR: {one_minus_pcr:.4f}")
                else:
                    print("  1-PCR: N/A (insufficient successful adversarial samples)")
                # 记录 1-PCR 结果（即便为 None 也保留占位，方便对齐）
                one_minus_pcr_results.append(one_minus_pcr)
            elif task == 'fairness':
                # X 是输入特征，G 是群体标签
                current_metric = value_fairness(model, X, Y_true, G)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current Fairness Gap |Acc(1)-Acc(0)|: {current_metric:.4f}")
            metric_results.append(current_metric)
        else:
            print(f"  Skipped non-trainable neuron: {neuron_name}")
        
        k += 1
    
    # 恢复原始权重
    model.set_weights(original_weights)
    
    # 非 adv 任务下，1-PCR 无意义，baseline_one_minus_pcr 置为 None
    if task != 'adv':
        return metric_results, [], None
    # 安全兜底：若 baseline_one_minus_pcr 未成功计算，则记为 0.0
    if baseline_one_minus_pcr is None:
        baseline_one_minus_pcr = 0.0
    return metric_results, one_minus_pcr_results, baseline_one_minus_pcr


def start_memory_monitor(interval=30):
    try:
        import psutil, os
        pid = os.getpid()
        proc = psutil.Process(pid)
        def _mon():
            while True:
                mem = proc.memory_info().rss / 1024.0**2
                print(f"[MEM_MON] pid={pid} RSS={mem:.1f}MB")
                time.sleep(interval)
        t = threading.Thread(target=_mon, daemon=True)
        t.start()
    except Exception as e:
        print("Memory monitor not started:", e)


def _gradact_reduce_channels(activation, gradient):
    """Per-example neuron contribution; conv mean over spatial axes, dense as-is."""
    a = np.asarray(activation, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if a.shape != g.shape or a.ndim < 2:
        raise ValueError('Gradient x Activation: incompatible activation/gradient: {} vs {}'.format(a.shape, g.shape))
    contribution = a * g
    if contribution.ndim > 2:
        contribution = np.mean(contribution, axis=tuple(range(1, contribution.ndim - 1)))
    if contribution.ndim != 2 or not np.all(np.isfinite(contribution)):
        raise ValueError('Invalid contribution matrix shape/values: {}'.format(contribution.shape))
    return contribution


def _gradact_build_graph(model, layer_idx, task, model_name):
    """One forward+backward contribution per sample. Gradients are of scalar sums."""
    layer_output = model.layers[layer_idx].output
    model_output = model.output
    shape = K.int_shape(model_output)
    class_count = int(shape[-1]) if shape and shape[-1] is not None else None
    if class_count is None:
        raise ValueError('Output class dimension must be statically known: {}'.format(model_name))

    labels_tensor = None
    if task == 'backdoor':
        if model_name not in BACKDOOR_TARGETS:
            raise ValueError('Unsupported backdoor dataset {}'.format(model_name))
        target = BACKDOOR_TARGETS[model_name]
        if target >= class_count:
            raise ValueError('Target {} >= output dimension {}'.format(target, class_count))
        # A fault means the triggered input is pushed towards the backdoor target.
        objective_per_sample = model_output[:, target]
    elif task in ('adv', 'fairness'):
        # The TRUE class soft score; for adv, its negative is the failure target.
        labels_tensor = tf.placeholder(tf.float32, shape=(None,), name='gradact_true_y_{}_{}'.format(task, layer_idx))
        if class_count == 1:
            pred = tf.reshape(model_output, (-1,))
            prob_correct = labels_tensor * pred + (1. - labels_tensor) * (1. - pred)
        else:
            one_hot = tf.one_hot(tf.cast(labels_tensor, tf.int32), depth=class_count, dtype=model_output.dtype)
            prob_correct = tf.reduce_sum(model_output * one_hot, axis=-1)
        objective_per_sample = -prob_correct if task == 'adv' else prob_correct
    elif task == 'safety':
        # Existing CFRepair safety metric selects the minimum-output class.
        # Convert -outputs to a soft probability and aggregate UNSAFE classes.
        negative_distance_probs = tf.nn.softmax(-model_output, axis=-1)
        if model_name == 'a19':
            unsafe = [3,4]
        elif model_name == 'a29':
            unsafe = list(range(2, class_count))
        else:
            unsafe = list(range(1, class_count))
        if not unsafe or max(unsafe) >= class_count:
            raise ValueError('Safety output incompatible with unsafe class definition')
        mask = np.zeros((class_count,), dtype=np.float32)
        mask[unsafe] = 1.0
        objective_per_sample = tf.reduce_sum(negative_distance_probs * tf.constant(mask), axis=-1)
    else:
        raise ValueError('Unsupported task: {}'.format(task))

    scalar_objective = tf.reduce_sum(objective_per_sample)
    grads = tf.gradients(scalar_objective, layer_output)[0]
    if grads is None:
        raise RuntimeError('Gradient is None for model layer {}'.format(layer_idx))
    inputs = [model.input]
    if labels_tensor is not None:
        inputs.append(labels_tensor)
    add_learning_phase = False
    try:
        lp = K.learning_phase()
        if not isinstance(lp, (int, np.integer)):
            inputs.append(lp)
            add_learning_phase = True
    except Exception:
        pass
    fn = K.function(inputs, [layer_output, grads])
    return fn, labels_tensor is not None, add_learning_phase


def _gradact_average(model, graph, X, Y=None, batch_size=GRADACT_BATCH_SIZE):
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        raise ValueError('No samples provided to Gradient x Activation')
    fn, needs_y, has_learning_phase = graph
    if needs_y:
        Y = np.asarray(Y).reshape(-1)
        if len(X) != len(Y):
            raise ValueError('Samples/labels length mismatch: {} != {}'.format(len(X), len(Y)))
    score_sum, total = None, 0
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        args = [X[start:stop]]
        if needs_y:
            args.append(Y[start:stop].astype(np.float32))
        if has_learning_phase:
            args.append(0)  # inference mode; no stochastic dropout
        activation, grads = fn(args)
        c = _gradact_reduce_channels(activation, grads)
        batch_sum = np.sum(c, axis=0)
        score_sum = batch_sum if score_sum is None else score_sum + batch_sum
        total += stop - start
    return score_sum / float(total)


def _gradact_model_path(task, model_name, resnet_18_flag=False):
    if resnet_18_flag:
        return os.path.join('..', 'model', '{}_resnet.h5'.format(model_name))
    if task == 'adv':
        return os.path.join('..', 'model', '{}_adv.h5'.format(model_name))
    return os.path.join('..', 'model', '{}.h5'.format(model_name))


def _gradact_select_data(model, task, model_name, requested_n, epsilon, shared_fairness):
    """Same CFRepair load_data interface, with original adversarial selection."""
    group_direction = None
    note = None
    if task == 'backdoor':
        clean, labels, fault = load_data(model_name, task='backdoor', k_sample=requested_n)
        clean, fault = np.asarray(clean), np.asarray(fault)
        labels = np.asarray(labels).reshape(-1)
        if len(clean) != len(fault) or len(clean) != len(labels):
            raise RuntimeError('Backdoor clean/fault/label size mismatch')
        clean_labels, fault_labels = None, None
        note = 'original CFRepair backdoor clean vs triggered samples; no extra label filtering'
    elif task == 'adv':
        fault, clean, labels = load_data(model_name, task='adv', epsilon=epsilon, k_sample=requested_n)
        clean, fault = np.asarray(clean), np.asarray(fault)
        labels = np.asarray(labels).reshape(-1)
        if len(clean) != len(fault) or len(clean) != len(labels):
            raise RuntimeError('Adversarial clean/fault/label size mismatch')
        n = len(labels)
        pred_fault = np.argmax(sess_run(model, model.output, fault), axis=-1)
        success_mask = pred_fault != labels
        n_success = int(np.sum(success_mask))
        print('[CGARepair:adv] original CFRepair successful adversarial samples: {}/{}'.format(n_success, n))
        if n_success:
            # This mirrors the original CFRepair's successful-attack filtering
            # and seeded random clean controls (not pairwise clean matching).
            fault = fault[success_mask]
            fault_labels = labels[success_mask]
            control_n = min(len(clean), len(fault))
            if len(fault) > control_n:
                chosen = np.random.choice(len(fault), size=control_n, replace=False)
                fault = fault[chosen]
                fault_labels = fault_labels[chosen]
            control_indices = np.random.choice(len(clean), size=control_n, replace=False)
            clean_labels = labels[control_indices]
            clean = clean[control_indices]
            note = 'original CFRepair adv-success filter and random clean matching'
        else:
            clean_labels = labels
            fault_labels = labels
            note = 'all adversarial and clean inputs because no successful attack was found'
    elif task == 'safety':
        # Keep the original CFRepair order: ct_x first, dr_x second.
        ct_x, dr_x = load_data(model_name, task='safety', k_sample=requested_n)
        fault, clean = np.asarray(ct_x), np.asarray(dr_x)
        # Match the original safety validation's handling of 1-D inputs.
        if fault.ndim == 1:
            fault = fault.reshape(-1, 1)
        if clean.ndim == 1:
            clean = clean.reshape(-1, 1)
        clean_labels, fault_labels = None, None
        note = 'counterexample ct_x versus drawdown dr_x from original load_data'
    elif task == 'fairness':
        if shared_fairness is None:
            shared_fairness = load_fairness_shared_bundle(model_name, requested_n)
        X, Y, G = _require_fairness_bundle(shared_fairness)
        baseline = _fairness_accuracy_stats(model, X, Y, G)
        group_direction = float(baseline['direction'])
        clean, fault = X[G == 0], X[G == 1]
        clean_labels, fault_labels = Y[G == 0], Y[G == 1]
        note = 'original CFRepair same X/Y/G fairness bundle and signed group gap'
    else:
        raise ValueError('Unknown task: {}'.format(task))
    if len(clean) == 0 or len(fault) == 0:
        raise RuntimeError('Empty clean/fault pool for {}'.format(task))
    return clean, clean_labels, fault, fault_labels, group_direction, note


def compute_gradient_activation_values(task, model_name, num_examples=1024,
                                       top_k=None, resnet_18_flag=False,
                                       epsilon=None, fairness_data=None):
    """Only replaces CFRepair's neuron ranking calculation; same players/scores API."""
    start = time.time()
    setup_tf()
    model_path = _gradact_model_path(task, model_name, resnet_18_flag)
    model = load_model(model_path, compile=False)
    model_tag = '{}_resnet'.format(model_name) if resnet_18_flag else model_name
    result_dir = os.path.join('..', 'rq1', 'LocalizationTiming', 'cache', os.environ['CGAT_METHOD'], os.environ['CGAT_DATASET'], os.environ['CGAT_TRIAL'],  model_tag)
    os.makedirs(result_dir, exist_ok=True)
    extra = '_eps{}'.format(epsilon) if task == 'adv' else ''
    cache_file = os.path.join(result_dir,
        'CGARepair_{}_samples{}_seed{}{}_v3.json'.format(task, num_examples, seed, extra))
    fprint = _fairness_bundle_fingerprint(fairness_data) if task == 'fairness' else None
    model_stat = os.stat(model_path)
    signature = [int(model_stat.st_size), int(model_stat.st_mtime_ns)]
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        if (cache.get('method') == GRADACT_VERSION and
            cache.get('model_signature') == signature and
            (task != 'fairness' or cache.get('fairness_sample_fingerprint') == fprint)):
            print('[CGARepair] Reusing GradAct ranking cache {}'.format(cache_file))
            return cache['players'], cache['scores']
    clean, clean_labels, fault, fault_labels, direction, note = _gradact_select_data(
        model, task, model_name, num_examples, epsilon, fairness_data)
    print('[CGARepair] {} samples: clean={}, faulty={}'.format(task, len(clean), len(fault)))
    players = []
    scores = []
    diagnostics = []
    for layer_idx, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if 'Conv' not in layer_type and 'Dense' not in layer_type:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        if kernel.ndim not in (2, 3, 4, 5):
            continue
        graph = _gradact_build_graph(model, layer_idx, task, model_name)
        clean_mean = _gradact_average(model, graph, clean, clean_labels)
        fault_mean = _gradact_average(model, graph, fault, fault_labels)
        if clean_mean.shape != fault_mean.shape or len(clean_mean) != kernel.shape[-1]:
            raise RuntimeError('GradAct layer {} score shape mismatch: {} vs {} vs {}'.format(
                layer_idx, clean_mean.shape, fault_mean.shape, kernel.shape))
        difference = fault_mean - clean_mean
        layer_scores = (np.maximum(direction * difference, 0.0)
                        if task == 'fairness' else np.abs(difference))
        if not np.all(np.isfinite(layer_scores)):
            raise RuntimeError('Non-finite GradAct scores in layer {}'.format(layer_idx))
        for idx, val in enumerate(layer_scores):
            players.append('{}_{}_{}'.format(layer_type, layer_idx, idx))
            scores.append(float(val))
        diagnostics.append({'layer_index': int(layer_idx), 'type': layer_type,
                            'neurons': int(len(layer_scores)),
                            'score_max': float(np.max(layer_scores))})
        print('[CGARepair] layer {} {} channels={} max_score={:.6g}'.format(
            layer_idx, layer_type, len(layer_scores), float(np.max(layer_scores))), flush=True)
    if not players or len(set(players)) != len(players):
        raise RuntimeError('Missing or duplicated neuron identifiers')
    if np.max(scores) == 0.0:
        print('[CGARepair][WARNING] All attribution scores are zero; review task objective and model outputs.')
    bundle = {
        'task': task,
        'model': model_name,
        'method': GRADACT_VERSION,
        'num_examples': int(num_examples),
        'seed': int(seed),
        'players': players,
        'scores': scores,
        'data_selection': note,
        'num_clean_used': int(len(clean)),
        'num_fault_used': int(len(fault)),
        'model_signature': signature,
        'fairness_sample_fingerprint': fprint,
        'direction': direction,
        'epsilon': epsilon,
        'layer_diagnostics': diagnostics,
        'runtime_sec': float(time.time()-start),
    }
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    print('[CGARepair] Saved ranking cache: {}'.format(cache_file))
    return players, scores


def main():
    # The command-line seed is not a positional Top-K value.
    global seed
    import argparse
    opt = argparse.ArgumentParser(add_help=False)
    opt.add_argument('--seed', type=int, default=256)
    opt.add_argument('--check-args', action='store_true')
    options, positional = opt.parse_known_args(sys.argv[1:])
    if len(positional) < 3:
        raise SystemExit('Usage: loc_CGARepair_test.py TASK MODEL SAMPLES [EPSILON] [RESNET] [TOP_K] --seed SEED')
    task, model_name = positional[:2]
    if task not in ('backdoor', 'adv', 'safety', 'fairness'):
        raise SystemExit('Unknown task: {}'.format(task))
    num_examples = int(positional[2])
    if num_examples <= 0:
        raise SystemExit('num_examples must be positive')
    extras = positional[3:]
    epsilon = None
    if task == 'adv':
        if not extras:
            raise SystemExit('adv requires an epsilon after the sample count')
        epsilon = extras.pop(0)
    resnet_18_flag = False
    if extras and extras[0].lower() in ('true','false','1','0','yes','no','y','n','t','f'):
        resnet_18_flag = extras.pop(0).lower() in ('true','1','yes','y','t')
    top_k = int(extras.pop(0)) if extras else None
    if extras:
        raise SystemExit('Unexpected extra arguments: {}'.format(extras))
    if top_k is not None and top_k <= 0:
        raise SystemExit('TOP_K must be positive')
    seed = int(options.seed)
    if not 0 <= seed < 2**32:
        raise SystemExit('--seed must be in [0, 4294967295]')
    print('[CGARepair] task={} model={} samples={} epsilon={} resnet={} top_k={} seed={}'.format(
        task, model_name, num_examples, epsilon, resnet_18_flag,
        'auto' if top_k is None else top_k, seed), flush=True)
    if options.check_args:
        return
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    import time
    start_time = time.time()

    # Fairness data are loaded exactly once. The same in-memory X/Y/G arrays are
    # used for localization, independent Step-2 scoring, and cumulative Top-K validation.
    fairness_shared_data = None
    actual_sample_count = int(num_examples)
    if task == 'fairness':
        fairness_shared_data = load_fairness_shared_bundle(model_name, num_examples)
        actual_sample_count = int(len(fairness_shared_data[0]))

    # Only the localization scorer changes; the CFRepair validation below is retained.
    players, scores = compute_gradient_activation_values(
        task, model_name, num_examples, top_k, resnet_18_flag, epsilon,
        fairness_data=fairness_shared_data,
    )
    # 排序并显示结果
    items = list(zip(players, scores))
    items.sort(key=lambda x: x[1], reverse=True)
    ranked_players = [p for p, _ in items]
    if not ranked_players:
        raise RuntimeError('No neurons found for localization')
    actual_top_k = (max(4, min(11, int(len(ranked_players) * 0.01)))
                    if top_k is None else int(top_k))
    actual_top_k = min(len(ranked_players), actual_top_k)
    if actual_top_k < 1:
        raise ValueError('Invalid top_k')
    print('[CGARepair] Total neurons={}  selected auto Top-k={}'.format(
        len(ranked_players), actual_top_k))
    
    print(f"\nTop 10 most important neurons:")
    for i in range(min(10, len(items))):
        print(f"  {i+1}. {items[i][0]}: {items[i][1]:.6f}")
    
    # 验证topk效果（使用自适应top_k）
    metric_results = None
    print(f"\n=== Validating Top-{actual_top_k} Neurons ===")
    if resnet_18_flag:
        model_path = os.path.join('..', 'model', f'{model_name}_resnet.h5')
    else:
        if task == 'adv':
            model_path = os.path.join('..', 'model', f'{model_name}_adv.h5')
        else:
            model_path = os.path.join('..', 'model', f'{model_name}.h5')
    model = load_model(model_path, compile=False)
    baseline_metric_before_disable = None
    
    if task == 'backdoor':
        X, Y, X_perturbed = load_data(model_name, task='backdoor', k_sample=num_examples)
        metric_results, one_minus_pcr_results, baseline_one_minus_pcr = validate_topk_neurons(
            model, X, X_perturbed, ranked_players, model_name, top_k=actual_top_k, task=task
        )
    elif task == 'safety':
        ct_x, dr_x = load_data(model_name, task='safety', k_sample=num_examples)
        # 对于safety任务，使用counterexample数据
        X = ct_x
        metric_results, one_minus_pcr_results, baseline_one_minus_pcr = validate_topk_neurons(
            model, X, None, ranked_players, model_name, top_k=actual_top_k, task=task
        )
    elif task == 'adv':
        X_adv, X_clean, Y_clean = load_data(model_name, task='adv', epsilon=epsilon, k_sample=num_examples)
        # 这里传参方式与上方 validate_topk_neurons 的 adv 分支一致：X=Y_clean, X_perturbed=X_adv
        metric_results, one_minus_pcr_results, baseline_one_minus_pcr = validate_topk_neurons(
            model, Y_clean, X_adv, ranked_players, model_name, top_k=actual_top_k, task=task
        )
    elif task == 'fairness':
        baseline_metric_before_disable = value_fairness(
            model, fairness_shared_data[0], fairness_shared_data[1], fairness_shared_data[2]
        )
        # Reuse exactly the same X/Y/G arrays used to compute the ranking.
        metric_results, one_minus_pcr_results, baseline_one_minus_pcr = validate_topk_neurons(
            model, fairness_shared_data, None, ranked_players, model_name,
            top_k=actual_top_k, task=task
        )
    # 保存最终结果
    if resnet_18_flag:
        out_dir = os.path.join('result', task, f'{model_name}_resnet')
    else:
        out_dir = os.path.join('result', task, model_name)
    # adv 输出目录包含 epsilon
    if task == 'adv' and epsilon is not None:
        out_dir = os.path.join(out_dir, f'epsilon_{epsilon}')
    os.makedirs(out_dir, exist_ok=True)
    # actual_top_k is computed once immediately after ranking.
    
    result_sample_count = actual_sample_count if task == 'fairness' else int(num_examples)
    result_path = os.path.join(
        out_dir,
        f'top{actual_top_k}_samples{result_sample_count}_CGARepair_seed{seed}.txt'
    )
    # Separate text ranking is necessary because original CFRepair overwrote
    # its ranking .txt with JSON metadata. Keep the .txt JSON convention intact.
    ranking_path = result_path[:-4] + '_ranked_players.txt'
    with open(ranking_path, 'w', encoding='utf-8') as f:
        f.write(','.join(ranked_players))
    selected_path = result_path[:-4] + '_selected_topk.txt'
    with open(selected_path, 'w', encoding='utf-8') as f:
        f.write(','.join(ranked_players[:actual_top_k]))
    scores_path = result_path[:-4] + '_scores.csv'
    with open(scores_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'neuron', 'score', 'in_topk'])
        for rank, (player, score) in enumerate(items, start=1):
            writer.writerow([rank, player, '{:.16g}'.format(score), int(rank <= actual_top_k)])
    meta = {
        'task': task,
        'seed': int(seed),
        'model': model_name,
        'num_examples': result_sample_count,
        'requested_num_examples': int(num_examples),
        'actual_sample_count': int(result_sample_count),
        'localization_validation_same_samples': bool(task == 'fairness'),
        'fairness_sample_fingerprint': (
            _fairness_bundle_fingerprint(fairness_shared_data)
            if task == 'fairness' else None
        ),
        'supported_fairness_datasets': list(SUPPORTED_FAIRNESS_DATASETS),
        'baseline_metric_before_disable': baseline_metric_before_disable,
        'top_k': actual_top_k,
        'num_players': len(ranked_players),
        'method': 'CGARepair',
        'localization_method': GRADACT_VERSION,
        'alpha': None,
        'steps': 1,
        'ranked_players_path': ranking_path,
        'selected_topk_path': selected_path,
        'scores_csv_path': scores_path,
        'time': int(time.time() - start_time),
        f'top{actual_top_k}_metric_results': metric_results
    }
    # 对抗攻击任务：额外记录 1-PCR 序列与基线 1-PCR
    if task == 'adv':
        if one_minus_pcr_results:
            meta[f'top{actual_top_k}_one_minus_pcr_results'] = one_minus_pcr_results
        if baseline_one_minus_pcr is not None:
            meta['baseline_one_minus_pcr'] = float(baseline_one_minus_pcr)
        if epsilon is not None:
            meta['epsilon'] = epsilon
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved CFRepair-format cumulative-ablation JSON to {result_path}')
    print(f'Saved full ranking to {ranking_path}')
    print(f'Saved selected top-k to {selected_path}')


if __name__ == '__main__':
    main()
