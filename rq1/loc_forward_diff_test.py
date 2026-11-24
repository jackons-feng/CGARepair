from curses import start_color
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K

# Enable TensorFlow v1 compatibility mode
tf.compat.v1.disable_eager_execution()
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

num_examples = 512
# 导入统一的数据加载模块
from load_data import load_data

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


def evaluate_safety(model, images, model_name, metric='safety', batch_size=256):
    """Evaluate safety violation rate or accuracy for safety tasks"""
    output = []
    num_batches = int(np.ceil(len(images) / batch_size))
    for batch in range(num_batches):
        batch_idxs = np.arange(batch * batch_size, min((batch + 1) * batch_size, len(images)))
        input_dic = {model.input: images[batch_idxs]}
        output.append(K.get_session().run(model.output, input_dic))
    try:
        output = np.concatenate(output, 0)
    except:
        output = np.array(output)
    
    pred = np.argmin(output, axis=-1)
    if metric == 'safety':
        if model_name == 'a19':
            val = (np.sum(pred == 3) + np.sum(pred == 4)) / len(output)
        elif model_name == 'a29':
            val = (len(images) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output)
        else:
            val = (np.sum(pred != 0)) / len(output)
    elif metric == 'drawndown':
        if model_name == 'a19':
            val = (len(output) - np.sum(pred == 3) - np.sum(pred == 4)) / len(output)
        elif model_name == 'a29':
            val = (np.sum(pred == 0) + np.sum(pred == 1)) / len(output)
        else:
            val = np.sum(pred == 0) / len(output)
    return val


"""
def evaluate_fairness(model, images, labels, batch_size=256):
    #Evaluate group accuracy gap |Acc(group=1) - Acc(group=0)| for fairness tasks.
    #labels: binary group label array where 0=female, 1=male (per data_pipeline meta)
    
    if images is None or labels is None or len(images) == 0:
        return 0.0
    output = []
    num_batches = int(np.ceil(len(images) / batch_size))
    for batch in range(num_batches):
        batch_idxs = np.arange(batch * batch_size, min((batch + 1) * batch_size, len(images)))
        input_dic = {model.input: images[batch_idxs]}
        output.append(K.get_session().run(model.output, input_dic))
    try:
        output = np.concatenate(output, 0)
    except Exception:
        output = np.array(output)
    pred = np.argmax(output, axis=-1)
    lbl = labels.astype(np.int64)
    mask0 = (lbl == 0)
    mask1 = (lbl == 1)
    acc0 = np.sum(pred[mask0] == lbl[mask0]) / np.maximum(1, np.sum(mask0))
    acc1 = np.sum(pred[mask1] == lbl[mask1]) / np.maximum(1, np.sum(mask1))
    return float(abs(acc1 - acc0))
"""

def evaluate_fairness(model, X, Y_true, G, batch_size=256):
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
    if G.max() > 1:
        G = G.copy()
        G[G == 1] = 0
        G[G == 2] = 1
    
    output = []
    num_batches = int(np.ceil(len(X) / batch_size))
    for batch in range(num_batches):
        batch_idxs = np.arange(batch * batch_size, min((batch + 1) * batch_size, len(X)))
        input_dic = {model.input: X[batch_idxs]}
        output.append(K.get_session().run(model.output, input_dic))

    output = np.concatenate(output, 0).flatten()
    pred = (output > 0.5).astype(int)

    # 分组准确率
    mask0 = (G == 0)
    mask1 = (G == 1)
    acc0 = np.mean(pred[mask0] == Y_true[mask0])
    acc1 = np.mean(pred[mask1] == Y_true[mask1])

    return float(abs(acc1 - acc0))


def forward_diff_scores(model, X, Y, X_perturbed, sample_n=512, task='backdoor',model_name=None, epsilon=None):
    """
    Forward difference: calculate neuron importance based on activation patterns
    between positive examples and negative examples
    """
    rng = np.random.default_rng(123)
    idx = rng.choice(len(X), min(sample_n, len(X)), replace=False)
    Xs = X[idx]

    conv_idx, dense_idx,_ = get_layer_indices(model)
    players = []
    scores = []

    if task == 'backdoor':
        Ys, Xs_perturbed = Y[idx], X_perturbed[idx]
        # ① 构建正例集和反例集
        # 正例集: (Xs, Ys) - 干净数据
        # 反例集: (Xs_perturbed, target_label) - 后门数据
        dic = {'gtsrb': [(32, 32, 3), (43, 33)], 'mnist': [(28, 28, 1), (10, 3)], 'fashion': [(28, 28, 1), (10, 3)]}
        target_label = dic[model_name][1][1]
        positive_data = Xs
        negative_data = Xs_perturbed
    elif task == 'adv':
        # 对于对抗样本任务，使用干净数据作为正例，对抗样本作为反例
        Ys = Y[idx]
        Xs_adv = X_perturbed[idx]  # X_perturbed 在这里是 X_adv
        positive_data = Xs  # 干净数据
        negative_data = Xs_adv  # 对抗样本
        print(f"Adv task - Positive samples (clean): {Xs.shape}")
        print(f"Adv task - Negative samples (adv): {Xs_adv.shape}")
    elif task == 'safety':
        # 对于安全任务，使用drawndown数据作为正例，counterexample数据作为反例
        # 注意：这里需要修改调用方式，传入counterexample数据
        # 暂时使用相同数据，实际调用时需要传入不同的数据集
        positive_data = Xs  # drawndown data (safe)
        negative_data = Xs  # counterexample data (unsafe) - 需要从外部传入
        print(f"Warning: Using same data for positive and negative examples in safety task")
        print(f"Input shape: {Xs.shape}, Data type: {task}")
    elif task == 'fairness':
        # For fairness, construct groups by sensitive attribute labels Y: 0 vs 1
        # Positive: group 0 (e.g., female), Negative: group 1 (e.g., male)
        if Y is None:
            raise ValueError("Fairness task requires Y (group labels) to be provided.")
        Gs = Y[idx]
        grp0 = Xs[Gs == 0]
        grp1 = Xs[Gs == 1]
        if len(grp0) == 0 or len(grp1) == 0:
            raise ValueError("Fairness task requires both groups (labels 0 and 1) to have samples after sampling.")
        # Balance groups for fair comparison
        n = int(min(len(grp0), len(grp1)))
        if n == 0:
            raise ValueError("Insufficient samples to balance groups for fairness forward diff.")
        idx0 = np.random.default_rng(123).choice(len(grp0), n, replace=False)
        idx1 = np.random.default_rng(321).choice(len(grp1), n, replace=False)
        positive_data = grp0[idx0]
        negative_data = grp1[idx1]
    else:
        raise ValueError(f"Unknown task: {task}")
    
    def get_layer_activations(model, X_data, layer_indices):
        """获取指定层的激活值"""
        activations = {}
        bs = 256
        
        for layer_idx in layer_indices:
            layer_outputs = []
            for b in range(int(np.ceil(len(X_data)/bs))):
                batch = X_data[b*bs:(b+1)*bs]
                # 获取中间层输出
                intermediate_layer_model = tf.keras.Model(
                    inputs=model.input, 
                    outputs=model.layers[layer_idx].output
                )
                layer_out = K.get_session().run(intermediate_layer_model.output, {model.input: batch})
                layer_outputs.append(layer_out)
            activations[layer_idx] = np.concatenate(layer_outputs, axis=0)
        return activations
    
    def calculate_importance_scores(activations):
        """
        ② 计算前向传播重要性：神经元输出（绝对值）占该层所有神经元输出（绝对值）之和的比值
        """
        importance_scores = {}
        
        for layer_idx, activation in activations.items():
            # ④ 对多维输出取绝对值并逐元素相加得到一维值
            if len(activation.shape) > 2:  # 卷积层
                # 对所有通道取绝对值并逐元素相加
                abs_activation = np.abs(activation)
                # 在空间维度上求和，保留通道维度
                channel_sums = np.sum(abs_activation, axis=tuple(range(1, len(abs_activation.shape)-1)))
            else:  # 全连接层
                channel_sums = np.abs(activation)
            
            # 计算每个神经元的重要性：该神经元输出 / 该层总输出
            total_layer_output = np.sum(channel_sums, axis=1, keepdims=True)
            # 避免除零
            total_layer_output = np.where(total_layer_output == 0, 1, total_layer_output)
            importance = channel_sums / total_layer_output
            
            # ③ 对正例集计算得出的前向数据平均
            avg_importance = np.mean(importance, axis=0)
            importance_scores[layer_idx] = avg_importance
            
        return importance_scores
    
    # 获取所有相关层（只包含卷积层和全连接层），与 loc_forward_diff.py 保持一致
    all_layer_indices = []
    layer_names = []
    layer_name_to_index = {}

    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if ('Conv' in layer_type or 'Dense' in layer_type) and hasattr(layer, 'output_shape') and layer.output_shape is not None:
            all_layer_indices.append(i)
            lname = f"{layer_type}_{i}"
            layer_names.append(lname)
            layer_name_to_index[lname] = i

    print(f"Processing {len(all_layer_indices)} trainable layers...")
    print(f"Layer mapping: {layer_name_to_index}")
    
    # 计算正例集的重要性
    print(f"Calculating importance for positive examples ({task})...")
    positive_activations = get_layer_activations(model, positive_data, all_layer_indices)
    positive_importance = calculate_importance_scores(positive_activations)
    
    # 计算反例集的重要性
    print(f"Calculating importance for negative examples ({task})...")
    negative_activations = get_layer_activations(model, negative_data, all_layer_indices)
    negative_importance = calculate_importance_scores(negative_activations)
    
    # 计算重要性差异并排序
    print("Computing importance differences...")
    for i, layer_idx in enumerate(all_layer_indices):
        layer_name = layer_names[i]
        
        if layer_idx in positive_importance and layer_idx in negative_importance:
            pos_imp = positive_importance[layer_idx]
            neg_imp = negative_importance[layer_idx]
            
            # 计算差异：反例集重要性 - 正例集重要性
            # 正值表示该神经元在反例数据中更重要
            importance_diff = neg_imp - pos_imp
            
            # 为每个神经元/通道添加玩家和分数
            # 安全上限：受模型真实通道/单元数约束，避免产生越界玩家
            layer = model.layers[layer_idx]
            upper = len(importance_diff)
            try:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    kshape = layer.get_weights()[0].shape if layer.get_weights() else None
                    if kshape is not None:
                        upper = min(upper, int(kshape[-1]))
                    if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                        upper = min(upper, int(layer.output_shape[-1]))
                elif isinstance(layer, tf.keras.layers.Dense):
                    if hasattr(layer, 'units') and layer.units is not None:
                        upper = min(upper, int(layer.units))
                    kshape = layer.get_weights()[0].shape if layer.get_weights() else None
                    if kshape is not None:
                        upper = min(upper, int(kshape[-1]))
            except Exception:
                pass

            for neuron_idx in range(upper):
                diff_score = float(importance_diff[neuron_idx])
                players.append(f'{layer_name}_{neuron_idx}')
                scores.append(diff_score)
    
    return players, scores


def forward_diff_scores_safety(model, positive_data, negative_data, sample_n=512):
    """
    专门用于安全任务的前向传播差异计算：
    正例集: drawndown数据（安全）
    反例集: counterexample数据（不安全）
    """
    rng = np.random.default_rng(123)
    
    # 分别从正例和反例中采样
    pos_idx = rng.choice(len(positive_data), min(sample_n, len(positive_data)), replace=False)
    neg_idx = rng.choice(len(negative_data), min(sample_n, len(negative_data)), replace=False)
    
    pos_samples = positive_data[pos_idx]  # drawndown data (safe)
    neg_samples = negative_data[neg_idx]  # counterexample data (unsafe)
    
    print(f"Safety task - Positive samples shape: {pos_samples.shape}")
    print(f"Safety task - Negative samples shape: {neg_samples.shape}")
    
    conv_idx, dense_idx,_ = get_layer_indices(model)
    players = []
    scores = []
    
    def get_layer_activations(model, X_data, layer_indices):
        """获取指定层的激活值"""
        activations = {}
        bs = 256
        
        for layer_idx in layer_indices:
            layer_outputs = []
            for b in range(int(np.ceil(len(X_data)/bs))):
                batch = X_data[b*bs:(b+1)*bs]
                # 获取中间层输出
                intermediate_layer_model = tf.keras.Model(
                    inputs=model.input, 
                    outputs=model.layers[layer_idx].output
                )
                layer_out = K.get_session().run(intermediate_layer_model.output, {model.input: batch})
                layer_outputs.append(layer_out)
            activations[layer_idx] = np.concatenate(layer_outputs, axis=0)
        return activations
    
    def calculate_importance_scores(activations):
        """计算前向传播重要性"""
        importance_scores = {}
        
        for layer_idx, activation in activations.items():
            # 对多维输出取绝对值并逐元素相加得到一维值
            if len(activation.shape) > 2:  # 卷积层
                abs_activation = np.abs(activation)
                channel_sums = np.sum(abs_activation, axis=tuple(range(1, len(abs_activation.shape)-1)))
            else:  # 全连接层
                channel_sums = np.abs(activation)
            
            # 计算每个神经元的重要性
            total_layer_output = np.sum(channel_sums, axis=1, keepdims=True)
            total_layer_output = np.where(total_layer_output == 0, 1, total_layer_output)
            importance = channel_sums / total_layer_output
            
            # 对正例集计算得出的前向数据平均
            avg_importance = np.mean(importance, axis=0)
            importance_scores[layer_idx] = avg_importance
            
        return importance_scores
    
    # 获取所有相关层的索引（只包含卷积层和全连接层）
    all_layer_indices = []
    layer_names = []
    layer_name_to_index = {}
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if ('Conv' in layer_type or 'Dense' in layer_type) and hasattr(layer, 'output_shape') and layer.output_shape is not None:
            all_layer_indices.append(i)
            layer_name = f"{layer_type}_{i}"
            layer_names.append(layer_name)
            layer_name_to_index[layer_name] = i
    
    print(f"Processing {len(all_layer_indices)} trainable layers for safety task...")
    
    # 计算正例集（安全数据）的重要性
    print("Calculating importance for safe data (positive examples)...")
    pos_activations = get_layer_activations(model, pos_samples, all_layer_indices)
    pos_importance = calculate_importance_scores(pos_activations)
    
    # 计算反例集（不安全数据）的重要性
    print("Calculating importance for unsafe data (negative examples)...")
    neg_activations = get_layer_activations(model, neg_samples, all_layer_indices)
    neg_importance = calculate_importance_scores(neg_activations)
    
    # 计算重要性差异并排序
    print("Computing importance differences for safety task...")
    for i, layer_idx in enumerate(all_layer_indices):
        layer_name = layer_names[i]
        
        if layer_idx in pos_importance and layer_idx in neg_importance:
            pos_imp = pos_importance[layer_idx]
            neg_imp = neg_importance[layer_idx]
            
            # 计算差异：反例集重要性 - 正例集重要性
            # 正值表示该神经元在不安全数据中更重要
            importance_diff = neg_imp - pos_imp
            
            # 为每个神经元/通道添加玩家和分数
            for neuron_idx, diff_score in enumerate(importance_diff):
                players.append(f'{layer_name}_{neuron_idx}')
                scores.append(float(diff_score))
    
    return players, scores


def validate_top5_neurons(model, X, X_perturbed, ranked_players, top_k=None, task='backdoor',model_name=None):
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
            dic = {'gtsrb': [(32, 32, 3), (43, 33)], 'mnist': [(28, 28, 1), (10, 3)], 'fashion': [(28, 28, 1), (10, 3)]}
            target_label = dic[model_name][1][1]
            out_perturbed = []
            bs = 256
            for b in range(int(np.ceil(len(X_data)/bs))):
                batch = X_data[b*bs:(b+1)*bs]
                out_perturbed.append(K.get_session().run(model.output, {model.input: batch}))
            pred_perturbed = np.argmax(np.concatenate(out_perturbed, axis=0), axis=-1)
            asr = np.sum(pred_perturbed == target_label) / len(pred_perturbed)
            return asr
        elif task == 'adv':
            # 对于对抗样本，计算预测改变率：删除神经元前后预测发生改变的比例
            # 这里需要比较删除前后的预测结果
            bs = 256
            pred_before = []
            for b in range(int(np.ceil(len(X_data)/bs))):
                batch = X_data[b*bs:(b+1)*bs]
                pred_before.append(K.get_session().run(model.output, {model.input: batch}))
            pred_before = np.argmax(np.concatenate(pred_before, axis=0), axis=-1)
            
            # 这里应该与删除前的预测比较，但为了简化，我们返回当前预测的分布
            # 实际使用时需要在外部保存删除前的预测结果
            return len(np.unique(pred_before)) / len(pred_before)  # 预测多样性作为指标
        elif task == 'safety':
            return evaluate_safety(model, X_data, model_name, metric='safety')
        elif task == 'fairness':
            # Here, X_data is images and X_perturbed is used to pass group labels
            X, Y_true, G = X_data
            return evaluate_fairness(model, X, Y_true, G)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def disable_neuron_by_name(model, neuron_name):
        """根据神经元名称禁用对应的权重"""
        # 支持两种命名：
        # 1) "LayerName::idx" 按实际 layer.name 匹配
        # 2) "Dense_8_3" / "Conv2D_5_12" 其中中间的数字为全局层索引，最后为神经元索引
        layer = None
        neuron_idx = None
        if '::' in neuron_name:
            layer_name, neuron_idx_str = neuron_name.split('::', 1)
            neuron_idx = int(neuron_idx_str)
            matched_layers = [i for i, l in enumerate(model.layers) if l.name == layer_name]
            if not matched_layers:
                print(f"Warning: Layer name {layer_name} not found in model")
                return False
            layer = model.layers[matched_layers[0]]
        else:
            # 形如 "Dense_8_3" -> 取最后一个 '_' 之后为神经元索引，倒数第二段为层索引
            if '_' not in neuron_name:
                print(f"Warning: Cannot parse neuron name {neuron_name}")
                return False
            try:
                base, neuron_idx_str = neuron_name.rsplit('_', 1)
                layer_idx_str = base.split('_')[-1]
                layer_idx = int(layer_idx_str)
                neuron_idx = int(neuron_idx_str)
            except Exception:
                print(f"Warning: Cannot parse neuron name {neuron_name}")
                return False
            if layer_idx < 0 or layer_idx >= len(model.layers):
                print(f"Warning: Layer index {layer_idx} out of range (total {len(model.layers)})")
                return False
            layer = model.layers[layer_idx]
        
        # 计算该层的输出通道/单元上限
        try:
            max_units = int(layer.output_shape[-1]) if hasattr(layer, 'output_shape') and layer.output_shape is not None else None
        except Exception:
            max_units = None

        # 获取层的权重
        weights = layer.get_weights()
        if len(weights) == 0:
            print(f"Warning: Layer {layer_idx} ({type(layer).__name__}) has no weights")
            return False

        # 禁用神经元
        if isinstance(layer, tf.keras.layers.Conv2D):
            # 卷积层：禁用对应的输出通道
            if len(weights) > 0:  # kernel weights
                kernel = weights[0]
                upper = kernel.shape[-1]
                if max_units is not None:
                    upper = min(upper, int(max_units))
                if neuron_idx < upper:  # 检查神经元索引是否有效
                    kernel[:, :, :, neuron_idx] = 0.0
                    weights[0] = kernel
                    layer.set_weights(weights)
                    return True
                else:
                    print(f"Warning: Neuron index {neuron_idx} out of range for layer {getattr(layer, 'name', 'unknown')} (max: {upper-1})")
                    return False
        elif isinstance(layer, tf.keras.layers.Dense):
            # 全连接层：禁用对应的输出神经元
            if len(weights) > 0:  # kernel weights
                kernel = weights[0]
                upper = kernel.shape[-1]
                if hasattr(layer, 'units') and layer.units is not None:
                    upper = min(upper, int(layer.units))
                if neuron_idx < upper:  # 检查神经元索引是否有效
                    kernel[:, neuron_idx] = 0.0
                    weights[0] = kernel
                    layer.set_weights(weights)
                    return True
                else:
                    print(f"Warning: Neuron index {neuron_idx} out of range for layer {getattr(layer, 'name', 'unknown')} (max: {upper-1})")
                    return False
        
        print(f"Warning: Could not disable neuron {neuron_name}")
        return False
    
    # 对于 adv 任务，需要先计算删除前的预测结果
    if task == 'adv':
        # 计算删除前的对抗样本预测
        bs = 256
        pred_before = []
        for b in range(int(np.ceil(len(X_perturbed)/bs))):
            batch = X_perturbed[b*bs:(b+1)*bs]
            pred_before.append(K.get_session().run(model.output, {model.input: batch}))
        pred_before = np.argmax(np.concatenate(pred_before, axis=0), axis=-1)
        
        # 计算初始攻击成功率（对抗样本被误分类的比例）
        # 注意：这里 X 参数实际上是 Y_clean（真实标签）
        initial_asr = np.sum(pred_before != X) / len(pred_before)
        print(f"Initial ASR (adv_pred != y_clean): {initial_asr:.4f}")
        
        baseline_metric = initial_asr
    elif task == 'backdoor':
        baseline_metric = calculate_metric(model, X_perturbed, task,model_name=model_name)
        print(f"Baseline ASR: {baseline_metric:.4f}")
    elif task == 'safety':
        baseline_metric = calculate_metric(model=model, X_data=X, task=task, model_name=model_name)
        print(f"Baseline Safety Violation Rate: {baseline_metric:.4f}")
    elif task == 'fairness':
        # X 是输入特征，Y_true 是真实标签
        X, Y_true, G = load_data(model_name, task='fairness', k_sample=num_examples)
        baseline_metric = evaluate_fairness(model, X, Y_true, G)
        print(f"Baseline Fairness Gap |Acc(1)-Acc(0)|: {baseline_metric:.4f}")
    
    # 保存原始权重
    original_weights = [np.copy(w) for w in model.get_weights()]
    
    # 逐步禁用top-k神经元（只处理可训练层的神经元）
    metric_results = []
    disabled_count = 0
    k = 0
    
    while disabled_count < top_k and k < len(ranked_players):
        neuron_name = ranked_players[k]
        print(f"Trying to disable neuron {k+1}: {neuron_name}")
        
        # 禁用神经元
        success = disable_neuron_by_name(model, neuron_name)
        if success:
            disabled_count += 1
            # 计算当前指标
            if task == 'adv':
                # 计算删除后的对抗样本预测
                pred_after = []
                for b in range(int(np.ceil(len(X_perturbed)/bs))):
                    batch = X_perturbed[b*bs:(b+1)*bs]
                    pred_after.append(K.get_session().run(model.output, {model.input: batch}))
                pred_after = np.argmax(np.concatenate(pred_after, axis=0), axis=-1)
                # 重新计算 ASR = adv_pred != y_clean
                aft_asr = np.sum(pred_after != X) / len(pred_after)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current ASR: {aft_asr:.4f}")
                metric_results.append(aft_asr)
            elif task == 'backdoor':
                current_metric = calculate_metric(model, X_perturbed, task, model_name)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current ASR: {current_metric:.4f}")
                metric_results.append(current_metric)
            elif task == 'safety':
                current_metric = calculate_metric(model, X, task, model_name)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current Safety Violation Rate: {current_metric:.4f}")
                metric_results.append(current_metric)
            elif task == 'fairness':
                # X 是输入特征，G 是群体标签
                current_metric = evaluate_fairness(model, X, Y_true, G)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current Fairness Gap |Acc(1)-Acc(0)|: {current_metric:.4f}")
                metric_results.append(current_metric)
        else:
            print(f"  Skipped non-trainable neuron: {neuron_name}")
        
        k += 1
    
    # 恢复原始权重
    model.set_weights(original_weights)
    
    return metric_results


def main():
    if len(sys.argv) < 3:
        print('Usage: python loc_forward_diff_test.py [task: backdoor|safety|adv] [model_name] [k_sample] [epsilon(adv only)] [resnet_flag(optional)] [k_top(optional)]')
        print('Example: python loc_forward_diff_test.py backdoor gtsrb 512')
        print('Example: python loc_forward_diff_test.py safety a19 512')
        print('Example: python loc_forward_diff_test.py adv mnist 512 0.2')
        print('Note: k_top will be automatically calculated as 1% of total neurons (min 4, max 11)')
        sys.exit(1)
    
    task = sys.argv[1]
    model_name = sys.argv[2]
    k_sample = int(sys.argv[3])
    
    # 解析 epsilon 参数（仅 adv 任务需要）
    epsilon = None
    if task == 'adv':
        if len(sys.argv) < 5:
            print('Error: adv task requires epsilon parameter')
            sys.exit(1)
        epsilon = sys.argv[4]
    
    # 解析 resnet_flag
    resnet_18_flag = False
    if task == 'adv':
        resnet_flag_idx = 5
    else:
        resnet_flag_idx = 4
    
    if len(sys.argv) > resnet_flag_idx:
        if sys.argv[resnet_flag_idx] in ('1', 'True', 'yes', 'y', 't'):
            resnet_18_flag = True
        else:
            resnet_18_flag = False
    # 解析 k_top
    k_top_idx = resnet_flag_idx + 1
    if len(sys.argv) > k_top_idx:
        k_top = int(sys.argv[k_top_idx])
    else:
        k_top = None  # 默认使用自适应top_k
    # 设置缓存路径 - 根据ResNet标志和epsilon调整
    if resnet_18_flag:
        cache_dir = os.path.join('result', 'loc_result', f'{model_name}_resnet')
    else:
        cache_dir = os.path.join('result', 'loc_result', model_name)
    
    # 对于 adv 任务，在缓存文件名中包含 epsilon
    if task == 'adv':
        cache_file = os.path.join(cache_dir, f'forward_difference_{task}_{epsilon}_{k_sample}.json')
    else:
        cache_file = os.path.join(cache_dir, f'forward_difference_{task}_{k_sample}.json')
    import time
    
    # 检查是否有缓存结果
    if os.path.exists(cache_file):
        print(f"=== Loading cached Forward Difference results from {cache_file} ===")
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        players = cache_data['players']
        scores = cache_data['scores']
        print(f"Loaded {len(players)} neurons from cache")
    else:
        print(f"=== Computing Forward Difference Localization for {task} task ===")
        start_time = time.time()
        if resnet_18_flag:
            model_path = os.path.join('..', 'model', f'{model_name}_resnet.h5')
        else:
            if task == 'adv':
                model_path = os.path.join('..', 'model', f'{model_name}_adv.h5')
            else:
                model_path = os.path.join('..', 'model', f'{model_name}.h5')
        model = load_model(model_path, compile=False)
        print("\n=== Model Summary ===")
        model.summary()
        try:
            print(f"Total params: {model.count_params()}")
        except Exception:
            pass
        
        if task == 'backdoor':
            X, Y, X_perturbed = load_data(model_name=model_name, task='backdoor', k_sample=k_sample)
            players, scores = forward_diff_scores(model, X, Y, X_perturbed, sample_n=k_sample, task='backdoor',model_name=model_name)
        elif task == 'safety':
            ct_x, dr_x = load_data(model_name=model_name, task='safety', k_sample=k_sample)
            # 对于安全任务，使用drawndown数据作为正例，counterexample数据作为反例
            # 修改forward_diff_scores函数调用，传入两个不同的数据集
            players, scores = forward_diff_scores_safety(model, dr_x, ct_x, sample_n=k_sample)
        elif task == 'adv':
            X_adv, X_clean, Y_clean = load_data(model_name=model_name, task='adv', epsilon=epsilon, k_sample=k_sample)
            players, scores = forward_diff_scores(model, X_clean, Y_clean, X_adv, sample_n=k_sample, task='adv', model_name=model_name, epsilon=epsilon)
        elif task == 'fairness':
            X, G, _ = load_data(model_name=model_name, task='fairness', k_sample=k_sample)
            players, scores = forward_diff_scores(model, X, G, None, sample_n=k_sample, task='fairness', model_name=model_name)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        
        # 保存缓存结果
        cache_data = {
            'task': task,
            'model': model_name,
            'sample_n': k_sample,
            'method': 'forward_difference',
            'player_schema': 'layer_name_v1',
            'players': players,
            'scores': scores,
            'time': int(time.time() - start_time)}
        if task == 'adv':
            cache_data['epsilon'] = epsilon
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved localization results to {cache_file}")
    
    # 排序并显示结果
    items = list(zip(players, scores))
    items.sort(key=lambda x: x[1], reverse=True)
    ranked = [p for p, _ in items]
    
    print(f"\nTop 10 most important neurons:")
    for i in range(min(10, len(items))):
        print(f"  {i+1}. {items[i][0]}: {items[i][1]:.6f}")
    
    # 验证topk效果（使用自适应top_k）
    print(f"\n=== Validating Top Neurons (Adaptive) ===")
    if resnet_18_flag:
        model_path = os.path.join('..', 'model', f'{model_name}_resnet.h5')
    else:
        if task == 'adv':
            model_path = os.path.join('..', 'model', f'{model_name}_adv.h5')
        else:
            model_path = os.path.join('..', 'model', f'{model_name}.h5')
    model = load_model(model_path, compile=False)
    
    if task == 'backdoor':
        X, Y, X_perturbed = load_data(model_name=model_name, task='backdoor', k_sample=k_sample)
        metric_results = validate_top5_neurons(model, X, X_perturbed, ranked, top_k=k_top, task='backdoor',model_name=model_name)
    elif task == 'safety':
        ct_x, dr_x = load_data(model_name=model_name, task='safety', k_sample=k_sample)
        metric_results = validate_top5_neurons(model, ct_x, None, ranked, top_k=k_top, task='safety', model_name=model_name)
    elif task == 'adv':
        X_adv, X_clean, Y_clean = load_data(model_name=model_name, task='adv', epsilon=epsilon, k_sample=k_sample)
        metric_results = validate_top5_neurons(model, Y_clean, X_adv, ranked, top_k=k_top, task='adv', model_name=model_name)
    elif task == 'fairness':
        X, G, _ = load_data(model_name=model_name, task='fairness', k_sample=k_sample)
        # Pass labels via X_perturbed parameter to reuse signature
        metric_results = validate_top5_neurons(model, X, G, ranked, top_k=k_top, task='fairness', model_name=model_name)
    
    # 保存最终结果 - 根据ResNet标志和epsilon调整路径
    if resnet_18_flag:
        # 对于ResNet模型，创建专门的路径
        out_dir = os.path.join('result', task, f'{model_name}_resnet')
    else:
        # 原始模型路径
        out_dir = os.path.join('result', task, model_name)
    
    # 对于 adv 任务，在输出目录中包含 epsilon
    if task == 'adv':
        out_dir = os.path.join(out_dir, f'epsilon_{epsilon}')
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 计算实际使用的top_k用于文件名
    if k_top is None:
        actual_top_k = max(4, min(11, int(len(ranked) * 0.01)))
    else:
        actual_top_k = k_top
    
    with open(os.path.join(out_dir, f'top{actual_top_k}_forward_diff.txt'), 'w', encoding='utf-8') as f:
        f.write(','.join(ranked))
    meta = {
        'task': task,
        'model': f'{model_name}_resnet' if resnet_18_flag else model_name, 
        'sample_n': k_sample, 
        'num_players': len(ranked), 
        'method': 'forward_difference',
        'resnet_18_flag': resnet_18_flag,
        'top_k': actual_top_k,
        f'top{actual_top_k}_metric_results': metric_results
    }
    if task == 'adv':
        meta['epsilon'] = epsilon
    with open(os.path.join(out_dir, f'top{actual_top_k}_forward_diff.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved ranking to {os.path.join(out_dir, f"top{actual_top_k}_forward_diff.txt")}')


if __name__ == '__main__':
    main()


