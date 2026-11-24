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

def load_data(model_name, task='backdoor'):
    """Load data for different tasks"""
    import h5py
    from tensorflow.keras.preprocessing import image
    import time
    time1 = time.time()
    
    if task == 'backdoor':
        # Configuration dictionary from run_backdoor.py
        dic = {'gtsrb': [(32, 32, 3), (43, 33)], 'mnist': [(28, 28, 1), (10, 3)], 'fashion': [(28, 28, 1), (10, 3)]}
        INPUT_SHAPE = dic[model_name][0]
        target_label = dic[model_name][1][1]
        print(f'target_label: {target_label}')
        MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float)).astype(int)
        IMG_DIR = '../data/img'
        IMG_FILENAME_TEMPLATE = '{}_visualize_%s_label_%d.png'.format(model_name)
        
        # Load clean data
        if model_name == 'gtsrb':
            # Try multiple possible paths for the GTSRB dataset
            possible_paths = [
                '../data/gtsrb/gtsrb_dataset_int.h5',
                './data/gtsrb/gtsrb_dataset_int.h5',
                'data/gtsrb/gtsrb_dataset_int.h5',
            ]
            
            dataset = {}
            data_filename = None
            for path in possible_paths:
                if os.path.exists(path):
                    data_filename = path
                    break
            
            with h5py.File(data_filename, 'r') as hf:
                for name in hf:
                    dataset[name] = np.array(hf.get(name))
            X = np.array(dataset['X_test'], dtype='float32')
            Y = np.argmax(np.array(dataset['Y_test'], dtype='float32'), axis=-1)
        elif model_name == 'fashion':
            (X, Y), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
            X = np.expand_dims(X.astype("float32") / 255, -1)
        else:  # mnist
            (X, Y), (_, _) = tf.keras.datasets.mnist.load_data()
            X = np.expand_dims(X.astype("float32") / 255, -1)
        
        # Generate perturbed (backdoor) data
        X_perturbed = get_perturbed_input(X, model_name, target_label, INPUT_SHAPE, MASK_SHAPE, IMG_DIR, IMG_FILENAME_TEMPLATE)
        time2 = time.time()
        print(f"Backdoor data loading time: {time2-time1:.2f} seconds")
        return X, Y, X_perturbed
    
    elif task == 'safety':
        # Load safety data (counterexamples and drawndown data)

        with h5py.File(f'../data/{model_name}/counterexample.h5', 'r') as hf:
            ct_x = hf['counterexample'][:]
        with h5py.File(f'../data/{model_name}/drawndown.h5', 'r') as hf:
            dr_x = hf['drawndown'][:]
        
        time2 = time.time()
        print(f"Safety data loading time: {time2-time1:.2f} seconds")
        return ct_x, dr_x
    
    else:
        raise ValueError(f"Unknown task: {task}")


def load_img_data(model_name):
    """Backward compatibility wrapper"""
    return load_data(model_name, task='backdoor')


def get_perturbed_input(x, model_name, target_label, INPUT_SHAPE, MASK_SHAPE, IMG_DIR, IMG_FILENAME_TEMPLATE):
    """Generate backdoor perturbed inputs following run_backdoor.py logic"""
    from tensorflow.keras.preprocessing import image
    
    mask = np.ones(MASK_SHAPE)  # Default: no mask (all ones)
    pattern = np.zeros((*INPUT_SHAPE[:2], 3))  # Default: no pattern (all zeros)

    mask_filename = IMG_FILENAME_TEMPLATE % ('mask', target_label)
    mask_path = '%s/%s' % (IMG_DIR, mask_filename)
    if os.path.isfile(mask_path):
        img = image.load_img(
            mask_path,
            color_mode='grayscale',
            target_size=INPUT_SHAPE)
        mask = image.img_to_array(img)
        mask /= 255
        mask = mask[:, :, 0]


    pattern_filename = IMG_FILENAME_TEMPLATE % ('pattern', target_label)
    pattern_path = '%s/%s' % (IMG_DIR, pattern_filename)
    if os.path.isfile(pattern_path):
        img = image.load_img(
            pattern_path,
            color_mode='rgb',
            target_size=INPUT_SHAPE)
        pattern = image.img_to_array(img)
    
    # Apply mask and pattern
    filtered = np.multiply(x, np.expand_dims(np.subtract(np.ones((MASK_SHAPE)), mask), axis=2))
    if model_name == 'gtsrb':
        fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
        x_out = np.add(filtered, fusion)
    else:
        pattern_gray = pattern[:, :, 1] / 255.
        fusion = np.expand_dims(np.multiply(pattern_gray, mask), axis=2)
        x_out = np.add(filtered, fusion)
    return x_out

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


def forward_diff_scores(model, X, Y, X_perturbed, sample_n=512, task='backdoor',model_name=None):
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
    elif task == 'safety':
        # 对于安全任务，使用drawndown数据作为正例，counterexample数据作为反例
        # 注意：这里需要修改调用方式，传入counterexample数据
        # 暂时使用相同数据，实际调用时需要传入不同的数据集
        positive_data = Xs  # drawndown data (safe)
        negative_data = Xs  # counterexample data (unsafe) - 需要从外部传入
        print(f"Warning: Using same data for positive and negative examples in safety task")
        print(f"Input shape: {Xs.shape}, Data type: {task}")
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
    
    # 获取所有相关层的索引（只包含卷积层和全连接层）
    all_layer_indices = []
    layer_names = []
    layer_name_to_index = {}  # 层名称到索引的映射
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        # 只处理卷积层和全连接层
        if ('Conv' in layer_type or 'Dense' in layer_type) and hasattr(layer, 'output_shape') and layer.output_shape is not None:
            all_layer_indices.append(i)
            layer_name = f"{layer_type}_{i}"
            layer_names.append(layer_name)
            layer_name_to_index[layer_name] = i
    
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
            for neuron_idx, diff_score in enumerate(importance_diff):
                players.append(f'{layer_name}_{neuron_idx}')
                scores.append(float(diff_score))
    
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
        elif task == 'safety':
            return evaluate_safety(model, X_data, model_name, metric='safety')
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def disable_neuron_by_name(model, neuron_name):
        """根据神经元名称禁用对应的权重"""
        # 解析神经元名称：LayerType_layerIdx_neuronIdx
        parts = neuron_name.split('_')
        if len(parts) < 3:
            print(f"Warning: Cannot parse neuron name {neuron_name}")
            return False
        
        layer_type = parts[0]
        layer_idx_str = parts[1]
        neuron_idx = int(parts[2])
        
        # 构建层名称
        layer_name = f"{layer_type}_{layer_idx_str}"
        
        # 检查是否是有效的层类型
        if not ('Conv' in layer_type or 'Dense' in layer_type):
            print(f"Warning: Layer type {layer_type} is not trainable (no weights)")
            return False
        
        # 找到对应的层
        layer_idx = int(layer_idx_str)
        if layer_idx >= len(model.layers):
            print(f"Warning: Layer index {layer_idx} out of range")
            return False
        
        layer = model.layers[layer_idx]
        
        # 获取层的权重
        weights = layer.get_weights()
        if len(weights) == 0:
            print(f"Warning: Layer {layer_idx} ({layer_type}) has no weights")
            return False
        
        # 禁用神经元
        if 'Conv' in layer_type:
            # 卷积层：禁用对应的输出通道
            if len(weights) > 0:  # kernel weights
                kernel = weights[0]
                if neuron_idx < kernel.shape[-1]:  # 检查神经元索引是否有效
                    kernel[:, :, :, neuron_idx] = 0.0
                    weights[0] = kernel
                    layer.set_weights(weights)
                    return True
                else:
                    print(f"Warning: Neuron index {neuron_idx} out of range for layer {layer_name}")
                    return False
        elif 'Dense' in layer_type:
            # 全连接层：禁用对应的输出神经元
            if len(weights) > 0:  # kernel weights
                kernel = weights[0]
                if neuron_idx < kernel.shape[-1]:  # 检查神经元索引是否有效
                    kernel[:, neuron_idx] = 0.0
                    weights[0] = kernel
                    layer.set_weights(weights)
                    return True
                else:
                    print(f"Warning: Neuron index {neuron_idx} out of range for layer {layer_name}")
                    return False
        
        print(f"Warning: Could not disable neuron {neuron_name}")
        return False
    
    # 计算基线指标
    if task == 'backdoor':
        baseline_metric = calculate_metric(model, X_perturbed, task,model_name=model_name)
        print(f"Baseline ASR: {baseline_metric:.4f}")
    elif task == 'safety':
        baseline_metric = calculate_metric(model=model, X_data=X, task=task, model_name=model_name)
        print(f"Baseline Safety Violation Rate: {baseline_metric:.4f}")
    
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
            if task == 'backdoor':
                current_metric = calculate_metric(model, X_perturbed, task, model_name)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current ASR: {current_metric:.4f}")
            elif task == 'safety':
                current_metric = calculate_metric(model, X, task, model_name)
                print(f"  Successfully disabled top-{disabled_count} neuron: {neuron_name}")
                print(f"  Current Safety Violation Rate: {current_metric:.4f}")
            metric_results.append(current_metric)
        else:
            print(f"  Skipped non-trainable neuron: {neuron_name}")
        
        k += 1
    
    # 恢复原始权重
    model.set_weights(original_weights)
    
    return metric_results


def main():
    if len(sys.argv) < 3:
        print('Usage: python loc_forward_diff.py [task: backdoor|safety] [model_name] [k_sample] [resnet_flag(optional)][k_top(optional)]  ')
        print('Example: python loc_forward_diff.py backdoor gtsrb 512 True ')
        print('Example: python loc_forward_diff.py safety a19 512')
        print('Note: k_top will be automatically calculated as 1% of total neurons (min 4, max 11)')
        sys.exit(1)
    
    task = sys.argv[1]
    model_name = sys.argv[2]
    k_sample = int(sys.argv[3])
    if len(sys.argv) > 4:
        if sys.argv[4] in ('1', 'True', 'yes', 'y', 't'):
            resnet_18_flag = True
        else:
            resnet_18_flag = False
    else:
        resnet_18_flag = False
    if len(sys.argv) > 5:
        k_top = int(sys.argv[5])
    else:
        k_top = None  # 默认使用自适应top_k
    # 设置缓存路径 - 根据ResNet标志调整
    if resnet_18_flag:
        cache_dir = os.path.join('result', 'loc_result', f'{model_name}_resnet')
    else:
        cache_dir = os.path.join('result', 'loc_result', model_name)
    os.makedirs(cache_dir, exist_ok=True)
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
            model_path = os.path.join('..', 'model', f'{model_name}.h5')
        model = load_model(model_path, compile=False)
        
        if task == 'backdoor':
            X, Y, X_perturbed = load_data(model_name=model_name, task='backdoor')
            players, scores = forward_diff_scores(model, X, Y, X_perturbed, sample_n=k_sample, task='backdoor',model_name=model_name)
        elif task == 'safety':
            ct_x, dr_x = load_data(model_name=model_name, task='safety')
            # 对于安全任务，使用drawndown数据作为正例，counterexample数据作为反例
            # 修改forward_diff_scores函数调用，传入两个不同的数据集
            players, scores = forward_diff_scores_safety(model, dr_x, ct_x, sample_n=k_sample)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        
        # 保存缓存结果
        cache_data = {
            'task': task,
            'model': model_name,
            'sample_n': k_sample,
            'method': 'forward_difference',
            'players': players,
            'scores': scores,
            'time': int(time.time() - start_time)}
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
        model_path = os.path.join('..', 'model', f'{model_name}.h5')
    model = load_model(model_path, compile=False)
    
    if task == 'backdoor':
        X, Y, X_perturbed = load_data(model_name=model_name, task='backdoor')
        metric_results = validate_top5_neurons(model, X, X_perturbed, ranked, top_k=k_top, task='backdoor',model_name=model_name)
    elif task == 'safety':
        ct_x, dr_x = load_data(model_name=model_name, task='safety')
        metric_results = validate_top5_neurons(model, ct_x, None, ranked, top_k=k_top, task='safety', model_name=model_name)
    
    # 保存最终结果 - 根据ResNet标志调整路径
    if resnet_18_flag:
        # 对于ResNet模型，创建专门的路径
        out_dir = os.path.join('result', task, f'{model_name}_resnet')
    else:
        # 原始模型路径
        out_dir = os.path.join('result', task, model_name)
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
    with open(os.path.join(out_dir, f'top{actual_top_k}_forward_diff.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved ranking to {os.path.join(out_dir, f"top{actual_top_k}_forward_diff.txt")}')


if __name__ == '__main__':
    main()


