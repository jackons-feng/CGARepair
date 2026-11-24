"""
统一的数据加载模块
支持 backdoor, safety, adv, fairness 四种任务的数据加载
"""
import os
import sys
import h5py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def load_data(model_name, task='backdoor', epsilon=None, k_sample=None, csv_path=None):
    """
    统一的数据加载函数
    
    参数:
        model_name: 模型名称 (gtsrb, mnist, fashion, a19, a29, a33, etc.)
        task: 任务类型 ('backdoor', 'safety', 'adv', 'fairness')
        epsilon: 对抗样本的扰动强度 (仅adv任务需要)
        k_sample: 采样数量 (如果指定,则从加载的数据中随机采样k_sample个样本)
    
    返回:
        根据不同任务返回不同的数据:
        - backdoor: (X_clean, Y_clean, X_perturbed)
        - safety: (X_counterexample, X_drawndown)
        - adv: (X_adv, X_clean, Y_clean)
        - fairness: (X, Y, None)
    """
    import time
    time1 = time.time()
    
    def _read_h5_first(h5_path, prefer_key=None):
        """从H5文件读取数据,优先使用prefer_key,否则读取第一个数据集"""
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Expected H5 not found: {h5_path}")
        with h5py.File(h5_path, 'r') as hf:
            if prefer_key is not None and prefer_key in hf:
                return hf[prefer_key][:]
            # fallback to first dataset
            for k in hf.keys():
                try:
                    return hf[k][:]
                except Exception:
                    continue
        raise RuntimeError(f"No readable dataset inside: {h5_path}")
    
    def _sample_data(data_tuple, k_sample):
        """从数据中随机采样k_sample个样本"""
        if k_sample is None:
            return data_tuple
        
        # 获取数据长度(使用第一个非None元素)
        data_len = None
        for d in data_tuple:
            if d is not None:
                data_len = len(d)
                break
        
        if data_len is None or k_sample >= data_len:
            return data_tuple
        
        # 随机采样索引
        rng = np.random.default_rng(123)
        idx = rng.choice(data_len, k_sample, replace=False)
        
        # 对每个数据应用采样
        sampled = []
        for d in data_tuple:
            if d is not None:
                sampled.append(d[idx])
            else:
                sampled.append(None)
        
        return tuple(sampled)

    def sample_per_group(X, Y, G, k_per_group):
        """每个组别采样 k_per_group 个样本；若没有群体标签则返回原数据。"""
        if G is None:
            print("  [Fairness] No group labels available; skipping per-group sampling.")
            return X, Y, G
        unique_groups = np.unique(G)
        sampled_idx = []
        rng = np.random.default_rng(123)
        for g in unique_groups:
            idxs = np.where(G == g)[0]
            n = min(k_per_group, len(idxs))
            sampled_idx.extend(rng.choice(idxs, n, replace=False))
        sampled_idx = np.array(sampled_idx)
        return X[sampled_idx], Y[sampled_idx], G[sampled_idx]


    
    # ========== ADV 任务 ==========
    if task == 'adv':
        if epsilon is None:
            raise ValueError("adv task requires epsilon parameter, e.g., '0.2'")
        
        print(f"=== Loading ADV data for {model_name}, epsilon={epsilon} ===")
        
        # 对于mnist,从txt文件加载攻击数据
        if model_name == 'mnist':
            adv_data_file = os.path.join('..', 'data', 'adv', 'mnist','train', f'epsilon{epsilon}.txt')
            adv_label_file = os.path.join('..', 'data', 'adv', 'mnist','train', 'label.txt')
            # 加载攻击数据
            with open(adv_data_file, 'r') as f:
                data_lines = f.readlines()
            X_adv = []
            for line in data_lines:
                X_adv.append(np.array(line.strip().split(','), dtype=float).reshape((28, 28, 1)))
            X_adv = np.array(X_adv, dtype='float32')
            
            # 加载标签
            with open(adv_label_file, 'r') as f:
                label_lines = f.readlines()
            Y_clean = np.array([int(line.strip()) for line in label_lines], dtype=int)
            
            # 加载干净数据(从keras.datasets)
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
            
            # 只取与攻击数据相同数量的干净数据
            X_clean = x_train[:len(X_adv)]
            
            print(f"ADV data loaded from txt files")
            print(f"Shapes -> X_adv:{X_adv.shape}, X_clean:{X_clean.shape}, Y_clean:{Y_clean.shape}")
            print(f"ADV data loading time: {time.time()-time1:.2f} seconds")
            
            # 采样
            result = _sample_data((X_adv, X_clean, Y_clean), k_sample)
            if k_sample:
                print(f"Sampled {k_sample} examples from {len(X_adv)}")
            return result
        else:
            adv_data_file = os.path.join('..', 'data', 'adv','cifar10', 'train', f'epsilon{epsilon}.npy')
            adv_label_file = os.path.join('..', 'data', 'adv','cifar10', 'train', 'label.npy')
            def load_from_npy(data_path, label_path, shape):
                x_flat = np.load(data_path)
                x = x_flat.reshape((-1,) + shape)
                y = np.load(label_path)
                return x, y
            def preprocess_cifar10(x):
                """Prepares a batch of CIFAR-10 images (0-1 float) for the model."""
                x = x.astype('float32')
                x -= np.array([0.4914, 0.4822, 0.4465])
                x /= np.array([0.247, 0.243, 0.261])
                return x
            x_adv, y_true = load_from_npy(adv_data_file, adv_label_file, (32, 32, 3))
            x_adv=preprocess_cifar10(x_adv)
            (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
            # Ensure clean data is also scaled to [0, 1] before preprocessing
            X_clean = preprocess_cifar10(x_train.astype('float32') / 255.0)
            Y_clean = y_true
            # 采样
            if k_sample:
                print(f"Sampled {k_sample} examples from {len(x_adv)}")
            return _sample_data((x_adv, X_clean, Y_clean), k_sample)

        
    
    # ========== SAFETY 任务 ==========
    elif task == 'safety':
        print(f"=== Loading SAFETY data for {model_name} ===")
        
        # safety数据来源于../data/{model_name}/counterexample.h5和drawndown.h5
        ct_path = os.path.join('..', 'data', model_name, 'counterexample.h5')
        dd_path = os.path.join('..', 'data', model_name, 'drawndown.h5')
        
        ct_x = _read_h5_first(ct_path, 'counterexample')
        dr_x = _read_h5_first(dd_path, 'drawndown')
        
        print(f"Safety data loaded from {os.path.join('..', 'data', model_name)}")
        print(f"Shapes -> counterexample:{ct_x.shape}, drawndown:{dr_x.shape}")
        print(f"Safety data loading time: {time.time()-time1:.2f} seconds")
        
        # 采样(分别对两个数据集采样)
        if k_sample:
            rng = np.random.default_rng(123)
            ct_idx = rng.choice(len(ct_x), min(k_sample, len(ct_x)), replace=False)
            dr_idx = rng.choice(len(dr_x), min(k_sample, len(dr_x)), replace=False)
            ct_x = ct_x[ct_idx]
            dr_x = dr_x[dr_idx]
            print(f"Sampled {len(ct_x)} counterexamples and {len(dr_x)} drawndown examples")
        
        return ct_x, dr_x
    
    # ========== BACKDOOR 任务 ==========
    elif task == 'backdoor':
        print(f"=== Loading BACKDOOR data for {model_name} ===")
        
        # 加载干净数据
        if model_name == 'gtsrb':
            # 使用pkl文件加载
            data_file = os.path.join('..', 'data', 'gtsrb', 'train_data.pkl')
            with open(data_file, 'rb') as f:
                train_data = pickle.load(f)
            
            if isinstance(train_data, dict):
                X_clean = np.array(train_data['data'], dtype='float32')
                Y_clean = np.array(train_data['label'], dtype=int)
            else:
                X_clean = np.array(train_data[0], dtype='float32')
                Y_clean = np.array(train_data[1], dtype=int)
            
            # GTSRB数据预处理: (N,3,48,48) -> (N,48,48,3) -> resize to (N,32,32,3) -> scale to [0,255]
            print(f"Original GTSRB shape: {X_clean.shape}, range: [{X_clean.min():.3f}, {X_clean.max():.3f}]")
            
            # 转置: (N,3,48,48) -> (N,48,48,3)
            X_clean = np.transpose(X_clean, (0, 2, 3, 1))
            
            # Resize到(32,32,3) - 分批处理避免内存问题
            batch_size = 5000
            resized_batches = []
            for i in range(0, len(X_clean), batch_size):
                batch = X_clean[i:i+batch_size]
                # 使用TensorFlow的resize函数
                import tensorflow.python.keras.backend as K
                resized_batch = K.get_session().run(
                    tf.image.resize(batch.astype('float32'), (32, 32))
                )
                resized_batches.append(resized_batch)
            X_clean = np.concatenate(resized_batches, axis=0)
            
            # 将[0,1]范围放大到[0,255]以匹配训练时的数据范围
            X_clean = X_clean * 255.0
            
            print(f"Processed GTSRB shape: {X_clean.shape}, range: [{X_clean.min():.3f}, {X_clean.max():.3f}]")
            print(f"GTSRB data loaded from pkl file")
        elif model_name == 'fashion':
            (x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
            X_clean = np.expand_dims(x_train.astype("float32") / 255, -1)
            Y_clean = y_train
            print(f"Fashion-MNIST data loaded from keras.datasets")
        elif model_name == 'mnist':
            (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
            X_clean = np.expand_dims(x_train.astype("float32") / 255, -1)
            Y_clean = y_train
            print(f"MNIST data loaded from keras.datasets")
        else:
            raise ValueError(f"Unknown model_name for backdoor task: {model_name}")
        
        # 生成后门数据(使用get_perturbed_input函数)
        # 这里需要先采样,然后生成后门数据
        if k_sample and k_sample < len(X_clean):
            rng = np.random.default_rng(123)
            idx = rng.choice(len(X_clean), k_sample, replace=False)
            X_clean = X_clean[idx]
            Y_clean = Y_clean[idx]
            print(f"Sampled {k_sample} clean examples from {len(X_clean)}")
        
        # 生成后门数据
        X_perturbed = _get_perturbed_input_batch(X_clean, model_name)
        
        print(f"Shapes -> X_clean:{X_clean.shape}, Y_clean:{Y_clean.shape}, X_perturbed:{X_perturbed.shape}")
        print(f"Backdoor data loading time: {time.time()-time1:.2f} seconds")
        
        return X_clean, Y_clean, X_perturbed
    
    # ========== FAIRNESS 任务 ==========
    elif task == 'fairness':
        print(f"=== Loading FAIRNESS data for {model_name} ===")

        fr_dir = os.path.join('..', 'data', 'fairness', model_name)
        x_path = os.path.join(fr_dir, 'x_train.h5')
        y_path = os.path.join(fr_dir, 'y_train.h5')
        g_path = os.path.join(fr_dir, 'g_train.h5')

        X = _read_h5_first(x_path, 'x').astype('float32')
        Y = _read_h5_first(y_path, 'y')
        G = _read_h5_first(g_path, 'g') if os.path.exists(g_path) else None
        Y = np.asarray(Y).reshape(-1)
        if G is None:
            print("  [Fairness] No g_train found; using labels as group tags (e.g., LFW).")
            G = Y.copy()
        else:
            G = np.asarray(G).reshape(-1)

        print(f"Fairness data loaded from {fr_dir}")
        print(f"Shapes -> X:{X.shape}, Y:{Y.shape}, G:{None if G is None else G.shape}")
        print(f"Fairness data loading time: {time.time()-time1:.2f} seconds")

        # 按组采样
        X, Y, G = sample_per_group(X, Y, G, k_per_group=2048)
        print(f"After sampling: X:{X.shape}, Y:{Y.shape}, G:{None if G is None else G.shape}")

        return X, Y, G





def _get_perturbed_input_batch(X, model_name):
    """
    批量生成后门数据
    
    参数:
        X: 干净数据 (N, H, W, C)
        model_name: 模型名称
    
    返回:
        X_perturbed: 后门数据 (N, H, W, C)
    """
    # 根据model_name确定参数
    IMG_DIR = os.path.join('..', 'data', 'img')
    IMG_FILENAME_TEMPLATE = '{}_visualize_%s_label_%d.png'.format(model_name)
    if model_name == 'gtsrb':
        INPUT_SHAPE = (32, 32)
        MASK_SHAPE = (32, 32)
        target_label = 33
    elif model_name in ['mnist', 'fashion']:
        INPUT_SHAPE = (28, 28)
        MASK_SHAPE = (28, 28)
        target_label = 3
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    # 加载mask和pattern
    mask = []
    pattern = []
    
    mask_filename = IMG_FILENAME_TEMPLATE % ('mask', target_label)
    mask_path = os.path.join(IMG_DIR, mask_filename)
    if os.path.isfile(mask_path):
        img = image.load_img(
            mask_path,
            color_mode='grayscale',
            target_size=INPUT_SHAPE)
        mask = image.img_to_array(img)
        mask /= 255
        mask = mask[:, :, 0]
    else:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    pattern_filename = IMG_FILENAME_TEMPLATE % ('pattern', target_label)
    pattern_path = os.path.join(IMG_DIR, pattern_filename)
    if os.path.isfile(pattern_path):
        img = image.load_img(
            pattern_path,
            color_mode='rgb',
            target_size=INPUT_SHAPE)
        pattern = image.img_to_array(img)
    else:
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")
    
    # 批量生成后门数据
    X_perturbed = []
    for x in X:
        filtered = np.multiply(x, np.expand_dims(np.subtract(np.ones(MASK_SHAPE), mask), axis=2))
        if model_name == 'gtsrb':
            fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
            x_out = np.add(filtered, fusion)
        else:
            pattern_gray = pattern[:, :, 1] / 255.
            fusion = np.expand_dims(np.multiply(pattern_gray, mask), axis=2)
            x_out = np.add(filtered, fusion)
        X_perturbed.append(x_out)
    
    return np.array(X_perturbed, dtype='float32')


if __name__ == '__main__':
    """测试数据加载功能"""
    print("=== Testing load_data function ===\n")
    
    # 测试backdoor任务
    print("1. Testing backdoor task (mnist):")
    try:
        X, Y, X_p = load_data('mnist', task='backdoor', k_sample=100)
        print(f"✓ Backdoor (mnist) loaded successfully\n")
    except Exception as e:
        print(f"✗ Backdoor (mnist) failed: {e}\n")
    
    # 测试safety任务
    print("2. Testing safety task (a19):")
    try:
        ct_x, dr_x = load_data('a19', task='safety', k_sample=100)
        print(f"✓ Safety (a19) loaded successfully\n")
    except Exception as e:
        print(f"✗ Safety (a19) failed: {e}\n")
    
    # 测试adv任务
    print("3. Testing adv task (mnist, epsilon=0.2):")
    try:
        X_adv, X_clean, Y = load_data('mnist', task='adv', epsilon='0.2', k_sample=100)
        print(f"✓ ADV (mnist) loaded successfully\n")
    except Exception as e:
        print(f"✗ ADV (mnist) failed: {e}\n")
    
    # 测试fairness任务
    print("4. Testing fairness task:")
    try:
        X, Y, G = load_data('some_model', task='fairness', k_sample=100)
        print(f"✓ Fairness loaded successfully\n")
    except Exception as e:
        print(f"✗ Fairness failed: {e}\n")