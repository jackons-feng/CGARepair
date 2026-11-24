"""
Compare different repair algorithms on backdoor task
Run: python compare_algorithms.py <model_name> <alpha> [seed] [algorithm]

Supported algorithms (based on reviewer feedback):
- NSGA-II: Recommended by Reviewer 1 for 2-objective problems
- MOSPO: Multi-Objective Sparrow Search Optimization algorithm
- Gradient-Pareto: Suggested by Reviewer 2 as gradient-based alternative
  * Uses differentiable loss functions (cross-entropy)
  * Uses automatic differentiation (tf.gradients)
  * Generates Pareto front via multiple weight combinations
"""

import os
import sys
import time
import json
import h5py
import pickle
import numpy as np
import tensorflow as tf
import traceback
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K
from repair_algorithms import (
    BackdoorRepairProblem, 
    NSGA2Repair, 
    MOSPORepair,
    GradientRepair
)

# Setup TensorFlow
tf.compat.v1.disable_eager_execution()
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

# Dataset configuration
DATASET_CONFIG = {
    'gtsrb': {'input_shape': (32, 32, 3), 'num_classes': 43, 'target_label': 33},
    'mnist': {'input_shape': (28, 28, 1), 'num_classes': 10, 'target_label': 3},
    'fashion': {'input_shape': (28, 28, 1), 'num_classes': 10, 'target_label': 3}
}


def get_layer_index(model, c_flag=True, d_flag=True, b_flag=True):
    """Get layer indices for conv, dense, and batch normalization layers"""
    conv, dense, batch = [], [], []
    weight_idx = 0
    
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
            
        lname = type(layer).__name__
        
        if b_flag and 'Batch' in lname:
            batch.append(weight_idx)
            weight_idx += len(weights)
        elif c_flag and 'Conv' in lname:
            conv.append(weight_idx)
            weight_idx += len(weights)
        elif d_flag and 'Dense' in lname:
            dense.append(weight_idx)
            weight_idx += len(weights)
    
    return conv, dense, batch


def get_layer_index_mapping(model):
    """
    Get mapping from actual layer index (in model.layers) to weight index
    
    This converts from the layer indices used in player names (actual model.layers indices)
    to the weight indices in model.get_weights()
    
    Returns:
        actual_to_weight: dict mapping (layer_type, actual_layer_idx) to weight index in model.get_weights()
    """
    actual_to_weight = {}
    weight_idx = 0
    
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        lname = type(layer).__name__
        
        if 'Conv' in lname:
            # For Conv layers, we only care about the first weight (the kernel weights)
            # Bias is the second weight (if present), but we typically don't repair biases
            # Verify that the first weight is indeed kernel weights (should be multi-dimensional)
            if len(weights) > 0 and len(weights[0].shape) < 2:
                raise ValueError(
                    f"Conv2D layer {i} has unexpected weight shape: {weights[0].shape}. "
                    f"Expected kernel weights to be multi-dimensional (at least 2D)."
                )
            actual_to_weight[('conv2d', i)] = weight_idx
            weight_idx += len(weights)
        elif 'Dense' in lname:
            # For Dense layers, we only care about the first weight (the kernel weights)
            # Bias is the second weight (if present), but we typically don't repair biases
            # Verify that the first weight is indeed kernel weights (should be 2D)
            if len(weights) > 0 and len(weights[0].shape) < 2:
                raise ValueError(
                    f"Dense layer {i} has unexpected weight shape: {weights[0].shape}. "
                    f"Expected kernel weights to be 2D (input_features, output_features)."
                )
            actual_to_weight[('dense', i)] = weight_idx
            weight_idx += len(weights)
        else:
            # For other layers (BatchNorm, etc.), just skip their weights
            # BatchNorm typically has 4 weights: gamma, beta, moving_mean, moving_variance
            # All are 1D tensors, so we skip them
            weight_idx += len(weights)
    
    return actual_to_weight

def load_data(model_name, keys=None):
    """Load dataset based on model name"""
    def process_data_batch(X_data, batch_size=5000):
        processed_batches = []
        for i in range(0, len(X_data), batch_size):
            batch = X_data[i:i+batch_size]
            batch = np.transpose(batch, (0, 2, 3, 1))
            batch = K.get_session().run(tf.image.resize(batch.astype('float32'), (32, 32)))
            batch = batch * 255.0
            processed_batches.append(batch)
        return np.concatenate(processed_batches, axis=0)
    if model_name == 'gtsrb':
        data_file = os.path.join('..', 'data', 'gtsrb', 'train_data.pkl')
        with open(data_file, 'rb') as f:
            train_data = pickle.load(f)
            X_clean = process_data_batch(train_data['data'])
            Y_clean = np.array(train_data['label'], dtype=int)
        return X_clean, Y_clean
    elif model_name == 'fashion':
        (x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
    else:  # mnist
         (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
         x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
        #raise NotImplementedError("MNIST data not implemented")
    return x_train, y_train


def get_perturbed_input(x, model_name, target_label, input_shape, mask_shape):
    """Generate backdoor-perturbed input using mask and pattern"""
    from tensorflow.keras.preprocessing import image
    
    mask = []
    pattern = []
    IMG_DIR = '../data/img'

    img_filename_template = '{}_visualize_%s_label_%d.png'.format(model_name)
    mask_filename = img_filename_template % ('mask', target_label)
    if os.path.isfile('%s/%s' % (IMG_DIR, mask_filename)):
        img = image.load_img(
            '%s/%s' % (IMG_DIR, mask_filename),
            color_mode='grayscale',
            target_size=input_shape)
        mask = image.img_to_array(img)
        mask /= 255
        mask = mask[:, :, 0]

    pattern_filename = img_filename_template % ('pattern', target_label)
    if os.path.isfile('%s/%s' % (IMG_DIR, pattern_filename)):
        img = image.load_img(
            '%s/%s' % (IMG_DIR, pattern_filename),
            color_mode='rgb',
            target_size=input_shape)
        pattern = image.img_to_array(img)
    
    filtered = np.multiply(x, np.expand_dims(np.subtract(np.ones(mask_shape), mask), axis=2))
    if model_name == 'gtsrb':
        fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
        x_out = np.add(filtered, fusion)
    else:
        pattern = pattern[:, :, 1] / 255.
        fusion = np.expand_dims(np.multiply(pattern, mask), axis=2)
        x_out = np.add(filtered, fusion)
    return x_out


def select_data(X, Y, X_p, num_examples):
    """Select data samples for evaluation"""
    candidate_x = X
    candidate_y = Y
    random_indices = np.random.choice(len(candidate_x), min(num_examples, len(candidate_x)), replace=False)
    return candidate_x[random_indices], candidate_y[random_indices], X_p[random_indices]
def prepare_repair_data(model_name, alpha, loc_dir, repair_ratio=0.01, num_examples=10000, resnet_flag=False):
    """Prepare data for backdoor repair"""
    if model_name == 'gtsrb' and resnet_flag:
        path_to_model = f'../model/{model_name}_resnet.h5'
    else:
        path_to_model = f'../model/{model_name}.h5'
    
    print(f"\n{'='*60}")
    print(f"Loading model: {path_to_model}")
    print(f"{'='*60}\n")
    
    model = load_model(path_to_model, compile=False)
    convs, denses, _ = get_layer_index(model, c_flag=True, d_flag=True, b_flag=True)
    
    # Create mapping from actual layer index to weight index
    actual_to_weight = get_layer_index_mapping(model)
    
    # Debug: Print actual layer mapping
    dense_layers_in_mapping = [k[1] for k in actual_to_weight.keys() if k[0] == 'dense']
    conv_layers_in_mapping = [k[1] for k in actual_to_weight.keys() if k[0] == 'conv2d']
    
    print(f"Model layer information:")
    print(f"  Conv layers: {len(convs)} (weight indices: {convs})")
    print(f"  Dense layers: {len(denses)} (weight indices: {denses})")
    print(f"  Actual layer mapping:")
    print(f"    Conv layers in model: {sorted(conv_layers_in_mapping)}")
    print(f"    Dense layers in model: {sorted(dense_layers_in_mapping)}")
    print()
    
    players_file = os.path.join(loc_dir, 'players.txt')
    backdoor_vals_file = os.path.join(loc_dir, 'backdoor_vals.txt')
    print(f"Loading players from: {players_file}")
    print(f"Loading backdoor_vals from: {backdoor_vals_file}")
    
    with open(players_file) as f:
        players = f.read().split(',')
    with open(backdoor_vals_file) as f:
        backdoor_vals = list(map(float, f.read().split(',')))
    
    dic_sr = dict(zip(players, backdoor_vals))
    dic_players = {player: dic_sr[player]for player in players}
    
    import math
    repair_num = math.ceil(len(players) * repair_ratio)
    selected_players = [x[0] for x in sorted(dic_players.items(), key=lambda x: x[1], reverse=True)[:repair_num]]
    
    print(f"Selected {len(selected_players)} players for repair (out of {len(players)} total players)")
    if len(selected_players) > 0:
        print(f"Sample players: {selected_players[:5]}")
    print()
    
    init_weights = []
    shapes = []
    for player in selected_players:
        player_parts = player.split('_')
        layer_type = player_parts[0].lower()  # Convert to lowercase to handle both "conv" and "Conv"
        # Normalize layer_type: convert "conv" or "conv2d" to "conv2d"
        if layer_type in ['conv', 'conv2d']:
            layer_type = 'conv2d'
        actual_layer_idx = int(player_parts[1])  # This is the actual layer index in model.layers
        channel_idx = int(player_parts[-1])
        
        # Map from actual layer index to weight index
        key = (layer_type, actual_layer_idx)
        if key not in actual_to_weight:
            # Provide helpful error message with available layers
            if layer_type == 'dense':
                available_dense_layers = [k[1] for k in actual_to_weight.keys() if k[0] == 'dense']
                raise IndexError(f"Dense layer index {actual_layer_idx} out of range. Available dense layers: {len(available_dense_layers)} (indices: {available_dense_layers}). Player: {player}")
            else:
                available_conv_layers = [k[1] for k in actual_to_weight.keys() if k[0] == 'conv2d']
                raise IndexError(f"{layer_type.capitalize()} layer index {actual_layer_idx} out of range. Available {layer_type} layers: {len(available_conv_layers)} (indices: {available_conv_layers}). Player: {player}")
        weight_index = actual_to_weight[key]
        
        # Debug: Check if weight_index is valid
        all_weights = model.get_weights()
        if weight_index >= len(all_weights):
            raise IndexError(
                f"Player {player}: Weight index {weight_index} out of range. "
                f"Model has {len(all_weights)} weights. "
                f"Layer: {layer_type} {actual_layer_idx}, key: {key}"
            )
        
        weight_tensor = all_weights[weight_index]
        
        # Verify that weight_index points to kernel weights, not bias or BatchNorm weights
        # For Conv2D: kernel shape should be (kernel_h, kernel_w, input_channels, output_channels) - 4D
        # For Dense: kernel shape should be (input_features, output_features) - 2D
        # If we get a 1D tensor, it might be a bias weight or BatchNorm weight, which is wrong
        if len(weight_tensor.shape) == 1:
            # This might be a bias weight or BatchNorm weight, which shouldn't happen
            # Check the actual layer to see what type it is
            actual_layer = model.layers[actual_layer_idx]
            layer_name = type(actual_layer).__name__
            if 'BatchNorm' in layer_name or 'BatchNormalization' in layer_name:
                raise ValueError(
                    f"Player {player}: weight_index {weight_index} points to a BatchNormalization layer weight "
                    f"(shape: {weight_tensor.shape}), but player references {layer_type} layer {actual_layer_idx}. "
                    f"This indicates a mapping error. The actual layer at index {actual_layer_idx} is {layer_name}, "
                    f"not a {layer_type} layer. Please check the layer indices in player names."
                )
            elif weight_tensor.shape[0] < channel_idx + 1:
                # This is likely a bias weight (1D with shape (output_channels,))
                # But we're trying to access channel_idx which is >= output_channels
                raise ValueError(
                    f"Player {player}: weight_index {weight_index} points to a 1D weight tensor "
                    f"(shape: {weight_tensor.shape}), which is likely a bias weight, not kernel weights. "
                    f"Kernel weights for {layer_type} layers should be multi-dimensional. "
                    f"This suggests the weight_index mapping is incorrect. "
                    f"Layer: {layer_type} {actual_layer_idx}, channel_idx: {channel_idx}"
                )
        
        # Debug: print weight tensor shape for troubleshooting
        print(f"DEBUG: Player {player}: weight_index={weight_index}, weight_shape={weight_tensor.shape}, channel_idx={channel_idx}, layer_type={layer_type}, actual_layer_idx={actual_layer_idx}")
        
        # For 1D weight tensors, we need special handling
        if len(weight_tensor.shape) == 0:
            # Scalar weight - shouldn't happen, but handle it
            raise ValueError(
                f"Player {player}: Weight tensor at index {weight_index} is a scalar (shape: {weight_tensor.shape}). "
                f"This should not happen for {layer_type} layers."
            )
        elif len(weight_tensor.shape) == 1:
            # 1D weight tensor: channel_idx should be within [0, shape[0])
            max_channels = weight_tensor.shape[0]
            if channel_idx < 0 or channel_idx >= max_channels:
                raise IndexError(
                    f"Player {player}: channel index {channel_idx} out of range for {layer_type} layer {actual_layer_idx} "
                    f"(weight_index {weight_index}). Weight shape: {weight_tensor.shape} (1D), max channels: {max_channels}. "
                    f"Player format: {player_parts}. Note: For 1D weights, channel_idx must be < {max_channels}"
                )
            # For 1D weights, extract the scalar value and convert to 1D array for consistency
            try:
                weight = np.array([weight_tensor[channel_idx]])
            except IndexError as e:
                raise IndexError(
                    f"Player {player}: Failed to access index {channel_idx} from 1D weight tensor. "
                    f"Weight shape: {weight_tensor.shape}, weight_index: {weight_index}, "
                    f"layer: {layer_type} {actual_layer_idx}. Original error: {str(e)}"
                ) from e
        else:
            # Multi-dimensional weight tensor: channel_idx refers to the last dimension
            max_channels = weight_tensor.shape[-1]
            if channel_idx < 0 or channel_idx >= max_channels:
                raise IndexError(
                    f"Player {player}: channel index {channel_idx} out of range for {layer_type} layer {actual_layer_idx} "
                    f"(weight_index {weight_index}). Weight shape: {weight_tensor.shape}, max channels: {max_channels}. "
                    f"Player format: {player_parts}"
                )
            try:
                # Use ellipsis to access the last dimension
                weight = weight_tensor[..., channel_idx]
            except IndexError as e:
                # Provide detailed error information
                error_msg = str(e)
                raise IndexError(
                    f"Player {player}: Failed to access channel {channel_idx} from {layer_type} layer {actual_layer_idx} "
                    f"(weight_index {weight_index}). Weight shape: {weight_tensor.shape}, "
                    f"max channels (last dim): {max_channels}, channel_idx: {channel_idx}. "
                    f"Player format: {player_parts}. Original error: {error_msg}"
                ) from e
        shapes.append(weight.shape)
        init_weights.append(weight.flatten())
    
    init_weights = np.concatenate(init_weights)

    weight_slices = []
    begin_index = 0
    for shape in shapes:
        end_index = begin_index + np.prod(shape)
        weight_slices.append((begin_index, end_index))
        begin_index = end_index
    
    data_x, data_y = load_data(model_name)
    config = DATASET_CONFIG[model_name]
    input_shape = config['input_shape']
    target_label = config['target_label']
    mask_shape = np.ceil(np.array(input_shape[0:2], dtype=float)).astype(int)
    data_p = get_perturbed_input(data_x, model_name, target_label, input_shape, mask_shape)
    data_x, data_y, data_p = select_data(data_x, data_y, data_p, num_examples)
    
    print(f"\n{'='*60}")
    print("Evaluating INITIAL model performance (before repair)")
    print(f"{'='*60}\n")
    
    predictions = model.predict(data_x, verbose=0, batch_size=256)
    predicted_labels = np.argmax(predictions, axis=1)
    initial_acc = np.sum(predicted_labels == data_y) / len(data_y)
    
    predictions_p = model.predict(data_p, verbose=0, batch_size=256)
    predicted_labels_p = np.argmax(predictions_p, axis=1)
    
    if model_name == 'gtsrb':
        non_target_mask = (data_y != target_label)
        if np.sum(non_target_mask) > 0:
            backdoor_success = np.sum(predicted_labels_p[non_target_mask] == target_label)
            initial_sr = backdoor_success / np.sum(non_target_mask)
        else:
            initial_sr = 0.0
    else:
        backdoor_success = np.sum(predicted_labels_p == target_label)
        initial_sr = backdoor_success / len(data_y)
    
    print(f"Initial Performance: Acc={initial_acc:.4f}, SR={initial_sr:.4f}")
    
    return {
        'model_path': path_to_model,
        'init_weights': init_weights,
        'selected_players': selected_players,
        'shapes': shapes,
        'weight_slices': weight_slices,
        'convs': convs,
        'denses': denses,
        'data_x': data_x,
        'data_y': data_y,
        'data_p': data_p,
        'model_name': model_name,
        'alpha': alpha,
        'actual_to_weight': actual_to_weight  # Mapping from (layer_type, actual_layer_idx) to weight index
    }


def run_comparison(model_name, algorithms, n_gen=100, pop_size=20, seed=None, resnet_flag=False):
    model_suffix = " (ResNet)" if (model_name == 'gtsrb' and resnet_flag) else ""
    print(f"\n{'='*60}")
    print(f"Comparing Repair Algorithms on {model_name}{model_suffix}")
    print(f"{'='*60}\n")

    if resnet_flag and model_name == 'gtsrb':
        loc_dir = f'./loc_result/{model_name}_resnet'
    else:
        loc_dir = f'./loc_result/{model_name}'
    
    print(f"Reading players.txt from: {loc_dir}")
    players_file = os.path.join(loc_dir, 'players.txt')
    if not os.path.exists(players_file):
        raise FileNotFoundError(f"players.txt not found at: {players_file}")
    
    data = prepare_repair_data(model_name, seed, loc_dir, resnet_flag=resnet_flag)
    
    thresholds = {
        'gtsrb': {'accuracy': 0.93, 'backdoor': 0.001},
        'mnist': {'accuracy': 0.9858, 'backdoor': 0.001},
        'fashion': {'accuracy': 0.9213, 'backdoor': 0.001}
    }
    
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo.name}")
        print(f"{'='*60}\n")
        
        # Clear session before each algorithm to avoid graph conflicts
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)
        
        start_time = time.time()
        
        try:
            # Create fresh problem instance for each algorithm
            problem = BackdoorRepairProblem(
                model_path=data['model_path'],
                init_weights=np.tile(data['init_weights'], (pop_size, 1)),
                selected_players=data['selected_players'],
                shapes=data['shapes'],
                convs=data['convs'],
                denses=data['denses'],
                data_x=data['data_x'],
                data_y=data['data_y'],
                data_p=data['data_p'],
                model_name=data['model_name'],
                threshold_acc=thresholds[model_name]['accuracy'],
                threshold_sr=thresholds[model_name]['backdoor']
            )
            
            # Initial population
            init_population = np.tile(data['init_weights'], (pop_size, 1))
            init_population += np.random.uniform(-0.05, 0.05, init_population.shape)
            
            res = algo.repair(problem, init_population, n_gen)
            elapsed_time = time.time() - start_time
            
            if hasattr(res, 'F') and hasattr(res, 'X'):
                final_f = res.F.reshape(-1, 2)
                final_x = res.X.reshape(len(final_f), -1)
                
                # Filter solutions by constraints
                # Objectives format may differ by algorithm:
                #   - NSGA-II: [backdoor_rate, -accuracy] (both minimized)
                #   - Others (MOSPO/Gradient): [backdoor_rate, accuracy]
                threshold_acc = thresholds[model_name]['accuracy']
                threshold_sr = thresholds[model_name]['backdoor']
                is_nsga = 'nsga' in algo.name.lower()
                
                # Find solutions that satisfy constraints
                valid_indices = []
                valid_solutions_info = []
                for i, (f, x) in enumerate(zip(final_f, final_x)):
                    backdoor_rate = f[0]
                    accuracy = (-f[1]) if is_nsga else f[1]
                    if accuracy >= threshold_acc and backdoor_rate <= threshold_sr:
                        valid_indices.append(i)
                        valid_solutions_info.append((i, backdoor_rate, accuracy))
                
                # Save ALL solutions (for plotting full Pareto), but keep a count of valid ones
                filtered_f = final_f
                filtered_x = final_x
                satisfies_constraints = len(valid_indices) > 0
                print(f"  Info: {len(valid_indices)}/{len(final_f)} solutions satisfy constraints (saving all solutions)")
                
                # Deduplication: Remove duplicate solutions (same objective values)
                if len(filtered_f) > 0:
                    unique_objectives = {}
                    unique_indices = []
                    
                    for i, f in enumerate(filtered_f):
                        backdoor_rate = f[0]
                        accuracy = (-f[1]) if is_nsga else f[1]
                        key = (round(backdoor_rate, 8), round(accuracy, 8))
                        
                        if key not in unique_objectives:
                            unique_objectives[key] = i
                            unique_indices.append(i)
                    
                    if len(unique_indices) < len(filtered_f):
                        print(f"  Deduplication: {len(filtered_f)} → {len(unique_indices)} unique solutions")
                        filtered_f = filtered_f[unique_indices]
                        filtered_x = filtered_x[unique_indices]
                    else:
                        print(f"  No duplicate solutions found")
                
                # Choose solution with minimum backdoor rate
                best_idx_in_filtered = int(np.argmin(filtered_f[:, 0]))
                best_backdoor = float(filtered_f[best_idx_in_filtered, 0])
                best_acc = float((-filtered_f[best_idx_in_filtered, 1]) if is_nsga else filtered_f[best_idx_in_filtered, 1])
                
                results[algo.name] = {
                    'success': True,
                    'time': elapsed_time,
                    'final_backdoor': best_backdoor,
                    'final_accuracy': best_acc,
                    'n_solutions': len(filtered_f),
                    'n_original_solutions': len(final_f),
                    'n_valid_solutions': len(valid_indices),
                    'satisfies_constraints': satisfies_constraints,
                    'pareto_front': filtered_f.tolist(),
                    'pareto_solutions': filtered_x,
                    'best_solution_idx': best_idx_in_filtered,
                    'objective_format': 'sr_negacc' if is_nsga else 'sr_acc'
                }
                
                print(f"\n{algo.name} Results:")
                print(f"  Time: {elapsed_time:.2f}s")
                print(f"  Original Pareto front: {len(final_f)} solutions")
                print(f"  Filtered Pareto front: {len(filtered_f)} solutions")
                print(f"  Best backdoor rate: {best_backdoor:.6f}")
                print(f"  Best accuracy: {best_acc:.6f}")
                print(f"  Satisfies constraints: {satisfies_constraints}")
            else:
                results[algo.name] = {'success': False, 'error': 'No results returned', 'time': elapsed_time}
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            results[algo.name] = {'success': False, 'error': str(e), 'time': elapsed_time}
            print(f"\n{algo.name} FAILED: {e}")
            traceback.print_exc()
    if resnet_flag and model_name == 'gtsrb':
        output_dir = f'./loc_result/{model_name}_resnet/{seed}'
    else:
        output_dir = f'./loc_result/{model_name}/{seed}'
    os.makedirs(output_dir, exist_ok=True)
    
    for algo_name, result in results.items():
        if result['success'] and 'pareto_solutions' in result:
            algo_dir_name = algo_name.lower().replace('-', '').replace('/', '')
            if 'nsga' in algo_dir_name: algo_dir_name = 'nsga2'
            elif 'mospo' in algo_dir_name: algo_dir_name = 'mospo'
            elif 'gradient' in algo_dir_name: algo_dir_name = 'gradient'
            
            algo_output_dir = os.path.join(output_dir, algo_dir_name)
            pareto_dir = os.path.join(algo_output_dir, 'pareto_front')
            os.makedirs(pareto_dir, exist_ok=True)
            
            print(f"\nSaving {algo_name} Pareto front solutions to {pareto_dir}")
            
            pareto_solutions = result['pareto_solutions']
            pareto_front = np.array(result['pareto_front'])
            pareto_records = []
            objective_format = result.get('objective_format', 'sr_negacc' if 'nsga' in algo_name.lower() else 'sr_acc')
            
            # Use the saved actual_to_weight mapping (same as used during optimization)
            actual_to_weight = data['actual_to_weight']
            
            for idx in range(len(pareto_solutions)):
                # Reload model for each solution to ensure clean state
                save_model = load_model(data['model_path'], compile=False)
                weights = save_model.get_weights()
                
                for player_idx, (player, shape) in enumerate(zip(data['selected_players'], data['shapes'])):
                    begin_idx, end_idx = data['weight_slices'][player_idx]
                    new_weight = pareto_solutions[idx][begin_idx:end_idx].reshape(shape)
                    
                    player_parts = player.split('_')
                    layer_type = player_parts[0].lower()
                    # Normalize layer_type: convert "conv" or "conv2d" to "conv2d"
                    if layer_type in ['conv', 'conv2d']:
                        layer_type = 'conv2d'
                    actual_layer_idx = int(player_parts[1])
                    channel_idx = int(player_parts[-1])
                    
                    # Map from actual layer index to weight index using saved mapping
                    key = (layer_type, actual_layer_idx)
                    if key not in actual_to_weight:
                        raise KeyError(f"Player {player} references {layer_type} layer {actual_layer_idx}, but this key is not in actual_to_weight mapping")
                    weight_index = actual_to_weight[key]
                    
                    # Validate weight_index and channel_idx
                    if weight_index >= len(weights):
                        raise IndexError(f"Player {player}: weight_index {weight_index} >= {len(weights)} (number of weight arrays)")
                    if channel_idx >= weights[weight_index].shape[-1]:
                        raise IndexError(f"Player {player}: channel_idx {channel_idx} >= {weights[weight_index].shape[-1]} (number of channels in weight array {weight_index})")
                    
                    weights[weight_index][..., channel_idx] = new_weight
                
                save_model.set_weights(weights)
                save_model.save_weights(os.path.join(pareto_dir, f'weights_{idx}.h5'))
                
                backdoor_rate = float(pareto_front[idx, 0])
                accuracy = float((-pareto_front[idx, 1]) if objective_format == 'sr_negacc' else pareto_front[idx, 1])
                # Normalize for CSV: ensure ASR in [0,1]
                if backdoor_rate > 1.0:
                    backdoor_rate = backdoor_rate / 100.0
                pareto_records.append({'index': idx, 'train_backdoor': backdoor_rate, 'train_accuracy': accuracy})
            
            with open(os.path.join(pareto_dir, 'pareto_front.txt'), 'w') as f:
                f.write("index,backdoor_rate,accuracy\n")
                for record in pareto_records:
                    f.write(f"{record['index']},{record['train_backdoor']:.10f},{record['train_accuracy']:.10f}\n")
            
            del result['pareto_solutions']
    
    output_file = os.path.join(output_dir, f'comparison_{int(time.time())}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison results saved to: {output_file}")
    
    print("\nSummary Table:")
    print(f"{'Algorithm':<20} {'Time(s)':<12} {'#Solutions':<12} {'Backdoor Rate':<15} {'Accuracy':<12} {'Status':<10}")
    print("-" * 85)
    for algo_name, result in results.items():
        if result['success']:
            n_sols = result.get('n_solutions', 1)
            print(f"{algo_name:<20} {result['time']:<12.2f} {n_sols:<12} {result['final_backdoor']:<15.6f} {result['final_accuracy']:<12.6f} {'Success':<10}")
        else:
            print(f"{algo_name:<20} {result['time']:<12.2f} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'Failed':<10}")
    
    return results


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_algorithms.py <model_name> <resnet> [--seed SEED] [--algo ALGORITHM]")
        print("\nAvailable algorithms:")
        print("  - nsga2    : NSGA-II ")
        print("  - mospo    : MOSPO ")
        print("  - gradient : Gradient-Pareto ")
        print("               Uses differentiable losses + automatic differentiation")
        print("               Generates Pareto front with multiple weight combinations")
        sys.exit(1)
    
    model_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if len(sys.argv) >= 4:
        resnet_flag = sys.argv[3]
        if resnet_flag in ['True', 'true', '1', 'yes', 'y']:
            resnet_flag = True
        else:
            resnet_flag = False
    else:
        resnet_flag = False
    algorithm_name = None
    if '--algo' in sys.argv:
        algo_idx = sys.argv.index('--algo')
        if algo_idx + 1 < len(sys.argv):
            algorithm_name = sys.argv[algo_idx + 1].lower()
        else:
            print("Error: --algo requires an algorithm name")
            sys.exit(1)
    
    all_algorithms = {
        'nsga2': NSGA2Repair(pop_size=20),
        'mospo': MOSPORepair(pop_size=20),
        'gradient': GradientRepair(learning_rate=0.001, n_weights=7, n_iterations_per_weight=100)
    }
    
    if algorithm_name:
        if algorithm_name not in all_algorithms:
            print(f"Error: Unknown algorithm '{algorithm_name}'")
            print(f"Available algorithms: {', '.join(all_algorithms.keys())}")
            sys.exit(1)
        algorithms = [all_algorithms[algorithm_name]]
    else:
        algorithms = list(all_algorithms.values())
    
    print(f"\nRunning comparison for model: {model_name}{' (ResNet)' if resnet_flag else ''}, seed: {seed}")

    n_gen = 100
    pop_size = 100
    results = run_comparison(
        model_name=model_name,
        algorithms=algorithms,
        n_gen=n_gen,
        pop_size=pop_size,
        seed=seed,
        resnet_flag=resnet_flag
    )

