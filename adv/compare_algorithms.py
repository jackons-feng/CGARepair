"""
Compare Different Repair Algorithms for Adversarial Robustness
Runs multiple repair algorithms and compares their performance

Supported algorithms (based on reviewer feedback):
- NSGA-II: Recommended by Reviewer 1 for 2-objective problems
- MOEA/D: Suggested by Reviewer 2 as decomposition-based alternative
- Gradient: Suggested by Reviewer 2 as gradient-based alternative

Usage:
    python compare_algorithms.py --epsilon <eps> --model_name <model> [--algorithms ALGO1,ALGO2,...] [--pop_size N] [--n_gen N]

Examples:
    python compare_algorithms.py --epsilon 0.05 --model_name mnist                           # Run all algorithms
    python compare_algorithms.py --epsilon 0.05 --model_name mnist --algorithms nsga2,moead  # Run specific algorithms
    python compare_algorithms.py --epsilon 0.05 --model_name mnist --pop_size 100 --n_gen 1000  # Custom parameters
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K


from repair_algorithms import (
    NSGA2Repair, MOEADRepair, GradientRepair, build_layer_lookup, parse_player_identifier
)

def load_adv(model_name, epsilon):
    """Load adversarial and clean data (train set only)
    
    参考 load_data.py 的 adv 加载逻辑，只使用 train 标记的数据
    """
    if model_name == 'mnist':
        # 加载对抗样本数据从txt文件
        adv_data_file = os.path.join('..', 'data', 'adv', 'mnist', 'train', f'epsilon{epsilon}.txt')
        adv_label_file = os.path.join('..', 'data', 'adv', 'mnist', 'train', 'label.txt')
        
        # 加载对抗数据
        with open(adv_data_file, 'r') as f:
            data_lines = f.readlines()
        sr_x = []
        for line in data_lines:
            sr_x.append(np.array(line.strip().split(','), dtype=float).reshape((28, 28, 1)))
        sr_x = np.array(sr_x, dtype='float32')
        
        # 加载标签
        with open(adv_label_file, 'r') as f:
            label_lines = f.readlines()
        sr_y = np.array([int(line.strip()) for line in label_lines], dtype=int)
        
        # 加载干净数据(从keras.datasets)
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
        
        # 只取与攻击数据相同数量的干净数据
        acc_x = x_train[:len(sr_x)]
        acc_y = y_train[:len(sr_x)]
        
        print(f"Loaded MNIST adversarial data from txt files (train set)")
        print(f"Shapes -> sr_x:{sr_x.shape}, sr_y:{sr_y.shape}, acc_x:{acc_x.shape}, acc_y:{acc_y.shape}")
    
    else:
        # 加载CIFAR-10对抗样本从npy文件
        adv_data_file = os.path.join('..', 'data', 'adv', 'cifar10', 'train', f'epsilon{epsilon}.npy')
        adv_label_file = os.path.join('..', 'data', 'adv', 'cifar10', 'train', 'label.npy')
        
        def preprocess_cifar10(x):
            """Prepares a batch of CIFAR-10 images (0-1 float) for the model."""
            x = x.astype('float32')
            x -= np.array([0.4914, 0.4822, 0.4465])
            x /= np.array([0.247, 0.243, 0.261])
            return x
        
        # 加载对抗数据
        x_adv_flat = np.load(adv_data_file)
        sr_x = x_adv_flat.reshape((-1, 32, 32, 3))
        sr_x = preprocess_cifar10(sr_x)
        sr_y = np.load(adv_label_file)
        
        # 加载干净数据
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
        acc_x = preprocess_cifar10(x_train.astype('float32') / 255.0)
        acc_y = y_train.flatten()
        
        # 只取与攻击数据相同数量的干净数据
        acc_x = acc_x[:len(sr_x)]
        acc_y = acc_y[:len(sr_x)]
        
        print(f"Loaded CIFAR-10 adversarial data from npy files (train set)")
        print(f"Shapes -> sr_x:{sr_x.shape}, sr_y:{sr_y.shape}, acc_x:{acc_x.shape}, acc_y:{acc_y.shape}")

    return sr_x, sr_y, acc_x, acc_y


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
def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def select_neurons_to_repair(loc_dic, repair_percentage=0.01):
    """Select neurons to repair based on Shapley values
    
    Args:
        loc_dic: Directory containing localization results
        repair_percentage: Percentage of neurons to repair
        
    Returns:
        Tuple of (selected_players, shapes, all_players)
    """
    # Load localization results
    with open(os.path.join(loc_dic, 'players.txt')) as f1, \
         open(os.path.join(loc_dic, 'adv_vals.txt')) as f2:
        players = f1.read().split(',')
        vals = list(map(float, f2.read().split(',')))
    
    # Create dictionary of player: shapley_value
    dic_players = dict(zip(players, vals))
    
    # Select top neurons based on Shapley values
    repair_num = max(4, min(11, math.ceil(len(players) * repair_percentage)))
    selected_players = [
        x[0] for x in sorted(dic_players.items(), key=lambda x: x[1], reverse=True)[:repair_num]
    ]
    
    return selected_players, players




def filter_valid_players(selected_players, convs, denses, layer_lookup=None):
    """Filter out player identifiers that reference missing layers."""
    valid_players = []
    for player in selected_players:
        try:
            layer_type, layer_pos, _ = parse_player_identifier(player, layer_lookup)
            if layer_type == "conv":
                if 0 <= layer_pos < len(convs):
                    valid_players.append(player)
                else:
                    raise IndexError(f"conv layer index {layer_pos} out of range")
            else:
                if 0 <= layer_pos < len(denses):
                    valid_players.append(player)
                else:
                    raise IndexError(f"dense layer index {layer_pos} out of range")
        except (ValueError, IndexError) as exc:
            print(f"  Skipping player '{player}': {exc}")
    return valid_players


def extract_neuron_shapes(model, selected_players, convs, denses, layer_lookup=None):
    """Extract weight shapes for selected neurons
    
    Args:
        model: Keras model
        selected_players: List of neuron identifiers
        convs: List of convolutional layer indices
        denses: List of dense layer indices
        
    Returns:
        List of weight shapes
    """
    shapes = []
    weights = model.get_weights()
    
    for player in selected_players:
        layer_type, layer_pos, neuron_idx = parse_player_identifier(player, layer_lookup)
        if layer_type == "conv":
            weight_index = convs[layer_pos]
        else:
            weight_index = denses[layer_pos]
        weight = weights[weight_index][..., neuron_idx]
        shapes.append(weight.shape)
    
    return shapes


def save_pareto_front(result, save_dir, algorithm_name, threshold_acc=0.971, threshold_adv=0.81, 
                      model_path=None, selected_players=None, shapes=None, convs=None, denses=None,
                      layer_lookup=None):
    """Save Pareto front solutions (filtered by constraints)
    
    Args:
        result: Optimization result object
        save_dir: Directory to save results
        algorithm_name: Name of the algorithm
        threshold_acc: Minimum accuracy threshold
        threshold_adv: Minimum robustness threshold
        model_path: Path to model (for weight reconstruction)
        selected_players: List of selected neurons
        shapes: Weight shapes for selected neurons
        convs: Conv layer indices
        denses: Dense layer indices
    """
    pareto_dir = os.path.join(save_dir, 'pareto_front')
    if not os.path.exists(pareto_dir):
        os.makedirs(pareto_dir)
    
    # Save ALL solutions for plotting; do not filter by constraints here
    if not hasattr(result, 'X') or result.X is None or len(result.X) == 0:
        print(f"  Warning: No solutions found in result, creating empty Pareto front")
        filtered_X = np.array([]).reshape(0, 0)
        filtered_F = np.array([]).reshape(0, 2)
    else:
        filtered_X = result.X
        if hasattr(result, 'F') and result.F is not None:
            filtered_F = result.F
        else:
            # Create dummy objectives if F is missing
            filtered_F = np.zeros((len(filtered_X), 2))
    
    # Optional: keep dedup step; comment out to preserve all points exactly as returned
    # (Here we preserve all points, including duplicates, for exact plotting)
    
    # Save Pareto front metadata (consistent with adv_moo_2.py format)
    metadata_file = os.path.join(pareto_dir, 'pareto_front.txt')
    with open(metadata_file, 'w') as f:
        f.write("index,train_sr_rate,train_accuracy\n")
        
        if len(filtered_F) > 0:
            # For multi-objective algorithms
            if len(filtered_F.shape) > 1 and filtered_F.shape[1] >= 2:
                for i, (f1, f2) in enumerate(filtered_F):
                    sr_rate = -f1  # Robustness
                    accuracy = -f2  # Clean accuracy
                    f.write(f"{i},{sr_rate:.10f},{accuracy:.10f}\n")
            elif hasattr(filtered_F, '__len__') and len(filtered_F) > 0:
                for i, f in enumerate(filtered_F):
                    sr_rate = -f if isinstance(f, (int, float)) else -f[0] if hasattr(f, '__len__') else 0.0
                    f.write(f"{i},{sr_rate:.10f},0.0\n")
    
    print(f"  Pareto front metadata saved to: {metadata_file}")
    print(f"  Saved {len(filtered_F)} unique solutions")
    
    # Save weights for each solution (CRITICAL for adv_test_2.py)
    if model_path and selected_players and shapes and convs and denses and len(filtered_X) > 0:
        print(f"  Saving weights for {len(filtered_X)} solutions...")
        from tensorflow.keras.models import load_model
        
        # Load base model
        model = load_model(model_path, compile=False)
        base_weights = model.get_weights()
        
        # Build player indices mapping
        player_indices = []
        for player in selected_players:
            layer_type, layer_pos, neuron_idx = parse_player_identifier(player, layer_lookup)
            if layer_type == "conv":
                layer_idx = convs[layer_pos]
            else:
                layer_idx = denses[layer_pos]
            player_indices.append((layer_idx, neuron_idx))
        
        # Build weight slices
        weight_slices = []
        begin_index = 0
        for shape in shapes:
            end_index = begin_index + np.prod(shape)
            weight_slices.append((begin_index, end_index, shape))
            begin_index = end_index
        
        # Save each solution
        for i, x in enumerate(filtered_X):
            # Reconstruct full weights from decision variables
            weights = [w.copy() if isinstance(w, np.ndarray) else w for w in base_weights]
            
            for (layer_idx, neuron_idx), (begin_idx, end_idx, shape) in zip(player_indices, weight_slices):
                new_weight = x[begin_idx:end_idx].reshape(shape)
                if len(weights[layer_idx].shape) == 4:  # Conv layer
                    weights[layer_idx][..., neuron_idx] = new_weight
                elif len(weights[layer_idx].shape) == 2:  # Dense layer
                    weights[layer_idx][:, neuron_idx] = new_weight
            
            # Save weights
            model.set_weights(weights)
            weight_file = os.path.join(pareto_dir, f'weights_{i}.h5')
            model.save_weights(weight_file)
        
        print(f" Saved {len(filtered_X)} weight files to: {pareto_dir}")
    else:
        print(f" Warning: Cannot save weights (missing model_path or neuron info)")
        print(f"    adv_test_2.py will not be able to load solutions!")
    
    # Update result object with filtered solutions
    result.F = filtered_F
    result.X = filtered_X
    
    # Note: Individual weight files are saved during optimization
    # in the repair algorithm's _evaluate method


def run_algorithm(algorithm_class, algorithm_name, model_path, selected_players, 
                  shapes, convs, denses, adv_x, adv_y, clean_x, clean_y,
                  save_dir, pop_size, n_gen, layer_lookup=None, **kwargs):
    """Run a single repair algorithm
    
    Args:
        algorithm_class: Algorithm class to instantiate
        algorithm_name: Name of the algorithm
        model_path: Path to the model
        selected_players: List of neurons to repair
        shapes: Weight shapes for selected neurons
        convs: Convolutional layer indices
        denses: Dense layer indices
        adv_x: Adversarial training data
        adv_y: Adversarial training labels
        clean_x: Clean training data
        clean_y: Clean training labels
        save_dir: Directory to save results
        pop_size: Population size
        n_gen: Number of generations
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Tuple of (result, elapsed_time)
    """
    print_separator()
    print(f"Running {algorithm_name}...")
    print_separator()
    
    start_time = time.time()
    
    # Create algorithm instance
    algo = algorithm_class(
        model_path=model_path,
        selected_players=selected_players,
        shapes=shapes,
        convs=convs,
        denses=denses,
        adv_x=adv_x,
        adv_y=adv_y,
        clean_x=clean_x,
        clean_y=clean_y,
        layer_lookup=layer_lookup,
    )
    
    # Run repair

    result = algo.repair(
        pop_size=pop_size,
        n_gen=n_gen,
        save_dir=os.path.join(save_dir, 'pareto_front'),
        **kwargs
    )
    
    elapsed_time = time.time() - start_time
    
    # Save Pareto front (with constraint filtering)
    threshold_acc = kwargs.get('threshold_acc', 0.971)
    threshold_adv = kwargs.get('threshold_adv', 0.81)
    save_pareto_front(result, save_dir, algorithm_name, threshold_acc, threshold_adv,
                      model_path=model_path, selected_players=selected_players, 
                      shapes=shapes, convs=convs, denses=denses, layer_lookup=layer_lookup)
    
    # Save algorithm info
    info_file = os.path.join(save_dir, 'algorithm_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"algorithm={algorithm_name}\n")
        f.write(f"pop_size={pop_size}\n")
        f.write(f"n_gen={n_gen}\n")
        f.write(f"elapsed_time={elapsed_time:.2f}\n")
        f.write(f"num_solutions={len(result.X) if hasattr(result, 'X') else 0}\n")
        for key, value in kwargs.items():
            f.write(f"{key}={value}\n")
    
    print(f"\n{algorithm_name} completed in {elapsed_time:.2f} seconds")
    print(f"  Results saved to: {save_dir}")
    
    return result, elapsed_time


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare repair algorithms for adversarial robustness')
    parser.add_argument('--epsilon', type=str, help='Adversarial perturbation epsilon')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--algorithms', type=str, default='all',
                        help='Comma-separated list of algorithms (nsga2,moead,gradient) or "all"')
    parser.add_argument('--pop_size', type=int, default=100,
                        help='Population size (default: 100)')
    parser.add_argument('--n_gen', type=int, default=1,
                        help='Number of generations (default: 100)')
    parser.add_argument('--repair_pct', type=float, default=0.01,
                        help='Percentage of neurons to repair (default: 0.01)')
    parser.add_argument('--adv_samples', type=int, default=10000,
                        help='Number of adversarial samples (default: all)')
    parser.add_argument('--clean_samples', type=int, default=10000,
                        help='Number of clean samples (default: all)')

    args = parser.parse_args()

    epsilon = args.epsilon
    model_name = args.model_name
    pop_size = args.pop_size
    n_gen = args.n_gen
    repair_percentage = args.repair_pct
    data_name = 'train'
    
    # Parse algorithms
    if args.algorithms.lower() == 'all':
        algorithms = ['nsga2', 'moead', 'gradient']
    else:
        algorithms = [a.strip().lower() for a in args.algorithms.split(',')]
    
    # Validate algorithms
    valid_algorithms = ['nsga2', 'moead', 'gradient']
    for algo in algorithms:
        if algo not in valid_algorithms:
            print(f"Error: Unknown algorithm '{algo}'")
            print(f"Valid algorithms: {', '.join(valid_algorithms)}")
            sys.exit(1)

    # Initialize TensorFlow
    TIME_START = time.time()
    tf.compat.v1.disable_eager_execution()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

    # Print configuration
    print_separator()
    print(f"Adversarial Repair - Algorithm Comparison")
    print_separator()
    print(f"Configuration:")
    print(f"  Epsilon: {epsilon}")
    print(f"  Algorithms: {', '.join([a.upper() for a in algorithms])}")
    print(f"  Population size: {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Repair percentage: {repair_percentage * 100:.2f}%")
    print(f"  Data: {data_name}")

    # Paths

    model_path = f'../model/{model_name}_adv.h5'
    loc_dic = f'loc_result/{model_name}/{epsilon}'
    
    # Check if localization results exist
    if not os.path.exists(loc_dic):
        print(f"\nError: Localization results not found at {loc_dic}")
        print("Please run RQ1 and read.py first to generate localization results.")
        sys.exit(1)

    # Load model
    print_separator('-')
    print("Loading model and data...")
    model = load_model(model_path, compile=False)
    
    # Get layer indices and name lookup for resolving players
    convs, denses,_= get_layer_index(model)
    layer_lookup = build_layer_lookup(model)
    
    # Select neurons to repair
    selected_players, all_players = select_neurons_to_repair(loc_dic, repair_percentage)
    selected_players = filter_valid_players(selected_players, convs, denses, layer_lookup)
    
    if not selected_players:
        raise RuntimeError("No valid neurons to repair after filtering localization results.")
    shapes = extract_neuron_shapes(model, selected_players, convs, denses, layer_lookup)
    
    print(f"  Total neurons: {len(all_players)}")
    print(f"  Selected for repair: {len(selected_players)} ({repair_percentage*100:.2f}%)")
    print(f"  Total parameters to optimize: {sum(np.prod(s) for s in shapes)}")

    # Load data
    print_separator('-')
    print(f"Loading {data_name} data...")

    adv_x, adv_y, clean_x, clean_y = load_adv(model_name, epsilon)
    
    # Subsample if requested
    if args.adv_samples and args.adv_samples < len(adv_x):
        indices = np.random.choice(len(adv_x), args.adv_samples, replace=False)
        adv_x = adv_x[indices]
        adv_y = adv_y[indices]
    
    if args.clean_samples and args.clean_samples < len(clean_x):
        indices = np.random.choice(len(clean_x), args.clean_samples, replace=False)
        clean_x = clean_x[indices]
        clean_y = clean_y[indices]
    
    print(f"  Adversarial examples: {len(adv_x)}")
    print(f"  Clean examples: {len(clean_x)}")

    # Set thresholds based on model (matching adv_moo_2.py)
    if model_name in ['mnist', 'mnist_adv']:
        threshold_acc = 0.971
        threshold_adv = 0.81
    else:
        threshold_acc = 0.921
        threshold_adv = 0.39
    
    print(f"  Thresholds: acc >= {threshold_acc}, robust >= {threshold_adv}")

    # Algorithm configurations
    algorithm_configs = {
        'nsga2': {
            'class': NSGA2Repair,
            'name': 'NSGA-II',
            'kwargs': {
                'threshold_acc': threshold_acc,
                'threshold_adv': threshold_adv
            }
        },
        'moead': {
            'class': MOEADRepair,
            'name': 'MOEA/D',
            'kwargs': {
                'n_neighbors': 20,
                'threshold_acc': threshold_acc,
                'threshold_adv': threshold_adv
            }
        },
        'gradient': {
            'class': GradientRepair,
            'name': 'Gradient-Adam',
            'kwargs': {
                'threshold_acc': threshold_acc,
                'threshold_adv': threshold_adv,
                'learning_rate': 0.001
            }
        }
    }

    # Run algorithms
    results = {}
    
    for algo_key in algorithms:
        config = algorithm_configs[algo_key]
        
        # Clear session before each algorithm to avoid graph conflicts
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)
        
        # Create save directory
        save_dir = f'comparison_results/{model_name}/{epsilon}/{algo_key}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Run algorithm
        try:
            result, elapsed = run_algorithm(
                algorithm_class=config['class'],
                algorithm_name=config['name'],
                model_path=model_path,
                selected_players=selected_players,
                shapes=shapes,
                convs=convs,
                denses=denses,
                adv_x=adv_x,
                adv_y=adv_y,
                clean_x=clean_x,
                clean_y=clean_y,
                save_dir=save_dir,
                pop_size=pop_size,
                n_gen=n_gen,
                layer_lookup=layer_lookup,
                **config['kwargs']
            )
            
            results[algo_key] = {
                'result': result,
                'elapsed': elapsed,
                'save_dir': save_dir
            }
        
        except Exception as e:
            print(f"\nError running {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print_separator()
    print("Comparison Summary")
    print_separator()
    
    if results:
        print(f"{'Algorithm':<15} {'Time (s)':<12} {'Solutions':<12} {'Save Directory'}")
        print('-' * 80)
        
        for algo_key, data in results.items():
            name = algorithm_configs[algo_key]['name']
            elapsed = data['elapsed']
            n_solutions = len(data['result'].X) if hasattr(data['result'], 'X') else 0
            save_dir = data['save_dir']
            
            print(f"{name:<15} {elapsed:<12.2f} {n_solutions:<12} {save_dir}")
        
        # Save comparison summary
        summary_file = f'comparison_results/{model_name}/{epsilon}/comparison_summary.txt'
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write(f"epsilon={epsilon}\n")
            f.write(f"pop_size={pop_size}\n")
            f.write(f"n_gen={n_gen}\n")
            f.write(f"repair_percentage={repair_percentage}\n")
            f.write(f"data={data_name}\n")
            f.write(f"adv_samples={len(adv_x)}\n")
            f.write(f"clean_samples={len(clean_x)}\n")
            f.write(f"\n")
            
            for algo_key, data in results.items():
                name = algorithm_configs[algo_key]['name']
                f.write(f"{algo_key}_time={data['elapsed']:.2f}\n")
                f.write(f"{algo_key}_solutions={len(data['result'].X) if hasattr(data['result'], 'X') else 0}\n")
        
        print(f"\nComparison summary saved to: {summary_file}")
    else:
        print("No algorithms completed successfully.")

    # Total execution time
    total_time = time.time() - TIME_START
    print_separator()
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print_separator()


if __name__ == '__main__':
    main()