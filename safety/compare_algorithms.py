"""
Compare different repair algorithms on safety task
Run: python compare_algorithms.py a19

Supported algorithms (based on reviewer feedback):
- NSGA-II: Recommended by Reviewer 1 for 2-objective problems
- MOEA/D: Suggested by Reviewer 2 as decomposition-based alternative
- Gradient: Suggested by Reviewer 2 as gradient-based alternative
"""

import os
import sys
import time
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K
from repair_algorithms import (
    SafetyRepairProblem, 
    NSGA2Repair, 
    MOEADRepair,
    GradientRepair
)

# Setup TensorFlow
tf.compat.v1.disable_eager_execution()
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))


def get_layer_index(model):
    """Get layer indices"""
    conv, dense, batch = [], [], []
    j = 0
    for layer in model.layers:
        weights = layer.get_weights()
        lname = type(layer).__name__
        if weights:
            if 'Batch' in lname:
                batch.append(j)
                j += 4
            elif 'Conv' in lname:
                conv.append(j)
                j += 2
            elif 'Dense' in lname:
                dense.append(j)
                j += 2
    return conv, dense, batch


def load_data(model_name):
    """Load safety data"""
    with h5py.File(f'../data/{model_name}/counterexample.h5', 'r') as hf:
        ct_x = hf['counterexample'][:]
    with h5py.File(f'../data/{model_name}/drawndown.h5', 'r') as hf:
        dr_x = hf['drawndown'][:]
    return ct_x, dr_x


def select_data(X, num_examples):
    """Randomly select data"""
    random_indices = np.random.choice(len(X), min(num_examples, len(X)), replace=False)
    return X[random_indices]


def prepare_repair_data(model_name, loc_dir, repair_ratio=0.01, num_examples=1024):
    """Prepare data for repair"""
    # Load model
    model_path = f'../model/{model_name}.h5'
    model = load_model(model_path, compile=False)
    
    # Get layer indices
    convs, denses, _ = get_layer_index(model)
    
    # Load localization results
    with open(os.path.join(loc_dir, 'players.txt')) as f:
        players = f.read().split(',')
    with open(os.path.join(loc_dir, 'safety_vals.txt')) as f:
        safety_vals = list(map(float, f.read().split(',')))
    
    # Select top neurons
    dic_sr = dict(zip(players, safety_vals))
    dic_players = {player: dic_sr[player] for player in players}
    
    import math
    repair_num = math.ceil(len(players) * repair_ratio)
    selected_players = [x[0] for x in sorted(dic_players.items(), key=lambda x: x[1], reverse=True)[:repair_num]]
    
    # Extract initial weights
    init_weights = []
    shapes = []
    for player in selected_players:
        if player.split('_')[0] == "conv":
            weight_index = convs[int(player.split('_')[1])]
            weight = model.get_weights()[weight_index][..., int(player.split('_')[-1])]
        else:
            weight_index = denses[int(player.split('_')[1])]
            weight = model.get_weights()[weight_index][..., int(player.split('_')[-1])]
        shapes.append(weight.shape)
        init_weights.append(weight.flatten())
    
    init_weights = np.concatenate(init_weights)
    
    # Load data
    ct_x, dr_x = load_data(model_name)
    ct_x = select_data(ct_x, num_examples)
    dr_x = select_data(dr_x, num_examples)
    
    return {
        'model_path': model_path,
        'init_weights': init_weights,
        'selected_players': selected_players,
        'shapes': shapes,
        'convs': convs,
        'denses': denses,
        'ct_x': ct_x,
        'dr_x': dr_x
    }


def run_comparison(model_name, algorithms, n_gen=100, pop_size=20, seed=None):
    """Run comparison of different algorithms
    
    Args:
        model_name: Name of the model to repair
        algorithms: List of algorithm instances to compare
        n_gen: Number of generations
        pop_size: Population size
        seed: Random seed for reproducibility (optional)
    """
    
    print(f"\n{'='*60}")
    print(f"Comparing Repair Algorithms on {model_name}")
    print(f"{'='*60}\n")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"Random seed: {seed}\n")
    
    # Prepare data
    loc_dir = f'./loc_result/{model_name}'
    data = prepare_repair_data(model_name, loc_dir)
    
    # Thresholds (relaxed from original for better convergence)
    thresholds = {
        'a19': {'accuracy': 0.999999999, 'safety': 0.00000001},
        'a29': {'accuracy': 0.999999999, 'safety': 0.00000001},
        'a33': {'accuracy': 0.999999999, 'safety': 0.00000001}
    }
    
    # Results storage
    results = {}
    
    # Run each algorithm
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
            problem = SafetyRepairProblem(
                model_path=data['model_path'],
                init_weights=np.tile(data['init_weights'], (pop_size, 1)),
                selected_players=data['selected_players'],
                shapes=data['shapes'],
                convs=data['convs'],
                denses=data['denses'],
                ct_x=data['ct_x'],
                dr_x=data['dr_x'],
                model_name=model_name,
                threshold_acc=thresholds[model_name]['accuracy'],
                threshold_safety=thresholds[model_name]['safety']
            )
            
            # Initial population
            init_population = np.tile(data['init_weights'], (pop_size, 1))
            init_population += np.random.uniform(-0.05, 0.05, init_population.shape)
            
            # Run repair
            res = algo.repair(problem, init_population, n_gen)
            elapsed_time = time.time() - start_time
            
            # Extract results
            if hasattr(res, 'F') and hasattr(res, 'X'):
                if res.F.ndim == 1:
                    final_f = res.F.reshape(1, -1)
                    final_x = res.X.reshape(1, -1)
                else:
                    final_f = res.F
                    final_x = res.X
                
                # Filter solutions by constraints
                threshold_acc = thresholds[model_name]['accuracy']
                threshold_safety = thresholds[model_name]['safety']
                
                valid_indices = []
                for i, f in enumerate(final_f):
                    safety_rate = f[0]
                    accuracy = -f[1]  # Convert back to positive
                    if accuracy >= threshold_acc and safety_rate <= threshold_safety:
                        valid_indices.append(i)
                
                # Keep only solutions that satisfy constraints. If none, mark as failure
                if valid_indices:
                    filtered_f = final_f[valid_indices]
                    filtered_x = final_x[valid_indices]
                    print(f"  ✓ Filtered Pareto front: {len(valid_indices)}/{len(final_f)} solutions satisfy constraints")
                else:
                    print(f"  ✗ No solution satisfies constraints (acc>={threshold_acc}, safety<={threshold_safety}). Marking as FAILED and skipping save.")
                    results[algo.name] = {
                        'success': False,
                        'error': 'No valid solutions under constraints',
                        'time': elapsed_time
                    }
                    continue
                
                # Deduplication: Remove duplicate solutions (same objective values)
                if len(filtered_f) > 0:
                    unique_objectives = {}
                    unique_indices = []
                    
                    for i, f in enumerate(filtered_f):
                        safety_rate = f[0]
                        accuracy = -f[1]  # Convert back to positive
                        key = (round(safety_rate, 8), round(accuracy, 8))
                        
                        if key not in unique_objectives:
                            unique_objectives[key] = i
                            unique_indices.append(i)
                    
                    if len(unique_indices) < len(filtered_f):
                        print(f"  Deduplication: {len(filtered_f)} → {len(unique_indices)} unique solutions")
                        filtered_f = filtered_f[unique_indices]
                        filtered_x = filtered_x[unique_indices]
                    else:
                        print(f"  No duplicate solutions found")
                
                results[algo.name] = {
                    'success': True,
                    'time': elapsed_time,
                    'final_safety': float(filtered_f[:, 0].min()) if filtered_f.shape[1] > 0 else None,
                    'final_accuracy': float(-filtered_f[:, 1].max()) if filtered_f.shape[1] > 1 else None,
                    'n_solutions': len(filtered_f),
                    'n_original_solutions': len(final_f),
                    'n_valid_solutions': len(valid_indices),
                    'pareto_front': filtered_f.tolist() if filtered_f.shape[0] < 100 else None,
                    'pareto_solutions': filtered_x  # Store solution weights
                }
                
                print(f"\n{algo.name} Results:")
                print(f"  Time: {elapsed_time:.2f}s")
                print(f"  Original Pareto front: {len(final_f)} solutions")
                print(f"  Filtered Pareto front: {len(filtered_f)} solutions")
                print(f"  Best Safety Rate: {results[algo.name]['final_safety']:.6f}")
                print(f"  Best Accuracy: {results[algo.name]['final_accuracy']:.6f}")
            else:
                results[algo.name] = {
                    'success': False,
                    'error': 'No results returned',
                    'time': elapsed_time
                }
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            results[algo.name] = {
                'success': False,
                'error': str(e),
                'time': elapsed_time
            }
            print(f"\n{algo.name} FAILED: {e}")
    
    # Save results and Pareto fronts
    output_dir = f'./comparison_results/{model_name}/{seed}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Pareto front weights for each algorithm
    for algo_name, result in results.items():
        if result['success'] and 'pareto_solutions' in result:
            # Determine algorithm directory name
            algo_dir_name = algo_name.lower().replace('-', '').replace('/', '')
            if 'nsga' in algo_dir_name:
                algo_dir_name = 'nsga2'
            elif 'moea' in algo_dir_name:
                algo_dir_name = 'moead'
            elif 'gradient' in algo_dir_name:
                algo_dir_name = 'gradient'
            
            # Create algorithm-specific directory
            algo_output_dir = os.path.join(output_dir, algo_dir_name)
            pareto_dir = os.path.join(algo_output_dir, 'pareto_front')
            os.makedirs(pareto_dir, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Saving {algo_name} Pareto front solutions...")
            print(f"{'='*60}")
            
            # Load base model
            base_model = load_model(data['model_path'], compile=False)
            
            # Save each Pareto solution
            pareto_solutions = result['pareto_solutions']
            pareto_front = np.array(result['pareto_front'])
            pareto_records = []
            
            for idx in range(len(pareto_solutions)):
                # Create model with repaired weights
                save_model = load_model(data['model_path'], compile=False)
                
                # Apply repaired weights
                weights = save_model.get_weights()
                start_idx = 0
                for player, shape in zip(data['selected_players'], data['shapes']):
                    weight_size = np.prod(shape)
                    new_weight = pareto_solutions[idx][start_idx:start_idx + weight_size].reshape(shape)
                    
                    if player.split('_')[0] == "conv":
                        weight_index = data['convs'][int(player.split('_')[1])]
                        weights[weight_index][..., int(player.split('_')[-1])] = new_weight
                    else:
                        weight_index = data['denses'][int(player.split('_')[1])]
                        weights[weight_index][..., int(player.split('_')[-1])] = new_weight
                    
                    start_idx += weight_size
                
                save_model.set_weights(weights)
                
                # Save weights
                weight_path = os.path.join(pareto_dir, f'weights_{idx}.h5')
                save_model.save_weights(weight_path)
                
                # Record performance (safety_rate, accuracy)
                safety_rate = float(pareto_front[idx, 0])
                accuracy = float(-pareto_front[idx, 1])  # Negate back to original
                
                pareto_records.append({
                    'index': idx,
                    'train_safety': safety_rate,
                    'train_accuracy': accuracy
                })
                
                print(f"  Solution {idx:3d}: safety={safety_rate:.8f}, accuracy={accuracy:.8f}")
            
            # Save Pareto front metadata
            pareto_file = os.path.join(pareto_dir, 'pareto_front.txt')
            with open(pareto_file, 'w') as f:
                f.write("index,safety_rate,accuracy\n")
                for record in pareto_records:
                    f.write(f"{record['index']},{record['train_safety']:.10f},{record['train_accuracy']:.10f}\n")
            
            print(f"\n  Saved {len(pareto_records)} solutions to: {pareto_dir}")
            
            # Remove pareto_solutions from results (too large for JSON)
            del result['pareto_solutions']
    
    # Save comparison results JSON
    output_file = os.path.join(output_dir, f'comparison_{int(time.time())}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Comparison results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Algorithm':<20} {'Time(s)':<12} {'Safety Rate':<15} {'Accuracy':<12} {'Status':<10}")
    print("-" * 70)
    for algo_name, result in results.items():
        if result['success']:
            print(f"{algo_name:<20} {result['time']:<12.2f} {result['final_safety']:<15.6f} {result['final_accuracy']:<12.6f} {'Success':<10}")
        else:
            print(f"{algo_name:<20} {result['time']:<12.2f} {'N/A':<15} {'N/A':<12} {'Failed':<10}")
    
    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_algorithms.py <model_name> [seed] [algorithm]")
        print("\nExamples:")
        print("  python compare_algorithms.py a19                    # Run all algorithms, no seed")
        print("  python compare_algorithms.py a19 5                  # Run all algorithms, seed=5")
        print("  python compare_algorithms.py a19 5 nsga2            # Run NSGA-II only, seed=5")
        print("  python compare_algorithms.py a19 5 moead            # Run MOEA/D only, seed=5")
        print("  python compare_algorithms.py a19 5 gradient         # Run Gradient only, seed=5")
        print("\nAvailable algorithms (based on reviewer feedback):")
        print("  - nsga2    : NSGA-II (Reviewer 1 recommendation)")
        print("  - moead    : MOEA/D (Reviewer 2 suggestion)")
        print("  - gradient : Gradient-based Adam (Reviewer 2 suggestion)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Parse seed and algorithm
    seed = None
    algorithm_name = None
    
    if len(sys.argv) > 2:
        # Try to parse as seed (integer)
        try:
            seed = int(sys.argv[2])
            if len(sys.argv) > 3:
                algorithm_name = sys.argv[3].lower()
        except ValueError:
            # Not a seed, must be algorithm name
            algorithm_name = sys.argv[2].lower()
    
    # Define all available algorithms (only the 3 recommended by reviewers)
    all_algorithms = {
        'nsga2': NSGA2Repair(pop_size=20),
        'moead': MOEADRepair(n_neighbors=15),  # Automatically adapts to population size
        'gradient': GradientRepair(learning_rate=0.01, n_weights=5)
    }
    
    # Select algorithms to run
    if algorithm_name:
        if algorithm_name not in all_algorithms:
            print(f"Error: Unknown algorithm '{algorithm_name}'")
            print(f"Available algorithms: {', '.join(all_algorithms.keys())}")
            sys.exit(1)
        algorithms = [all_algorithms[algorithm_name]]
        print(f"\nRunning single algorithm: {algorithm_name.upper()}")
    else:
        # Run all three algorithms
        algorithms = list(all_algorithms.values())
        print(f"\nRunning all algorithms")
    
    print(f"Model: {model_name}")
    print(f"Seed: {seed if seed is not None else 'None (random)'}")
    print(f"Algorithms: {[algo.name for algo in algorithms]}")
    
    # Run comparison
    results = run_comparison(
        model_name=model_name,
        algorithms=algorithms,
        n_gen=50,
        pop_size=20,
        seed=seed
    )

