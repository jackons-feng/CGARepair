"""
Repair algorithms for backdoor task
Supports: NSGA-II, MOSPO, Gradient-based (Adam)

Objectives:
- Backdoor success rate (SR): minimize
- Clean accuracy (ACC): maximize

Note: NSGA-III removed as reviewers suggested NSGA-II is more appropriate for 2-objective problems.
CMA-ES not included as it was not mentioned by reviewers.
"""

import numpy as np
import os
import h5py
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions


class BackdoorRepairProblem(Problem):
    """Multi-objective optimization problem for backdoor repair
    
    Optimized version following backdoor_moo_2.py:
    - Model reuse (avoid repeated loading)
    - Weight caching
    - Pre-computed indices
    - Perfect solution tracking
    """
    
    def __init__(self, model_path, init_weights, selected_players, shapes, 
                 convs, denses, data_x, data_y, data_p, model_name, threshold_acc=0.963, threshold_sr=0.00001):
        super().__init__(
            n_var=len(init_weights[0]),
            n_obj=2,  # Objectives (returned here as [SR, -ACC] for NSGA-II compatibility)
            n_ieq_constr=0,  # No constraints (MOEA/D/NSGA-II don't use explicit constraints here)
            xl=-5.0, 
            xu=5.0,
            vtype=float
        )
        self.model_path = model_path
        self.init_weights_template = init_weights
        self.selected_players = selected_players
        self.shapes = shapes
        self.convs = convs
        self.denses = denses
        self.data_x = data_x
        self.data_y = data_y
        self.data_p = data_p
        self.model_name = model_name
        self.threshold_acc = threshold_acc
        self.threshold_sr = threshold_sr
        
        # Performance optimization: reuse model and cache weights
        self.model = load_model(model_path, compile=False)
        self.base_weights = self.model.get_weights()
        
        # Validate labels on initialization
        # Get num_classes from model output shape
        test_output = self.model.predict(data_x[:1], verbose=0)
        num_classes = test_output.shape[-1]
        
        # Validate and fix labels if necessary
        if np.any(self.data_y < 0) or np.any(self.data_y >= num_classes):
            invalid_labels = np.where((self.data_y < 0) | (self.data_y >= num_classes))[0]
            print(f"Warning: Invalid labels in data_y found: {self.data_y[invalid_labels[:10]]} (expected range [0, {num_classes-1}])")
            print(f"Total invalid labels: {len(invalid_labels)} out of {len(self.data_y)}")
            self.data_y = np.clip(self.data_y, 0, num_classes - 1)
            print(f"Labels clipped to valid range [0, {num_classes-1}]")
        
        # Create mapping from actual layer index to weight index
        # Player names use actual layer indices (from model.layers), not sequential indices
        # This must match the logic in compare_algorithms.py's get_layer_index_mapping()
        actual_to_weight = {}
        weight_idx = 0
        for i, layer in enumerate(self.model.layers):
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
        
        # Pre-compute player indices for faster evaluation
        self.player_indices = []
        for j, player in enumerate(selected_players):
            parts = player.split('_')
            layer_type = parts[0].lower()  # Convert to lowercase to handle both "conv" and "Conv"
            # Normalize layer_type: convert "conv" or "conv2d" to "conv2d"
            if layer_type in ['conv', 'conv2d']:
                layer_type = 'conv2d'
            actual_layer_idx = int(parts[1])  # This is the actual layer index in model.layers
            neuron_idx = int(parts[-1])
            
            # Map from actual layer index to weight index
            key = (layer_type, actual_layer_idx)
            if key not in actual_to_weight:
                raise KeyError(f"Player {player} references {layer_type} layer {actual_layer_idx}, but this layer doesn't exist or has no weights")
            weight_index = actual_to_weight[key]
            
            # Verify that weight_index points to kernel weights, not bias or BatchNorm weights
            weight_tensor = self.base_weights[weight_index]
            actual_layer = self.model.layers[actual_layer_idx]
            actual_layer_name = type(actual_layer).__name__
            
            # Check if the actual layer is BatchNormalization
            if 'BatchNorm' in actual_layer_name or 'BatchNormalization' in actual_layer_name:
                raise ValueError(
                    f"Player {player}: weight_index {weight_index} points to a BatchNormalization layer weight "
                    f"(shape: {weight_tensor.shape}), but player references {layer_type} layer {actual_layer_idx}. "
                    f"This indicates a mapping error. The actual layer at index {actual_layer_idx} is {actual_layer_name}, "
                    f"not a {layer_type} layer. Please check the layer indices in player names."
                )
            
            # Verify that weight tensor is multi-dimensional (kernel weights, not bias)
            if len(weight_tensor.shape) == 1:
                # This is likely a bias weight (1D with shape (output_channels,))
                raise ValueError(
                    f"Player {player}: weight_index {weight_index} points to a 1D weight tensor "
                    f"(shape: {weight_tensor.shape}), which is likely a bias weight, not kernel weights. "
                    f"Kernel weights for {layer_type} layers should be multi-dimensional. "
                    f"This suggests the weight_index mapping is incorrect. "
                    f"Layer: {layer_type} {actual_layer_idx}, neuron_idx: {neuron_idx}"
                )
            
            # Validate neuron/channel index against the last dimension of the corresponding weight tensor
            max_channels = weight_tensor.shape[-1]
            if neuron_idx < 0 or neuron_idx >= max_channels:
                raise IndexError(
                    f"Player {player}: channel/neuron index {neuron_idx} out of range for weight_index "
                    f"{weight_index} with {max_channels} channels (shape={weight_tensor.shape})"
                )
            self.player_indices.append((weight_index, neuron_idx, j))
        
        # Track perfect solutions
        self.perfect_solutions = []

        # Pre-compute weight slicing indices for consistency with other scripts
        self.weight_slices = []
        begin_index = 0
        for shape in self.shapes:
            end_index = begin_index + np.prod(shape)
            self.weight_slices.append((begin_index, end_index))
            begin_index = end_index
        self.evaluation_count = 0
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population with optimized model reuse
        
        Internally, we compute SR and ACC consistent with backdoor_moo_2.value().
        Returned objective vector is [SR_with_penalty, -ACC_with_penalty] to remain compatible
        with pymoo's minimization (used by NSGA-II). Other algorithms (e.g., MOPSO, Gradient)
        convert the second entry back to positive ACC for their own dominance/selection logic.
        """
        f1_list = []  # backdoor success rate (with penalty)
        f2_list = []  # negative accuracy (with penalty for low ACC)
        
        for i in range(x.shape[0]):
            self.evaluation_count += 1
            
            # Reset to base weights (faster than get_weights each time)
            weights = [w.copy() for w in self.base_weights]
            
            # Set neuron weights using pre-computed indices
            begin_index = 0
            weight = x[i]
            for layer_idx, neuron_idx, shape_idx in self.player_indices:
                end_index = begin_index + np.prod(self.shapes[shape_idx])
                try:
                    weights[layer_idx][..., neuron_idx] = weight[begin_index:end_index].reshape(self.shapes[shape_idx])
                except IndexError as e:
                    raise IndexError(
                        f"Weight assignment out of bounds: layer_idx={layer_idx}, neuron_idx={neuron_idx}, "
                        f"weight_shape={weights[layer_idx].shape}, slice_shape={self.shapes[shape_idx]}, "
                        f"begin={begin_index}, end={end_index}"
                    ) from e
                begin_index = end_index
            
            self.model.set_weights(weights)
            
            # Evaluate objectives
            backdoor_rate = self._compute_backdoor_rate(self.model, self.data_x, self.data_p, self.data_y)
            accuracy = self._compute_accuracy(self.model, self.data_x, self.data_y)
            
            # Track perfect solutions
            if accuracy >= self.threshold_acc and backdoor_rate <= self.threshold_sr:
                self.perfect_solutions.append({
                    'weights': x[i].copy(),
                    'backdoor_rate': backdoor_rate,
                    'accuracy': accuracy,
                    'evaluation': self.evaluation_count
                })
                print(f"Perfect solution found at evaluation {self.evaluation_count}!")
            
            # Objectives with penalty for constraint violations
            # Penalty for accuracy below threshold
            acc_penalty = max(0, self.threshold_acc - accuracy) * 10.0
            # Penalty for backdoor above threshold
            backdoor_penalty = max(0, backdoor_rate - self.threshold_sr) * 10.0
            
            f1 = backdoor_rate + backdoor_penalty
            f2 = -accuracy + acc_penalty  # Negative for minimization
            
            f1_list.append(f1)
            f2_list.append(f2)
        
        out['F'] = np.column_stack([f1_list, f2_list])
    
    
    def evaluate_population(self, x):
        """Evaluate a population of solutions
        
        Args:
            x: Population array (numpy array, shape: (n_pop, n_var))
        
        Returns:
            Objective values (numpy array, shape: (n_pop, n_obj))
        """
        out = {}
        self._evaluate(x, out)
        return out['F']
    
    def evaluate_single(self, x):
        """Evaluate a single solution
        
        Args:
            x: Solution array (numpy array, shape: (n_var,))
        
        Returns:
            Objective values (numpy array, shape: (n_obj,))
        """
        x = x.reshape(1, -1)  # Reshape to (1, n_var)
        out = {}
        self._evaluate(x, out)
        return out['F'][0]  # Return first (and only) row
    
    def _compute_backdoor_rate(self, model, images, images_p, labels, batch_size=1024):
        """Compute backdoor success rate"""
        output = self._sess_run(model, model.output, images_p, batch_size)
        pred_p = np.argmax(output, axis=-1)
        
        # Get target label from dataset config
        target_labels = {
            'gtsrb': 33, 'mnist': 3, 'fashion': 3
        }
        target_label = target_labels.get(self.model_name, 3)
        
        # Get num_classes from model output shape
        num_classes = output.shape[-1]
        
        # Convert labels to integer array
        labels = np.asarray(labels, dtype=np.int32)
        
        # Validate predictions and labels are in valid range [0, num_classes-1]
        if np.any(pred_p < 0) or np.any(pred_p >= num_classes):
            invalid_preds = np.where((pred_p < 0) | (pred_p >= num_classes))[0]
            print(f"Warning: Invalid predictions found: {pred_p[invalid_preds[:5]]} (expected range [0, {num_classes-1}])")
            pred_p = np.clip(pred_p, 0, num_classes - 1).astype(np.int32)
        
        if np.any(labels < 0) or np.any(labels >= num_classes):
            invalid_labels = np.where((labels < 0) | (labels >= num_classes))[0]
            print(f"Warning: Invalid labels found: {labels[invalid_labels[:5]]} (expected range [0, {num_classes-1}])")
            labels = np.clip(labels, 0, num_classes - 1).astype(np.int32)
        
        # Validate target_label is in valid range
        target_label = int(target_label)
        if target_label < 0 or target_label >= num_classes:
            print(f"Warning: target_label {target_label} out of range [0, {num_classes-1}], clipping")
            target_label = int(np.clip(target_label, 0, num_classes - 1))
        
        val = (np.sum(pred_p == target_label) - np.sum((labels == target_label) & (pred_p == target_label))) / len(images)
        return val
    
    def _compute_accuracy(self, model, images, labels, batch_size=1024):
        """Compute accuracy"""
        output = self._sess_run(model, model.output, images, batch_size)
        pred_labels = np.argmax(output, axis=-1)
        
        # Get num_classes from model output shape
        num_classes = output.shape[-1]
        
        # Convert labels to integer array
        labels = np.asarray(labels, dtype=np.int32)
        
        # Validate predictions and labels are in valid range [0, num_classes-1]
        if np.any(pred_labels < 0) or np.any(pred_labels >= num_classes):
            invalid_preds = np.where((pred_labels < 0) | (pred_labels >= num_classes))[0]
            print(f"Warning: Invalid predictions found: {pred_labels[invalid_preds[:5]]} (expected range [0, {num_classes-1}])")
            pred_labels = np.clip(pred_labels, 0, num_classes - 1).astype(np.int32)
        
        if np.any(labels < 0) or np.any(labels >= num_classes):
            invalid_labels = np.where((labels < 0) | (labels >= num_classes))[0]
            print(f"Warning: Invalid labels found: {labels[invalid_labels[:5]]} (expected range [0, {num_classes-1}])")
            labels = np.clip(labels, 0, num_classes - 1).astype(np.int32)
        
        corr_predictions = pred_labels == labels
        val = np.sum(corr_predictions) / corr_predictions.shape[0]
        return val
    
    def _sess_run(self, model, variable, images, batch_size=1024):
        """Run model in batches"""
        output = []
        num_batches = int(np.ceil(len(images) / batch_size))
        for batch in range(num_batches):
            batch_idxs = np.arange(
                batch * batch_size, min((batch + 1) * batch_size, len(images))
            )
            input_dic = {model.input: images[batch_idxs]}
            batch_output = K.get_session().run(variable, input_dic)
            
            # Validate output shape
            if len(batch_output.shape) < 2:
                raise ValueError(f"Model output has unexpected shape: {batch_output.shape}")
            if batch_output.shape[-1] <= 0:
                raise ValueError(f"Model output has invalid number of classes: {batch_output.shape[-1]}")
            
            output.append(batch_output)
        
        result = np.concatenate(output, 0)
        
        # Final validation
        if result.shape[-1] <= 0:
            raise ValueError(f"Concatenated output has invalid number of classes: {result.shape[-1]}")
        
        return result


class RepairAlgorithm:
    """Base class for repair algorithms"""
    
    def __init__(self, name):
        self.name = name
        self.results = None
    
    def repair(self, problem, init_population, n_gen, seed=None):
        """Run repair algorithm"""
        raise NotImplementedError


class NSGA2Repair(RepairAlgorithm):
    """NSGA-II repair algorithm
    
    Recommended by Reviewer 1: "This paper uses NSGA-III but only considers 
    two objective functions. Would NSGA-II be more appropriate?"
    """
    
    def __init__(self, pop_size=20):
        super().__init__("NSGA-II")
        self.pop_size = pop_size
    
    def repair(self, problem, init_population, n_gen, seed=None):
        """Run NSGA-II optimization
        
        Args:
            problem: BackdoorRepairProblem instance
            init_population: Initial population (numpy array)
            n_gen: Number of generations
            seed: Random seed for reproducibility
        """
        mutation = PolynomialMutation(prob=0.9, eta=20)
        crossover = UniformCrossover(prob=1.0)
        
        algorithm = NSGA2(
            pop_size=self.pop_size,
            mutation=mutation,
            crossover=crossover,
            sampling=init_population
        )
        
        termination = get_termination('n_gen', n_gen)
        res = minimize(problem, algorithm, termination, seed=seed, verbose=True)
        self.results = res
        return res


def dominates(f1, f2):
    """Check if f1 dominates f2 with mixed objective directions
    
    Args:
        f1: Objective values of solution 1 as [SR, ACC]
        f2: Objective values of solution 2 as [SR, ACC]
    
    Returns:
        True if f1 dominates f2, False otherwise
    """
    # Mixed objectives: minimize SR (index 0), maximize ACC (index 1)
    better_or_equal = (f1[0] <= f2[0]) and (f1[1] >= f2[1])
    strictly_better = (f1[0] < f2[0]) or (f1[1] > f2[1])
    return better_or_equal and strictly_better


def update_archive(archive_solutions, archive_objectives, new_solutions, new_objectives):
    """Update archive with new solutions (non-dominated solutions only)
    
    Args:
        archive_solutions: Current archive solutions (numpy array, shape: (n_archive, n_var))
        archive_objectives: Current archive objectives (numpy array, shape: (n_archive, n_obj))
        new_solutions: New solutions to consider (numpy array, shape: (n_new, n_var))
        new_objectives: New objectives to consider (numpy array, shape: (n_new, n_obj))
    
    Returns:
        Updated archive (solutions, objectives)
    """
    # Normalize inputs to numpy arrays
    archive_solutions = np.asarray(archive_solutions)
    archive_objectives = np.asarray(archive_objectives)
    new_solutions = np.asarray(new_solutions)
    new_objectives = np.asarray(new_objectives)

    # Sanity checks and alignment for new batch
    if new_solutions.ndim == 1 and new_solutions.size > 0:
        new_solutions = new_solutions.reshape(1, -1)
    if new_objectives.ndim == 1 and new_objectives.size > 0:
        new_objectives = new_objectives.reshape(1, -1)

    if new_solutions.size > 0 and new_objectives.size > 0:
        n_new_sol = new_solutions.shape[0]
        n_new_obj = new_objectives.shape[0]
        if n_new_sol != n_new_obj:
            n_align = min(n_new_sol, n_new_obj)
            print(f"[WARN] update_archive: trimming new arrays to same length "
                  f"{n_align} from solutions={n_new_sol}, objectives={n_new_obj}")
            new_solutions = new_solutions[:n_align]
            new_objectives = new_objectives[:n_align]

    # Sanity checks and alignment for archive batch
    if archive_solutions.size > 0 and archive_objectives.size > 0:
        n_arc_sol = archive_solutions.shape[0]
        n_arc_obj = archive_objectives.shape[0]
        if n_arc_sol != n_arc_obj:
            n_align = min(n_arc_sol, n_arc_obj)
            print(f"[WARN] update_archive: trimming archive arrays to same length "
                  f"{n_align} from solutions={n_arc_sol}, objectives={n_arc_obj}")
            archive_solutions = archive_solutions[:n_align]
            archive_objectives = archive_objectives[:n_align]

    # Combine archive and new solutions
    if archive_solutions.size == 0:
        combined_solutions = new_solutions
        combined_objectives = new_objectives
    elif new_solutions.size == 0:
        combined_solutions = archive_solutions
        combined_objectives = archive_objectives
    else:
        combined_solutions = np.vstack([archive_solutions, new_solutions])
        combined_objectives = np.vstack([archive_objectives, new_objectives])
    
    # Find non-dominated solutions (using mixed objective dominance)
    # Final consistency check
    if combined_solutions.shape[0] != combined_objectives.shape[0]:
        n_align = min(combined_solutions.shape[0], combined_objectives.shape[0])
        print(f"[WARN] update_archive: final trim to {n_align} due to mismatch "
              f"combined_solutions={combined_solutions.shape[0]}, "
              f"combined_objectives={combined_objectives.shape[0]}")
        combined_solutions = combined_solutions[:n_align]
        combined_objectives = combined_objectives[:n_align]

    n = combined_objectives.shape[0]
    is_non_dominated = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_non_dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if is_non_dominated[j] and dominates(combined_objectives[j], combined_objectives[i]):
                is_non_dominated[i] = False
                break
    
    # Extract non-dominated solutions
    if np.any(is_non_dominated):
        archive_solutions = combined_solutions[is_non_dominated].copy()
        archive_objectives = combined_objectives[is_non_dominated].copy()
        
        # Ensure arrays are contiguous
        archive_solutions = np.ascontiguousarray(archive_solutions)
        archive_objectives = np.ascontiguousarray(archive_objectives)
    else:
        # Return empty arrays with correct shape
        if len(combined_solutions) > 0:
            archive_solutions = np.empty((0, combined_solutions.shape[1]))
            archive_objectives = np.empty((0, combined_objectives.shape[1]))
        else:
            archive_solutions = np.empty((0, new_solutions.shape[1]))
            archive_objectives = np.empty((0, new_objectives.shape[1]))
    return archive_solutions, archive_objectives


def compute_crowding_distance(objectives):
    """Compute crowding distance for archive truncation
    
    Args:
        objectives: Objective values (numpy array, shape: (n, n_obj))
    
    Returns:
        Crowding distances (numpy array, shape: (n,))
    """
    n, n_obj = objectives.shape
    if n <= 2:
        return np.ones(n) * np.inf
    
    distances = np.zeros(n)
    
    for obj_idx in range(n_obj):
        # Sort by this objective
        sorted_indices = np.argsort(objectives[:, obj_idx])
        sorted_objectives = objectives[sorted_indices, obj_idx]
        
        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Compute range
        obj_range = sorted_objectives[-1] - sorted_objectives[0]
        if obj_range < 1e-10:
            continue
        
        # Compute distances for interior points
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            distances[idx] += (sorted_objectives[i + 1] - sorted_objectives[i - 1]) / obj_range
    
    return distances


def truncate_archive(archive_solutions, archive_objectives, max_size=100):
    """Truncate archive to max_size using crowding distance
    
    Args:
        archive_solutions: Archive solutions (numpy array)
        archive_objectives: Archive objectives (numpy array)
        max_size: Maximum archive size
    
    Returns:
        Truncated archive (solutions, objectives)
    """
    n = len(archive_solutions)
    if n <= max_size:
        return archive_solutions, archive_objectives
    
    # Compute crowding distances
    distances = compute_crowding_distance(archive_objectives)
    
    # Select top max_size solutions by crowding distance
    # Sort by distance (ascending), take last max_size (highest distances)
    sorted_indices = np.argsort(distances)
    # Ensure we don't take more indices than available
    num_to_select = min(max_size, n)
    selected_indices = sorted_indices[-num_to_select:]
    
    # Ensure indices are valid (should always be true, but check for safety)
    if len(selected_indices) > 0:
        assert np.all(selected_indices >= 0) and np.all(selected_indices < n), \
            f"Invalid indices: min={selected_indices.min()}, max={selected_indices.max()}, n={n}"
    
    # Additional safety: filter out any invalid indices (shouldn't happen, but just in case)
    valid_mask = (selected_indices >= 0) & (selected_indices < n)
    if not np.all(valid_mask):
        print(f"Warning: Filtering out invalid indices in truncate_archive")
        selected_indices = selected_indices[valid_mask]
        
    return archive_solutions[selected_indices], archive_objectives[selected_indices]


def select_global_best(archive_solutions, archive_objectives):
    """Select global best (gbest) from archive using random selection
    
    Args:
        archive_solutions: Archive solutions (numpy array)
        archive_objectives: Archive objectives (numpy array)
    
    Returns:
        Selected gbest solution (numpy array)
    """
    n = len(archive_solutions)
    if n == 0:
        raise ValueError("Archive is empty, cannot select gbest")
    
    # Random selection from archive (ensure index is in valid range [0, n-1])
    # Use np.random.choice for safer random selection
    # This ensures we always get a valid index in [0, n-1]
    try:
        idx = np.random.choice(n)
    except ValueError:
        # Fallback if np.random.choice fails (shouldn't happen)
        idx = 0
    
    # Convert to Python int to avoid numpy integer type issues
    idx = int(idx)
    
    # Final validation before indexing (double-check)
    if idx < 0 or idx >= n:
        raise IndexError(f"Invalid index {idx} for archive of size {n}. This should never happen!")
    
    return archive_solutions[idx]


class MOSPORepair(RepairAlgorithm):
    """MOPSO (Multi-Objective Particle Swarm Optimization) repair algorithm
    
    Complete implementation of Multi-Objective Particle Swarm Optimization.
    Uses PSO velocity update with archive-based global best selection.
    """
    
    def __init__(self, pop_size=20, w=0.5, c1=1.5, c2=1.5, archive_size=100):
        super().__init__("MOPSO")
        self.pop_size = pop_size
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.archive_size = archive_size  # Maximum archive size
    
    def repair(self, problem, init_population, n_gen, seed=None):
        """Run MOPSO optimization
        
        Args:
            problem: BackdoorRepairProblem instance
            init_population: Initial population (numpy array)
            n_gen: Number of generations (iterations)
            seed: Random seed for reproducibility
        
        Returns:
            Result object with X (solutions) and F (objectives)
        """
        if seed is not None:
            np.random.seed(seed)
        
        print(f"\n{'='*80}")
        print(f"MOPSO (Multi-Objective Particle Swarm Optimization)")
        print(f"{'='*80}")
        print(f"Population size: {self.pop_size}")
        print(f"Generations: {n_gen}")
        print(f"Inertia weight (w): {self.w}")
        print(f"Cognitive coefficient (c1): {self.c1}")
        print(f"Social coefficient (c2): {self.c2}")
        print(f"Archive size: {self.archive_size}")
        print(f"{'='*80}\n")
        
        # Initialize particles (positions)
        n_var = problem.n_var
        particles = init_population.copy()
        
        # Clip particles to bounds
        particles = np.clip(particles, problem.xl, problem.xu)
        
        # Initialize velocities (zero or small random)
        velocities = np.zeros_like(particles)
        
        # Evaluate initial population
        print("Evaluating initial population...")
        raw_objectives = problem.evaluate_population(particles)  # shape: (n, 2) -> [SR, -ACC]
        # Convert to [SR, ACC] for MOPSO's mixed-objective dominance
        objectives = np.column_stack([raw_objectives[:, 0], -raw_objectives[:, 1]])
        
        # Initialize personal best (pbest)
        pbest = particles.copy()
        pbest_objectives = objectives.copy()
        
        # Initialize archive (non-dominated solutions)
        archive_solutions = np.empty((0, n_var))
        archive_objectives = np.empty((0, problem.n_obj))
        archive_solutions, archive_objectives = update_archive(
            archive_solutions, archive_objectives, particles, objectives
        )
        
        print(f"Initial archive size: {len(archive_solutions)}")
        
        # Main optimization loop
        for iteration in range(n_gen):
            print(f"\nGeneration {iteration + 1}/{n_gen}")
            
            # Initialize objectives array for this generation (store as [SR, ACC])
            objectives = np.zeros((self.pop_size, problem.n_obj))
            
            for i in range(self.pop_size):
                # Select global best (gbest) from archive
                if len(archive_solutions) > 0:
                    try:
                        gbest = select_global_best(archive_solutions, archive_objectives)
                    except (IndexError, ValueError) as e:
                        print(f"Error selecting gbest: {e}")
                        print(f"Archive size: {len(archive_solutions)}")
                        print(f"Archive shape: {archive_solutions.shape if hasattr(archive_solutions, 'shape') else 'N/A'}")
                        raise
                else:
                    # Fallback: use particle's own pbest
                    gbest = pbest[i]
                
                # PSO velocity update
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])
                    + self.c2 * r2 * (gbest - particles[i])
                )
                
                # Update particle position
                particles[i] += velocities[i]
                
                # Clip to bounds
                particles[i] = np.clip(particles[i], problem.xl, problem.xu)
                
                # Evaluate new position
                try:
                    f_raw = problem.evaluate_single(particles[i])  # [SR, -ACC]
                except Exception as e:
                    raise RuntimeError(
                        f"Error during evaluate_single for particle {i} at generation {iteration + 1}: {e}"
                    ) from e
                # Convert to [SR, ACC] for dominance/archive
                f_new = np.array([f_raw[0], -f_raw[1]], dtype=float)
                
                # Update personal best using mixed dominance ([SR, ACC])
                if dominates(f_new, pbest_objectives[i]):
                    # New solution dominates pbest
                    pbest[i] = particles[i].copy()
                    pbest_objectives[i] = f_new.copy()
                elif not dominates(pbest_objectives[i], f_new):
                    # Non-dominated: randomly choose one to maintain diversity
                    if np.random.rand() < 0.5:
                        pbest[i] = particles[i].copy()
                        pbest_objectives[i] = f_new.copy()
                
                # Store objective for archive update
                objectives[i] = f_new
            
            # Update archive with new solutions (objectives already computed above)
            archive_solutions, archive_objectives = update_archive(
                archive_solutions, archive_objectives, particles, objectives
            )
            
            # Also update archive with pbest solutions
            archive_solutions, archive_objectives = update_archive(
                archive_solutions, archive_objectives, pbest, pbest_objectives
            )
            
            # Truncate archive if too large
            archive_solutions, archive_objectives = truncate_archive(
                archive_solutions, archive_objectives, max_size=self.archive_size
            )
            
            print(f"  Archive size: {len(archive_solutions)}")
            
            # Print best solution info
            if len(archive_solutions) > 0:
                # Mixed-objective normalization:
                # - Normalize SR to [0,1] (lower is better)
                # - Normalize ACC to [0,1] and convert to (1 - norm_acc) so lower is better
                sr_vals = archive_objectives[:, 0]
                acc_vals = archive_objectives[:, 1]
                sr_min, sr_max = sr_vals.min(), sr_vals.max()
                acc_min, acc_max = acc_vals.min(), acc_vals.max()
                norm_sr = (sr_vals - sr_min) / (sr_max - sr_min + 1e-12)
                norm_acc = (acc_vals - acc_min) / (acc_max - acc_min + 1e-12)
                score = norm_sr + (1.0 - norm_acc)  # lower is better
                best_idx = np.argmin(score)
                best_obj = archive_objectives[best_idx]
                print(f"  Best (normalized trade-off): SR={best_obj[0]:.6f}, Acc={best_obj[1]:.6f}")
        
        print(f"\n{'='*80}")
        print(f"MOPSO optimization completed")
        print(f"Final archive size: {len(archive_solutions)}")
        print(f"{'='*80}\n")
        
        # Create result object compatible with pymoo
        class MOPSOResult:
            def __init__(self, X, F):
                self.X = X
                self.F = F  # Stored as [SR, ACC]
        
        self.results = MOPSOResult(
            X=archive_solutions,
            F=archive_objectives
        )
        
        return self.results



class GradientRepair(RepairAlgorithm):
    """Gradient-based Pareto front generation using differentiable loss functions
    
    Suggested by Reviewer 2: "The very choice of a genetic solver is questionable here: 
    gradient-based constrained optimisation can directly exploit differentiable violation 
    measures and routinely produces Pareto fronts of equal quality at a fraction of the 
    sampling cost."
    
    Key improvements over previous implementation:
    1. Uses differentiable loss functions (cross-entropy) instead of non-differentiable metrics
    2. Uses automatic differentiation (tf.gradients) instead of numerical approximation
    3. Generates Pareto front by running optimization with multiple weight combinations
    4. Much faster and more accurate than numerical gradients
    """
    
    def __init__(self, learning_rate=0.001, n_weights=11, n_iterations_per_weight=100):
        super().__init__("Gradient-Pareto")
        self.learning_rate = learning_rate
        self.n_weights = n_weights  # Number of weight combinations to try
        self.n_iterations_per_weight = n_iterations_per_weight  # Iterations per weight
    
    def repair(self, problem, init_population, n_gen, seed=None):
        """Generate Pareto front using gradient-based optimization with differentiable losses
        
        Strategy:
        1. Use cross-entropy loss (differentiable) instead of accuracy/backdoor rate
        2. Use tf.gradients for automatic differentiation (much faster than numerical)
        3. Run optimization multiple times with different weight combinations
        4. Each run produces one Pareto point
        
        Args:
            problem: BackdoorRepairProblem instance
            init_population: Initial population (numpy array)
            n_gen: Total number of gradient steps (divided among weight combinations)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            tf.compat.v1.set_random_seed(seed)
        
        print(f"\n{'='*80}")
        print(f"Gradient-based Pareto Front Generation (Reviewer 2's Suggestion)")
        print(f"{'='*80}")
        print(f"Strategy: Differentiable losses + Automatic differentiation + Multiple weights")
        print(f"Number of weight combinations: {self.n_weights}")
        print(f"Iterations per weight: {self.n_iterations_per_weight}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Total gradient steps: {self.n_weights * self.n_iterations_per_weight}")
        print(f"(Compare to NSGA-II/MOEA/D: {n_gen} generations × population size evaluations)")
        print(f"{'='*80}\n")
        
        # Get target label for backdoor
        target_labels = {'gtsrb': 33, 'mnist': 3, 'fashion': 3}
        target_label = target_labels.get(problem.model_name, 3)
        
        # Prepare data
        X_clean = problem.data_x
        y_clean = problem.data_y
        X_backdoor = problem.data_p
        
        # Convert labels to one-hot
        n_classes = problem.model.output_shape[-1]
        
        # Validate and clip labels to valid range [0, n_classes-1]
        # Convert to integer and ensure valid range
        y_clean = np.asarray(y_clean, dtype=np.int32)
        if np.any(y_clean < 0) or np.any(y_clean >= n_classes):
            invalid_labels = np.where((y_clean < 0) | (y_clean >= n_classes))[0]
            print(f"Warning: Invalid y_clean labels found: {y_clean[invalid_labels[:5]]} (expected range [0, {n_classes-1}])")
            y_clean = np.clip(y_clean, 0, n_classes - 1).astype(np.int32)
        
        # Validate target_label is in valid range
        target_label = int(target_label)
        if target_label < 0 or target_label >= n_classes:
            print(f"Warning: target_label {target_label} out of range [0, {n_classes-1}], clipping")
            target_label = int(np.clip(target_label, 0, n_classes - 1))
        
        # Final safety check before to_categorical
        assert np.all(y_clean >= 0) and np.all(y_clean < n_classes), \
            f"y_clean contains invalid values: min={y_clean.min()}, max={y_clean.max()}, n_classes={n_classes}"
        assert 0 <= target_label < n_classes, \
            f"target_label {target_label} out of range [0, {n_classes-1}]"
        
        y_clean_onehot = tf.keras.utils.to_categorical(y_clean, n_classes)
        y_backdoor_target = np.full(len(X_backdoor), target_label, dtype=np.int32)
        y_backdoor_onehot = tf.keras.utils.to_categorical(y_backdoor_target, n_classes)
        
        # Build TensorFlow graph for gradient computation
        K.clear_session()
        
        # Create new graph and session
        graph = tf.Graph()
        with graph.as_default():
            # Create new session for this graph
            sess = tf.compat.v1.Session(graph=graph)
            with sess.as_default():
                # Set Keras session
                K.set_session(sess)
                
                # Reload model in new session
                model = load_model(problem.model_path, compile=False)
                
                # Function to set neuron weights in model
                def set_neuron_weights_tf(weight_values):
                    """Set neuron weights from flat array"""
                    weights = [w.copy() for w in problem.base_weights]
                    begin_index = 0
                    for layer_idx, neuron_idx, shape_idx in problem.player_indices:
                        end_index = begin_index + np.prod(problem.shapes[shape_idx])
                        weights[layer_idx][..., neuron_idx] = weight_values[begin_index:end_index].reshape(problem.shapes[shape_idx])
                        begin_index = end_index
                    model.set_weights(weights)
                
                # Placeholders for input data
                X_clean_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X_clean.shape[1:])
                y_clean_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, n_classes))
                X_backdoor_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X_backdoor.shape[1:])
                y_backdoor_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, n_classes))
                weight_ph = tf.compat.v1.placeholder(tf.float32, shape=())  # Scalar weight for multi-objective
                
                # Model outputs
                logits_clean = model(X_clean_ph)
                logits_backdoor = model(X_backdoor_ph)
                
                # Differentiable loss functions (key improvement!)
                # L_acc: Cross-entropy on clean data (minimize to maximize accuracy)
                loss_acc = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_clean_ph, logits=logits_clean)
                )
                
                # L_backdoor: Negative cross-entropy between backdoor predictions and target
                # We want the model to be WRONG on backdoor, so we maximize cross-entropy
                # (or equivalently, minimize negative cross-entropy)
                loss_backdoor = -tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_backdoor_ph, logits=logits_backdoor)
                )
                
                # Combined loss: weighted sum
                # loss_total = weight * loss_acc + (1 - weight) * loss_backdoor
                loss_total = weight_ph * loss_acc + (1.0 - weight_ph) * loss_backdoor
                
                # CRITICAL: Create masks to only optimize selected neurons
                # Build a dictionary: weight_idx -> list of neuron indices
                weight_to_neurons = {}
                for layer_idx, neuron_idx, shape_idx in problem.player_indices:
                    if layer_idx not in weight_to_neurons:
                        weight_to_neurons[layer_idx] = []
                    weight_to_neurons[layer_idx].append(neuron_idx)
                
                # Get all trainable variables
                all_trainable_vars = model.trainable_weights
                
                # Compute gradients
                gradients = tf.gradients(loss_total, all_trainable_vars)
                
                # Create masked gradients: zero out gradients for non-selected neurons
                masked_gradients = []
                
                # Map trainable variables to weight indices
                # Trainable variables are typically in order: [conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, ...]
                # We need to find which trainable variable corresponds to which weight index
                all_weights = model.get_weights()
                trainable_var_to_weight_idx = {}
                
                # Build mapping from trainable var to weight index
                trainable_idx = 0
                for weight_idx, weight in enumerate(all_weights):
                    # Check if this weight is trainable by comparing with trainable_vars
                    if trainable_idx < len(all_trainable_vars):
                        var = all_trainable_vars[trainable_idx]
                        if tuple(weight.shape) == tuple(var.shape.as_list()):
                            trainable_var_to_weight_idx[trainable_idx] = weight_idx
                            trainable_idx += 1
                n_masked_layers = 0
                for var_idx, (grad, var) in enumerate(zip(gradients, all_trainable_vars)):
                    if grad is None:
                        masked_gradients.append(None)
                        continue
                    
                    # Get corresponding weight index
                    weight_idx = trainable_var_to_weight_idx.get(var_idx, None)
                    
                    if weight_idx is not None and weight_idx in weight_to_neurons:
                        # This layer has selected neurons - create a mask
                        mask_shape = var.shape.as_list()
                        mask = np.zeros(mask_shape, dtype=np.float32)
                        
                        # Enable gradient for selected neurons (last dimension)
                        for neuron_idx in weight_to_neurons[weight_idx]:
                            mask[..., neuron_idx] = 1.0
                        
                        # Apply mask to gradient
                        mask_tensor = tf.constant(mask, dtype=tf.float32)
                        masked_grad = grad * mask_tensor
                        masked_gradients.append(masked_grad)
                        n_masked_layers += 1
                    else:
                        # Not a selected layer - zero out gradient
                        masked_gradients.append(tf.zeros_like(grad))
                # Create optimizer with masked gradients
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_op = optimizer.apply_gradients(zip(masked_gradients, all_trainable_vars))
                
                # CRITICAL: Only initialize optimizer variables, NOT model weights!
                # global_variables_initializer() would reset the loaded model weights
                # Instead, we only initialize the optimizer's internal variables (momentum, etc.)
                optimizer_vars = optimizer.variables()
                sess.run(tf.compat.v1.variables_initializer(optimizer_vars))
                
                # Verify model weights are correct after loading
                print("\nVerifying model weights after loading...")
                pred_clean_test = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[:100]})
                pred_backdoor_test = sess.run(logits_backdoor, feed_dict={X_backdoor_ph: X_backdoor[:100]})
                acc_test = np.mean(np.argmax(pred_clean_test, axis=1) == y_clean[:100])
                backdoor_test = np.mean(np.argmax(pred_backdoor_test, axis=1) == target_label)
                print(f"Initial model performance: Acc={acc_test:.4f}, Backdoor={backdoor_test:.4f}")
                
                if acc_test < 0.5:
                    print("WARNING: Model accuracy is very low! Model weights may not be loaded correctly.")
                    print("Attempting to reload weights...")
                    # Force reload from problem.base_weights
                    model.set_weights([w.copy() for w in problem.base_weights])
                    # Re-verify
                    pred_clean_test = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[:100]})
                    acc_test = np.mean(np.argmax(pred_clean_test, axis=1) == y_clean[:100])
                    print(f"After reload: Acc={acc_test:.4f}")
                
                # Storage for Pareto front
                pareto_solutions = []
                pareto_objectives = []
                
                # Generate weight combinations: w ∈ [0, 1]
                # w=0: pure backdoor removal, w=1: pure accuracy preservation
                weights = np.linspace(0.0, 1.0, self.n_weights)
                
                # Use different starting points from initial population
                n_init = len(init_population)
                
                for i, weight_combo in enumerate(weights):
                    print(f"\n{'─'*80}")
                    print(f"Weight combination {i+1}/{self.n_weights}: w_acc={weight_combo:.3f}, w_backdoor={1-weight_combo:.3f}")
                    print(f"{'─'*80}")
                    
                    # CRITICAL: Start from original model weights, not perturbed weights!
                    # This ensures we don't break the model
                    
                    # Step 1: Reset optimizer state (momentum, etc.)
                    sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))
                    
                    # Step 2: Set model weights AFTER resetting optimizer
                    # (optimizer reset might affect model weights in some TF versions)
                    base_weights_copy = [w.copy() for w in problem.base_weights]
                    model.set_weights(base_weights_copy)
                    
                    # Step 3: Verify weights are correct
                    pred_test = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[:100]})
                    acc_test = np.mean(np.argmax(pred_test, axis=1) == y_clean[:100])
                    print(f"  Starting accuracy: {acc_test:.4f}")
                    if acc_test < 0.5:
                        raise RuntimeError(f"Model broken after weight reset! Acc={acc_test:.4f}")
                    
                    # Gradient descent loop
                    for iteration in range(self.n_iterations_per_weight):
                        # Run one gradient step
                        _, loss_val, loss_acc_val, loss_backdoor_val = sess.run(
                            [train_op, loss_total, loss_acc, loss_backdoor],
                            feed_dict={
                                X_clean_ph: X_clean,
                                y_clean_ph: y_clean_onehot,
                                X_backdoor_ph: X_backdoor,
                                y_backdoor_ph: y_backdoor_onehot,
                                weight_ph: weight_combo
                            }
                        )
                        
                        # Print progress
                        if iteration % 20 == 0 or iteration == self.n_iterations_per_weight - 1:
                            # Evaluate actual metrics using sess.run instead of model.predict
                            pred_clean_logits = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean})
                            pred_backdoor_logits = sess.run(logits_backdoor, feed_dict={X_backdoor_ph: X_backdoor})
                            
                            acc = np.mean(np.argmax(pred_clean_logits, axis=1) == y_clean)
                            backdoor_rate = np.mean(np.argmax(pred_backdoor_logits, axis=1) == target_label)
                            
                            print(f"  Iter {iteration:3d}: Loss={loss_val:.4f}, "
                                  f"Acc={acc:.4f}, Backdoor={backdoor_rate:.6f}")
                    
                    # Extract final solution
                    final_weights = model.get_weights()
                    
                    # Convert back to decision variable format
                    # Calculate total size from shapes
                    total_size = sum(np.prod(shape) for shape in problem.shapes)
                    x_final = np.zeros(total_size)
                    begin_index = 0
                    for layer_idx, neuron_idx, shape_idx in problem.player_indices:
                        end_index = begin_index + np.prod(problem.shapes[shape_idx])
                        x_final[begin_index:end_index] = final_weights[layer_idx][..., neuron_idx].flatten()
                        begin_index = end_index
                    
                    # Evaluate final solution directly in this session
                    # Use full evaluation data
                    pred_clean_full = sess.run(logits_clean, feed_dict={X_clean_ph: problem.data_x})
                    pred_backdoor_full = sess.run(logits_backdoor, feed_dict={X_backdoor_ph: problem.data_p})
                    
                    acc_final = np.mean(np.argmax(pred_clean_full, axis=1) == problem.data_y)
                    backdoor_rate_final = np.mean(np.argmax(pred_backdoor_full, axis=1) == target_label)
                    
                    # Objectives for GradientRepair are stored as [SR, ACC] (no negative accuracy)
                    f_final = np.array([backdoor_rate_final, acc_final], dtype=float)
                    
                    pareto_solutions.append(x_final)
                    pareto_objectives.append(f_final)
                    
                    print(f"  ✓ Final: SR={f_final[0]:.6f}, Acc={f_final[1]:.6f}")
                    
                    # Check if perfect
                    if f_final[0] <= problem.threshold_sr and f_final[1] >= problem.threshold_acc:
                        print(f"  🎯 Perfect solution found!")
                
                print(f"\n{'='*80}")
                print(f"Pareto front generation completed")
                print(f"Generated {len(pareto_solutions)} solutions")
                print(f"{'='*80}\n")
                
                # Close session
                sess.close()
        
        # Create result object compatible with pymoo
        class GradientResult:
            def __init__(self, X, F):
                self.X = X
                self.F = F  # Stored as [SR, ACC]
        
        self.results = GradientResult(
            X=np.array(pareto_solutions),
            F=np.array(pareto_objectives)
        )
        
        return self.results
