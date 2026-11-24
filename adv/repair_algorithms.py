"""
Repair algorithms for adversarial robustness task
Supports: NSGA-II, MOEA/D, Gradient-based (Adam)

Based on reviewer feedback:
- NSGA-II: Recommended by Reviewer 1 for 2-objective problems
- MOEA/D: Suggested by Reviewer 2 as decomposition-based alternative
- Gradient: Suggested by Reviewer 2 as gradient-based alternative
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K
from abc import ABC, abstractmethod

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def build_layer_lookup(model):
    """Build lookup tables mapping layer names to sequential indices."""
    lookup = {
        'conv_names': [],
        'dense_names': [],
        'name_to_info': {}
    }
    
    weight_idx = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        
        lname = type(layer).__name__
        lower_name = layer.name.lower()
        info = None
        
        if 'Batch' in lname:
            pass
        elif 'Conv' in lname:
            info = {
                'type': 'conv',
                'seq_index': len(lookup['conv_names']),
                'weight_idx': weight_idx
            }
            lookup['conv_names'].append(lower_name)
        elif 'Dense' in lname or 'Fc' in lname:
            info = {
                'type': 'dense',
                'seq_index': len(lookup['dense_names']),
                'weight_idx': weight_idx
            }
            lookup['dense_names'].append(lower_name)
        
        if info:
            lookup['name_to_info'][lower_name] = info
        
        weight_idx += len(weights)
    
    return lookup


def parse_player_identifier(player, layer_lookup=None):
    """Parse player string using either layer names or sequential indices."""
    parts = player.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid player identifier format: {player}")
    
    layer_label = parts[0].lower()
    if layer_label.startswith('conv'):
        layer_type = 'conv'
    elif layer_label.startswith('dense') or layer_label.startswith('fc'):
        layer_type = 'dense'
    else:
        raise ValueError(f"Unsupported layer type in player identifier: {player}")
    
    try:
        neuron_idx = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid neuron index in player identifier: {player}") from exc
    
    seq_index = None
    if layer_lookup:
        layer_key = '_'.join(parts[:-1]).lower()
        info = layer_lookup['name_to_info'].get(layer_key)
        if info and info['type'] == layer_type:
            seq_index = info['seq_index']
    
    if seq_index is None:
        if len(parts) < 3:
            raise ValueError(f"Missing layer index in player identifier: {player}")
        try:
            seq_index = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid layer index in player identifier: {player}") from exc
    
    return layer_type, seq_index, neuron_idx


class RepairAlgorithm(ABC):
    """Abstract base class for repair algorithms"""
    
    def __init__(self, model_path, selected_players, shapes, convs, denses, 
                 adv_x, adv_y, clean_x, clean_y, batch_size=1024, layer_lookup=None):
        """Initialize repair algorithm
        
        Args:
            model_path: Path to the model file
            selected_players: List of neuron identifiers to repair
            shapes: List of weight shapes for selected neurons
            convs: List of convolutional layer indices
            denses: List of dense layer indices
            adv_x: Adversarial training examples
            adv_y: Labels for adversarial examples
            clean_x: Clean training examples
            clean_y: Labels for clean examples
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.selected_players = selected_players
        self.shapes = shapes
        self.convs = convs
        self.denses = denses
        self.adv_x = adv_x
        self.adv_y = adv_y
        self.clean_x = clean_x
        self.clean_y = clean_y
        self.batch_size = batch_size
        
        # Load model
        self.model = load_model(model_path, compile=False)
        self.base_weights = self.model.get_weights()
        self.layer_lookup = layer_lookup or build_layer_lookup(self.model)
        
        # Pre-compute indices for efficiency
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Pre-compute layer and weight indices for selected neurons"""
        self.player_indices = []
        for player in self.selected_players:
            layer_type, layer_pos, neuron_idx = parse_player_identifier(player, self.layer_lookup)
            if layer_type == "conv":
                if not 0 <= layer_pos < len(self.convs):
                    raise IndexError(f"conv layer index {layer_pos} out of range for player {player}")
                layer_idx = self.convs[layer_pos]
            else:
                if not 0 <= layer_pos < len(self.denses):
                    raise IndexError(f"dense layer index {layer_pos} out of range for player {player}")
                layer_idx = self.denses[layer_pos]
            self.player_indices.append((layer_idx, neuron_idx))
        
        self.weight_slices = []
        begin_index = 0
        for shape in self.shapes:
            end_index = begin_index + np.prod(shape)
            self.weight_slices.append((begin_index, end_index, shape))
            begin_index = end_index
    
    def sess_run(self, model, variable, images):
        """Run model inference in batches"""
        output = []
        num_batches = int(np.ceil(len(images) / self.batch_size))
        for batch in range(num_batches):
            batch_idxs = np.arange(
                batch * self.batch_size, 
                min((batch + 1) * self.batch_size, len(images))
            )
            input_dic = {model.input: images[batch_idxs]}
            output.append(K.get_session().run(variable, input_dic))
        return np.concatenate(output, 0)
    
    def evaluate(self, model, adv_x, adv_y, clean_x, clean_y):
        """Evaluate model on adversarial and clean data
        
        Matching adv_moo_2.py logic:
        - Adversarial robustness: accuracy of adv_x with adv_y labels
        - Clean accuracy: accuracy of clean_x with clean_y labels
        
        Returns:
            Tuple of (adv_robustness, clean_accuracy)
        """
        # Adversarial robustness (accuracy on adversarial examples)
        output_adv = self.sess_run(model, model.output, adv_x)
        pred_adv = np.argmax(output_adv, axis=-1)
        adv_robustness = np.sum(pred_adv == adv_y) / len(adv_y)
        
        # Clean accuracy
        output_clean = self.sess_run(model, model.output, clean_x)
        pred_clean = np.argmax(output_clean, axis=-1)
        clean_accuracy = np.sum(pred_clean == clean_y) / len(clean_y)
        
        return adv_robustness, clean_accuracy
    
    @abstractmethod
    def repair(self, pop_size=200, n_gen=3000, save_dir=None, **kwargs):
        """Run the repair algorithm
        
        Args:
            pop_size: Population size
            n_gen: Number of generations
            save_dir: Directory to save results
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Pareto front solutions
        """
        pass


class NSGA2Repair(RepairAlgorithm):
    """NSGA-II based repair algorithm"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "NSGA-II"
    
    def repair(self, pop_size=200, n_gen=3000, save_dir=None, 
               threshold_acc=0.971, threshold_adv=0.81):
        """Run NSGA-II repair
        
        Args:
            pop_size: Population size
            n_gen: Number of generations
            save_dir: Directory to save results
            threshold_acc: Minimum accuracy threshold
            threshold_adv: Minimum robustness threshold (1 - adv_error)
            
        Returns:
            Optimization result object
        """
        print(f"\nRunning {self.name} repair...")
        print(f"  Population size: {pop_size}")
        print(f"  Generations: {n_gen}")
        print(f"  Thresholds: acc >= {threshold_acc}, robust >= {threshold_adv}")
        
        # Extract initial weights
        init_weights = []
        for player in self.selected_players:
            layer_type, layer_pos, neuron_idx = parse_player_identifier(player, self.layer_lookup)
            if layer_type == "conv":
                weight_index = self.convs[layer_pos]
            else:
                weight_index = self.denses[layer_pos]
            weight = self.base_weights[weight_index][..., neuron_idx]
            init_weights.append(weight.flatten())
        
        init_weights = np.concatenate(init_weights)
        init_population = np.tile(init_weights, (pop_size, 1))
        
        # Create problem
        problem = AdversarialRepairProblem(
            n_var=len(init_weights),
            threshold_acc=threshold_acc,
            threshold_adv=threshold_adv,
            repair_algo=self,
            save_dir=save_dir
        )
        
        # Setup algorithm - NSGA-II (no reference directions needed)
        mutation = PolynomialMutation(prob=0.9, eta=20)
        crossover = UniformCrossover(prob=1.0)
        
        algorithm = NSGA2(
            pop_size=pop_size,
            mutation=mutation,
            crossover=crossover,
            sampling=init_population
        )
        
        # Run optimization
        termination = get_termination('n_gen', n_gen)
        try:
            res = minimize(
                problem,
                algorithm,
                termination,
                seed=1023,
                verbose=True
            )
        except Exception as e:
            print(f"\nError during {self.name} optimization: {e}")
            import traceback
            traceback.print_exc()
            # Return empty result object
            class EmptyResult:
                def __init__(self):
                    self.X = np.array([]).reshape(0, len(init_weights))
                    self.F = np.array([]).reshape(0, 2)
            return EmptyResult()
        
        if res is None:
            print(f"\nWarning: {self.name} returned None, creating empty result")
            class EmptyResult:
                def __init__(self):
                    self.X = np.array([]).reshape(0, len(init_weights))
                    self.F = np.array([]).reshape(0, 2)
            return EmptyResult()
        
        print(f"\n{self.name} completed!")
        if hasattr(res, 'X') and res.X is not None:
            print(f"  Pareto front size: {len(res.X)}")
        else:
            print(f"  Warning: No Pareto front found in result")
            # Create empty result if X is None
            if not hasattr(res, 'X') or res.X is None:
                res.X = np.array([]).reshape(0, len(init_weights))
                res.F = np.array([]).reshape(0, 2)
        
        return res


class MOEADRepair(RepairAlgorithm):
    """MOEA/D based repair algorithm
    
    Note: MOEA/D does not support constraints, so we use an unconstrained problem.
    Constraint filtering is done in post-processing (compare_algorithms.py).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "MOEA/D"
    
    def repair(self, pop_size=200, n_gen=3000, save_dir=None,
               threshold_acc=0.971, threshold_adv=0.81, n_neighbors=20):
        """Run MOEA/D repair
        
        Args:
            pop_size: Population size
            n_gen: Number of generations
            save_dir: Directory to save results
            threshold_acc: Minimum accuracy threshold (used for saving, not as constraint)
            threshold_adv: Minimum robustness threshold (used for saving, not as constraint)
            n_neighbors: Number of neighboring subproblems
            
        Returns:
            Optimization result object
        """
        print(f"\nRunning {self.name} repair...")
        print(f"  Population size: {pop_size}")
        print(f"  Generations: {n_gen}")
        print(f"  Neighbors: {n_neighbors}")
        print(f"  Note: MOEA/D does not support constraints - filtering done in post-processing")
        
        # Extract initial weights
        init_weights = []
        for player in self.selected_players:
            layer_type, layer_pos, neuron_idx = parse_player_identifier(player, self.layer_lookup)
            if layer_type == "conv":
                weight_index = self.convs[layer_pos]
            else:
                weight_index = self.denses[layer_pos]
            weight = self.base_weights[weight_index][..., neuron_idx]
            init_weights.append(weight.flatten())
        
        init_weights = np.concatenate(init_weights)
        init_population = np.tile(init_weights, (pop_size, 1))
        
        # Create problem WITHOUT constraints (MOEA/D limitation)
        problem = AdversarialRepairProblemUnconstrained(
            n_var=len(init_weights),
            threshold_acc=threshold_acc,
            threshold_adv=threshold_adv,
            repair_algo=self,
            save_dir=save_dir
        )
        
        # Setup algorithm - dynamically determine n_partitions based on pop_size
        # For 2 objectives: n_partitions + 1 = pop_size
        n_partitions = pop_size - 1
        
        # Generate reference directions for 2-objective problem
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
        
        # If ref_dirs doesn't match pop_size exactly, adjust init_population
        n_ref = len(ref_dirs)
        if n_ref != pop_size:
            # Extend or trim init_population to match ref_dirs
            if n_ref > pop_size:
                # Duplicate some individuals
                extra_needed = n_ref - pop_size
                indices = np.random.choice(pop_size, extra_needed, replace=True)
                extra_pop = init_population[indices] + np.random.uniform(-0.01, 0.01, (extra_needed, init_population.shape[1]))
                init_population = np.vstack([init_population, extra_pop])
            else:
                # Trim population
                init_population = init_population[:n_ref]
        
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=min(n_neighbors, n_ref - 1),  # Ensure n_neighbors < pop_size
            prob_neighbor_mating=0.9,
            mutation=PolynomialMutation(prob=0.9),
            crossover=UniformCrossover(prob=1.0),
            sampling=init_population
        )
        
        # Run optimization
        termination = get_termination('n_gen', n_gen)
        try:
            res = minimize(
                problem,
                algorithm,
                termination,
                seed=1023,
                verbose=True
            )
        except Exception as e:
            print(f"\nError during {self.name} optimization: {e}")
            import traceback
            traceback.print_exc()
            # Return empty result object
            class EmptyResult:
                def __init__(self):
                    self.X = np.array([]).reshape(0, len(init_weights))
                    self.F = np.array([]).reshape(0, 2)
            return EmptyResult()
        
        if res is None:
            print(f"\nWarning: {self.name} returned None, creating empty result")
            class EmptyResult:
                def __init__(self):
                    self.X = np.array([]).reshape(0, len(init_weights))
                    self.F = np.array([]).reshape(0, 2)
            return EmptyResult()
        
        print(f"\n{self.name} completed!")
        if hasattr(res, 'X') and res.X is not None:
            print(f"  Pareto front size: {len(res.X)}")
        else:
            print(f"  Warning: No Pareto front found in result")
            # Create empty result if X is None
            if not hasattr(res, 'X') or res.X is None:
                res.X = np.array([]).reshape(0, len(init_weights))
                res.F = np.array([]).reshape(0, 2)
        
        return res


class GradientRepair(RepairAlgorithm):
    """Gradient-based Pareto front generation using differentiable loss functions
    
    Suggested by Reviewer 2: "The very choice of a genetic solver is questionable here: 
    gradient-based constrained optimisation can directly exploit differentiable violation 
    measures and routinely produces Pareto fronts of equal quality at a fraction of the 
    sampling cost."
    
    Key improvements over genetic algorithms:
    1. Uses differentiable loss functions (cross-entropy) instead of non-differentiable metrics
    2. Uses automatic differentiation (tf.gradients) instead of numerical approximation
    3. Generates Pareto front by running optimization with multiple weight combinations
    4. Much faster and more accurate than evolutionary algorithms
    """
    
    def __init__(self, *args, learning_rate=0.001, n_weights=3, n_iterations_per_weight=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Gradient-Adam"
        self.learning_rate = learning_rate
        self.n_weights = n_weights  # Number of weight combinations to try
        self.n_iterations_per_weight = n_iterations_per_weight  # Iterations per weight
    
    def repair(self, pop_size=200, n_gen=3000, save_dir=None,
               threshold_acc=0.971, threshold_adv=0.81, 
               learning_rate=None, n_weights=None, n_iterations_per_weight=None):
        """Generate Pareto front using gradient-based optimization with differentiable losses
        
        Strategy (same as backdoor implementation):
        1. Use cross-entropy loss (differentiable) instead of accuracy/robustness metrics
        2. Use tf.gradients for automatic differentiation (much faster than numerical)
        3. Run optimization multiple times with different weight combinations
        4. Each run produces one Pareto point
        
        Args:
            pop_size: Not used (kept for interface compatibility)
            n_gen: Total number of iterations (distributed across weight combinations)
            save_dir: Directory to save results
            threshold_acc: Minimum accuracy threshold
            threshold_adv: Minimum robustness threshold
            learning_rate: Learning rate for Adam (default: use self.learning_rate)
            n_weights: Number of weight combinations to try (default: use self.n_weights)
            n_iterations_per_weight: Iterations per weight (default: use self.n_iterations_per_weight or n_gen // n_weights)
            
        Returns:
            Optimization result object with Pareto front
        """
        # Use instance defaults if not specified
        if learning_rate is None:
            learning_rate = self.learning_rate
        if n_weights is None:
            n_weights = self.n_weights
        if n_iterations_per_weight is None:
            if hasattr(self, 'n_iterations_per_weight') and self.n_iterations_per_weight is not None:
                n_iterations_per_weight = self.n_iterations_per_weight
            else:
                n_iterations_per_weight = max(1, n_gen // n_weights)
        
        print(f"\n{'='*80}")
        print(f"Gradient-based Pareto Front Generation (Reviewer 2's Suggestion)")
        print(f"{'='*80}")
        print(f"Strategy: Differentiable losses + Automatic differentiation + Multiple weights")
        print(f"Number of weight combinations: {n_weights}")
        print(f"Iterations per weight: {n_iterations_per_weight}")
        print(f"Learning rate: {learning_rate}")
        print(f"Total gradient steps: {n_weights * n_iterations_per_weight}")
        print(f"(Compare to NSGA-II/MOEA/D: {n_gen} generations × population size evaluations)")
        print(f"{'='*80}\n")
        
        # Prepare data
        X_clean = self.clean_x
        y_clean = self.clean_y
        X_adv = self.adv_x
        y_adv = self.adv_y
        
        # Convert labels to one-hot
        n_classes = self.model.output_shape[-1]
        y_clean_onehot = tf.keras.utils.to_categorical(y_clean, n_classes)
        y_adv_onehot = tf.keras.utils.to_categorical(y_adv, n_classes)
        
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
                model = load_model(self.model_path, compile=False)
                
                # Placeholders for input data
                X_clean_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X_clean.shape[1:])
                y_clean_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, n_classes))
                X_adv_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + X_adv.shape[1:])
                y_adv_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, n_classes))
                weight_ph = tf.compat.v1.placeholder(tf.float32, shape=())  # Scalar weight for multi-objective
                
                # Model outputs
                logits_clean = model(X_clean_ph)
                logits_adv = model(X_adv_ph)
                
                # Differentiable loss functions (key improvement!)
                # L_acc: Cross-entropy on clean data (minimize to maximize accuracy)
                loss_acc = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_clean_ph, logits=logits_clean)
                )
                
                # L_robust: Cross-entropy on adversarial data (minimize to maximize robustness)
                # We want the model to be CORRECT on adversarial examples
                loss_robust = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_adv_ph, logits=logits_adv)
                )
                
                # Combined loss: weighted sum
                # loss_total = weight * loss_acc + (1 - weight) * loss_robust
                loss_total = weight_ph * loss_acc + (1.0 - weight_ph) * loss_robust
                
                # CRITICAL: Create masks to only optimize selected neurons
                # Build a dictionary: weight_idx -> list of neuron indices
                weight_to_neurons = {}
                for (layer_idx, neuron_idx), _ in zip(self.player_indices, self.weight_slices):
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
                
                print(f"Debug: Total trainable vars: {len(all_trainable_vars)}")
                print(f"Debug: Weight indices with selected neurons: {list(weight_to_neurons.keys())}")
                print(f"Debug: Trainable var to weight idx mapping: {trainable_var_to_weight_idx}")
                
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
                        print(f"Debug: Masked layer {var_idx} (weight_idx={weight_idx}), shape={mask_shape}, neurons={weight_to_neurons[weight_idx]}")
                    else:
                        # Not a selected layer - zero out gradient
                        masked_gradients.append(tf.zeros_like(grad))
                
                print(f"Debug: Total layers with non-zero gradients: {n_masked_layers}/{len(all_trainable_vars)}")
                
                # Create optimizer with masked gradients
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.apply_gradients(zip(masked_gradients, all_trainable_vars))
                
                # CRITICAL: Only initialize optimizer variables, NOT model weights!
                optimizer_vars = optimizer.variables()
                sess.run(tf.compat.v1.variables_initializer(optimizer_vars))
                
                # Verify model weights are correct after loading
                print("\nVerifying model weights after loading...")
                pred_clean_test = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[:100]})
                pred_adv_test = sess.run(logits_adv, feed_dict={X_adv_ph: X_adv[:100]})
                acc_test = np.mean(np.argmax(pred_clean_test, axis=1) == y_clean[:100])
                robust_test = np.mean(np.argmax(pred_adv_test, axis=1) == y_adv[:100])
                print(f"Initial model performance: Acc={acc_test:.4f}, Robust={robust_test:.4f}")
                
                if acc_test < 0.5:
                    print("WARNING: Model accuracy is very low! Model weights may not be loaded correctly.")
                    print("Attempting to reload weights...")
                    # Force reload from base_weights
                    model.set_weights([w.copy() for w in self.base_weights])
                    # Re-verify
                    pred_clean_test = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[:100]})
                    acc_test = np.mean(np.argmax(pred_clean_test, axis=1) == y_clean[:100])
                    print(f"After reload: Acc={acc_test:.4f}")
                
                # Storage for Pareto front
                pareto_solutions = []
                pareto_objectives = []
                
                # Generate weight combinations: w ∈ [0, 1]
                # w=0: pure robustness, w=1: pure accuracy
                weights = np.linspace(0.0, 1.0, n_weights)
                
                for i, weight_combo in enumerate(weights):
                    print(f"\n{'─'*80}")
                    print(f"Weight combination {i+1}/{n_weights}: w_acc={weight_combo:.3f}, w_robust={1-weight_combo:.3f}")
                    print(f"{'─'*80}")
                    
                    # CRITICAL: Start from original model weights, not perturbed weights!
                    # This ensures we don't break the model
                    
                    # Step 1: Reset optimizer state (momentum, etc.)
                    sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))
                    
                    # Step 2: Set model weights AFTER resetting optimizer
                    base_weights_copy = [w.copy() for w in self.base_weights]
                    model.set_weights(base_weights_copy)
                    
                    # Step 3: Verify weights are correct
                    pred_test = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[:100]})
                    acc_test = np.mean(np.argmax(pred_test, axis=1) == y_clean[:100])
                    print(f"  Starting accuracy: {acc_test:.4f}")
                    if acc_test < 0.5:
                        raise RuntimeError(f"Model broken after weight reset! Acc={acc_test:.4f}")
                    
                    # Gradient descent loop with mini-batch training
                    batch_size = min(512, len(X_clean))  # Use mini-batch to avoid OOM
                    n_batches = (len(X_clean) + batch_size - 1) // batch_size
                    
                    for iteration in range(n_iterations_per_weight):
                        # Sample a random batch for this iteration
                        batch_idx = np.random.choice(len(X_clean), batch_size, replace=False)
                        X_clean_batch = X_clean[batch_idx]
                        y_clean_batch = y_clean_onehot[batch_idx]
                        X_adv_batch = X_adv[batch_idx]
                        y_adv_batch = y_adv_onehot[batch_idx]
                        
                        # Run one gradient step on the batch
                        _, loss_val, loss_acc_val, loss_robust_val = sess.run(
                            [train_op, loss_total, loss_acc, loss_robust],
                            feed_dict={
                                X_clean_ph: X_clean_batch,
                                y_clean_ph: y_clean_batch,
                                X_adv_ph: X_adv_batch,
                                y_adv_ph: y_adv_batch,
                                weight_ph: weight_combo
                            }
                        )
                        
                        # Print progress
                        if iteration % 20 == 0 or iteration == n_iterations_per_weight - 1:
                            # Evaluate on a sample to avoid OOM
                            eval_size = min(1000, len(X_clean))
                            eval_idx = np.random.choice(len(X_clean), eval_size, replace=False)
                            
                            pred_clean_logits = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[eval_idx]})
                            pred_adv_logits = sess.run(logits_adv, feed_dict={X_adv_ph: X_adv[eval_idx]})
                            
                            acc = np.mean(np.argmax(pred_clean_logits, axis=1) == y_clean[eval_idx])
                            robust = np.mean(np.argmax(pred_adv_logits, axis=1) == y_adv[eval_idx])
                            
                            print(f"  Iter {iteration:3d}: Loss={loss_val:.4f}, "
                                  f"Acc={acc:.4f}, Robust={robust:.6f}")

                    
                    # Extract final solution
                    final_weights = model.get_weights()
                    
                    # Convert back to decision variable format
                    x_final = []
                    for (layer_idx, neuron_idx), (begin_idx, end_idx, shape) in zip(
                        self.player_indices, self.weight_slices
                    ):
                        x_final.append(final_weights[layer_idx][..., neuron_idx].flatten())
                    x_final = np.concatenate(x_final)
                    
                    # Evaluate final solution using batch evaluation to avoid OOM
                    eval_batch_size = 512
                    all_clean_preds = []
                    all_adv_preds = []
                    
                    for i in range(0, len(X_clean), eval_batch_size):
                        batch_end = min(i + eval_batch_size, len(X_clean))
                        pred_clean_batch = sess.run(logits_clean, feed_dict={X_clean_ph: X_clean[i:batch_end]})
                        pred_adv_batch = sess.run(logits_adv, feed_dict={X_adv_ph: X_adv[i:batch_end]})
                        all_clean_preds.append(np.argmax(pred_clean_batch, axis=1))
                        all_adv_preds.append(np.argmax(pred_adv_batch, axis=1))
                    
                    all_clean_preds = np.concatenate(all_clean_preds)
                    all_adv_preds = np.concatenate(all_adv_preds)
                    
                    acc_final = np.mean(all_clean_preds == y_clean)
                    robust_final = np.mean(all_adv_preds == y_adv)
                    
                    # Objectives: [-robustness, -accuracy]
                    f_final = np.array([-robust_final, -acc_final])
                    
                    pareto_solutions.append(x_final)
                    pareto_objectives.append(f_final)
                    
                    print(f"  ✓ Final: Robust={robust_final:.6f}, Acc={acc_final:.6f}")
                    
                    # Check if perfect
                    if robust_final >= threshold_adv and acc_final >= threshold_acc:
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
                self.F = F
        
        return GradientResult(
            X=np.array(pareto_solutions),
            F=np.array(pareto_objectives)
        )


class AdversarialRepairProblem(Problem):
    """Multi-objective optimization problem for adversarial repair (with constraints)
    
    Used by NSGA-II which supports constraints.
    """
    
    def __init__(self, n_var, threshold_acc, threshold_adv, repair_algo, save_dir=None):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=2,
            xl=-5.0,
            xu=5.0,
            vtype=float
        )
        self.threshold_acc = threshold_acc
        self.threshold_adv = threshold_adv
        self.repair_algo = repair_algo
        self.save_dir = save_dir
        self.best_fitness = 10.0
        self.eval_count = 0
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population"""
        weights = [w.copy() if isinstance(w, np.ndarray) else w 
                   for w in self.repair_algo.base_weights]
        
        adv_errors = []
        accuracies = []
        constraint_acc = []
        constraint_adv = []
        
        for i in range(x.shape[0]):
            self.eval_count += 1
            
            try:
                # Set neuron weights
                solution = x[i]
                for (layer_idx, neuron_idx), (begin_idx, end_idx, shape) in zip(
                    self.repair_algo.player_indices, self.repair_algo.weight_slices
                ):
                    weights[layer_idx][..., neuron_idx] = solution[begin_idx:end_idx].reshape(shape)
                
                # Update model
                self.repair_algo.model.set_weights(weights)
                
                # Evaluate (returns adv_robustness, clean_accuracy)
                adv_robustness, clean_accuracy = self.repair_algo.evaluate(
                    self.repair_algo.model,
                    self.repair_algo.adv_x,
                    self.repair_algo.adv_y,
                    self.repair_algo.clean_x,
                    self.repair_algo.clean_y
                )
                
                # Save promising solutions
                if self.save_dir and clean_accuracy >= self.threshold_acc and adv_robustness >= self.threshold_adv:
                    fitness = -adv_robustness - clean_accuracy  # Lower is better
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        index = len([f for f in os.listdir(self.save_dir) if f.startswith('weights_')])
                        self.repair_algo.model.save_weights(
                            os.path.join(self.save_dir, f'weights_{index}.h5')
                        )
                
                # Objectives: minimize negative values (to maximize positive values)
                # This matches adv_moo_2.py format
                adv_errors.append(-adv_robustness)  # Minimize -robustness = maximize robustness
                accuracies.append(-clean_accuracy)  # Minimize -accuracy = maximize accuracy
                constraint_acc.append(-clean_accuracy + self.threshold_acc)
                constraint_adv.append(-adv_robustness + self.threshold_adv)
            except Exception as e:
                # If evaluation fails, use worst-case values
                print(f"  Warning: Evaluation failed for individual {i}: {e}")
                adv_errors.append(1.0)  # Worst robustness
                accuracies.append(1.0)  # Worst accuracy
                constraint_acc.append(1.0)  # Worst constraint violation
                constraint_adv.append(1.0)  # Worst constraint violation
        
        out['F'] = np.column_stack([adv_errors, accuracies])
        out['G'] = np.column_stack([constraint_adv, constraint_acc])
        
        # Print progress
        if self.eval_count % (x.shape[0] * 10) == 0:
            gen = self.eval_count // x.shape[0]
            best_robust = max([-e for e in adv_errors])  # Convert back from negative
            best_acc = max([-a for a in accuracies])
            print(f"  Gen {gen:3d}: Best robust={best_robust:.6f}, Best acc={best_acc:.6f}")


class AdversarialRepairProblemUnconstrained(Problem):
    """Multi-objective optimization problem for adversarial repair (without constraints)
    
    Used by MOEA/D which does not support constraints.
    Constraint filtering is done in post-processing (compare_algorithms.py).
    """
    
    def __init__(self, n_var, threshold_acc, threshold_adv, repair_algo, save_dir=None):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,  # No constraints for MOEA/D
            xl=-5.0,
            xu=5.0,
            vtype=float
        )
        self.threshold_acc = threshold_acc
        self.threshold_adv = threshold_adv
        self.repair_algo = repair_algo
        self.save_dir = save_dir
        self.best_fitness = 10.0
        self.eval_count = 0
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population"""
        weights = [w.copy() if isinstance(w, np.ndarray) else w 
                   for w in self.repair_algo.base_weights]
        
        adv_errors = []
        accuracies = []
        
        for i in range(x.shape[0]):
            self.eval_count += 1
            
            try:
                # Set neuron weights
                solution = x[i]
                for (layer_idx, neuron_idx), (begin_idx, end_idx, shape) in zip(
                    self.repair_algo.player_indices, self.repair_algo.weight_slices
                ):
                    weights[layer_idx][..., neuron_idx] = solution[begin_idx:end_idx].reshape(shape)
                
                # Update model
                self.repair_algo.model.set_weights(weights)
                
                # Evaluate (returns adv_robustness, clean_accuracy)
                adv_robustness, clean_accuracy = self.repair_algo.evaluate(
                    self.repair_algo.model,
                    self.repair_algo.adv_x,
                    self.repair_algo.adv_y,
                    self.repair_algo.clean_x,
                    self.repair_algo.clean_y
                )
                
                # Save promising solutions (still useful even without constraints)
                if self.save_dir and clean_accuracy >= self.threshold_acc and adv_robustness >= self.threshold_adv:
                    fitness = -adv_robustness - clean_accuracy  # Lower is better
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        index = len([f for f in os.listdir(self.save_dir) if f.startswith('weights_')])
                        self.repair_algo.model.save_weights(
                            os.path.join(self.save_dir, f'weights_{index}.h5')
                        )
                
                # Objectives: minimize negative values (to maximize positive values)
                adv_errors.append(-adv_robustness)  # Minimize -robustness = maximize robustness
                accuracies.append(-clean_accuracy)  # Minimize -accuracy = maximize accuracy
            except Exception as e:
                # If evaluation fails, use worst-case values
                print(f"  Warning: Evaluation failed for individual {i}: {e}")
                adv_errors.append(1.0)  # Worst robustness
                accuracies.append(1.0)  # Worst accuracy
        
        out['F'] = np.column_stack([adv_errors, accuracies])
        # No constraints for MOEA/D
        
        # Print progress
        if self.eval_count % (x.shape[0] * 10) == 0:
            gen = self.eval_count // x.shape[0]
            best_robust = max([-e for e in adv_errors])  # Convert back from negative
            best_acc = max([-a for a in accuracies])
            print(f"  Gen {gen:3d}: Best robust={best_robust:.6f}, Best acc={best_acc:.6f}")