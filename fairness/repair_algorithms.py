"""
Repair algorithms for fairness task
Supports: NSGA-II, MOEA/D, Gradient-based (Adam)

Note: NSGA-III removed as reviewers suggested NSGA-II is more appropriate for 2-objective problems.
"""

import numpy as np
import os
import h5py
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


class FairnessRepairProblem(Problem):
    """Multi-objective optimization problem for fairness repair
    
    Optimized version following fairness_moo_2.py:
    - Model reuse (avoid repeated loading)
    - Weight caching
    - Pre-computed indices
    - Perfect solution tracking
    """
    
    def __init__(self, model_path, init_weights, selected_players, shapes, 
                 convs, denses, test_x, test_y, threshold_acc=0.96, threshold_fairness=0.008):
        super().__init__(
            n_var=len(init_weights[0]),
            n_obj=2,
            n_ieq_constr=0,  # No hard constraints, use objectives only
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
        self.test_x = test_x
        self.test_y = test_y
        self.threshold_acc = threshold_acc
        self.threshold_fairness = threshold_fairness
        
        # Performance optimization: reuse model and cache weights
        self.model = load_model(model_path, compile=False)
        self.base_weights = self.model.get_weights()
        
        # Pre-compute player indices for faster evaluation
        self.player_indices = []
        for j, player in enumerate(selected_players):
            parts = player.split('_')
            if parts[0] == "conv":
                layer_idx = convs[int(parts[1])]
            else:
                layer_idx = denses[int(parts[1])]
            neuron_idx = int(parts[-1])
            self.player_indices.append((layer_idx, neuron_idx, j))
        
        # Track perfect solutions
        self.perfect_solutions = []
        self.evaluation_count = 0
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate population with optimized model reuse"""
        f1_list = []  # fairness gap
        f2_list = []  # negative accuracy (for minimization)
        
        for i in range(x.shape[0]):
            self.evaluation_count += 1
            
            # Reset to base weights (faster than get_weights each time)
            weights = [w.copy() for w in self.base_weights]
            
            # Set neuron weights using pre-computed indices
            begin_index = 0
            weight = x[i]
            for layer_idx, neuron_idx, shape_idx in self.player_indices:
                end_index = begin_index + np.prod(self.shapes[shape_idx])
                weights[layer_idx][..., neuron_idx] = weight[begin_index:end_index].reshape(self.shapes[shape_idx])
                begin_index = end_index
            
            self.model.set_weights(weights)
            
            # Evaluate objectives
            fairness_gap = self._compute_fairness(self.model, self.test_x, self.test_y)
            accuracy = self._compute_accuracy(self.model, self.test_x, self.test_y)
            
            # Track perfect solutions
            if accuracy >= self.threshold_acc and fairness_gap <= self.threshold_fairness:
                self.perfect_solutions.append({
                    'weights': x[i].copy(),
                    'fairness_gap': fairness_gap,
                    'accuracy': accuracy,
                    'evaluation': self.evaluation_count
                })
                print(f"Perfect solution found at evaluation {self.evaluation_count}! "
                      f"Fairness={fairness_gap:.6f}, Acc={accuracy:.6f}")
            
            f1_list.append(fairness_gap)
            f2_list.append(-accuracy)  # Negative for minimization
        
        out['F'] = np.column_stack([f1_list, f2_list])
    
    def _compute_fairness(self, model, images, labels, batch_size=1024):
        """Compute fairness gap (difference between female and male accuracy)"""
        output = self._sess_run(model, model.output, images, batch_size)
        pred_labels = np.argmax(output, axis=-1)
        corr_predictions = pred_labels == labels
        
        # Calculate per-group accuracy
        female_acc = np.sum((labels == 0) & corr_predictions) / np.sum(labels == 0)
        male_acc = np.sum((labels == 1) & corr_predictions) / np.sum(labels == 1)
        
        # Return absolute fairness gap
        return abs(male_acc - female_acc)
    
    def _compute_accuracy(self, model, images, labels, batch_size=1024):
        """Compute overall accuracy"""
        output = self._sess_run(model, model.output, images, batch_size)
        pred_labels = np.argmax(output, axis=-1)
        corr_predictions = pred_labels == labels
        
        return np.sum(corr_predictions) / len(corr_predictions)
    
    def _sess_run(self, model, variable, images, batch_size=1024):
        """Run model in batches"""
        output = []
        num_batches = int(np.ceil(len(images) / batch_size))
        for batch in range(num_batches):
            batch_idxs = np.arange(
                batch * batch_size, min((batch + 1) * batch_size, len(images))
            )
            input_dic = {model.input: images[batch_idxs]}
            output.append(K.get_session().run(variable, input_dic))
        return np.concatenate(output, 0)


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
            problem: FairnessRepairProblem instance
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


class MOEADRepair(RepairAlgorithm):
    """MOEA/D repair algorithm
    
    Suggested by Reviewer 2: "NSGA-III is introduced without a shred of evidence 
    that its reference-point mechanism outperforms decomposition-based alternatives 
    (MOEA/D, SMS-EMOA)..."
    """
    
    def __init__(self, n_neighbors=15):
        super().__init__("MOEA/D")
        self.n_neighbors = n_neighbors
    
    def repair(self, problem, init_population, n_gen, seed=None):
        """Run MOEA/D optimization
        
        Args:
            problem: FairnessRepairProblem instance
            init_population: Initial population (numpy array)
            n_gen: Number of generations
            seed: Random seed for reproducibility
        """
        # Determine n_partitions based on initial population size
        # For 2 objectives: n_partitions + 1 = pop_size
        pop_size = len(init_population)
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
        
        mutation = PolynomialMutation(prob=0.9, eta=20)
        crossover = UniformCrossover(prob=1.0)
        
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=min(self.n_neighbors, n_ref - 1),  # Ensure n_neighbors < pop_size
            prob_neighbor_mating=0.7,
            mutation=mutation,
            crossover=crossover,
            sampling=init_population
        )
        
        termination = get_termination('n_gen', n_gen)
        res = minimize(problem, algorithm, termination, seed=seed, verbose=True)
        self.results = res
        return res


class GradientRepair(RepairAlgorithm):
    """Gradient-based Pareto front generation using TensorFlow automatic differentiation
    
    Suggested by Reviewer 2: "The very choice of a genetic solver is questionable here: 
    gradient-based constrained optimisation can directly exploit differentiable violation 
    measures and routinely produces Pareto fronts of equal quality at a fraction of the 
    sampling cost."
    
    Key improvements over numerical gradients:
    1. Use TensorFlow's automatic differentiation (much faster)
    2. Use differentiable loss functions (cross-entropy based)
    3. Optimize only selected neurons (masked gradients)
    4. Multiple weight combinations for Pareto front
    
    Strategy:
    - For fairness: Use cross-entropy loss weighted by group
    - For accuracy: Use standard cross-entropy loss
    """
    
    def __init__(self, learning_rate=0.001, n_weights=11, n_iterations_per_weight=100):
        super().__init__("Gradient-Adam")
        self.learning_rate = learning_rate
        self.n_weights = n_weights
        self.n_iterations_per_weight = n_iterations_per_weight
    
    def repair(self, problem, init_population, n_gen, seed=None):
        """Generate Pareto front using gradient-based optimization with differentiable losses
        
        Strategy:
        1. Use differentiable loss functions based on cross-entropy
        2. Use tf.gradients for automatic differentiation
        3. Run optimization multiple times with different weight combinations
        4. Each run produces one Pareto point
        
        Args:
            problem: FairnessRepairProblem instance
            init_population: Initial population (numpy array)
            n_gen: Total number of gradient steps (divided among weight combinations)
            seed: Random seed for reproducibility
        """
        # Use n_gen to determine iterations per weight if not explicitly set
        if hasattr(self, 'n_iterations_per_weight') and self.n_iterations_per_weight is None:
            self.n_iterations_per_weight = max(10, n_gen // self.n_weights)
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
        
        # Prepare data
        test_x = problem.test_x
        test_y = problem.test_y
        
        # Convert labels to one-hot for cross-entropy
        n_classes = problem.model.output_shape[-1]
        test_y_onehot = tf.keras.utils.to_categorical(test_y, n_classes)
        
        # Build TensorFlow graph for gradient computation
        K.clear_session()
        
        try:
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
                    
                    # Placeholders for input data
                    X_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + test_x.shape[1:])
                    y_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, n_classes))
                    labels_ph = tf.compat.v1.placeholder(tf.int32, shape=(None,))  # Gender labels (0=female, 1=male)
                    weight_ph = tf.compat.v1.placeholder(tf.float32, shape=())
                    
                    # Model outputs
                    logits = model(X_ph)
                    
                    # Differentiable loss functions
                    # L_acc: Cross-entropy loss (minimize to maximize accuracy)
                    loss_acc = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits)
                    )
                    
                    # L_fairness: Difference between group-specific losses
                    # We want to minimize |loss_female - loss_male|
                    # This is a differentiable approximation of the fairness gap
                    
                    # Create masks for each group
                    female_mask = tf.cast(tf.equal(labels_ph, 0), tf.float32)
                    male_mask = tf.cast(tf.equal(labels_ph, 1), tf.float32)
                    
                    # Per-sample cross-entropy
                    per_sample_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits)
                    
                    # Group-specific losses
                    loss_female = tf.reduce_sum(per_sample_ce * female_mask) / (tf.reduce_sum(female_mask) + 1e-8)
                    loss_male = tf.reduce_sum(per_sample_ce * male_mask) / (tf.reduce_sum(male_mask) + 1e-8)
                    
                    # Fairness loss: absolute difference (use squared difference for differentiability)
                    loss_fairness = tf.square(loss_female - loss_male)
                    
                    # Combined loss: weighted sum
                    loss_total = weight_ph * loss_acc + (1.0 - weight_ph) * loss_fairness
                    
                    # Create masks to only optimize selected neurons
                    weight_to_neurons = {}
                    for layer_idx, neuron_idx, shape_idx in problem.player_indices:
                        if layer_idx not in weight_to_neurons:
                            weight_to_neurons[layer_idx] = []
                        weight_to_neurons[layer_idx].append(neuron_idx)
                    
                    # Get all trainable variables
                    all_trainable_vars = model.trainable_weights
                    
                    # Compute gradients
                    gradients = tf.gradients(loss_total, all_trainable_vars)
                    
                    # Create masked gradients
                    masked_gradients = []
                    
                    # Build mapping from trainable var to weight index
                    all_weights = model.get_weights()
                    trainable_var_to_weight_idx = {}
                    trainable_idx = 0
                    for weight_idx, weight in enumerate(all_weights):
                        if trainable_idx < len(all_trainable_vars):
                            var = all_trainable_vars[trainable_idx]
                            if tuple(weight.shape) == tuple(var.shape.as_list()):
                                trainable_var_to_weight_idx[trainable_idx] = weight_idx
                                trainable_idx += 1
                    
                    print(f"Debug: Total trainable vars: {len(all_trainable_vars)}")
                    print(f"Debug: Weight indices with selected neurons: {list(weight_to_neurons.keys())}")
                    
                    n_masked_layers = 0
                    for var_idx, (grad, var) in enumerate(zip(gradients, all_trainable_vars)):
                        if grad is None:
                            masked_gradients.append(None)
                            continue
                        
                        weight_idx = trainable_var_to_weight_idx.get(var_idx, None)
                        
                        if weight_idx is not None and weight_idx in weight_to_neurons:
                            # This layer has selected neurons - create a mask
                            mask_shape = var.shape.as_list()
                            mask = np.zeros(mask_shape, dtype=np.float32)
                            
                            # Enable gradient for selected neurons (last dimension)
                            for neuron_idx in weight_to_neurons[weight_idx]:
                                mask[..., neuron_idx] = 1.0
                            
                            mask_tensor = tf.constant(mask, dtype=tf.float32)
                            masked_grad = grad * mask_tensor
                            masked_gradients.append(masked_grad)
                            n_masked_layers += 1
                        else:
                            # Not a selected layer - zero out gradient
                            masked_gradients.append(tf.zeros_like(grad))
                    
                    print(f"Debug: Total layers with non-zero gradients: {n_masked_layers}/{len(all_trainable_vars)}")
                    
                    # Create optimizer with masked gradients + global norm clipping
                    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
                    grads_for_clip = [g if g is not None else tf.zeros_like(v) for g, v in zip(masked_gradients, all_trainable_vars)]
                    clipped_grads, _ = tf.clip_by_global_norm(grads_for_clip, 1.0)
                    train_op = optimizer.apply_gradients(zip(clipped_grads, all_trainable_vars))
                    
                    # Only initialize optimizer variables
                    optimizer_vars = optimizer.variables()
                    sess.run(tf.compat.v1.variables_initializer(optimizer_vars))
                    
                    # Verify model weights
                    print("\nVerifying model weights after loading...")
                    pred_test = sess.run(logits, feed_dict={X_ph: test_x[:100]})
                    pred_labels = np.argmax(pred_test, axis=-1)
                    acc_test = np.mean(pred_labels == test_y[:100])
                    print(f"Initial model performance: Acc={acc_test:.4f}")
                    
                    if acc_test < 0.5:
                        print("WARNING: Model accuracy is very low! Model weights may not be loaded correctly.")
                        model.set_weights([w.copy() for w in problem.base_weights])
                        pred_test = sess.run(logits, feed_dict={X_ph: test_x[:100]})
                        pred_labels = np.argmax(pred_test, axis=-1)
                        acc_test = np.mean(pred_labels == test_y[:100])
                        print(f"After reload: Acc={acc_test:.4f}")
                    
                    # Storage for Pareto front
                    pareto_solutions = []
                    pareto_objectives = []
                    
                    # Generate weight combinations
                    weights = np.linspace(0.0, 1.0, self.n_weights)
                    
                    for i, weight_combo in enumerate(weights):
                        print(f"\n{'─'*80}")
                        print(f"Weight combination {i+1}/{self.n_weights}: w_acc={weight_combo:.3f}, w_fairness={1-weight_combo:.3f}")
                        print(f"{'─'*80}")
                        
                        # Reset to base weights
                        sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))
                        base_weights_copy = [w.copy() for w in problem.base_weights]
                        model.set_weights(base_weights_copy)
                        
                        # Gradient descent loop with balanced mini-batch training
                        batch_size = min(512, len(test_x))
                        # Precompute gender indices for balanced sampling
                        gender0_idx = np.where(test_y == 0)[0]
                        gender1_idx = np.where(test_y == 1)[0]
                        
                        for iteration in range(self.n_iterations_per_weight):
                            # Balanced sampling by gender
                            half = batch_size // 2
                            sel0 = np.random.choice(gender0_idx, half, replace=True if len(gender0_idx) < half else False)
                            sel1 = np.random.choice(gender1_idx, batch_size - half, replace=True if len(gender1_idx) < (batch_size - half) else False)
                            batch_idx = np.concatenate([sel0, sel1])
                            
                            # Run one gradient step
                            _, loss_val, loss_acc_val, loss_fairness_val = sess.run(
                                [train_op, loss_total, loss_acc, loss_fairness],
                                feed_dict={
                                    X_ph: test_x[batch_idx],
                                    y_ph: test_y_onehot[batch_idx],
                                    labels_ph: test_y[batch_idx],
                                    weight_ph: weight_combo
                                }
                            )
                            
                            # Print progress
                            if iteration % 20 == 0 or iteration == self.n_iterations_per_weight - 1:
                                # Evaluate on a sample
                                eval_size = min(1000, len(test_x))
                                eval_idx = np.random.choice(len(test_x), eval_size, replace=False)
                                
                                pred_logits = sess.run(logits, feed_dict={X_ph: test_x[eval_idx]})
                                pred_labels = np.argmax(pred_logits, axis=-1)
                                corr_predictions = pred_labels == test_y[eval_idx]
                                
                                # Compute actual metrics
                                accuracy = np.mean(corr_predictions)
                                
                                # Fairness gap
                                female_mask_np = (test_y[eval_idx] == 0)
                                male_mask_np = (test_y[eval_idx] == 1)
                                female_acc = np.sum(corr_predictions & female_mask_np) / (np.sum(female_mask_np) + 1e-8)
                                male_acc = np.sum(corr_predictions & male_mask_np) / (np.sum(male_mask_np) + 1e-8)
                                fairness_gap = abs(female_acc - male_acc)
                                
                                print(f"  Iter {iteration:3d}: Loss={loss_val:.4f}, "
                                      f"Fairness={fairness_gap:.6f}, Acc={accuracy:.6f}")
                        
                        # Extract final solution
                        final_weights = model.get_weights()
                        
                        # Convert to decision variable format
                        total_size = sum(np.prod(shape) for shape in problem.shapes)
                        x_final = np.zeros(total_size)
                        begin_index = 0
                        for layer_idx, neuron_idx, shape_idx in problem.player_indices:
                            end_index = begin_index + np.prod(problem.shapes[shape_idx])
                            x_final[begin_index:end_index] = final_weights[layer_idx][..., neuron_idx].flatten()
                            begin_index = end_index
                        
                        # Evaluate final solution directly in current session
                        # (avoid cross-graph issues with problem._evaluate)
                        pred_logits = sess.run(logits, feed_dict={X_ph: test_x})
                        pred_labels = np.argmax(pred_logits, axis=-1)
                        corr_predictions = pred_labels == test_y
                        
                        # Compute actual metrics
                        accuracy = np.mean(corr_predictions)
                        
                        # Fairness gap
                        female_mask_np = (test_y == 0)
                        male_mask_np = (test_y == 1)
                        female_acc = np.sum(corr_predictions & female_mask_np) / (np.sum(female_mask_np) + 1e-8)
                        male_acc = np.sum(corr_predictions & male_mask_np) / (np.sum(male_mask_np) + 1e-8)
                        fairness_gap = abs(female_acc - male_acc)
                        
                        f_final = np.array([fairness_gap, -accuracy])
                        
                        pareto_solutions.append(x_final)
                        pareto_objectives.append(f_final)
                        
                        print(f"  ✓ Final: Fairness={f_final[0]:.6f}, Neg_Acc={f_final[1]:.6f} (Acc={-f_final[1]:.6f})")
                        
                        # Check if perfect
                        if f_final[0] <= problem.threshold_fairness and -f_final[1] >= problem.threshold_acc:
                            print(f"  🎯 Perfect solution found!")
            
            print(f"\n{'='*80}")
            print(f"Pareto front generation completed")
            print(f"Generated {len(pareto_solutions)} solutions")
            print(f"{'='*80}\n")
            
            # Close session
            sess.close()
        
        except Exception as e:
            print(f"ERROR in GradientRepair: {e}")
            import traceback
            traceback.print_exc()
            # Return empty result on error
            self.results = GradientResult(
                X=np.array([]).reshape(0, sum(np.prod(shape) for shape in problem.shapes)),
                F=np.array([]).reshape(0, 2)
            )
            return self.results
        
        # Create result object compatible with pymoo
        class GradientResult:
            def __init__(self, X, F):
                self.X = X
                self.F = F
        
        if len(pareto_solutions) == 0:
            print("ERROR: No solutions generated!")
            # Return empty result
            self.results = GradientResult(
                X=np.array([]).reshape(0, sum(np.prod(shape) for shape in problem.shapes)),
                F=np.array([]).reshape(0, 2)
            )
        else:
            self.results = GradientResult(
                X=np.array(pareto_solutions),
                F=np.array(pareto_objectives)
            )
        
        return self.results
