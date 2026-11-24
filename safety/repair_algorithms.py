"""
Repair algorithms for safety task
Supports: NSGA-II, MOEA/D, Gradient-based (Adam)

Note: NSGA-III removed as reviewers suggested NSGA-II is more appropriate for 2-objective problems.
CMA-ES not included as it was not mentioned by reviewers.
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


class SafetyRepairProblem(Problem):
    """Multi-objective optimization problem for safety repair
    
    Optimized version following safety_moo_2.py:
    - Model reuse (avoid repeated loading)
    - Weight caching
    - Pre-computed indices
    - Perfect solution tracking
    """
    
    def __init__(self, model_path, init_weights, selected_players, shapes, 
                 convs, denses, ct_x, dr_x, model_name, threshold_acc=0.95, threshold_safety=0.01):
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
        self.ct_x = ct_x
        self.dr_x = dr_x
        self.model_name = model_name
        self.threshold_acc = threshold_acc
        self.threshold_safety = threshold_safety
        
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
        f1_list = []  # safety violation rate
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
            safety_rate = self._compute_safety(self.model, self.ct_x)
            accuracy = self._compute_accuracy(self.model, self.dr_x)
            
            # Track perfect solutions
            if accuracy == 1.0 and safety_rate == 0.0:
                self.perfect_solutions.append({
                    'weights': x[i].copy(),
                    'safety_rate': safety_rate,
                    'accuracy': accuracy,
                    'evaluation': self.evaluation_count
                })
                print(f"Perfect solution found at evaluation {self.evaluation_count}!")
            
            f1_list.append(safety_rate)
            f2_list.append(-accuracy)  # Negative for minimization
        
        out['F'] = np.column_stack([f1_list, f2_list])
    
    def _compute_safety(self, model, images, batch_size=1024):
        """Compute safety violation rate"""
        output = self._sess_run(model, model.output, images, batch_size)
        pred = np.argmin(output, axis=-1)
        
        if self.model_name == 'a19':
            val = (np.sum(pred == 3) + np.sum(pred == 4)) / len(output)
        elif self.model_name == 'a29':
            val = (len(images) - np.sum(pred == 0) - np.sum(pred == 1)) / len(output)
        else:
            val = (np.sum(pred != 0)) / len(output)
        return val
    
    def _compute_accuracy(self, model, images, batch_size=1024):
        """Compute drawndown accuracy"""
        output = self._sess_run(model, model.output, images, batch_size)
        pred = np.argmin(output, axis=-1)
        
        if self.model_name == 'a19':
            val = (len(output) - np.sum(pred == 3) - np.sum(pred == 4)) / len(output)
        elif self.model_name == 'a29':
            val = (np.sum(pred == 0) + np.sum(pred == 1)) / len(output)
        else:
            val = np.sum(pred == 0) / len(output)
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
            problem: SafetyRepairProblem instance
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
            problem: SafetyRepairProblem instance
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
    - For safety: Use softmax output and target specific classes
    - For accuracy: Use softmax output and target correct predictions
    """
    
    def __init__(self, learning_rate=0.001, n_weights=11, n_iterations_per_weight=100):
        super().__init__("Gradient-Adam")
        self.learning_rate = learning_rate
        self.n_weights = n_weights
        self.n_iterations_per_weight = n_iterations_per_weight
    
    def repair(self, problem, init_population, n_gen, seed=None):
        """Generate Pareto front using gradient-based optimization with differentiable losses
        
        Strategy:
        1. Use differentiable loss functions based on model outputs
        2. Use tf.gradients for automatic differentiation
        3. Run optimization multiple times with different weight combinations
        4. Each run produces one Pareto point
        
        Args:
            problem: SafetyRepairProblem instance
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
        
        # Prepare data
        ct_x = problem.ct_x  # Counterexample data (safety violations)
        dr_x = problem.dr_x  # Drawndown data (accuracy)
        model_name = problem.model_name
        
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
                
                # Placeholders for input data
                ct_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + ct_x.shape[1:])
                dr_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + dr_x.shape[1:])
                weight_ph = tf.compat.v1.placeholder(tf.float32, shape=())
                
                # Model outputs (note: safety models output distances, use argmin for prediction)
                logits_ct = model(ct_ph)
                logits_dr = model(dr_ph)
                
                # Differentiable loss functions
                # For safety: We want to minimize violations
                # The original uses argmin and checks specific classes
                # We approximate this with a differentiable loss
                
                if model_name == 'a19':
                    # Safety: minimize P(class=3) + P(class=4)
                    # Accuracy: maximize P(class!=3 and class!=4)
                    probs_ct = tf.nn.softmax(-logits_ct)  # Negative because argmin
                    probs_dr = tf.nn.softmax(-logits_dr)
                    loss_safety = tf.reduce_mean(probs_ct[:, 3] + probs_ct[:, 4])
                    loss_accuracy = -tf.reduce_mean(probs_dr[:, 0] + probs_dr[:, 1] + probs_dr[:, 2])
                    
                elif model_name == 'a29':
                    # Safety: maximize P(class=0 or class=1)
                    # Accuracy: maximize P(class=0 or class=1)
                    probs_ct = tf.nn.softmax(-logits_ct)
                    probs_dr = tf.nn.softmax(-logits_dr)
                    loss_safety = -tf.reduce_mean(probs_ct[:, 0] + probs_ct[:, 1])
                    loss_accuracy = -tf.reduce_mean(probs_dr[:, 0] + probs_dr[:, 1])
                    
                else:  # a09
                    # Safety: maximize P(class=0)
                    # Accuracy: maximize P(class=0)
                    probs_ct = tf.nn.softmax(-logits_ct)
                    probs_dr = tf.nn.softmax(-logits_dr)
                    loss_safety = -tf.reduce_mean(probs_ct[:, 0])
                    loss_accuracy = -tf.reduce_mean(probs_dr[:, 0])
                
                # Combined loss: weighted sum
                loss_total = weight_ph * loss_accuracy + (1.0 - weight_ph) * loss_safety
                
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
                
                # Create optimizer with masked gradients
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_op = optimizer.apply_gradients(zip(masked_gradients, all_trainable_vars))
                
                # Only initialize optimizer variables
                optimizer_vars = optimizer.variables()
                sess.run(tf.compat.v1.variables_initializer(optimizer_vars))
                
                # Verify model weights
                print("\nVerifying model weights after loading...")
                pred_ct_test = sess.run(logits_ct, feed_dict={ct_ph: ct_x[:100]})
                pred_dr_test = sess.run(logits_dr, feed_dict={dr_ph: dr_x[:100]})
                print(f"Initial outputs - CT shape: {pred_ct_test.shape}, DR shape: {pred_dr_test.shape}")
                
                # Storage for Pareto front
                pareto_solutions = []
                pareto_objectives = []
                
                # Generate weight combinations
                weights = np.linspace(0.0, 1.0, self.n_weights)
                
                for i, weight_combo in enumerate(weights):
                    print(f"\n{'─'*80}")
                    print(f"Weight combination {i+1}/{self.n_weights}: w_acc={weight_combo:.3f}, w_safety={1-weight_combo:.3f}")
                    print(f"{'─'*80}")
                    
                    # Reset to base weights
                    sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))
                    base_weights_copy = [w.copy() for w in problem.base_weights]
                    model.set_weights(base_weights_copy)
                    
                    # Gradient descent loop with mini-batch training
                    batch_size = min(512, len(ct_x))
                    
                    for iteration in range(self.n_iterations_per_weight):
                        # Sample random batch
                        batch_idx_ct = np.random.choice(len(ct_x), batch_size, replace=False)
                        batch_idx_dr = np.random.choice(len(dr_x), batch_size, replace=False)
                        
                        # Run one gradient step
                        _, loss_val, loss_safety_val, loss_acc_val = sess.run(
                            [train_op, loss_total, loss_safety, loss_accuracy],
                            feed_dict={
                                ct_ph: ct_x[batch_idx_ct],
                                dr_ph: dr_x[batch_idx_dr],
                                weight_ph: weight_combo
                            }
                        )
                        
                        # Print progress
                        if iteration % 20 == 0 or iteration == self.n_iterations_per_weight - 1:
                            # Evaluate on a sample
                            eval_size = min(1000, len(ct_x))
                            eval_idx_ct = np.random.choice(len(ct_x), eval_size, replace=False)
                            eval_idx_dr = np.random.choice(len(dr_x), eval_size, replace=False)
                            
                            pred_ct = sess.run(logits_ct, feed_dict={ct_ph: ct_x[eval_idx_ct]})
                            pred_dr = sess.run(logits_dr, feed_dict={dr_ph: dr_x[eval_idx_dr]})
                            
                            # Compute actual metrics
                            pred_ct_class = np.argmin(pred_ct, axis=-1)
                            pred_dr_class = np.argmin(pred_dr, axis=-1)
                            
                            if model_name == 'a19':
                                safety_rate = (np.sum(pred_ct_class == 3) + np.sum(pred_ct_class == 4)) / len(pred_ct)
                                accuracy = (len(pred_dr) - np.sum(pred_dr_class == 3) - np.sum(pred_dr_class == 4)) / len(pred_dr)
                            elif model_name == 'a29':
                                safety_rate = (len(pred_ct) - np.sum(pred_ct_class == 0) - np.sum(pred_ct_class == 1)) / len(pred_ct)
                                accuracy = (np.sum(pred_dr_class == 0) + np.sum(pred_dr_class == 1)) / len(pred_dr)
                            else:
                                safety_rate = np.sum(pred_ct_class != 0) / len(pred_ct)
                                accuracy = np.sum(pred_dr_class == 0) / len(pred_dr)
                            
                            print(f"  Iter {iteration:3d}: Loss={loss_val:.4f}, "
                                  f"Safety={safety_rate:.6f}, Acc={accuracy:.6f}")
                    
                    # Extract final solution
                    final_weights = model.get_weights()

                    # Convert to decision variable format (for saving later)
                    total_size = sum(np.prod(shape) for shape in problem.shapes)
                    x_final = np.zeros(total_size)
                    begin_index = 0
                    for layer_idx, neuron_idx, shape_idx in problem.player_indices:
                        end_index = begin_index + np.prod(problem.shapes[shape_idx])
                        x_final[begin_index:end_index] = final_weights[layer_idx][..., neuron_idx].flatten()
                        begin_index = end_index

                    # Evaluate final solution within the SAME graph/session to avoid cross-graph errors
                    # Compute logits on full datasets in batches
                    def run_in_batches(data, tensor, placeholder, bs=1024):
                        outs = []
                        n = len(data)
                        for b in range(0, n, bs):
                            outs.append(sess.run(tensor, feed_dict={placeholder: data[b:b+bs]}))
                        return np.concatenate(outs, axis=0)

                    pred_ct = run_in_batches(ct_x, logits_ct, ct_ph)
                    pred_dr = run_in_batches(dr_x, logits_dr, dr_ph)

                    pred_ct_class = np.argmin(pred_ct, axis=-1)
                    pred_dr_class = np.argmin(pred_dr, axis=-1)

                    if model_name == 'a19':
                        safety_rate = (np.sum(pred_ct_class == 3) + np.sum(pred_ct_class == 4)) / len(pred_ct)
                        accuracy = (len(pred_dr) - np.sum(pred_dr_class == 3) - np.sum(pred_dr_class == 4)) / len(pred_dr)
                    elif model_name == 'a29':
                        safety_rate = (len(pred_ct) - np.sum(pred_ct_class == 0) - np.sum(pred_ct_class == 1)) / len(pred_ct)
                        accuracy = (np.sum(pred_dr_class == 0) + np.sum(pred_dr_class == 1)) / len(pred_dr)
                    else:
                        safety_rate = np.sum(pred_ct_class != 0) / len(pred_ct)
                        accuracy = np.sum(pred_dr_class == 0) / len(pred_dr)

                    # Objective vector: [safety_rate, -accuracy] (minimization)
                    f_final = np.array([safety_rate, -accuracy], dtype=np.float32)
                    
                    pareto_solutions.append(x_final)
                    pareto_objectives.append(f_final)
                    
                    print(f"  ✓ Final: Safety={f_final[0]:.6f}, Neg_Acc={f_final[1]:.6f} (Acc={-f_final[1]:.6f})")
                    
                    # Check if perfect
                    if f_final[0] <= problem.threshold_safety and -f_final[1] >= problem.threshold_acc:
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
        
        self.results = GradientResult(
            X=np.array(pareto_solutions),
            F=np.array(pareto_objectives)
        )
        
        return self.results

