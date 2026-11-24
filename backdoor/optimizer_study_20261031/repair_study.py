"""CGARepair backdoor LOCAL WEIGHT repair: one CLI, MOEA/D or true MOPSO.

Examples:
  python backdoor_moo_2.py mnist 42 --optimizer moead --quick
  python backdoor_moo_2.py mnist 42 --optimizer mopso --quick
  python backdoor_moo_2.py gtsrb 42 --resnet --optimizer mopso

Read the original CGARepair ranking cache; preserve its adaptive 1% (4..11)
selection. Both optimizers evaluate exactly the same training objective, model,
selected weights and SR definition. Test checkpoints with backdoor_test_2.py.
Python >=3.7. MOEA/D needs the same pymoo package as the user's original script;
MOPSO is implemented below (external Pareto archive + crowding selection).
"""
import argparse
import csv
from datetime import datetime, timezone
import math
import os
import pickle
import re
import time
from pathlib import Path
import numpy as np

from read_CGARepair import load_ranking, export_ranking, resolve_model_name, model_tag
from cgarepair_backdoor_utils import (DATASET_CONFIG, initialize_tf, load_dataset,
    model_file, trigger_inputs, select_subset, calculate_metrics, save_json)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def run_directory(root, tag, seed, optimizer, run_tag='', quick=False):
    suffix = ('_' + run_tag) if run_tag else ('_quick' if quick else '')
    return Path(root).expanduser() / tag / 'CGARepair' / ('seed_' + str(seed)) / (optimizer + suffix)


def map_players_to_weights(model, selected_players):
    """Map real Keras layer indices to global model.get_weights() offsets.

    IMPORTANT: count BatchNorm/bias/other layer arrays too; the old script skipped
    their offsets, corrupting ResNet layer selections.
    """
    flattened = model.get_weights()
    mapping = {}
    offset = 0
    for layer_index, layer in enumerate(model.layers):
        local = layer.get_weights()
        if local and ('Conv' in type(layer).__name__ or 'Dense' in type(layer).__name__):
            mapping[layer_index] = (offset, local[0], type(layer).__name__)
        offset += len(local)
    if offset != len(flattened):
        raise RuntimeError('model.get_weights offset mismatch {} != {}; nested layers unsupported'.format(
            offset, len(flattened)))
    specs = []
    used_slices = set()
    total = 0
    import re
    for player in selected_players:
        match = re.fullmatch(r'(Conv\w*|Dense)_(\d+)_(\d+)', str(player), re.I)
        if match is None:
            raise ValueError('Bad CGARepair layer identifier: {}'.format(player))
        layer_type, layer_idx, unit = match.group(1), int(match.group(2)), int(match.group(3))
        if layer_idx not in mapping:
            raise KeyError('Selected layer {} does not appear in model mapping'.format(player))
        weight_idx, weight, actual_type = mapping[layer_idx]
        if layer_type.lower() != actual_type.lower():
            raise ValueError('Layer type mismatch for {}: actual {}'.format(player, actual_type))
        if weight.ndim < 2 or weight.ndim > 5:
            raise ValueError('Unsupported kernel dimensionality: {} / {}'.format(player, weight.shape))
        if unit < 0 or unit >= weight.shape[-1]:
            raise ValueError('Neuron index out of range for {} with kernel {}'.format(player, weight.shape))
        output_shape = getattr(model.layers[layer_idx], 'output_shape', None)
        if output_shape is not None:
            if isinstance(output_shape, tuple) and output_shape[-1] is not None:
                if int(output_shape[-1]) != int(weight.shape[-1]):
                    raise ValueError('Output channel != kernel last axis for {}: {}; '
                                     'depthwise/separable layers need explicit handling'.format(player, weight.shape))
        if (weight_idx, unit) in used_slices:
            raise ValueError('Duplicate parameter slice for {}'.format(player))
        used_slices.add((weight_idx, unit))
        shape = tuple(weight[..., unit].shape)
        n_values = int(np.prod(shape))
        specs.append({'player': player, 'layer_index': layer_idx, 'weight_index': weight_idx,
                      'neuron_index': unit, 'shape': shape, 'start': total, 'stop': total+n_values,
                      'num_parameters': n_values})
        total += n_values
    if not specs:
        raise ValueError('No selected repair parameters')
    return specs


def original_vector(weights, specs):
    return np.concatenate([np.asarray(weights[s['weight_index']][..., s['neuron_index']])
                           .reshape(-1) for s in specs]).astype(np.float64)


def apply_vector(weights, specs, x):
    x = np.asarray(x)
    for spec in specs:
        weights[spec['weight_index']][..., spec['neuron_index']] = (
            x[spec['start']:spec['stop']].reshape(spec['shape']))
    return weights


class RepairEvaluator:
    def __init__(self, model, session, specs, clean, labels, triggered, model_name,
                 sr_definition='legacy', batch_size=512):
        self.model = model
        self.session = session
        self.specs = specs
        self.model_name = model_name
        self.sr_definition = sr_definition
        self.batch_size = int(batch_size)
        self.clean = clean
        self.labels = labels
        self.triggered = triggered
        self.target_label = DATASET_CONFIG[model_name]['target_label']
        self.base_weights = [w.copy() for w in model.get_weights()]
        self.work_weights = [w.copy() for w in self.base_weights]
        self.evaluations = 0
        self.eval_wall_sec = 0.0
        self.traces = []

    def evaluate_batch(self, X):
        vectors = np.atleast_2d(np.asarray(X, dtype=np.float64))
        values = np.zeros((len(vectors), 2), dtype=np.float64)
        for i, vector in enumerate(vectors):
            start = time.perf_counter()
            apply_vector(self.work_weights, self.specs, vector)
            self.model.set_weights(self.work_weights)
            v = calculate_metrics(self.model, self.clean, self.labels, self.triggered,
                                  self.target_label, self.session,
                                  sr_definition=self.sr_definition, batch_size=self.batch_size)
            values[i] = [v['sr'], -v['clean_accuracy']]
            self.evaluations += 1
            self.eval_wall_sec += time.perf_counter()-start
        return values

    def baseline_metrics(self):
        self.model.set_weights(self.base_weights)
        return calculate_metrics(self.model, self.clean, self.labels, self.triggered,
                                 self.target_label, self.session,
                                 sr_definition=self.sr_definition, batch_size=self.batch_size)

    def save_vector_weights(self, vector, filename):
        apply_vector(self.work_weights, self.specs, vector)
        self.model.set_weights(self.work_weights)
        self.model.save_weights(str(filename))

    def reset(self):
        self.model.set_weights(self.base_weights)


def dominates(a, b):
    return bool(np.all(a <= b + 1.e-12) and np.any(a < b - 1.e-12))


def non_dominated_indices(F):
    F = np.atleast_2d(np.asarray(F, dtype=float))
    alive = []
    for i in range(len(F)):
        if any(dominates(F[j], F[i]) for j in range(len(F)) if j != i):
            continue
        # Equal objective vectors are redundant for front evaluation; preserve first.
        if any(np.allclose(F[i], F[j], atol=1e-12, rtol=0.0) for j in alive):
            continue
        alive.append(i)
    return np.asarray(alive, dtype=int)


def crowding_distance(F):
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n = len(F)
    d = np.zeros(n, dtype=float)
    if n <= 2:
        return np.full(n, np.inf)
    for objective in range(F.shape[1]):
        order = np.argsort(F[:, objective], kind='mergesort')
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        span = F[order[-1], objective] - F[order[0], objective]
        if span > 1.e-14:
            for i in range(1, n-1):
                d[order[i]] += (F[order[i+1], objective] - F[order[i-1], objective]) / span
    return d


def reduce_archive(X, F, max_size):
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)
    ids = non_dominated_indices(F)
    X, F = X[ids], F[ids]
    while len(X) > max_size:
        crowd = crowding_distance(F)
        finite = np.where(np.isfinite(crowd))[0]
        if not len(finite):
            break
        drop = finite[np.argmin(crowd[finite])]
        X = np.delete(X, drop, axis=0)
        F = np.delete(F, drop, axis=0)
    if len(X) > max_size:
        X, F = X[:max_size], F[:max_size]
    return X, F


def choose_leader(archive_x, archive_f, rng):
    if len(archive_x) == 1:
        return archive_x[0]
    a, b = rng.choice(len(archive_x), size=2, replace=True)
    crowd = crowding_distance(archive_f)
    return archive_x[a if crowd[a] >= crowd[b] else b]


def run_mopso(evaluator, x0, lower, upper, pop_size, max_evals, args, seed):
    """Proper multiobjective PSO: Pareto archive, crowding leaders, personal bests."""
    rng = np.random.RandomState(int(seed))
    span = upper - lower
    positions = np.clip(x0 + rng.uniform(-args.init_jitter, args.init_jitter,
                                         (pop_size, len(x0))), lower, upper)
    velocities = rng.uniform(-0.005, 0.005, positions.shape) * span
    velocity_bound = args.vmax_fraction * span
    objectives = evaluator.evaluate_batch(positions)
    pbest_pos, pbest_f = positions.copy(), objectives.copy()
    archive_x, archive_f = reduce_archive(positions, objectives, args.archive_size)
    history = []
    generation = 1
    while evaluator.evaluations < max_evals:
        trial_n = min(pop_size, max_evals-evaluator.evaluations)
        next_pos = positions.copy()
        r1 = rng.uniform(size=(trial_n, len(x0)))
        r2 = rng.uniform(size=(trial_n, len(x0)))
        leader = np.stack([choose_leader(archive_x, archive_f, rng) for _ in range(trial_n)])
        velocities[:trial_n] = (args.inertia * velocities[:trial_n]
                                + args.c1 * r1 * (pbest_pos[:trial_n]-positions[:trial_n])
                                + args.c2 * r2 * (leader-positions[:trial_n]))
        velocities[:trial_n] = np.clip(velocities[:trial_n], -velocity_bound, velocity_bound)
        next_pos[:trial_n] = np.clip(positions[:trial_n] + velocities[:trial_n], lower, upper)
        # Bounce against box boundaries, preventing continued outward motion.
        touching = (next_pos[:trial_n] <= lower) | (next_pos[:trial_n] >= upper)
        velocities[:trial_n][touching] *= -0.5
        new_f = evaluator.evaluate_batch(next_pos[:trial_n])
        positions[:trial_n] = next_pos[:trial_n]
        for i in range(trial_n):
            if dominates(new_f[i], pbest_f[i]) or (not dominates(pbest_f[i], new_f[i])
                                                   and rng.random_sample() < 0.5):
                pbest_pos[i], pbest_f[i] = positions[i].copy(), new_f[i].copy()
        archive_x, archive_f = reduce_archive(
            np.concatenate([archive_x, positions[:trial_n]], axis=0),
            np.concatenate([archive_f, new_f], axis=0), args.archive_size)
        generation += 1
        if generation % args.log_every == 0 or evaluator.evaluations >= max_evals:
            best_sr = float(np.min(archive_f[:, 0]))
            max_acc = float(np.max(-archive_f[:, 1]))
            history.append({'generation': generation, 'evaluations': evaluator.evaluations,
                            'min_sr': best_sr, 'max_accuracy': max_acc,
                            'archive_size': int(len(archive_x))})
            print('[MOPSO] gen={} FEs={} archive={} minSR={:.6f} maxACC={:.6f}'.format(
                generation, evaluator.evaluations, len(archive_x), best_sr, max_acc), flush=True)
    return archive_x, archive_f, history


def run_moead(evaluator, x0, lower, upper, pop_size, max_evals, args, seed):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
    except ImportError as e:
        raise ImportError('MOEA/D requires the pymoo version compatible with the original '
                          'backdoor_moo_2.py. Import error: {}'.format(e))

    class PymooBackdoorProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(x0), n_obj=2, xl=lower, xu=upper, vtype=float)
        def _evaluate(self, X, out, *args_, **kwargs_):
            out['F'] = evaluator.evaluate_batch(X)

    ref_dirs = get_reference_directions('das-dennis', 2, n_partitions=pop_size-1)
    if len(ref_dirs) != pop_size:
        raise RuntimeError('Expected exactly {} MOEAD reference directions, got {}'.format(pop_size, len(ref_dirs)))
    rng = np.random.RandomState(int(seed))
    init_pop = np.clip(x0 + rng.uniform(-args.init_jitter, args.init_jitter,
                                        (pop_size, len(x0))), lower, upper)
    mutation_prob = (args.mutation_prob if args.mutation_prob is not None
                     else 0.9)
    algorithm = MOEAD(ref_dirs=ref_dirs,
                      n_neighbors=min(args.neighbors, pop_size-1),
                      prob_neighbor_mating=args.neighbor_prob,
                      mutation=PolynomialMutation(prob=mutation_prob, prob_var=1.0/len(x0), eta=args.mutation_eta),
                      crossover=UniformCrossover(prob=args.crossover_prob),
                      sampling=init_pop)
    problem = PymooBackdoorProblem()
    termination = get_termination('n_eval', int(max_evals))
    res = minimize(problem, algorithm, termination, seed=int(seed), verbose=False,
                   save_history=False)
    population = res.pop
    if population is None:
        raise RuntimeError('MOEA/D returned no population')
    X = np.asarray(population.get('X'), dtype=float)
    F = np.asarray(population.get('F'), dtype=float)
    X, F = reduce_archive(X, F, args.archive_size)
    if not len(X):
        raise RuntimeError('MOEA/D final population yielded empty nondominated front')
    history = [{'generation': int(getattr(res.algorithm, 'n_gen', 0)),
                'evaluations': int(evaluator.evaluations),
                'min_sr': float(np.min(F[:, 0])),
                'max_accuracy': float(np.max(-F[:, 1])),
                'archive_size': int(len(X))}]
    print('[MOEAD] generations={} FEs={} final_front={}'.format(
        history[0]['generation'], evaluator.evaluations, len(X)), flush=True)
    return X, F, history



# NSGA-III ablation: same TRAIN problem, initial population, mutation, crossover,
# selection policy and input bounds as native MOEA/D.
def run_nsga3(evaluator, x0, lower, upper, pop_size, max_evals, args, seed):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
    except ImportError as e:
        raise ImportError('NSGA-III requires the pymoo version compatible with the original '
                          'backdoor_moo_2.py. Import error: {}'.format(e))

    class PymooBackdoorProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(x0), n_obj=2, xl=lower, xu=upper, vtype=float)
        def _evaluate(self, X, out, *args_, **kwargs_):
            out['F'] = evaluator.evaluate_batch(X)

    ref_dirs = get_reference_directions('das-dennis', 2, n_partitions=pop_size-1)
    if len(ref_dirs) != pop_size:
        raise RuntimeError('Expected exactly {} MOEAD reference directions, got {}'.format(pop_size, len(ref_dirs)))
    rng = np.random.RandomState(int(seed))
    init_pop = np.clip(x0 + rng.uniform(-args.init_jitter, args.init_jitter,
                                        (pop_size, len(x0))), lower, upper)
    mutation_prob = (args.mutation_prob if args.mutation_prob is not None
                     else 0.9)
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs,
                      mutation=PolynomialMutation(prob=mutation_prob, prob_var=1.0/len(x0), eta=args.mutation_eta),
                      crossover=UniformCrossover(prob=args.crossover_prob),
                      sampling=init_pop, eliminate_duplicates=False)
    problem = PymooBackdoorProblem()
    termination = get_termination('n_eval', int(max_evals))
    res = minimize(problem, algorithm, termination, seed=int(seed), verbose=False,
                   save_history=False)
    population = res.pop
    if population is None:
        raise RuntimeError('NSGA-III returned no population')
    X = np.asarray(population.get('X'), dtype=float)
    F = np.asarray(population.get('F'), dtype=float)
    X, F = reduce_archive(X, F, args.archive_size)
    if not len(X):
        raise RuntimeError('NSGA-III final population yielded empty nondominated front')
    history = [{'generation': int(getattr(res.algorithm, 'n_gen', 0)),
                'evaluations': int(evaluator.evaluations),
                'min_sr': float(np.min(F[:, 0])),
                'max_accuracy': float(np.max(-F[:, 1])),
                'archive_size': int(len(X))}]
    print('[NSGA3] generations={} FEs={} final_front={}'.format(
        history[0]['generation'], evaluator.evaluations, len(X)), flush=True)
    return X, F, history



def pick_train_solution(F, threshold_acc, threshold_sr):
    """Use training objectives ONLY. Do not pick the best model using TEST metrics."""
    F = np.asarray(F)
    sr, acc = F[:, 0], -F[:, 1]
    meets_both = (sr <= threshold_sr) & (acc >= threshold_acc)
    meets_utility = acc >= threshold_acc
    if np.any(meets_both):
        pool = np.where(meets_both)[0]
        policy = 'feasible_min_sr_then_max_acc'
        order = np.lexsort((-acc[pool], sr[pool]))
    elif np.any(meets_utility):
        pool = np.where(meets_utility)[0]
        policy = 'acc_guard_min_sr_then_max_acc'
        order = np.lexsort((-acc[pool], sr[pool]))
    else:
        pool = np.arange(len(F))
        policy = 'no_utility_feasible_choose_max_acc_then_min_sr'
        order = np.lexsort((sr[pool], -acc[pool]))
    return int(pool[order[0]]), policy


def parser():
    p = argparse.ArgumentParser(description='CGARepair backdoor weight repair, MOEAD/MOPSO shared objectives')
    p.add_argument('model_name', help='mnist | fashion | gtsrb | gtsrb_resnet')
    p.add_argument('seed', type=int, help='optimization seed (separate from localization seed)')
    p.add_argument('legacy_resnet', nargs='?', default=None,
                   help='Legacy positional flag, e.g. gtsrb 42 True')
    p.add_argument('--resnet', action='store_true')
    p.add_argument('--optimizer', '--algorithm', choices=('moead', 'nsga3', 'mopso'), default='moead')
    p.add_argument('--loc-json', help='Explicit CGARepair_backdoor_samples512_seed256_v3.json')
    p.add_argument('--loc-root', help='Folder containing result/loc_result/<dataset>')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--model-root', help='Folder containing pretrained .h5 checkpoints')
    p.add_argument('--data-root', help='Folder containing img/ and gtsrb/ data')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-tag', default='', help='Label for hyperparameter experiment; separate output directory')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--quick', action='store_true', help='Smoke test: 12 individuals, 36 FEs, 128 samples')
    p.add_argument('--pop-size', type=int, default=40)
    p.add_argument('--max-gen', type=int, default=40)
    p.add_argument('--max-evals', type=int, default=None,
                   help='Common max fitness evaluations for both optimizers; default pop_size*max_gen')
    p.add_argument('--eval-samples', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--search-radius', type=float, default=0.5,
                   help='Same absolute per-parameter bounds centered at pretrained weights')
    p.add_argument('--init-jitter', type=float, default=0.01)
    p.add_argument('--sr-definition', choices=('legacy', 'standard'), default='legacy',
                   help='legacy: non-target success / ALL images (original code); '
                        'standard: non-target success / non-target images')
    p.add_argument('--threshold-acc', type=float, default=None)
    p.add_argument('--threshold-sr', type=float, default=None)
    p.add_argument('--save-front-limit', type=int, default=10,
                   help='Max Pareto weight .h5 files (best checkpoint saved regardless)')
    p.add_argument('--archive-size', type=int, default=60)
    p.add_argument('--log-every', type=int, default=5)
    # MOEA/D
    p.add_argument('--neighbors', type=int, default=15)
    p.add_argument('--neighbor-prob', type=float, default=0.7)
    p.add_argument('--mutation-eta', type=float, default=20.0)
    p.add_argument('--mutation-prob', type=float, default=0.9)
    p.add_argument('--crossover-prob', type=float, default=1.0)
    # MOPSO
    p.add_argument('--inertia', type=float, default=0.6)
    p.add_argument('--c1', type=float, default=1.5)
    p.add_argument('--c2', type=float, default=1.5)
    p.add_argument('--vmax-fraction', type=float, default=0.1)
    return p


def run(args):
    wall_start = time.perf_counter()
    is_resnet = args.resnet or str(args.legacy_resnet).lower() in ('true', '1', 'yes', 'y')
    model_name, is_resnet = resolve_model_name(args.model_name, is_resnet)
    tag = model_tag(model_name, is_resnet)
    if args.quick:
        args.pop_size, args.max_gen, args.eval_samples = 12, 3, 128
        if args.max_evals is None:
            args.max_evals = 36
    if args.pop_size < 3 or args.eval_samples < 1 or args.max_gen < 1:
        raise ValueError('pop-size>=3, eval-samples>=1, max-gen>=1 required')
    max_evals = int(args.max_evals if args.max_evals is not None else args.pop_size*args.max_gen)
    if max_evals < args.pop_size:
        raise ValueError('--max-evals cannot be smaller than population size')
    if args.search_radius <= 0.0 or args.init_jitter < 0.0:
        raise ValueError('search-radius must be positive, init-jitter must be nonnegative')
    if args.archive_size < 2 or args.save_front_limit < 0:
        raise ValueError('archive-size must be >=2 and save-front-limit >=0')
    run_dir = run_directory(args.output_root, tag, args.seed, args.optimizer, args.run_tag, args.quick)
    summary_path = run_dir / 'summary.json'
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError('Existing {}. Use --run-tag for a new trial or --overwrite to replace.'.format(summary_path))
    run_dir.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    players, scores, loc = load_ranking(model_name, is_resnet,
        args.loc_seed, args.loc_samples, args.loc_json, args.loc_root)
    exported = export_ranking(players, scores, loc, export_dir=run_dir / 'localization_input')
    localization_read_sec = time.perf_counter()-t
    print('[Locator] {} Top-{} from {} (loc_seed={}, opt_seed={})'.format(
        tag, loc['top_k'], loc['source_json'], args.loc_seed, args.seed), flush=True)
    print('[Locator] selected={}'.format(players), flush=True)

    t = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    from tensorflow.keras.models import load_model
    model_path = model_file(model_name, is_resnet, args.model_root)
    model = load_model(model_path, compile=False)
    src_signature = loc.get('source_model_signature')
    if isinstance(src_signature, (list, tuple)) and len(src_signature) >= 1:
        if int(src_signature[0]) != os.stat(model_path).st_size:
            raise ValueError('Localization and optimizer checkpoint sizes do not match: {} vs {}. '
                             'Use the same model as CGARepair localization.'.format(
                                 src_signature[0], os.stat(model_path).st_size))
    specs = map_players_to_weights(model, players)
    original_weights = model.get_weights()
    x0 = original_vector(original_weights, specs)
    model_setup_sec = time.perf_counter()-t
    print('[Repair] selected_neurons={} decision_variables={} model={}'.format(
        len(specs), len(x0), model_path), flush=True)

    t = time.perf_counter()
    x, y = load_dataset(model_name, 'train', args.data_root, session)
    x_p = trigger_inputs(x, model_name, args.data_root)
    ex_x, ex_y, ex_p = select_subset(x, y, x_p, args.eval_samples, args.seed)
    del x, y, x_p
    training_data_sec = time.perf_counter()-t
    evaluator = RepairEvaluator(model, session, specs, ex_x, ex_y, ex_p, model_name,
                                args.sr_definition, args.batch_size)
    t = time.perf_counter()
    baseline = evaluator.baseline_metrics()
    baseline_sec = time.perf_counter()-t
    print('[Baseline] {} SR={:.6f}, Acc={:.6f}, standardASR={}'.format(
        args.sr_definition, baseline['sr'], baseline['clean_accuracy'], baseline['standard_asr']), flush=True)
    cfg = DATASET_CONFIG[model_name]
    threshold_acc = cfg['threshold_acc'] if args.threshold_acc is None else args.threshold_acc
    threshold_sr = cfg['threshold_sr'] if args.threshold_sr is None else args.threshold_sr
    lower = np.maximum(-5.0, x0-args.search_radius)
    upper = np.minimum(5.0, x0+args.search_radius)
    if np.any(lower >= upper):
        raise ValueError('Degenerate search range: check --search-radius')
    t = time.perf_counter()
    try:
        if args.optimizer == 'moead':
            archive_x, archive_f, history = run_moead(
                evaluator, x0, lower, upper, args.pop_size, max_evals, args, args.seed)
        elif args.optimizer == 'nsga3':
            archive_x, archive_f, history = run_nsga3(
                evaluator, x0, lower, upper, args.pop_size, max_evals, args, args.seed)
        else:
            archive_x, archive_f, history = run_mopso(
                evaluator, x0, lower, upper, args.pop_size, max_evals, args, args.seed)
    finally:
        evaluator.reset()
    optimizer_wall_sec = time.perf_counter()-t
    if evaluator.evaluations != max_evals:
        raise RuntimeError('Actual evaluations {} != expected {} (check pymoo generation size)'.format(
            evaluator.evaluations, max_evals))
    best_idx, select_policy = pick_train_solution(archive_f, threshold_acc, threshold_sr)
    feasible = (archive_f[:, 0] <= threshold_sr) & (-archive_f[:, 1] >= threshold_acc)
    # Do not evaluate the test set or choose candidates based on test metrics.
    t = time.perf_counter()
    front_dir = run_dir / 'pareto_front'
    front_dir.mkdir(exist_ok=True)
    evaluator.save_vector_weights(archive_x[best_idx], run_dir / 'best_weights.h5')
    evaluator.reset()
    # Saving all full ResNet checkpoints can be enormous; always save Pareto vectors
    # in pareto_front.pkl, and only a bounded number of .h5 checkpoints.
    save_ids = list(range(min(len(archive_x), args.save_front_limit)))
    if best_idx not in save_ids and args.save_front_limit > 0:
        save_ids[-1] = best_idx
    records = []
    for idx in sorted(set(save_ids)):
        filename = front_dir / 'weights_{}.h5'.format(idx)
        evaluator.save_vector_weights(archive_x[idx], filename)
        records.append({'index': idx, 'train_backdoor': float(archive_f[idx, 0]),
                        'train_accuracy': float(-archive_f[idx, 1]), 'path': str(filename)})
    evaluator.reset()
    with open(str(run_dir / 'pareto_front.txt'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['index', 'backdoor_rate', 'accuracy'])
        for rec in records:
            w.writerow([rec['index'], '{:.12g}'.format(rec['train_backdoor']),
                        '{:.12g}'.format(rec['train_accuracy'])])
    with open(str(run_dir / 'pareto_front.pkl'), 'wb') as f:
        pickle.dump({'X': archive_x, 'F': archive_f, 'history': history,
                     'selected_index': best_idx, 'source_locator': loc['source_json'],
                     'total_evaluations': evaluator.evaluations}, f, protocol=4)
    artifact_save_sec = time.perf_counter()-t
    timing = {
        'localization_previous_run_sec': loc['source_runtime_sec'],
        'localization_in_this_run': False,
        'localization_read_export_sec': float(localization_read_sec),
        'model_loading_mapping_sec': float(model_setup_sec),
        'training_data_and_trigger_sec': float(training_data_sec),
        'baseline_evaluation_sec': float(baseline_sec),
        'optimization_wall_sec': float(optimizer_wall_sec),
        'fitness_evaluation_within_optimization_sec': float(evaluator.eval_wall_sec),
        'checkpoint_and_front_save_sec': float(artifact_save_sec),
        'total_repair_run_wall_sec': float(time.perf_counter()-wall_start),
        'notes': 'Previous localization time is NOT added to this repair run; '
                 'fitness evaluation is a SUBSET of optimization time, not extra.',
    }
    selected_spec = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}
                     for s in specs]
    summary = {
        'schema': 'CGARepair_backdoor_weight_repair_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'backdoor', 'model': model_name, 'model_tag': tag,
        'optimizer': args.optimizer, 'method': 'CGARepair_{}'.format(args.optimizer.upper()),
        'optimization_seed': int(args.seed), 'localization_seed': int(args.loc_seed),
        'localization': loc, 'exported_locator_dir': exported,
        'model_path': model_path, 'selected_players': players, 'top_k': int(loc['top_k']),
        'num_players_total': int(loc['num_players']),
        'changed_weight_slots': int(len(x0)),
        'parameter_budget_percent': float(100.0*len(x0)/max(1, model.count_params())),
        'parameter_scope': 'Selected Conv/Dense kernel output channels; biases and BN frozen',
        'repair_specs': selected_spec,
        'sr_definition': args.sr_definition,
        'sr_definition_note': ('legacy: successful non-target attacks / all examples'
                               if args.sr_definition == 'legacy' else
                               'standard: successful non-target attacks / non-target examples'),
        'train_eval_samples': int(len(ex_x)),
        'original_train_metrics': baseline,
        'threshold_acc': float(threshold_acc), 'threshold_sr': float(threshold_sr),
        'max_evaluations_requested': int(max_evals),
        'evaluations_actual': int(evaluator.evaluations),
        'front_size': int(len(archive_x)),
        'front_feasible_count': int(np.sum(feasible)),
        'selected_index': int(best_idx),
        'selection_policy': select_policy,
        'selected_train_sr': float(archive_f[best_idx, 0]),
        'selected_train_acc': float(-archive_f[best_idx, 1]),
        'best_weights_file': str((run_dir / 'best_weights.h5').resolve()),
        'saved_front_indices': sorted(set(save_ids)),
        'optimizer_parameters': {
            'population_size': int(args.pop_size), 'max_generations': int(args.max_gen),
            'search_radius': float(args.search_radius), 'init_jitter': float(args.init_jitter),
            'neighbor_count': int(args.neighbors), 'neighbor_mating_prob': float(args.neighbor_prob),
            'mutation_eta': float(args.mutation_eta),
            'mutation_prob': float(args.mutation_prob) if args.mutation_prob is not None else 0.9,
            'crossover_prob': float(args.crossover_prob),
            'inertia': float(args.inertia), 'c1': float(args.c1), 'c2': float(args.c2),
            'vmax_fraction': float(args.vmax_fraction), 'archive_size': int(args.archive_size),
        },
        'history': history,
        'timing_sec': timing,
        'optimization_time_sec': timing['optimization_wall_sec'],
        'total_time_sec': timing['total_repair_run_wall_sec'],
    }
    save_json(summary_path, summary)
    print('[Done] {} {}: selected train SR={:.6f}, ACC={:.6f} / baseline SR={:.6f}, ACC={:.6f}'.format(
        tag, args.optimizer, summary['selected_train_sr'], summary['selected_train_acc'],
        baseline['sr'], baseline['clean_accuracy']))
    print('[Timing] local(previous)={}s, optimization={:.2f}s, save={:.2f}s, total={:.2f}s'.format(
        timing['localization_previous_run_sec'], timing['optimization_wall_sec'],
        timing['checkpoint_and_front_save_sec'], timing['total_repair_run_wall_sec']))
    print('[Files] summary={}, best_weights={}, front={}'.format(summary_path,
          run_dir / 'best_weights.h5', run_dir / 'pareto_front.pkl'))
    return summary


def main():
    args = parser().parse_args()
    run(args)


if __name__ == '__main__':
    main()
