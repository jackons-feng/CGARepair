"""CGARepair safety repair: MOEA/D or MOPSO, one command-line script.

Objectives (exactly the original safety task metrics):
  1. min counterexample safety violation (argmin unsafe classes)
  2. max drawndown retention (argmin safe classes)

The original safety_moo_2.py mistakenly instantiated NSGA2 while describing MOEA/D.
This script actually uses pymoo MOEAD (or self-contained MOPSO).
Only selected Conv/Dense kernel output channel weights are modified.
Python >=3.7, TF graph-mode and pymoo 0.6.x for MOEA/D.
"""
import argparse
import csv
from datetime import datetime, timezone
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

from read_CGARepair import load_ranking, export_ranking, model_name_checked
from cgarepair_safety_utils import (CONFIG, initialize_tf, load_safety_data,
                                    model_file, calculate_metrics, save_json)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def run_directory(root, model, seed, optimizer, run_tag='', quick=False):
    suffix = ('_' + run_tag) if run_tag else ('_quick' if quick else '')
    return (Path(root).expanduser() / model / 'CGARepair' /
            ('seed_' + str(seed)) / (optimizer + suffix))


def map_players_to_weights(model, selected_players):
    """Use real model.layers indices and count *all* intervening weight arrays.

    Older ResNet repairs could unintentionally omit BatchNorm offsets. This
    mapping deliberately includes them while keeping biases/BN frozen.
    """
    flattened = model.get_weights()
    mapping = {}
    offset = 0
    for layer_idx, layer in enumerate(model.layers):
        local = layer.get_weights()
        if local and ('Conv' in type(layer).__name__ or 'Dense' in type(layer).__name__):
            mapping[layer_idx] = (offset, local[0], type(layer).__name__)
        offset += len(local)
    if offset != len(flattened):
        raise RuntimeError('Model weight offsets do not match model.get_weights; '
                           'nested Keras models require explicit support')
    result = []
    seen = set()
    start = 0
    for name in selected_players:
        m = re.fullmatch(r'(Conv\w*|Dense)_(\d+)_(\d+)', str(name), flags=re.IGNORECASE)
        if m is None:
            raise ValueError('Bad CGARepair player identifier: {}'.format(name))
        expected_type, layer_idx, neuron = m.group(1), int(m.group(2)), int(m.group(3))
        if layer_idx not in mapping:
            raise KeyError('Selected layer is missing or not Conv/Dense: {}'.format(name))
        weight_idx, kernel, true_type = mapping[layer_idx]
        if expected_type.lower() != true_type.lower():
            raise ValueError('CGARepair/model layer type mismatch: {} actual={}'.format(name, true_type))
        if kernel.ndim not in (2, 3, 4, 5):
            raise ValueError('Unsupported parameter tensor {} {}'.format(name, kernel.shape))
        if not 0 <= neuron < kernel.shape[-1]:
            raise ValueError('Neuron index outside kernel shape: {} {}'.format(name, kernel.shape))
        key = (weight_idx, neuron)
        if key in seen:
            raise ValueError('Duplicate parameter slice: {}'.format(name))
        seen.add(key)
        shape = tuple(kernel[..., neuron].shape)
        length = int(np.prod(shape))
        result.append({'player': name, 'layer_index': layer_idx,
                       'weight_index': weight_idx, 'neuron_index': neuron,
                       'shape': shape, 'start': start, 'stop': start+length,
                       'num_parameters': length})
        start += length
    if not result:
        raise ValueError('Empty repair mask')
    return result


def original_vector(weights, specs):
    return np.concatenate([weights[s['weight_index']][..., s['neuron_index']].reshape(-1)
                           for s in specs]).astype(np.float64)


def apply_vector(weights, specs, vector):
    vector = np.asarray(vector, dtype=np.float64)
    for s in specs:
        weights[s['weight_index']][..., s['neuron_index']] = (
            vector[s['start']:s['stop']].reshape(s['shape']))
    return weights


def dominates(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b + 1.e-12) and np.any(a < b - 1.e-12))


def non_dominated_indices(F):
    F = np.atleast_2d(np.asarray(F, dtype=float))
    result = []
    for i in range(len(F)):
        if any(dominates(F[j], F[i]) for j in range(len(F)) if j != i):
            continue
        if any(np.allclose(F[i], F[j], atol=1.e-12, rtol=0) for j in result):
            continue
        result.append(i)
    return np.asarray(result, dtype=int)


def crowding_distance(F):
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n = len(F)
    if n <= 2:
        return np.full(n, np.inf)
    crowd = np.zeros(n, dtype=float)
    for j in range(F.shape[1]):
        order = np.argsort(F[:, j], kind='mergesort')
        crowd[order[0]] = np.inf
        crowd[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span > 1.e-14:
            for k in range(1, n-1):
                crowd[order[k]] += (F[order[k+1], j] - F[order[k-1], j])/span
    return crowd


def reduce_archive(X, F, max_size):
    X, F = np.asarray(X, dtype=np.float64), np.asarray(F, dtype=np.float64)
    keep = non_dominated_indices(F)
    X, F = X[keep], F[keep]
    while len(X) > max_size:
        crowd = crowding_distance(F)
        finite = np.where(np.isfinite(crowd))[0]
        idx = finite[np.argmin(crowd[finite])] if len(finite) else len(X)-1
        X, F = np.delete(X, idx, axis=0), np.delete(F, idx, axis=0)
    return X, F


def choose_leader(archive_x, archive_f, rng):
    if len(archive_x) == 1:
        return archive_x[0]
    a, b = rng.choice(len(archive_x), size=2, replace=True)
    crowd = crowding_distance(archive_f)
    return archive_x[a if crowd[a] >= crowd[b] else b]


class SafetyRepairEvaluator:
    """Train-only fitness evaluations, historical nondominated archive."""
    def __init__(self, model, session, specs, counterexamples, drawdown,
                 model_name, batch_size=512, archive_size=60):
        self.model = model
        self.session = session
        self.specs = specs
        self.ct_x = counterexamples
        self.dr_x = drawdown
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.base_weights = [w.copy() for w in model.get_weights()]
        self.work_weights = [w.copy() for w in self.base_weights]
        self.evaluations = 0
        self.eval_wall_sec = 0.0
        self.archive_size = int(archive_size)
        self.archive_x = None
        self.archive_f = None

    def baseline_metrics(self):
        self.model.set_weights(self.base_weights)
        return calculate_metrics(self.model, self.model_name, self.ct_x, self.dr_x,
                                 self.session, self.batch_size)

    def _archive_one(self, x, f):
        if self.archive_f is None:
            self.archive_x = np.asarray(x, dtype=np.float64).reshape(1, -1).copy()
            self.archive_f = np.asarray(f, dtype=np.float64).reshape(1, 2).copy()
            return
        if np.any(np.all(np.abs(self.archive_f-f) <= 1.e-12, axis=1)):
            return
        if any(dominates(old, f) for old in self.archive_f):
            return
        keep = [i for i in range(len(self.archive_f)) if not dominates(f, self.archive_f[i])]
        self.archive_x = np.concatenate([self.archive_x[keep], np.asarray(x).reshape(1, -1)])
        self.archive_f = np.concatenate([self.archive_f[keep], np.asarray(f).reshape(1, 2)])
        if len(self.archive_x) > self.archive_size:
            self.archive_x, self.archive_f = reduce_archive(
                self.archive_x, self.archive_f, self.archive_size)

    def evaluate_batch(self, candidates):
        candidates = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        output = np.empty((len(candidates), 2), dtype=np.float64)
        for i, vector in enumerate(candidates):
            start = time.perf_counter()
            apply_vector(self.work_weights, self.specs, vector)
            self.model.set_weights(self.work_weights)
            metrics = calculate_metrics(self.model, self.model_name, self.ct_x, self.dr_x,
                                        self.session, self.batch_size)
            f = np.array([metrics['safety_violation_rate'],
                          -metrics['drawdown_retention']], dtype=np.float64)
            output[i] = f
            self.evaluations += 1
            self._archive_one(vector, f)
            self.eval_wall_sec += time.perf_counter() - start
        return output

    def save_vector_weights(self, vector, filepath):
        apply_vector(self.work_weights, self.specs, vector)
        self.model.set_weights(self.work_weights)
        self.model.save_weights(str(filepath))

    def reset(self):
        self.model.set_weights(self.base_weights)


def run_mopso(ev, x0, lower, upper, pop_size, max_evals, args, seed):
    rng = np.random.RandomState(int(seed))
    span = upper-lower
    positions = np.clip(x0+rng.uniform(-args.init_jitter, args.init_jitter,
                                        (pop_size, len(x0))), lower, upper)
    positions[0] = np.clip(x0, lower, upper)
    velocity = rng.uniform(-.005, .005, positions.shape)*span
    velocity_bound = args.vmax_fraction*span
    scores = ev.evaluate_batch(positions)
    pbest, pbest_f = positions.copy(), scores.copy()
    archive_x, archive_f = ev.archive_x.copy(), ev.archive_f.copy()
    history = []
    gen = 1
    while ev.evaluations < max_evals:
        n = min(pop_size, max_evals-ev.evaluations)
        r1 = rng.uniform(size=(n, len(x0)))
        r2 = rng.uniform(size=(n, len(x0)))
        leader = np.stack([choose_leader(archive_x, archive_f, rng) for _ in range(n)])
        velocity[:n] = (args.inertia*velocity[:n]
                        + args.c1*r1*(pbest[:n]-positions[:n])
                        + args.c2*r2*(leader-positions[:n]))
        velocity[:n] = np.clip(velocity[:n], -velocity_bound, velocity_bound)
        proposal = np.clip(positions[:n]+velocity[:n], lower, upper)
        touching = (proposal <= lower) | (proposal >= upper)
        velocity[:n][touching] *= -.5
        f_next = ev.evaluate_batch(proposal)
        positions[:n] = proposal
        for i in range(n):
            if dominates(f_next[i], pbest_f[i]) or (
                not dominates(pbest_f[i], f_next[i]) and rng.rand() < 0.5):
                pbest[i], pbest_f[i] = positions[i].copy(), f_next[i].copy()
        archive_x, archive_f = ev.archive_x.copy(), ev.archive_f.copy()
        gen += 1
        if gen % args.log_every == 0 or ev.evaluations >= max_evals:
            hist = {'generation': gen, 'evaluations': int(ev.evaluations),
                    'minimum_safety_violation_rate': float(np.min(archive_f[:, 0])),
                    'maximum_drawdown_retention': float(np.max(-archive_f[:, 1])),
                    'archive_size': int(len(archive_x))}
            history.append(hist)
            print('[MOPSO] gen={} FEs={} archive={} minSafety={:.6f} maxRetention={:.6f}'.format(
                gen, ev.evaluations, len(archive_x), hist['minimum_safety_violation_rate'],
                hist['maximum_drawdown_retention']), flush=True)
    return archive_x, archive_f, history


def run_moead(ev, x0, lower, upper, pop_size, max_evals, args, seed):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
    except ImportError as exc:
        raise ImportError('MOEA/D requires the same compatible pymoo 0.6 environment as '
                          'the original repair script: {}'.format(exc))

    class SafetyProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(x0), n_obj=2, xl=lower, xu=upper, vtype=float)

        def _evaluate(self, X, out, *unused, **kwunused):
            out['F'] = ev.evaluate_batch(X)

    ref_dirs = get_reference_directions('das-dennis', 2, n_partitions=pop_size-1)
    if len(ref_dirs) != pop_size:
        raise RuntimeError('MOEA/D reference directions do not equal pop size')
    rng = np.random.RandomState(int(seed))
    initial = np.clip(x0+rng.uniform(-args.init_jitter, args.init_jitter,
                                      (pop_size, len(x0))), lower, upper)
    initial[0] = np.clip(x0, lower, upper)
    # 0.9 was used by the successful legacy MOEA/D repair, unlike 1/num_variables.
    mutation_prob = 0.9 if args.mutation_prob is None else float(args.mutation_prob)
    kwargs = {'prob': mutation_prob, 'eta': args.mutation_eta}
    if args.mutation_prob_var is not None:
        import inspect
        if 'prob_var' not in inspect.signature(PolynomialMutation).parameters:
            raise RuntimeError('Installed pymoo PolynomialMutation lacks prob_var. '
                               'Omit --mutation-prob-var or upgrade pymoo.')
        kwargs['prob_var'] = args.mutation_prob_var
    algo = MOEAD(ref_dirs=ref_dirs,
                 n_neighbors=min(args.neighbors, pop_size-1),
                 prob_neighbor_mating=args.neighbor_prob,
                 mutation=PolynomialMutation(**kwargs),
                 crossover=UniformCrossover(prob=args.crossover_prob),
                 sampling=initial)
    result = minimize(SafetyProblem(), algo, get_termination('n_eval', int(max_evals)),
                      seed=int(seed), verbose=False, save_history=False)
    if result.pop is None or ev.archive_x is None:
        raise RuntimeError('MOEA/D yielded no evaluated solutions')
    # Include all historical evaluated solutions, not only MOEA/D's last population.
    X, F = ev.archive_x.copy(), ev.archive_f.copy()
    hist = [{'generations': int(getattr(result.algorithm, 'n_gen', 0)),
             'evaluations': int(ev.evaluations),
             'minimum_safety_violation_rate': float(np.min(F[:, 0])),
             'maximum_drawdown_retention': float(np.max(-F[:, 1])),
             'archive_size': int(len(X))}]
    print('[MOEAD] generations={} FEs={} historical_front={}'.format(
        hist[0]['generations'], ev.evaluations, len(X)), flush=True)
    return X, F, hist



# NSGA-III ablation: same TRAIN problem, initial population, mutation, crossover,
# selection policy and input bounds as native MOEA/D.
def run_nsga3(ev, x0, lower, upper, pop_size, max_evals, args, seed):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
    except ImportError as exc:
        raise ImportError('NSGA-III requires the same compatible pymoo 0.6 environment as '
                          'the original repair script: {}'.format(exc))

    class SafetyProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(x0), n_obj=2, xl=lower, xu=upper, vtype=float)

        def _evaluate(self, X, out, *unused, **kwunused):
            out['F'] = ev.evaluate_batch(X)

    ref_dirs = get_reference_directions('das-dennis', 2, n_partitions=pop_size-1)
    if len(ref_dirs) != pop_size:
        raise RuntimeError('MOEA/D reference directions do not equal pop size')
    rng = np.random.RandomState(int(seed))
    initial = np.clip(x0+rng.uniform(-args.init_jitter, args.init_jitter,
                                      (pop_size, len(x0))), lower, upper)
    initial[0] = np.clip(x0, lower, upper)
    # 0.9 was used by the successful legacy MOEA/D repair, unlike 1/num_variables.
    mutation_prob = 0.9 if args.mutation_prob is None else float(args.mutation_prob)
    kwargs = {'prob': mutation_prob, 'eta': args.mutation_eta}
    if args.mutation_prob_var is not None:
        import inspect
        if 'prob_var' not in inspect.signature(PolynomialMutation).parameters:
            raise RuntimeError('Installed pymoo PolynomialMutation lacks prob_var. '
                               'Omit --mutation-prob-var or upgrade pymoo.')
        kwargs['prob_var'] = args.mutation_prob_var
    algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs,
                 mutation=PolynomialMutation(**kwargs),
                 crossover=UniformCrossover(prob=args.crossover_prob),
                 sampling=initial, eliminate_duplicates=False)
    result = minimize(SafetyProblem(), algo, get_termination('n_eval', int(max_evals)),
                      seed=int(seed), verbose=False, save_history=False)
    if result.pop is None or ev.archive_x is None:
        raise RuntimeError('NSGA-III yielded no evaluated solutions')
    # Include all historical evaluated solutions, not only MOEA/D's last population.
    X, F = ev.archive_x.copy(), ev.archive_f.copy()
    hist = [{'generations': int(getattr(result.algorithm, 'n_gen', 0)),
             'evaluations': int(ev.evaluations),
             'minimum_safety_violation_rate': float(np.min(F[:, 0])),
             'maximum_drawdown_retention': float(np.max(-F[:, 1])),
             'archive_size': int(len(X))}]
    print('[NSGA3] generations={} FEs={} historical_front={}'.format(
        hist[0]['generations'], ev.evaluations, len(X)), flush=True)
    return X, F, hist



def choose_train_solution(F, min_retention, max_violation):
    F = np.asarray(F, dtype=float)
    error, acc = F[:, 0], -F[:, 1]
    good = (error <= max_violation) & (acc >= min_retention)
    utility = acc >= min_retention
    if np.any(good):
        ids = np.where(good)[0]
        policy = 'meets_both_min_error_then_max_acc'
        order = np.lexsort((-acc[ids], error[ids]))
    elif np.any(utility):
        ids = np.where(utility)[0]
        policy = 'utility_guard_min_error_then_max_acc'
        order = np.lexsort((-acc[ids], error[ids]))
    else:
        ids = np.arange(len(F))
        policy = 'no_utility_feasible_max_acc_then_min_error'
        order = np.lexsort((error[ids], -acc[ids]))
    return int(ids[order[0]]), policy


def get_parser():
    p = argparse.ArgumentParser(description='CGARepair safety: MOEA/D / MOPSO local weight repair')
    p.add_argument('model_name', choices=('a19', 'a29', 'a33'))
    p.add_argument('seed', type=int, help='Optimizer seed independent from localization seed')
    p.add_argument('--optimizer', '--algorithm', choices=('moead', 'nsga3', 'mopso'), default='moead')
    p.add_argument('--loc-json', help='Exact CGARepair safety localization JSON file')
    p.add_argument('--loc-root', help='Localization root (if auto search is insufficient)')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--model-root', help='Folder containing a19.h5, a29.h5, a33.h5')
    p.add_argument('--data-root', help='Directory containing a19/a29/a33 safety HDF5 subfolders')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-tag', default='', help='Output suffix; existing runs are preserved')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--quick', action='store_true', help='12 population, 36 FEs, 128 train rows/set')
    p.add_argument('--pop-size', type=int, default=40)
    p.add_argument('--max-gen', type=int, default=40)
    p.add_argument('--max-evals', type=int, default=None)
    p.add_argument('--eval-samples', type=int, default=1024,
                   help='Rows selected independently from counterexample/drawdown; 0=all')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--search-radius', type=float, default=.5,
                   help='Search limits = intersection of [-5,5] and base weight +/- radius')
    p.add_argument('--init-jitter', type=float, default=.01)
    p.add_argument('--threshold-acc', type=float, default=None,
                   help='Minimum drawndown retention, original safety setting 0.999')
    p.add_argument('--threshold-safety', type=float, default=None,
                   help='Maximum counterexample violation rate, default 0')
    p.add_argument('--archive-size', type=int, default=60)
    p.add_argument('--save-front-limit', type=int, default=10)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--neighbors', type=int, default=15)
    p.add_argument('--neighbor-prob', type=float, default=.7)
    p.add_argument('--mutation-prob', type=float, default=.9,
                   help='Same individual-level probability as original 0.9')
    p.add_argument('--mutation-prob-var', type=float, default=None,
                   help='Optional pymoo variable-level mutation probability')
    p.add_argument('--mutation-eta', type=float, default=20.)
    p.add_argument('--crossover-prob', type=float, default=1.)
    p.add_argument('--inertia', type=float, default=.6)
    p.add_argument('--c1', type=float, default=1.5)
    p.add_argument('--c2', type=float, default=1.5)
    p.add_argument('--vmax-fraction', type=float, default=.1)
    return p


def run(args):
    wall_start = time.perf_counter()
    model = model_name_checked(args.model_name)
    if args.quick:
        args.pop_size, args.max_gen, args.eval_samples = 12, 3, 128
        if args.max_evals is None:
            args.max_evals = 36
    if args.pop_size < 3 or args.max_gen < 1 or args.eval_samples < 0 or args.batch_size < 1:
        raise ValueError('pop-size>=3, max-gen>=1, eval-samples>=0, batch-size>=1 required')
    max_evals = int(args.max_evals if args.max_evals is not None
                    else args.pop_size * args.max_gen)
    if max_evals < args.pop_size:
        raise ValueError('max-evals must be >= pop-size')
    if args.search_radius <= 0 or args.init_jitter < 0:
        raise ValueError('search-radius>0 and init-jitter>=0 are required')
    if args.archive_size < 2 or args.save_front_limit < 0 or args.log_every <= 0:
        raise ValueError('archive-size>=2, save-front-limit>=0, log-every>0 required')
    if not 0. <= args.mutation_prob <= 1.:
        raise ValueError('mutation-prob must be in [0,1]')
    if args.mutation_prob_var is not None and not 0. <= args.mutation_prob_var <= 1.:
        raise ValueError('mutation-prob-var must be in [0,1]')
    if not 0. <= args.neighbor_prob <= 1. or not 0. <= args.crossover_prob <= 1.:
        raise ValueError('neighbor-prob and crossover-prob must be in [0,1]')
    outdir = run_directory(args.output_root, model, args.seed,
                           args.optimizer, args.run_tag, args.quick)
    if (outdir/'summary.json').exists() and not args.overwrite:
        raise FileExistsError('Existing {}. Use a new --run-tag or --overwrite.'.format(
            outdir/'summary.json'))
    outdir.mkdir(parents=True, exist_ok=True)

    begin = time.perf_counter()
    players, scores, loc = load_ranking(model, args.loc_seed, args.loc_samples,
                                         args.loc_json, args.loc_root)
    export_dir = export_ranking(players, scores, loc, outdir/'localization_input')
    locator_read_sec = time.perf_counter() - begin
    print('[Locator] {} Top-{} from {} (loc_seed={}, opt_seed={})'.format(
        model, loc['top_k'], loc['source_json'], args.loc_seed, args.seed), flush=True)
    print('[Locator] selected={}'.format(players), flush=True)

    begin = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    from tensorflow.keras.models import load_model
    path = model_file(model, args.model_root)
    network = load_model(path, compile=False)
    signature = loc.get('source_model_signature')
    if isinstance(signature, (list, tuple)) and signature:
        if os.stat(path).st_size != int(signature[0]):
            raise RuntimeError('CGARepair locator checkpoint size differs from repair model: {} != {}'
                               .format(signature[0], os.stat(path).st_size))
    specs = map_players_to_weights(network, players)
    x0 = original_vector(network.get_weights(), specs)
    model_sec = time.perf_counter() - begin
    print('[Repair] {} selected_neurons={} decision_variables={} model={}'.format(
        model, len(specs), len(x0), path), flush=True)

    begin = time.perf_counter()
    ct_x, dr_x, provenance = load_safety_data(model, split='train',
                                              data_root=args.data_root,
                                              samples=args.eval_samples, seed=args.seed)
    train_data_sec = time.perf_counter() - begin
    evaluator = SafetyRepairEvaluator(network, session, specs, ct_x, dr_x, model,
                                      batch_size=args.batch_size, archive_size=args.archive_size)
    begin = time.perf_counter()
    baseline = evaluator.baseline_metrics()
    baseline_sec = time.perf_counter() - begin
    print('[Baseline] safety_violation={:.6f} drawdown_retention={:.6f} (ct={}, dr={})'.format(
        baseline['safety_violation_rate'], baseline['drawdown_retention'],
        len(ct_x), len(dr_x)), flush=True)

    min_retention = (CONFIG[model]['threshold_acc'] if args.threshold_acc is None
                     else float(args.threshold_acc))
    max_violation = (CONFIG[model]['threshold_safety'] if args.threshold_safety is None
                     else float(args.threshold_safety))
    if not (0 <= min_retention <= 1 and 0 <= max_violation <= 1):
        raise ValueError('Thresholds must be in [0,1]')
    low = np.maximum(-5., x0 - args.search_radius)
    high = np.minimum(5., x0 + args.search_radius)
    if np.any(low >= high):
        raise RuntimeError('Invalid bounds, pretrained weight may be outside [-5,5]. '
                           'Check model/locator and search configuration.')
    if np.any((x0 < low) | (x0 > high)):
        raise RuntimeError('Pretrained weights outside configured search box')

    begin = time.perf_counter()
    try:
        if args.optimizer == 'moead':
            X, F, history = run_moead(evaluator, x0, low, high, args.pop_size,
                                      max_evals, args, args.seed)
        elif args.optimizer == 'nsga3':
            X, F, history = run_nsga3(evaluator, x0, low, high, args.pop_size,
                                      max_evals, args, args.seed)
        else:
            X, F, history = run_mopso(evaluator, x0, low, high, args.pop_size,
                                      max_evals, args, args.seed)
    finally:
        evaluator.reset()
    optimization_sec = time.perf_counter() - begin
    if evaluator.evaluations != max_evals:
        raise RuntimeError('Actual evaluations {} != expected {} (check pymoo generation size)'.format(
            evaluator.evaluations, max_evals))
    chosen, policy = choose_train_solution(F, min_retention, max_violation)
    feasible = (F[:, 0] <= max_violation) & (-F[:, 1] >= min_retention)

    begin = time.perf_counter()
    front_dir = outdir/'pareto_front'
    front_dir.mkdir(parents=True, exist_ok=True)
    best_file = outdir/'best_weights.h5'
    evaluator.save_vector_weights(X[chosen], best_file)
    evaluator.reset()
    export_ids = list(range(min(len(X), args.save_front_limit)))
    if chosen not in export_ids and export_ids:
        export_ids[-1] = chosen
    records = []
    for idx in sorted(set(export_ids)):
        fpath = front_dir/'weights_{}.h5'.format(idx)
        evaluator.save_vector_weights(X[idx], fpath)
        records.append({'index': idx, 'train_safety_violation': float(F[idx, 0]),
                        'train_drawdown_retention': float(-F[idx, 1]), 'file': str(fpath)})
    evaluator.reset()
    with open(str(outdir/'pareto_front.txt'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'train_safety_violation', 'train_drawdown_retention'])
        for record in records:
            writer.writerow([record['index'], record['train_safety_violation'],
                             record['train_drawdown_retention']])
    with open(str(outdir/'pareto_front.pkl'), 'wb') as file:
        pickle.dump({'X': X, 'F': F, 'history': history,
                     'selected_index': int(chosen),
                     'total_evaluations': int(evaluator.evaluations),
                     'source_locator': loc['source_json']}, file, protocol=4)
    save_sec = time.perf_counter() - begin
    timing = {
        'localization_previous_run_sec': loc['source_runtime_sec'],
        'localization_previous_run_not_included_in_repair_wall': True,
        'localization_read_export_sec': float(locator_read_sec),
        'model_load_and_weight_mapping_sec': float(model_sec),
        'train_data_load_sec': float(train_data_sec),
        'train_baseline_eval_sec': float(baseline_sec),
        'optimization_wall_sec': float(optimization_sec),
        'fitness_eval_within_optimization_sec': float(evaluator.eval_wall_sec),
        'pareto_and_checkpoint_save_sec': float(save_sec),
        'total_repair_run_wall_sec': float(time.perf_counter() - wall_start),
        'notes': 'Previous localization runtime is historical; '
                 'fitness evaluations are INCLUDED in optimization wall.',
    }
    summary = {
        'schema': 'CGARepair_safety_MOEAD_MOPSO_weight_repair_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'safety', 'model': model, 'optimizer': args.optimizer,
        'method': 'CGARepair_{}'.format(args.optimizer.upper()),
        'optimization_seed': int(args.seed), 'localization_seed': int(args.loc_seed),
        'localization': loc, 'localization_exported_dir': export_dir,
        'model_file': path, 'selected_players': players, 'top_k': int(loc['top_k']),
        'total_ranked_neurons': int(loc['num_players']),
        'repair_specs': [{k: list(v) if isinstance(v, tuple) else v for k,v in item.items()}
                         for item in specs],
        'decision_variables': int(len(x0)),
        'parameter_budget_percent': float(100.*len(x0)/max(network.count_params(), 1)),
        'parameter_scope': 'Selected Conv/Dense kernel output channels; BN and biases frozen',
        'train_data': provenance, 'original_train_metrics': baseline,
        'objective_definitions': ['safety_violation_rate=minimize', '-drawdown_retention=minimize'],
        'threshold_drawdown_retention': float(min_retention),
        'threshold_safety_violation': float(max_violation),
        'max_evaluations_requested': int(max_evals),
        'evaluations_actual': int(evaluator.evaluations),
        'front_size': int(len(X)), 'front_feasible_count': int(np.sum(feasible)),
        'selected_index': int(chosen), 'selection_policy': policy,
        'selected_train_safety_violation': float(F[chosen, 0]),
        'selected_train_drawdown_retention': float(-F[chosen, 1]),
        'best_weights_file': str(best_file.resolve()),
        'optimizer_parameters': {
            'population_size': int(args.pop_size), 'max_gen': int(args.max_gen),
            'max_evaluations': int(max_evals), 'init_jitter': float(args.init_jitter),
            'search_radius': float(args.search_radius),
            'neighbor_count': int(args.neighbors), 'neighbor_prob': float(args.neighbor_prob),
            'mutation_prob': float(args.mutation_prob),
            'mutation_prob_var': args.mutation_prob_var,
            'mutation_eta': float(args.mutation_eta),
            'crossover_prob': float(args.crossover_prob),
            'inertia': float(args.inertia), 'c1': float(args.c1), 'c2': float(args.c2),
            'vmax_fraction': float(args.vmax_fraction), 'archive_size': int(args.archive_size),
        },
        'optimizer_history': history,
        'timing_sec': timing,
        'total_time_sec': float(timing['total_repair_run_wall_sec']),
    }
    save_json(outdir/'summary.json', summary)
    print('[Done] {} {} train safety={:.6f}, retention={:.6f} '
          '/ baseline safety={:.6f}, retention={:.6f}'.format(
             model, args.optimizer, summary['selected_train_safety_violation'],
             summary['selected_train_drawdown_retention'],
             baseline['safety_violation_rate'], baseline['drawdown_retention']), flush=True)
    print('[Timing] localization(previous)={}s, optimization={:.2f}s, '
          'repair_total={:.2f}s'.format(loc['source_runtime_sec'],
            optimization_sec, timing['total_repair_run_wall_sec']), flush=True)
    print('[Files] summary={} best={} front={}'.format(outdir/'summary.json',
          best_file, outdir/'pareto_front.pkl'), flush=True)
    return summary


def main():
    run(get_parser().parse_args())


if __name__ == '__main__':
    main()
