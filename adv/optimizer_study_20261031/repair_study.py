"""CGARepair adversarial local-weight repair: MOEA/D and MOPSO in one CLI.

Only the selected Conv/Dense kernel output channels may change. Targets are
min(ADV classification error) and max(clean classification accuracy).

Python >=3.7, TensorFlow graph mode, numpy and (for MOEA/D) pymoo 0.6.x.
No imports from read.py: CFRepair's original reader remains untouched.
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

from read_CGARepair import load_ranking, export_ranking, epsilon_string, model_name_checked
from cgarepair_adv_utils import (CONFIG, initialize_tf, load_adv_and_clean,
                                 model_file, calculate_metrics, save_json)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def run_directory(root, model, epsilon, seed, optimizer, run_tag='', quick=False):
    suffix = ('_' + run_tag) if run_tag else ('_quick' if quick else '')
    return (Path(root).expanduser() / 'adv' / model / ('epsilon_' + epsilon) /
            'CGARepair' / ('seed_' + str(seed)) / (optimizer + suffix))


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


class AdvRepairEvaluator:
    """One repaired model instance and a bounded historical nondominated archive."""
    def __init__(self, model, session, specs, adv_x, adv_y, clean_x, clean_y,
                 batch_size=512, archive_size=60):
        self.model, self.session, self.specs = model, session, specs
        self.adv_x, self.adv_y = adv_x, adv_y
        self.clean_x, self.clean_y = clean_x, clean_y
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
        return calculate_metrics(self.model, self.adv_x, self.adv_y,
                                 self.clean_x, self.clean_y, self.session, self.batch_size)

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
            metrics = calculate_metrics(self.model, self.adv_x, self.adv_y,
                                        self.clean_x, self.clean_y, self.session, self.batch_size)
            f = np.array([metrics['adv_error_rate'], -metrics['clean_accuracy']], dtype=np.float64)
            output[i] = f
            self.evaluations += 1
            self._archive_one(vector, f)
            self.eval_wall_sec += time.perf_counter()-start
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
                    'minimum_adv_error': float(np.min(archive_f[:, 0])),
                    'maximum_clean_accuracy': float(np.max(-archive_f[:, 1])),
                    'archive_size': int(len(archive_x))}
            history.append(hist)
            print('[MOPSO] gen={} FEs={} archive={} minAdvErr={:.6f} maxAcc={:.6f}'.format(
                gen, ev.evaluations, len(archive_x), hist['minimum_adv_error'],
                hist['maximum_clean_accuracy']), flush=True)
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

    class AdvProblem(Problem):
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
    result = minimize(AdvProblem(), algo, get_termination('n_eval', int(max_evals)),
                      seed=int(seed), verbose=False, save_history=False)
    if result.pop is None or ev.archive_x is None:
        raise RuntimeError('MOEA/D yielded no evaluated solutions')
    # Include all historical evaluated solutions, not only MOEA/D's last population.
    X, F = ev.archive_x.copy(), ev.archive_f.copy()
    hist = [{'generations': int(getattr(result.algorithm, 'n_gen', 0)),
             'evaluations': int(ev.evaluations),
             'minimum_adv_error': float(np.min(F[:, 0])),
             'maximum_clean_accuracy': float(np.max(-F[:, 1])),
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

    class AdvProblem(Problem):
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
    result = minimize(AdvProblem(), algo, get_termination('n_eval', int(max_evals)),
                      seed=int(seed), verbose=False, save_history=False)
    if result.pop is None or ev.archive_x is None:
        raise RuntimeError('NSGA-III yielded no evaluated solutions')
    # Include all historical evaluated solutions, not only MOEA/D's last population.
    X, F = ev.archive_x.copy(), ev.archive_f.copy()
    hist = [{'generations': int(getattr(result.algorithm, 'n_gen', 0)),
             'evaluations': int(ev.evaluations),
             'minimum_adv_error': float(np.min(F[:, 0])),
             'maximum_clean_accuracy': float(np.max(-F[:, 1])),
             'archive_size': int(len(X))}]
    print('[NSGA3] generations={} FEs={} historical_front={}'.format(
        hist[0]['generations'], ev.evaluations, len(X)), flush=True)
    return X, F, hist



def choose_train_solution(F, min_acc, max_error):
    F = np.asarray(F, dtype=float)
    error, acc = F[:, 0], -F[:, 1]
    good = (error <= max_error) & (acc >= min_acc)
    utility = acc >= min_acc
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
    p = argparse.ArgumentParser(description='CGARepair adversarial repair: MOEA/D or MOPSO')
    p.add_argument('model_name', choices=('mnist', 'cifar10'))
    p.add_argument('epsilon', help='MNIST 0.05, CIFAR10 0.1')
    p.add_argument('seed', type=int, help='Repair seed, independent of locator seed')
    p.add_argument('--optimizer', '--algorithm', choices=('moead', 'nsga3', 'mopso'), default='moead')
    p.add_argument('--loc-json', help='Full path to the CGARepair ADV rank cache')
    p.add_argument('--loc-root', help='Root containing the CGARepair loc_result/<dataset> caches')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=512)
    p.add_argument('--model-root', help='Directory with mnist_adv.h5 and cifar10_adv.h5')
    p.add_argument('--data-root', help='Directory with adv/<dataset>/<split>/epsilon*.npy or txt')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-tag', default='', help='Suffix of optimizer output directory')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--quick', action='store_true', help='12 pop, 36 FEs, 128 sample smoke test')
    p.add_argument('--pop-size', type=int, default=40)
    p.add_argument('--max-gen', type=int, default=40)
    p.add_argument('--max-evals', type=int, default=None)
    p.add_argument('--eval-samples', type=int, default=1024,
                   help='Per-set sample count (separate ADV and clean subsets)')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--search-radius', type=float, default=.5,
                   help='Bounds are max(-5, x0-radius) to min(5, x0+radius)')
    p.add_argument('--init-jitter', type=float, default=.01)
    p.add_argument('--threshold-acc', type=float, default=None)
    p.add_argument('--threshold-adv-error', type=float, default=None)
    p.add_argument('--archive-size', type=int, default=60)
    p.add_argument('--save-front-limit', type=int, default=10)
    p.add_argument('--log-every', type=int, default=5)
    p.add_argument('--neighbors', type=int, default=15)
    p.add_argument('--neighbor-prob', type=float, default=.7)
    p.add_argument('--mutation-prob', type=float, default=None,
                   help='Default 0.9 reproduces legacy mutation probability')
    p.add_argument('--mutation-prob-var', type=float, default=None)
    p.add_argument('--mutation-eta', type=float, default=20.0)
    p.add_argument('--crossover-prob', type=float, default=1.0)
    p.add_argument('--inertia', type=float, default=.6)
    p.add_argument('--c1', type=float, default=1.5)
    p.add_argument('--c2', type=float, default=1.5)
    p.add_argument('--vmax-fraction', type=float, default=.1)
    return p


def run(args):
    wall_start = time.perf_counter()
    model = model_name_checked(args.model_name)
    eps = epsilon_string(args.epsilon)
    if args.quick:
        args.pop_size, args.max_gen, args.eval_samples = 12, 3, 128
        if args.max_evals is None:
            args.max_evals = 36
    if args.pop_size < 3 or args.max_gen < 1 or args.eval_samples < 1:
        raise ValueError('--pop-size>=3, --max-gen>=1, --eval-samples>=1 required')
    max_evals = int(args.max_evals if args.max_evals is not None
                    else args.pop_size*args.max_gen)
    if max_evals < args.pop_size:
        raise ValueError('max-evals must not be smaller than pop-size')
    if args.search_radius <= 0 or args.init_jitter < 0:
        raise ValueError('search-radius must be >0, init-jitter must be >=0')
    if args.archive_size < 2 or args.save_front_limit < 0:
        raise ValueError('archive-size>=2 and save-front-limit>=0 required')
    if args.mutation_prob is not None and not 0 <= args.mutation_prob <= 1:
        raise ValueError('mutation-prob must be within [0,1]')
    outdir = run_directory(args.output_root, model, eps, args.seed,
                           args.optimizer, args.run_tag, args.quick)
    if (outdir/'summary.json').exists() and not args.overwrite:
        raise FileExistsError('Existing {}. Use --run-tag or --overwrite.'.format(outdir/'summary.json'))
    outdir.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    players, scores, loc = load_ranking(model, eps, args.loc_seed, args.loc_samples,
                                         args.loc_json, args.loc_root)
    exported = export_ranking(players, scores, loc, outdir/'localization_input')
    loc_read_sec = time.perf_counter()-t
    print('[Locator] {} eps={} Top-{} from {}'.format(model, eps, loc['top_k'],
                                                       loc['source_json']), flush=True)
    print('[Locator] selected={}'.format(players), flush=True)

    t = time.perf_counter()
    tf, K, session = initialize_tf(args.seed)
    from tensorflow.keras.models import load_model
    filepath = model_file(model, args.model_root)
    network = load_model(filepath, compile=False)
    signature = loc.get('source_model_signature')
    if isinstance(signature, (tuple, list)) and signature:
        if int(signature[0]) != os.stat(filepath).st_size:
            raise ValueError('Locator checkpoint/model differ in file size: {} vs {}'
                             .format(signature[0], os.stat(filepath).st_size))
    specs = map_players_to_weights(network, players)
    base_vector = original_vector(network.get_weights(), specs)
    setup_sec = time.perf_counter()-t
    print('[Repair] {} eps={} neurons={} decision_variables={} checkpoint={}'.format(
        model, eps, len(specs), len(base_vector), filepath), flush=True)

    t = time.perf_counter()
    adv_x, adv_y, clean_x, clean_y, data_info = load_adv_and_clean(
        model, eps, split='train', data_root=args.data_root, samples=args.eval_samples, seed=args.seed)
    data_sec = time.perf_counter()-t
    evaluator = AdvRepairEvaluator(network, session, specs, adv_x, adv_y, clean_x, clean_y,
                                   batch_size=args.batch_size, archive_size=args.archive_size)
    t = time.perf_counter()
    baseline = evaluator.baseline_metrics()
    baseline_sec = time.perf_counter()-t
    print('[Baseline] ADV error={:.6f}, robustness={:.6f}, cleanAcc={:.6f} '
          '(adv_n={}, clean_n={})'.format(baseline['adv_error_rate'], baseline['adv_robustness'],
                                        baseline['clean_accuracy'], len(adv_y), len(clean_y)), flush=True)
    min_acc = (CONFIG[model]['threshold_acc'] if args.threshold_acc is None
               else args.threshold_acc)
    max_err = (CONFIG[model]['threshold_adv_error'] if args.threshold_adv_error is None
               else args.threshold_adv_error)
    low = np.maximum(-5.0, base_vector-args.search_radius)
    high = np.minimum(5.0, base_vector+args.search_radius)
    if np.any(low >= high):
        raise RuntimeError('Invalid parameter bounds, including pretrained values outside [-5,5]')

    t = time.perf_counter()
    try:
        if args.optimizer == 'moead':
            X, F, history = run_moead(evaluator, base_vector, low, high,
                                      args.pop_size, max_evals, args, args.seed)
        elif args.optimizer == 'nsga3':
            X, F, history = run_nsga3(evaluator, base_vector, low, high,
                                      args.pop_size, max_evals, args, args.seed)
        else:
            X, F, history = run_mopso(evaluator, base_vector, low, high,
                                      args.pop_size, max_evals, args, args.seed)
    finally:
        evaluator.reset()
    optimization_sec = time.perf_counter()-t
    if evaluator.evaluations != max_evals:
        raise RuntimeError('Actual evaluations {} != expected {} (check pymoo generation size)'.format(
            evaluator.evaluations, max_evals))
    chosen_idx, selection_policy = choose_train_solution(F, min_acc, max_err)
    feasible = (F[:, 0] <= max_err) & (-F[:, 1] >= min_acc)

    t = time.perf_counter()
    front_dir = outdir/'pareto_front'
    front_dir.mkdir(parents=True, exist_ok=True)
    best_file = outdir/'best_weights.h5'
    evaluator.save_vector_weights(X[chosen_idx], best_file)
    evaluator.reset()
    save_ids = list(range(min(len(X), args.save_front_limit)))
    if chosen_idx not in save_ids and save_ids:
        save_ids[-1] = chosen_idx
    records = []
    for idx in sorted(set(save_ids)):
        file = front_dir / 'weights_{}.h5'.format(idx)
        evaluator.save_vector_weights(X[idx], file)
        records.append({'index': idx, 'train_adv_error': float(F[idx, 0]),
                        'train_robustness': float(1-F[idx, 0]),
                        'train_accuracy': float(-F[idx, 1]), 'file': str(file)})
    evaluator.reset()
    with open(str(outdir/'pareto_front.txt'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'train_adv_error', 'train_robustness', 'train_accuracy'])
        for item in records:
            writer.writerow([item['index'], item['train_adv_error'],
                             item['train_robustness'], item['train_accuracy']])
    with open(str(outdir/'pareto_front.pkl'), 'wb') as f:
        pickle.dump({'X': X, 'F': F, 'history': history,
                     'selected_index': int(chosen_idx),
                     'total_evaluations': int(evaluator.evaluations),
                     'source_locator': loc['source_json']}, f, protocol=4)
    saving_sec = time.perf_counter()-t
    timing = {
        'localization_previous_run_sec': loc['source_runtime_sec'],
        'localization_previous_run_not_included_in_repair_wall': True,
        'localization_read_export_sec': float(loc_read_sec),
        'model_load_and_weight_mapping_sec': float(setup_sec),
        'train_data_load_sec': float(data_sec),
        'train_baseline_eval_sec': float(baseline_sec),
        'optimization_wall_sec': float(optimization_sec),
        'fitness_eval_within_optimization_sec': float(evaluator.eval_wall_sec),
        'pareto_and_checkpoint_save_sec': float(saving_sec),
        'total_repair_run_wall_sec': float(time.perf_counter()-wall_start),
        'notes': 'Prior locator elapsed is historical; fitness eval is included in optimization wall.',
    }
    summary = {
        'schema': 'CGARepair_adv_MOEAD_MOPSO_weight_repair_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'adv', 'model': model, 'epsilon': eps, 'optimizer': args.optimizer,
        'method': 'CGARepair_{}'.format(args.optimizer.upper()),
        'optimization_seed': int(args.seed), 'localization_seed': int(args.loc_seed),
        'localization': loc, 'localization_exported_dir': exported,
        'model_file': filepath,
        'selected_players': players, 'top_k': int(loc['top_k']),
        'total_ranked_neurons': int(loc['num_players']),
        'repair_specs': [{k: list(v) if isinstance(v, tuple) else v for k, v in s.items()}
                         for s in specs],
        'decision_variables': int(len(base_vector)),
        'parameter_budget_percent': float(100.*len(base_vector)/max(network.count_params(), 1)),
        'parameter_scope': 'Selected Conv/Dense kernel outputs only; BN and biases unchanged',
        'train_data': data_info,
        'original_train_metrics': baseline,
        'objective_definitions': ['adv_error_rate=minimize', '-clean_accuracy=minimize'],
        'threshold_accuracy': float(min_acc), 'threshold_adv_error': float(max_err),
        'max_evaluations_requested': int(max_evals),
        'evaluations_actual': int(evaluator.evaluations),
        'front_size': int(len(X)), 'front_feasible_count': int(np.sum(feasible)),
        'selected_index': int(chosen_idx),
        'selection_policy': selection_policy,
        'selected_train_adv_error': float(F[chosen_idx, 0]),
        'selected_train_robustness': float(1-F[chosen_idx, 0]),
        'selected_train_clean_accuracy': float(-F[chosen_idx, 1]),
        'best_weights_file': str(best_file.resolve()),
        'optimizer_parameters': {
            'population_size': int(args.pop_size), 'max_gen': int(args.max_gen),
            'max_evaluations': int(max_evals), 'init_jitter': float(args.init_jitter),
            'search_radius': float(args.search_radius),
            'neighbor_count': int(args.neighbors), 'neighbor_prob': float(args.neighbor_prob),
            'mutation_prob': float(.9 if args.mutation_prob is None else args.mutation_prob),
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
    print('[Done] {} eps={} {} train err={:.6f} robustness={:.6f} acc={:.6f} '
          '/ baseline err={:.6f} acc={:.6f}'.format(model, eps, args.optimizer,
          summary['selected_train_adv_error'], summary['selected_train_robustness'],
          summary['selected_train_clean_accuracy'], baseline['adv_error_rate'],
          baseline['clean_accuracy']), flush=True)
    print('[Timing] localization_previous={}s optimization={:.2f}s total_repair={:.2f}s'.format(
        loc['source_runtime_sec'], optimization_sec, timing['total_repair_run_wall_sec']), flush=True)
    print('[Files] summary={} best={} front={}'.format(outdir/'summary.json',
                                                      best_file, outdir/'pareto_front.pkl'), flush=True)
    return summary


def main():
    args = get_parser().parse_args()
    run(args)


if __name__ == '__main__':
    main()
