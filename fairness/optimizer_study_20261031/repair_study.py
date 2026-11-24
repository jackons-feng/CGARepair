"""CGARepair fairness local repair using genuine MOEA/D or MOPSO.

Targets: (minimize absolute sensitive-group accuracy gap, maximize overall accuracy).
Bank, Adult, Default; original CFRepair read.py is not touched.
CGARepair ranking uses actual model.layers indices, with BatchNorm offsets accounted for.
All weights are selected using a TRAIN sample; TEST is strictly untouched here.
"""
import argparse
import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np

from read_CGARepair import load_ranking, export_ranking, verify_model_signature
from cgarepair_fairness_utils import (
    check_dataset, model_path_for, load_fairness_split, set_up_tf_graph,
    evaluate_model, build_neuron_specs, apply_vector, save_json, sha256_file,
)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def dominates(a, b, tol=1e-12):
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


def crowding_distance(f):
    f = np.asarray(f, dtype=float)
    n = len(f)
    if n <= 2:
        return np.full(n, np.inf)
    scores = np.zeros(n)
    for j in range(f.shape[1]):
        ids = np.argsort(f[:, j], kind='mergesort')
        scores[ids[0]] = np.inf
        scores[ids[-1]] = np.inf
        span = f[ids[-1], j] - f[ids[0], j]
        if span > 0:
            for idx in range(1, n - 1):
                scores[ids[idx]] += (f[ids[idx + 1], j] - f[ids[idx - 1], j]) / span
    return scores


class HistoricalArchive:
    """Keep historical nondominated candidates; never select checkpoint on TEST."""
    def __init__(self, max_size, x0):
        self.max_size = int(max_size)
        self.x0 = np.asarray(x0, dtype=float)
        self.X, self.F = [], []

    def update(self, x, f):
        f = np.asarray(f, dtype=float)
        if not np.all(np.isfinite(f)):
            raise RuntimeError('Optimizer produced nonfinite objective')
        remove = []
        for i, old in enumerate(self.F):
            if np.all(np.abs(old - f) < 1e-12):
                # Prefer smallest perturbation if objective values tie.
                if np.linalg.norm(x - self.x0) < np.linalg.norm(self.X[i] - self.x0):
                    self.X[i] = np.asarray(x, dtype=float).copy()
                return
            if dominates(old, f):
                return
            if dominates(f, old):
                remove.append(i)
        for idx in reversed(remove):
            del self.X[idx]
            del self.F[idx]
        self.X.append(np.asarray(x, dtype=float).copy())
        self.F.append(f.copy())
        while len(self.X) > self.max_size:
            dist = crowding_distance(np.asarray(self.F))
            finite = np.flatnonzero(np.isfinite(dist))
            # Preserve min-gap and max-accuracy extremes.
            drop = int(finite[np.argmin(dist[finite])]) if len(finite) else len(self.F) - 1
            del self.X[drop]
            del self.F[drop]

    def as_arrays(self):
        if not self.X:
            raise RuntimeError('No optimizer candidates in archive')
        return np.asarray(self.X, dtype=np.float64), np.asarray(self.F, dtype=np.float64)


class FairnessEvaluator:
    def __init__(self, model, session, base_weights, specs, data_x, data_y, data_g,
                 x0, batch_size=1024, archive_size=120):
        self.model = model
        self.session = session
        self.specs = specs
        self.base_weights = [a.copy() for a in base_weights]
        self.work_weights = [a.copy() for a in base_weights]
        self.x, self.y, self.g = data_x, data_y, data_g
        self.batch_size = int(batch_size)
        self.archive = HistoricalArchive(archive_size, x0)
        self.evaluations = 0
        self.evaluate_wall_sec = 0.0

    def evaluate_batch(self, candidates):
        xs = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        values = np.empty((len(xs), 2), dtype=np.float64)
        for idx, vector in enumerate(xs):
            started = time.perf_counter()
            apply_vector(self.work_weights, self.specs, vector)
            self.model.set_weights(self.work_weights)
            m = evaluate_model(self.model, self.x, self.y, self.g, self.session,
                               batch_size=self.batch_size)
            f = np.asarray((m['fairness_gap'], -m['accuracy']), dtype=np.float64)
            values[idx, :] = f
            self.evaluations += 1
            self.archive.update(vector, f)
            self.evaluate_wall_sec += time.perf_counter() - started
        return values

    def save_weights(self, vector, filepath):
        apply_vector(self.work_weights, self.specs, vector)
        self.model.set_weights(self.work_weights)
        self.model.save_weights(str(filepath))

    def reset(self):
        self.model.set_weights(self.base_weights)


def run_moead(ev, x0, lower, upper, args, budget):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.core.callback import Callback
    except ImportError as exc:
        raise ImportError('MOEA/D requires pymoo compatible with your old environment: {}'.format(exc))

    class FairnessProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(x0), n_obj=2, n_ieq_constr=0,
                             xl=lower, xu=upper, vtype=float)

        def _evaluate(self, X, out, *unused, **kwunused):
            out['F'] = ev.evaluate_batch(X)

    class Progress(Callback):
        def __init__(self):
            super().__init__()
            self.history = []

        def notify(self, algorithm):
            generation = int(algorithm.n_gen)
            if generation % args.log_every and ev.evaluations < budget:
                return
            if not ev.archive.F:
                return
            front = np.asarray(ev.archive.F)
            row = {'generation': generation, 'evaluations': int(ev.evaluations),
                   'min_fairness_gap': float(np.min(front[:, 0])),
                   'max_accuracy': float(np.max(-front[:, 1])),
                   'front_size': int(len(front))}
            self.history.append(row)
            print('[MOEAD] gen={} FEs={} historical_front={} minGap={:.6f} maxAcc={:.6f}'.format(
                generation, row['evaluations'], row['front_size'],
                row['min_fairness_gap'], row['max_accuracy']), flush=True)

    refs = get_reference_directions('das-dennis', 2, n_partitions=args.pop_size - 1)
    if len(refs) != args.pop_size:
        raise RuntimeError('Expected {} MOEA/D refs, got {}'.format(args.pop_size, len(refs)))
    rng = np.random.RandomState(int(args.seed))
    initial = x0 + rng.uniform(-args.init_jitter, args.init_jitter,
                               (args.pop_size, len(x0)))
    initial = np.clip(initial, lower, upper)
    initial[0, :] = x0

    mutargs = {'prob': args.mutation_prob, 'eta': args.mutation_eta}
    if args.mutation_prob_var is not None:
        import inspect
        if 'prob_var' not in inspect.signature(PolynomialMutation).parameters:
            raise ValueError('Installed pymoo does not support --mutation-prob-var. '
                             'Omit it to use the original MOEA/D mutation configuration.')
        mutargs['prob_var'] = args.mutation_prob_var
    mutation = PolynomialMutation(**mutargs)
    algorithm = MOEAD(ref_dirs=refs,
                      n_neighbors=min(args.neighbors, args.pop_size - 1),
                      prob_neighbor_mating=args.neighbor_prob,
                      sampling=initial,
                      crossover=UniformCrossover(prob=args.crossover_prob),
                      mutation=mutation)
    callback = Progress()
    result = minimize(FairnessProblem(), algorithm,
                      get_termination('n_eval', int(budget)),
                      seed=int(args.seed), verbose=False, callback=callback,
                      save_history=False)
    if ev.evaluations == 0:
        raise RuntimeError('No MOEA/D solution evaluated')
    front_x, front_f = ev.archive.as_arrays()
    print('[MOEAD] generations={} FEs={} historical_front={}'.format(
        int(getattr(result.algorithm, 'n_gen', 0)), ev.evaluations, len(front_x)))
    return front_x, front_f, callback.history, int(getattr(result.algorithm, 'n_gen', 0))




# NSGA-III ablation: same TRAIN problem, initial population, mutation, crossover,
# selection policy and input bounds as native MOEA/D.
def run_nsga3(ev, x0, lower, upper, args, budget):
    try:
        from pymoo.core.problem import Problem
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.operators.mutation.pm import PolynomialMutation
        from pymoo.operators.crossover.ux import UniformCrossover
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        from pymoo.core.callback import Callback
    except ImportError as exc:
        raise ImportError('NSGA-III requires pymoo compatible with your old environment: {}'.format(exc))

    class FairnessProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(x0), n_obj=2, n_ieq_constr=0,
                             xl=lower, xu=upper, vtype=float)

        def _evaluate(self, X, out, *unused, **kwunused):
            out['F'] = ev.evaluate_batch(X)

    class Progress(Callback):
        def __init__(self):
            super().__init__()
            self.history = []

        def notify(self, algorithm):
            generation = int(algorithm.n_gen)
            if generation % args.log_every and ev.evaluations < budget:
                return
            if not ev.archive.F:
                return
            front = np.asarray(ev.archive.F)
            row = {'generation': generation, 'evaluations': int(ev.evaluations),
                   'min_fairness_gap': float(np.min(front[:, 0])),
                   'max_accuracy': float(np.max(-front[:, 1])),
                   'front_size': int(len(front))}
            self.history.append(row)
            print('[NSGA3] gen={} FEs={} historical_front={} minGap={:.6f} maxAcc={:.6f}'.format(
                generation, row['evaluations'], row['front_size'],
                row['min_fairness_gap'], row['max_accuracy']), flush=True)

    refs = get_reference_directions('das-dennis', 2, n_partitions=args.pop_size - 1)
    if len(refs) != args.pop_size:
        raise RuntimeError('Expected {} MOEA/D refs, got {}'.format(args.pop_size, len(refs)))
    rng = np.random.RandomState(int(args.seed))
    initial = x0 + rng.uniform(-args.init_jitter, args.init_jitter,
                               (args.pop_size, len(x0)))
    initial = np.clip(initial, lower, upper)
    initial[0, :] = x0

    mutargs = {'prob': args.mutation_prob, 'eta': args.mutation_eta}
    if args.mutation_prob_var is not None:
        import inspect
        if 'prob_var' not in inspect.signature(PolynomialMutation).parameters:
            raise ValueError('Installed pymoo does not support --mutation-prob-var. '
                             'Omit it to use the original MOEA/D mutation configuration.')
        mutargs['prob_var'] = args.mutation_prob_var
    mutation = PolynomialMutation(**mutargs)
    algorithm = NSGA3(pop_size=args.pop_size, ref_dirs=refs,
                      sampling=initial,
                      crossover=UniformCrossover(prob=args.crossover_prob),
                      mutation=mutation, eliminate_duplicates=False)
    callback = Progress()
    result = minimize(FairnessProblem(), algorithm,
                      get_termination('n_eval', int(budget)),
                      seed=int(args.seed), verbose=False, callback=callback,
                      save_history=False)
    if ev.evaluations == 0:
        raise RuntimeError('No NSGA-III solution evaluated')
    front_x, front_f = ev.archive.as_arrays()
    print('[NSGA3] generations={} FEs={} historical_front={}'.format(
        int(getattr(result.algorithm, 'n_gen', 0)), ev.evaluations, len(front_x)))
    return front_x, front_f, callback.history, int(getattr(result.algorithm, 'n_gen', 0))



def _mopso_leader(archive_x, archive_f, rng):
    """Crowding-distance tournament, preferring underrepresented Pareto regions."""
    if len(archive_x) == 1:
        return archive_x[0]
    # Both competitors are drawn from the historical nondominated archive.
    a, b = rng.randint(0, len(archive_x), size=2)
    distances = crowding_distance(archive_f)
    return archive_x[a if distances[a] >= distances[b] else b]


def run_mopso(ev, x0, lower, upper, args, budget):
    """Historical-archive MOPSO: same objective/bounds/initial pop/FEs as MOEA/D.

    All objective evaluations use the same fixed TRAIN sample.  No TEST data
    are loaded, and checkpoint selection is handled downstream on TRAIN only.
    """
    rng = np.random.RandomState(int(args.seed))
    n = int(args.pop_size)
    dimension = len(x0)
    span = np.asarray(upper - lower, dtype=np.float64)
    positions = x0 + rng.uniform(-args.init_jitter, args.init_jitter, (n, dimension))
    positions = np.clip(positions, lower, upper)
    positions[0, :] = x0
    velocity = rng.uniform(-.005, .005, (n, dimension)) * span
    vmax = args.vmax_fraction * span

    scores = ev.evaluate_batch(positions)
    pbest = positions.copy()
    pbest_f = scores.copy()
    history = []
    generation = 1
    while ev.evaluations < budget:
        count = min(n, budget - ev.evaluations)
        archive_x, archive_f = ev.archive.as_arrays()
        leaders = np.asarray([_mopso_leader(archive_x, archive_f, rng)
                              for _ in range(count)], dtype=np.float64)
        r1 = rng.uniform(size=(count, dimension))
        r2 = rng.uniform(size=(count, dimension))
        velocity[:count] = (args.inertia * velocity[:count]
                            + args.c1 * r1 * (pbest[:count] - positions[:count])
                            + args.c2 * r2 * (leaders - positions[:count]))
        velocity[:count] = np.clip(velocity[:count], -vmax, vmax)
        proposal = np.clip(positions[:count] + velocity[:count], lower, upper)
        touching = (proposal <= lower) | (proposal >= upper)
        velocity[:count][touching] *= -.5
        new_f = ev.evaluate_batch(proposal)
        positions[:count] = proposal

        for i in range(count):
            if dominates(new_f[i], pbest_f[i]):
                update = True
            elif dominates(pbest_f[i], new_f[i]):
                update = False
            elif np.all(np.abs(new_f[i] - pbest_f[i]) < 1e-12):
                # If both objectives are exactly tied, prefer less weight drift.
                update = np.linalg.norm(positions[i] - x0) < np.linalg.norm(pbest[i] - x0)
            else:
                update = bool(rng.rand() < .5)
            if update:
                pbest[i] = positions[i].copy()
                pbest_f[i] = new_f[i].copy()
        generation += 1
        if generation % args.log_every == 0 or ev.evaluations >= budget:
            _, front = ev.archive.as_arrays()
            row = {'generation': generation, 'evaluations': int(ev.evaluations),
                   'min_fairness_gap': float(np.min(front[:, 0])),
                   'max_accuracy': float(np.max(-front[:, 1])),
                   'front_size': int(len(front))}
            history.append(row)
            print('[MOPSO] gen={} FEs={} historical_front={} minGap={:.6f} maxAcc={:.6f}'.format(
                generation, row['evaluations'], row['front_size'],
                row['min_fairness_gap'], row['max_accuracy']), flush=True)

    front_x, front_f = ev.archive.as_arrays()
    if ev.evaluations != budget:
        raise RuntimeError('MOPSO budget mismatch: {} != {}'.format(ev.evaluations, budget))
    print('[MOPSO] generations={} FEs={} historical_front={}'.format(
        generation, ev.evaluations, len(front_x)), flush=True)
    return front_x, front_f, history, generation


def select_train_solution(F, baseline_gap, baseline_acc, min_acc, gap_target):
    F = np.asarray(F, dtype=float)
    gap, acc = F[:, 0], -F[:, 1]
    eligible = np.flatnonzero((acc >= min_acc - 1e-12) &
                              (gap <= baseline_gap + 1e-12))
    if len(eligible) == 0:
        return None, 'no_train_candidate_meets_accuracy_and_no_fairness_regression'
    if gap_target is not None:
        target = eligible[gap[eligible] <= gap_target + 1e-12]
        if len(target):
            eligible = target
            policy = 'target_feasible_min_gap_then_max_acc'
        else:
            policy = 'utility_guard_min_gap_then_max_acc'
    else:
        policy = 'utility_guard_min_gap_then_max_acc'
    indices = np.lexsort((-acc[eligible], gap[eligible]))
    winner = int(eligible[indices[0]])
    return winner, policy


def output_dir(args):
    suffix = ('_' + args.run_tag) if args.run_tag else ('_quick' if args.quick else '')
    if args.quick and args.run_tag and not suffix.endswith('_quick'):
        suffix += '_quick'
    return Path(args.output_root) / args.dataset / 'CGARepair' / ('seed_' + str(args.seed)) / (args.optimizer + suffix)


def parse_args():
    p = argparse.ArgumentParser(description='CGARepair fairness local repair by MOEA/D or MOPSO')
    p.add_argument('dataset', choices=('bank', 'adult', 'default'))
    p.add_argument('seed', type=int, help='Optimization seed, e.g. 42')
    p.add_argument('--optimizer', choices=('moead', 'nsga3', 'mopso'), default='moead')
    p.add_argument('--loc-seed', type=int, default=256)
    p.add_argument('--loc-samples', type=int, default=4096)
    p.add_argument('--loc-json')
    p.add_argument('--loc-root')
    p.add_argument('--top-k', type=int)
    p.add_argument('--skip-signature-check', action='store_true')
    p.add_argument('--model-root')
    p.add_argument('--data-root')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--pop-size', type=int, default=100)
    p.add_argument('--max-gen', type=int, default=100)
    p.add_argument('--max-evals', type=int, default=10000)
    p.add_argument('--eval-samples', type=int, default=10000,
                   help='TRAIN sample count; 0 means FULL TRAIN; NEVER reads TEST')
    p.add_argument('--eval-batch-size', type=int, default=1024)
    p.add_argument('--search-radius', type=float, default=10.0)
    p.add_argument('--init-jitter', type=float, default=.01)
    p.add_argument('--neighbors', type=int, default=15)
    p.add_argument('--neighbor-prob', type=float, default=.7)
    p.add_argument('--mutation-prob', type=float, default=.9)
    p.add_argument('--mutation-eta', type=float, default=20.)
    p.add_argument('--mutation-prob-var', type=float)
    p.add_argument('--crossover-prob', type=float, default=1.)
    p.add_argument('--inertia', type=float, default=.6, help='MOPSO inertia weight')
    p.add_argument('--c1', type=float, default=1.5, help='MOPSO cognitive coefficient')
    p.add_argument('--c2', type=float, default=1.5, help='MOPSO social coefficient')
    p.add_argument('--vmax-fraction', type=float, default=.1, help='MOPSO velocity limit as fraction of search span')
    p.add_argument('--max-acc-drop', type=float, default=.02,
                   help='TRAIN clean-accuracy drop allowed, fractional units (0.02=2pp)')
    p.add_argument('--min-acc', type=float, default=None,
                   help='Override dynamic utility guard')
    p.add_argument('--gap-target', type=float, default=.05,
                   help='Optional TRAIN gap target, DOES NOT set pymoo hard constraint')
    p.add_argument('--archive-size', type=int, default=120)
    p.add_argument('--save-front-weights', action='store_true')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--run-tag', default='')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    args.dataset = check_dataset(args.dataset)
    if args.quick:
        args.pop_size, args.max_gen, args.max_evals = 12, 4, 48
        args.eval_samples = 256
        args.archive_size = min(25, args.archive_size)
    if args.pop_size < 3 or args.max_gen < 1 or args.max_evals < args.pop_size:
        p.error('Need pop-size>=3, max-gen>=1, max-evals>=pop-size')
    if args.eval_samples < 0 or args.search_radius <= 0 or args.eval_batch_size <= 0:
        p.error('Invalid eval-samples, search-radius, or eval-batch-size')
    if not 0 <= args.max_acc_drop <= 1 or not 0 <= args.mutation_prob <= 1:
        p.error('Invalid max-acc-drop or mutation-prob')
    if args.mutation_prob_var is not None and not 0 <= args.mutation_prob_var <= 1:
        p.error('Invalid mutation-prob-var')
    if args.min_acc is not None and not 0 <= args.min_acc <= 1:
        p.error('Invalid min-acc')
    if args.gap_target is not None and not 0 <= args.gap_target <= 1:
        p.error('Invalid gap-target')
    if not 0 <= args.inertia <= 2 or args.c1 < 0 or args.c2 < 0 or not 0 < args.vmax_fraction <= 1:
        p.error('MOPSO needs inertia in [0,2], c1/c2>=0, vmax-fraction in (0,1]')
    if args.archive_size < 2 or args.log_every < 1:
        p.error('archive-size>=2, log-every>=1 are required')
    return args


def run(args):
    wall_start = time.perf_counter()
    run_dir = output_dir(args)
    if (run_dir / 'summary.json').exists() and not args.overwrite:
        raise FileExistsError('Run exists: {}. Choose another --run-tag or --overwrite.'.format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_start = time.perf_counter()
    model_path = model_path_for(args.dataset, args.model_root)
    players, scores, provenance = load_ranking(
        args.dataset, args.loc_seed, args.loc_samples,
        args.loc_json, args.loc_root, top_k=args.top_k)
    signature_check = 'skipped_by_user' if args.skip_signature_check else verify_model_signature(
        provenance, model_path)
    export_ranking(players, scores, provenance, run_dir / 'localization_input')
    read_wall = time.perf_counter() - stage_start
    print('[Locator] {} Top-{} from {} (loc_seed={}, opt_seed={})'.format(
        args.dataset, len(players), provenance['source_json'], args.loc_seed, args.seed), flush=True)
    print('[Locator] selected={}'.format(players), flush=True)

    stage_start = time.perf_counter()
    np.random.seed(args.seed)
    tf, K, session = set_up_tf_graph()
    tf.compat.v1.set_random_seed(int(args.seed))
    from tensorflow.keras.models import load_model
    model = load_model(str(model_path), compile=False)
    specs, x0, original_weights = build_neuron_specs(model, players)
    X, Y, G, train_manifest = load_fairness_split(
        args.dataset, 'train', args.data_root,
        sample_count=args.eval_samples, sample_seed=args.seed)
    # Old model/data schema mismatch should be detected before spending 10k FEs.
    declared_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if tuple(declared_shape[1:]) != tuple(X.shape[1:]):
        raise ValueError('Model expects {}, train inputs are {}'.format(declared_shape, X.shape))
    baseline = evaluate_model(model, X, Y, G, session, args.eval_batch_size, include_f1=True)
    min_acc = float(args.min_acc) if args.min_acc is not None else max(0., baseline['accuracy'] - args.max_acc_drop)
    print('[Repair] dataset={} selected_neurons={} decision_variables={} train_samples={}'.format(
        args.dataset, len(specs), len(x0), len(X)), flush=True)
    print('[Baseline] TRAIN Gap={:.6f} Acc={:.6f} F1={:.6f} guardMinAcc={:.6f}'.format(
        baseline['fairness_gap'], baseline['accuracy'], baseline['f1'], min_acc), flush=True)
    data_wall = time.perf_counter() - stage_start

    lower = np.maximum(-5., x0 - args.search_radius)
    upper = np.minimum(5., x0 + args.search_radius)
    if np.any(lower > upper) or np.any((x0 < lower - 1e-10) | (x0 > upper + 1e-10)):
        raise ValueError('Some original weights are outside [-5,5], so the legacy bounds '
                         'cannot contain the original model. Confirm model and adjust code explicitly.')
    n_gens = min(args.max_gen, args.max_evals // args.pop_size)
    budget = int(n_gens * args.pop_size)
    evaluator = FairnessEvaluator(model, session, original_weights, specs, X, Y, G, x0,
                                  batch_size=args.eval_batch_size,
                                  archive_size=args.archive_size)
    stage_start = time.perf_counter()
    if args.optimizer == 'moead':
        front_x, front_f, history, actual_gens = run_moead(
            evaluator, x0, lower, upper, args, budget)
    elif args.optimizer == 'nsga3':
        front_x, front_f, history, actual_gens = run_nsga3(
            evaluator, x0, lower, upper, args, budget)
    else:
        front_x, front_f, history, actual_gens = run_mopso(
            evaluator, x0, lower, upper, args, budget)
    optimization_wall = time.perf_counter() - stage_start
    if evaluator.evaluations != budget:
        raise RuntimeError('Actual evaluations {} != expected {} (check pymoo generation size)'.format(
            evaluator.evaluations, budget))
    selected_index, policy = select_train_solution(front_f, baseline['fairness_gap'],
                                                    baseline['accuracy'], min_acc, args.gap_target)
    if selected_index is None:
        best_vector = x0.copy()
        print('[Selector] No candidate meets guard and non-worsening gap; restore baseline.', flush=True)
    else:
        best_vector = front_x[selected_index]
    stage_start = time.perf_counter()
    evaluator.save_weights(best_vector, run_dir / 'best_weights.h5')
    best_train = evaluate_model(model, X, Y, G, session,
                                args.eval_batch_size, include_f1=True)
    evaluator.reset()

    # Archive metadata, baseline model remains a valid guard fallback.
    front_dir = run_dir / 'pareto_front'
    front_dir.mkdir(parents=True, exist_ok=True)
    with open(str(run_dir / 'pareto_front.pkl'), 'wb') as file:
        pickle.dump({'X': front_x, 'F': front_f, 'players': players,
                     'optimizer': args.optimizer, 'evaluation_scope': 'train'},
                    file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(str(front_dir / 'pareto_front.csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'train_fairness_gap', 'train_accuracy',
                         'meets_accuracy_guard', 'improves_or_preserves_fairness'])
        for index, row in enumerate(front_f):
            writer.writerow([index, '{:.12g}'.format(row[0]), '{:.12g}'.format(-row[1]),
                             int(-row[1] >= min_acc - 1e-12),
                             int(row[0] <= baseline['fairness_gap'] + 1e-12)])
    if args.save_front_weights:
        for i, vector in enumerate(front_x):
            evaluator.save_weights(vector, front_dir / ('weights_' + str(i) + '.h5'))
    evaluator.reset()
    save_wall = time.perf_counter() - stage_start
    previous_loc = provenance.get('source_runtime_sec')
    time_data = {
        'localization_previous_run_sec': previous_loc,
        'localization_previous_run_excluded_from_current_wall': True,
        'localization_read_sec': float(read_wall),
        'train_data_model_baseline_sec': float(data_wall),
        'optimization_wall_sec': float(optimization_wall),
        'objective_evaluation_wall_sec': float(evaluator.evaluate_wall_sec),
        'save_results_sec': float(save_wall),
        'repair_run_total_wall_sec': float(time.perf_counter() - wall_start),
        'independent_test_wall_sec': None}
    summary = {
        'task': 'fairness', 'method': 'CGARepair', 'optimizer': args.optimizer,
        'dataset': args.dataset, 'seed': int(args.seed), 'loc_seed': int(args.loc_seed),
        'localization': provenance, 'model_signature_check': signature_check,
        'model_path': str(model_path), 'model_sha256': sha256_file(model_path),
        'train_data': train_manifest,
        'evaluation_scope': 'train_only', 'test_used_for_selection': False,
        'fairness_metric': 'absolute_gap_of_sensitive_group_classification_accuracies',
        'objectives': ['min_group_accuracy_gap', 'max_overall_accuracy'],
        'selected_neurons': players, 'parameter_specs': specs,
        'decision_variables': int(len(x0)),
        'total_original_model_parameters': int(sum(int(a.size) for a in original_weights)),
        'best_vector_l1_shift': float(np.sum(np.abs(best_vector - x0))),
        'best_vector_l2_shift': float(np.linalg.norm(best_vector - x0)),
        'modified_selected_variables': int(np.count_nonzero(np.abs(best_vector-x0) > 1e-7)),
        'baseline_train': baseline, 'repaired_train': best_train,
        'delta_fairness_gap': float(best_train['fairness_gap'] - baseline['fairness_gap']),
        'delta_accuracy': float(best_train['accuracy'] - baseline['accuracy']),
        'min_train_accuracy': float(min_acc), 'target_train_gap': float(args.gap_target),
        'best_front_index': selected_index,
        'selection_policy': policy,
        'train_accuracy_guard_met': bool(best_train['accuracy'] >= min_acc - 1e-12),
        'train_gap_target_met': bool(best_train['fairness_gap'] <= args.gap_target + 1e-12),
        'population_size': int(args.pop_size), 'max_evals_requested': int(args.max_evals),
        'eval_budget_effective': int(budget), 'total_evaluations': int(evaluator.evaluations),
        'generations': int(actual_gens), 'historical_pareto_size': int(len(front_x)),
        'optimizer_params': {
            'search_radius': args.search_radius, 'absolute_bounds': [-5, 5],
            'init_jitter': args.init_jitter, 'neighbors': args.neighbors,
            'neighbor_prob': args.neighbor_prob, 'mutation_prob': args.mutation_prob,
            'mutation_eta': args.mutation_eta, 'mutation_prob_var': args.mutation_prob_var,
            'crossover_prob': args.crossover_prob,
            'inertia': args.inertia, 'c1': args.c1, 'c2': args.c2,
            'vmax_fraction': args.vmax_fraction},
        'history': history, 'timing_sec': time_data,
        'files': {'best_weights': 'best_weights.h5',
                  'pareto_vectors': 'pareto_front.pkl',
                  'pareto_metadata': 'pareto_front/pareto_front.csv'},
    }
    save_json(run_dir / 'summary.json', summary)
    save_json(run_dir / 'time_summary.json', time_data)
    print('[Done] {} {}: TRAIN gap {:.6f}->{:.6f}; acc {:.6f}->{:.6f}'.format(
        args.dataset, args.optimizer.upper(), baseline['fairness_gap'], best_train['fairness_gap'],
        baseline['accuracy'], best_train['accuracy']), flush=True)
    print('[Timing] localization(previous)={}s, optimization={:.2f}s, total={:.2f}s'.format(
        previous_loc, optimization_wall, time_data['repair_run_total_wall_sec']), flush=True)
    print('[Files] {} {}'.format(run_dir / 'summary.json', run_dir / 'best_weights.h5'), flush=True)
    return summary


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
