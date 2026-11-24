#!/usr/bin/env python3
"""Audit and aggregate seed-wise RQ1 cumulative-ablation results for 3 locators."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from run_rq1_baselines import SUITES, root_for, is_valid, metrics_from, DEFAULT_SEEDS

METHODS = ('CGARepair', 'BIRDNN', 'ccbr')


def mean_std(vals):
    values = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None) if values else (None, None)


def fmt(v):
    return '' if v is None else '{:.8g}'.format(float(v))


def write_table(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def candidate_files(root, task, model, samples, epsilon, resnet, seed, method):
    sub = root_for(root, task, model, epsilon, resnet)
    return sorted(sub.glob('top*_samples{}_{}_seed{}.txt'.format(samples, method, seed)))


def load_record(root, task, model, samples, epsilon, resnet, seed, method):
    for path in candidate_files(root, task, model, samples, epsilon, resnet, seed, method):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            k, n = int(data['top_k']), int(data['num_players'])
            expected = min(n, max(4, min(11, int(n * 0.01))))
            if n <= 0 or k != expected or data.get('task') != task or data.get('model') != model:
                continue
            if int(data.get('seed', -1)) != seed:
                continue
            if method != 'CGARepair' and not is_valid(path, method, task, model, samples, seed):
                continue
            metrics = metrics_from(data, k)
            if metrics is None:
                continue
            baseline = data.get('baseline_metric_before_disable', None)
            baseline = float(baseline) if baseline is not None else None
            if baseline is not None and (not math.isfinite(baseline) or baseline < -1e-8 or baseline > 1 + 1e-8):
                continue
            fp = data.get('fairness_sample_fingerprint')
            if task == 'fairness' and not fp:
                continue
            select_path = data.get('selected_topk_path')
            if not select_path:
                select_path = str(path)[:-4] + '_selected_topk.txt'
            if not Path(select_path).exists():
                selected = []
            else:
                selected = [x.strip() for x in Path(select_path).read_text(encoding='utf-8').strip().split(',') if x.strip()]
            return {
                'data': data, 'path': str(path), 'method': method,
                'task': task, 'model': model, 'seed': seed,
                'n': n, 'k': k, 'samples': samples, 'metrics': metrics,
                'baseline': baseline, 'final': metrics[-1],
                'improvement': (baseline - metrics[-1]) if baseline is not None else None,
                'time': data.get('localization_time_sec'),
                'fairness_fp': fp, 'selected': selected,
            }
        except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError):
            pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='result')
    p.add_argument('--out', default='result/rq1_three_method_summary')
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    args = p.parse_args()
    root, out = Path(args.root), Path(args.out)
    audit = []
    valid = defaultdict(list)
    per_seed = []
    paired = defaultdict(list)
    topk_pairs = []
    all_specs = [(task, model, samples, eps, rn) for task, configs in SUITES.items() for model, samples, eps, rn in configs]

    for task, model, samples, eps, rn in all_specs:
        for seed in args.seeds:
            records = {}
            for method in METHODS:
                rec = load_record(root, task, model, samples, eps, rn, seed, method)
                if not rec:
                    audit.append('MISSING_OR_INVALID {} {} seed={} method={}'.format(task, model, seed, method))
                    continue
                records[method] = rec
                valid[(task, model, method)].append(rec)
                per_seed.append({
                    'task': task, 'model': model, 'method': method, 'seed': seed,
                    'k': rec['k'], 'num_players': rec['n'], 'samples': samples,
                    'baseline': fmt(rec['baseline']), 'topk_final': fmt(rec['final']),
                    'improvement': fmt(rec['improvement']),
                    'localization_time_sec': fmt(rec['time']),
                    'fairness_fingerprint': rec['fairness_fp'] or '',
                    'json_path': rec['path'],
                })
            if len(records) != len(METHODS):
                continue
            if len({(r['n'], r['k']) for r in records.values()}) != 1:
                audit.append('INCOMPARABLE_CANDIDATE_SET {} {} seed={} sizes={}'.format(task, model, seed, {
                    m: (r['n'], r['k']) for m, r in records.items()}))
                continue
            if task == 'fairness':
                fingerprints = {m: r['fairness_fp'] for m, r in records.items()}
                if len(set(fingerprints.values())) != 1:
                    audit.append('INCOMPARABLE_FAIRNESS_SAMPLES {} {} seed={} fingerprints={}'.format(task, model, seed, fingerprints))
                    continue
                baseline_vals = [r['baseline'] for r in records.values() if r['baseline'] is not None]
                if len(baseline_vals) > 1 and max(baseline_vals) - min(baseline_vals) > 1e-7:
                    audit.append('INCOMPARABLE_FAIRNESS_BASELINE {} {} seed={} {}'.format(task, model, seed, baseline_vals))
                    continue
            for method, rec in records.items():
                paired[(task, model, method)].append(rec)
            for j, m1 in enumerate(METHODS):
                for m2 in METHODS[j+1:]:
                    a, b = set(records[m1]['selected']), set(records[m2]['selected'])
                    if len(a) != records[m1]['k'] or len(b) != records[m2]['k']:
                        audit.append('MISSING_SELECTED_TOPK {} {} seed={} {} {}'.format(task, model, seed, m1, m2))
                        continue
                    topk_pairs.append({
                        'task': task, 'model': model, 'seed': seed,
                        'method_a': m1, 'method_b': m2,
                        'intersection': len(a & b), 'jaccard': fmt(len(a & b) / len(a | b)),
                    })
    def aggregate(groups):
        result = []
        for (task, model, method), records in sorted(groups.items()):
            b, bs = mean_std([r['baseline'] for r in records])
            v, vs = mean_std([r['final'] for r in records])
            im, ims = mean_std([r['improvement'] for r in records])
            t, ts = mean_std([r['time'] for r in records])
            result.append({
                'task': task, 'model': model, 'method': method,
                'n_seeds': len(records), 'expected_seeds': len(args.seeds),
                'k': records[0]['k'], 'num_players': records[0]['n'],
                'baseline_mean': fmt(b), 'baseline_std': fmt(bs),
                'topk_mean': fmt(v), 'topk_std': fmt(vs),
                'improvement_mean': fmt(im), 'improvement_std': fmt(ims),
                'localization_seconds_mean': fmt(t), 'localization_seconds_std': fmt(ts),
            })
        return result
    agg_cols = ['task', 'model', 'method', 'n_seeds', 'expected_seeds', 'k', 'num_players',
                'baseline_mean', 'baseline_std', 'topk_mean', 'topk_std', 'improvement_mean',
                'improvement_std', 'localization_seconds_mean', 'localization_seconds_std']
    write_table(out/'summary_by_method.csv', agg_cols, aggregate(valid))
    write_table(out/'summary_paired.csv', agg_cols, aggregate(paired))
    write_table(out/'per_seed.csv', ['task', 'model', 'method', 'seed', 'k', 'num_players', 'samples',
                 'baseline', 'topk_final', 'improvement', 'localization_time_sec',
                 'fairness_fingerprint', 'json_path'], per_seed)
    curves = []
    for (task, model, method), records in sorted(paired.items()):
        for step in range(1, records[0]['k'] + 1):
            values = [r['metrics'][step-1] for r in records]
            mu, sd = mean_std(values)
            curves.append({
                'task': task, 'model': model, 'method': method, 'step': step,
                'n_seeds': len(values), 'metric_mean': fmt(mu), 'metric_std': fmt(sd),
            })
    write_table(out/'curves_paired.csv', ['task', 'model', 'method', 'step', 'n_seeds', 'metric_mean', 'metric_std'], curves)
    write_table(out/'pairwise_topk.csv', ['task', 'model', 'seed', 'method_a', 'method_b', 'intersection', 'jaccard'], topk_pairs)
    audit.append('INFO: For backdoor/adv/safety, existing CGARepair outputs do not provide a common evaluation-set fingerprint; exact method-paired sample identity cannot be verified.')
    audit.append('INFO: Localization+ablation use training-derived samples; results are internal RQ1 diagnostics, not held-out TEST performance.')
    audit.append('INFO: CGARepair non-fairness baseline_metric_before_disable is absent in old JSONs, so its improvement columns may be blank.')
    (out/'audit.txt').write_text('\n'.join(audit)+'\n',encoding='utf-8')
    n_complete = sum(1 for key, records in paired.items() if len(records) == len(args.seeds))
    print('Summary:', out)
    print('Valid per-seed records:', len(per_seed), 'full-5-seed paired method/config series:', n_complete)
    print('Audit entries:', len(audit))

if __name__ == '__main__':
    main()
