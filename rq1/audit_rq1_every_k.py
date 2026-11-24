#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only RQ1 per-step Top-k audit. No localization/inference is run.

Recompute each step's 5-seed mean and sample SD from ORIGINAL per-seed JSONs.
Compare against curves_paired.csv (non-ADV) and adv_1pcr_curves.csv (ADV).
Exclude only backdoor/mnist, giving 10 configs * 3 methods and 189 curve points.

Run from the cfrepair_RQ1 experiment root:
  python summarize_adv_1pcr.py   # only if adv_1pcr_curves.csv missing
  python audit_rq1_every_k.py
  python audit_rq1_every_k.py --verbose
"""
import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CONFIGS = [
    ('adv', 'mnist', 4),
    ('adv', 'cifar10', 11),
    ('backdoor', 'fashion', 4),
    ('backdoor', 'gtsrb', 11),
    ('safety', 'a19', 4),
    ('safety', 'a29', 4),
    ('safety', 'a33', 4),
    ('fairness', 'adult', 7),
    ('fairness', 'bank', 7),
    ('fairness', 'default', 7),
]
METHODS = ('CGARepair', 'BIRDNN', 'ccbr')
SEEDS = (42, 123, 256, 1024, 2026)
EXPECTED_KEYS = {(task, model, method, seed) for task, model, _ in CONFIGS for method in METHODS for seed in SEEDS}


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, cols, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def validate_values(seq, k, path, name, dict_mode=False):
    if not isinstance(seq, list):
        raise ValueError('{} missing/non-list {}'.format(path, name))
    if dict_mode and seq and isinstance(seq[0], dict):
        # The BIRDNN baseline can log attempted but skipped units; count only actual disables.
        seq = [x.get('metric_after_disable') for x in seq if x.get('status') == 'disabled']
    if len(seq) != k:
        raise ValueError('{}: {} has {} valid steps, expected {}'.format(path, name, len(seq), k))
    values = [float(v) for v in seq]
    if not all(math.isfinite(v) and -1.e-8 <= v <= 1+1.e-8 for v in values):
        raise ValueError('{}: {} contains nonfinite/out-of-range entries'.format(path, name))
    return values


def read_raw(row):
    p = Path(row['json_path'])
    if not p.is_file():
        raise FileNotFoundError('Missing raw JSON: {}'.format(p))
    data = json.loads(p.read_text(encoding='utf-8'))
    task, model, method, seed = row['task'], row['model'], row['method'], int(row['seed'])
    k = int(row['k'])
    if (data.get('task'), data.get('model'), data.get('method')) != (task, model, method):
        raise ValueError('Task/model/method mismatch in {}'.format(p))
    if int(data.get('seed', -1)) != seed or int(data.get('top_k', -1)) != k:
        raise ValueError('Seed/Top-K mismatch in {}'.format(p))
    num = int(data.get('num_players', -1))
    expected_k = min(num, max(4, min(11, num//100)))
    if num <= 0 or k != expected_k:
        raise ValueError('{}: Top-K={}, expected {} for num_players={}'.format(p,k,expected_k,num))
    if task == 'adv':
        field = 'top{}_one_minus_pcr_results'.format(k)
        values = validate_values(data.get(field), k, p, field)
    else:
        field = 'top{}_metric_results'.format(k)
        values = validate_values(data.get(field), k, p, field, dict_mode=True)
    fingerprint = data.get('fairness_sample_fingerprint') if task == 'fairness' else None
    if task == 'fairness' and not fingerprint:
        raise ValueError('{} missing fairness sample fingerprint'.format(p))
    return (values, fingerprint, num)


def reported_curves(summary):
    result = {}
    normal = summary/'curves_paired.csv'
    adv = summary/'adv_1pcr_curves.csv'
    if not normal.exists() or not adv.exists():
        raise FileNotFoundError('Need {} and {} (run summarize_adv_1pcr.py if 1-PCR CSV missing)'.format(normal, adv))
    for path,is_adv in ((normal,False),(adv,True)):
        for r in read_csv(path):
            task = 'adv' if is_adv else r['task']
            if (task,r['model']) == ('backdoor','mnist') or (task == 'adv' and not is_adv):
                continue
            key = (task,r['model'],r['method'],int(r['step']))
            if key in result:
                raise ValueError('Duplicate curve row {} in {}'.format(key,path))
            mean = float(r['one_minus_pcr_mean'] if is_adv else r['metric_mean'])
            sd = float(r['one_minus_pcr_std'] if is_adv else r['metric_std'])
            result[key] = (int(r['n_seeds']),mean,sd)
    return result


def run(args):
    summary = args.summary
    per_seed = summary/'per_seed.csv'
    if not per_seed.exists():
        raise FileNotFoundError('Missing {}; run summarize_rq1_three_methods.py first'.format(per_seed))
    rows = read_csv(per_seed)
    groups = defaultdict(dict)
    fingerprint_by_seed = defaultdict(dict)
    nplayers_by_config = defaultdict(dict)
    warnings = []
    failures = []
    observed = set()
    for row in rows:
        key = (row['task'],row['model'],row['method'],int(row['seed']))
        if key not in EXPECTED_KEYS:
            continue
        if key in observed:
            failures.append('DUPLICATE {}'.format(key))
            continue
        observed.add(key)
        try:
            series,fp,nplayers = read_raw(row)
            groups[key[:3]][key[3]] = series
            nplayers_by_config[key[:2]][key[2]] = nplayers
            if fp:
                fingerprint_by_seed[(key[0],key[1],key[3])][key[2]] = fp
        except Exception as exc:
            failures.append('INVALID {}: {}'.format(key,exc))
    missing=EXPECTED_KEYS-observed
    for k in sorted(missing):
        failures.append('MISSING {}'.format(k))
    for (task,model), m in sorted(nplayers_by_config.items()):
        if len(set(m.values())) != 1:
            failures.append('DIFFERENT_CANDIDATE_COUNTS {}/{}: {}'.format(task,model,m))
    for key,fps in sorted(fingerprint_by_seed.items()):
        if len(fps) != 3 or len(set(fps.values())) != 1:
            failures.append('FAIRNESS_SAMPLE_MISMATCH {}: {}'.format(key,fps))
    curves = reported_curves(summary)
    checked = 0
    report_rows = []
    for task,model,kmax in CONFIGS:
        for method in METHODS:
            key = (task,model,method)
            by_seed = groups.get(key,{})
            if set(by_seed) != set(SEEDS):
                failures.append('INCOMPLETE {}: {} instead of {}'.format(key,sorted(by_seed),list(SEEDS)))
                continue
            if any(len(v)!=kmax for v in by_seed.values()):
                failures.append('WRONG_STEP_COUNT {}: {}'.format(key,{s:len(v) for s,v in by_seed.items()}))
                continue
            if len({tuple(v) for v in by_seed.values()}) == 1:
                warnings.append('ZERO_VARIATION {}/{}/{}: identical full Top-k metric sequence over 5 seeds; this does NOT prove random sampling varied'.format(*key))
            for step in range(1,kmax+1):
                samples = [by_seed[s][step-1] for s in SEEDS]
                mean = statistics.mean(samples)
                sd = statistics.stdev(samples)
                upper,lower = min(1.,mean+sd),max(0.,mean-sd)
                curvekey=(task,model,method,step)
                old=curves.get(curvekey)
                if old is None:
                    failures.append('CURVE_MISSING {}'.format(curvekey))
                    continue
                n,old_mean,old_sd=old
                # Aggregation's non-ADV CSV uses ~8 significant digits.
                if n != 5 or not math.isclose(mean,old_mean,rel_tol=0,abs_tol=6e-8) or not math.isclose(sd,old_sd,rel_tol=0,abs_tol=6e-8):
                    failures.append('CURVE_MISMATCH {} raw(mu={:.10f},sd={:.10f}) csv(n={},mu={:.10f},sd={:.10f})'.format(curvekey,mean,sd,n,old_mean,old_sd))
                    continue
                checked += 1
                row=dict(task=task,model=model,method=method,step=step,n_seeds=5,
                         mean_pct='{:.6f}'.format(mean*100),std_pct='{:.6f}'.format(sd*100),
                         lower_pct='{:.6f}'.format(lower*100),upper_pct='{:.6f}'.format(upper*100))
                row.update({'seed_{}'.format(s):'{:.10f}'.format(by_seed[s][step-1]*100) for s in SEEDS})
                report_rows.append(row)
                if args.verbose:
                    print('{:8s} {:8s} {:9s} k={:2d}  {:8.4f} +/- {:7.4f}%  [{:8.4f}, {:8.4f}]'.format(task,model,method,step,mean*100,sd*100,lower*100,upper*100))
    total= sum(k for _,_,k in CONFIGS)*len(METHODS)
    if len(curves) != total:
        failures.append('UNEXPECTED_CURVE_ROWS: {} expected {}'.format(len(curves),total))
    # Warnings are limitations of inference, not calculation errors.
    warnings.append('NON_FAIRNESS_SAMPLE_IDENTITY: Cannot independently verify all methods used identical evaluation samples without shared sample fingerprints.')
    warnings.append('FIXED_SAMPLES: Five different seed labels alone do not prove five independent localization samples; see prior load_data fingerprint check.')
    cols=['task','model','method','step','n_seeds','mean_pct','std_pct','lower_pct','upper_pct']+['seed_{}'.format(s) for s in SEEDS]
    summary.mkdir(parents=True,exist_ok=True)
    outcsv=summary/'audit_every_k_10models.csv'
    outreport=summary/'audit_every_k_10models.txt'
    write_csv(outcsv,cols,report_rows)
    lines = ['RQ1 EVERY-K AUDIT',
             'Excluded: backdoor/mnist',
             'Adversarial metric: 1-PCR; other metrics: original ablation metric',
             '10 configs x 3 methods x 5 seeds = 150 input records',
             '{} / {} per-step comparisons passed'.format(checked,total),
             'FAILURES: {}'.format(len(failures)),
             *['FAIL: '+s for s in failures],
             'WARNINGS: {}'.format(len(warnings)),
             *['WARN: '+s for s in warnings],
             'Raw JSON and existing aggregate CSVs were READ ONLY; no localization or model inference performed.']
    outreport.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\nAUDIT: {} / {} Top-k points matched raw seed-wise JSON'.format(checked,total))
    print('Models: {} | Methods: {} | Seeds: {} | Input records: {}'.format(len(CONFIGS),len(METHODS),len(SEEDS),len(observed)))
    print('Errors: {} | Warnings: {}'.format(len(failures),len(warnings)))
    print('Per-step mean/SD/bounds:',outcsv)
    print('Audit details:',outreport)
    if failures:
        print('\nFirst errors:')
        for f in failures[:12]: print(' -',f)
        return 1
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,default=Path('result/rq1_three_method_summary'))
    p.add_argument('--verbose',action='store_true',help='Print each of 189 Top-k points')
    try:
        code=run(p.parse_args())
    except Exception as exc:
        print('AUDIT FAILED:',exc,file=sys.stderr)
        code=2
    sys.exit(code)
