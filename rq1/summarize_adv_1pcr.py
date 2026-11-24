#!/usr/bin/env python3
"""Summarize Top-K 1-PCR for the 3 RQ1 localization methods from original per-seed JSONs.

Run at the cfrepair_RQ1 experiment root AFTER summarize_rq1_three_methods.py:
    python summarize_adv_1pcr.py

The old summary ZIP alone does not contain PCR values, so raw result/adv/... JSONs
must still be present in the current directory.
"""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('result/rq1_three_method_summary')
SRC = ROOT / 'per_seed.csv'
METHOD_ORDER = ('CGARepair', 'BIRDNN', 'ccbr')
MODEL_ORDER = ('mnist', 'cifar10')
EXPECTED_SEEDS = (256, 42, 123, 1024, 2026)


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=columns)
        wr.writeheader()
        wr.writerows(rows)


def check_series(record):
    model = record['model']
    method = record['method']
    seed = int(record['seed'])
    k = int(record['k'])
    fn = Path(record['json_path'])
    if not fn.is_file():
        raise FileNotFoundError(str(fn))
    data = json.loads(fn.read_text(encoding='utf-8'))
    if data.get('task') != 'adv' or data.get('model') != model or data.get('method') != method:
        raise ValueError('task/model/method mismatch')
    if int(data.get('seed', -1)) != seed or int(data.get('top_k', -1)) != k:
        raise ValueError('seed/top_k mismatch')
    field = 'top{}_one_minus_pcr_results'.format(k)
    values = data.get(field)
    if not isinstance(values, list) or len(values) != k:
        raise ValueError('missing/invalid {}, got length {}'.format(field, len(values) if isinstance(values, list) else 'N/A'))
    vals = [float(v) for v in values]
    if not all(math.isfinite(v) and -1e-8 <= v <= 1+1e-8 for v in vals):
        raise ValueError('1-PCR outside [0,1] or undefined')
    return vals


def avg_sd(vals):
    return statistics.mean(vals), statistics.stdev(vals) if len(vals)>1 else 0.


def main():
    if not SRC.exists():
        raise SystemExit('Missing {}; run python summarize_rq1_three_methods.py first'.format(SRC))
    raw = list(csv.DictReader(SRC.open(encoding='utf-8-sig', newline='')))
    per = []
    curves = []
    summary = []
    issues = []
    group = defaultdict(list)
    expected_total = 2 * 3 * len(EXPECTED_SEEDS)
    for row in raw:
        if row['task'] != 'adv':
            continue
        ds, method, seed = row['model'], row['method'], int(row['seed'])
        if ds not in MODEL_ORDER or method not in METHOD_ORDER or seed not in EXPECTED_SEEDS:
            issues.append('Unexpected series: {}/{}/{}'.format(ds,method,seed))
            continue
        try:
            vals = check_series(row)
        except Exception as ex:
            issues.append('INVALID {}/{}/seed{}: {}'.format(ds,method,seed,ex))
            continue
        group[(ds,method)].append((seed,vals))
        per.append({'model': ds, 'method': method, 'seed': seed, 'k': len(vals),
                    'topk_1_pcr': '{:.10f}'.format(vals[-1]),
                    'topk_pcr': '{:.10f}'.format(1-vals[-1]),
                    'json_path': row['json_path']})
    for ds in MODEL_ORDER:
        for method in METHOD_ORDER:
            series = sorted(group.get((ds,method),[]))
            seeds = [s for s,_ in series]
            if seeds != sorted(EXPECTED_SEEDS):
                issues.append('INCOMPLETE {}/{}: seeds={}, required={}'.format(ds,method,seeds,list(EXPECTED_SEEDS)))
                continue
            sizes = {len(x) for _,x in series}
            if len(sizes) != 1:
                issues.append('INCONSISTENT_TOPK {}/{}: {}'.format(ds,method,sizes))
                continue
            k = sizes.pop()
            vals = [v[-1] for _,v in series]
            mu,sd=avg_sd(vals)
            summary.append({'model':ds,'method':method,'n_seeds':len(series),'k':k,
                            'topk_1_pcr_mean':'{:.10f}'.format(mu),
                            'topk_1_pcr_std':'{:.10f}'.format(sd),
                            'topk_pcr_mean':'{:.10f}'.format(1-mu),
                            'topk_pcr_std':'{:.10f}'.format(sd)})
            for i in range(k):
                step=[v[i] for _,v in series]
                m,s=avg_sd(step)
                curves.append({'model':ds,'method':method,'step':i+1,'n_seeds':len(series),
                               'one_minus_pcr_mean':'{:.10f}'.format(m),
                               'one_minus_pcr_std':'{:.10f}'.format(s)})
    write_csv(ROOT/'adv_1pcr_per_seed.csv',
              ['model','method','seed','k','topk_1_pcr','topk_pcr','json_path'],per)
    write_csv(ROOT/'adv_1pcr_summary.csv',
              ['model','method','n_seeds','k','topk_1_pcr_mean','topk_1_pcr_std','topk_pcr_mean','topk_pcr_std'],summary)
    write_csv(ROOT/'adv_1pcr_curves.csv',
              ['model','method','step','n_seeds','one_minus_pcr_mean','one_minus_pcr_std'],curves)
    (ROOT/'adv_1pcr_audit.txt').write_text('\n'.join(issues or ['OK: all expected 5-seed 1-PCR series present'])+'\n', encoding='utf-8')
    print('Valid per-seed 1-PCR records: {} / {}'.format(len(per),expected_total))
    print('Model       Method       Top-K  1-PCR (mean +/- SD, %)   PCR (mean, %)')
    for r in summary:
        print('{:<11} {:<12} {:>4}   {:>6.2f} +/- {:<6.2f}       {:>6.2f}'.format(
            r['model'],r['method'],r['k'],100*float(r['topk_1_pcr_mean']),
            100*float(r['topk_1_pcr_std']),100*float(r['topk_pcr_mean'])))
    print('Audit entries:',len(issues))
    if issues:
        print('\n'.join(issues))
    print('Saved:',ROOT/'adv_1pcr_summary.csv')


if __name__ == '__main__':
    main()
