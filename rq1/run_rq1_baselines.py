#!/usr/bin/env python3
"""Run the two RQ1 locator baselines for a task; only skip audited outputs."""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROTOCOL = 'rq1_baseline_multiseed_v2'
BASE = Path(__file__).resolve().parent
DEFAULT_SEEDS = [256, 42, 123, 1024, 2026]
SUITES = {
    'backdoor': [('mnist', 512, None, False), ('fashion', 512, None, False), ('gtsrb', 512, None, True)],
    'adv': [('mnist', 512, '0.05', False), ('cifar10', 512, '0.1', False)],
    'safety': [('a19', 512, None, False), ('a29', 512, None, False), ('a33', 512, None, False)],
    'fairness': [('bank', 4096, None, False), ('adult', 4096, None, False), ('default', 4096, None, False)],
}
METHODS = {'birdnn': ('loc_BIRDNN_test.py', 'BIRDNN'), 'ccbr': ('loc_ccbr_test.py', 'ccbr')}


def root_for(root, task, model, epsilon, resnet):
    sub = '{}_resnet'.format(model) if resnet else model
    p = root / task / sub
    if task == 'adv':
        p = p / ('epsilon_' + str(epsilon))
    return p


def metrics_from(data, k):
    seq = data.get('top{}_metric_results'.format(k), [])
    if not isinstance(seq, list):
        return None
    if seq and isinstance(seq[0], dict):
        seq = [r.get('metric_after_disable') for r in seq if r.get('status') == 'disabled']
    if len(seq) != k:
        return None
    try:
        vals = [float(x) for x in seq]
    except (TypeError, ValueError):
        return None
    return vals if all(math.isfinite(v) and -1e-8 <= v <= 1 + 1e-8 for v in vals) else None


def is_valid(path, method, task, model, n_samples, seed):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        n = int(data['num_players'])
        k = int(data['top_k'])
        expected_k = min(n, max(4, min(11, int(n * 0.01))))
        if not (n > 0 and k == expected_k):
            return False
        checks = (
            data.get('rq1_protocol_version') == PROTOCOL,
            data.get('method') == method,
            data.get('task') == task,
            data.get('model') == model,
            int(data.get('seed', -1)) == seed,
            int(data.get('requested_num_examples', -1)) == n_samples,
            int(data.get('actual_sample_count', -1)) == n_samples,
            int(re.search(r'top(\d+)_samples(\d+)_', path.name).group(1)) == k,
            int(re.search(r'top(\d+)_samples(\d+)_', path.name).group(2)) == n_samples,
            metrics_from(data, k) is not None,
        )
        if not all(checks):
            return False
        if task == 'fairness' and not data.get('fairness_sample_fingerprint'):
            return False
        selected = data.get('selected_topk_path')
        if selected and not Path(selected).exists():
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=SUITES)
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=['birdnn', 'ccbr'])
    ap.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--force', action='store_true', help='Rerun even valid results')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = Path('result')
    config = [x for x in SUITES[args.task] if args.models is None or x[0] in args.models]
    if not config:
        ap.error('No matching models in this task')
    if len(set(args.seeds)) != len(args.seeds):
        ap.error('Duplicate seeds')
    todo = len(config) * len(args.methods) * len(args.seeds)
    print('[RQ1 BASELINES] task={} methods={} seeds={} jobs={}'.format(args.task, args.methods, args.seeds, todo), flush=True)
    counter = 0
    for model, samples, epsilon, resnet in config:
        p = root_for(root, args.task, model, epsilon, resnet)
        for seed in args.seeds:
            for meth in args.methods:
                counter += 1
                script, tag = METHODS[meth]
                if not (Path(script)).is_file():
                    raise SystemExit('Missing script: {}. Unzip entire package beside load_data.py.'.format(script))
                candidates = sorted(p.glob('top*_samples{}_{}_seed{}.txt'.format(samples, tag, seed)))
                valid = next((x for x in candidates if is_valid(x, tag, args.task, model, samples, seed)), None)
                prefix = '[{}/{}] {} {} seed={}'.format(counter, todo, tag, model, seed)
                if valid and not args.force:
                    print(prefix + ' SKIP validated: ' + str(valid), flush=True)
                    continue
                cmd = [sys.executable, '-u', script, args.task, model, str(samples)]
                if args.task == 'adv':
                    cmd.append(str(epsilon))
                if resnet:
                    cmd.append('True')
                cmd += ['--seed', str(seed)]
                print(prefix + ' RUN: ' + ' '.join(cmd), flush=True)
                if args.dry_run:
                    continue
                # Preserve invalid old files before they can be overwritten.
                backup_dir = root / '_rq1_pre_multiseed_backup' / args.task / p.relative_to(root / args.task)
                for old in candidates:
                    if not is_valid(old, tag, args.task, model, samples, seed):
                        backup_dir.mkdir(parents=True, exist_ok=True)
                        dest = backup_dir / old.name
                        if dest.exists():
                            dest = backup_dir / (old.name + '.' + str(time.time_ns()) + '.bak')
                        shutil.copy2(old, dest)
                        print('  [BACKUP] {} -> {}'.format(old, dest), flush=True)
                logdir = root / 'rq1_baseline_logs' / args.task
                logdir.mkdir(parents=True, exist_ok=True)
                logfile = logdir / '{}_{}_seed{}.log'.format(meth, model, seed)
                with logfile.open('w', encoding='utf-8') as log:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    try:
                        for line in proc.stdout:
                            sys.stdout.write(line)
                            log.write(line)
                            log.flush()
                    except KeyboardInterrupt:
                        proc.terminate()
                        raise
                    rc = proc.wait()
                if rc:
                    raise SystemExit('{} FAILED (code={}). Log: {}'.format(prefix, rc, logfile))
                fresh = sorted(p.glob('top*_samples{}_{}_seed{}.txt'.format(samples, tag, seed)))
                if not any(is_valid(x, tag, args.task, model, samples, seed) for x in fresh):
                    raise SystemExit('{} produced no valid result; inspect {}'.format(prefix, logfile))
    print('[DONE] {}: all requested methods/seeds finished or skipped.'.format(args.task), flush=True)

if __name__ == '__main__':
    main()
