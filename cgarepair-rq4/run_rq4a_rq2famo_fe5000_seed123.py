#!/usr/bin/env python3
"""Paired CGARepair optimizer / evaluation-budget study.

No original files are edited. Each child process runs from the corresponding
native task directory, using original task modules and TRAIN/TEST protocols.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SEEDS=(123,)
TASKS={
    'adv': {'dir':'adv','model':'mnist','eps':'0.05', 'loc_samples':512,'pop_compare':100},
    'backdoor': {'dir':'backdoor','model':'fashion','loc_samples':512,'pop_compare':100},
    'safety': {'dir':'safety','model':'a33','loc_samples':512,'pop_compare':100},
    'fairness': {'dir':'fairness','model':'bank','loc_samples':4096,'pop_compare':100},
}
STUDY_DIR='optimizer_study_20261031'
CONFIG = str(Path(__file__).resolve().parent / 'config' / 'rq4a_rq2famo_fe5000_seed123.json')


def _resolve_release_paths(value, root):
    if isinstance(value, dict):
        return {k: _resolve_release_paths(v, root)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_release_paths(v, root) for v in value]
    if isinstance(value, str):
        prefixes = (
            "adv/", "backdoor/", "fairness/", "safety/",
            "rq1/", "model/", "cgarepair-rq4/"
        )
        if value.startswith(prefixes):
            return str((Path(root) / value).resolve())
    return value


def load_config(project):
    f=project/'cgarepair-rq4'/'config'/CONFIG
    if not f.is_file():
        raise FileNotFoundError('Installer did not create {}'.format(f))
    with open(str(f),encoding='utf-8') as h:
        data=json.load(h)
    if not isinstance(data,dict):
        raise ValueError('study_config.json must contain an object')
    return data

def jobs(study='both',task='all',seed=None,budget=None,include_moead_controls=False,optimizer='all'):
    out=[]
    tasknames=list(TASKS) if task=='all' else [task]
    runseeds=list(SEEDS) if seed is None else [int(seed)]
    for t in tasknames:
        c=TASKS[t]
        for s in runseeds:
            if study in ('both','optimizer'):
                algorithms=('nsga3','moead') if include_moead_controls else ('nsga3',)
                for alg in algorithms:
                    if optimizer not in ('all',alg): continue
                    b=5000
                    if budget is not None and budget!=b: continue
                    out.append({'study':'optimizer','task':t,'seed':s,'optimizer':alg,
                                'budget':b,'pop':c['pop_compare'],
                                'tag':'optim5000_p{}'.format(c['pop_compare'])})
            if study in ('both','budget') and t in ('adv','fairness'):
                bpop=100
                if optimizer not in ('all','moead'): continue
                for b in (2500,5000,5000):
                    if budget is not None and budget!=b: continue
                    out.append({'study':'budget','task':t,'seed':s,'optimizer':'moead',
                                'budget':b,'pop':bpop,
                                'tag':'budget{}_p{}'.format(b,bpop)})
    out.sort(key=lambda j:(j['study'],j['task'],j['optimizer'],j['budget'],j['seed']))
    return out

def work_paths(project,job,quick=False):
    task_dir=project/TASKS[job['task']]['dir']
    tag=job['tag']+('_smoke_quick' if quick else '')
    root=project/'cgarepair-rq4/results/optimizer_comparison_rq2famo_fe5000'
    ident='{}_{}_{}_{}_{}_{}'.format(job['study'],job['task'],job['optimizer'],job['budget'],job['pop'],job['seed'])
    if quick: ident+='_smoke'
    return task_dir,tag,root,root/'logs'/ident

def result_dir(job,tag,root):
    c=TASKS[job['task']]
    if job['task']=='adv':
        return root/'adv'/c['model']/('epsilon_'+c['eps'])/'CGARepair'/('seed_'+str(job['seed']))/(job['optimizer']+'_'+tag)
    if job['task']=='fairness':
        return root/c['model']/'CGARepair'/('seed_'+str(job['seed']))/(job['optimizer']+'_'+tag)
    return root/c['model']/'CGARepair'/('seed_'+str(job['seed']))/(job['optimizer']+'_'+tag)

def get_extra(cfg, task, phase):
    entries=cfg.get('task_extra_args',{}).get(task,{})
    vals=entries.get(phase,[])
    if not isinstance(vals,list) or any(not isinstance(x,(str,int,float)) for x in vals):
        raise ValueError('task_extra_args.{}.{} must be a JSON list of CLI values'.format(task,phase))
    return [str(v) for v in vals]

def cmd_for(project,job,phase,config,quick=False):
    c=TASKS[job['task']]
    task_dir,tag,root,log_dir=work_paths(project,job,quick)
    script=task_dir/STUDY_DIR/('repair_study.py' if phase=='repair' else 'test_study.py')
    cmd=[sys.executable,'-u',str(script)]
    if job['task']=='adv':cmd.extend([c['model'],c['eps'],str(job['seed'])])
    else:cmd.extend([c['model'],str(job['seed'])])
    cmd.extend(['--optimizer',job['optimizer'],'--run-tag',tag,'--output-root',str(root)])
    if phase=='repair':
        maxeval=(48 if job['task']=='fairness' else 36) if quick else job['budget']
        pop=12 if quick else job['pop']
        if maxeval%pop:
            raise ValueError('Budget {} not divisible by pop_size {}'.format(maxeval,pop))
        cmd.extend(['--loc-seed','256','--loc-samples',str(c['loc_samples']),
                    '--pop-size',str(pop),'--max-gen',str(maxeval//pop),
                    '--max-evals',str(maxeval)])
        if quick:cmd.append('--quick')
    else:
        if job['task']=='fairness': pass  # native fairness full test has no --run-dir
        else:cmd.extend(['--run-dir',str(result_dir(job,tag,root)),'--test-samples',str(0 if not quick else 100)])
    # Extra flags allow local path / protocol alignment without changing this package.
    cmd.extend(get_extra(config,job['task'],phase))
    return cmd,task_dir,log_dir

def verify_summary(job,summary,quick=False):
    reported_opt=summary.get('optimizer')
    if reported_opt!=job['optimizer']:
        raise ValueError('Optimizer mismatch {} != {}'.format(reported_opt,job['optimizer']))
    reported_seed=summary.get('optimization_seed',summary.get('seed',-1))
    if int(reported_seed)!=job['seed']:
        raise ValueError('Seed mismatch {} != {}'.format(reported_seed,job['seed']))
    expected=(48 if job['task']=='fairness' else 36) if quick else job['budget']
    if job['task']=='fairness':
        actual=summary.get('total_evaluations')
        pop=summary.get('population_size')
        requested=summary.get('eval_budget_effective')
    else:
        actual=summary.get('evaluations_actual')
        pop=summary.get('optimizer_parameters',{}).get('population_size')
        requested=summary.get('max_evaluations_requested')
    if actual is None or int(actual)!=expected:
        raise ValueError('Actual FE {} != expected {}'.format(actual,expected))
    if requested is not None and int(requested)!=expected:
        raise ValueError('Requested FE {} != expected {}'.format(requested,expected))
    if pop is not None and int(pop)!=(12 if quick else job['pop']):
        raise ValueError('Population size {} != expected {}'.format(pop,job['pop']))
    selected=summary.get('selected_players',summary.get('selected_neurons',[]))
    if not isinstance(selected,list) or not selected:
        raise ValueError('Repair summary missing exact selected neuron list')
    return True

def execute(cmd,cwd,log_file):
    log_file.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['PYTHONPATH']=str(cwd)+os.pathsep+env.get('PYTHONPATH','')
    with open(str(log_file),'w',encoding='utf-8') as f:
        f.write('COMMAND: '+subprocess.list2cmdline(cmd)+'\n')
        f.flush()
        proc=subprocess.Popen(cmd,cwd=str(cwd),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              universal_newlines=True,bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc=proc.wait()
    if rc:
        raise RuntimeError('Process exited with code {}. See {}'.format(rc,log_file))

def preflight(project,selected):
    try:
        import pymoo
        print('[ENV] pymoo version:',getattr(pymoo,'__version__','unknown'))
    except ImportError:
        print('[ENV] pymoo not importable: install in the ORIGINAL nsrepair conda env')
        return False
    ok=True
    for task in sorted(set(x['task'] for x in selected)):
        directory=project/TASKS[task]['dir']
        for needed in [directory/'read_CGARepair.py',directory/('cgarepair_'+task+'_utils.py'),
                       directory/STUDY_DIR/'repair_study.py',directory/STUDY_DIR/'test_study.py']:
            if not needed.is_file():
                print('[MISSING]',needed);ok=False
            else:print('[OK]',needed)
        stamp=directory/STUDY_DIR/'SOURCE_PROVENANCE.json'
        if stamp.is_file():
            import hashlib
            with open(str(stamp),encoding='utf-8') as f:provenance=json.load(f)
            native=directory/(provenance['repair_source'].split('(')[0]+'.py')
            if native.is_file():
                fingerprint=hashlib.sha256(native.read_bytes()).hexdigest()
                if fingerprint!=provenance['repair_sha256']:
                    print('[NOTICE] Native optimizer source differs from bundled source:',native)
                    print('  Verify that local updates to objectives, locator, and TRAIN selection are reflected.')
                else:print('[OK] source version matched:',native)
            else:print('[NOTICE] Original CLI source not found for version comparison:',native)
    return ok

def main():
    p=argparse.ArgumentParser(description='CGARepair optimizer and budget study (five seeds)')
    p.add_argument('--project-root',default=None,help='NSrepair folder; default parent of installed study folder')
    p.add_argument('--action',choices=('plan','preflight','repair','test','all','summarize'),default='plan')
    p.add_argument('--study',choices=('optimizer','budget','both'),default='both')
    p.add_argument('--task',choices=('all',)+tuple(TASKS),default='all')
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--budget',type=int,choices=(2500,5000,5000))
    p.add_argument('--optimizer',choices=('all','moead','nsga3'),default='all')
    p.add_argument('--include-moead-controls',action='store_true',help='Strict matched optimizer control; adds 20 MOEA/D runs')
    p.add_argument('--quick',action='store_true',help='Separate 36-FE quick smoke-test outputs; NOT paper results')
    p.add_argument('--limit',type=int,default=0,help='0=all matching jobs; 1=first job')
    p.add_argument('--force',action='store_true',help='Overwrite the selected study output with a new run')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args()
    project=Path(args.project_root).expanduser().resolve() if args.project_root else Path(__file__).resolve().parent.parent
    cfg=load_config(project)
    cfg = _resolve_release_paths(cfg, project)
    selected=jobs(args.study,args.task,args.seed,args.budget,args.include_moead_controls,args.optimizer)
    if args.limit>0:selected=selected[:args.limit]
    if not selected and args.action!='summarize':
        raise SystemExit('No jobs match these filters')
    if args.action=='summarize':
        from summarize_study import summarize
        summarize(project,quick=args.quick)
        return
    print('Project:',project,'Jobs:',len(selected),'Action:',args.action)
    if args.action=='preflight':
        if not preflight(project,selected):raise SystemExit(2)
        return
    for i,job in enumerate(selected,1):
        taskdir,tag,root,log=work_paths(project,job,args.quick)
        expected=result_dir(job,tag,root)
        print('[{}/{}] {}:{} seed={} optimizer={} FE={} pop={} out={}'.format(
            i,len(selected),job['study'],job['task'],job['seed'],job['optimizer'],(48 if job['task']=='fairness' else 36) if args.quick else job['budget'],
            12 if args.quick else job['pop'],expected))
        if args.action=='plan':
            for phase in ('repair','test'):
                command,cwd,_=cmd_for(project,job,phase,cfg,args.quick)
                print('  {}: (cd {}) {}'.format(phase,cwd,subprocess.list2cmdline(command)))
            continue
        try:
            if args.action in ('repair','all'):
                repair_json=expected/'summary.json'
                if repair_json.exists() and not args.force:
                    with open(str(repair_json),encoding='utf-8') as f:summary=json.load(f)
                    verify_summary(job,summary,args.quick)
                    print('  [SKIP] verified existing repair')
                else:
                    command,cwd,_=cmd_for(project,job,'repair',cfg,args.quick)
                    if args.force:command.append('--overwrite')
                    execute(command,cwd,log/'repair.log')
                    if not repair_json.is_file():raise RuntimeError('Missing '+str(repair_json))
                    with open(str(repair_json),encoding='utf-8') as f:summary=json.load(f)
                    verify_summary(job,summary,args.quick)
            if args.action in ('test','all'):
                repair_json=expected/'summary.json'
                if not repair_json.is_file():raise RuntimeError('Repair summary missing: '+str(repair_json))
                with open(str(repair_json),encoding='utf-8') as f:summary=json.load(f)
                verify_summary(job,summary,args.quick)
                test_json=expected/'test_results.json'
                if test_json.exists() and not args.force:
                    print('  [SKIP] existing TEST:',test_json)
                else:
                    command,cwd,_=cmd_for(project,job,'test',cfg,args.quick)
                    if args.force and job['task']!='fairness':command.append('--overwrite')
                    execute(command,cwd,log/'test.log')
                    if not test_json.is_file():raise RuntimeError('Missing '+str(test_json))
        except Exception as exc:
            print('[ERROR]',type(exc).__name__,exc,file=sys.stderr)
            if not args.continue_on_error:raise
    if args.action in ('test','all'):
        print('[DONE] TRAIN/TEST stage finished')
        print('[INFO] RQ4 statistics require separate audited aggregation')

if __name__=='__main__':
    main()
