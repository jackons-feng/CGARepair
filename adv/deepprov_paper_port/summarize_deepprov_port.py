#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from deepprov_io import root_from_script
SEEDS=(42,123,256,1024,2026)
EPSILON_BY_DATASET={'mnist':0.05,'cifar10':0.1}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=str(root_from_script()))
    p.add_argument('--output',default=None)
    a=p.parse_args()
    root=Path(a.root).resolve()
    out=Path(a.output).resolve() if a.output else root/'adv'/'deepprov_paper_port_summary'
    out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for ds in ('mnist','cifar10'):
        for seed in SEEDS:
            eps=EPSILON_BY_DATASET[ds]
            path=root/'adv'/'loc_result'/ds/str(seed)/('deepprov_paper_port_eps{:g}'.format(eps))/'deepprov_port_test_results.json'
            if not path.exists():
                rows.append({'dataset':ds,'epsilon':eps,'seed':seed,'status':'MISSING','clean_acc':'','adv_acc':'','aer':'','json':str(path)})
                continue
            with path.open('r',encoding='utf-8') as f: obj=json.load(f)
            if obj['dataset']!=ds or int(obj['seed'])!=seed or float(obj['epsilon'])!=eps:
                raise RuntimeError('Result identity mismatch: '+str(path))
            rows.append({'dataset':ds,'epsilon':eps,'seed':seed,'status':obj['train_status'],
                         'clean_acc':obj['after']['clean_acc'],
                         'adv_acc':obj['after']['adv_acc'],
                         'aer':obj['after']['adversarial_error_rate'], 'json':str(path)})
    with (out/'per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','epsilon','seed','status','clean_acc','adv_acc','aer','json']);w.writeheader();w.writerows(rows)
    summaries=[]
    for ds in ('mnist','cifar10'):
        valid=[r for r in rows if r['dataset']==ds and r['status']=='TRAIN_FEASIBLE']
        present=[r for r in rows if r['dataset']==ds and r['status']!='MISSING']
        status='PAPER_READY_PORT' if len(valid)==5 else 'INCOMPLETE_OR_INFEASIBLE'
        d={'dataset':ds,'epsilon':EPSILON_BY_DATASET[ds],'n_test_results':len(present),'n_train_feasible':len(valid),'status':status}
        for key in ('clean_acc','adv_acc','aer'):
            if len(valid)==5:
                values=np.array([float(r[key]) for r in valid])
                d[key+'_mean_pct']=float(np.mean(values)*100)
                d[key+'_std_pct']=float(np.std(values,ddof=1)*100)
            else:
                d[key+'_mean_pct']=''
                d[key+'_std_pct']=''
        summaries.append(d)
        print(ds,status,'valid=',len(valid),'/5')
    with (out/'means.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0].keys()));w.writeheader();w.writerows(summaries)
    print('[SAVED]',out)

if __name__=='__main__': main()
