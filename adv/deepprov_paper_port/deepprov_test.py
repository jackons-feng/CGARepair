#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-selection independent TEST for DeepProv Keras activation-action PORT."""
from __future__ import print_function
import argparse
import json
import sys
import time
from pathlib import Path
from deepprov_io import (root_from_script,adv_split,clean_split,
                         ensure,sha256,write_json)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=('mnist','cifar10'))
    p.add_argument('seed',type=int)
    p.add_argument('--epsilon',type=float,default=None, help='Default: MNIST 0.05, CIFAR-10 0.1')
    p.add_argument('--root',default=str(root_from_script()))
    p.add_argument('--run-dir',default=None)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.epsilon is None:
        args.epsilon = 0.05 if args.dataset == 'mnist' else 0.1
    root=Path(args.root).resolve()
    run=Path(args.run_dir).resolve() if args.run_dir else root/'adv'/'loc_result'/args.dataset/str(args.seed)/('deepprov_paper_port_eps{:g}'.format(args.epsilon))
    manifest_path=ensure(run/'train_manifest.json')
    report_path=run/'deepprov_port_test_results.json'
    if report_path.is_file() and not args.overwrite:
        print('[SKIP TEST] Exists:',report_path)
        return 0
    with open(str(manifest_path),'r',encoding='utf-8') as f:
        m=json.load(f)
    if (m.get('dataset') != args.dataset or m.get('seed') != args.seed or
            float(m.get('epsilon')) != float(args.epsilon) or m.get('train_selection_only') is not True):
        raise ValueError('TRAIN manifest identity/provenance mismatch')
    checkpoint=ensure(m['checkpoint'])
    original=ensure(m['model']['original_model'])
    if sha256(checkpoint)!=m['checkpoint_sha256']:
        raise ValueError('TRAIN-selected model bytes changed after selection')
    if sha256(original)!=m['model']['original_model_sha256']:
        raise ValueError('Original pretrained model changed after TRAIN selection')
    print('[FIXED CHECKPOINT]',checkpoint)
    print('[METHOD] Runtime activation-action port, no TEST selection')
    print('[SOURCE]',m['method'])
    import tensorflow as tf
    from deepprov_runtime import DeepProvPaperActOn, accuracy
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    t0=time.time()
    x_clean,y_clean=clean_split(args.dataset,'test',tf)
    x_adv,y_adv,adv_info=adv_split(root,args.dataset,'test',args.epsilon)
    if adv_info['source_sha256']==m['adv_train']['source_sha256']:
        raise RuntimeError('Exact adversarial TRAIN/TEST file hash collision; refuse to publish')
    if len(x_clean)!=len(y_clean) or len(x_adv)!=len(y_adv):
        raise ValueError('TEST data shapes mismatch')
    if len(x_clean) != len(y_clean) or len(x_adv) != len(y_adv):
        raise ValueError('TEST input/label count mismatch')
    if len(x_clean) != 10000 or len(x_adv) == 0:
        raise ValueError('Invalid TEST sample count')
    print('[TEST COUNTS] clean={} adv={}'.format(
        len(x_clean), len(x_adv)))
    print('[TEST]',adv_info['source_path'],'n=',len(x_adv))
    base=tf.keras.models.load_model(str(original),compile=False)
    repaired=tf.keras.models.load_model(str(checkpoint),compile=False,
                                         custom_objects={'DeepProvPaperActOn':DeepProvPaperActOn})
    before_clean=accuracy(base,(x_clean,y_clean),args.batch_size)
    before_adv=accuracy(base,(x_adv,y_adv),args.batch_size)
    after_clean=accuracy(repaired,(x_clean,y_clean),args.batch_size)
    after_adv=accuracy(repaired,(x_adv,y_adv),args.batch_size)
    result={'method':m['method'],'dataset':args.dataset,'seed':args.seed,'epsilon':args.epsilon,
        'train_status':m['status'],'is_full_test':True,'train_selected_fixed_checkpoint':True,
        'test_did_not_select_checkpoint':True,'train_model_hash_verified':True,
        'test_source':adv_info,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
        'before':{'clean_acc':before_clean,'adv_acc':before_adv,'adversarial_error_rate':1-before_adv},
        'after':{'clean_acc':after_clean,'adv_acc':after_adv,'adversarial_error_rate':1-after_adv},
        'delta_adv_acc_pp':100*(after_adv-before_adv),
        'delta_clean_acc_pp':100*(after_clean-before_clean),
        'elapsed_sec':time.time()-t0,
        'comparison_scope':"Keras runtime action port, NOT author's full IPG/GNN implementation"}
    write_json(report_path,result)
    print('[BEFORE] Clean={:.6f} AdvAcc={:.6f} AER={:.6f}'.format(before_clean,before_adv,1-before_adv))
    print('[AFTER]  Clean={:.6f} AdvAcc={:.6f} AER={:.6f}'.format(after_clean,after_adv,1-after_adv))
    print('[SAVED]',report_path)
    if m['status']!='TRAIN_FEASIBLE':
        print('[WARNING] TRAIN found no feasible action: results exploratory; cannot claim effective repair')
    return 0

if __name__=='__main__':
    sys.exit(main())
