#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepProv paper-mechanism Keras PORT: TRAIN-only candidate/selection.

This is NOT the authors' full PyTorch IPG/GNN artifact. The model action is
faithful to official mod_models.py; localization attribution is a gradient
proxy replacing the official pretrained IPG/GNN explainer. Paper table labels
should say 'DeepProv activation-intervention port', not 'original DeepProv'.
"""
from __future__ import print_function
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
from deepprov_math import summarize_per_channel, tradeoff_score
from deepprov_io import (root_from_script, clean_split, adv_split, hash_model,
                         sha256, write_json, ensure)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=('mnist','cifar10'))
    p.add_argument('seed',type=int)
    p.add_argument('--epsilon',type=float,default=None, help='Default: MNIST 0.05, CIFAR-10 0.1')
    p.add_argument('--root',default=str(root_from_script()))
    p.add_argument('--model-path',default=None)
    p.add_argument('--output-root',default=None)
    p.add_argument('--loc-sample',type=int,default=256)
    p.add_argument('--eval-sample',type=int,default=768)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--candidate-pool',type=int,default=32)
    p.add_argument('--per-layer-top',type=int,default=12)
    p.add_argument('--max-actions',type=int,default=24)
    p.add_argument('--max-clean-drop',type=float,default=0.03)
    p.add_argument('--alpha',type=float,default=1.0)
    p.add_argument('--beta-scale',type=float,default=1.0,
                   help='Scale empirical node beta; default paper-inspired 1')
    p.add_argument('--reference',choices=('mean','kde'),default='kde')
    p.add_argument('--kde-bandwidth',type=float,default=0.1)
    p.add_argument('--allow-zero-TS',action='store_true',
                   help='Only for diagnosis; paper uses strictly positive TS')
    p.add_argument('--overwrite',action='store_true')
    return p


def sample(x,y,n,seed):
    rng=np.random.RandomState(seed)
    if n<=0 or n>=len(y): return x,y
    ix=np.sort(rng.choice(len(y),size=n,replace=False))
    return x[ix],y[ix]


def set_action_state(model, actions, references, betas):
    """Update non-trainable masks on ONE existing graph (no graph explosion)."""
    by_layer={}
    for row in actions: by_layer.setdefault(row['layer_name'],[]).append(row)
    for name, ref in references.items():
        try:
            layer=model.get_layer('dp_act_on_'+name)
        except ValueError:
            # No selected candidate in this layer => no action node inserted.
            continue
        mask=np.zeros_like(ref,dtype=np.float32)
        reference=np.asarray(ref,dtype=np.float32).copy()
        beta=np.zeros_like(ref,dtype=np.float32)
        for action in by_layer.get(name,[]):
            k=int(action['neuron_idx'])
            mask[...,k]=1.0
            if action['category']=='nullify':
                reference[...,k]=0.0
                beta[...,k]=1e30
            else:
                beta[...,k]=betas[name][k]
        layer.set_weights([reference,mask,beta])


def main():
    args=parser().parse_args()
    if args.epsilon is None:
        args.epsilon = 0.05 if args.dataset == 'mnist' else 0.1
    if not (0<args.alpha<=1): raise ValueError('Official alpha range is (0,1]')
    if args.beta_scale<=0: raise ValueError('Beta scale must be positive')
    root=Path(args.root).resolve()
    model_path=Path(args.model_path).resolve() if args.model_path else root/'model'/(args.dataset+'_adv.h5')
    ensure(model_path) # NEVER silently fall back to clean pretrained model.
    result_root=Path(args.output_root).resolve() if args.output_root else root/'adv'/'loc_result'/args.dataset/str(args.seed)/('deepprov_paper_port_eps{:g}'.format(args.epsilon))
    manifest_path=result_root/'train_manifest.json'
    repaired_path=result_root/'repaired_model.h5'
    if manifest_path.exists() and not args.overwrite:
        print('[SKIP TRAIN] Existing manifest: {}'.format(manifest_path))
        return 0
    if result_root.exists() and any(result_root.iterdir()) and not args.overwrite:
        raise FileExistsError('Nonempty output directory; use --overwrite only if intentional: {}'.format(result_root))
    result_root.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    from deepprov_runtime import (list_layers,gather_activations_and_proxy,
                                  build_action_model,accuracy,model_predict)
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
    tf.compat.v1.set_random_seed(args.seed)
    np.random.seed(args.seed)
    t0=time.time()
    print('[METHOD] DeepProv runtime-activation Keras PORT (NOT full IPG/GNN)')
    print('[MODEL]',model_path)
    model=tf.keras.models.load_model(str(model_path),compile=False)
    print('[DATA] Loading TRAIN splits only for model selection')
    x_clean,y_clean=clean_split(args.dataset,'train',tf)
    x_adv,y_adv,adv_info=adv_split(root,args.dataset,'train',args.epsilon)
    if model.input_shape[1:] != x_clean.shape[1:] or model.input_shape[1:] != x_adv.shape[1:]:
        raise ValueError('Model input mismatch. {} vs {} vs {}'.format(model.input_shape,x_clean.shape,x_adv.shape))
    loc_b=sample(x_clean,y_clean,args.loc_sample,args.seed+11)
    loc_a=sample(x_adv,y_adv,args.loc_sample,args.seed+12)
    eval_b=sample(x_clean,y_clean,args.eval_sample,args.seed+201)
    eval_a=sample(x_adv,y_adv,args.eval_sample,args.seed+202)
    layers=list_layers(model)
    if not layers: raise RuntimeError('No supported hidden Dense/Conv2D layers found')
    print('[LOCALIZATION] layers={}'.format(layers))
    acts_b,attr_b=gather_activations_and_proxy(model,*loc_b,layer_names=layers,batch_size=args.batch_size)
    acts_a,attr_a=gather_activations_and_proxy(model,*loc_a,layer_names=layers,batch_size=args.batch_size)
    candidates=[]
    refs={}
    betas={}
    for name in layers:
        cs, reference, beta=summarize_per_channel(acts_b[name],acts_a[name],
                        attr_b[name],attr_a[name],seed=args.seed,reference_mode=args.reference,
                        bandwidth=args.kde_bandwidth)
        for row in cs: row['layer_name']=name
        refs[name]=reference
        betas[name]=beta*args.beta_scale
        candidates.extend(cs[:args.per_layer_top])
    candidates.sort(key=lambda r:({'nullify':0,'prioritized':1,'regular':2}[r['category']],-r['score'],r['layer_name'],r['neuron_idx']))
    candidates=candidates[:args.candidate_pool]
    print('[CANDIDATES]',len(candidates),'from',len(layers),'layers')
    if not candidates:
        write_json(result_root/'diagnostic_no_candidates.json',{'reason':'no_candidate','model':str(model_path)})
        raise RuntimeError('No localization candidates; cannot claim successful reproduction')
    # Build activation-intervention graph ONCE, then update its masks for trials.
    graph=build_action_model(model,candidates,refs,betas,alpha=args.alpha)
    set_action_state(graph,[],refs,betas)
    base_clean=accuracy(model,eval_b,args.batch_size)
    base_adv=accuracy(model,eval_a,args.batch_size)
    identity_clean=accuracy(graph,eval_b,args.batch_size)
    identity_adv=accuracy(graph,eval_a,args.batch_size)
    probe_x=eval_b[0][:min(64,len(eval_b[0]))]
    max_output_delta=float(np.max(np.abs(model_predict(model,probe_x,args.batch_size) -
                                          model_predict(graph,probe_x,args.batch_size))))
    if (abs(base_clean-identity_clean)>1e-10 or
            abs(base_adv-identity_adv)>1e-10 or max_output_delta>1e-5):
        raise RuntimeError('ZERO-MASK PARITY FAILED: max output delta={:.8g}; refuse results'.format(max_output_delta))
    print('[ZERO-MASK PARITY] PASS clean={:.6f} adv={:.6f} max_output_delta={:.2e}'.format(
          base_clean,base_adv,max_output_delta))
    clean_floor=max(0.0,base_clean-args.max_clean_drop)
    ranking=[]
    for i,cand in enumerate(candidates):
        set_action_state(graph,[cand],refs,betas)
        ca=accuracy(graph,eval_b,args.batch_size)
        aa=accuracy(graph,eval_a,args.batch_size)
        ts=tradeoff_score(base_clean,base_adv,ca,aa)
        ranking.append({'candidate':cand,'single_clean':ca,'single_adv':aa,'single_TS':ts})
        if (i+1)%8==0: print('[SINGLE] {}/{} checked'.format(i+1,len(candidates)))
    ranking.sort(key=lambda r:(-r['single_TS'],-r['single_adv'],r['candidate']['layer_name']))
    accepted=[]
    history=[]
    prev_clean,prev_adv=base_clean,base_adv
    prev_ts=0.0
    for item in ranking:
        trial=accepted+[item['candidate']]
        set_action_state(graph,trial,refs,betas)
        ca=accuracy(graph,eval_b,args.batch_size)
        aa=accuracy(graph,eval_a,args.batch_size)
        ts=tradeoff_score(base_clean,base_adv,ca,aa)
        ok=(ca>=clean_floor and aa>=prev_adv-1e-9 and (ts>prev_ts+1e-9 or (args.allow_zero_TS and ts>=prev_ts-1e-9)))
        history.append({'candidate':item['candidate'],'clean':ca,'adv':aa,'TS':ts,'accepted':bool(ok)})
        if ok:
            accepted=trial
            prev_clean,prev_adv,prev_ts=ca,aa,ts
            print('[ACCEPT] count={} clean={:.6f} adv={:.6f} TS={:.6f}'.format(len(accepted),ca,aa,ts))
        if len(accepted)>=args.max_actions: break
    # Group search on TRAIN only, not the paper's exact multi-iteration algorithm.
    # It can find synergistic action sets when one node alone changes no label.
    group_history=[]
    if not accepted and not args.allow_zero_TS:
        ordered=[r['candidate'] for r in ranking]
        for k in (2,4,8,16,24):
            k=min(k,len(ordered),args.max_actions)
            if k<=0: continue
            trial=ordered[:k]
            set_action_state(graph,trial,refs,betas)
            ca=accuracy(graph,eval_b,args.batch_size)
            aa=accuracy(graph,eval_a,args.batch_size)
            ts=tradeoff_score(base_clean,base_adv,ca,aa)
            feasible=(ca>=clean_floor and aa>=base_adv and ts>1e-9)
            group_history.append({'k':k,'clean':ca,'adv':aa,'TS':ts,'feasible':bool(feasible)})
            if feasible and (not accepted or ts>prev_ts):
                accepted=trial
                prev_clean,prev_adv,prev_ts=ca,aa,ts
    set_action_state(graph,accepted,refs,betas)
    status='TRAIN_FEASIBLE' if accepted and prev_ts>0 else 'NO_TRAIN_FEASIBLE_ACTION'
    if status=='TRAIN_FEASIBLE':
        print('[TRAIN SELECTED] {} actions, delta AdvAcc={:.6f}, TS={:.6f}'.format(len(accepted),prev_adv-base_adv,prev_ts))
    else:
        print('[NO TRAIN FEASIBLE ACTION] preserving original behavior; do not report repair success')
    # Save complete runtime model, not `weights_final_*.h5`.
    graph.save(str(repaired_path),include_optimizer=False)
    manifest={'method':'DeepProv_paper_mechanism_Keras_port_not_official_GNN',
       'official_source':'https://github.com/um-dsp/DeepProv',
       'dataset':args.dataset,'seed':int(args.seed),'epsilon':args.epsilon,
       'status':status, 'alpha':args.alpha,'reference_mode':args.reference,
       'beta_scale':args.beta_scale,'model':hash_model(model_path),
       'adv_train':adv_info, 'train_selection_only':True,
       'loc_n_benign':len(loc_b[1]),'loc_n_adv':len(loc_a[1]),
       'eval_n_benign':len(eval_b[1]),'eval_n_adv':len(eval_a[1]),
       'original_train_clean':base_clean,'original_train_adv':base_adv,
       'repaired_train_clean':prev_clean,'repaired_train_adv':prev_adv,
       'train_TS':prev_ts,'clean_floor':clean_floor,
       'candidate_count':len(candidates),'selected_count':len(accepted),
       'selected_actions':accepted, 'single_ranking':ranking,
       'incremental_history':history,'group_diagnostics':group_history,
       'checkpoint':str(repaired_path.resolve()),'checkpoint_sha256':sha256(repaired_path),
       'elapsed_sec':time.time()-t0,
       'limitations':['Official graph IPG-GNN attribution replaced by gradient-times-activation proxy',
                     'Convolution reference uses channel-pooled KDE',
                     'Beta uses sampled clean/adv activations without proven image-level pairing',
                     'No TEST access during TRAIN selection']}
    write_json(manifest_path,manifest)
    print('[SAVED]',repaired_path)
    print('[SAVED]',manifest_path)
    return 0

if __name__=='__main__':
    sys.exit(main())
