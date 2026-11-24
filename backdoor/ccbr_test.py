#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent TEST evaluator for CCBR activation-scaling NSGA-III repair.

Aligned with the provided backdoor_moo_1.py (NOT compare_algorithms.py):

  loc_result/<fashion|mnist|gtsrb|gtsrb_resnet>/<seed>/nsga3/
      pareto_front.pkl                 # TRAIN objectives, solution vectors
      final_results.txt                # TRAIN-selected best valid solution index
      pareto_front/repair_weights_N.npy # activation scaling vector, NOT .h5 weights

  loc_result/<tag>/ccbr_players.txt
  loc_result/<tag>/ccbr_backdoor_vals.txt
  ../model/<tag>.h5
  ../data/img/<dataset>_visualize_{mask|pattern}_label_<target>.png

Examples (run from the same directory as backdoor_moo_1.py):
  python -u ccbr_test.py fashion 42 --evasion
  python -u ccbr_test.py gtsrb_resnet 42 --evasion
  python -u ccbr_test.py gtsrb 42 True --test-file ../data/gtsrb/test_data.pkl --evasion

TEST is never used to choose a solution. Primary SR reproduces the CCBR
TRAIN convention (successes on non-target inputs / ALL inputs). Standard
conditional ASR (successes / non-target inputs) is also reported.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

DATASET_CONFIG = {
    'mnist':   {'target_label': 3, 'input_shape': (28, 28, 1), 'threshold_acc': 0.9858, 'threshold_sr': 0.01},
    'fashion': {'target_label': 3, 'input_shape': (28, 28, 1), 'threshold_acc': 0.85,   'threshold_sr': 0.01},
    'gtsrb':   {'target_label': 33,'input_shape': (32, 32, 3), 'threshold_acc': 0.92,   'threshold_sr': 0.01},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='TEST saved CCBR NSGA-III activation scaling on hold-out data')
    p.add_argument('model_name', help='fashion / gtsrb_resnet / gtsrb / mnist')
    p.add_argument('seed', type=int, help='Seed used for the matching CCBR repair run')
    p.add_argument('legacy_resnet', nargs='?', default='', help='Optional positional True for GTSRB ResNet')
    p.add_argument('--resnet', action='store_true', help='Use gtsrb_resnet.h5 and loc_result/gtsrb_resnet')
    p.add_argument('--optimizer', '--algorithm', default='nsga3', choices=('nsga3',),
                   help='CCBR backdoor_moo_1.py implements nsga3 (not nsga2)')
    p.add_argument('--output-root', default='loc_result')
    p.add_argument('--run-dir', help='Explicit folder containing pareto_front.pkl')
    p.add_argument('--model-root', default='../model')
    p.add_argument('--model-path', help='Optional direct original poisoned .h5 model')
    p.add_argument('--data-root', default='../data')
    p.add_argument('--test-file', help='Hold-out GTSRB TEST .pkl/.npz file; NEVER train_data.pkl')
    p.add_argument('--img-dir', help='Trigger PNG folder (default: <data-root>/img)')
    p.add_argument('--loc-dir', help='Folder containing ccbr_players.txt and ccbr_backdoor_vals.txt')
    p.add_argument('--solution-index', type=int,
                   help='Explicit index fixed BEFORE TEST; default is original TRAIN best-valid selection')
    p.add_argument('--checkpoint', help='Explicit preselected .npy activation scaling vector (not a model .h5)')
    p.add_argument('--test-samples', type=int, default=0,
                   help='0=full hold-out TEST; positive sample count for a quick smoke test')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--evasion', action='store_true', help='Evaluate rotation/compression/noise/brightness on triggered TEST')
    p.add_argument('--evasion-samples', type=int, default=200)
    p.add_argument('--evaluate-front', action='store_true',
                   help='Evaluate each saved solution for ANALYSIS only; never use TEST to reselect')
    p.add_argument('--out-dir', help='Save JSON and CSV here; default: CCBR repair run directory')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    if args.test_samples < 0 or args.evasion_samples <= 0 or args.batch_size <= 0:
        p.error('--test-samples must be nonnegative; batch/evasion sample sizes must be positive')
    if args.solution_index is not None and args.solution_index < 0:
        p.error('--solution-index must be nonnegative')
    if args.solution_index is not None and args.checkpoint:
        p.error('Pass either --solution-index or --checkpoint, not both')
    return args


def model_identity(args):
    m = args.model_name.lower().strip().replace('-', '_')
    resnet = args.resnet or str(args.legacy_resnet).lower() in ('true','1','yes','y')
    if m in ('gtsrb_resnet', 'gtsrbresnet'):
        m, resnet = 'gtsrb', True
    if m not in DATASET_CONFIG:
        raise ValueError('Expected fashion, gtsrb_resnet, gtsrb, or mnist; got %r' % m)
    if resnet and m != 'gtsrb':
        raise ValueError('--resnet applies only to GTSRB')
    return m, 'gtsrb_resnet' if resnet else m


def read_train_front(run_dir):
    path = run_dir / 'pareto_front.pkl'
    if not path.is_file():
        raise FileNotFoundError('CCBR TRAIN Pareto file missing: %s' % path)
    with path.open('rb') as f:
        record = pickle.load(f)  # Read only trusted, locally generated training files.
    if not isinstance(record, dict) or 'X' not in record or 'F' not in record:
        raise ValueError('Invalid CCBR Pareto pkl: expected X and F')
    x = np.asarray(record['X'], dtype=np.float64)
    f = np.asarray(record['F'], dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if f.ndim == 1:
        f = f.reshape(1, -1)
    if x.ndim != 2 or f.ndim != 2 or f.shape[1] != 2 or len(x) != len(f):
        raise ValueError('Unexpected Pareto X/F shapes: %s %s' % (x.shape, f.shape))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(f)):
        raise ValueError('Nonfinite X or F in TRAIN Pareto pkl')
    return record, x, f


def train_selected_index(run_dir, record, f, model_name):
    """Exactly reproduce backdoor_moo_1.py: among feasible, max TRAIN Acc."""
    config = DATASET_CONFIG[model_name]
    threshold_acc = float(record.get('threshold_acc', config['threshold_acc']))
    threshold_sr = float(record.get('threshold_sr', config['threshold_sr']))
    train_sr, train_acc = f[:, 0], -f[:, 1]
    feasible = np.where((train_sr <= threshold_sr) & (train_acc >= threshold_acc))[0]
    if len(feasible) == 0:
        raise ValueError('No TRAIN-feasible CCBR solution. The repair script did NOT select a best-valid '
                         'solution. Specify a TRAIN-preselected --solution-index explicitly if needed; '
                         'do not choose from TEST scores.')
    index = int(feasible[np.argmax(train_acc[feasible])])
    final_file = run_dir / 'final_results.txt'
    if final_file.is_file():
        match = re.search(r'Best solution index:\s*(\d+)', final_file.read_text(encoding='utf-8'))
        if match and int(match.group(1)) != index:
            raise ValueError('TRAIN selection conflict: final_results.txt selects %s but Pareto pkl selects %s. '
                             'Check that files are from the same run.' % (match.group(1), index))
    return index, {'source': 'TRAIN Pareto F + original CCBR best-valid rule',
                   'policy': 'TRAIN feasible (SR <= threshold_sr and ACC >= threshold_acc), then max TRAIN ACC',
                   'train_feasible_count': int(len(feasible)),
                   'train_selection_verified': True,
                   'threshold_acc': threshold_acc, 'threshold_sr': threshold_sr}


def selected_scaling(run_dir, args, record, x, f, model_name):
    if args.checkpoint:
        file = Path(args.checkpoint).expanduser().resolve()
        selection = {'source': 'explicit preselected .npy', 'policy': 'fixed by caller before TEST',
                     'train_selection_verified': False, 'train_solution_index': None}
        idx = None
    else:
        if args.solution_index is not None:
            idx = int(args.solution_index)
            selection = {'source': 'explicit TRAIN solution index',
                         'policy': 'fixed by caller before TEST, not independently verified as best-valid',
                         'train_selection_verified': False}
        else:
            idx, selection = train_selected_index(run_dir, record, f, model_name)
        if idx >= len(x):
            raise ValueError('Solution index %d >= length of TRAIN Pareto front %d' % (idx, len(x)))
        file = run_dir / 'pareto_front' / ('repair_weights_%d.npy' % idx)
        selection['train_solution_index'] = int(idx)
        selection['train_sr'] = float(f[idx, 0])
        selection['train_accuracy'] = float(-f[idx, 1])
    if not file.is_file():
        raise FileNotFoundError('CCBR activation scaling vector missing: %s' % file)
    if file.suffix.lower() != '.npy':
        raise ValueError('CCBR saves activation scalars as .npy, not .h5: %s' % file)
    coeff = np.asarray(np.load(str(file), allow_pickle=False), dtype=np.float64).reshape(-1)
    if len(coeff) < 1 or not np.all(np.isfinite(coeff)):
        raise ValueError('Invalid CCBR activation scaling vector: %s' % file)
    if idx is not None and not np.allclose(coeff, x[idx], rtol=1e-6, atol=1e-7):
        raise ValueError('Stored .npy vector is inconsistent with TRAIN Pareto X[%d]' % idx)
    return coeff.astype(np.float32), file.resolve(), selection


def read_repair_indices(loc_dir, model):
    """Exactly reconstruct prepare_repair_data(): top min(11, ceil(1%))."""
    player_file = loc_dir / 'ccbr_players.txt'
    val_file = loc_dir / 'ccbr_backdoor_vals.txt'
    if not player_file.is_file() or not val_file.is_file():
        raise FileNotFoundError('Missing CCBR locator files: %s and %s' % (player_file, val_file))
    players = [v.strip() for v in player_file.read_text(encoding='utf-8').split(',') if v.strip()]
    raw_vals = [v.strip() for v in val_file.read_text(encoding='utf-8').split(',') if v.strip()]
    if len(players) != len(raw_vals) or not players:
        raise ValueError('CCBR player/score count mismatch or empty: %d vs %d' % (len(players), len(raw_vals)))
    values = list(map(float, raw_vals))
    dic = dict(zip(players, values))
    # Source uses a dict; insertion order handles duplicated player names the same way.
    repair_num = min(11, math.ceil(len(players) * 0.01))
    selected = [name for name, _ in sorted(dic.items(), key=lambda p: p[1], reverse=True)[:repair_num]]
    rep_indices = []
    for player in selected:
        parts = player.split('_')
        if len(parts) < 3:
            raise ValueError('Invalid CCBR player name: %s' % player)
        layer_idx, neuron_idx = int(parts[1]), int(parts[-1])
        if layer_idx >= len(model.layers):
            print('[WARNING] CCBR source ignores out-of-range layer: %s' % player)
            continue
        rep_indices.append((layer_idx, neuron_idx))
    if not rep_indices:
        raise ValueError('All selected CCBR neurons are invalid; no repair possible')
    return selected, rep_indices, {'player_file': str(player_file.resolve()),
                                   'backdoor_vals_file': str(val_file.resolve()),
                                   'repair_num': repair_num}


def setup_tf(seed):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    try:
        import tensorflow.python.keras.backend as K
        K.set_session(sess)
    except (ImportError, AttributeError):
        try:
            tf.compat.v1.keras.backend.set_session(sess)
        except AttributeError:
            pass
    np.random.seed(int(seed))
    tf.compat.v1.set_random_seed(int(seed))
    return tf, sess


def normalize_labels(labels):
    y = np.asarray(labels)
    return (np.argmax(y, axis=1) if y.ndim == 2 and y.shape[-1] > 1 else y.reshape(-1)).astype(np.int64)


def load_gtsrb_test(path, tf, sess):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('GTSRB TEST split not found: %s' % path)
    if 'train' in path.name.lower():
        raise ValueError('Independent TEST must not be train_data.pkl: %s' % path)
    if path.suffix.lower() in ('.pkl', '.pickle'):
        with path.open('rb') as f:
            data = pickle.load(f)
    elif path.suffix.lower() == '.npz':
        with np.load(str(path), allow_pickle=False) as z:
            data = {k: np.asarray(z[k]) for k in z.files}
    else:
        raise ValueError('GTSRB TEST file needs .pkl or .npz')
    if not isinstance(data, dict):
        raise ValueError('GTSRB TEST data expected a dict')
    xx = next((data[k] for k in ('data','x','X') if k in data), None)
    yy = next((data[k] for k in ('label','labels','y','Y') if k in data), None)
    if xx is None or yy is None:
        raise ValueError('GTSRB TEST pkl must contain data and label keys')
    xx = np.asarray(xx)
    yy = normalize_labels(yy)
    if xx.ndim != 4 or len(xx) != len(yy):
        raise ValueError('GTSRB TEST image/label shape mismatch')
    if xx.shape[1] == 3 and xx.shape[-1] != 3:
        xx = np.transpose(xx, (0,2,3,1))
    if xx.shape[-1] != 3:
        raise ValueError('GTSRB expected RGB 3-channel input')
    # Training code does (CHW->HWC) -> tf.image.resize(32,32) -> multiply 255.
    if float(np.min(xx)) < -1e-3 or float(np.max(xx)) > 1.01:
        raise ValueError('Expected GTSRB TEST PKL images in [0,1] as in CCBR TRAIN preprocessing; '
                         'got range %.4f ... %.4f' % (np.min(xx), np.max(xx)))
    ph = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, 3))
    op = tf.image.resize(ph, (32,32))
    result = []
    for begin in range(0, len(xx), 1000):
        batch = xx[begin:begin+1000].astype(np.float32)
        resized = sess.run(op, feed_dict={ph: batch})
        result.append((resized * 255.0).astype(np.float32))
    return np.concatenate(result, axis=0), yy


def load_test_data(model_name, args, tf, sess):
    if model_name in ('mnist','fashion'):
        provider = tf.keras.datasets.mnist if model_name == 'mnist' else tf.keras.datasets.fashion_mnist
        (_, _), (x,y) = provider.load_data()
        x = np.expand_dims(x.astype(np.float32)/255.0, -1)
        return x, normalize_labels(y), 'keras:%s/TEST' % model_name
    if args.test_file:
        path = Path(args.test_file)
    else:
        root = Path(args.data_root)
        candidates = [root/'gtsrb'/name for name in
                      ('test_data.pkl','test_data.npz','test.pkl','test.npz')]
        present = [v for v in candidates if v.is_file()]
        if not present:
            raise FileNotFoundError('No GTSRB hold-out TEST data found under %s. '
                                    'Use --test-file /path/to/test_data.pkl' % (root/'gtsrb'))
        if len(present) > 1:
            raise ValueError('Multiple GTSRB TEST files: use --test-file to disambiguate: %s' % present)
        path = present[0]
    x,y = load_gtsrb_test(path,tf,sess)
    return x,y,str(path.resolve())


def trigger_images(model_name, clean, args, tf):
    from tensorflow.keras.preprocessing import image
    target = DATASET_CONFIG[model_name]['target_label']
    h,w,c = DATASET_CONFIG[model_name]['input_shape']
    imgdir = Path(args.img_dir or (Path(args.data_root)/'img')).expanduser().resolve()
    maskfile = imgdir / ('%s_visualize_mask_label_%d.png' % (model_name,target))
    pattfile = imgdir / ('%s_visualize_pattern_label_%d.png' % (model_name,target))
    for path in (maskfile,pattfile):
        if not path.is_file():
            raise FileNotFoundError('Missing original CCBR backdoor trigger PNG: %s' % path)
    mask = image.img_to_array(image.load_img(str(maskfile), color_mode='grayscale', target_size=(h,w)))[:,:,0] / 255.0
    patt = image.img_to_array(image.load_img(str(pattfile), color_mode='rgb', target_size=(h,w)))
    if c == 1:
        patt = patt[:,:,1:2] / 255.0  # Exactly CCBR's GREEN channel for grayscale datasets.
    # GTSRB pattern stays in [0,255]; MNIST/Fashion pattern is in [0,1].
    poisoned = clean.astype(np.float32) * (1.0-mask[None,:,:,None]) + patt[None] * mask[None,:,:,None]
    return poisoned.astype(np.float32), {'mask':str(maskfile),'pattern':str(pattfile)}


class CCBRActivationEvaluator:
    """Reproduce CCBR problem's split_model + activation scaling evaluation."""
    def __init__(self, tf, sess, model, rep_index):
        self.tf = tf
        self.sess = sess
        self.model = model
        self.rep_index = rep_index
        self.has_merge = any(('Add' in type(layer).__name__) or ('Concatenate' in type(layer).__name__)
                             for layer in model.layers)
        layers = sorted(set(layer_idx for layer_idx,_ in rep_index))
        if len(layers) == 1 and 0 <= layers[0] < len(model.layers):
            self.split_layer = layers[0]
        elif self.has_merge:
            add_layers = [i for i,l in enumerate(model.layers) if 'Add' in type(l).__name__]
            if add_layers:
                self.split_layer = add_layers[-1]
            else:
                dense_layers = [i for i,l in enumerate(model.layers) if 'Dense' in type(l).__name__]
                self.split_layer = max(0, dense_layers[-1]-1) if dense_layers else len(model.layers)//2
        else:
            dense_layers = [i for i,l in enumerate(model.layers) if 'Dense' in type(l).__name__]
            self.split_layer = max(0, dense_layers[-1]-1) if dense_layers else len(model.layers)//2
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Lambda
        if self.has_merge:
            self.model1 = Model(inputs=model.input, outputs=model.layers[self.split_layer].output,
                                name='ccbr_test_model1')
            if self.split_layer >= len(model.layers)-1:
                self.tail_mode = 'identity'
            else:
                dense_layers = [layer for layer in model.layers if 'Dense' in type(layer).__name__]
                if not dense_layers:
                    raise ValueError('CCBR ResNet manual head needs a Dense layer')
                parts = dense_layers[-1].get_weights()
                if len(parts) == 2:
                    self.head_w,self.head_b = parts
                elif len(parts) == 1:
                    self.head_w = parts[0]
                    self.head_b = np.zeros(self.head_w.shape[1],dtype=np.float32)
                else:
                    raise ValueError('CCBR ResNet last Dense has unsupported weights')
                self.tail_mode = 'manual_dense'
        else:
            start = 1 if 'InputLayer' in type(model.layers[0]).__name__ else 0
            z = model.input
            for i in range(start,self.split_layer+1):
                z = model.layers[i](z)
            self.model1 = Model(inputs=model.input,outputs=z,name='ccbr_test_model1')
            shape = tuple(self.model1.output.shape.as_list()[1:])
            zz = Input(shape=shape,name='ccbr_test_split_input')
            if self.split_layer >= len(model.layers)-1:
                self.model2 = Model(zz,Lambda(lambda t:t,name='ccbr_test_identity_tail')(zz))
            else:
                tail = zz
                for i in range(self.split_layer+1,len(model.layers)):
                    tail = model.layers[i](tail)
                self.model2 = Model(zz,tail,name='ccbr_test_model2')
            self.tail_mode = 'keras'
        active = [(layer_idx,neuron_idx) for layer_idx,neuron_idx in rep_index if layer_idx == self.split_layer]
        print('[CCBR] split_layer=%d, tail_mode=%s, usable neurons=%d/%d' %
              (self.split_layer,self.tail_mode,len(active),len(rep_index)))
        if not active:
            print('[WARNING] No selected neurons on split layer: repair will be no-op as in CCBR source')

    def original_predict(self, x):
        return self.sess.run(self.model.output, feed_dict={self.model.input: x})

    def repaired_predict(self, images, coeff):
        hidden = self.sess.run(self.model1.output,feed_dict={self.model1.input: images})
        hidden2 = hidden.copy()
        for j, (layer_idx, neuron_idx) in enumerate(self.rep_index):
            if layer_idx != self.split_layer:  # Original CCBR skip rule.
                continue
            if hidden.ndim == 2:
                if neuron_idx < hidden.shape[1]:
                    hidden2[:,neuron_idx] = coeff[j] * hidden[:,neuron_idx]
            elif hidden.ndim == 4:
                if neuron_idx < hidden.shape[-1]:
                    hidden2[:,:,:,neuron_idx] = coeff[j] * hidden[:,:,:,neuron_idx]
            else:
                raise ValueError('Unsupported CCBR hidden shape: %s' % (hidden.shape,))
        if self.tail_mode == 'keras':
            return self.sess.run(self.model2.output,feed_dict={self.model2.input:hidden2})
        if self.tail_mode == 'identity':
            return hidden2
        if hidden2.ndim == 4:
            hidden2 = np.mean(hidden2,axis=(1,2))
        elif hidden2.ndim != 2:
            raise ValueError('CCBR manual Dense tail unsupported hidden ndim %s' % hidden2.ndim)
        if hidden2.shape[1] != self.head_w.shape[0]:
            raise ValueError('CCBR manual head mismatch, expected %s, got %s at split layer %s' %
                             (self.head_w.shape[0], hidden2.shape[1],self.split_layer))
        # CCBR returns softmax(head logits); argmax is unchanged without softmax.
        return hidden2 @ self.head_w + self.head_b

    def predict_labels(self, x, batch_size, coeff=None):
        labels=[]
        for begin in range(0,len(x),batch_size):
            batch = x[begin:begin+batch_size]
            raw = self.original_predict(batch) if coeff is None else self.repaired_predict(batch,coeff)
            labels.append(np.argmax(raw,axis=-1).astype(np.int64).reshape(-1))
        return np.concatenate(labels,axis=0)


def score_preds(clean_pred, poison_pred, true_y, target):
    true_y = np.asarray(true_y).reshape(-1)
    eligible = true_y != target
    n = int(len(true_y))
    num_eligible = int(np.sum(eligible))
    successes = int(np.sum((poison_pred == target) & eligible))
    acc = float(np.mean(clean_pred == true_y))
    return {'clean_accuracy':acc,'sr': float(successes/n), # EXACT CCBR train convention.
            'legacy_sr':float(successes/n),
            'standard_asr': float(successes/num_eligible) if num_eligible else None,
            'attack_success_count': successes, 'non_target_count':num_eligible,
            'total_count': n}


def evaluate(evaluator, clean, triggered, labels, target, batch_size, coeff=None):
    clean_pred = evaluator.predict_labels(clean,batch_size,coeff)
    attacked_pred = evaluator.predict_labels(triggered,batch_size,coeff)
    return score_preds(clean_pred,attacked_pred,labels,target)


def evaluate_triggered(evaluator, images, labels, target, batch_size, coeff=None):
    pred = evaluator.predict_labels(images,batch_size,coeff)
    eligible = np.asarray(labels) != target
    n = len(labels)
    success = int(np.sum((pred == target) & eligible))
    return {'sr':success/n, 'standard_asr':success/int(np.sum(eligible)) if np.any(eligible) else None,
            'successes':success, 'non_target_count':int(np.sum(eligible)), 'total_count':n}


def evasion_cases(tf,sess,poisoned,labels,model_name,max_n,seed):
    from scipy.ndimage import rotate
    rng=np.random.RandomState(int(seed)+1000)
    idx=rng.choice(len(poisoned),min(max_n,len(poisoned)),replace=False)
    x=poisoned[idx]
    y=np.asarray(labels)[idx]
    h,w,_=DATASET_CONFIG[model_name]['input_shape']
    limit=255.0 if model_name=='gtsrb' else 1.0
    results={}
    rotations=[]
    for angle in (-15,-10,10,15):
        rotations.append(np.clip(np.stack([rotate(im,angle,axes=(0,1),reshape=False,order=1,mode='nearest')
                                           for im in x]),0,limit).astype(np.float32))
    results['rotation']=(np.concatenate(rotations),np.tile(y,4))
    placeholder=tf.compat.v1.placeholder(tf.float32,shape=(None,h,w,DATASET_CONFIG[model_name]['input_shape'][2]))
    compressed=[]
    for ratio in (0.5,0.7):
        size=(max(1,int(h*ratio)),max(1,int(w*ratio)))
        small=tf.image.resize(placeholder,size)
        large=tf.image.resize(small,(h,w))
        compressed.append(np.clip(sess.run(large,feed_dict={placeholder:x}),0,limit).astype(np.float32))
    results['compression']=(np.concatenate(compressed),np.tile(y,2))
    noise_scales=(10.0,20.0) if model_name=='gtsrb' else (0.04,0.08)
    noises=[np.clip(x+rng.normal(0,sigma,size=x.shape),0,limit).astype(np.float32)
            for sigma in noise_scales]
    results['noise']=(np.concatenate(noises),np.tile(y,2))
    results['brightness']=(np.concatenate([np.clip(x*v,0,limit).astype(np.float32)
                                           for v in (0.7,1.3)]),np.tile(y,2))
    return results


def sha256_file(file):
    h=hashlib.sha256()
    with open(str(file),'rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):
            h.update(buf)
    return h.hexdigest()


def save_json(path,payload):
    with open(str(path),'w',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)


def run(args):
    start=time.perf_counter()
    model_name,tag=model_identity(args)
    cfg=DATASET_CONFIG[model_name]
    run_dir=(Path(args.run_dir) if args.run_dir else Path(args.output_root)/tag/str(args.seed)/'nsga3').expanduser().resolve()
    loc_dir=(Path(args.loc_dir) if args.loc_dir else Path(args.output_root)/tag).expanduser().resolve()
    record,x,f=read_train_front(run_dir)
    coeff,ckpt,selection=selected_scaling(run_dir,args,record,x,f,model_name)

    outdir=Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir
    outdir.mkdir(parents=True,exist_ok=True)
    paths=[outdir/'ccbr_test_results.json',outdir/'ccbr_test_metrics.csv',outdir/'ccbr_time_summary.json']
    if args.evaluate_front: paths.append(outdir/'ccbr_pareto_test_results.csv')
    existing=[p for p in paths if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError('Output exists: %s; use --overwrite to replace' % existing)

    t=time.perf_counter()
    tf,sess=setup_tf(args.seed)
    path=Path(args.model_path or (Path(args.model_root)/(tag+'.h5'))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Original model not found: %s' % path)
    model=tf.keras.models.load_model(str(path),compile=False)
    selected_players,rep_indices,loc_meta=read_repair_indices(loc_dir,model)
    if len(rep_indices)!=len(coeff):
        raise ValueError('CCBR rep_indices length %d != scaling coeff length %d. Check '
                         'locator files and repair output are from the same run.' % (len(rep_indices),len(coeff)))
    evaluator=CCBRActivationEvaluator(tf,sess,model,rep_indices)
    model_build_time=time.perf_counter()-t

    t=time.perf_counter()
    clean,labels,test_source=load_test_data(model_name,args,tf,sess)
    if tuple(clean.shape[1:]) != tuple(cfg['input_shape']):
        raise ValueError('Unexpected test input shape %s vs %s' % (clean.shape[1:],cfg['input_shape']))
    total_test_count=len(labels)
    if args.test_samples>0 and args.test_samples<len(labels):
        rng=np.random.RandomState(int(args.seed)+424242)
        take=rng.choice(len(labels),args.test_samples,replace=False)
        clean,labels=clean[take],labels[take]
    triggered,trigger_meta=trigger_images(model_name,clean,args,tf)
    data_time=time.perf_counter()-t

    t=time.perf_counter()
    original=evaluate(evaluator,clean,triggered,labels,cfg['target_label'],args.batch_size)
    repaired=evaluate(evaluator,clean,triggered,labels,cfg['target_label'],args.batch_size,coeff)
    eval_time=time.perf_counter()-t
    print('[CCBR TEST] %s seed=%s, TEST=%s/%s' % (tag,args.seed,len(labels),total_test_count))
    print('[TRAIN SELECTED] index=%s scaling=%s' % (selection.get('train_solution_index'),ckpt))
    print('[BEFORE] ACC=%.4f%%  CCBR SR=%.4f%%  standard ASR=%s' %
          (100*original['clean_accuracy'],100*original['sr'],
           '%.4f%%' % (100*original['standard_asr']) if original['standard_asr'] is not None else 'N/A'))
    print('[AFTER]  ACC=%.4f%%  CCBR SR=%.4f%%  standard ASR=%s' %
          (100*repaired['clean_accuracy'],100*repaired['sr'],
           '%.4f%%' % (100*repaired['standard_asr']) if repaired['standard_asr'] is not None else 'N/A'))
    print('[DELTA]  ACC=%+.4f pp  CCBR SR=%+.4f pp' %
          (100*(repaired['clean_accuracy']-original['clean_accuracy']),
           100*(repaired['sr']-original['sr'])))
    if not selection['train_selection_verified']:
        print('[WARNING] Explicit selection used: confirm chosen BEFORE any TEST evaluation')

    front_results=[]
    t=time.perf_counter()
    if args.evaluate_front:
        for i in range(len(x)):
            p=run_dir/'pareto_front'/('repair_weights_%d.npy' % i)
            if not p.is_file():
                raise FileNotFoundError('Pareto scaling vector missing: %s' % p)
            scale=np.asarray(np.load(str(p),allow_pickle=False),dtype=np.float32).reshape(-1)
            metrics=evaluate(evaluator,clean,triggered,labels,cfg['target_label'],args.batch_size,scale)
            front_results.append({'index':int(i),'train_sr':float(f[i,0]),'train_accuracy':float(-f[i,1]),
                                  'test_metrics':metrics})
        with (outdir/'ccbr_pareto_test_results.csv').open('w',newline='',encoding='utf-8') as fp:
            writer=csv.writer(fp)
            writer.writerow(['index','train_sr','train_accuracy','test_sr','test_accuracy','test_standard_asr'])
            for entry in front_results:
                met=entry['test_metrics']
                writer.writerow([entry['index'],entry['train_sr'],entry['train_accuracy'],
                                 met['sr'],met['clean_accuracy'],met['standard_asr']])
    front_time=time.perf_counter()-t

    evasion_results=None
    t=time.perf_counter()
    if args.evasion:
        evasion_results={}
        for name,(cases,y) in evasion_cases(tf,sess,triggered,labels,model_name,args.evasion_samples,args.seed).items():
            evasion_results[name]={
                'original':evaluate_triggered(evaluator,cases,y,cfg['target_label'],args.batch_size),
                'train_selected_repaired':evaluate_triggered(evaluator,cases,y,cfg['target_label'],args.batch_size,coeff)}
    evasion_time=time.perf_counter()-t

    total_time=time.perf_counter()-start
    timing={'model_load_and_split_sec':model_build_time,'test_data_and_trigger_sec':data_time,
            'before_after_test_eval_sec':eval_time,'front_test_eval_sec':front_time,
            'evasion_test_eval_sec':evasion_time,'total_test_wall_sec':total_time,
            'previous_repair_optimization_sec':record.get('optimization_time')}
    result={'schema':'CCBR_NSAG3_activation_scaling_holdout_test_v2',
            'method':'CCBR','optimizer':'nsga3',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'model':tag,'seed':int(args.seed),
            'test_source':test_source,'test_count':int(len(labels)),
            'test_full_count':int(total_test_count),
            'test_is_subset':bool(total_test_count!=len(labels)),
            'trigger':trigger_meta,'original_model_file':str(path),
            'localization':loc_meta,'selected_players':selected_players,
            'rep_index_info':rep_indices,'split_layer':int(evaluator.split_layer),
            'tail_mode':evaluator.tail_mode,
            'activation_scaling_file':str(ckpt),'scaling_sha256':sha256_file(ckpt),
            'selection':selection,'selection_uses_TEST':False,
            'sr_definition':'CCBR source: triggered target predictions from non-target labels / ALL labels',
            'original_test_metrics':original,'repaired_test_metrics':repaired,
            'delta_accuracy':float(repaired['clean_accuracy']-original['clean_accuracy']),
            'delta_sr':float(repaired['sr']-original['sr']),
            'evasion_results':evasion_results,'pareto_analysis_only':front_results,
            'timing_sec':timing}
    save_json(paths[0],result)
    with paths[1].open('w',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh)
        wr.writerow(['model','seed','test_count','selected_index','original_acc','repaired_acc',
                     'original_ccbr_sr','repaired_ccbr_sr','original_standard_asr','repaired_standard_asr',
                     'delta_acc','delta_ccbr_sr','test_total_sec'])
        wr.writerow([tag,args.seed,len(labels),selection.get('train_solution_index'),
                     original['clean_accuracy'],repaired['clean_accuracy'],original['sr'],repaired['sr'],
                     original['standard_asr'],repaired['standard_asr'],
                     repaired['clean_accuracy']-original['clean_accuracy'],
                     repaired['sr']-original['sr'],total_time])
    save_json(paths[2],{'method':'CCBR','model':tag,'seed':int(args.seed),'timing_sec':timing,
                        'note':'Previous optimization time is read from TRAIN Pareto metadata; '
                               'the historical optimization and current TEST were separate runs.'})
    print('[TIME] TEST %.2fs, previous TRAIN optimization=%s' %
          (total_time,record.get('optimization_time')))
    print('[FILES] %s / %s / %s' % (paths[0],paths[1],paths[2]))
    return result


def main(argv=None):
    a=parse_args(argv)
    run(a)


if __name__=='__main__':
    main()
