"""TensorFlow/Keras runtime activation intervention port of DeepProv.

NOT the author's original IPG/GNN inference provenance code. This module
reproduces the *runtime action* in official artifact/library/mod_models.py:
  x[eligible] = x[eligible] - alpha * (x[eligible] - benign_reference)
where eligible = selected & (abs(delta) < beta).

It builds a *new Keras model* with an action layer inserted in its forward
path. Simply writing different Dense/Conv weights is NOT DeepProv's act_on.
"""
from __future__ import print_function
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

# User's nsrepair environment uses TF 1.x-style graph execution.
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
try:
    sess = K.get_session()
except Exception:
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    try:
        K.set_session(sess)
    except Exception:
        tf.compat.v1.keras.backend.set_session(sess)


class DeepProvPaperActOn(tf.keras.layers.Layer):
    """Non-trainable, serializable node action layer with per-node beta."""
    def __init__(self, alpha=1.0, **kwargs):
        super(DeepProvPaperActOn, self).__init__(**kwargs)
        self.alpha = float(alpha)
        if not (0 <= self.alpha <= 1):
            raise ValueError('DeepProv alpha must be in [0,1]')

    def build(self, input_shape):
        dimensions = tuple(int(d) for d in input_shape[1:])
        if any(d <= 0 for d in dimensions):
            raise ValueError('DeepProv needs statically known activation shape')
        self.reference = self.add_weight(name='benign_reference', shape=dimensions,
                                         initializer='zeros', trainable=False)
        self.action_mask = self.add_weight(name='action_mask', shape=dimensions,
                                           initializer='zeros', trainable=False)
        self.beta = self.add_weight(name='beta_threshold', shape=dimensions,
                                    initializer='zeros', trainable=False)
        super(DeepProvPaperActOn, self).build(input_shape)

    def call(self, inputs, **kwargs):
        shape = [1] + [int(d) for d in self.reference.shape.as_list()]
        ref = tf.reshape(self.reference, shape)
        threshold = tf.reshape(self.beta, shape)
        selected = tf.reshape(self.action_mask, shape) > 0.5
        diff = inputs - ref
        allowed = tf.logical_and(selected, tf.less(tf.abs(diff), threshold))
        return tf.where(allowed, inputs - self.alpha * diff, inputs)

    def get_config(self):
        cfg = super(DeepProvPaperActOn, self).get_config()
        cfg.update({'alpha':self.alpha})
        return cfg


def _as_list(obj):
    return list(obj) if isinstance(obj,(list,tuple)) else [obj]


def _key(t):
    try: return t.ref()
    except Exception: return getattr(t,'name',id(t))


def _map(obj, lookup):
    if isinstance(obj, list): return [_map(v,lookup) for v in obj]
    if isinstance(obj,tuple): return tuple(_map(v,lookup) for v in obj)
    key = _key(obj)
    if key not in lookup:
        raise KeyError('Cannot map Keras graph tensor {}'.format(getattr(obj,'name',obj)))
    return lookup[key]


def build_action_model(original, selected, references, beta_by_layer, alpha=1.0):
    """Clone Keras Functional/Sequential graph and inject act_on after layers.

    Selected action dicts: {'layer_name', 'neuron_idx', 'category'}.
    The learned weights of original layers are copied bit-for-bit; the repair
    is encoded in nontrainable extra layer tensors.
    """
    actions = {}
    for item in selected:
        actions.setdefault(item['layer_name'], []).append(item)
    if not actions:
        return original
    inputs=[]
    mapping={}
    original_inputs=_as_list(original.inputs)
    for i, inp in enumerate(original_inputs):
        dims=tuple(int(d) if d is not None else None for d in inp.shape.as_list()[1:])
        new_input=tf.keras.layers.Input(shape=dims, name='deep_prov_input_{}'.format(i))
        inputs.append(new_input)
        mapping[_key(inp)] = new_input
    layer_pairs=[]
    inject=[]
    for original_layer in original.layers:
        if isinstance(original_layer, tf.keras.layers.InputLayer):
            for old, new in zip(_as_list(original_layer.output),inputs):
                mapping[_key(old)]=new
            continue
        config=original_layer.get_config()
        config.pop('batch_input_shape',None)
        config.pop('batch_shape',None)
        clone=original_layer.__class__.from_config(config)
        layer_pairs.append((original_layer,clone))
        for node in original_layer._inbound_nodes:
            mapped_inputs=_map(node.input_tensors,mapping)
            if isinstance(mapped_inputs,(list,tuple)) and len(mapped_inputs)==1:
                mapped_inputs=mapped_inputs[0]
            new_output=clone(mapped_inputs)
            if original_layer.name in actions:
                if isinstance(new_output,(list,tuple)):
                    raise ValueError('Action layer must have single tensor output: {}'.format(original_layer.name))
                act=DeepProvPaperActOn(alpha=alpha,name='dp_act_on_'+original_layer.name)
                new_output=act(new_output)
                inject.append((original_layer.name,act,actions[original_layer.name]))
            old_outputs=_as_list(node.output_tensors)
            new_outputs=_as_list(new_output)
            if len(old_outputs)!=len(new_outputs):
                raise RuntimeError('Output count changed for {}'.format(original_layer.name))
            for a,b in zip(old_outputs,new_outputs):
                mapping[_key(a)]=b
    outputs=[_map(out,mapping) for out in _as_list(original.outputs)]
    repaired=tf.keras.models.Model(inputs=inputs[0] if len(inputs)==1 else inputs,
                                  outputs=outputs[0] if len(outputs)==1 else outputs,
                                  name='DeepProv_Activation_Port')
    for src, dst in layer_pairs:
        if src.get_weights():
            dst.set_weights(src.get_weights())
    for name, act, items in inject:
        ref=np.asarray(references[name],dtype=np.float32).copy()
        mask=np.zeros_like(ref,dtype=np.float32)
        beta=np.zeros_like(ref,dtype=np.float32)
        per_channel=np.asarray(beta_by_layer[name],np.float32)
        for item in items:
            j=int(item['neuron_idx'])
            mask[...,j]=1.0
            if item.get('category')=='nullify':
                ref[...,j]=0.0
                beta[...,j]=np.finfo(np.float32).max / 8.0
            else:
                beta[...,j]=per_channel[j]
        act.set_weights([ref,mask,beta])
    # Original layers must remain unchanged, otherwise it isn't an act_on port.
    for src, dst in layer_pairs:
        for left,right in zip(src.get_weights(),dst.get_weights()):
            if not np.array_equal(left,right):
                raise AssertionError('Original trainable weights changed during graph rebuild')
    return repaired


def model_predict(model, x, batch_size=64):
    # Use graph-mode session, consistent with old user code.
    out=[]
    for start in range(0,len(x),batch_size):
        end=min(start+batch_size,len(x))
        feed={model.input:x[start:end]}
        phase=K.learning_phase()
        if not isinstance(phase,int):
            feed[phase]=0
        out.append(K.get_session().run(model.output,feed_dict=feed))
    return np.concatenate(out,axis=0)


def accuracy(model, pair, batch_size=64):
    x,y=pair
    if not len(y): raise ValueError('Empty evaluation set')
    return float(np.mean(np.argmax(model_predict(model,x,batch_size),axis=-1)==y))


def gather_activations_and_proxy(model, x, y, layer_names, batch_size=32):
    """Empirical activation + gradient-times-activation structural PROXY.

    Official DeepProv's structural attribution is GNN graph attribution. This
    is a Keras-compatible surrogate, not an equivalent GNN reproduction.
    """
    if len(y)!=len(x): raise ValueError('x/y mismatch')
    labels_ph=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,),name='deep_prov_loc_labels')
    scores=tf.reduce_sum(model.output * tf.one_hot(labels_ph, int(model.output_shape[-1])), axis=1)
    layer_tensors=[model.get_layer(name).output for name in layer_names]
    gradients=tf.gradients(tf.reduce_sum(scores), layer_tensors)
    if any(v is None for v in gradients):
        bad=[name for name,g in zip(layer_names,gradients) if g is None]
        raise RuntimeError('Cannot form gradient proxy for layers {}'.format(bad))
    chunks={name:[] for name in layer_names}
    attrib={name:np.zeros(int(model.get_layer(name).output.shape[-1]),np.float64) for name in layer_names}
    n=0
    for start in range(0,len(y),batch_size):
        end=min(start+batch_size,len(y))
        feed={model.input:x[start:end],labels_ph:np.asarray(y[start:end],dtype=np.int32)}
        phase=K.learning_phase()
        if not isinstance(phase,int): feed[phase]=0
        outputs=K.get_session().run(layer_tensors+gradients,feed_dict=feed)
        n+=end-start
        for idx,name in enumerate(layer_names):
            a=outputs[idx]
            g=outputs[idx+len(layer_names)]
            chunks[name].append(a)
            dims=tuple(range(a.ndim-1))
            attrib[name]+=np.sum(np.abs(a*g),axis=dims)
    if n==0: raise ValueError('Empty localization data')
    return ({name:np.concatenate(chunks[name],axis=0) for name in layer_names},
            {name:attrib[name]/n for name in layer_names})


def list_layers(model):
    result=[]
    last_id=len(model.layers)-1
    for i,layer in enumerate(model.layers):
        if i==last_id: continue
        if not isinstance(layer,(tf.keras.layers.Dense,tf.keras.layers.Conv2D)):
            continue
        shape=layer.output_shape
        if not isinstance(shape,(tuple,list)) or len(shape) not in (2,4):
            continue
        result.append(layer.name)
    return result
