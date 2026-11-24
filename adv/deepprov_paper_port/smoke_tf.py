#!/usr/bin/env python3
"""Run in user's TensorFlow nsrepair environment before real DeepProv repair."""
from __future__ import print_function
import numpy as np
import tensorflow as tf
from deepprov_runtime import (build_action_model,DeepProvPaperActOn,model_predict)
from deepprov_train import set_action_state
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
np.random.seed(13)
model_input=tf.keras.layers.Input(shape=(4,))
hidden=tf.keras.layers.Dense(6,activation='relu',name='hidden')(model_input)
model_out=tf.keras.layers.Dense(2,activation='softmax',name='classifier')(hidden)
model=tf.keras.models.Model(inputs=model_input,outputs=model_out)
x=np.random.randn(16,4).astype(np.float32)
y0=model_predict(model,x)
a={'layer_name':'hidden','neuron_idx':0,'category':'regular'}
references={'hidden':np.full((6,),0.12,np.float32)}
betas={'hidden':np.full((6,),100.0,np.float32)}
wrapped=build_action_model(model,[a],references,betas,alpha=1)
set_action_state(wrapped,[],references,betas)
y1=model_predict(wrapped,x)
assert np.allclose(y0,y1,atol=1e-6),np.max(np.abs(y0-y1))
set_action_state(wrapped,[a],references,betas)
y2=model_predict(wrapped,x)
assert np.any(np.abs(y2-y1)>1e-8), 'Action layer not on active forward path'
wrapped.save('deepprov_smoke_tmp.h5',include_optimizer=False)
restored=tf.keras.models.load_model('deepprov_smoke_tmp.h5',compile=False,
    custom_objects={'DeepProvPaperActOn':DeepProvPaperActOn})
y3=model_predict(restored,x)
assert np.allclose(y2,y3,atol=1e-6),np.max(np.abs(y3-y2))
print('[SMOKE PASS] graph clone, no-action parity, activation effect, save/reload')
