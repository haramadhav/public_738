# -*- coding: utf-8 -*-
"""
EECS-738 Machine Learning
Lab_4 (CIFAR)
Neural Networks (Functional API, Shallow, and Deep ResNet)
@author: h067t028
Hara Madhav Talasila
"""

# Load libraries
import keras
from keras.datasets import mnist, cifar10 # for loading the raw_data
# for NN
from sequential_neural_network import hara_NN # for baseline (Lab_3)
from def_stack import * # for my defs
from keras.models import Sequential, Model # for NN model, API
from keras.layers import * # Input, Dense, Dropout # all as *
from keras.optimizers import SGD, Adam # for compiling
import numpy as np # for math

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

## Prep data
# Reshape
var_input_shape = np.prod(x_train.shape[1:])
x_train = x_train.reshape(x_train.shape[0], var_input_shape)
x_test  = x_test.reshape(x_test.shape[0], var_input_shape)
# Convert unit8 to float
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
# Normalize
x_train /= 255 # x_train = x_train/255
x_test  /= 255 # x_test = x_test/255
# Convert labels to categorical orthonormal vectors
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)


# Baseline NN model from functional API
inputs = Input(shape = (var_input_shape,), name = 'INPUT_input')
x = Dense(64, activation='relu', name = 'INPUT')(inputs)
x = Dense(64, activation='relu', name = 'Hidden_1')(x)
x = Dense(64, activation='relu', name = 'Hidden_2')(x)
x = Dense(64, activation='relu', name = 'Hidden_3')(x)
outputs = Dense(10, activation='softmax', name = 'OUTPUT')(x)
model = Model(inputs=inputs, outputs=outputs, name = 'P8_API' )
# model.summary()
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test)
succ_P8 = hara_NN_model_HIST_LOG(history)
keras.utils.plot_model(model, show_shapes=True, to_file = 'P8_API_Model.png')
del [model, history, inputs, x, outputs]

var_title   = ['P9_skip2',  'P10_Deep', 'P10_skip1',    'P10_skip3']
var_skip    = [2,           2,          1,              3]
var_neurons = [128,         128,        128,            128]
var_dropout = [0.5,         0.5,        0.5,            0.5]
var_blocks  = [5,           12,         5,              5]
var_lr      = [0.001,       0.001,      0.001,          0.001]
var_bs      = [128,         128,        128,            128]
idx_list    = range(len(var_title))
# var_title   = ['P10_1',     'P10_2',    'P10_3',    'P10_4']
# var_skip    = [2,           2,          1,              3]
# var_neurons = [64,          128,        128,            128]
# var_dropout = [0.0,         0.5,        0.5,            0.5]
# var_blocks  = [5,           12,         5,              5]
# var_lr      = [0.002,       0.01,       0.003,          0.02]
# var_bs      = [256,         128,        128,            128]
# idx_list    = [0, 1]

for idx in idx_list:
    model = hara_NN_model_RES(var_title = var_title[idx],
                              var_input_shape = var_input_shape, # DATA
                              var_skip = var_skip[idx],
                              var_neurons = var_neurons[idx],
                              var_dropout = var_dropout[idx], 
                              var_blocks = var_blocks[idx])
    # model.summary()
    keras.utils.plot_model(model, show_shapes=True, 
                           to_file = model.name+'_Model.png')
    history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test,
                                var_opt = Adam(amsgrad=True,
                                               learning_rate = var_lr[idx]),
                                var_loss = 'binary_crossentropy',
                                var_batch_size = var_bs[idx])
    succ_P = hara_NN_model_HIST_LOG(history)
    del [model, history]