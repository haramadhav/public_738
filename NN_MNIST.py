# -*- coding: utf-8 -*-
"""
EECS-738 Machine Learning
Lab_4 (MNIST)
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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


# Baseline NN model from Lab-3
success_P3      = hara_NN(var_title = 'P3_Lab_3', # Baseline
                          var_shape = [64,64,64,10],var_dropout = 0, 
                          var_decay = 1e-6, var_momentum = 0.9, #comp opt
                          var_batch_size = 64, var_epochs = 20, #fit
                          var_verbose = 0)

# Baseline NN model from my def_stack using Sequential model
model   = hara_NN_model_SEQ(var_title = 'P3_Base')
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test)
succ_P3 = hara_NN_model_HIST_LOG(history)
keras.utils.plot_model(model, show_shapes=True, to_file = 'P3_Base_Model.png')
del [model, history]

# Baseline NN model from functional API
inputs = Input(shape = (var_input_shape,), name = 'INPUT_input')
x = Dense(64, activation='relu', name = 'INPUT')(inputs)
x = Dense(64, activation='relu', name = 'Hidden_1')(x)
x = Dense(64, activation='relu', name = 'Hidden_2')(x)
x = Dense(64, activation='relu', name = 'Hidden_3')(x)
outputs = Dense(10, activation='softmax', name = 'OUTPUT')(x)
model = Model(inputs=inputs, outputs=outputs, name = 'P4_API' )
# model.summary()
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test)
succ_P4 = hara_NN_model_HIST_LOG(history)
keras.utils.plot_model(model, show_shapes=True, to_file = 'P4_API_Model.png')
del [model, history, inputs, x, outputs]
 
var_title   = ['P5_skip2',  'P6_Deep',  'P7_skip1',     'P7_skip3']
var_skip    = [2,           2,          1,              3]
var_neurons = [128,         128,        128,            128]
var_dropout = [0.5,         0.5,        0.5,            0.5]
var_blocks  = [5,           12,         5,              5]
for idx in range(len(var_title)):
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
                                var_opt = Adam(amsgrad=True),
                                var_loss = 'binary_crossentropy',
                                var_batch_size = 128)
    succ_P = hara_NN_model_HIST_LOG(history)
    del [model, history]