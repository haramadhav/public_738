# -*- coding: utf-8 -*-
"""
EECS-738 Machine Learning
Lab_4 (definitions stack)
Neural Networks (Functional API, Shallow, and Deep ResNet)
@author: h067t028
Hara Madhav Talasila
"""

# Load libraries
import keras
from keras.datasets import mnist # for loading the raw_data
# for NN
from keras.models import Sequential, Model # for NN model
from keras.layers import * # Input, Dense, Dropout # all as *
from keras.optimizers import SGD, Adam # for compiling
from keras.callbacks import Callback # for epoch times
# for math in printing metrics
import numpy as np
# for beautiful plots
from matplotlib.pyplot import *

class epoch_time_logger(Callback):
  def on_train_begin(self, logs = {}): # initialize
      self.time_vector = []
  def on_epoch_begin(self, epoch, logs={}):
      self.t0 = time.time()
  def on_epoch_end(self, epoch, logs={}):
      self.time_vector.append(time.time()-self.t0)
      # time_vector is a log of each epoch time
epoch_time_log = epoch_time_logger()

def hara_NN_model_SEQ(var_title = 'NN', var_input_shape = 784,
            var_shape = [64,64,64,10],var_dropout = 0, # Baseline):
            ):
    # INPUT layer or first layer
    model = Sequential(name = var_title)
    model.add(  Dense(var_shape[0], activation='relu', 
                      input_shape=(var_input_shape,), name = 'INPUT') )
    if var_dropout: model.add(  Dropout(var_dropout, name = 'Drop_INPUT'))
    for idx in range(len(var_shape)-1):
        model.add(  Dense(var_shape[idx], activation='relu', 
                          name = 'Hidden_'+str(idx+1) ) )
        if var_dropout: model.add(  Dropout(var_dropout, 
                                            name = 'Drop_'+str(idx+1) ) )
    model.add(  Dense(var_shape[idx+1], activation='softmax', 
                      name = 'OUTPUT') )
    # print('_________________________________________________________________')
    # model.summary() # PRINT SUMMARY
    return model

def hara_NN_model_RES(var_title = 'RES', var_input_shape = 784,
                      var_skip = 2, var_neurons = 128, 
                      var_dropout = 0.5, var_blocks = 5
                      ):
    for idx in range(1,1+var_blocks):
        blk = 'B'+str(idx)
        if idx == 1:
            inputs = Input(shape = (var_input_shape,), name = 'INPUT_img') 
            x = Dense(var_neurons, activation = 'relu',
                      name = blk+'_IN')(inputs)
            blk_out = Dense(var_neurons, activation = 'relu',
                            name = blk+'_OUT')(x)
        elif idx == var_blocks:
            x = Dense(var_neurons, activation = 'relu',
                      name = blk+'_IN')(blk_out)
            x = Dropout(var_dropout, name = blk+'_DRP')(x)
            outputs = Dense(10, activation='softmax',
                            name = blk+'_OUT')(x)
        else:
            x = Dense(var_neurons, name = blk+'_IN')(blk_out)
            x = BatchNormalization(name = blk+'_BN1')(x)
            x = Activation('relu', name = blk+'_ACT1')(x)
            if var_skip >= 2:
                x = Dropout(var_dropout, name = blk+'_DRP1')(x)
                x = Dense(var_neurons, name = blk+'_WGT')(x)
                x = BatchNormalization(name = blk+'_BN2')(x)
            if var_skip >= 3:
                x = Activation('relu', name = blk+'_ACT2')(x)
                x = Dropout(var_dropout, name = blk+'_DRP2')(x)
                x = Dense(var_neurons, name = blk+'_WGT2')(x)
                x = BatchNormalization(name = blk+'_BN3')(x)
            blk_out = add([x, blk_out], name = blk+'_OUT')
    model = Model(inputs, outputs, name = var_title)
    return model

def hara_NN_model_RUN(model, x_train, y_train, x_test, y_test, 
                      var_batch_size = 64, var_epochs = 20, var_verbose = 0, 
                      var_decay = 1e-6, var_momentum = 0.9,
                      var_learning_rate = 0.01, #default
                      var_loss = 'categorical_crossentropy',
                      var_opt = 'sgd'):
    if var_opt == 'sgd':
        var_opt = SGD(learning_rate = var_learning_rate, 
                      decay = var_decay, momentum = var_momentum,
                      nesterov = True)
    model.compile(optimizer = var_opt,
                  loss = var_loss,
                  metrics=['accuracy'] )    

    history = model.fit(x_train, y_train,
                        batch_size = var_batch_size,
                        epochs = var_epochs,
                        verbose = var_verbose,
                        validation_data = (x_test, y_test),
                        callbacks = [epoch_time_log])
    return history

def hara_NN_model_HIST_LOG(history, epoch_time_log = epoch_time_log):
    success = 0
    print(history.model.name)
    print('#FIT#\tAcc_MAX\t\tAcc_mean\t\tLoss_MIN\tLoss_mean',
          '\nTrain\t', round(np.max(history.history['accuracy'])*100,2), 
          '\t\t', round(np.mean(history.history['accuracy'])*100,2),
          '\t\t\t', round(np.min(history.history['loss']),2),
          '\t\t', round(np.mean(history.history['loss']),2),
          '\nTest\t', round(np.max(history.history['val_accuracy'])*100,2), 
          '\t\t', round(np.mean(history.history['val_accuracy'])*100,2),
          '\t\t\t', round(np.min(history.history['val_loss']),2),
          '\t\t', round(np.mean(history.history['val_loss']),2),
          )
    print('#TIME#\tTotal: ', round(np.sum(epoch_time_log.time_vector),2),'s',
          '\tMean: ', round(np.mean(epoch_time_log.time_vector),2),'s',
           )
    print('_________________________________________________________________')
    
    # PLOTS
    figure()
    x_axis = list(range(1,1+len(history.epoch)))
    # subplot(311)
    plot(x_axis, 100*np.array(history.history['accuracy']),'r:*', label = 'Train') 
    plot(x_axis, 100*np.array(history.history['val_accuracy']),'b:o', label = 'Test')
    grid(color='k', linestyle='-', linewidth=0.1)
    xlabel('Epochs -->')
    ylabel('Accuracy, %')
    ylim(90,100)
    legend(loc = 4)
    suptitle(history.model.name)
    savefig(history.model.name + '_Accuracy.png', dpi = 300)
    figure()
    # subplot(312)
    plot(x_axis, history.history['loss'],'r:*', label = 'Train') 
    plot(x_axis, history.history['val_loss'],'b:o', label = 'Test')
    grid(color='k', linestyle='-', linewidth=0.1)
    xlabel('Epochs -->')
    ylabel('Loss')
    ylim(0,1)
    legend(loc = 1)
    suptitle(history.model.name)
    savefig(history.model.name + '_Loss.png', dpi = 300)
    figure()
    # subplot(313)
    plot(x_axis, epoch_time_log.time_vector,'r:*', label = 'dt') 
    grid(color='k', linestyle='-', linewidth=0.1)
    xlabel('Epochs -->')
    ylabel('Time, s')
    ylim(1,14)
    legend(loc = 1)
    suptitle(history.model.name)
    savefig(history.model.name + '_Time.png', dpi = 300)
    
    success = 1
    return success