# -*- coding: utf-8 -*-
"""
EECS-738 Machine Learning
Lab_3
Neural Networks
@author: h067t028
Hara Madhav Talasila
"""

# Load libraries
import keras
from keras.datasets import mnist # for loading the raw_data
# for NN
from keras.models import Sequential # for NN model
from keras.layers import Dense, Dropout # for layers, Dropout
from keras.optimizers import SGD # for compiling
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
    
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## Prep data
# Reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test  = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
# Convert unit8 to float
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
# Normalize
x_train /= 255 # x_train = x_train/255
x_test  /= 255 # x_test = x_test/255
# Convert labels to categorical orthonormal vectors
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)


## NN model
def hara_NN(var_title = 'NN',
            var_shape = [32,32,10],var_dropout = 0, # Baseline
            var_decay = 0, var_momentum = 0, #compile optimizer
            var_batch_size = 64, var_epochs = 20, #fit
            var_verbose = 0):
    success = 0
    # INPUT layer or first layer
    model = Sequential(name = var_title)
    model.add(  Dense(var_shape[0], activation='relu', input_shape=(784,), 
                      name = 'INPUT') )
    if var_dropout: model.add(  Dropout(var_dropout, name = 'Drop_INPUT'))
    for idx in range(len(var_shape)-1):
        # first hidden layer or second layer
        model.add(  Dense(var_shape[idx], activation='relu', 
                          name = 'Hidden_'+str(idx+1) ) )
        if var_dropout: model.add(  Dropout(var_dropout, 
                                            name = 'Drop_'+str(idx+1) ) )
        # second hidden layer or third layer
        # model.add(  Dense(32, activation='relu', name = 'Hidden_2') )
    # OUTPUT layer or fourth layer
    model.add(  Dense(var_shape[idx+1], activation='softmax', 
                      name = 'OUTPUT') )
    print('_________________________________________________________________')
    model.summary() # PRINT SUMMARY
    keras.utils.plot_model(model, show_shapes=True, 
                           to_file = 'Model_P3_baseline.png')

    model.compile(optimizer = SGD(learning_rate = 0.01, #default
                                  decay = var_decay,
                                  momentum = var_momentum),
                  loss = 'categorical_crossentropy',
                  metrics=['accuracy'] )    

    history = model.fit(x_train, y_train,
                        batch_size = var_batch_size,
                        epochs = var_epochs,
                        verbose = var_verbose,
                        validation_data = (x_test, y_test),
                        callbacks = [epoch_time_log])
    
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
    # score = model.evaluate(x_test, y_test,
    #                         batch_size = var_batch_size, 
    #                         verbose = var_verbose)
    # print('EVALUATE--> Test loss = ', score[0],'Test accuracy:', score[1])
    
    # PLOTS
    figure()
    x_axis = list(range(1,1+var_epochs))
    # subplot(311)
    plot(x_axis, 100*np.array(history.history['accuracy']),'r:*', label = 'Train') 
    plot(x_axis, 100*np.array(history.history['val_accuracy']),'b:o', label = 'Test')
    grid(color='k', linestyle='-', linewidth=0.1)
    xlabel('Epochs -->')
    ylabel('Accuracy, %')
    ylim(30,100)
    legend(loc = 4)
    suptitle(var_title)
    savefig('Accuracy_' + var_title + '.png', dpi = 300)
    figure()
    # subplot(312)
    plot(x_axis, history.history['loss'],'r:*', label = 'Train') 
    plot(x_axis, history.history['val_loss'],'b:o', label = 'Test')
    grid(color='k', linestyle='-', linewidth=0.1)
    xlabel('Epochs -->')
    ylabel('Loss')
    ylim(0,2)
    legend(loc = 1)
    suptitle(var_title)
    savefig('Loss_' + var_title + '.png', dpi = 300)
    figure()
    # subplot(313)
    plot(x_axis, epoch_time_log.time_vector,'r:*', label = 'dt') 
    grid(color='k', linestyle='-', linewidth=0.1)
    xlabel('Epochs -->')
    ylabel('Time, s')
    ylim(0.6,4)
    legend(loc = 1)
    suptitle(var_title)
    savefig('Time_' + var_title + '.png', dpi = 300)
    
    success = 1
    return success


# success_P3      = hara_NN(var_title = 'P3_Baseline')
# success_P4_1    = hara_NN(var_title = 'P4_Add_Neuron_64', 
#                           var_shape=[64,64,10])
# success_P4_2    = hara_NN(var_title = 'P4_Add_Neuron_128', 
#                           var_shape=[128,128,10])
# success_P4_3    = hara_NN(var_title = 'P4_Add_Neuron_256', 
#                           var_shape=[256,256,10])
# success_P5      = hara_NN(var_title = 'P5_Momentum', 
#                           var_decay = 0.000001, 
#                           var_momentum = 0.9)
# success_P6_1    = hara_NN(var_title = 'P6_Batch_size_128', 
#                           var_batch_size = 128)
# success_P6_2    = hara_NN(var_title = 'P6_Batch_size_32', 
#                           var_batch_size = 32)
# success_P7      = hara_NN(var_title = 'P7_Add_layer', 
#                           var_shape=[32,32,32,10])
# success_P8_1      = hara_NN(var_title = 'P8_Shape_128_dropout', 
#                           var_shape=[128,128,10],
#                           var_dropout = 0.5)
# success_P8_2      = hara_NN(var_title = 'P8_Shape_64_dropout', 
#                           var_shape=[64,64,10],
#                           var_dropout = 0.5)
# success_P8_3      = hara_NN(var_title = 'P8_Shape_32_dropout', 
#                           var_shape=[32,32,10],
#                           var_dropout = 0.5)
# success_P8_4      = hara_NN(var_title = 'P8_Shape_256_dropout', 
#                           var_shape=[256,256,10],
#                           var_dropout = 0.5)
# success_P8_5      = hara_NN(var_title = 'P8_Shape_512_dropout', 
#                           var_shape=[512,512,10],
#                           var_dropout = 0.5)
# success_P9      = hara_NN(var_title = 'P9_BestHara', 
#                           var_shape=[64,64,10],
#                           var_dropout = 0,
#                           var_decay = 0.000001,
#                           var_momentum = 0.9,
#                           var_batch_size = 128)
# success_P9      = hara_NN(var_title = 'P9_BestHara', 
#                           var_shape=[1024,1024,10],
#                           var_dropout = 0.5,
#                           var_decay = 0.000001,
#                           var_momentum = 0.9,
#                           var_batch_size = 64)
# success_P9      = hara_NN(var_title = 'P9_BestHara', 
#                           var_shape=[1024,1024,10],
#                           var_dropout = 0.5,
#                           var_decay = 0.000001,
#                           var_momentum = 0.9,
#                           var_batch_size = 32)
