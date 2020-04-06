from keras.models import Sequential, Model # for NN model, API
from keras.layers import * # Input, Dense, Dropout # all as *
from keras.optimizers import SGD, Adam # for compiling

# SHALLOW Residual Network
inputs = Input(shape=(var_input_shape,), name='INPUT_img') 
x = Dense(128, activation='relu', name='B1_IN')(inputs)
block_1_output = Dense(128, activation='relu', name='B1_OUT')(x)

x = Dense(128, name = 'B2_IN')(block_1_output)
x = BatchNormalization(name = 'B2_BN1')(x)
x = Activation('relu', name = 'B2_ACT')(x)
x = Dropout(0.5, name = 'B2_DRP')(x)
x = Dense(128, name = 'B2_WGT')(x)
x = BatchNormalization(name = 'B2_BN2')(x)
block_2_output = add([x, block_1_output], name='B2_OUT')

x = Dense(128, name = 'B3_IN')(block_2_output)
x = BatchNormalization(name = 'B3_BN1')(x)
x = Activation('relu', name = 'B3_ACT')(x)
x = Dropout(0.5, name = 'B3_DRP')(x)
x = Dense(128, name = 'B3_WGT')(x)
x = BatchNormalization(name = 'B3_BN2')(x)
block_3_output = add([x, block_2_output], name='B3_OUT')

x = Dense(128, name='B4_IN')(block_3_output)
x = BatchNormalization(name = 'B4_BN1')(x)
x = Activation('relu', name = 'B4_ACT')(x)
x = Dropout(0.5, name = 'B4_DRP')(x)
x = Dense(128, name = 'B4_WGT')(x)
x = BatchNormalization(name = 'B4_BN2')(x)
block_4_output = add([x, block_3_output], name='B4_OUT')

x = Dense(128, activation='relu', name='B5_IN')(block_4_output)
x = Dropout(0.5, name = 'B5_DRP')(x)
outputs = Dense(10, activation='softmax', name='B5_OUT')(x)

model = Model(inputs, outputs, name='P5_ResNet_Blackboard')
model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file = model.name+'_Model.png')
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test,
                            var_opt = Adam(amsgrad=True),
                            var_loss = 'binary_crossentropy',
                            var_batch_size = 128)
succ_P5 = hara_NN_model_HIST_LOG(history)
del [model, history, inputs, x, outputs]

# SHALLOW Residual Network
model = hara_NN_model_RES(var_title = 'P5_Shallow', var_input_shape = 784,
                          var_skip = 2, var_neurons = 128,
                          var_dropout = 0.5, var_blocks = 5)
# model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file = model.name+'_Model.png')
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test,
                            var_opt = Adam(amsgrad=True),
                            var_loss = 'binary_crossentropy',
                            var_batch_size = 128)
succ_P5 = hara_NN_model_HIST_LOG(history)
del [model, history]

# DEEP Residual Network
model = hara_NN_model_RES(var_title = 'P6_Deep', var_input_shape = 784,
                          var_skip = 2, var_neurons = 128,
                          var_dropout = 0.5, var_blocks = 12)
# model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file = model.name+'_Model.png')
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test,
                            var_opt = Adam(amsgrad=True),
                            var_loss = 'binary_crossentropy',
                            var_batch_size = 128)
succ_P6 = hara_NN_model_HIST_LOG(history)
del [model, history]

# SHALLOW Residual Network (Skip_1)
model = hara_NN_model_RES(var_title = 'P7_skip1', var_input_shape = 784,
                          var_skip = 1, var_neurons = 128,
                          var_dropout = 0.5, var_blocks = 5)
# model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file = model.name+'_Model.png')
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test,
                            var_opt = Adam(amsgrad=True),
                            var_loss = 'binary_crossentropy',
                            var_batch_size = 128)
succ_P7_1 = hara_NN_model_HIST_LOG(history)
del [model, history]
# SHALLOW Residual Network (Skip_3)
model = hara_NN_model_RES(var_title = 'P7_skip3', var_input_shape = 784,
                          var_skip = 3, var_neurons = 128,
                          var_dropout = 0.5, var_blocks = 5)
# model.summary()
keras.utils.plot_model(model, show_shapes=True, to_file = model.name+'_Model.png')
history = hara_NN_model_RUN(model, x_train, y_train, x_test, y_test,
                            var_opt = Adam(amsgrad=True),
                            var_loss = 'binary_crossentropy',
                            var_batch_size = 128)
succ_P7_2 = hara_NN_model_HIST_LOG(history)
del [model, history]