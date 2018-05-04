# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:01:02 2016
ECE802-608 Final Project: Modification of classic LSTM
Case 1): no input signals, keep bias and hidden units

"""

#work directory
import os
wdir = '/home/dong/Documents/MSU/Courses/ECE885/Modified-LSTM-master/'
os.chdir(wdir)


###############################################################################
#                   Model1: modification without input signal                 #
###############################################################################
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from slim21 import LSTMs
import keras


# ###############################################################################
# ##                          Test1: MNIST Data                                ##
# ###############################################################################
# ##########################
# #(1)  Pixel-wise input   #
# #########################
# Load the data, shuffle and split them into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Data normalization
X_means = np.mean(X_train, axis=0)
X_stds = np.std(X_train, axis=0)
X_train = (X_train - X_means)/(X_stds+1e-6)
X_test = (X_test - X_means)/(X_stds+1e-6)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
#learning rate schedule
learning_rate = 0.0001


#early stopping
# earlystopper = EarlyStopping(monitor='val_acc', patience=15, verbose=1, mode='max')
#check point
#filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath = os.path.join(wdir,'model1','LSTM1MNIST1weights.best.hdf5') #LSTM_model1
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')

# create a model
batch_size = 32
nb_epochs = 100
hidden_units = 50
model = Sequential()
consume_less='mem'
LSTM1 = LSTMs(implementation=1, units=hidden_units,
              activation='tanh',
              input_shape=(28,28), model='LSTM6')


model.add(LSTM1)
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()
rmsprop = RMSprop(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=rmsprop,
              metrics=['accuracy'])

#fit and evalute the model
lstm1_mnist_hist1 = model.fit(X_train, Y_train,
                              batch_size=batch_size,
                              nb_epoch=nb_epochs,
                              verbose=1,
                              validation_data=(X_test, Y_test),
                              callbacks=[checkpoint])
scores = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

A = lstm1_mnist_hist1.history['acc']
B = lstm1_mnist_hist1.history['val_acc']
C = lstm1_mnist_hist1.history['loss']
D = lstm1_mnist_hist1.history['val_loss']

savePath = wdir
model.save(os.path.join(savePath,'LSTM_M1_MNIST1.h5')) #save complied model

#plot acc and loss vs epochs
import matplotlib.pyplot as plt
print(lstm1_mnist_hist1.history.keys())
#accuracy
plt.plot(lstm1_mnist_hist1.history['acc'])
plt.plot(lstm1_mnist_hist1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig(os.path.join(savePath,'model1','MNIST1_AccVsEpoch.jpeg'),
            dpi=1000, bbox_inches='tight')
plt.show()
#loss
plt.plot(lstm1_mnist_hist1.history['loss'])
plt.plot(lstm1_mnist_hist1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig(os.path.join(savePath,'model1','MNIST1_LossVsEpoch.jpeg'),
            dpi=1000, bbox_inches='tight')
plt.show()

#save training history
import numpy as np
np.save(os.path.join(savePath,'model1','lstm1_mnist_hist1.npy'),
        lstm1_mnist_hist1.history)
#load
hist = np.load(os.path.join(savePath,'model1','lstm1_mnist_hist1.npy')).item() #dict
acc = hist['acc']
loss = hist['loss']
val_acc = hist['val_acc']
val_loss = hist['val_loss']



# ########################
# #(2)  Row-wise input   #
# ########################
# Load the data, shuffle and split them into train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(X_train.shape[0], 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 28, 28)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# # Data normalization
# X_means = np.mean(X_train, axis=0)
# X_stds = np.std(X_train, axis=0)
# X_train = (X_train - X_means)/(X_stds+1e-6)
# X_test = (X_test - X_means)/(X_stds+1e-6)
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# nb_classes = 10
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
#
# #learning rate schedule
# learning_rate = 0.001
# import keras
# class decaylr_loss(keras.callbacks.Callback):
#     def __init__(self):
#         super(decaylr_loss, self).__init__()
#     def on_epoch_end(self,epoch,logs={}):
#         loss=logs.items()[1][1]
#         #loss=logs.get('val_loss')
#         print('loss: ', loss)
#         old_lr = 0.001
#         new_lr= old_lr*np.exp(loss)
#         print('New learning rate: ', new_lr)
#         K.set_value(self.model.optimizer.lr, new_lr)
# lrate = decaylr_loss()
# #early stopping
# patience = 50
# earlystopper = EarlyStopping(monitor='val_acc', patience=patience,
#                              verbose=1, mode='max')
# #check point
# from keras.callbacks import ModelCheckpoint
# #filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# filepath = os.path.join(wdir,'model1','LSTM1MNIST2weights.best.hdf5') #LSTM_model1
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
#                              verbose=1,
#                              save_best_only=True, mode='max')
#
# # create a model
# batch_size = 32
# nb_epochs = 200 #increases to 200
# hidden_units = 50 #changed to 50
# model = Sequential()
# consume_less='mem'
# from keras.layers.recurrent import LSTM1   #recurrent_m1.py
# model.add(LSTM1(output_dim=hidden_units,
#                input_shape = (28,28),
#                inner_init='glorot_uniform',
#                forget_bias_init='one',
#                activation='tanh',
#                inner_activation='sigmoid',
#                consume_less=consume_less))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
# rmsprop = RMSprop(lr=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=rmsprop,
#               metrics=['accuracy'])
# model.summary()
#
# #fit and evaluate the model
# lstm1_mnist_hist2 = model.fit(X_train, Y_train,
#                               batch_size=batch_size,
#                               nb_epoch=nb_epochs,
#                               verbose=1,
#                               validation_data=(X_test, Y_test),
#                               callbacks=[earlystopper, lrate, checkpoint])
# scores = model.evaluate(X_test, Y_test, verbose=0)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
#
# savePath = wdir
# model.save(os.path.join(savePath,'LSTM_M1_MNIST2.h5'))
#
# #plot acc and loss vs epochs
# import matplotlib.pyplot as plt
# print(lstm1_mnist_hist2.history.keys())
# #accuracy
# plt.plot(lstm1_mnist_hist2.history['acc'])
# plt.plot(lstm1_mnist_hist2.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.savefig(os.path.join(savePath,'model1','MNIST2_AccVsEpoch.jpeg'),
#             dpi=1000, bbox_inches='tight')
# plt.show()
# #loss
# plt.plot(lstm1_mnist_hist2.history['loss'])
# plt.plot(lstm1_mnist_hist2.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='best')
# plt.savefig(os.path.join(savePath,'model1','MNIST2_LossVsEpoch.jpeg'),
#             dpi=1000, bbox_inches='tight')
# plt.show()
#
# #save training history
# import numpy as np
# np.save(os.path.join(savePath,'model1','lstm1_mnist_hist2.npy'),
#         lstm1_mnist_hist2.history)
# #load
# hist = np.load(os.path.join(savePath,'model1','lstm1_mnist_hist2.npy')).item() #dict
# acc = hist['acc']
# loss = hist['loss']
# val_acc = hist['val_acc']
# val_loss = hist['val_loss']
#
# #######################
# # Check-pointed LSTM  #
# #######################
# model.load_weights(filepath)
# model = Sequential()
# model.add(LSTM1(output_dim=hidden_units,
#                input_shape = (28,28),
#                inner_init='glorot_uniform',
#                forget_bias_init='one',
#                activation='tanh',
#                inner_activation='sigmoid',
#                consume_less=consume_less))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
# rmsprop = RMSprop(lr=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=rmsprop,
#               metrics=['accuracy'])
# scores = model.evaluate(X_test,Y_test,verbose=0)
# print("%s: %.2f%%" %(model.metrics_names[1],scores[1]*100))
