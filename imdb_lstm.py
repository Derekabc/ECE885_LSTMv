# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 09:01:02 2016
ECE802-608 Final Project: Modification of classic LSTM
Case 1): no input signals, keep bias and hidden units


"""

# work directory
import os
wdir = '/home/dong/Documents/MSU/Courses/ECE885/Modified-LSTM-master/'
os.chdir(wdir)


###############################################################################
##                              Test2: IMDB Data                             ##
###############################################################################

import os
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from keras.datasets import imdb
from keras import backend as K
import matplotlib.pyplot as plt
from slim21 import LSTMs

np.random.seed(1337)  # for reproducibility

#load data
print('Loading data...')
max_features = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
#data sequence
print('Pad sequences (samples x time)')
from keras.preprocessing import sequence
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# build a LSTM model

#callback control
learning_rate = 0.00001

# class decaylr_loss(keras.callbacks.Callback):
#     def __init__(self):
#         super(decaylr_loss, self).__init__()
#     def on_epoch_end(self,epoch,logs={}):
#         loss=list(logs.items())[1][1]
#         #loss = logs.get('val_loss')
#         print('loss: ', loss)
#         old_lr = 0.0001
#         new_lr= old_lr*np.exp(loss)
#         print('New learning rate: ', new_lr)
#         K.set_value(self.model.optimizer.lr, new_lr)

# lrate = decaylr_loss()
earlystopper = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
filepath = os.path.join(wdir,'model2','LSTM2IMDBweights.best.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True,mode='max')

hidden_units = 128
batch_size = 32
nb_epoch = 100
print('Build model...')
LSTM2 = LSTMs(implementation=1, units=hidden_units,
              activation='sigmoid',
              input_shape=(784,1), model='LSTM6')

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM2)
model.add(Dense(1, activation='sigmoid'))
rmsprop = RMSprop(lr=learning_rate)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])
model.summary()

#fit and evaluate the model
lstm2_imdb_hist = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, y_test),
                            callbacks=[earlystopper, checkpoint])

A = lstm2_imdb_hist.history['acc']
B = lstm2_imdb_hist.history['val_acc']
C = lstm2_imdb_hist.history['loss']
D = lstm2_imdb_hist.history['val_loss']
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

savePath = wdir
model.save(os.path.join(savePath,'LSTM_M2_IMDB.h5'))

#plot acc and loss vs epochs
print(lstm2_imdb_hist.history.keys())
#accuracy
plt.plot(lstm2_imdb_hist.history['acc'])
plt.plot(lstm2_imdb_hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(savePath,'model2','IMDB_AccVsEpoch.jpeg'),
            dpi=1000, bbox_inches='tight')
plt.show()

#loss
plt.plot(lstm2_imdb_hist.history['loss'])
plt.plot(lstm2_imdb_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(savePath,'model2','IMDB_LossVsEpoch.jpeg'),
            dpi=1000, bbox_inches='tight')
plt.show()


#
# #save training history
# import numpy as np
# np.save(os.path.join(savePath,'model1','lstm1_imdb_hist.npy'),
#         lstm1_imdb_hist.history)
# #load
# hist = np.load(os.path.join(savePath,'model1','lstm1_imdb_hist.npy')).item() #dict
# acc = hist['acc']
# loss = hist['loss']
# val_acc = hist['val_acc']
# val_loss = hist['val_loss']
# #figure again
# plt.plot(acc)
# plt.plot(val_acc)
# plt.title('model accuracy',fontsize=20)
# plt.ylabel('accuracy',fontsize=18)
# plt.xlabel('epoch',fontsize=18)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(['train', 'test'], loc='best',prop={'size':15})











