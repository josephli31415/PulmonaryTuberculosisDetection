import random
import cv2 as cv
import os
from tensorflow.keras.layers import Reshape, add, Input, Activation, Conv1D, Flatten, Dense, Dropout, GRU, LSTM, Bidirectional, SimpleRNN, Conv2D, MaxPooling2D, Concatenate, BatchNormalization
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, multiply, ZeroPadding2D, AveragePooling2D
import math
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from keras import regularizers
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from confusion_matrix import DrawConfusionMatrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
file_root = 'coughtb/'
#file_list = os.listdir(file_root)
data_list = []
for i in range(1, 1001):
    file_path = file_root+str(i)+'.png'
    spect = cv.imread(filepath)
    data_list.append(spect)
data_x = np.array(data_list)
label = pd.read_csv('experiment2.csv', header=None)
label_y = label.to_numpy().reshape(1000,1)

inputs1 = input(shape = (240, 320, 3))
norm_layer = preprocessing.Normalization
norm_layer.adapt(data_x)
x2 = norm_layer(inputs1)
x = norm_layer(inputs1)
x = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=3, strides=2, padding="SAME")(x)
x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same',activation='relu' )(x)
# x = MaxPooling2D(pool_size=3, strides=2, padding="SAME")(x)
x = Flatten()(x)
x = Dropout(rate = 0.4)(x)
x = Dense(1024, activation='relu')(x)
#print(x.shape)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)

data = np.load('experiment2.npy')
# data = data.reshape(142, -1)
#inputs2 = Input(shape=(837,1))
data = data.reshape (1000, 59, 31)
inputs2 = Input(shape = (59,31))
norm_layer = preprocessing.Normalization()
norm_layer.adapt(data)
x1 = norm_layer(inputs2)
x1 = Birirectional(LSTM(units=32, return_sequences=True))(x1)
x1 = Bidirectional(LSTM(units=64, return_sequences=True))(x1)
x1 = Flatten()(x1)
x1 = Dropout(0.5)(x1)
x1 = Dense(64, activation='relu')(x1)
# x1 = Dense(32, activation='relu')(x1)
# x = Dense(2, activation='softmax')(x)
x1 = BatchNormalization()(x1)
x = Concatenate()([x,x1]) 
x= Dense(64, activation='relu')(x)
x1 = BatchNormalization()(x1)
x = Dense(1, kernel_regularizer=regularizers.l1(0.01),
    bias_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l1_l2(0.01),
    activation='sigmoid')(x)

inputs3 = Input(shape=(2, ))
# x = Concatenate()([x, x1])
# x2 = Dense(2, activation='relu')(inputs3)
x4 = Dense(1, activation='sigmoid')(inputs3)
loss = tf.losses.BinaryCrossentropy()
opts = tf.optimizers.Adam(learning_rate=0.001)
model1 = tf.keras.Model(inputs=[inputs1, inputs2], outputs=x)
model1.summary()
model1.compile(optimizer=opts, loss=loss, metrics=['accuracy'])
divide_rate = 0.8
divide_num = round(len(data_x)*divide_rate)

num = 2000
for i in range(num):
    x = random.randint(0, len(data_x)-1)
    y = random.randint(0, len(data_x)-1)
    data_x[x], data_x[y] = data_x[y], data_x[x]
    data[x], data[y] = data[y], data[x]
    label_y[x], label_y[y] = label_y[y], label_y[x]
train_x1 = data_x[:divide_num]
train_y1 = label_y[:divide_num]
train_x2 = data[:divide_num]
test_x1 = data_x[divide_num:]
test_y1 = label_y[divide_num:]
test_x2 = data[divide_num:]

history1 = model1.fit([train_x1, train_x2], train_y1,epochs=30, batch_size=32)
# history2 = model2.fit(train_x1, train_y1, epochs=100, batch_size=32)
# history = model.fit([train_x1, train_x2], train_y1,epochs=30, batch_size=32)


pre = model1.evaluate([test_X1, test_x2], test_y1)
print('test_loss:', pre[0], '-test_acc:', pre[1])

rtb = 0
rhea = 0
for i in range(0,200):
    if test_y1[i] == 0:
        rtb = rtb + 1
    else:
        rhea = rhea + 1
print(rtb)
print(rhea)
tb = 0
hea = 0
pred = model1.predict([test_x1, test_x2], test_y1)
pred = np.round(pred)
# print(pred)
for i in range(0,200)
    if test_y1[i] == 0 and pred[i] != test_y1[i]:
        tb = tb + 1
    if test_y1[i] == 1 and pred[i] != test_y1[i]:
        hea = hea +1



