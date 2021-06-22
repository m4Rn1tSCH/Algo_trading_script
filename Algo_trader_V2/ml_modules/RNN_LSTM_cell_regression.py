# -*- coding: utf-8 -*-
"""
Created on 6/22/2021; 12:17 PM

@author: Bill Jaenke
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:18:48 2020

@author: bill-
"""
# RNN with Two Layers to predict a single numerical value
# does not reuse its own states
# uses multivariate data with time windows

import pandas as pd
import numpy as np
from datetime import datetime as dt
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column, data
from tensorflow.keras import Model, layers, regularizers

'''
RNN Regression
single-step and multi-step model for a recurrent neural network
'''
# Pie chart States - works
state_ct = Counter(list(df['state']))
# The * operator can be used in conjunction with zip() to unzip the list.
labels, values=zip(*state_ct.items())
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
fig1, ax = plt.subplots(figsize=(20, 12))
ax.pie(values, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')
#ax.title('Transaction locations of user {df[unique_mem_id][0]}')
ax.legend(loc='center right')
plt.show()

# Pie chart transaction type -works
trans_ct = Counter(list(df['transaction_category_name']))
# The * operator can be used in conjunction with zip() to unzip the list.
labels_2, values_2=zip(*trans_ct.items())
#Pie chart, where the slices will be ordered and plotted counter-clockwise:
fig1, ax = plt.subplots(figsize=(20, 12))
ax.pie(values_2, labels=labels_2, autopct='%1.1f%%', shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')
#ax.title('Transaction categories of user {df[unique_mem_id][0]}')
ax.legend(loc='center right')
plt.show()

ax_desc=df['description'].astype('int64', errors='ignore')
ax_amount=df['amount'].astype('int64', errors='ignore')
sns.pairplot(df)
sns.boxplot(x=ax_desc, y=ax_amount)
sns.heatmap(df)

print("TF-version:", tf.__version__)
# sns.pairplot(df[['amount', 'amount_mean_lag7', 'amount_std_lag7']])


TRAIN_SPLIT = 750

# normalize the training set; but not yet split up
train_dataset = dataset[:TRAIN_SPLIT]
train_ds_norm = tf.keras.utils.normalize(train_dataset)
val_ds_norm = tf.keras.utils.normalize(dataset[TRAIN_SPLIT:])

# train dataset is already shortened and normalized
y_train_multi = train_ds_norm.pop('amount_mean_lag7')
X_train_multi = train_ds_norm[:TRAIN_SPLIT]
# referring to previous dataset; second slice becomes validation data until end of the data
y_val_multi = val_ds_norm.pop('amount_mean_lag7')
X_val_multi = val_ds_norm

print("Shape y_training:", y_train_multi.shape)
print("Shape X_training:", X_train_multi.shape)
print("Shape y_validation:", y_val_multi.shape)
print("Shape X_validation:", X_val_multi.shape)

# buffer_size can be equivalent to the entire length of the df; that way all of it is being shuffled
BUFFER_SIZE = len(train_dataset)

# Batch refers to the chunk of the dataset that is used for validating the predictions
BATCH_SIZE = 21

# size of data chunk that is fed per time period
# weekly expenses are the label; one week's expenses are fed to the layer
timestep = 7

# pass as tuples to convert to tensor slices
#   if pandas dfs fed --> .values to retain rectangular shape and avoid ragged tensors
#   if 2 separate df slices (X/y) fed --> no .values and reshaping needed

# turn the variables into arrays; convert to:
# (X= batch_size(examples), Y=timesteps, Z=features)

# training dataframe
X_train_multi = np.array(X_train_multi)
X_train_multi = np.reshape(X_train_multi, (X_train_multi.shape[0], 1, X_train_multi.shape[1]))
train_data_multi = tf.data.Dataset.from_tensor_slices((X_train_multi, y_train_multi))
# generation of batches
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False).repeat()

# validation dataframe
X_val_multi = np.array(X_val_multi)
X_val_multi = np.reshape(X_val_multi, (X_val_multi.shape[0], 1, X_val_multi.shape[1]))
val_data_multi = tf.data.Dataset.from_tensor_slices((X_val_multi, y_val_multi))
# generation of batches
val_data_multi = val_data_multi.batch(BATCH_SIZE, drop_remainder=False).repeat()

# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (None, 1, 21) (height is treated like time).
# stateful=True to reuse weights of one step
input_dim = 28

units = 128
output_size = 1

# Build the RNN model
def build_model(allow_cudnn_kernel=True):

    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = tf.keras.layers.LSTM(units, stateful=True, input_shape=(None, X_train_multi.shape[2]))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units, stateful=True),
                                         input_shape=(None, input_dim))

    model = tf.keras.models.Sequential([lstm_layer,
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(output_size)
                                        ])
    return model


model = build_model(allow_cudnn_kernel=True)

model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mae'])

# fix divisibility problem (sample size to steps per epoch)
# model.fit(X_train_multi,y_train_multi,
#           validation_data=(X_val_multi, y_val_multi),
#           epochs=250,
#           # evaluation steps need to consume all samples without remainder
#           steps_per_epoch=125,
#           validation_steps=250)

model.fit(train_data_multi, epochs=250,
          steps_per_epoch=125,
          # evaluation steps need to consume all samples without remainder
          validation_data=val_data_multi,
          validation_steps=125)
