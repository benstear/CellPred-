#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:51:16 2018

@author: dawnstear
"""
import os
#s.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow import contrib
import numpy as np

from sklearn import cross_validation #, model_selection
from sklearn.utils import shuffle
import matplotlib as plt
from matplotlib import pyplot

tf.logging.set_verbosity(tf.logging.DEBUG)
# In order of ascending severity, they are DEBUG, INFO, WARN, ERROR, and FATAL
    
cellcount, genecount = np.shape(data)
data_shuffled = shuffle(data)

X_ = data_shuffled.drop('Labels',axis=1)
X_ = data_shuffled.drop('TYPE',axis=1)
y_ = data_shuffled['Labels']  # X_ and y_ are used b/c X and y are also name of placeholders in TF... 

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_.values,y_.values,test_size=0.2)
                                                                # pass in X_.values so its a numpy.ndarray not dframe 

#Define number of neurons per layer and batch size
batch_size = 30
num_input = genecount-1 # **** must preserve gene (feature) order when inputing data
n_hidden_1 = 512   # num_input = genecount-1 bc we have to take out one column (TYPE) bc its not needed
n_hidden_2 = 256
n_hidden_3 = 64
n_hidden_4 = 256
n_hidden_5 = 128
num_classes = len(celldistro) # specify how many cell types (labels in dataset)

# Create placeholders for train X,y and test X,y. Shape = (None, n_inputs)...
# bc we dont know how many samples we will have per batch yet
X = tf.placeholder(tf.float32, shape=(batch_size,num_input),name="X") # use -1, not None for dynamic batch size
y = tf.placeholder(tf.int64, shape=(batch_size), name="y")

# Build Architecture
with tf.name_scope("dnn"):
    layer1 = tf.layers.dense(inputs=X, units=n_hidden_1, activation=tf.nn.elu,name='Layer_1',reuse=True) # add names?
    dropout1 = tf.layers.dropout(inputs=layer1, rate=0.3,name='Dropout_1') # use Estimator, not ModeKeys
    layer2 = tf.layers.dense(inputs=dropout1,units=n_hidden_2, activation=tf.nn.elu,name='Layer_2',reuse=True)
    dropout2 = tf.layers.dropout(inputs=layer2, rate=0.3,name='Dropout_2')
    layer3 = tf.layers.dense(inputs=dropout2, units=n_hidden_3, activation=tf.nn.elu,name='Layer_3',reuse=True)
    
    output = tf.layers.dense(inputs=layer3,units=num_classes,activation=tf.nn.softmax,name='Output',reuse=True)
    #output_tensor = tf.Print(output,[output])
    
######### Cost fn = softmax xentropy
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output) # check sparse_xentropy docs
    loss = tf.reduce_mean(xentropy,name="loss")           # y or y_ ^^^

######### Optimizer = SGD    
lr=0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(lr)
    training_op = optimizer.minimize(loss)
    
### Now how to evaluate...
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(output,y,1)  # check top_k docs
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

# *------End of Construction phase -------*   
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# os.path.join(file, extension)   -writer for tensorboard viz
writer = tf.summary.FileWriter("/Users/dawnstear/desktop/tmp")

# *--------Set Some hyperparameters & initialize Metic Vectors --------*
n_epochs = 28
epochvec = range(1,n_epochs+1)
accTrain = []
accVal = []

def next_batch(num, data, labels):  # from stack overflow
    ''' Return a total of `num` random samples and labels. '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
'''
  TODO:
      
 -check that labels were applied correctly    
 -next batch method, ok
 -use coo sparse dframe/matrix
 -softmax logits needs labels to be ints 0-9 so change labels to 0 through 9  ?   
 -change shuffle method    
 -save trainedmodelwith date and time title 
 -implement Tensorboard
 -Implement multithreading for cluster
 - implement pie chart of cell types and Venn diagram
 - plot accuracy as function of n epochs or n genes
 -kfold and randomize, labelencode....keras'''
    
# *--------------- Run Session -------------------*
with tf.Session() as sess:
    init.run() # sess.run(init)  ??
    print('Just initialized')
    for epoch in range(n_epochs):
        print('Starting Epoch %d ' % epoch)
        for iteration in range(cellcount//batch_size): # examples = 55,000
            
            X_batch, y_batch = next_batch(batch_size, X_train, y_train)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
       # acc_val = accuracy.eval(feed_dict={X:X_test, y:y_test})
        accTrain = np.append(accTrain,acc_train)
      #print('Epoch %d completed' % epoch)
      #  accVal = np.append(accVal,acc_val) 
      #  print('Epoch:%d - Train Acc :%f - Validation Acc: %f' % (epoch, acc_train,acc_val)) 
    sess.close()  
    
    
    
writer.add_graph(sess.graph) # outside session


# --------------- Plot Metrics -------------------
fig, ax = pyplot.subplots()
ax.plot(epochvec,accTrain)
#ax.plot(epochvec,accVal)
ax.set(xlabel='Epoch', ylabel='Accuracies',
       title='Accuracy Over Training Phase') # include time, epoch and batch size in title
ax.grid()
fig.savefig("/Users/dawnstear/desktop/tmp/")


#tf.trainable_variables(scope=None)

