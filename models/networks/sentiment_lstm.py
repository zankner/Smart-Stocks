#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
tf.keras.backend.clear_session()

import numpy as np

import sys
sys.path.insert(0,'pre')
from pipeline import *



# In[2]:

train_ds = input_fn('data/train_input_paths.txt','data/train_output_paths.txt',{'batch_size':32,'buffer_size':5})
test_ds = input_fn('data/test_input_path.txt','data/test_output_path.txt',{'batch_size':32,'buffer_size':5})

# In[11]:



embedding_dim = 300
num_features = 6120
save_path = 'saved_models/word2vec_model'
EPOCHS = 50 


# In[3]:


class word_2_vec(Model):
    def __init__(self):
        super(word_2_vec, self).__init__()
        self.dense_1 = Dense(embedding_dim)
        self.dense_2 = Dense(num_features,activation='softmax')
        
    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

model = word_2_vec()


# In[4]:


loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


# In[5]:


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


# In[7]:


@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)


# In [8]:

@tf.function
def test_step(test_inputs, test_labels):
  test_predictions = model(test_inputs)
  t_loss = loss_object(test_labels, test_predictions)
  test_loss(t_loss)
  test_accuracy(test_labels, test_predictions)


# In [9]:

for epoch in range(EPOCHS):
    for inputs,labels in train_ds:
        train_step(inputs,labels)
    
    for test_inputs,test_labels in test_ds:
        test_step(test_inputs,test_labels)
   
    template = 'Epoch: {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'

    print(template.format(epoch+1,train_loss.result(),train_accuracy.result()*100,test_loss.result(),test_accuracy.result()*100))

model.save_weights(save_path,save_format='tf')

new_model = word_2_vec()
new_model.load_weights(save_path)
