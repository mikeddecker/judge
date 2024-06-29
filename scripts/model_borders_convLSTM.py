#!/usr/bin/env python
# coding: utf-8

# In[1]:


# keras CNN predict air, between, ground or not jumping


# In[2]:


# !pip3 install tensorflow


# In[3]:


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle
from utils_misc import pickle_load_or_create


# In[4]:


from DataGeneratorBordersDB import DataGeneratorSkillBorders


# In[5]:


config = pickle_load_or_create('convLSTM_192_3x3_rgb', lambda:{
    'convolution': (3,3),
    'time_length':8,
    'dim':128,
    'rgb':True,
    'unique_labels': 4,
}, config=True)
config


# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(ConvLSTM2D(filters=64, kernel_size=config['convolution'],
                     input_shape=(config['time_length'], config['dim'],  config['dim'], 3 if config['rgb'] else 1),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())

# Uncomment if you want to add more ConvLSTM2D layers
# model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
#                      padding='same', return_sequences=True))
# model.add(BatchNormalization())

# Apply TimeDistributed Dense layer to each time step
model.add(TimeDistributed(Flatten()))  # Flatten each frame
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(TimeDistributed(Dense(config['unique_labels'], activation='softmax')))

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[7]:


model.summary()


# In[ ]:





# In[ ]:





# In[8]:


# Parameters
params = {'dim': (config['dim'],config['dim']),
          'time_length': config['time_length'],
          'n_classes': config['unique_labels'],
          'n_channels': 3 if config['rgb'] else 1,
          'shuffle': True,
}

training_generator = DataGeneratorSkillBorders(train=True, **params)
test_generator = DataGeneratorSkillBorders(train=False, **params)


# In[9]:


training_generator.batch_order


# In[ ]:





# In[10]:


get_ipython().run_cell_magic('time', '', 'X, y = training_generator.__getitem__(25)\n')


# In[11]:


X.shape


# In[12]:


X


# In[13]:


y.shape


# In[14]:


y


# In[15]:


y.shape


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(training_generator, epochs=2, 
                    validation_data=test_generator, shuffle=False)


# In[ ]:





# In[ ]:


pd.DataFrame(history.history)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'model.fit')


# In[ ]:


with open('../models/frames_skillborder_convLSTM_model_history.pkl', 'wb') as handle:
    pickle.dump(history, handle)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




