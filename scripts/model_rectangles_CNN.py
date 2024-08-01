#!/usr/bin/env python
# coding: utf-8

# # keras CNN predict rectangle box

# In[1]:


# !pip3 install tensorflow


# In[2]:


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

# Suppres warnings from positioning like
# [h264 @ 0x56bf4fb5da40] reference picture missing during reorder
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"


# In[19]:


models = [
    'frames_skillborder_CNN_model_96pixels_history'
]
model_name = models[0]


# In[ ]:





# In[21]:


model = pickle_load_or_create('../models/frames_skillborder_CNN_model_96pixels_history', lambda: None, True)
print(model)


# In[22]:


from DataGeneratorFrames import DataGeneratorRectangles


# In[24]:


config = pickle_load_or_create(model_name, lambda:{
    'convolution': (3,3),
    'dim':256,
    'rgb':True,
    'unique_labels': 3,
}, config=True)
config


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, TimeDistributed, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam

if model is None:
    model = Sequential()
    model.add(Conv2D(filters=24, kernel_size=config['convolution'],
                     input_shape=(config['dim'], config['dim'], 3 if config['rgb'] else 1)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=48, kernel_size=(3, 3)))
    model.add(MaxPool2D())
    model.add(BatchNormalization())
    
    model.add(Flatten())  # Flatten each frame
    model.add(Dense(64, activation='relu'))
    model.add(Dense(config['unique_labels'], activation='linear'))
    
    model.compile(optimizer=Adam(), 
                  loss='mean_absolute_error', 
                  metrics=['mean_absolute_error', 'mean_squared_error'])
else:
    model = model.model


# In[26]:


model.summary()


# In[ ]:





# In[ ]:





# In[27]:


# Parameters
params = {'dim': (config['dim'],config['dim']),
          'n_classes': config['unique_labels'],
          'n_channels': 3 if config['rgb'] else 1,
          'shuffle': True,
}

training_generator = DataGeneratorRectangles(train=True, **params)
test_generator = DataGeneratorRectangles(train=False, **params)


# In[28]:


training_generator.batch_order


# In[ ]:





# In[29]:


get_ipython().run_cell_magic('time', '', 'X, y = training_generator.__getitem__(25)\n')


# In[30]:


X.shape


# In[ ]:





# In[31]:


y.shape


# In[32]:


y[:5]


# In[ ]:





# In[33]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(training_generator, epochs=3,\n                    validation_data=test_generator, shuffle=False)\n')


# In[34]:


pd.DataFrame(history.history)


# In[50]:


get_ipython().run_line_magic('pinfo', 'model.fit')


# In[36]:


with open(f"../models/{model_name}.pkl", 'wb') as handle:
    pickle.dump(history, handle)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




