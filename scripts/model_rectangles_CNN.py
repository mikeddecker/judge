#!/usr/bin/env python
# coding: utf-8

# # keras CNN predict rectangle box

# In[ ]:


# !pip3 install tensorflow


# In[ ]:


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


# In[ ]:


models = [
    'rectangles_poging_3'
]
model_name = models[0]


# In[ ]:


root = '/media/miked/Elements/Judge/FINISHED-DB-READY/'


# In[ ]:


model = pickle_load_or_create(f"../models/{model_name}", lambda: None, True)
print(model)


# In[ ]:


from DataGeneratorFrames import DataGeneratorRectangles


# In[ ]:


config = pickle_load_or_create(model_name, lambda:{
    'convolution': (3,3),
    'dim':256,
    'rgb':True,
    'unique_labels': 3,
}, config=True)
config


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, TimeDistributed, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam

if model is None:
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=config['convolution'],
                     input_shape=(config['dim'], config['dim'], 3 if config['rgb'] else 1),
                     activation='sigmoid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='sigmoid'))
    model.add(MaxPool2D())
    model.add(BatchNormalization())

    model.add(Flatten())  # Flatten each frame
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(config['unique_labels'], activation='sigmoid'))
    
    model.compile(optimizer=Adam(), 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
else:
    model = model.model


# In[ ]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:


# Parameters
params = {'dim': (config['dim'],config['dim']),
          'n_classes': config['unique_labels'],
          'n_channels': 3 if config['rgb'] else 1,
          'shuffle': True,
}

training_generator = DataGeneratorRectangles(rootfolder=root, batch_size=8, train=True, **params)
test_generator = DataGeneratorRectangles(rootfolder=root, batch_size=8, train=False, **params)


# In[ ]:


training_generator.batch_order


# In[ ]:


X, y = training_generator.__getitem__(25)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)


# In[ ]:


history = model.fit(training_generator, epochs=1,
                    validation_data=test_generator, shuffle=False,
                    callbacks=[early_stopping, reduce_lr])


# In[ ]:


pd.DataFrame(history.history)


# In[ ]:





# In[ ]:


with open(f"../models/{model_name}.pkl", 'wb') as handle:
    pickle.dump(history, handle)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




