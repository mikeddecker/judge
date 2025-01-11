#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement googlenetsmall
# Build an inception module
# get_ipython().run_line_magic('pip', 'install sqlalchemy')

DIM = 512

import functools
import keras
from helpers import iou

DefaultConv = functools.partial(
    keras.layers.Conv2D, kernel_size=(3, 3), strides=(2, 2),
    padding="same", activation="relu", kernel_initializer="he_normal")

DefaultMaxPool = functools.partial(
    keras.layers.MaxPool2D,
    pool_size=(3,3), strides=(2,2), padding="same")

def get_model(input_shape, num_classes, use_batch_norm=True, **kwargs):
  model = keras.Sequential(**kwargs)
  model.add(keras.layers.Input(shape=input_shape))
  model.add(DefaultConv(filters=24, kernel_size=(5,5), strides=(2,2),  padding='same'))
  if use_batch_norm:
    model.add(keras.layers.BatchNormalization())
  model.add(DefaultMaxPool())
  model.add(keras.layers.Dropout(0.25))
  model.add(DefaultConv(filters=32))
  model.add(DefaultConv(filters=48))
  if use_batch_norm:
    model.add(keras.layers.BatchNormalization())
  model.add(DefaultMaxPool())
  model.add(DefaultConv(filters=64))

  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(units=128, activation="relu"))
  model.add(keras.layers.Dense(units=num_classes, activation='sigmoid'))

  return model


# In[5]:


model = get_model(input_shape=(DIM,DIM,3), num_classes=4, use_batch_norm=True)
model.summary()


# In[6]:




# In[7]:


# Compile the model with IoU loss
# model.compile(optimizer='adam', loss=keras.losses.Huber(), metrics=[iou_loss])
model.compile(optimizer='adam', loss='mse', metrics=[iou])


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from FrameLoader import FrameLoader
from DataGeneratorFrames import DataGeneratorFrames
from DataRepository import DataRepository

repo = DataRepository()
repo.load_relativePaths_of_videos_with_framelabels()

train_generator = DataGeneratorFrames(
    frameloader=FrameLoader(repo),
    train_test_val="train",
    dim=(DIM,DIM),
    batch_size=32,
)

val_generator = DataGeneratorFrames(
    frameloader=FrameLoader(repo),
    train_test_val="test",
    dim=(DIM,DIM),
    batch_size=32,
)


callbacks = [
    ModelCheckpoint('model_best.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
    EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=8,
    callbacks=callbacks,
    verbose=1,
    validation_data=val_generator
)


keras.models.save_model(
    model,
    filepath="randomModel.keras",
    overwrite=True
)

model.save_weights("randomModel.weights.h5")

