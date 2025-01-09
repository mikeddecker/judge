#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning mobilenetv3 small version

# In[1]:


DIM = 512


# In[2]:


import functools
import keras
from helpers import iou
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from FrameLoader import FrameLoader
from DataGeneratorFrames import DataGeneratorFrames
from DataRepository import DataRepository


# In[3]:


DefaultConv = functools.partial(
    keras.layers.Conv2D, kernel_size=(3, 3), strides=(2, 2),
    padding="same", activation="relu", kernel_initializer="he_normal")

DefaultMaxPool = functools.partial(
    keras.layers.MaxPool2D,
    pool_size=(3,3), strides=(2,2), padding="same")


# In[4]:


mobilenetv3small = keras.applications.MobileNetV3Small(
input_shape=(DIM,DIM,3),
include_top=False,
weights="imagenet",
dropout_rate=0.2,
pooling='avg',
name="MobileNetV3Small",
)
mobilenetv3small.summary()


# In[5]:


def get_model(input_shape, num_classes, use_batch_norm=True, **kwargs):
  model = keras.Sequential(**kwargs)
  mobilenetv3small = keras.applications.MobileNetV3Small(
    input_shape=(DIM,DIM,3),
    include_top=False,
    weights="imagenet",
    dropout_rate=0.2,
    pooling='avg',
    name="MobileNetV3Small",
  )
  mobilenetv3small.trainable = False
  model.add(mobilenetv3small)
  model.add(keras.layers.Dense(units=512, activation="relu"))
  model.add(keras.layers.Dense(units=num_classes, activation='sigmoid'))

  return model


# In[6]:


model = get_model(input_shape=(DIM,DIM,3), num_classes=4, use_batch_norm=True)
model.summary()


# In[7]:


model.compile(optimizer='adam', loss='mse', metrics=[iou])


# In[9]:


repo = DataRepository()

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


# In[ ]:


callbacks = [
    ModelCheckpoint('model_best.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
    EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=10,
    callbacks=callbacks,
    verbose=1,
    validation_data=val_generator
)


# In[ ]:


keras.models.save_model(
    model,
    filepath="mobilenetv3small.keras",
    overwrite=True
)

model.save_weights("mobilenetv3small.weights.h5")

