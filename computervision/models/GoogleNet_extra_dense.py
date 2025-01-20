#!/usr/bin/env python
# coding: utf-8

# Implement GoogleNet
# Build an inception module

import functools
import keras

DefaultConv = functools.partial(
    keras.layers.Conv2D, kernel_size=(1, 1), strides=(1, 1),
    padding="same", activation="relu")

class InceptionModule(keras.layers.Layer):
  def __init__(self, filters11, filters33_reduce, filters33,
               filters55_reduce, filters55, filters_pool_proj,
               use_batch_norm=True,**kwargs):
    super().__init__(**kwargs)
    self.conv11 = DefaultConv(filters=filters11)

    self.conv33_reduce = DefaultConv(filters=filters33_reduce)
    self.conv33 = DefaultConv(filters=filters33, kernel_size=(3,3))

    self.conv55_reduce = DefaultConv(filters=filters55_reduce)
    self.conv55 = DefaultConv(filters=filters55, kernel_size=(5,5))

    self.max_pool33 = keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')
    self.conv_pool = DefaultConv(filters=filters_pool_proj)

    self.use_batch_norm = use_batch_norm
    if use_batch_norm:
      self.batch_layer = keras.layers.BatchNormalization()

  def call(self, inputs):
    path1 = self.conv11(inputs)
    path2 = self.conv33(self.conv33_reduce(inputs))
    path3 = self.conv55(self.conv55_reduce(inputs))
    path4 = self.conv_pool(self.max_pool33(inputs))

    iModule = keras.layers.Concatenate(axis=-1)([path1, path2, path3, path4])
    logits = self.batch_layer(iModule) if self.use_batch_norm else iModule

    return logits


DefaultMaxPool = functools.partial(
    keras.layers.MaxPool2D,
    pool_size=(3,3), strides=(2,2), padding="same")

def get_model(modelinfo, **kwargs):
  use_batch_norm = modelinfo['use_batch_norm']
  
  inputs = keras.layers.Input(shape=(modelinfo['dim'],modelinfo['dim'],3))
  hidden = DefaultConv(filters=64, kernel_size=(7,7), strides=(2,2),  padding='same')(inputs)
  if use_batch_norm:
    hidden = keras.layers.BatchNormalization()(hidden)
  hidden = DefaultMaxPool()(hidden)
  hidden = DefaultConv(filters=64)(hidden)
  hidden = DefaultConv(filters=192, kernel_size=(3,3))(hidden)
  if use_batch_norm:
    hidden = keras.layers.BatchNormalization()(hidden)
  hidden = DefaultMaxPool()(hidden)

  hidden = InceptionModule(filters11=64, filters33_reduce=96, filters33=128,
    filters55_reduce=16, filters55=32, filters_pool_proj=32,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = InceptionModule(filters11=128, filters33_reduce=128, filters33=192,
    filters55_reduce=32, filters55=96, filters_pool_proj=64,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = DefaultMaxPool()(hidden)

  hidden = InceptionModule(filters11=192, filters33_reduce=96, filters33=208,
    filters55_reduce=16, filters55=48, filters_pool_proj=64,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = InceptionModule(filters11=160, filters33_reduce=112, filters33=224,
    filters55_reduce=24, filters55=64, filters_pool_proj=64,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = InceptionModule(filters11=128, filters33_reduce=128, filters33=256,
    filters55_reduce=24, filters55=64, filters_pool_proj=64,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = InceptionModule(filters11=112, filters33_reduce=144, filters33=288,
    filters55_reduce=32, filters55=64, filters_pool_proj=64,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = InceptionModule(filters11=256, filters33_reduce=160, filters33=320,
    filters55_reduce=32, filters55=128, filters_pool_proj=128,
    use_batch_norm=use_batch_norm)(hidden)

  hidden = DefaultMaxPool()(hidden)
  hidden = InceptionModule(filters11=256, filters33_reduce=160, filters33=320,
    filters55_reduce=32, filters55=128, filters_pool_proj=128,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = InceptionModule(filters11=384, filters33_reduce=192, filters33=384,
    filters55_reduce=48, filters55=128, filters_pool_proj=128,
    use_batch_norm=use_batch_norm)(hidden)
  hidden = keras.layers.GlobalAveragePooling2D()(hidden)
  hidden = keras.layers.Dropout(0.4)(hidden)
  hidden = keras.layers.Flatten()(hidden)
  hidden = keras.layers.Dense(units=1000, activation="relu")(hidden)
  hidden = keras.layers.Dense(units=500, activation="relu")(hidden)
  y = keras.layers.Dense(units=4, activation='sigmoid', name='xywh')(hidden)
  # y = keras.layers.Dense(units=1, activation='sigmoid', name='y')(hidden)
  # w = keras.layers.Dense(units=1, activation='sigmoid', name='w')(hidden)
  # h = keras.layers.Dense(units=1, activation='sigmoid', name='h')(hidden)

  model = keras.Model(inputs=inputs, outputs=y)

  return model