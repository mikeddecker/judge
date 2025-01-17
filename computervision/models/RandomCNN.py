#!/usr/bin/env python
# coding: utf-8

# Implement googlenetsmall
# Build an inception module

DIM = 384

import functools
import keras

DefaultConv = functools.partial(
    keras.layers.Conv2D, kernel_size=(3, 3), strides=(2, 2),
    padding="same", activation="relu", kernel_initializer="he_normal")

DefaultMaxPool = functools.partial(
    keras.layers.MaxPool2D,
    pool_size=(3,3), strides=(2,2), padding="same")

def get_model(input_shape, num_classes, use_batch_norm=True, **kwargs):
    model = keras.Sequential(**kwargs)
    model.add(keras.layers.Input(shape=input_shape))
    model.add(DefaultConv(filters=32, kernel_size=(5,5), strides=(1,1)))
    model.add(keras.layers.Dropout(0.05))
    model.add(DefaultConv(filters=48, kernel_size=(5,5), strides=(2,2),  padding='same'))
    if use_batch_norm:
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.05))
    model.add(DefaultMaxPool())
    model.add(DefaultConv(filters=48, strides=(1)))
    model.add(DefaultConv(filters=64))
    model.add(DefaultConv(filters=96, strides=(1)))
    model.add(DefaultConv(filters=96))
    if use_batch_norm:
        model.add(keras.layers.BatchNormalization())
    model.add(DefaultMaxPool())
    model.add(keras.layers.Dropout(0.05))
    model.add(DefaultConv(filters=128))

    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation="relu"))
    model.add(keras.layers.Dense(units=num_classes, activation='sigmoid'))

    return model
