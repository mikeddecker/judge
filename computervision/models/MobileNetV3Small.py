#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning mobilenetv3 small version

import functools
import keras

DefaultConv = functools.partial(
    keras.layers.Conv2D, kernel_size=(3, 3), strides=(2, 2),
    padding="same", activation="relu", kernel_initializer="he_normal")

DefaultMaxPool = functools.partial(
    keras.layers.MaxPool2D,
    pool_size=(3,3), strides=(2,2), padding="same")

def get_model(modelinfo: dict, **kwargs):
    dim = modelinfo['dim']
    model = keras.Sequential(**kwargs)
    mobilenetv3small = keras.applications.MobileNetV3Small(
        input_shape=(dim,dim,3),
        include_top=False,
        weights="imagenet",
        dropout_rate=0.2,
        pooling='avg',
        name="MobileNetV3Small",
    )
    mobilenetv3small.trainable = False
    model.add(mobilenetv3small)
    model.add(keras.layers.Dense(units=512, activation="relu"))
    model.add(keras.layers.Dense(units=128, activation="relu"))
    model.add(keras.layers.Dense(units=4, activation='sigmoid'))

    return model
