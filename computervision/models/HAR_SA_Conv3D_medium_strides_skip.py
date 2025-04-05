# -*- coding: utf-8 -*-
# FeedForward
# EncoderBlock
# Embedding with position
# GPTDecoderBlock
# Based on implementation of encoder/decoder in class &
# Based on https://keras.io/examples/vision/object_detection_using_vision_transformer/

import keras
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Layer, ConvLSTM2D, ConvLSTM3D, Conv2D, Conv3D, Multiply, Add, Activation, TimeDistributed, Flatten, Dense
from tensorflow.keras import backend as K

import sys
sys.path.append('..')
from api.helpers import ConfigHelper

class SelfAttention(Layer):
    """Self-attention layer for ConvLSTM"""
    def __init__(self, kernel_size=1, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        # Input shape: (batch_size, timesteps, height, width, channels)
        self.query = Conv2D(input_shape[-1], kernel_size=self.kernel_size, 
                           padding='same', use_bias=False)
        self.key = Conv2D(input_shape[-1], kernel_size=self.kernel_size, 
                         padding='same', use_bias=False)
        self.value = Conv2D(input_shape[-1], kernel_size=self.kernel_size, 
                           padding='same', use_bias=False)
        self.gamma = self.add_weight(name='gamma', shape=[1], 
                                    initializer='zeros', trainable=True)
        super(SelfAttention, self).build(input_shape)
        
    def call(self, x):
        # x shape: (batch_size, timesteps, height, width, channels)
        batch_size, timesteps, h, w, c = K.int_shape(x)
        
        # Reshape to apply 2D convs on spatial dimensions
        x_reshaped = K.reshape(x, (-1, h, w, c))
        
        # Project to query, key, value
        q = self.query(x_reshaped)
        k = self.key(x_reshaped)
        v = self.value(x_reshaped)
        
        # Reshape back to include timesteps
        q = K.reshape(q, (-1, timesteps, h, w, c))
        k = K.reshape(k, (-1, timesteps, h, w, c))
        v = K.reshape(v, (-1, timesteps, h, w, c))
        
        # Compute attention scores
        attn_scores = tf.einsum('bthwc,bthwc->bthw', q, k)
        attn_scores = Activation('softmax')(attn_scores)
        
        # Apply attention to values
        out = tf.einsum('bthw,bthwc->bthwc', attn_scores, v)
        
        # Add skip connection and learnable weight
        out = self.gamma * out + x
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape

def get_model(modelinfo, df_table_counts: pd.DataFrame):
    """Build a Self-Attention ConvLSTM model"""
    input_shape = (modelinfo['timesteps'],modelinfo['dim'],modelinfo['dim'],3)
    kernel_size = 3
    filters = 32
    inputs = tf.keras.Input(shape=input_shape)
    
    # ConvLSTM layers with self-attention
    x = Conv3D(filters=filters, kernel_size=kernel_size,
                  padding='same', activation="relu")(inputs)
    x = Conv3D(filters=int(filters * 1.5), kernel_size=kernel_size,
                  padding='same', activation="relu")(inputs)
    x = SelfAttention()(x)

    x = Conv3D(filters=int(filters * 2), kernel_size=kernel_size,
                  padding='same', activation="relu")(x)
    
    skip = x
     
    x = Conv3D(filters=int(filters * 2), kernel_size=kernel_size,
                  padding='same', activation="relu")(x)
    
    x = SelfAttention()(x) + skip

    x = Conv3D(filters=int(filters * 3), kernel_size=kernel_size, strides=(1,2,2), padding='same', activation="relu")(x)    
    x = Conv3D(filters=int(filters * 4), kernel_size=kernel_size, strides=1, padding='same', activation="relu")(x)

    x = skip = SelfAttention()(x)

    x = Conv3D(filters=int(filters * 4), kernel_size=kernel_size, strides=1, padding='same', activation="relu")(x)    
    x = Conv3D(filters=int(filters * 4), kernel_size=kernel_size, strides=1, padding='same', activation="relu")(x)

    x = SelfAttention()(x) + skip
    
    x = Conv3D(filters=int(filters * 6), kernel_size=kernel_size, strides=2, padding='same', activation="relu")(x)
    
    # Additional processing
    x = Conv3D(filters=int(filters*6), kernel_size=kernel_size, strides=1, activation='relu', padding='same')(x)
    x = Conv3D(filters=int(filters*8), kernel_size=kernel_size, strides=(1,2,2), activation='relu', padding='same')(x)
    x = Conv3D(filters=int(filters*8), kernel_size=1, strides=2, activation='relu')(x)
    x = Flatten()(x)
    # x = Dense(512, activation='softmax')(x)
    features = Dense(256, activation='softmax')(x)

    # Output layer
    dd_config = ConfigHelper.get_discipline_DoubleDutch_config()
    outputs = {}
    for key, value in dd_config.items():
        if key == "Tablename":
            continue
        if value[0] == "Categorical":
            tablename = "skill"
            match (key):
                case 'Skill':
                    tablename = 'skills'
                case 'Turner1' | 'Turner2':
                    tablename = "turners"
                case 'Type':
                    tablename = 'types'
            classes = int(df_table_counts.iloc[0][tablename])
            outputs[key] = keras.layers.Dense(classes, activation='softmax', name=key)(features)
        else:
            outputs[key] = keras.layers.Dense(1, activation='sigmoid', name=key)(features)

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=outputs)