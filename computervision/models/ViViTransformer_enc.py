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
import sys
sys.path.append('..')
from api.helpers import ConfigHelper


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=keras.activations.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

class Patches(keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = keras.ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = keras.ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
    
class TimePatches(keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = keras.ops.shape(images)
        batch_size = input_shape[0]
        timestep = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        def create_single_timepatch(video_input):
            patches = keras.ops.image.extract_patches(video_input, size=self.patch_size)
            patches = keras.ops.reshape(
                patches,
                (
                    num_patches_h * num_patches_w * timestep,
                    self.patch_size * self.patch_size * channels,
                ),
            )

            return patches
        
        patches = tf.map_fn(create_single_timepatch, images)

        return patches
    
    def compute_output_shape(self, input_shape):
        timesteps = input_shape[1]
        patches_per_timestep = (input_shape[2] / self.patch_size) ** 2
        output_shape = (input_shape[0], int(timesteps * patches_per_timestep), self.patch_size * self.patch_size * input_shape[-1])
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class TimePatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, timesteps, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.timesteps = timesteps
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches*timesteps, output_dim=projection_dim
        )

    def call(self, patch):
        positions = keras.ops.expand_dims(
            keras.ops.arange(start=0, stop=self.num_patches*self.timesteps, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

def get_model(modelinfo, df_table_counts: pd.DataFrame):
    inputs = keras.Input(shape=(modelinfo['timesteps'],modelinfo['dim'],modelinfo['dim'],3))
    patches = TimePatches(modelinfo['patch_size'])(inputs)
    num_patches = (modelinfo['dim'] // modelinfo['patch_size']) ** 2
    encoded_patches = TimePatchEncoder(num_patches, modelinfo['timesteps'], modelinfo['dim_embedding'])(patches)
    print("shape of encoded_patches", encoded_patches.shape)

    # Create multiple layers of the Transformer block.
    for _ in range(modelinfo['encoder_blocks']):
        # Layer normalization 1.
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=modelinfo['num_heads'], key_dim=modelinfo['dim_embedding'], dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=[modelinfo['dim_embedding'] ** 2, modelinfo['dim_embedding']], dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.3)(representation)

    features = mlp(representation, hidden_units=modelinfo['mlp_head_units'], dropout_rate=0.3)


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
