# -*- coding: utf-8 -*-
# FeedForward
# EncoderBlock
# Embedding with position
# GPTDecoderBlock
# Based on implementation of encoder/decoder in class &
# Based on https://keras.io/examples/vision/object_detection_using_vision_transformer/

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # Override function to avoid error while saving model
    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update(
    #         {
    #             "input_shape": input_shape,
    #             "patch_size": PATCH_SIZE,
    #             "num_patches": num_patches,
    #             "projection_dim": projection_dim,
    #             "num_heads": num_heads,
    #             "transformer_units": transformer_units,
    #             "transformer_layers": transformer_layers,
    #             "mlp_head_units": mlp_head_units,
    #         }
    #     )
    #     return config

    def call(self, patch):
        positions = keras.ops.expand_dims(
            keras.ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

def get_model(modelinfo):
    inputs = keras.Input(shape=(modelinfo['dim'],modelinfo['dim'],3))
    patches = Patches(modelinfo['patch_size'])(inputs)
    num_patches = (modelinfo['dim'] // modelinfo['patch_size']) ** 2
    encoded_patches = PatchEncoder(num_patches, modelinfo['dim_embedding'])(patches)

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

    bounding_box = keras.layers.Dense(4)(features)

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)
