import keras
import numpy as np
import sys
import tensorflow as tf
import pandas as pd

sys.path.append('.')

from helpers import iou
from FrameLoader import FrameLoader
from DataGeneratorFrames import DataGeneratorFrames
from DataRepository import DataRepository

from models.GoogleNet import get_model as get_model_googlenet
from models.MobileNetV3Small import get_model as get_model_mobilenet
from models.RandomCNN import get_model as get_model_randomcnn
from models.vitransformer_enc import get_model as get_model_vit

def train_model(model, info_train):
    """Returns history object"""
    DIM = selected_info['dim']

    repo = DataRepository()
    train_generator = DataGeneratorFrames(
        frameloader=FrameLoader(repo),
        train_test_val="train",
        dim=(DIM,DIM),
        batch_size=info_train['batch_size'],
    )
    val_generator = DataGeneratorFrames(
        frameloader=FrameLoader(repo),
        train_test_val="test",
        dim=(DIM,DIM),
        batch_size=info_train['batch_size'],
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint('weights/last_trained_model_best.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    if info_train["early_stopping"]:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1))
    
    optimizer = keras.optimizers.Adam(learning_rate=info_train['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=[iou])

    history = model.fit(
        train_generator,
        epochs=info_train['epochs'],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_generator
    )

    return history

###############################################################################

info_googlenet = {
    'name' : 'googlenet',
    'dim' : 512,
    'batch_size' : 8,
    'learning_rate' : 1e-4,
    'use_batch_norm' : True,
    'get_model_function' : get_model_googlenet,
}
info_vit = {
    'name' : 'vision_transformer',
    'dim' : 224,
    'patch_size' : 16, # (224 / 16) **2 = 196 patches
    'dim_embedding' : 64,
    'num_heads': 4,
    'encoder_blocks': 4,
    'mlp_head_units' : [1024, 256, 64],  # Size of the dense layers
    'batch_size' : 8,
    'min_epochs' : 15,
    'learning_rate' : 1e-3,
    'weight_decay' : 1e-4,
    'get_model_function' : get_model_vit,
}
info_mobilenet = {
    'name' : 'mobilenet',
    'dim' : 224, # pre-trained default
    'batch_size' : 32,
    'min_epochs' : 15,
    'has_frozen_layers' : True,
    'learning_rate' : 1e-3,
    'get_model_function' : get_model_mobilenet,
}

###############################################################################
selected_info = info_mobilenet
# TODO : continue training
# TODO : save model
# TODO : write results
###############################################################################

trainings_info = {
    'epochs' : 2,
    'early_stopping' : False,
    'early_stopping_patience' : 3,
    'batch_size' : selected_info['batch_size'],
    'learning_rate' : 1e-4 if 'learning_rate' not in selected_info.keys() else selected_info['learning_rate'],
}
trainings_info['weight_decay'] = trainings_info['learning_rate'] / 10 if 'weight_decay' not in selected_info.keys() else selected_info['weight_decay']

model = selected_info['get_model_function'](selected_info)
model.summary()

history = train_model(model, info_train=trainings_info)
if 'has_frozen_layers' in selected_info.keys():
    trainings_info['epochs'] = 2
    model.trainable = True
    history = train_model(model, info_train=trainings_info)

print(history)

import matplotlib.pyplot as plt
def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_history("loss")

df_history = pd.DataFrame(history.history)
df_history["epoch"] = df_history.index + 1
df_history