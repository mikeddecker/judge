import keras
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd

from datetime import datetime

sys.path.append('.')

from helpers import iou, my_mse_loss_fn
from FrameLoader import FrameLoader
from DataGeneratorFrames import DataGeneratorFrames
from DataRepository import DataRepository

from models.GoogleNet import get_model as get_model_googlenet
from models.GoogleNet_extra_dense import get_model as get_model_googlenet_extra_dense
from models.MobileNetV3Small import get_model as get_model_mobilenet
from models.RandomCNN import get_model as get_model_randomcnn
from models.vitransformer_enc import get_model as get_model_vit

def train_model(model: keras.Sequential, info_train, from_scratch=True):
    """Returns history object"""
    DIM = selected_info['dim']
    weight_path = f"weights/{selected_info['name']}.weights.h5"
    print(selected_info["name"], os.path.exists(weight_path))
    if not from_scratch and os.path.exists(weight_path):
        print("loading saved weights")
        model.load_weights(weight_path)

    repo = DataRepository()
    train_generator = DataGeneratorFrames(
        frameloader=FrameLoader(repo),
        train_test_val="train",
        dim=(DIM,DIM),
        batch_size=info_train['batch_size'],
    )
    val_generator = DataGeneratorFrames(
        frameloader=FrameLoader(repo),
        train_test_val="val",
        dim=(DIM,DIM),
        batch_size=info_train['batch_size'],
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint('weights/last_trained_model_best.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    if info_train["early_stopping"]:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=trainings_info["restore_best_weights"], verbose=1))
    
    optimizer = keras.optimizers.Adam(learning_rate=info_train['learning_rate'])
    model.compile(optimizer=optimizer, loss=[my_mse_loss_fn], metrics=[iou])

    history = model.fit(
        train_generator,
        epochs=info_train['epochs'],
        callbacks=callbacks,
        verbose=1,
        validation_data=val_generator
    )

    if 'unfreeze_pre_trained_layers_after_training' in info_train.keys():
        pass
        # TODO : save history, add them after next training round

    repo2 = DataRepository() # Ensure connection didn't time out, by creating a new one
    df_history = pd.DataFrame(history.history)
    print(selected_info["name"])
    print(selected_info)
    print(df_history)
    df_history["modelname"] = [selected_info['name'] for _ in range(len(df_history))]
    df_history["train_date"] = [info_train['train_date'] for _ in range(len(df_history))]
    print(df_history)
    last_epoch_nr = int(repo2.get_last_epoch_nr(selected_info['name']))
    print("last_epoch", last_epoch_nr, from_scratch)
    if last_epoch_nr > 0:
        # Return when training from scratch has worse results than last time
        # TODO : make a function to update val_iou based on last validation set
        last_result = repo2.get_last_epoch_values(modelname=selected_info["name"], epoch=last_epoch_nr)
        print('result now', df_history.loc[df_history.index[-1], 'val_iou'])
        print('last result was: ', last_result.loc[0, 'val_iou'])
        if not trainings_info['save_anyway'] and df_history.loc[df_history.index[-1], 'val_iou'] < last_result.loc[0, 'val_iou']:
            print("RESULTS WEREN'T BETTER")
            return df_history

        df_history["epoch"] = df_history.index + 1 + (0 if from_scratch else last_epoch_nr)
    else:
        df_history["epoch"] = df_history.index + 1
    
    model.save_weights(weight_path)
    repo2.save_train_results(df_history, from_scratch=from_scratch)

    return df_history

###############################################################################

info_googlenet = {
    'name' : 'googlenet',
    'dim' : 512,
    'batch_size' : 8,
    'learning_rate' : 1e-4,
    'use_batch_norm' : True,
    'get_model_function' : get_model_googlenet,
}
info_googlenet['name'] = f"googlenet_d{info_googlenet['dim']}"
info_googlenet_extra_dense = {
    'dim' : 512,
    'batch_size' : 8,
    'learning_rate' : 1e-4,
    'use_batch_norm' : True,
    'get_model_function' : get_model_googlenet_extra_dense,
}
info_googlenet_extra_dense['name'] = f"googlenet_extra_dense_d{info_googlenet_extra_dense["dim"]}"
info_vit = {
    'name' : 'vision_transformer',
    'dim' : 240,
    'patch_size' : 12, # (224 / 16) **2 = 196 patches
    'dim_embedding' : 128,
    'num_heads': 4,
    'encoder_blocks': 4,
    'mlp_head_units' : [2048, 1024, 256, 64],  # Size of the dense layers
    'batch_size' : 8,
    'min_epochs' : 15,
    'learning_rate' : 3e-3,
    'weight_decay' : 4e-5,
    'get_model_function' : get_model_vit,
}
info_vit['name'] = f"vision_transformer_d{info_vit['dim']}_p{info_vit['patch_size']}_e{info_vit['dim_embedding']}_nh{info_vit['num_heads']}"
info_ViViT = {
    'video_model' : True,
    'name' : 'vision_transformer',
    'dim' : 224,
    'patch_size' : 14,
    'timesteps' : 8,
    'batch_size' : 1,
    'dim_embedding' : 64,
    'num_heads': 4,
    'encoder_blocks': 4,
    'mlp_head_units' : [2048, 1024, 256, 64],  # Size of the dense layers
    'min_epochs' : 15,
    'learning_rate' : 3e-3,
    'weight_decay' : 4e-5,
    'get_model_function' : get_model_vit,
}
info_ViViT['name'] = f"video_vision_transformer_d{info_ViViT['dim']}_p{info_ViViT['patch_size']}_e{info_ViViT['dim_embedding']}_nh{info_ViViT['num_heads']}"


info_mobilenet = {
    'name' : 'mobilenet',
    'dim' : 224, # pre-trained default
    'batch_size' : 32,
    'min_epochs' : 15,
    'has_frozen_layers' : True,
    'learning_rate' : 8e-3,
    'get_model_function' : get_model_mobilenet,
}

###############################################################################
selected_info = info_ViViT
###############################################################################

trainings_info = {
    'epochs' : 1, # Take more if first train round of random or transformer
    'early_stopping' : True,
    'restore_best_weights' : False,
    'early_stopping_patience' : 6,
    'batch_size' : selected_info['batch_size'],
    'learning_rate' : 8e-4 if 'learning_rate' not in selected_info.keys() else selected_info['learning_rate'],
    'train_date' : datetime.now().strftime("%Y%m%d"),
    'save_anyway' : True,
}
trainings_info['weight_decay'] = trainings_info['learning_rate'] / 20 if 'weight_decay' not in selected_info.keys() else selected_info['weight_decay']

model = selected_info['get_model_function'](selected_info)
model.summary()

history = train_model(model, info_train=trainings_info, from_scratch=True)

print(history)
