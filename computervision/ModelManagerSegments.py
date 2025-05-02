import keras
import numpy as np
import os
import sys
import tensorflow as tf
import pandas as pd

from datetime import datetime

sys.path.append('.')

from helpers import iou, my_mse_loss_fn, metric_mse_segmentation_close_accuracy, off_by_0_1, off_by_0_2, off_by_0_33
from managers.FrameLoader import FrameLoader
from managers.DataGeneratorSegmentation import DataGeneratorSegmentation
from managers.DataRepository import DataRepository

sys.path.append('..')
from api.helpers.ConfigHelper import get_discipline_DoubleDutch_config

from models.GoogleNet import get_model as get_model_googlenet
from models.GoogleNet_extra_dense import get_model as get_model_googlenet_extra_dense
from models.MobileNetV3Small import get_model as get_model_mobilenet
from models.RandomCNN import get_model as get_model_randomcnn
from models.vitransformer_enc import get_model as get_model_vit
from models.ViViTransformer_enc_segmentation import get_model as get_model_ViViT


def train_model(model: keras.Sequential, info_train, from_scratch=True):
    """Returns history object"""
    DIM = selected_info['dim']
    weight_path = f"weights/{selected_info['name']}.weights.h5"
    print(selected_info["name"], os.path.exists(weight_path))
    if not from_scratch and os.path.exists(weight_path):
        print("loading saved weights")
        model.load_weights(weight_path)

    repo = DataRepository()
    train_generator = DataGeneratorSegmentation(
        frameloader=FrameLoader(repo),
        train_test_val="train",
        dim=(DIM,DIM),
        timesteps=info_train['timesteps'],
        batch_size=info_train['batch_size'],
    )
    val_generator = DataGeneratorSegmentation(
        frameloader=FrameLoader(repo),
        train_test_val="val",
        dim=(DIM,DIM),
        timesteps=info_train['timesteps'],
        batch_size=info_train['batch_size'],
    )
    
    callbacks = []
    callbacks = [
        keras.callbacks.ModelCheckpoint('weights/last_trained_segmentation_model_best.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ]
    if info_train["early_stopping"]:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=trainings_info["restore_best_weights"], verbose=1))

    # "binary_crossentropy"
    optimizer = keras.optimizers.AdamW(learning_rate=info_train['learning_rate'])
    model.compile(optimizer=optimizer, loss="mse", metrics=[off_by_0_33, off_by_0_2 , off_by_0_1, 'accuracy'])

    history = model.fit(
        train_generator,
        epochs=info_train['epochs'],
        callbacks=callbacks,
        validation_data=val_generator,
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
    accuracy_columns = [col for col in df_history.columns if ('_accuracy' in col or '_lambda' in col) and not 'val_' in col]
    val_accuracy_columns =  [col for col in df_history.columns if ('_accuracy' in col or '_lambda' in col) and 'val_' in col]
    print(accuracy_columns)
    df_history["accuracy"] = df_history[accuracy_columns].mean(axis=1)
    df_history["val_accuracy"] = df_history[val_accuracy_columns].mean(axis=1)
    print(df_history)
    print("loss in columns", "loss" in df_history.columns)
    print("acc in columns", "loss" in df_history.columns)
    print("val loss in columns", "val_loss" in df_history.columns)
    print("val acc in columns", "val_accuracy" in df_history.columns)
    last_epoch_nr = int(repo2.get_last_epoch_nr(selected_info['name'], type='DD'))
    print("last_epoch", last_epoch_nr, from_scratch)
    if last_epoch_nr > 0:
        # Return when training from scratch has worse results than last time
        # TODO : make a function to update val_iou based on last validation set
        last_result = repo2.get_last_epoch_values(modelname=selected_info["name"], epoch=last_epoch_nr, type='DD')
        print('result now', df_history.loc[df_history.index[-1], 'val_accuracy'])
        print('last result was: ', last_result.loc[0, 'val_accuracy'])
        if not trainings_info['save_anyway'] and df_history.loc[df_history.index[-1], 'val_iou'] < last_result.loc[0, 'val_iou']:
            print("RESULTS WEREN'T BETTER")
            return df_history

        df_history["epoch"] = df_history.index + 1 + (0 if from_scratch else last_epoch_nr)
    else:
        df_history["epoch"] = df_history.index + 1
    
    model.save_weights(weight_path)
    repo2.save_train_results(df_history, from_scratch=from_scratch, skills=True)

    return df_history

###############################################################################

info_ViViT = {
    'video_model' : True,
    'name' : 'vision_transformer',
    'dim' : 224,
    'patch_size' : 14,
    'timesteps' : 16,
    'batch_size' : 1,
    'dim_embedding' : 32,
    'num_heads': 4,
    'encoder_blocks': 4,
    'mlp_head_units' : [128 + 512, 1024, 256, 64],  # Size of the dense layers
    'min_epochs' : 10,
    'learning_rate' : 1e-4,
    'weight_decay' : 4e-5,
    'get_model_function' : get_model_ViViT,
}
info_ViViT['name'] = f"segmentation_video_vision_transformer_adamw_d{info_ViViT['dim']}_p{info_ViViT['patch_size']}_e{info_ViViT['dim_embedding']}_nh{info_ViViT['num_heads']}"



###############################################################################
selected_info = info_ViViT
###############################################################################

trainings_info = {
    'epochs' : 3, # Take more if first train round of random or transformer
    'early_stopping' : True,
    'restore_best_weights' : False,
    'early_stopping_patience' : 25,
    'batch_size' : selected_info['batch_size'],
    'timesteps' : None if 'timesteps' not in selected_info.keys() else selected_info['timesteps'],
    'learning_rate' : 8e-4 if 'learning_rate' not in selected_info.keys() else selected_info['learning_rate'],
    'train_date' : datetime.now().strftime("%Y%m%d"),
    'save_anyway' : True,
}
trainings_info['weight_decay'] = trainings_info['learning_rate'] / 20 if 'weight_decay' not in selected_info.keys() else selected_info['weight_decay']

model = selected_info['get_model_function'](selected_info)
model.summary()

history = train_model(model, info_train=trainings_info, from_scratch=True)

print(history)
