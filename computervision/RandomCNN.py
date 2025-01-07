#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement googlenetsmall
# Build an inception module
# get_ipython().run_line_magic('pip', 'install sqlalchemy')

DIM = 512

# In[2]:


import functools
import keras

DefaultConv = functools.partial(
    keras.layers.Conv2D, kernel_size=(3, 3), strides=(2, 2),
    padding="same", activation="relu")


# In[4]:


DefaultMaxPool = functools.partial(
    keras.layers.MaxPool2D,
    pool_size=(3,3), strides=(2,2), padding="same")

def get_model(input_shape, num_classes, use_batch_norm=True, **kwargs):
  model = keras.Sequential(**kwargs)
  model.add(keras.layers.Input(shape=input_shape))
  model.add(DefaultConv(filters=24, kernel_size=(5,5), strides=(2,2),  padding='same'))
  if use_batch_norm:
    model.add(keras.layers.BatchNormalization())
  model.add(DefaultMaxPool())
  model.add(keras.layers.Dropout(0.25))
  model.add(DefaultConv(filters=32))
  model.add(DefaultConv(filters=48))
  if use_batch_norm:
    model.add(keras.layers.BatchNormalization())
  model.add(DefaultMaxPool())
  model.add(DefaultConv(filters=64))

  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(units=128, activation="relu"))
  model.add(keras.layers.Dense(units=num_classes, activation='sigmoid'))

  return model


# In[5]:


model = get_model(input_shape=(DIM,DIM,3), num_classes=4, use_batch_norm=True)
model.summary()


# In[6]:


def iou(y_true, y_pred):
    """
    Calculate IoU loss between the true and predicted bounding boxes.

    y_true and y_pred should have the shape (batch_size, 4), where each element is
    [center_x, center_y, width, height].
    """
    # Convert (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
    true_xmin = y_true[..., 0] - 0.5 * y_true[..., 2]
    true_ymin = y_true[..., 1] - 0.5 * y_true[..., 3]
    true_xmax = y_true[..., 0] + 0.5 * y_true[..., 2]
    true_ymax = y_true[..., 1] + 0.5 * y_true[..., 3]

    pred_xmin = y_pred[..., 0] - 0.5 * y_pred[..., 2]
    pred_ymin = y_pred[..., 1] - 0.5 * y_pred[..., 3]
    pred_xmax = y_pred[..., 0] + 0.5 * y_pred[..., 2]
    pred_ymax = y_pred[..., 1] + 0.5 * y_pred[..., 3]

    # Calculate the intersection area
    inter_xmin = keras.ops.maximum(true_xmin, pred_xmin)
    inter_ymin = keras.ops.maximum(true_ymin, pred_ymin)
    inter_xmax = keras.ops.minimum(true_xmax, pred_xmax)
    inter_ymax = keras.ops.minimum(true_ymax, pred_ymax)

    inter_width = keras.ops.maximum(0.0, inter_xmax - inter_xmin)
    inter_height = keras.ops.maximum(0.0, inter_ymax - inter_ymin)
    intersection_area = inter_width * inter_height

    # Calculate the union area
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union_area = true_area + pred_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou



# In[7]:


# Compile the model with IoU loss
# model.compile(optimizer='adam', loss=keras.losses.Huber(), metrics=[iou_loss])
model.compile(optimizer='adam', loss='mse', metrics=[iou])


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from FrameLoader import FrameLoader
from DataGeneratorFrames import DataGeneratorFrames
from DataRepository import DataRepository

repo = DataRepository()
repo.load_relativePaths_of_videos_with_framelabels()

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


callbacks = [
    ModelCheckpoint('model_best.keras', save_best_only=True, monitor='loss', mode='min', verbose=1),
    EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=7,
    callbacks=callbacks,
    verbose=1,
    validation_data=val_generator
)


# # In[9]:


# X, y = train_generator.__getitem__(5)
# X.shape, y.shape


# # In[ ]:


keras.models.save_model(
    model,
    filepath="randomModel.keras",
    overwrite=True
)

model.save_weights("randomModel.weights.h5")

