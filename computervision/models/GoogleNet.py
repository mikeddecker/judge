#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Implement GoogleNet
# Build an inception module
# get_ipython().run_line_magic('pip', 'install sqlalchemy')


# In[2]:
import sys
sys.path.append(".")
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
    # print('path1', path1.shape)
    # print('path2', path2.shape)
    # print('path3', path3.shape)
    # print('path4', path4.shape)

    iModule = keras.layers.Concatenate(axis=-1)([path1, path2, path3, path4])
    # print('imodule', iModule.shape)
    logits = self.batch_layer(iModule) if self.use_batch_norm else iModule

    return logits


# In[4]:


DefaultMaxPool = functools.partial(
    keras.layers.MaxPool2D,
    pool_size=(3,3), strides=(2,2), padding="same")

def get_googlenet_model(input_shape, num_classes, use_batch_norm=True, **kwargs):
  model = keras.Sequential(**kwargs)
  model.add(keras.layers.Input(shape=input_shape))
  model.add(DefaultConv(filters=64, kernel_size=(7,7), strides=(2,2),  padding='same'))
  if use_batch_norm:
    model.add(keras.layers.BatchNormalization())
  model.add(DefaultMaxPool())
  model.add(DefaultConv(filters=64))
  model.add(DefaultConv(filters=192, kernel_size=(3,3)))
  if use_batch_norm:
    model.add(keras.layers.BatchNormalization())
  model.add(DefaultMaxPool())

  model.add(InceptionModule(filters11=64, filters33_reduce=96, filters33=128,
    filters55_reduce=16, filters55=32, filters_pool_proj=32,
    use_batch_norm=use_batch_norm))
  model.add(InceptionModule(filters11=128, filters33_reduce=128, filters33=192,
    filters55_reduce=32, filters55=96, filters_pool_proj=64,
    use_batch_norm=use_batch_norm))
  model.add(DefaultMaxPool())

  model.add(InceptionModule(filters11=192, filters33_reduce=96, filters33=208,
    filters55_reduce=16, filters55=48, filters_pool_proj=64,
    use_batch_norm=use_batch_norm))
  model.add(InceptionModule(filters11=160, filters33_reduce=112, filters33=224,
    filters55_reduce=24, filters55=64, filters_pool_proj=64,
    use_batch_norm=use_batch_norm))
  model.add(InceptionModule(filters11=128, filters33_reduce=128, filters33=256,
    filters55_reduce=24, filters55=64, filters_pool_proj=64,
    use_batch_norm=use_batch_norm))
  model.add(InceptionModule(filters11=112, filters33_reduce=144, filters33=288,
    filters55_reduce=32, filters55=64, filters_pool_proj=64,
    use_batch_norm=use_batch_norm))
  model.add(InceptionModule(filters11=256, filters33_reduce=160, filters33=320,
    filters55_reduce=32, filters55=128, filters_pool_proj=128,
    use_batch_norm=use_batch_norm))

  model.add(DefaultMaxPool())
  model.add(InceptionModule(filters11=256, filters33_reduce=160, filters33=320,
    filters55_reduce=32, filters55=128, filters_pool_proj=128,
    use_batch_norm=use_batch_norm))
  model.add(InceptionModule(filters11=384, filters33_reduce=192, filters33=384,
    filters55_reduce=48, filters55=128, filters_pool_proj=128,
    use_batch_norm=use_batch_norm))
  model.add(keras.layers.GlobalAveragePooling2D())
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(units=1000, activation="relu"))
  model.add(keras.layers.Dense(units=4, activation='sigmoid'))

  return model


# In[5]:


model = get_googlenet_model(input_shape=(1024,1024,3), num_classes=4)
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

DIM = 512
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
    ModelCheckpoint('model_best.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

history = model.fit(
    train_generator,
    epochs=12,
    callbacks=callbacks,
    verbose=1,
    validation_data=val_generator
)


# In[9]:


# X, y = train_generator.__getitem__(5)
# X.shape, y.shape


# In[ ]:


keras.models.save_model(
    model,
    filepath="googlenet.keras",
    overwrite=True
)

model.save_weights("googlenet.weights.h5")

