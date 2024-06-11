#!/usr/bin/env python
# coding: utf-8

# In[1]:


# keras CNN predict air, between, ground or not jumping


# In[2]:


get_ipython().system('pip3 install tensorflow')


# In[3]:


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pickle

from my_classes import DataGeneratorSkillBorders


# In[5]:


video_border_labels_path = '../data/df_video_border_labels.pkl'
video_folder = '../videos/'

video_names = [
    '20240201_atelier_001.mp4',
    '20240201_atelier_002.mp4',
    '20240201_atelier_003.mp4',
    '20240201_atelier_004.mp4',
    '20240201_atelier_005.mp4',
    '20240209_atelier_006.mp4',
    '20240209_atelier_007.mp4',
    '20240209_atelier_008.mp4',
]

train_videos = [ video_folder + trainvid for trainvid in video_names]


# In[6]:


def pickle_load_or_create(path, cols):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        return pd.DataFrame(columns=cols)


# In[7]:


df_labels = pickle_load_or_create(video_border_labels_path, [])
df_labels


# In[8]:

print(df_labels)
df_labels.loc[df_labels.border == 5, 'border'] = 3
df_labels.loc[df_labels.border == 9, 'border'] = 4


# In[9]:


def get_random_frame(videos, grey=True, scale=1):
    """
    videos: array of video_paths
    df_video_border_labels: panda dataframe ['path', 'frame', 'borderlabel']
          0 : ground
          1 : heels of ground
          2 : air
          3 : Fault
          4 : no skipper or not jumping
    """
    path = videos[np.random.randint(0, len(videos)-1)]
    
    cap = cv2.VideoCapture(path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_nr = np.random.randint(0, video_length-1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    res, frame = cap.read()
    frame = cv2.resize(frame, dsize=(0,0), fx=scale, fy=scale)
    # frame = cv2.cvtColor(frame, 7)
    cap.release()
    # cv2.destroyAllWindows()

    return path, frame_nr, frame


# In[10]:


input_shape = get_random_frame(train_videos, grey=True, scale=0.4)[2].shape
input_shape


# In[11]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[12]:


model.summary()


# In[13]:


unique_labels = df_labels['border'].unique()
unique_labels


# In[14]:


model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(5))


# In[15]:


model.summary()


# In[ ]:





# In[16]:


from my_classes import DataGeneratorSkillBorders
# Parameters
params = {'dim': (64,64),
          'batch_size': 16,
          'n_classes': len(unique_labels),
          'n_channels': 3,
          'shuffle': True,
          'train': True}

training_generator = DataGeneratorSkillBorders(df_labels, video_folder=video_folder, **params)
test_generator = DataGeneratorSkillBorders(df_labels, video_folder=video_folder, train=False)


# In[17]:


get_ipython().run_cell_magic('time', '', 'X, y = training_generator.__getitem__(3)\n')


# In[18]:


X


# In[19]:


y


# In[ ]:





# In[20]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(training_generator, epochs=10, 
                    validation_data=test_generator)


# In[ ]:





# In[26]:


pd.DataFrame(history.history)


# In[ ]:





# In[32]:


with open('../models/last_cnn_model_history.pkl', 'wb') as handle:
    pickle.dump(history, handle)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




