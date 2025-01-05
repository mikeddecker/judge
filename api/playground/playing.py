#!/usr/bin/env python
# coding: utf-8

# # Getting started with transforms v2

# In[2]:


from pathlib import Path
import os
import sys
sys.path.append("..")
import torch
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.transforms import v2
from torchvision.io import decode_image

from torchvision.transforms.v2.functional import crop
import sqlalchemy
import pandas as pd

import cv2
from services.videoService import VideoService
from dotenv import load_dotenv
load_dotenv()

torch.manual_seed(1)

# If you're trying to run that on Colab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
from playground.helpers import plot

STORAGE_DIR = os.getenv("STORAGE_DIR")
videoService = VideoService(STORAGE_DIR)
## Loading from video
# Using cv2, as pytorch video api is time based and not frame based (to set specific frame)


def get_frames(path, frameNrs, keepWrongBGRColors=True):
    frames = {}
    cap = cv2.VideoCapture(path)
    for frameNr in frameNrs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
        _, frame = cap.read()
        frame = frame if keepWrongBGRColors else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[frameNr] = frame

    cap.release()

    return frames

def get_connection():
    HOST = '127.0.0.1'
    PORT = '3377'
    DATABASE = 'judge'
    USERNAME = 'root'
    PASSWORD = 'root'
    DATABASE_CONNECTION=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    engine = sqlalchemy.create_engine(DATABASE_CONNECTION)#
    return engine.connect()


con = get_connection()


df_videoIds = pd.read_sql("SELECT DISTINCT videoId FROM FrameLabels", con)

# Getting relative paths for each videoId
infos = {}

for idx, row in df_videoIds.tail().iterrows():
    videoId = int(row["videoId"])
    infos[videoId] = videoService.get(videoId).to_dict(include_frames=False)

    if idx==5:
        print(infos)

videopath = "/media/miked/Elements/Judge/FINISHED-DB-READY/competition/belgium/DD3/bk-sipiro-dd3-2024-senioren-luka-j1.MP4"


# In[9]:


frames = get_frames(videopath, [3, 555], keepWrongBGRColors=False)


# In[10]:


torch_image = torch.from_numpy(frames[555]).permute(2, 0, 1)


# In[11]:


plot([torch_image])


# ## Loading image with border

# In[20]:


# In[21]:




# In[54]:


qry = "SELECT * FROM FrameLabels WHERE videoId = 101"
videopath = "/media/miked/Elements/Judge/FINISHED-DB-READY/competition/belgium/DD3/bk-sipiro-dd3-2024-junioren-lore-j1.MP4"
width = 1280
height = 720
df = pd.read_sql_query(qry, con)
df = df.sample(frac=1.)
df.head()


# In[55]:


df.info()


# In[82]:


frames = get_frames(videopath, df["frameNr"].iloc[0:1], keepWrongBGRColors=False)
frames = [v for v in frames.values()]
plot(frames)


# In[83]:


info = df.iloc[0]
info.x * width, info.y * height, info.width * width, info.height * height


# In[ ]:


from torchvision import tv_tensors  # we'll describe this a bit later, bare with us
import torchvision

img = frames[0]
torch_image = torch.from_numpy(img).permute(2, 0, 1)
boxes = tv_tensors.BoundingBoxes(
    [
        [info.x * width, info.y * height, info.width * width, info.height * height],     
        # [200.55,200,200,200]
    ],
    format="CXCYWH", canvas_size=torch_image.shape[-2:])
boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
non_tensor = tv_tensors.BoundingBoxes(boxes_xyxy, format="XYXY", canvas_size=torch_image.shape[-2:])

plot([(torch_image, non_tensor)])


# In[88]:




# In[ ]:




