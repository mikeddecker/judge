#!/usr/bin/env python
# coding: utf-8

# # OpenCV help methods

from utils_cv2 import label_frames_seq
from utils_misc import pickle_save

video_border_labels_path = 'df_video_border_labels_seq'
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


df_labels = label_frames_seq()
print(df_labels.loc[len(df_labels)-25:len(df_labels)])

pickle_save(video_border_labels_path, df_labels)

val_counts = df_labels.border.value_counts() / len(df_labels)
print(val_counts)
