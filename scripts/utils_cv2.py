#!/usr/bin/env python
# coding: utf-8

# # OpenCV help methods

get_ipython().system('pip3 install opencv-python pandas')

# Imports
import os
import numpy as np
import pandas as pd
import pickle
import cv2 # pip install opencv-python
import json

from utils_misc import pickle_load_or_create, pickle_save

# # constanten
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

# # functies

def allowed_keys():
    return ['n','f','a','i','h','m','t','g','p','y','w','r']

def match_label(key): # TODO : to json setting
    """
    n - no jump(er)
    f - fault
    a - air
    i - interaction
    h, m, t - preparing for jump / multiple, heels of ground, on the tips/toes
    g - ground
    p - power
    y - gym
    w - wrapped
    r - released
    """
    match key:
        case 'g' | '0' | 'ground':
            return 0
        case 'h' | 'm' | 't' | '1' | 'multiple-start' | 'heels' | 'toes' | 'side-multiple' :
            return 1
        case 'a' | 'air' | '2':
            return 2
        case 'i' | 'interaction':
            return 3
        case 'p' | 'power':
            return 4
        case 'y' | 'gym':
            return 5
        case 'w' | 'wrapped':
            return 6
        case 'r' | 'released':
            return 7
        case 'f' | 'fout':
            return 8
        case 'n' | 'no-jump' | 'no-skipper': 
            return 9

def get_random_frame(path):
    """
    Gets a random frame from the given video path
    
    return (frame_nr, frame)
    """
    cap = cv2.VideoCapture(path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nr = np.random.randint(0, video_length-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    _, frame = cap.read()
    cap.release()

    return frame_nr, frame

def get_frame(path, frame_nr, release_cap=True):
    """Gets a specific frame from the given video path"""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    _, frame = cap.read()
    if release_cap:
        cap.release()

    return frame

def get_video_length(path):
    """Returns the framelength of the video"""
    # TODO : Add frame length to DB
    cap = cv2.VideoCapture(path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return video_length

def show_frame_and_await_key(frame, scale=0.4, release_cap=True):
    """
    Displays a single frame using OpenCV, waits for a key press, 
    and then closes the frame window.

    Args:
        frame (numpy.ndarray): The image frame to display.

    Returns:
        int: The ASCII value of the key pressed.
    """
    if scale != 1.0:
        frame = cv2.resize(frame, dsize=(0,0), fx=scale, fy=scale)

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(0)

    return key

def is_frame_already_labeled(path, frame_nr, df_labels):
    return len(df_labels[(df_labels['path'] == path) & (df_labels['frame'] == frame_nr)]) > 0

def has_frame_received_new_or_the_equal_label(path, frame_nr, df_labels, label):
    if is_frame_already_labeled(path, frame_nr, df_labels):
        return label == df_labels[(df_labels['path'] == path) & (df_labels['frame'] == frame_nr)].loc[0]
    else:
        return True

def get_random_video_name():
    vid_id = np.random.randint(0, len(video_names) - 2)
    return video_names[vid_id]

def label_frames_seq():
    """
    """
    quit_key = 'q'
    key_pressed = 'none'

    df_labels = pickle_load_or_create(video_border_labels_path, lambda: pd.DataFrame(columns=['path', 'frame', 'border']), config=False)

    vid_name = get_random_video_name()
    vid_path = video_folder + vid_name
    vid_len = get_video_length(vid_path)
    last_frame = df_labels[df_labels['path'] == vid_name]['frame'].max()
    print(last_frame)
    frame_nr =  0 if np.isnan(last_frame) else last_frame + 1

    while key_pressed != quit_key:
        if vid_len <= frame_nr:
            break
        
        # while is_already_labeled(vid_name, frame_nr, df_labels):
            # vid_name, frame_nr, frame = get_random_frame(video_names, video_folder, df_labels)
        
        frame = get_frame(vid_path, frame_nr, release_cap=True)

        key_pressed = chr(show_frame_and_await_key(frame, 0.5))
        if key_pressed in allowed_keys():
            label = match_label(key_pressed)
            df_labels.loc[len(df_labels)] = [vid_name, frame_nr, label]
            frame_nr += 1

    # Close the frame window
    cv2.destroyAllWindows()

    return df_labels

def remove_last_labels(n):
    df_labels = df_labels.loc[0:len(df_labels)-n-1]
    df_labels
    pickle_save(video_border_labels_path, df_labels)

