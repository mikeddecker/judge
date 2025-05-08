import numpy as np
import torch

import os
import cv2
import shutil
import pandas as pd

# from colorama import Fore, Style
from typing import List
from dotenv import load_dotenv
from managers.DataRepository import DataRepository

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer



load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
LABELED_FRAMES_FOLDER = "labeled-frames"
IMAGES_FOLDER = "images"
LABELS_FOLDER = "labels"
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS")
VIDEO_IMAGE_PREVIEW_FOLDER = os.getenv("VIDEO_IMAGE_PREVIEW_FOLDER")


os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, IMAGES_FOLDER), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, IMAGES_FOLDER, "train"), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, IMAGES_FOLDER, "test"), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, IMAGES_FOLDER, "val"), exist_ok=True)
if os.path.exists(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER)):
    shutil.rmtree(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER)) # Always fresh start for labels
os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER, "train"), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER, "test"), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER, "val"), exist_ok=True)


def get_video_path(repo, videoId):
    return os.path.join(STORAGE_DIR, repo.VideoNames.loc[videoId, "name"])


repo = DataRepository()

previous_videoId = 0
previous_frameNr = 0
cap = None
for train_test in ["train", "val"]:
    frames = repo.get_framelabels(train_test, type=2)

    for idx, row in frames.iterrows():
        videoId = int(row["videoId"])
        frameNr = int(row["frameNr"])
        x = row["x"]
        y = row["y"]
        w = row["width"]
        h = row["height"]

        video_path = get_video_path(repo, videoId)
        image_name = f"v{videoId:06d}fnr{frameNr:07d}"
        label_name = os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER, train_test, f"{image_name}.txt")
        image_name = os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, IMAGES_FOLDER, train_test, f"{image_name}.jpg")
        if not os.path.exists(image_name):
            print(f"creating {image_name}")
        
            if previous_frameNr != frameNr:
                cap = cv2.VideoCapture(video_path)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
            _, frame = cap.read()

            cv2.imwrite(image_name, frame)

        with open(label_name, 'a') as f:
            f.write(f"{0} {x} {y} {w} {h}\n")

        previous_videoId = videoId
        previous_frameNr = frameNr




variant = 'yolo11n.pt'
model = YOLO(variant)

args = dict(model=variant, data="jumpers.yml", epochs=300, batch=16, patience=15, lr0=0.001)
trainer = DetectionTrainer(overrides=args)
trainer.train()
print("all done")