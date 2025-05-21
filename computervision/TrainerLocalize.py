import numpy as np
import torch

import os
import json
import cv2
import shutil
import pandas as pd

# from colorama import Fore, Style
from typing import List
from dotenv import load_dotenv
from managers.DataRepository import DataRepository

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from datetime import datetime 
from localizor_with_strats import validate_localize


load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
LABELED_FRAMES_FOLDER = "ultralytics-yolo"
IMAGES_FOLDER = "images"
LABELS_FOLDER = "labels"
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS")

def train_yolo_model(variant: str, repo: DataRepository):
    """modeldir: e.g. runs/detect/train5"""
    def get_video_path(repo, videoId):
        return os.path.join(STORAGE_DIR, repo.VideoNames.loc[videoId, "name"])

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

            previous_frameNr = frameNr


    num_val_images = len(os.listdir(os.path.join(STORAGE_DIR, LABELED_FRAMES_FOLDER, LABELS_FOLDER, 'val')))

    model = YOLO(variant)
    args = dict(model=variant, data="jumpers.yml", epochs=300, batch=16, patience=15, lr0=0.001)
    results = model.train(**args)

    # 'ap_class_index', 'box', 'class_result', 'confusion_matrix', 'curves', 'curves_results', 
    # 'fitness', 'keys', 'maps', 'mean_results', 'names', 'on_plot', 'plot', 'process', 'results_dict', 'save_dir', 'speed', 'task'

    simplified = {
        'ap_class_index': results.ap_class_index.tolist(),
        'fitness': results.fitness,
        'keys': results.keys,
        'maps': results.maps.tolist(),
        'names': results.names,
        'num_val_images': num_val_images,
        'results_dict': results.results_dict,
        'speed': results.speed,
        'date': datetime.now()
    }

    save_dir = results.save_dir

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(simplified, f, sort_keys=True, default=str)

    return save_dir

if __name__ == "__main__":
    repo = DataRepository()
    size = 'n'
    variant = f'yolo11{size}.pt'
    save_dir = '/home/miked/code/judge/runs/detect/train11'
    # save_dir = train_yolo_model(variant=variant, repo=repo)
    # modelname = f"yolov11{size}_{save_dir.split('/')[-1]}" # TODO : get os seperator
    # validate_localize(modeldir=save_dir, repo=repo, modelname='yolov11n_train9')

