import cv2
import json
import numpy as np
import os
import pandas as pd

from constants import ENVS
from datetime import datetime 
from dotenv import load_dotenv
from localizor_with_strats import validate_localize
from managers.DataRepository import DataRepository
from ultralytics import YOLO


IMAGES_FOLDER = "images"
LABELS_FOLDER = "labels"

def train_yolo_model(variant: str, repo: DataRepository):
    """modeldir: e.g. runs/detect/train5"""
    def get_video_path(repo, videoId):
        return os.path.join(ENVS.DIRS.VIDEOS, repo.VideoNames.loc[videoId, "name"])

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
            label_name = os.path.join(ENVS.DIRS.YOLO_LABELS, LABELS_FOLDER, train_test, f"{image_name}.txt")
            image_name = os.path.join(ENVS.DIRS.YOLO_LABELS, IMAGES_FOLDER, train_test, f"{image_name}.jpg")
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


    num_val_images = len(os.listdir(os.path.join(ENVS.DIRS.YOLO_LABELS, LABELS_FOLDER, 'val')))

    # TODO : make jumpers.yml dynamic based on config
    model = YOLO(variant)
    args = dict(model=variant, data="jumpers.yml", epochs=300, batch=8, patience=8, lr0=0.0001)
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
    # modelname = f"yolov11{size}_{save_dir.name}"
    # validate_localize(modeldir=save_dir, repo=repo, modelname=modelname)

