import cv2
import json
import numpy as np
import os
import yaml
import shutil
import torch
import errno

from colorama import Fore, Style
from constants import ENVS
from datetime import datetime
from helpers import safe_rmtree
from localizor_with_strats import validate_localize
from managers.RepoGeneral import DataRepository
from ultralytics import YOLO

IMAGES_FOLDER = "images"
LABELS_FOLDER = "labels"

def train_yolo_model(variant: str, repo: DataRepository):
    """modeldir: e.g. runs/detect/train5"""
    def get_video_path(repo, videoId):
        return os.path.join(ENVS.DIRS.VIDEOS, repo.VideoNames.loc[videoId, "name"])
    
    def __generate_yolo_yaml(repo: DataRepository):
        # TODO : Should only be when training right?
        data = {
            'path': os.path.join(ENVS.DIRS.YOLO_LABELS),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(repo.get_frame_label_types())},
            'weights': [3,5,2]
        }

        # Save to a YAML file
        with open('jumpers.yml', 'w') as file:
            yaml.dump(data, file, sort_keys=False)

    __generate_yolo_yaml(repo)

    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images'), exist_ok=True)
    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'val'), exist_ok=True)
    shutil.rmtree(os.path.join(ENVS.DIRS.YOLO_LABELS, 'labels'))
    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'labels', 'test'), exist_ok=True)
    os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'labels', 'val'), exist_ok=True)

    os.makedirs(ENVS.DIRS.WEIGHTS.YOLO, exist_ok=True)
    weightdir = os.path.join(ENVS.DIRS.WEIGHTS.YOLO, variant)
    os.makedirs(weightdir, exist_ok=True)

    small_dataset = True
    previous_frameNr = 0
    previous_videoId = 0
    cap = None
    for train_test in ["train", "val"]:
        frames = repo.get_framelabels(train_test)
        frames = frames.sort_values(['videoId', 'frameNr'])
        
        imagecount = np.unique(frames[['videoId', 'frameNr']].values)
        if len(imagecount) < 75:
            patience = 30
            lr = 0.0002
            batch_size = 1
        elif len(imagecount) < 200:
            patience = 22
            lr = 0.0001
            batch_size = 2
        elif len(imagecount) < 500:
            patience = 15
            lr = 0.0001 / 2
            batch_size = 4
        else:
            patience = 8
            lr = 0.00001
            batch_size = 8

        for idx, row in frames.iterrows():
            videoId = int(row["videoId"])
            frameNr = int(row["frameNr"])
            x = row["x"]
            y = row["y"]
            w = row["width"]
            h = row["height"]
            classIdx = row['labeltype']

            video_path = get_video_path(repo, videoId)
            image_name = f"v{videoId:06d}fnr{frameNr:07d}"
            label_name = os.path.join(ENVS.DIRS.YOLO_LABELS, LABELS_FOLDER, train_test, f"{image_name}.txt")
            image_name = os.path.join(ENVS.DIRS.YOLO_LABELS, IMAGES_FOLDER, train_test, f"{image_name}.jpg")
            
            if not os.path.exists(image_name):
                print(f"creating {image_name}")
            
                cap = cv2.VideoCapture(video_path)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
                _, frame = cap.read()

                cv2.imwrite(image_name, frame)

            with open(label_name, 'a') as f:
                f.write(f"{classIdx} {x} {y} {w} {h}\n")

    num_val_images = len(os.listdir(os.path.join(ENVS.DIRS.YOLO_LABELS, LABELS_FOLDER, 'val')))

    variantWeightName = f'yolo11{variant}.pt' # TODO : use recipes.weights if set (optional)
    model = YOLO(variantWeightName)
    args = dict(
        model=variantWeightName, 
        data="jumpers.yml", 
        epochs=300, 
        batch=batch_size, 
        patience=patience, 
        lr0=lr,
        project=weightdir
    )

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

    trained_raw_avg_team_box_iou = validate_localize(modeldir=save_dir, repo=repo)
    
    # Post validation, keep best weights
    for previous_round_dir in os.listdir(weightdir):
        previous_folder = os.path.join(weightdir, previous_round_dir)
        if os.path.abspath(previous_folder) == os.path.abspath(save_dir):
            # Skip last trained one
            continue
        
        if not os.path.exists(os.path.join(previous_folder, 'weights', 'best.pt')):
            # The path if a training failed, and no weights exist.
            previous_raw_avg_team_box_io = 0
        else:
            # Re-validate previous trained model
            previous_raw_avg_team_box_io = validate_localize(modeldir=previous_folder, repo=repo)
        
        torch.cuda.synchronize()
        
        model_improved = previous_raw_avg_team_box_io * 1.01 < trained_raw_avg_team_box_iou
        print(f"Training {f"{Fore.GREEN}increased" if model_improved else f"{Fore.RED}decreased"}{Style.RESET_ALL} from {previous_raw_avg_team_box_io:.4f} to {trained_raw_avg_team_box_iou:.4f} ({Fore.GREEN if model_improved else Fore.RED}{(trained_raw_avg_team_box_iou - previous_raw_avg_team_box_io):.4f}{Style.RESET_ALL})")
        try:
            if model_improved:
                shutil.rmtree(previous_folder)
            else:
                shutil.rmtree(save_dir)
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                print(
                    f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} "
                    f"Could not remove directory (not empty): {e.filename}. Continuing."
                )
                continue
            else:
                raise

