# %%
import os
import cv2
import shutil
import pandas as pd
import numpy as np
import json

# from colorama import Fore, Style
from typing import List
from dotenv import load_dotenv
from managers.DataRepository import DataRepository
from ultralytics import YOLO
import torch

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
CROPPED_VIDEOS_FOLDER = "cropped-videos"
# IMAGES_FOLDER = "images"
# LABELS_FOLDER = "labels"
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS")
VIDEO_IMAGE_PREVIEW_FOLDER = os.getenv("VIDEO_IMAGE_PREVIEW_FOLDER")


# %%
from moviepy import ImageSequenceClip

# %%
print(STORAGE_DIR), print(CROPPED_VIDEOS_FOLDER)
# computervision/labeled_frame_nrs_to_imgs.ipynb

# %%
os.makedirs(os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER), exist_ok=True)

def get_video_path(repo, videoId):
    return os.path.join(STORAGE_DIR, repo.VideoNames.loc[videoId, "name"])

# %%
repo = DataRepository()

# %%
modeltrainround = 7
modelpath = f"/home/miked/code/judge/runs/detect/train{modeltrainround}" if modeltrainround > 0 else f"/home/miked/code/judge/runs/detect/train"
modelpath

# %%
# From https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
# Load a model
model = YOLO(os.path.join(modelpath, "weights", "best.pt"))


# %%
[676]

# %%
DIM = 224
# videoIds = repo.get_dd3_videoIds()["id"].to_list()
videoIds = [676]
PEOPLE = 3 # TODO : provide as a parameter or solve when skippers are entering the field

for videoId in videoIds:
    print("*"*30 + str(videoId) + "*"*30)

    videoPath = get_video_path(repo, videoId=videoId)
    videoOutputPath = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f"{DIM}_{videoId}_opencv.mp4")
    videoOutputPathCorrect = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f"{DIM}_{videoId}_v3.mp4")
    
    # if os.path.exists(videoOutputPathCorrect):
    #     continue

    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_time = int(333 / fps - 7)
    padding_x = width // 30
    padding_y = height // 35
    N = 4
    avgIOUlastNseconds = 0.0
    secondary_avgIOUlastNseconds = 0.0
    N_FRAMES = N * fps
    SQRT = 8

    smootval = 0.87
    smootval_shrink = 0.955
    smooted_x1_min = None
    smooted_y1_min = None
    smooted_x2_max = None
    smooted_y2_max = None
    predictions = {
        "x1": [],
        "y1": [],
        "x2": [],
        "y2": [],
        "frameNr" : [],
    }
    
    max_w = 0
    max_h = 0
    min_w = width
    min_h = height
    max_wh = max(width, height)

    times_with_no_jumper = 0
    cropping_started = False
    frames = []
    i = 0
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # fourcc = 0x00000021
    # out = cv2.VideoWriter(videoOutputPath, fourcc, fps, (DIM, DIM))
    ret, frame = cap.read()
    while ret:
        # i += 1
        result = model(frame, verbose=False)
        xyxy_boxes = result[0].boxes.xyxy
        predictions["frameNr"].append(i)
        
        if xyxy_boxes.shape[0] > 0:
            cropping_started = True
            x1_min = max(0, int(xyxy_boxes[:, 0].min().item()) - padding_x)
            y1_min = max(0, int(xyxy_boxes[:, 1].min().item()) - padding_y)
            x2_max = min(width, int(xyxy_boxes[:, 2].max().item()) + padding_x)
            y2_max = min(height, int(xyxy_boxes[:, 3].max().item()) + padding_y)
            times_with_no_jumper = 0
        elif times_with_no_jumper < 8 and 'x1_min' in locals():
            times_with_no_jumper += 1
        else:
            x1_min = 0
            y1_min = 0
            x2_max = width
            y2_max = height
            times_with_no_jumper += 1

        if len(xyxy_boxes) > PEOPLE + 1:
            def pairwise_distances(x):
                diffs = x.unsqueeze(1) - x.unsqueeze(0)  # shape: (N, N, 4)
                dists = torch.norm(diffs, dim=2)         # shape: (N, N)
                return dists

            dists = pairwise_distances(xyxy_boxes)

            ratio = (x2_max-x1_min) / (y2_max - y1_min)
            avg_dists = dists.mean(dim=1)
            std = avg_dists.std()
            mean = avg_dists.mean()

            boxesNearOtherBoxes = avg_dists < mean + std * 1.25
            boxesNearOtherBoxesIndexes = boxesNearOtherBoxes.nonzero(as_tuple=True)
            xyxy_boxes = xyxy_boxes[boxesNearOtherBoxesIndexes]


        if smooted_x1_min is None or smooted_x1_min == 1:
            smooted_x1_min = int(x1_min)
            smooted_y1_min = int(y1_min)
            smooted_x2_max = int(x2_max)
            smooted_y2_max = int(y2_max)
            secondary_smooted_x1_min = x1_min
            secondary_smooted_y1_min = y1_min
            secondary_smooted_x2_max = x2_max
            secondary_smooted_y2_max = y2_max

        else:
            intersection_width = max(0, min(x2_max, smooted_x2_max) - max(x1_min, smooted_x1_min))
            intersection_height = max(0, min(y2_max, smooted_y2_max) - max(y1_min, smooted_y1_min))
            intersection_area = intersection_width * intersection_height
            area_new_box = (x2_max - x1_min) * (y2_max - y1_min)
            area_smooted_box = (smooted_x2_max - smooted_x1_min) * (smooted_y2_max - smooted_y1_min)
            union = area_smooted_box + area_new_box - intersection_area
            iou = intersection_area / union
            avgIOUlastNseconds = ((N_FRAMES - 1) * avgIOUlastNseconds + iou) / N / fps 
            avgSqrd = avgIOUlastNseconds*avgIOUlastNseconds
            iou_threshold = avgSqrd * avgSqrd

            # secondary
            secondary_intersection_width = max(0, min(x2_max, secondary_smooted_x2_max) - max(x1_min, secondary_smooted_x1_min))
            secondary_intersection_height = max(0, min(y2_max, secondary_smooted_y2_max) - max(y1_min, secondary_smooted_y1_min))
            secondary_intersection_area = secondary_intersection_width * secondary_intersection_height
            area_new_box = (x2_max - x1_min) * (y2_max - y1_min)
            area_secondary_smooted_box = (secondary_smooted_x2_max - secondary_smooted_x1_min) * (secondary_smooted_y2_max - secondary_smooted_y1_min)
            secondary_union = area_secondary_smooted_box + area_new_box - secondary_intersection_area
            secondary_iou = secondary_intersection_area / secondary_union
            secondary_avgIOUlastNseconds = ((N_FRAMES / 2 - 1) * secondary_avgIOUlastNseconds + secondary_iou) / N_FRAMES * 2 
            secondary_avgSqrd = secondary_avgIOUlastNseconds*secondary_avgIOUlastNseconds
            secondary_iou_threshold = secondary_avgSqrd * secondary_avgSqrd

            # print(f"{iou > iou_threshold} : {iou:.3f} - 2nd = {secondary_iou:.3f} - T = {secondary_iou_threshold:.3f} --> follow? = {secondary_iou > secondary_iou_threshold} - {secondary_avgIOUlastNseconds} - N preds = {len(xyxy_boxes)}")
            # print(i, iou, iou_threshold)
            if iou > iou_threshold:
                movement_corrector = iou ** (1/SQRT) # In order to just follow current predictions (when big movements are made)
                smooted_x1_min = int((smootval * smooted_x1_min + (1-smootval) * x1_min if x1_min < smooted_x1_min else smootval_shrink * smooted_x1_min + (1-smootval_shrink) * x1_min) * movement_corrector + (1- movement_corrector) * x1_min)
                smooted_y1_min = int((smootval * smooted_y1_min + (1-smootval) * y1_min if y1_min < smooted_x1_min else smootval_shrink * smooted_y1_min + (1-smootval_shrink) * y1_min) * movement_corrector + (1- movement_corrector) * y1_min)
                smooted_x2_max = int((smootval * smooted_x2_max + (1-smootval) * x2_max if x2_max > smooted_x2_max else smootval_shrink * smooted_x2_max + (1-smootval_shrink) * x2_max) * movement_corrector + (1- movement_corrector) * x2_max)
                smooted_y2_max = int((smootval * smooted_y2_max + (1-smootval) * y2_max if y2_max > smooted_y2_max else smootval_shrink * smooted_y2_max + (1-smootval_shrink) * y2_max) * movement_corrector + (1- movement_corrector) * y2_max)
            


        # Cutting the image
        w_jumpers = smooted_x2_max - smooted_x1_min
        h_jumpers = smooted_y2_max - smooted_y1_min
        max_w = max(max_w, w_jumpers)
        max_h = max(max_h, h_jumpers)
        min_w = min(min_w, w_jumpers)
        min_h = min(max_h, h_jumpers)


        max_wh_jumpers = max(w_jumpers, h_jumpers)
        offset_x = (max_wh_jumpers - w_jumpers) // 2
        offset_y = (max_wh_jumpers - h_jumpers) // 2

        leftover_pixels_x = smooted_x1_min - offset_x
        if leftover_pixels_x < 0:
            crop_x1 = 0
            offset_x_left = abs(leftover_pixels_x)
        else:
            crop_x1 = leftover_pixels_x
            offset_x_left = 0
        
        leftover_pixels_x_right = width - (smooted_x2_max + offset_x)
        crop_x2 = width if leftover_pixels_x_right < 0 else min(smooted_x2_max + offset_x, width)

        leftover_pixels_y = smooted_y1_min - offset_y
        if leftover_pixels_y < 0:
            crop_y1 = 0
            offset_y_left = abs(leftover_pixels_y)
        else:
            crop_y1 = leftover_pixels_y
            offset_y_left = 0
        
        leftover_pixels_y_right = width - (smooted_y2_max + offset_y)
        crop_y2 = width if leftover_pixels_y_right < 0 else min(smooted_y2_max + offset_y, height)

        predictions["x1"].append(crop_x1)
        predictions["y1"].append(crop_y1)
        predictions["x2"].append(crop_x2)
        predictions["y2"].append(crop_y2)

        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        zeros = np.zeros((max_wh_jumpers, max_wh_jumpers, 3), dtype=np.uint8)
        zeros[offset_y_left:offset_y_left+(crop_y2-crop_y1), offset_x_left:offset_x_left+(crop_x2 - crop_x1)] = cropped_frame
        zeros = cv2.resize(zeros, (DIM, DIM))
        zeros = cv2.cvtColor(zeros, cv2.COLOR_BGR2RGB)

        frames.append(zeros)
        
        ret, frame = cap.read()
        i += 1

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    df_predictions = pd.DataFrame(predictions)
    df_predictions["videoId"] = videoId
    df_predictions["locationChecked"] = False
    df_predictions["segmentsChecked"] = False
    df_predictions.to_sql(name="Predictions_Localization", if_exists='append', con=repo.con, chunksize=500, index=False)
    repo.con.commit()
    
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(videoOutputPathCorrect)

    # To browser runnable codec
    # os.system(f"ffmpeg -i {videoOutputPath} -vcodec libx264 {videoOutputPathCorrect}")

# %%
df_predictions

# %%


# %%



# %%



