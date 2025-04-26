import os
import cv2
import math
import shutil
import pandas as pd
import numpy as np

import time
# from colorama import Fore, Style
from typing import List
from dotenv import load_dotenv
from managers.DataRepository import DataRepository
from ultralytics import YOLO

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
modeltrainround = 9
modelpath = f"/home/miked/code/judge/runs/detect/train{modeltrainround}" if modeltrainround > 0 else f"/home/miked/code/judge/runs/detect/train"
modelpath

# %%
# From https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
# Load a model
model = YOLO(os.path.join(modelpath, "weights", "best.pt"))


# %%
def calculate_iou(new_x_min, new_y_min, new_x_max, new_y_max, old_x_min, old_y_min, old_x_max, old_y_max):
    intersection_width = max(0, min(new_x_max, old_x_max) - max(new_x_min, old_x_min))
    intersection_height = max(0, min(new_y_max, old_y_max) - max(new_y_min, old_y_min))
    intersection_area = intersection_width * intersection_height
    
    area_new_box = (new_x_max - new_x_min) * (new_y_max - new_y_min)
    area_old_box = (old_x_max - old_x_min) * (old_y_max - old_y_min)
    
    union = area_new_box + area_old_box - intersection_area
    if union == 0:
        return 0.0  # To avoid division by zero
    
    iou = intersection_area / union
    return iou

def calculate_angle(new_x_min, new_y_min, new_x_max, new_y_max, old_x_min, old_y_min, old_x_max, old_y_max):
    new_center_x = (new_x_min + new_x_max) / 2
    new_center_y = (new_y_min + new_y_max) / 2
    old_center_x = (old_x_min + old_x_max) / 2
    old_center_y = (old_y_min + old_y_max) / 2

    delta_x = new_center_x - old_center_x
    delta_y = new_center_y - old_center_y

    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)

    return angle_rad, angle_deg

def calculate_cosine_similarity(new_x_min, new_y_min, new_x_max, new_y_max, old_x_min, old_y_min, old_x_max, old_y_max):
    # Calculate centers
    new_center_x = (new_x_min + new_x_max) / 2
    new_center_y = (new_y_min + new_y_max) / 2
    old_center_x = (old_x_min + old_x_max) / 2
    old_center_y = (old_y_min + old_y_max) / 2

    # Calculate movement vector
    delta_x = new_center_x - old_center_x
    delta_y = new_center_y - old_center_y

    # Reference vector (moving horizontally right)
    ref_x = 1
    ref_y = 0

    # Dot product
    dot_product = delta_x * ref_x + delta_y * ref_y

    # Magnitudes
    magnitude_movement = math.sqrt(delta_x**2 + delta_y**2)
    magnitude_ref = 1  # (1,0) has length 1

    if magnitude_movement == 0:
        return 1.0  # No movement = fully similar

    cosine_similarity = dot_product / (magnitude_movement * magnitude_ref)

    return cosine_similarity


# %%
DIM = 256
videoIds = repo.get_dd3_videoIds()["id"].to_list()
# videoIds = [1271]

for videoId in videoIds:
    start = time.time()
    videoPath = get_video_path(repo, videoId=videoId)
    videoOutputPath = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f"{DIM}_{videoId}.mp4")
    videoOutputPathCorrect = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f"{DIM}_{videoId}.mp4")
    videoOutputPathCorrectOK = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, "OK", f"{DIM}_{videoId}.mp4")
    videoOutputPathCorrectNOK = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, "OK_NET_NIET_PERFECT", f"{DIM}_{videoId}.mp4")
    videoOutputPathCorrectAlmostOK = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, "SLECHT", f"{DIM}_{videoId}.mp4")

    if os.path.exists(videoOutputPath) or os.path.exists(videoOutputPathCorrect) or os.path.exists(videoOutputPathCorrectOK) or os.path.exists(videoOutputPathCorrectAlmostOK) or os.path.exists(videoOutputPathCorrectNOK):
        continue

    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_time = int(333 / fps - 7)
    padding_x = width // 60
    padding_y = height // 50
    N = 3
    K = 0.5
    avgIOUlastNseconds = 0.0
    secondary_avgIOUlastNseconds = 0.0
    cos_similarity = 0
    avgGradenLastKseconds = 0
    previouscos_similarity= 0
    avgGradenVerschilLastKseconds = 0

    smootval = 0.90
    smootval_stationary = 0.9
    smootval_follow = 0.66
    smootval_shrink_stationary = 0.95
    smootval_shrink_follow = 0.80
    smooted_x1_min = None
    smooted_y1_min = None
    smooted_x2_max = None
    smooted_y2_max = None
    previous_x_min = None

    max_w = 0
    max_h = 0
    min_w = width
    min_h = height
    max_wh = max(width, height)

    i = 0
    times_with_no_jumper = 0
    cropping_started = False
    frames = []

    ret, frame = cap.read()
    while ret:
        i += 1
        result = model(frame, verbose=False)
        xyxy_boxes = result[0].boxes.xyxy

        # ## DD3 only, additional labeling
        # if xyxy_boxes.shape[0] != 3:
        #     print(f"video {videoId}, frame {i-1} heeft {xyxy_boxes.shape[0]} boxes")
        
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

        if previous_x_min is None:
            previous_x_min = x1_min
            previous_x_max = x2_max
            previous_y_min = y1_min
            previous_y_max = y2_max

        if smooted_x1_min is None or smooted_x1_min == 1:
            smooted_x1_min = x1_min
            smooted_y1_min = y1_min
            smooted_x2_max = x2_max
            smooted_y2_max = y2_max
            secondary_smooted_x1_min = x1_min
            secondary_smooted_y1_min = y1_min
            secondary_smooted_x2_max = x2_max
            secondary_smooted_y2_max = y2_max
        else:
            iou = calculate_iou(
                new_x_min=x1_min, old_x_min=smooted_x1_min,
                new_x_max=x2_max, old_x_max=smooted_x2_max,
                new_y_min=y1_min, old_y_min=smooted_y1_min,
                new_y_max=y2_max, old_y_max=smooted_y2_max,
            )
            avgIOUlastNseconds = ((N * fps - 1) * avgIOUlastNseconds + iou) / N / fps 
            iou_threshold = avgIOUlastNseconds ** 4

            iou_with_previous = calculate_iou(
                new_x_min=x1_min, old_x_min=previous_x_min,
                new_x_max=x2_max, old_x_max=previous_x_max,
                new_y_min=y1_min, old_y_min=previous_y_min,
                new_y_max=y2_max, old_y_max=previous_y_max,
            )

            if i % 2 == 0:
                previous_cos_similarity = cos_similarity
                cos_similarity = calculate_cosine_similarity(
                    new_x_min=x1_min, old_x_min=previous_x_min,
                    new_x_max=x2_max, old_x_max=previous_x_max,
                    new_y_min=y1_min, old_y_min=previous_y_min,
                    new_y_max=y2_max, old_y_max=previous_y_max,
                )

            avgGradenLastKseconds = ((K * fps - 1) * avgGradenLastKseconds + cos_similarity) / K / fps 
            # avgGradenVerschilLastKseconds = ((K * fps - 1) * avgGradenVerschilLastKseconds + abs(cos_similarity- previous_cos_similarity)) / K / fps 

            # print(i, iou, iou_threshold)
            if iou > iou_threshold:
                smootval = smootval_stationary * (1 - abs(avgGradenLastKseconds)) + smootval_follow * abs(avgGradenLastKseconds)
                smootval_shrink = smootval_shrink_stationary * (1 - abs(avgGradenLastKseconds)) + smootval_shrink_follow * abs(avgGradenLastKseconds)
                # print(f"{i/fps:.2f} - {smootval:.3f} - {smootval_shrink:.3f} --- {iou > iou_threshold} : {iou:.3f} - 2nd = {iou_with_previous:.3f} - {1 - avgGradenLastKseconds ** 4:.3f} - {1 - avgGradenLastKseconds ** 2:.3f}     {cos_similarity:3.2f} - {avgGradenLastKseconds:.2f} - ")
                smooted_x1_min = int(smootval * smooted_x1_min + (1-smootval) * x1_min) if x1_min < smooted_x1_min else int(smootval_shrink * smooted_x1_min + (1-smootval_shrink) * x1_min)
                smooted_y1_min = int(smootval * smooted_y1_min + (1-smootval) * y1_min) if y1_min < smooted_x1_min else int(smootval_shrink * smooted_y1_min + (1-smootval_shrink) * y1_min)
                smooted_x2_max = int(smootval * smooted_x2_max + (1-smootval) * x2_max) if x2_max > smooted_x2_max else int(smootval_shrink * smooted_x2_max + (1-smootval_shrink) * x2_max)
                smooted_y2_max = int(smootval * smooted_y2_max + (1-smootval) * y2_max) if y2_max > smooted_y2_max else int(smootval_shrink * smooted_y2_max + (1-smootval_shrink) * y2_max)
            
            if i % 2 == 0:
                previous_x_min = x1_min
                previous_x_max = x2_max
                previous_y_min = y1_min
                previous_y_max = y2_max
                
            secondary_smooted_x1_min = int(smootval * secondary_smooted_x1_min + (1-smootval) * x1_min) if x1_min < secondary_smooted_x1_min else int(smootval_shrink * smooted_x1_min + (1-smootval_shrink) * x1_min)
            secondary_smooted_y1_min = int(smootval * secondary_smooted_y1_min + (1-smootval) * y1_min) if y1_min < secondary_smooted_x1_min else int(smootval_shrink * smooted_y1_min + (1-smootval_shrink) * y1_min)
            secondary_smooted_x2_max = int(smootval * secondary_smooted_x2_max + (1-smootval) * x2_max) if x2_max > secondary_smooted_x2_max else int(smootval_shrink * smooted_x2_max + (1-smootval_shrink) * x2_max)
            secondary_smooted_y2_max = int(smootval * secondary_smooted_y2_max + (1-smootval) * y2_max) if y2_max > secondary_smooted_y2_max else int(smootval_shrink * smooted_y2_max + (1-smootval_shrink) * y2_max)

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
            offset_x = abs(leftover_pixels_x)
        else:
            crop_x1 = leftover_pixels_x
            offset_x = 0
        
        leftover_pixels_x_right = width - (smooted_x2_max + offset_x)
        crop_x2 = width if leftover_pixels_x_right < 0 else min(smooted_x2_max + offset_x, width)

        leftover_pixels_y = smooted_y1_min - offset_y
        if leftover_pixels_y < 0:
            crop_y1 = 0
            offset_y = abs(leftover_pixels_y)
        else:
            crop_y1 = leftover_pixels_y
            offset_y = 0
        
        leftover_pixels_y_right = width - (smooted_y2_max + offset_y)
        crop_y2 = width if leftover_pixels_y_right < 0 else min(smooted_y2_max + offset_y, height)

        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        zeros = np.zeros((max_wh_jumpers, max_wh_jumpers, 3), dtype=np.uint8)
        zeros[offset_y:offset_y+(crop_y2-crop_y1), offset_x:offset_x+(crop_x2 - crop_x1)] = cropped_frame
        zeros = cv2.resize(zeros, (DIM, DIM))
        zeros = cv2.cvtColor(zeros, cv2.COLOR_BGR2RGB)

        frames.append(zeros)
        
        ret, frame = cap.read()

    # print("min width", min_w, "max_w", max_w)   
    # print("min_height", min_h, "max_h", max_h)
    # print(i)

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(videoOutputPath)
    print(f"Took {time.time()-start:.2f}s")

    # To browser runnable codec
    # os.system(f"ffmpeg -i {videoOutputPath} -vcodec libx264 {videoOutputPath}")

# %%
 

# %%


# %%



