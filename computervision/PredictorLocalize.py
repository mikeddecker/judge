# %%
import os
import cv2
import shutil
import pandas as pd
import numpy as np

# from colorama import Fore, Style
from typing import List
from dotenv import load_dotenv
from DataRepository import DataRepository
from ultralytics import YOLO

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
CROPPED_VIDEOS_FOLDER = "cropped-videos"
# IMAGES_FOLDER = "images"
# LABELS_FOLDER = "labels"
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS")
VIDEO_IMAGE_PREVIEW_FOLDER = os.getenv("VIDEO_IMAGE_PREVIEW_FOLDER")
DIM = 224
PEOPLE = 3


print("STORAGE_DIR", STORAGE_DIR)
print("CROPPED_VIDEOS_FOLDER", CROPPED_VIDEOS_FOLDER)

os.makedirs(os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER), exist_ok=True)

def get_video_path(repo, videoId):
    return os.path.join(STORAGE_DIR, repo.VideoNames.loc[videoId, "name"])

def get_last_yolo_modelpath():
    modeltrainround = 4
    return f"/home/miked/code/judge/runs/detect/train{modeltrainround}" if modeltrainround > 0 else f"/home/miked/code/judge/runs/detect/train"

def predict_video(model, repo, videoId, saveFrameResults=True, saveCroppedVideo=True):
    raw_predictions_and_crops = {}

    videoPath = get_video_path(repo, videoId=videoId)
    videoOutputPath = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f"{DIM}_{videoId}.mp4")
    print("input", videoPath)
    print("output", videoOutputPath)

    cap = cv2.VideoCapture(videoPath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

    ret, frame = cap.read()
    result = model(frame, verbose=False)
    xyxy_boxes = result[0].boxes.xyxy
    print(xyxy_boxes)

    ###############
    # Predictions #
    ###############
    predictions_per_second = 2000
    videoPath = get_video_path(repo, videoId=videoId)
    videoOutputPath = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f"{DIM}_{videoId}.mp4")
    videoPath, videoOutputPath

    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_to_skip = fps // predictions_per_second
    frames_to_skip
    padding_x = width // 60
    padding_y = height // 50
    N = 3
    avgIOUlastNseconds = 0.0

    smootval = 0.85
    smootval_shrink = 0.945
    smooted_x1_min = None
    smooted_y1_min = None
    smooted_x2_max = None
    smooted_y2_max = None

    max_w = 0
    max_h = 0
    min_w = width
    min_h = height
    max_wh = max(width, height)

    i = 0
    times_with_no_jumper = 0
    cropping_started = False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(videoOutputPath, fourcc, fps, (DIM, DIM))
    ret, frame = cap.read()
    while ret:
        i += 1
        result = model(frame, verbose=False)
        xyxy_boxes = result[0].boxes.xyxy
        
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

        if smooted_x1_min is None or smooted_x1_min == 1:
            smooted_x1_min = x1_min
            smooted_y1_min = y1_min
            smooted_x2_max = x2_max
            smooted_y2_max = y2_max

        else:
            intersection_width = max(0, min(x2_max, smooted_x2_max) - max(x1_min, smooted_x1_min))
            intersection_height = max(0, min(y2_max, smooted_y2_max) - max(y1_min, smooted_y1_min))
            intersection_area = intersection_width * intersection_height
            area_new_box = (x2_max - x1_min) * (y2_max - y1_min)
            area_smooted_box = (smooted_x2_max - smooted_x1_min) * (smooted_y2_max - smooted_y1_min)
            union = area_smooted_box + area_new_box - intersection_area
            iou = intersection_area / union
            avgIOUlastNseconds = ((N * fps - 1) * avgIOUlastNseconds + iou) / N / fps 
            avgSqrd = avgIOUlastNseconds*avgIOUlastNseconds
            iou_threshold = avgSqrd * avgSqrd
            # print(i, iou, iou_threshold)
            if iou > iou_threshold:
                smooted_x1_min = int(smootval * smooted_x1_min + (1-smootval) * x1_min) if x1_min < smooted_x1_min else int(smootval_shrink * smooted_x1_min + (1-smootval_shrink) * x1_min)
                smooted_y1_min = int(smootval * smooted_y1_min + (1-smootval) * y1_min) if y1_min < smooted_x1_min else int(smootval_shrink * smooted_y1_min + (1-smootval_shrink) * y1_min)
                smooted_x2_max = int(smootval * smooted_x2_max + (1-smootval) * x2_max) if x2_max > smooted_x2_max else int(smootval_shrink * smooted_x2_max + (1-smootval_shrink) * x2_max)
                smooted_y2_max = int(smootval * smooted_y2_max + (1-smootval) * y2_max) if y2_max > smooted_y2_max else int(smootval_shrink * smooted_y2_max + (1-smootval_shrink) * y2_max)

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

        raw_predictions_and_crops[i] = {
            "videoId": videoId,
            "frameNr": i-1,
            "rawPredictions" : xyxy_boxes,
            "rawBoxes" : len(xyxy_boxes),
            "jumpersLocation" : { 
                "y1": crop_y1, 
                "y2": crop_y2,
                "x1": crop_x1,
                "x2": crop_x2,
            }
        }

        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        zeros = np.zeros((max_wh_jumpers, max_wh_jumpers, 3), dtype=np.uint8)
        zeros[offset_y:offset_y+(crop_y2-crop_y1), offset_x:offset_x+(crop_x2 - crop_x1)] = cropped_frame
        zeros = cv2.resize(zeros, (DIM, DIM))

        out.write(zeros)
            
        ret, frame = cap.read()

    print("min width", min_w, "max_w", max_w)   
    print("min_height", min_h, "max_h", max_h)
    print(i)
    print(framecrops)
    
    df = pd.DataFrame(framecrops)
    df.to_sql(name='Predictions_Localization', con=repo.con, if_exists='append', index=False)

    
    # Release everything if job is finished
    out.release()
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    repo = DataRepository()
    modelpath = get_last_yolo_modelpath()
    model = YOLO(os.path.join(modelpath, "weights", "best.pt"))
    videoId = 2278
    predict_video(model, repo, videoId)