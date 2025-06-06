import os
import cv2
import json
import math
import shutil
import pandas as pd
import numpy as np
import time
from moviepy import ImageSequenceClip

# from colorama import Fore, Style
from helpers import get_localize_strategy_list
from typing import List
from dotenv import load_dotenv
from managers.DataRepository import DataRepository
from pprint import pprint
from ultralytics import YOLO

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS")
FOLDER_VIDEORESULTS = os.getenv("FOLDER_VIDEORESULTS")

COLUMNS = [ 'xmin', 'xmax', 'ymin', 'ymax' ]
DIM = 224

strategyparams = {
    'raw' : {
        'N' : 3
    },
    'cosine' : {
        'N' : 3
    },
    'smoothing': {
        'N' : 3,
        'smoothval' : 0.86,
        'smoothval_shrink': 0.92,
    },
    'smoothing_skip_small_iou': {
        'N' : 3,
        'smoothval' : 0.86,
        'smoothval_shrink': 0.92,
    }
}


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

def calculate_iou_df(df_a, df_b):
    xa, ya, wa, ha = df_a['x'], df_a['y'], df_a['width'], df_a['height']
    xb, yb, wb, hb = df_b['x'], df_b['y'], df_b['width'], df_b['height']

    # Min and max values
    xa_min, ya_min = xa - wa / 2, ya - ha / 2
    xa_max, ya_max = xa + wa / 2, ya + ha / 2

    xb_min, yb_min = xb - wb / 2, yb - hb / 2
    xb_max, yb_max = xb + wb / 2, yb + hb / 2

    # Intersection coordinates
    inter_x_min = np.maximum(xa_min, xb_min)
    inter_y_min = np.maximum(ya_min, yb_min)
    inter_x_max = np.minimum(xa_max, xb_max)
    inter_y_max = np.minimum(ya_max, yb_max)

    # Intersection area
    inter_width = np.maximum(0, inter_x_max - inter_x_min)
    inter_height = np.maximum(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    # Union area
    area1 = wa * ha
    area2 = wb * hb
    union = area1 + area2 - intersection

    # IoU calculation
    return np.where(union > 0, intersection / union, 0)


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
    """ChatGPT assisted"""
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

def calculate_smoothed_values(strat:str, params: dict, previous_values:dict, i:int, xmin:float, xmax:float, ymin:float, ymax:float):
    """Returns new coordinates xmin, xmax, ymin, ymax"""
    match (strat):
        case 'raw':
            return xmin, xmax, ymin, ymax
        case 'smoothing':
            FPS = params['FPS']
            smoothval = params['smoothval']
            smoothval_shrink = params['smoothval_shrink']
            next_xmin = int(smoothval * previous_values['xmin'][-1] + (1-smoothval) * xmin if xmin < previous_values['xmin'][-1] else smoothval_shrink * previous_values['xmin'][-1] + (1-smoothval_shrink) * xmin)
            next_xmax = int(smoothval * previous_values['xmax'][-1] + (1-smoothval) * xmax if xmax > previous_values['xmax'][-1] else smoothval_shrink * previous_values['xmax'][-1] + (1-smoothval_shrink) * xmax)
            next_ymin = int(smoothval * previous_values['ymin'][-1] + (1-smoothval) * ymin if ymin < previous_values['xmin'][-1] else smoothval_shrink * previous_values['xmin'][-1] + (1-smoothval_shrink) * ymin)
            next_ymax = int(smoothval * previous_values['ymax'][-1] + (1-smoothval) * ymax if ymax > previous_values['ymax'][-1] else smoothval_shrink * previous_values['ymax'][-1] + (1-smoothval_shrink) * ymax)
            return next_xmin, next_xmax, next_ymin, next_ymax
        case 'cosine':
            raise NotImplementedError()
            # stratparams['avgIOUlastNseconds'] = 0.0
            # stratparams['secondary_avgIOUlastNseconds'] = 0.0
            # stratparams['cos_similarity'] = 0
            # stratparams['avgGradenLastKseconds'] = 0
            # stratparams['previouscos_similarity'] = 0
            # stratparams['avgGradenVerschilLastKseconds'] = 0

        case 'smoothing_skip_small_iou':
            iou = calculate_iou(
                new_x_min=xmin, old_x_min=previous_values['xmin'][-1],
                new_x_max=xmax, old_x_max=previous_values['xmax'][-1],
                new_y_min=ymin, old_y_min=previous_values['xmin'][-1],
                new_y_max=ymax, old_y_max=previous_values['ymax'][-1],
            )
            FPS = params['FPS']
            smoothval = params['smoothval']
            smoothval_shrink = params['smoothval_shrink']
            if iou < 0.90 * (FPS / (FPS + 0 if 'frames_skipped' not in params.keys() else params['frames_skipped'])):
                next_xmin = int(smoothval * previous_values['xmin'][-1] + (1-smoothval) * xmin if xmin < previous_values['xmin'][-1] else smoothval_shrink * previous_values['xmin'][-1] + (1-smoothval_shrink) * xmin)
                next_xmax = int(smoothval * previous_values['xmax'][-1] + (1-smoothval) * xmax if xmax > previous_values['xmax'][-1] else smoothval_shrink * previous_values['xmax'][-1] + (1-smoothval_shrink) * xmax)
                next_ymin = int(smoothval * previous_values['ymin'][-1] + (1-smoothval) * ymin if ymin < previous_values['xmin'][-1] else smoothval_shrink * previous_values['xmin'][-1] + (1-smoothval_shrink) * ymin)
                next_ymax = int(smoothval * previous_values['ymax'][-1] + (1-smoothval) * ymax if ymax > previous_values['ymax'][-1] else smoothval_shrink * previous_values['ymax'][-1] + (1-smoothval_shrink) * ymax)
                return next_xmin, next_xmax, next_ymin, next_ymax
            else:
                params['frames_skipped'] = 1 if 'frames_skipped' not in params.keys() else params['frames_skipped'] + 1
                return previous_values['xmin'][-1], previous_values['xmax'][-1], previous_values['ymin'][-1], previous_values['ymax'][-1]
        case _:
            raise NotImplementedError(f"Unrecognized strat ({strat})")
    

def localize_jumpers(
        model: YOLO, repo: DataRepository, modelname: str,
        videoId: int, dim: int, strategies: list, stratparams: dict,
        save_as_JSON=True, save_as_mp4=False, 
        padding=False
    ):
    """Returns df_coordinates containing x, y, width, height"""
    start = time.time()

    videoPath = repo.get_video_path(videoId=videoId)
    rawPredictedBoxesPath = os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{videoId}_raw_boxes_{modelname}.json")
    strat_model_name = f"{videoId}_crop_d{dim}_{modelname}" # TODO : include strat
    videoOutputPath = os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{strat_model_name}.mp4")

    # if os.path.exists(videoOutputPath):
    #     return None

    bad_results = [] # TODO : List containing videocrops deemed bad
    if strat_model_name in bad_results:
        return None

    cap = cv2.VideoCapture(videoPath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_wh = max(width, height)
    percent_padding_x = 0.08 if padding else 0
    percent_padding_y = 0.06 if padding else 0
    padding_x = percent_padding_x * dim
    padding_y = percent_padding_y * dim

    for s in strategies:
        stratparams[s]['FPS'] = fps
    
    smoothed_values = {s: {
        'xmin' : [],
        'xmax' : [],
        'ymin' : [],
        'ymax' : [],
    } for s in strategies}

    times_with_no_jumper = 0
    frames = []
    predicted_boxes = []

    if save_as_mp4:
        # Parameter initialisation for mp4
        max_w = 0
        max_h = 0
        min_w = width
        min_h = height
        max_wh = max(width, height)
    
    print(f"Predicting video {videoId}")

    i = 0
    ret, frame = cap.read()
    while ret:
        result = model(frame, verbose=False)
        xyxy_boxes = result[0].boxes.xyxy
        predicted_boxes.append(xyxy_boxes.tolist())
        
        if xyxy_boxes.shape[0] > 0:
            xmin = max(0, int(xyxy_boxes[:, 0].min().item()) - padding_x)
            ymin = max(0, int(xyxy_boxes[:, 1].min().item()) - padding_y)
            xmax = min(width, int(xyxy_boxes[:, 2].max().item()) + padding_x)
            ymax = min(height, int(xyxy_boxes[:, 3].max().item()) + padding_y)
            times_with_no_jumper = 0
        else:
            xmin = 0
            ymin = 0
            xmax = width
            ymax = height
            times_with_no_jumper += 1

        for s in strategies:
            if i == 0:
                smoothed_values[s]['xmin'].append(int(xmin))
                smoothed_values[s]['xmax'].append(int(xmax))
                smoothed_values[s]['ymin'].append(int(ymin))
                smoothed_values[s]['ymax'].append(int(ymax))
            else:
                smoothed_xmin, smoothed_xmax, smoothed_ymin, smoothed_ymax = calculate_smoothed_values(strat=s, params=stratparams[s], previous_values=smoothed_values[s], i=i, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                smoothed_values[s]['xmin'].append(smoothed_xmin)
                smoothed_values[s]['xmax'].append(smoothed_xmax)
                smoothed_values[s]['ymin'].append(smoothed_ymin)
                smoothed_values[s]['ymax'].append(smoothed_ymax)

        if save_as_mp4:
            if len(strategies) > 1:
                raise ValueError(f"Strategies may only have 1 item if saving as mp4")

            smoothed_xmin = int(smoothed_values[s]['xmin'][-1])
            smoothed_xmax = int(smoothed_values[s]['xmax'][-1])
            smoothed_ymin = int(smoothed_values[s]['ymin'][-1])
            smoothed_ymax = int(smoothed_values[s]['ymax'][-1])
            w_jumpers = smoothed_xmax - smoothed_xmin
            h_jumpers = smoothed_ymax - smoothed_ymin
            
            # TODO : check, might not be needed
            max_w = max(max_w, w_jumpers)
            max_h = max(max_h, h_jumpers)
            min_w = min(min_w, w_jumpers)
            min_h = min(min_h, h_jumpers)


            max_wh_jumpers = max(w_jumpers, h_jumpers)
            offset_x = (max_wh_jumpers - w_jumpers) // 2
            offset_y = (max_wh_jumpers - h_jumpers) // 2

            leftover_pixels_x = smoothed_xmin - offset_x
            if leftover_pixels_x < 0:
                crop_x1 = 0
                offset_x = abs(leftover_pixels_x)
            else:
                crop_x1 = leftover_pixels_x
                offset_x = 0
            
            leftover_pixels_x_right = width - (smoothed_xmax + offset_x)
            crop_x2 = width if leftover_pixels_x_right < 0 else min(smoothed_xmax + offset_x, width)

            leftover_pixels_y = smoothed_ymin - offset_y
            if leftover_pixels_y < 0:
                crop_y1 = 0
                offset_y = abs(leftover_pixels_y)
            else:
                crop_y1 = leftover_pixels_y
                offset_y = 0
            
            leftover_pixels_y_right = width - (smoothed_ymax + offset_y)
            crop_y2 = width if leftover_pixels_y_right < 0 else min(smoothed_ymax + offset_y, height)

            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            zeros = np.zeros((max_wh_jumpers, max_wh_jumpers, 3), dtype=np.uint8)
            zeros[offset_y:offset_y+(crop_y2-crop_y1), offset_x:offset_x+(crop_x2 - crop_x1)] = cropped_frame
            zeros = cv2.resize(zeros, (dim, dim))
            zeros = cv2.cvtColor(zeros, cv2.COLOR_BGR2RGB)

            frames.append(zeros)
            
        i += 1
        ret, frame = cap.read()

    with open(rawPredictedBoxesPath, "w") as fp:
        json.dump(predicted_boxes, fp, indent=4)

    cap.release()
    cv2.destroyAllWindows()

    if save_as_mp4:
        strat_model_name = f"{videoId}_crop_d{dim}_{modelname}_{s}" # TODO : include strat
        videoOutputPath = os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{strat_model_name}.mp4")

        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(videoOutputPath)

    if save_as_mp4 and save_as_JSON:
        with open(os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{strat_model_name}.json"), 'w') as f:
            json.dump(smoothed_values, f, sort_keys=True, indent=4)    

    for s in strategies:
        # Transform absolute XYXY to relative xywh
        smoothed_values[s] = pd.DataFrame(smoothed_values[s])
        smoothed_values[s]['x'] = (smoothed_values[s]['xmin'] + smoothed_values[s]['xmax']) / 2 / width
        smoothed_values[s]['y'] = (smoothed_values[s]['ymin'] + smoothed_values[s]['ymax']) / 2 / height
        smoothed_values[s]['width'] = (smoothed_values[s]['xmax'] - smoothed_values[s]['xmin']) / width
        smoothed_values[s]['height'] = (smoothed_values[s]['ymax'] - smoothed_values[s]['ymin']) / height

    return smoothed_values 

def validate_localize(modeldir: str, repo: DataRepository, modelname: str):
    """Validates localize methods on a specific run"""
    strategies = get_localize_strategy_list()
    strategies = ['raw', 'smoothing', 'smoothing_skip_small_iou']

    df_videos_with_boxes = repo.get_videos_having_boxes_of_type(type=1).sample(frac=1.0)
    total_frames = df_videos_with_boxes['frameLength'].sum()
    videoIds = df_videos_with_boxes['id'].tolist()
    print("Total frames", total_frames)

    model = YOLO(os.path.join(modeldir, "weights", "best.pt"))
    saveAsMp4 = False
    saveAsJSON = True

    full_team_relative_boxes_both = {
        'train': repo.get_framelabels(train_test_val='train', type=1),
        'val' : repo.get_framelabels(train_test_val='val', type=1),
    }

    min_ious = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    min_iou_text = "min_Iou_gt_"
    min_iou_dict = { f"{min_iou_text}{min_iou:.1f}" : set() for min_iou in min_ious }
    ious_all = {
        s: { 
            tv: {
                **{ 'sum': 0, 'min': 1, 'max': 0, 'avg': 0, 'total': 0, 'videoIds': set() },
                **min_iou_dict
            } for tv in ['train', 'val']
        } for s in strategies
    }
    invalid_frames = []
    errors = []

    valstart = time.time()
    completed_videoIds = []
    for videoId in videoIds:
        train_or_val = 'val' if videoId % 10 == 5 else 'train'
        full_team_relative_boxes = full_team_relative_boxes_both[train_or_val]
        full_team_relative_boxes_of_videoId = full_team_relative_boxes[full_team_relative_boxes['videoId'] == videoId]
        if len(full_team_relative_boxes_of_videoId) == 0:
            continue

        try:

            # df_coordinates contains both:
            # Xmin Ymin Xmax Ymax
            # and relative x y w h
            df_coordinates = localize_jumpers(
                model=model,
                modelname=modelname,
                repo=repo,
                videoId=videoId,
                dim=DIM,
                strategies=strategies,
                stratparams=strategyparams,
                save_as_JSON=saveAsJSON,
                save_as_mp4=saveAsMp4,
                padding=False
            )

            ##################################
            # Validate localized coordinates #
            ##################################
            # Dict with relative boxes
            frameNrs = full_team_relative_boxes_of_videoId['frameNr']
            if max(frameNrs) >= len(df_coordinates['raw']):
                print("Invalid framecounts... skipping videoId, maxFrameNr, count", videoId, max(frameNrs), len(df_coordinates['raw']))
                invalid_frames.append({
                    'videoId' : videoId,
                    'num_boxes' : len(df_coordinates['raw']),
                    'maxFrameNr': max(frameNrs),
                })
                # TODO : fix or delete videos (videos are of a particular competition, where it seems a conversion mistake has been made from m2ts to mp4)
                continue

            for s in strategies:
                predicted_relative_boxes = df_coordinates[s].iloc[frameNrs]

                full_team_relative_boxes_of_videoId.index = full_team_relative_boxes_of_videoId['frameNr']
                assert predicted_relative_boxes.shape[0] == full_team_relative_boxes_of_videoId.shape[0]

                ious_video = calculate_iou_df(predicted_relative_boxes, full_team_relative_boxes_of_videoId)
                
                ious_all[s][train_or_val]['sum'] += ious_video.sum()
                ious_all[s][train_or_val]['total'] += len(ious_video)
                ious_all[s][train_or_val]['min'] = min(ious_all[s][train_or_val]['min'], ious_video.min())
                ious_all[s][train_or_val]['max'] = max(ious_all[s][train_or_val]['max'], ious_video.max())
                ious_all[s][train_or_val]['avg'] = ious_all[s][train_or_val]['sum'] / ious_all[s][train_or_val]['total']
                ious_all[s][train_or_val]['videoIds'].add(videoId)
                ious_all[s][train_or_val]['videos'] = len(ious_all[s][train_or_val]['videoIds'])
                for min_iou in min_ious:
                    ious_all[s][train_or_val][f"{min_iou_text}{min_iou:.1f}"].add(videoId)

        except Exception as e:
            raise e
        finally:
            print(f"Error count", len(invalid_frames))

        completed_videoIds.append(videoId)
        # ious_all[s][train_or_val]['avg'] = ious_all[s][train_or_val]['sum'] / ious_all[s][train_or_val]['total']
        frames_predicted = df_videos_with_boxes[df_videos_with_boxes['id'].isin(completed_videoIds)]['frameLength'].sum()
        expected_end = (time.time() - valstart) / frames_predicted * total_frames
        seconds_left = (time.time() - valstart) / frames_predicted * (total_frames- frames_predicted)
        print(f"Currently {frames_predicted} / {total_frames} frames ({frames_predicted/total_frames*100:.1f}%) - elapsed {time.time()-valstart:.2f}s - estimated total time target {expected_end:.0f}s ---> seconds left: {seconds_left:.0f}s")
    
    for e in errors:
        print(e)
    
    print(f"Invalid frames")
    print(pd.DataFrame(invalid_frames))
    with open(os.path.join(modeldir, 'invalid_videos_framecount_and_effictive_frames_not_matching.json'), 'w') as f:
        json.dump(invalid_frames, f, sort_keys=True, default=str)    
    
    print(f"Took {time.time()-valstart:.2f}s")
    


    # Save validation
    with open(os.path.join(modeldir, 'localize_ious.json'), 'w') as f:
        json.dump(ious_all, f, sort_keys=True, default=str)



def predict_and_save_locations(modeldir: str, repo: DataRepository, modelname: str, videoIds: int):
    """Validates localize methods on a specific run"""

    strategies = ['smoothing']

    model = YOLO(os.path.join(modeldir, "weights", "best.pt"))
    valstart = time.time()
    completed_videoIds = []
    for videoId in videoIds:
        try:
            # df_coordinates contains both:
            # Xmin Ymin Xmax Ymax
            # and relative x y w h
            df_coordinates = localize_jumpers(
                model=model,
                modelname=modelname,
                repo=repo,
                videoId=videoId,
                dim=DIM,
                strategies=strategies,
                stratparams=strategyparams,
                save_as_JSON=True,
                save_as_mp4=True,
                padding=True
            )

        except Exception as e:
            raise e

        completed_videoIds.append(videoId)
        if len(videoIds) > 1:
            print(f"Completed {len(completed_videoIds)}/{len(videoIds)} videos")    
    