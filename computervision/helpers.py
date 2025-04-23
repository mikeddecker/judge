import cv2
import keras
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import pandas as pd

sys.path.append('..')

from api.helpers import ConfigHelper
from managers.FrameLoader import FrameLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot(imgs, bboxes=None, row_title=None, **imshow_kwargs):
    """
    Plot images in a grid with optional bounding boxes.

    imgs: list of images or list of lists of images (in BGR format).
    bboxes: list of bounding boxes corresponding to each image.
            Each bounding box is in the format [center_x, center_y, width, height], normalized (0 to 1).
            If None, no bounding boxes are drawn.
    row_title: optional list of titles for each row.
    **imshow_kwargs: additional arguments passed to `imshow`.
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = 2  # 2 images per row

    # Adjust the figure size to make the images bigger and fit 2 images per row
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(12, 6))
    
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            # Ensure img is in the correct format (BGR to RGB)
            if img.shape[-1] == 3:  # If the image has 3 channels (likely BGR from OpenCV)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            ax = axs[row_idx, col_idx]
            ax.imshow(img, **imshow_kwargs)
            
            # If bounding boxes are provided, draw them on the image
            if bboxes is not None:
                bbox = bboxes[row_idx * num_cols + col_idx]  # Get the bounding box for the current image
                if bbox is not None:
                    # Unpack the bounding box (center_x, center_y, width, height)
                    center_x, center_y, width, height = bbox

                    # Get the image dimensions
                    img_height, img_width = img.shape[:2]

                    # Convert the normalized bounding box to pixel coordinates
                    xmin = (center_x - width / 2) * img_width
                    ymin = (center_y - height / 2) * img_height
                    xmax = (center_x + width / 2) * img_width
                    ymax = (center_y + height / 2) * img_height

                    # Create a rectangle patch for the bounding box
                    rect = patches.Rectangle(
                        (xmin, ymin), xmax - xmin, ymax - ymin,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)  # Add the bounding box to the image
            ax.axis('off')

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    # No need for tight_layout() to avoid any unwanted resizing of images
    plt.show()

def my_mse_loss_fn(y_true, y_pred):
    """
    A custom MSE loss function where x and y positions are multiplied by 2,
    but w and h remain the same.
    """
    # Split y_true and y_pred into x, y, w, h components
    x_true, y_true, w_true, h_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x_pred, y_pred, w_pred, h_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    
    # Compute squared differences for each component (x, y, w, h)
    squared_difference_x = keras.ops.square(x_true - x_pred) * 2
    squared_difference_y = keras.ops.square(y_true - y_pred) * 2
    squared_difference_w = keras.ops.square(w_true - w_pred)
    squared_difference_h = keras.ops.square(h_true - h_pred)
    
    # Combine the squared differences (you could use mean or sum depending on your needs)
    total_squared_difference = squared_difference_x + squared_difference_y + squared_difference_w + squared_difference_h
    
    # Return the mean of the squared differences as the loss
    return keras.ops.mean(total_squared_difference)

def metric_mse_max_numeric_accuracy(max: int, y_true, y_pred):
    rounded_y_true = keras.ops.round(y_true * max)
    rounded_y_pred = keras.ops.round(y_pred * max)
    return keras.ops.mean(keras.ops.equal(rounded_y_pred, rounded_y_true))

def metric_mse_segmentation_close_accuracy(N, y_true, y_pred):
    rounded_y_true = keras.ops.round(y_true * N)
    rounded_y_pred = keras.ops.round(y_pred * N)
    return keras.ops.mean(keras.ops.equal(rounded_y_pred, rounded_y_true))

def off_by_0_1(y_true, y_pred):
    return metric_mse_segmentation_close_accuracy(10, y_true=y_true, y_pred=y_pred)

def off_by_0_2(y_true, y_pred):
    return metric_mse_segmentation_close_accuracy(5, y_true=y_true, y_pred=y_pred)

def off_by_0_33(y_true, y_pred):
    return metric_mse_segmentation_close_accuracy(3, y_true=y_true, y_pred=y_pred)

def iou(y_true, y_pred):
    """
    Calculate IoU loss between the true and predicted bounding boxes.

    y_true and y_pred should have the shape (batch_size, 4), where each element is
    [center_x, center_y, width, height].
    """
    # Convert (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
    true_xmin = y_true[..., 0] - 0.5 * y_true[..., 2]
    true_ymin = y_true[..., 1] - 0.5 * y_true[..., 3]
    true_xmax = y_true[..., 0] + 0.5 * y_true[..., 2]
    true_ymax = y_true[..., 1] + 0.5 * y_true[..., 3]

    pred_xmin = y_pred[..., 0] - 0.5 * y_pred[..., 2]
    pred_ymin = y_pred[..., 1] - 0.5 * y_pred[..., 3]
    pred_xmax = y_pred[..., 0] + 0.5 * y_pred[..., 2]
    pred_ymax = y_pred[..., 1] + 0.5 * y_pred[..., 3]

    # Calculate the intersection area
    inter_xmin = keras.ops.maximum(true_xmin, pred_xmin)
    inter_ymin = keras.ops.maximum(true_ymin, pred_ymin)
    inter_xmax = keras.ops.minimum(true_xmax, pred_xmax)
    inter_ymax = keras.ops.minimum(true_ymax, pred_ymax)

    inter_width = keras.ops.maximum(0.0, inter_xmax - inter_xmin)
    inter_height = keras.ops.maximum(0.0, inter_ymax - inter_ymin)
    intersection_area = inter_width * inter_height

    # Calculate the union area
    true_area = (true_xmax - true_xmin) * (true_ymax - true_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union_area = true_area + pred_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def load_skill_batch_X_torch(frameloader:FrameLoader, videoId:int, dim:tuple[int,int], frameStart:int, frameEnd:int, timesteps:int, normalized:bool, augment:bool):
    try:
        loaded_frames = frameloader.get_skill_torch(videoId, dim=dim, 
                                                    start=frameStart, 
                                                    end=frameEnd,
                                                    timesteps=timesteps,
                                                    normalized=normalized,
                                                    augment=augment)
        loaded_frames = torch.from_numpy(loaded_frames).float().to(device)  # [C, timesteps, H, W]
        flip_image = random.random() < 0.5 if augment else 0
        if flip_image:
            loaded_frames = torch.flip(loaded_frames, dims=(3,))
        return loaded_frames, flip_image

    except Exception as err:
        print(f"*"*80)
        print(f"Failed for videoId = {videoId}, frameStart = {frameStart}, frameEnd = {frameEnd}")
        print(str(err))
        print(f"*"*80)
        raise err

def load_skill_batch_y_torch(skillinfo_row, flip_turner:bool=False):
    """"skillinfo_row is a pandas dataframe row"""

    # Prepare targets - no batch dimension needed
    y = {}
    for key, value in ConfigHelper.get_discipline_DoubleDutch_config().items():
        if key == "Tablename":
            continue
            
        key_lower = key[0].lower() + key[1:]
        if flip_turner and key in ["Turner1", "Turner2"]:
            key = "Turner2" if key == "Turner1" else "Turner1"

        target_value = skillinfo_row[key_lower]
        
        if value[0] == "Categorical":
            # Convert to 0-based index and long tensor
            y[key] = torch.tensor(int(target_value) - 1, dtype=torch.long).to(device)
        elif value[0] == "Numerical":
            # Normalize and convert to float tensor
            normalized_value = target_value / value[2]
            y[key] = torch.tensor(normalized_value, dtype=torch.float).to(device)
        else:  # Boolean flags
            y[key] = torch.tensor(bool(target_value), dtype=torch.float).to(device)
            
    return y

def load_segment_batch_y_torch(frameStart:int, frameEnd:int, df_splitpoint_values:pd.DataFrame):
    y = df_splitpoint_values[(df_splitpoint_values['frameNr'] >= frameStart) & (df_splitpoint_values['frameNr'] < frameEnd)]['splitpoint'].to_numpy()
    return torch.from_numpy(y).float().to(device)

def load_segment_batch_X_torch(frameloader:FrameLoader, videoId:int, dim:tuple[int,int], frameStart:int, frameEnd:int, timesteps:int, normalized:bool, augment:bool=False):
    try:
        loaded_frames = frameloader.get_segment(videoId, dim=dim, 
                                                    start=frameStart, 
                                                    end=frameEnd,
                                                    normalized=normalized,
                                                    augment=augment)
        loaded_frames = torch.from_numpy(loaded_frames).float().to(device)  # [C, timesteps, H, W]
        flip_image = random.random() < 0.5 if augment else 0
        if flip_image:
            loaded_frames = torch.flip(loaded_frames, dims=(3,))
        return loaded_frames

    except Exception as err:
        print(f"*"*80)
        print(f"Failed for videoId = {videoId}, frameStart = {frameStart}, frameEnd = {frameEnd}")
        print(str(err))
        print(f"*"*80)
        raise err


def calculate_splitpoint_values(videoId: int, frameLength:int, df_Skills:pd.DataFrame, fps:float, Nsec_frames_around=1/6):
    """Creates a dataframe: 'videoId', 'frameNr', 'splitpoint'
    Where splitpoint is the value 0 -> 1 whether the video needs to be split at that point or not"""
    splitpoint_values = {
        'videoId' : [videoId for _ in range(frameLength)],
        'frameNr' : range(frameLength),
        'splitpoint' : [0 for _ in range(frameLength)],
    }

    frames_around_splitpoint = round(Nsec_frames_around * fps)
    for _, skillrow in df_Skills.iterrows():
        frameStart = skillrow["frameStart"]
        frameEnd = skillrow["frameEnd"]

        currentFrameStart = frameStart - frames_around_splitpoint
        currentFrameEnd = frameEnd - frames_around_splitpoint
        while currentFrameStart < frameStart + frames_around_splitpoint:
            framesApart = abs(currentFrameStart - frameStart)
            splitvalue = 1 - (framesApart / frames_around_splitpoint) ** 2
            splitvalue *= splitvalue

            currentFrameStart += 1
            currentFrameEnd += 1

            splitpoint_values['splitpoint'][currentFrameStart] = splitvalue
            if currentFrameEnd < frameLength:
                splitpoint_values['splitpoint'][currentFrameEnd] = splitvalue

    return pd.DataFrame(splitpoint_values)

def adaptSkillLabels(df_skills: pd.DataFrame, balancedType: str):
    if balancedType == 'jump_return_push_frog_other':
        df_skills['skill'] = np.where(
            df_skills['skill'] <= 5,
            df_skills['skill'],
            5
        )
        return df_skills
    elif balancedType == 'limit_10procent':
        return df_skills
    raise ValueError(f"Unrecognized type: {balancedType}")
    
def mapBalancedSkillIndexToLabel(balancedType: str, index:int):
    if balancedType == 'jump_return_push_frog_other':
        categories = {
            0: 'jump',
            1: 'return from power',
            2: 'pushup',
            3: 'frog',
            4: 'other',
        }
        return categories[index]
    
def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size