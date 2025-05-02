import cv2
import math
import random
import numpy as np
import os
import sys
import random
from dotenv import load_dotenv

load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

class FrameLoader:
    def __init__(self, datarepo):
        self.VideoNames = datarepo.VideoNames
        self.VideoNames.index = self.VideoNames["id"]

    def __get_cropped_video_path(self, videoId, dim:int = 224):
        CROPPED_VIDEOS_FOLDER = 'cropped-videos'
        vpathUNK = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f'{dim}_{videoId}.mp4')
        vpathOK = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, "OK", f"{dim}_{videoId}.mp4")
        vpathNOK = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, "OK_NET_NIET_PERFECT", f"{dim}_{videoId}.mp4")
        vpathAlmostOK = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, "SLECHT", f"{dim}_{videoId}.mp4")
        
        vpath = ""
        if os.path.exists(vpathOK):
            vpath = vpathOK
        elif os.path.exists(vpathAlmostOK):
            vpath = vpathAlmostOK
        elif os.path.exists(vpathUNK):
            vpath = vpathUNK
        elif os.path.exists(vpathNOK):
            vpath = vpathNOK

        if not os.path.exists(vpath):
            raise ValueError("path does not exist", vpath)

        return vpath
        

    def get_frame_original(self, videoId, frameNr, dim, original_x, original_y, original_width, original_height, printId=False):
        if printId:
            print(frameNr)
        vpath = os.path.join(STORAGE_DIR, self.VideoNames.loc[videoId, "name"])
        cap = cv2.VideoCapture(vpath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_wh = max(width, height)
        scaled_width = dim if width == max_wh else int(dim * width / max_wh)
        scaled_height = dim if height == max_wh else int(dim * height / max_wh)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
        _, frame = cap.read()

        frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

        zeros = np.zeros((dim, dim, 3), dtype=np.uint8)
        offset_x = (dim - scaled_width) // 2
        offset_y = (dim - scaled_height) // 2
        scale_x = 1 if width == max_wh else width / height # multiplier for y values
        scale_y = 1 if height == max_wh else height / width
        
        # Place the resized frame on the canvas at the calculated offset
        zeros[offset_y:offset_y+scaled_height, offset_x:offset_x+scaled_width] = frame

        offset_x = (original_x * scale_x *dim + offset_x) /dim            
        offset_y = (original_y * scale_y *dim + offset_y) /dim
        y = [offset_x, offset_y, original_width * scale_x, original_height * scale_y]
        cap.release()
        return zeros, y
    
    def get_frame(self, videoId, frameNr, dim, original_x, original_y, original_width, original_height, printId=False):
        if printId:
            print(frameNr)
        vpath = os.path.join(STORAGE_DIR, self.VideoNames.loc[videoId, "name"])
        cap = cv2.VideoCapture(vpath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x1 = math.floor(original_x * width - width * original_width / 2)
        x2 = math.ceil(original_x * width + width * original_width / 2)
        y1 = math.floor(original_y * height - height * original_height / 2)
        y2 = math.ceil(original_y * height + height * original_height / 2)

        top_height = y1
        bottom_height = height - y2
        left_width = x1
        right_width = width - x2
        cut_top = int(math.pow(random.random(), 4) / 2 * top_height)
        cut_bottom = int(math.pow(random.random(), 4) / 2 * bottom_height)
        cut_left = int(math.pow(random.random(), 4) / 2 * left_width)
        cut_right = int(math.pow(random.random(), 4) / 2 * right_width)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
        _, frame = cap.read()
        frame = frame[cut_top:height-cut_bottom, cut_left:width - cut_right]
        height = frame.shape[0]
        width = frame.shape[1]
        x1 = x1 - cut_left
        x2 = x2 - cut_left
        y1 = y1 - cut_top
        y2 = y2 - cut_top
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width > height or box_height > width:
            zeros = np.zeros((max(width, height), max(width, height), 3), dtype=np.uint8)
            left_pad = 0 if width == max(width, height) else abs(height - width) // 2
            top_pad = 0 if height == max(width, height) else abs(height - width) // 2
            zeros[top_pad:top_pad+height, left_pad:left_pad+width] = frame
            height = zeros.shape[0]
            width = zeros.shape[1]
            frame = zeros
            x1 = x1 + left_pad
            x2 = x2 + left_pad
            y1 = y1 + top_pad
            y2 = y2 + top_pad
        else:
            horizontal_margin =  min(height, width) - box_width
            vertical_margin = min(height, width) - box_height
            assert horizontal_margin >= 0
            assert vertical_margin >= 0
            multiplier = random.random()
            assert multiplier <= 1 and multiplier >= 0
            left_shift = max(0, int(x1 - random.random() * horizontal_margin)) if min(height,width) == height else 0
            top_shift = max(0, int(y1 - random.random() * vertical_margin)) if min(height,width) == width else 0
            left_shift = min(left_shift, width - min(height, width))
            top_shift = min(top_shift, height - min(height, width))
            assert top_shift + min(height, width) - top_shift == left_shift+min(height, width) - left_shift
            assert left_shift+min(height, width) <= width
            assert top_shift+min(height, width) <= height
            frame = frame[top_shift:top_shift + min(height, width), left_shift: left_shift+min(height, width)]
            assert frame.shape[0] == frame.shape[1]
            x1 = x1 - left_shift
            x2 = x2 - left_shift
            y1 = y1 - top_shift
            y2 = y2 - top_shift

        # Random padding around
        r = random.random()
        if r < 0.2:
            bigger_dim = int((1 + r) * frame.shape[0])
            shift = int(frame.shape[0] / 2 * r)
            zeros = np.zeros((bigger_dim, bigger_dim, 3), dtype=np.uint8)
            zeros[shift:shift+frame.shape[0], shift:shift+frame.shape[1]] = frame
            frame = zeros

            x1 += shift
            x2 += shift
            y1 += shift
            y2 += shift
        
        # Random flip
        if random.random() < 0.5:
            frame = cv2.flip(frame, 1)
            x1 = frame.shape[0] - x1
            x2 = frame.shape[0] - x2

        center_x = (x2 + x1) / 2
        center_y = (y2 + y1) / 2
        x = center_x / frame.shape[1]
        y = center_y / frame.shape[0]

        box_width = x2 - x1
        box_height = y2 - y1

        w = box_width / frame.shape[1]
        h = box_height / frame.shape[0]

        frame = cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA)
        
        cap.release()
        return frame, [x, y, w, h]
    
    def get_frames(self, videoId, frameInfo, dim, printId=False):
        # On sampling: frameInfo = { 'frameNr' : (x,y,w,h) }
        # On loading:  frameInfo = { 'frameNr' : (loaded_frame, y) }
        if printId:
            print(frameInfo)
        vpath = os.path.join(STORAGE_DIR, self.VideoNames.loc[videoId, "name"])
        cap = cv2.VideoCapture(vpath)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_wh = max(width, height)
        scaled_width = dim if width == max_wh else int(dim * width / max_wh)
        scaled_height = dim if height == max_wh else int(dim * height / max_wh)

        loaded_frames = {}
        for frameNr, bbox in frameInfo.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
            _, frame = cap.read()
            frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

            original_x = bbox[0]
            original_y = bbox[1]
            original_width = bbox[2]
            original_height = bbox[3]
            
            zeros = np.zeros((dim, dim, 3), dtype=np.uint8)
            offset_x = (dim - scaled_width) // 2
            offset_y = (dim - scaled_height) // 2
            scale_x = 1 if width == max_wh else width / height # multiplier for y values
            scale_y = 1 if height == max_wh else height / width
            
            # Place the resized frame on the canvas at the calculated offset
            zeros[offset_y:offset_y+scaled_height, offset_x:offset_x+scaled_width] = frame

            offset_x = (original_x * scale_x *dim + offset_x) /dim            
            offset_y = (original_y * scale_y *dim + offset_y) /dim
            y = [offset_x, offset_y, original_width * scale_x, original_height * scale_y]
            loaded_frames[frameNr] = (zeros, y)

        cap.release()

        return loaded_frames
    
    def get_skill(self, videoId: int, dim: tuple[int, int],
                  start: int, end: int, timesteps: int, normalized: bool = True, augment=False, flip_image=False):
        vpath = os.path.join(STORAGE_DIR, 'cropped-videos', f'224_{videoId}.mp4')
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        _, frame = cap.read()

        frames_per_timestep = (end - start) / timesteps

        # Shift skill a little
        if normalized and augment and random.random() < 0.7:
            start = start - 2 * frames_per_timestep * random.random() + 2 * frames_per_timestep * random.random()
            end = end - 2 * frames_per_timestep * random.random() + 2 * frames_per_timestep * random.random()
            frames_per_timestep = (end - start) / timesteps

        # pad = False
        # if augment and random.random() < 0.6:
        #     padding = int(random.random() * 50)
        #     top = int(random.random() * padding)
        #     left = int(random.random() * padding)
        #     pad = True

        frames = []
        currentFrame = start
        while currentFrame < end and len(frames) < timesteps:
            if round(currentFrame) < int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
                # print("current", currentFrame, "| pos", cap.get(cv2.CAP_PROP_POS_FRAMES), '| len frames', len(frames))                
                # if augment and pad:
                #     zeros = np.full((frame.shape[0] + padding, frame.shape[1] + padding, frame.shape[2]), 127)
                #     zeros[top:top+frame.shape[0], left:left+frame.shape[1]] = frame
                #     zeros = zeros.astype(float)
                #     frame = cv2.resize(zeros, (frame.shape[0], frame.shape[1])).astype(int)

                if flip_image:
                    frame = cv2.flip(frame, 1)

                frame = frame if not normalized else (frame / 255)
                frames.append(frame)
                currentFrame += frames_per_timestep
            else:
                # print("current", currentFrame, "| pos", cap.get(cv2.CAP_PROP_POS_FRAMES), '| len frames', len(frames))
                _, frame = cap.read()
                continue

        assert len(frames) == timesteps, f"Something went wrong, frames doesn't have length of timesteps = {timesteps}, got {len(frames)}"
        
        return np.array(frames), flip_image
    
    def get_segment(self, videoId: int, dim: tuple[int, int],
                  start: int, end: int, normalized: bool = True, augment=False, channels_last=False):
        """Returns frames in interval [start, end["""
        vpath = self.__get_cropped_video_path(videoId=videoId, dim=dim[0])

        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        _, frame = cap.read()

        frames = []
        currentFrame = start
        while currentFrame < end:
            frame = frame if not normalized and frame is not None else (frame / 255)
            frames.append(frame)
            currentFrame += 1
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros(shape=(dim[0], dim[0], 3), dtype=int)
            if frame is None:
                print("@"*80)
                print(f"Read a None frame", videoId, currentFrame)
                print("@"*80)


        assert len(frames) == end-start, f"Something went wrong, frames doesn't have length of timesteps = {end-start}, got {len(frames)}"
        
        return np.array(frames) if channels_last else np.transpose(np.array(frames), (3, 0, 1, 2))

    def get_skill_torch(self, videoId: int, dim: tuple[int, int],
                  start: int, end: int, timesteps: int, normalized: bool = True, augment=False):
        DIM = dim[0]
        vpath = self.__get_cropped_video_path(videoId=videoId, dim=DIM)
        
        cap = cv2.VideoCapture(vpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = cap.read()

        frames_per_timestep = (end - start) / timesteps

        # Shift skill a little
        # if normalized and augment and random.random() < 0.7:
        #     start = start - 2 * frames_per_timestep * random.random() + 2 * frames_per_timestep * random.random()
        #     end = end - 2 * frames_per_timestep * random.random() + 2 * frames_per_timestep * random.random()
        #     frames_per_timestep = (end - start) / timesteps

        # pad = False
        # if augment and random.random() < 0.6:
        #     padding = int(random.random() * 50)
        #     top = int(random.random() * padding)
        #     left = int(random.random() * padding)
        #     pad = True

        frames = []
        currentFrame = start
        while len(frames) < timesteps:
            if round(currentFrame) <= int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
                # print("current", currentFrame, "| pos", cap.get(cv2.CAP_PROP_POS_FRAMES), '| len frames', len(frames))                
                # if augment and pad:
                #     zeros = np.full((frame.shape[0] + padding, frame.shape[1] + padding, frame.shape[2]), 127)
                #     zeros[top:top+frame.shape[0], left:left+frame.shape[1]] = frame
                #     zeros = zeros.astype(float)
                #     frame = cv2.resize(zeros, (frame.shape[0], frame.shape[1])).astype(int)

                # if flip_image:
                #     frame = cv2.flip(frame, 1)

                # frame = frame if not normalized else (frame / 255)
                frames.append(frame)
                currentFrame += frames_per_timestep
            else:
                # print("current", currentFrame, "| pos", cap.get(cv2.CAP_PROP_POS_FRAMES), '| len frames', len(frames))
                _, frame = cap.read()
                continue

        assert len(frames) == timesteps, f"Something went wrong, frames doesn't have length of timesteps = {timesteps}, got {len(frames)}"
        
        return np.transpose(np.array(frames), (3, 0, 1, 2))