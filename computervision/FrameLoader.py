import cv2
import os
import numpy as np
import sys
from dotenv import load_dotenv

load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

class FrameLoader:
    def __init__(self, datarepo):
        self.VideoNames = datarepo.VideoNames
        self.VideoNames.index = self.VideoNames["id"]

    def get_frame(self, videoId, frameNr, dim, original_x, original_y, original_width, original_height, printId=False):
        # print(original_x, original_y, original_width, original_height)
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
    
    def get_frames(self, relative_path, frameNrs, keepWrongBGRColors=True):
        frames = {}
        cap = cv2.VideoCapture(os.path.join(STORAGE_DIR, relative_path))
        for frameNr in frameNrs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
            _, frame = cap.read()
            frame = frame if keepWrongBGRColors else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[frameNr] = frame

        cap.release()

        return frames