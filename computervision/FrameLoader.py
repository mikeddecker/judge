import cv2
import os
import sys
from dotenv import load_dotenv

load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

class FrameLoader:
    def __init__(self, datarepo):
        self.VideoNames = datarepo.VideoNames
        self.VideoNames.index = self.VideoNames["id"]

    def get_frame(self, videoId, frameNr, keepWrongBGRColors=True):
        print(os.path.join(STORAGE_DIR, self.VideoNames.loc[videoId, "name"]))
        cap = cv2.VideoCapture(os.path.join(STORAGE_DIR, self.VideoNames.loc[videoId, "name"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
        print("frames")
        _, frame = cap.read()

        frame = frame if keepWrongBGRColors else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return frame
        
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