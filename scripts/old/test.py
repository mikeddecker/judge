import os
import cv2
from dotenv import load_dotenv
load_dotenv()
STORAGE_DIR = '/media/miked/Elements/Judge/FINISHED-DB-READY'
print(STORAGE_DIR)
videopath = os.path.join(STORAGE_DIR, 'competition', 'belgium', 'DD3', 'bk-handles-dd3-2024-junioren-mixed-j1.MP4')
videopath = '/home/miked/Documents/C0003-00.02.47.040-00.04.04.800-seg02.mov'
cap = cv2.VideoCapture(videopath)
if not cap.isOpened():
    raise IOError("Cannot open camera")
info = {}
info["fps"] = cap.get(cv2.CAP_PROP_FPS)
info["frameLength"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(info)
cap.release()
cv2.destroyAllWindows()
