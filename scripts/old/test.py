import os
import cv2
from dotenv import load_dotenv
load_dotenv()
STORAGE_DIR = '/media/miked/Elements/Judge/FINISHED-DB-READY'
print(STORAGE_DIR)
videopath = os.path.join(STORAGE_DIR, 'competition', 'belgium', 'DD3', 'bk-handles-dd3-2024-junioren-mixed-j1.MP4')
videopath = '/media/miked/Elements/Judge/2022-arne-groot-bk/test/vlc-record-2022-05-26-15h47m10s-dshow___-.mp4'
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
