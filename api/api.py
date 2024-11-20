import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from typing import Union, Optional
from managers.videoManager import browse_videos

VIDEO_FOLDER="/home/miked/Pictures/Screenshots"

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/browse")
def read_browse(folderName: str = ""):
    return {
        "folderName": folderName,
        "content" : browse_videos(subFolder=folderName)
        }
# Watch out with urlencodings.
# belgium/2024/food/fruit becomes belgium%2F2024%2Ffood%2Ffruit
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = 'NO no no no'):
#     return {"item_id": item_id, "q": q}


# Specify the path to your image
IMAGE_PATH = "/home/miked/Pictures/Screenshots/timetable.png"
VIDEO_PATH = "/home/miked/Pictures/Screenshots/Screencast.webm"

@app.get("/videoimage/{img_path}")
def get_image(img_path : str):
    full_img_path = os.path.join(VIDEO_FOLDER, img_path)
    print(full_img_path)
    if os.path.exists(full_img_path):
        return FileResponse(full_img_path)
    return {"error": "Image not found"}

# @app.get("/video/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = 'NO no no no'):
#     return {"item_id": item_id, "q": q}
