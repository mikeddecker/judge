import os
import urllib.parse
from repos.videoRepo import random_hello

API_URL="127.0.0.1:8123"
VIDEO_FOLDER="/home/miked/Pictures/Screenshots"
SUPPORTED_VIDEO_FORMATS = [
    'webm',
    'jpg',
    'png',
] # Temporarily image formats

def browse_videos(subFolder = "", page = 1, limit = 10):

    return {
        "folders" : get_folders(subFolder=subFolder),
        "videos" : get_videos(subFolder=subFolder),
    }




## Help functions
def get_videos(subFolder = ""):
    """Returns the basic information needed in to browse in the video
    - preview_image : "API_URL"
    - video : "API_URL"
    - title : ""
    """
    # Read from the most inner nested to outside
    return list(
        map(
            lambda vid : {
                "title" : vid,
                "image-preview" : os.path.join(API_URL, "videoimage", subFolder, urllib.parse.quote(vid))
            },
            # Map file string title to
            filter(
                # Filter supported formats e.g. ".mp4"
                lambda f : f.split('.')[-1].lower() in SUPPORTED_VIDEO_FORMATS,
                filter(
                    # Filter files from files and subfolders
                    lambda content : os.path.isfile(os.path.join(VIDEO_FOLDER, subFolder, content)), 
                    os.listdir(os.path.join(VIDEO_FOLDER, subFolder))
                )
            )
        )
    )

def get_folders(subFolder = ""):
    return list(filter(lambda content : os.path.isdir(os.path.join(VIDEO_FOLDER, subFolder, content)), os.listdir(os.path.join(VIDEO_FOLDER, subFolder))))