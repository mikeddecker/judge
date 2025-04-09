import os
from dotenv import load_dotenv
from flask import send_file, Response, jsonify
from flask_restful import Resource, current_app
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
VIDEO_IMAGE_PREVIEW_FOLDER = os.getenv("VIDEO_IMAGE_PREVIEW_FOLDER")
CROPPED_VIDEOS_FOLDER = "cropped-videos"
CROPPED_VIDEOS_STATUSES = ["OK", "OK_NET_NIET_PERFECT", "SLECHT"]

class VideoRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404

        videoinfo = self.videoService.get(videoId)
        video_path = os.path.join(STORAGE_DIR, videoinfo.get_relative_video_path())
        with open(video_path, 'rb') as f:
            return Response(f.read())

class VideoRouterCropped(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404
        
        DIM = 224
        for status in CROPPED_VIDEOS_STATUSES:
            video_path = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, status, f"{DIM}_{videoId}.mp4")
            print(status, video_path)
            if os.path.exists(video_path):
                with open(video_path, 'rb') as f:
                    return Response(f.read())
        
        video_path = os.path.join(STORAGE_DIR, CROPPED_VIDEOS_FOLDER, f"{DIM}_{videoId}.mp4")
        print("bestaat het?", video_path, "@"*50,os.path.exists(video_path))
        with open(video_path, 'rb') as f:
            return Response(f.read())

class VideoInfoRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404

        videoinfo = self.videoService.get(videoId)
        return videoinfo.to_dict(), 200

class VideoImageRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        image_path = os.path.join(STORAGE_DIR, VIDEO_IMAGE_PREVIEW_FOLDER, f"{videoId}.jpg")
        if not os.path.exists(image_path):
            image_path = f"/home/miked/Videos/images/0.png"
        with open(image_path, 'rb') as f:
            return Response(f.read(), mimetype='image/jpg')