import cv2
import os

from config import ENVS
from flask import Response, request
from flask_restful import Resource
from helpers.ValueHelper import ValueHelper
from services.folderService import FolderService
from services.videoService import VideoService

class VideoRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404

        videoinfo = self.videoService.get(videoId)
        video_path = os.path.join(ENVS.DIRS.VIDEOS, videoinfo.get_relative_video_path())
        with open(video_path, 'rb') as f:
            return Response(f.read())

class VideoRouterCropped(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404
        
        video_path = os.path.join(ENVS.DIRS.GENERATED_VIDEODATA, f"{videoId}_cropped.mp4")
        if os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                return Response(f.read())
        else:
            # TODO : check return message
            return '', 200

class VideoInfoRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404

        videoinfo = self.videoService.get(videoId)
        return videoinfo.to_dict(), 200
    
class VideoPredictionRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404
        
        return self.videoService.getVideoPredictions(videoId), 200

class VideoPredictionRouter_HasLocalizePredictions(Resource):
    def __init__(self, **kwargs):
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404
        
        return self.videoService.has_predicted_boxes(videoId), 200

class VideoPredictionRouter_GetLocalizePredictions(Resource):
    def __init__(self, **kwargs):
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404
        
        return self.videoService.load_predicted_boxes(videoId), 200

class VideoImageRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        image_path = os.path.join(ENVS.DIRS.GENERATED_VIDEODATA, f"{videoId}", f"{videoId}.jpg")
        if not os.path.exists(image_path):
            image_path = f"/home/miked/Videos/images/0.png"
        with open(image_path, 'rb') as f:
            return Response(f.read(), mimetype='image/jpg')
        
    def post(self, videoId: int):
        if not self.videoService.exists_in_database(id=videoId):
            return f"VideoId {videoId} does not exist", 404
        
        videoinfo = self.videoService.get(videoId)
        frameNr = request.get_json()

        cropped = True
        croptext = "_cropped" if cropped else ""

        raise NotImplementedError('Disabled')

        videopath = 'TODO' # TODO
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            return "Cannot open camera", 500
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
        res, frame = cap.read()
        filename = 'TODO' # TODO
        cv2.imwrite(filename, frame)
        return 'ok', 200
