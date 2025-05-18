import os
from dotenv import load_dotenv
from flask import send_file, Response, jsonify, request
from flask_restful import Resource, current_app
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper
import cv2

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
FOLDER_VIDEORESULTS = os.getenv("FOLDER_VIDEORESULTS")
CROPPED_VIDEOS_FOLDER = "cropped-videos"
CROPPED_VIDEOS_STATUSES = ["OK", "OK_NET_NIET_PERFECT", "SLECHT"]

def get_cropped_video_path(videoId, dim:int = 224):
    """Duplicated method from computer vision"""
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
        if os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                return Response(f.read())
        else:
            return '', 200

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
    
class VideoPredictionRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return ve, 404
        
        return self.videoService.getVideoPredictions(videoId), 200

class VideoImageRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        image_path = os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{videoId}.jpg")
        if not os.path.exists(image_path):
            image_path = f"/home/miked/Videos/images/0.png"
        with open(image_path, 'rb') as f:
            return Response(f.read(), mimetype='image/jpg')
        
    def post(self, videoId: int):
        print("@"*80)
        print("@"*80)
        print(videoId, type(videoId))
        if not self.videoService.exists_in_database(id=videoId):
            return f"VideoId {videoId} does not exist", 404
        
        videoinfo = self.videoService.get(videoId)
        frameNr = request.get_json()
        print(frameNr)

        cropped = True
        croptext = "_cropped" if cropped else ""

        videopath = os.path.join(STORAGE_DIR, videoinfo.get_relative_video_path()) if not cropped else get_cropped_video_path(videoId=videoId)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            return "Cannot open camera", 500
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
        res, frame = cap.read()
        filename = os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{videoId}_{frameNr}{croptext}.jpg")
        cv2.imwrite(filename, frame)
        return 'ok', 200
