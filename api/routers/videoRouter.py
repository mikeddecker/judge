import os
from dotenv import load_dotenv
from flask_restful import Resource, current_app
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

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
        return videoinfo.to_dict(), 200
