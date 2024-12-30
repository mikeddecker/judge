import os
from dotenv import load_dotenv
from flask import send_file, Response, request
from flask_restful import Resource, current_app
from domain.frameinfo import FrameInfo
from services.folderService import FolderService
from services.videoService import VideoService
from services.storageService import StorageService
from helpers.ValueHelper import ValueHelper

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

class StorageRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        self.storageService = StorageService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self):
        try:
            return self.storageService.discover_drive_cached_version(), 200
        except Exception as ve:
            return Response(str(ve), status=500)

