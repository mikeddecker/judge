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

class DownloadRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        self.storageService = StorageService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def post(self):
        data = request.get_json()
        
        try:
            src = data.get('src').upper()
            url = data.get('URL')
            name = data.get('name')
            folderId = data.get('folderId')
            if self.folderService.exists_in_database(id=folderId):
                if src == 'YT':
                    try:
                        ValueHelper.check_raise_yt_url(url)
                    except ValueError as ve:
                        return Response(str(ve), status=400)
                    self.storageService.download_video(name=name, url=url, folderId=folderId)
                else:
                    Response(f"Source {src} not yet supported", status=400)
                pass
            else:
                return Response(f'FolderId {folderId} does not exist', status=400)
            return True, 200
        except Exception as e:
            return Response(str(e), status=500)
