from flask import Response
from flask_restful import Resource
from services.folderService import FolderService
from services.videoService import VideoService
from services.storageService import StorageService

class StorageRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.storageService = StorageService()
        super().__init__(**kwargs)
    
    def get(self):
        try:
            return self.storageService.discover_drive_cached_version(deleteOrphans=False), 200
        except Exception as ve:
            return Response(str(ve), status=500)

class OrphanDeleterRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.storageService = StorageService()
        super().__init__(**kwargs)
    
    def get(self):
        try:
            return self.storageService.discover_drive_cached_version(deleteOrphans=True), 200
        except Exception as ve:
            return Response(str(ve), status=500)
