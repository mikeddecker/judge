import os
import time
from dotenv import load_dotenv
from flask import send_file
from flask_restful import Resource
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

class FolderRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, folderId: int=None):
        if folderId:
            try:
                ValueHelper.check_raise_id(folderId)
                if not self.folderService.exists_in_database(id=folderId):
                    return f"FolderId {folderId} does not exist", 404
            except ValueError as ve:
                return ve, 404
            f = self.folderService.get(folderId).to_dict()
            children = self.folderService.get_children(f["Id"])
            f["Children"] = []
            for c in children:
                f["Children"].append(c.to_dict())
            f["Videos"] = {}
            for vidinfo in self.videoService.get_videos(folderId=f["Id"]):
                f["Videos"][vidinfo.Id] = vidinfo.to_dict()
            f["VideoCount"] = len(f["Videos"].keys())
            return f, 200
        else:
            # Modify to represent the same output as get(folderId)
            return {
                "Id" : 0,
                "Name" : "root",
                "Children" : [f.to_dict() for f in self.folderService.get_root_folders()],
                "Parent" : None,
                "Videos" : dict(),
                "VideoCount" : 0,
            }, 200
