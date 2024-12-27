import os
from domain.folder import Folder
from helpers.ValueHelper import ValueHelper
from repository.db import db
from repository.videoRepo import VideoRepository

API_URL="127.0.0.1:8123"
VIDEO_FOLDER="/home/miked/Videos"
SUPPORTED_VIDEO_FORMATS = [
    'webm',
    'mp4'
    'jpg',
    'png',
] # Temporarily media formats

class VideoService:
    PROPERTIES = [
        "VideoRepo",
        "StorageFolder",
    ]
    def __init__(self, storage_folder: str):
        ValueHelper.check_raise_string(storage_folder)
        self.VideoRepo = VideoRepository(db=db)

        if not os.path.exists(storage_folder):
            raise NotADirectoryError(f"StorageFolder {storage_folder} does not exist")
        self.StorageFolder = storage_folder
        
    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name in self.PROPERTIES:
                raise AttributeError(f"Cannot modify {name} once it's set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def add(self, name: str, folder: Folder):
        """Adds the given video to the database
        
        TODO : After localization; add meta information loader
        TODO : nice to have, add warning if name is double
        """
        if self.exists_in_database(name=name, folder=folder):
            raise LookupError(f"Video {name} already added in the database")
        if self.exists_path_on_drive(name=name, folder=folder):
            raise ValueError(f"Video {name} does not exist in {folder.get_relative_path()}")

        return self.VideoRepo.add(
            name=name,
            folder=folder
        )
    
    def exists_in_database(self, id: int = None, name: str = None, folder: Folder = None):
        """Check existence in database
        If id provided, ignore name and folder"""
        if id:
            ValueHelper.check_raise_id(id)
            return self.VideoRepo.exists(id=id)
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        return self.VideoRepo.exists_by_name(name=name, folder=folder)
    
    def exists_path_on_drive(self, name: str, folder: Folder):
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        return os.path.exists(os.path.join(self.StorageFolder, folder.get_relative_path(), name))

    def get(self, id: int):
        """Get video with the corresponding Id"""

    def get_videos(self, folderId: int):
        """Returns videos in the given folder"""
        raise NotImplementedError("")
