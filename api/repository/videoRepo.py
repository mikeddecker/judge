from domain.videoinfo import VideoInfo
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from flask_sqlalchemy import SQLAlchemy
from helpers.ValueHelper import ValueHelper
from repository.models import Video as VideoInfoDB, Folder as FolderDB
from repository.MapToDomain import MapToDomain
from repository.MapToDB import MapToDB
from typing import List

class VideoRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db

    # TODO : make width, height ... requirements
    def add(
            self, name: str, folder: Folder, frameLength: int,
            width=1920, height=1080, fps=30,
            training=True, qualitative=True, obstruction=False, private=True
        ) -> VideoInfo:
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        ValueHelper.check_raise_id(frameLength)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"Folder must be provided")
        new_video = VideoInfoDB(
            name = name,
            folderId = folder.Id,
            folder = MapToDB.map_folder(db=self.db, folder=folder),
            frameLength = frameLength,
            width = width,
            height = height,
            fps = fps,
            training = training,
            qualitative = qualitative,
            obstruction = obstruction,
            private = private
        )
        self.db.session.add(new_video)
        self.db.session.commit()
        return MapToDomain.map_video(new_video)
    
    def count(self) -> int:
        return self.db.session.query(VideoInfoDB).count()
    
    def exists(self, id: int) -> bool:
        ValueHelper.check_raise_id(id)
        return self.db.session.query(FolderDB.id).filter_by(id=id).scalar() is not None
    
    def exists_by_name(self, name: str, folder: Folder) -> bool:
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"folder must be provided")
        return self.db.session.query(VideoInfoDB).filter_by(name=name, folderId=folder.Id).scalar() is not None
        
    def get(self, id: int) -> VideoInfo:
        return MapToDomain.map_video(self.db.session.get(VideoInfoDB, ident=id))
    
    def get_videos(self, folderId: int) -> List[VideoInfo]:
        """Return videos in the given folder"""
        ValueHelper.check_raise_id(folderId)
        videosDB = self.db.session.query(VideoInfoDB).filter_by(folderId=folderId).all()
        return [MapToDomain.map_video(v) for v in videosDB]
    