import math
from domain.videoinfo import VideoInfo
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.frameinfo import FrameInfo
from flask_sqlalchemy import SQLAlchemy
from helpers.ValueHelper import ValueHelper
from repository.models import Video as VideoInfoDB, Folder as FolderDB, FrameLabel
from repository.MapToDomain import MapToDomain
from repository.MapToDB import MapToDB
from typing import List

class VideoRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db

    # TODO : make width, height ... requirements
    def add(
            self, name: str, folder: Folder, frameLength: int,
            width: float, height: float, fps: float,
            training=True, qualitative=True, obstruction=False, private=True,
            srcinfo:str=None
        ) -> VideoInfo:
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        ValueHelper.check_raise_id(frameLength)
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"Width, height, fps must be > 0", width, height, fps)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"Folder must be provided")
        new_video = VideoInfoDB(
            name = name,
            folderId = folder.Id,
            folder = self.db.session.get(FolderDB, ident=folder.Id),
            frameLength = frameLength,
            width = width,
            height = height,
            fps = fps,
            training = training,
            qualitative = qualitative,
            obstruction = obstruction,
            private = private,
            sourceInfo = srcinfo,
        )
        self.db.session.add(new_video)
        self.db.session.commit()
        return MapToDomain.map_video(new_video)
    
    def add_frameInfo(self, frameInfo: FrameInfo, video: VideoInfo):
        ValueHelper.check_raise_frameNr(frameInfo.FrameNr)
        ValueHelper.check_raise_id(video.Id)
        frame_label_DB = MapToDB.map_frameInfo(video=video, frameInfo=frameInfo)
        self.db.session.add(frame_label_DB)
        self.db.session.commit()

    def count(self) -> int:
        return self.db.session.query(VideoInfoDB).count()
    
    def exists(self, id: int) -> bool:
        ValueHelper.check_raise_id(id)
        return self.db.session.query(VideoInfoDB).filter_by(id=id).scalar() is not None
    
    def exists_by_name(self, name: str, folder: Folder) -> bool:
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"folder must be provided")
        return self.db.session.query(VideoInfoDB).filter_by(name=name, folderId=folder.Id).scalar() is not None
    
    def exists_frameInfo(self, videoId: int, frameNr: int):
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(frameNr)
        return self.db.session.query(FrameLabel).filter_by(videoId=videoId, frameNr=frameNr).first() is not None
  
    def delete(self, id: int):
        # TODO : check if no frames or skills are connected
        ValueHelper.check_raise_id(id)
        videoInfoDB = self.db.session.get(VideoInfoDB, ident=id)
        self.db.session.delete(videoInfoDB)
        self.db.session.commit()

    def get(self, id: int) -> VideoInfo:
        return MapToDomain.map_video(self.db.session.get(VideoInfoDB, ident=id))
    
    def get_videos(self, folderId: int) -> List[VideoInfo]:
        """Return videos in the given folder"""
        ValueHelper.check_raise_id(folderId)
        videosDB = self.db.session.query(VideoInfoDB).filter_by(folderId=folderId).all()
        return [MapToDomain.map_video(v) for v in videosDB]
    
    def has_frames(self, videoId):
        return self.db.session.query(FrameLabel).filter_by(videoId=videoId).count() > 0

    def is_already_downloaded(self, src_info: str) -> bool:
        return self.db.session.query(VideoInfoDB).filter_by(sourceInfo=src_info).count() > 0

    def remove_frameInfo(self, frameNr: int, videoId: int, frameinfo: FrameInfo):
        ValueHelper.check_raise_frameNr(frameNr)
        ValueHelper.check_raise_id(videoId)
        frameLabelDBs = self.db.session.query(FrameLabel).filter_by(frameNr=frameNr, videoId=videoId).all()
        print(frameLabelDBs)
        def calculate_distance(label1, label2):
            print('label1 is', type(label1), label1)
            print('label2 is', type(label2), label2)
            return math.sqrt((label1.x - label2.X) ** 2 + (label1.y - label2.Y) ** 2)

        closest_label = min(frameLabelDBs, key=lambda label: calculate_distance(label, frameinfo))
        self.db.session.delete(closest_label)
        self.db.session.commit()
    
    def update_frameInfo(self, frameInfo: FrameInfo, video: VideoInfo):
        ValueHelper.check_raise_frameNr(frameInfo.FrameNr)
        ValueHelper.check_raise_id(video.Id)
        frame_label_DB = self.db.session.query(FrameLabel).filter_by(frameNr=frameInfo.FrameNr, videoId=video.Id)
        frame_label_DB.x = frameInfo.X
        frame_label_DB.y = frameInfo.Y
        frame_label_DB.width = frameInfo.Width
        frame_label_DB.height = frameInfo.Height
        frame_label_DB.jumperVisible = frameInfo.JumperVisible
        self.db.session.commit()
