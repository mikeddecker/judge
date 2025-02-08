import os
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.frameinfo import FrameInfo
from domain.skill import Skill
from helpers.ValueHelper import ValueHelper
from helpers.ConfigHelper import get_discipline_DoubleDutch_config
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from typing import List

SUPPORTED_VIDEO_FORMATS = [
    'webm',
    'mp4'
    'jpg',
    'png',
] # Temporarily media formats

class VideoService:
    """Provides the video information of videos"""
    PROPERTIES = [
        "VideoRepo",
        "FolderRepo",
        "StorageFolder",
    ]
    def __init__(self, storage_folder: str):
        ValueHelper.check_raise_string(storage_folder)
        self.VideoRepo = VideoRepository(db=db)
        self.FolderRepo = FolderRepository(db=db)

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

    def add(self, name: str, folder: Folder, frameLength: float, width: float, height: float, fps: float, ytid: str = None) -> VideoInfo:
        """Adds the given video to the database
        
        TODO : After localization; add meta information loader
        TODO : nice to have, add warning if name is double
        TODO : enforce mp4, wav...
        """
        if not folder:
            raise ValueError(f"Foler may not be None")
        if not self.FolderRepo.exists(id=folder.Id):
            raise LookupError(f"Id of folder is invalid")
        if self.exists_in_database(name=name, folder=folder):
            raise LookupError(f"Video {name} already added in the database")
        if not self.exists_on_drive(name=name, folder=folder):
            raise ValueError(f"Video {name} does not exist in {folder.get_relative_path()}")
        if frameLength <= 0 or width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(f"FrameLength, width and height must be > 0, got", frameLength, width, height)

        return self.VideoRepo.add(
            name=name,
            folder=folder,
            frameLength=frameLength,
            width=width,
            height=height,
            fps=fps,
            srcinfo=ytid, # TODO : make better
        )
    
    def add_skill(self, videoinfo: VideoInfo, frameStart: int, frameEnd: int, skillinfo: dict) -> Skill:
        assert isinstance(skillinfo, dict), "Skillinfo is not a dict"
        assert len(skillinfo) > 0, "Skillinfo is empty"
        config = get_discipline_DoubleDutch_config()

        ValueHelper.check_raise_frameNr(frameStart)
        ValueHelper.check_raise_frameNr(frameEnd)
        assert frameEnd > frameStart, f"End is smaller than start: s = {frameStart}, e = {frameEnd}"
        assert frameEnd - frameStart > 5, f"Skill must be long enough"
        if videoinfo.has_skill_overlap(frameStart, frameEnd):
            raise ValueError(f"Skill has overlap with another skill, {frameStart} -> {frameEnd}")

        # Check skillinfo values
        for key, value in config.items():
            if key != 'Tablename':
                assert key in skillinfo.keys(), f"Skillinfo does not provide info for {key}"
            if value[0] == "Numerical":
                min = value[1]
                max = value[2]
                try:
                    skillinfo[key] = int(skillinfo[key])
                except ValueError:
                    raise ValueError(f"Skillspecification of {key} must be an integer, but got {skillinfo[key]} ({type(skillinfo[key])})")
                assert skillinfo[key] >= min and skillinfo[key] <= max, f"Skillinfo for {key} must be between {min} and {max}, got {skillinfo[key]}"
            elif value[0] == "Categorical":
                try:
                    skillinfo[key] = int(skillinfo[key])
                except ValueError:
                    raise ValueError(f"Skillspecification of {key} must be an integer, but got {skillinfo[key]} ({type(skillinfo[key])})")
                self.VideoRepo.exists_skillinfo(discipline=config["Tablename"], table_name_part=config[key][1], uc=skillinfo[key])
            elif value[0] == "Boolean":
                assert isinstance(skillinfo[key], bool), f"Boolean value {key} must be a boolean, got {skillinfo[key]}"

        insertedId =self.VideoRepo.add_skill(videoId=videoinfo.Id, disciplineConfig=config, skillinfo=skillinfo, start=frameStart, end=frameEnd)
        skill = Skill(id=insertedId, disciplineConfig=config, skillinfo=skillinfo, start=frameStart, end=frameEnd)
        # videoinfo.add_skill(skill)
        return skill

    def update_skill(self, id: int, videoinfo: VideoInfo, frameStart: int, frameEnd: int, skillinfo: dict) -> VideoInfo:
        assert isinstance(skillinfo, dict), "Skillinfo is not a dict"
        assert len(skillinfo) > 0, "Skillinfo is empty"
        config = get_discipline_DoubleDutch_config()

        ValueHelper.check_raise_frameNr(frameStart)
        ValueHelper.check_raise_frameNr(frameEnd)
        assert frameEnd > frameStart, f"End is smaller than start: s = {frameStart}, e = {frameEnd}"
        assert frameEnd - frameStart > 5, f"Skill must be long enough"
        if videoinfo.has_skill_overlap(frameStart, frameEnd, skillId=id):
            raise ValueError(f"Skill has overlap with another skill, {frameStart} -> {frameEnd}")

        # Check skillinfo values
        for key, value in config.items():
            if key != 'Tablename':
                assert key in skillinfo.keys(), f"Skillinfo does not provide info for {key}"
            if value[0] == "Numerical":
                min = value[1]
                max = value[2]
                assert isinstance(skillinfo[key], int), f"Skillspecification of {key} must be in integer, got {skillinfo[key]}"
                assert skillinfo[key] >= min and skillinfo[key] <= max, f"Skillinfo {key} must be between {min} and {max}, got {skillinfo[key]}"
            elif value[0] == "Categorical":
                assert isinstance(skillinfo[key], int), f"Skillspecification of {key} must be in integer, got {skillinfo[key]}"
                self.VideoRepo.exists_skillinfo(discipline=config["Tablename"], table_name_part=config[key][1], uc=skillinfo[key])
            elif value[0] == "Boolean":
                assert isinstance(skillinfo[key], bool), f"Boolean value {key} must be a boolean, got {skillinfo[key]}"

        self.VideoRepo.update_skill(id=id, videoId=videoinfo.Id, disciplineConfig=config, skillinfo=skillinfo, start=frameStart, end=frameEnd)
        video = self.VideoRepo.get(videoinfo.Id)
        for s in self.VideoRepo.get_skills(video.Id):
            video.add_skill(s)
        return video


    def remove_skill(self, disciplineconfig: dict, videoinfo: VideoInfo, frameStart: int, frameEnd: int) -> VideoInfo:
        skill = videoinfo.get_skill(frameStart, frameEnd)
        self.VideoRepo.remove_skill(disciplineconfig, videoinfo.Id, skill.FrameStart, skill.FrameEnd)
        videoinfo.remove_skill(skill)
        return videoinfo
    
    def count(self) -> int:
        return self.VideoRepo.count()
    
    def exists_in_database(self, id: int = None, name: str = None, folder: Folder = None) -> bool:
        """Check existence in database
        If id provided, ignore name and folder"""
        if id:
            ValueHelper.check_raise_id(id)
            return self.VideoRepo.exists(id=id)
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"When no id, but name is given, folder also needs to be given; got {folder}")
        return self.VideoRepo.exists_by_name(name=name, folder=folder)
    
    def exists_on_drive(self, name: str, folder: Folder) -> bool:
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"Folder must be provided, got {folder}")
        return os.path.exists(os.path.join(self.StorageFolder, folder.get_relative_path(), name))

    def get(self, id: int) -> VideoInfo:
        """Get video with the corresponding Id"""
        ValueHelper.check_raise_id(id)
        if not self.exists_in_database(id=id):
            raise LookupError(f"VideoId {id} does not exist")
        video = self.VideoRepo.get(id=id)
        for skill in self.VideoRepo.get_skills(videoId=id):
            video.add_skill(skill)
        return video

    def is_already_downloaded(self, sourceinfo:str):
        """Sourceinfo = yt_id"""
        return self.VideoRepo.is_already_downloaded(sourceinfo)

    def delete_from_database(self, id: int):
        # TODO : check skills & segments
        ValueHelper.check_raise_id(id)
        if not self.exists_in_database(id=id):
            raise LookupError(f"VideoId {id} does not exist")
        if self.VideoRepo.has_frames(videoId=id):
            videoinfo = self.VideoRepo.get(id=id)
            for f_nr in videoinfo.Frames:
                self.VideoRepo.remove_frameInfo(videoId=id, frameNr=f_nr)
        self.VideoRepo.delete(id=id)

    def get_videos(self, folderId: int) -> List[VideoInfo]:
        """Returns videos in the given folder that are inserted in the database"""
        ValueHelper.check_raise_id(folderId)
        if not self.FolderRepo.exists(folderId):
            raise LookupError(f"FolderId {folderId} does not exist")
        return self.VideoRepo.get_videos(folderId=folderId)
    
    def get_skills(self, videoId: int) -> List[Skill]:
        ValueHelper.check_raise_id(videoId)
        if not self.VideoRepo.exists(videoId):
            raise LookupError(f"VideoId {videoId} does not exist")
        return self.VideoRepo.get_skills(videoId)
    
    def get_skilloptions(self, skilltype: str, tablepart: str):
        assert skilltype == "DoubleDutch", f"Currently only DoubleDutch supported, got {skilltype}"
        assert tablepart in ["Type", "Turner", "Skill"], f"Only tables Type, turner & skills are created, got {tablepart}"
        return self.VideoRepo.get_skilloptions(skilltype, tablepart)

    
    # TODO : nice to have
    def rename(self, id: int, new_name):
        raise NotImplementedError("Nice to have, end of journey")
    
    def remove_frameInfo(self, frameNr, video: VideoInfo, frameinfo: FrameInfo) -> VideoInfo:
        ValueHelper.check_raise_frameNr(frameNr)
        if video is None or not isinstance(video, VideoInfo):
            raise ValueError(f"frameInfo is not {VideoInfo}, got {video}")
        if not video.has_frame_been_labeled(frameinfo):
            raise ValueError(f"Frame {frameinfo} not labeled")
        self.VideoRepo.remove_frameInfo(frameNr, videoId=video.Id, frameinfo=frameinfo)
        video.remove_framelabel(frameNr=frameNr, label=frameinfo)
        return video

    def set_frameInfo(self, frameInfo: FrameInfo, video: VideoInfo):
        if frameInfo is None or not isinstance(frameInfo, FrameInfo):
            raise ValueError(f"frameInfo is not {FrameInfo}, got {frameInfo}")
        if video is None or not isinstance(video, VideoInfo):
            raise ValueError(f"frameInfo is not {VideoInfo}, got {video}")
        if frameInfo.FrameNr >= video.FrameLength:
            raise ValueError(f"FrameNr out of bounds, max {video.FrameLength}, got {frameInfo.FrameNr}")
        # if frameInfo.Width < 0.3 or frameInfo
        video.add_framelabel(frameInfo)
        if not frameInfo.LabelType == 2 and self.VideoRepo.exists_frameInfo(videoId=video.Id, frameNr=frameInfo.FrameNr):
            self.VideoRepo.update_frameInfo(video=video, frameInfo=frameInfo)
        else:
            self.VideoRepo.add_frameInfo(video=video, frameInfo=frameInfo)
        return video

    # TODO : nice to have
    def upload(self):
        raise NotImplementedError("Nice to have, end of journey")
