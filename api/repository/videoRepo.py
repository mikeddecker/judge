import math

from domain.videoinfo import VideoInfo
from domain.folder import Folder
from domain.skill import Skill
from domain.videoinfo import VideoInfo
from domain.frameinfo import FrameInfo
from flask_sqlalchemy import SQLAlchemy
from helpers.ValueHelper import ValueHelper
from repository.MapToDomain import MapToDomain
from repository.MapToDB import MapToDB
from repository.models import Video as VideoInfoDB, Folder as FolderDB, FrameLabel, Skill, FrameLabelType
from sqlalchemy import desc, func, and_
from typing import List

class VideoRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db

    # TODO : make width, height ... requirements
    def add(
            self, name: str, folder: Folder, frameLength: int,
            width: float, height: float, fps: float, duration: float,
            training=True, qualitative=True, obstruction=False, private=True,
            srcinfo:str=None,
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
            duration = duration,
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
    
    def exists_frameInfo(self, video: VideoInfo, frameInfo: FrameInfo):
        ValueHelper.check_raise_id(video.Id)
        ValueHelper.check_raise_frameNr(frameInfo.FrameNr)
        return self.db.session.query(FrameLabel).filter_by(
            videoId=video.Id, 
            frameNr=frameInfo.FrameNr,
            x=frameInfo.X,
            y=frameInfo.Y,
            width=frameInfo.Width,
            height=frameInfo.Height,
        ).first() is not None
  
    def delete(self, id: int):
        # TODO : check if no frames or skills are connected
        ValueHelper.check_raise_id(id)
        videoInfoDB = self.db.session.get(VideoInfoDB, ident=id)
        self.db.session.delete(videoInfoDB)
        self.db.session.commit()

    def get(self, id: int) -> VideoInfo:
        return MapToDomain.map_video(self.db.session.get(VideoInfoDB, ident=id))
    
    def get_videoId(self, name: str, folder: Folder) -> int:
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"folder must be provided")
        return self.db.session.query(VideoInfoDB).filter_by(name=name, folderId=folder.Id).one().id
    
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
        def calculate_distance(label1, label2):
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

    def get_team_boxes(self, video_id: int = None):
        xmin = func.min(FrameLabel.x - FrameLabel.width / 2).label("xmin")
        xmax = func.max(FrameLabel.x + FrameLabel.width / 2).label("xmax")
        ymin = func.min(FrameLabel.y - FrameLabel.height / 2).label("ymin")
        ymax = func.max(FrameLabel.y + FrameLabel.height / 2).label("ymax")

        width = (func.max(FrameLabel.x + FrameLabel.width / 2) -
                func.min(FrameLabel.x - FrameLabel.width / 2)).label("width")
        x = ((func.max(FrameLabel.x + FrameLabel.width / 2) +
            func.min(FrameLabel.x - FrameLabel.width / 2)) / 2).label("x")
        height = (func.max(FrameLabel.y + FrameLabel.height / 2) -
                func.min(FrameLabel.y - FrameLabel.height / 2)).label("height")
        y = ((func.max(FrameLabel.y + FrameLabel.height / 2) +
            func.min(FrameLabel.y - FrameLabel.height / 2)) / 2).label("y")

        # Base filters
        filters = [FrameLabel.labeltype == 1]
        
        # Add conditional filter
        if video_id is not None:
            filters.append(FrameLabel.videoId == video_id)
        else:
            filters.append(FrameLabel.videoId % 10 == 5)

        query = (
            self.db.session.query(
                FrameLabel.videoId,
                FrameLabel.frameNr,
                xmin,
                xmax,
                ymin,
                ymax,
                width,
                x,
                height,
                y
            )
            .filter(and_(*filters))
            .group_by(FrameLabel.videoId, FrameLabel.frameNr)
            .order_by(FrameLabel.videoId, FrameLabel.frameNr)
        )

        return {
            row.frameNr: {
                'xmin': row.xmin,
                'xmax': row.xmax,
                'ymin': row.ymin,
                'ymax': row.ymax,
                'x': row.x,
                'y': row.y,
                'width': row.width,
                'height': row.height,
            }
            for row in query.all()
        }

    ##########
    # Skills #
    ##########
    def add_skill(self, videoId: int, skillinfo: dict, start: int, end: int) -> int:
        """Let the service be responsible for good values in the dicts"""
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(start)
        ValueHelper.check_raise_frameNr(end)

        # Likewise checks can be done, to check whether values of layerproperties exist
        assert self.db.session.query(VideoInfoDB).filter_by(id=videoId).count() > 0, f"VideoId {videoId} does not exist"        

        skill = Skill(
            videoId = videoId,
            frameStart = start,
            frameEnd = end,
            skillinfo = skillinfo,
        )

        self.db.session.add(skill)
        self.db.session.commit()
        return skill.id
    
    def update_skill(self, id: int, videoId: int, skillinfo: dict, start: int, end: int) -> int:
        """Let the service be responsible for good values in the dicts"""
        ValueHelper.check_raise_id(id)
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(start)
        ValueHelper.check_raise_frameNr(end)

        skill = self.db.session.query(Skill).filter_by(id=id).first()
        assert skill is not None, f"Skill {id} does not exist"
        assert self.db.session.query(VideoInfoDB).filter_by(id=videoId).count() > 0, f"VideoId {videoId} does not exist"        

        skill.frameStart = start
        skill.frameEnd = end
        skill.skillinfo = skillinfo

        self.db.session.commit()
    
    def get_skills(self, videoId: int) -> List[Skill]:
        skillsDB = self.db.session.query(Skill).filter_by(videoId=videoId).all()
        return [MapToDomain.map_skill(s) for s in skillsDB]
    
    def get_previous_skill(self, videoId: int, frameEnd: int) -> tuple[dict, str, int]:
        """Returns prev_skillinfo, prev_skillname, base_level"""
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(frameEnd)
        DDskillDB = self.db.session.query(Skill).filter(Skill.videoId==videoId).filter(Skill.frameEnd <= frameEnd).order_by(desc(Skill.frameEnd)).first()
        if DDskillDB is None:
            return None, None, 0
        skillDB = self.db.session.query(Skill).filter_by(id=DDskillDB.skill).first()
        return MapToDomain.map_skill(DDskillDB), skillDB.name, 0 if skillDB.level_dd == "/" else int(str.split(skillDB.level_dd, '-')[-1])

    def remove_skill(self, videoId, start: int, end: int):
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(start)
        ValueHelper.check_raise_frameNr(end)

        skillDB = self.db.session.query(Skill).filter_by(frameStart=start, frameEnd=end).first()
        self.db.session.delete(skillDB)
        self.db.session.commit()

    def update_skills_completed(self, videoId: int, completed: bool):
        ValueHelper.check_raise_id(videoId)
        if not isinstance(completed, bool):
            raise ValueError(f"Completed must be a boolean {completed}")
        videoDB = self.db.session.query(VideoInfoDB).filter_by(id=videoId).first()
        videoDB.completed_skill_labels = completed
        self.db.session.commit()

    def initiate(self):
        print('init db')
        if FrameLabelType.query.count() == 0:
            self.db.session.add(FrameLabelType(id=1, info='foreground-person'))
            self.db.session.add(FrameLabelType(id=2, info='background-person'))
            self.db.session.commit()

    def get_frame_label_types(self) -> dict:
        return {t.id: t.info for t in FrameLabelType.query.all()}

