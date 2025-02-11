import math
from domain.videoinfo import VideoInfo
from domain.folder import Folder
from domain.skill import Skill
from domain.videoinfo import VideoInfo
from domain.frameinfo import FrameInfo
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from helpers.ValueHelper import ValueHelper
from repository.models import Video as VideoInfoDB, Folder as FolderDB, FrameLabel, Skillinfo_DoubleDutch, Skillinfo_DoubleDutch_Skill, Skillinfo_DoubleDutch_Turner, Skillinfo_DoubleDutch_Type
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
    
    def exists_skillinfo(self, discipline: str, table_name_part: str, uc: int) -> bool:
        ValueHelper.check_raise_id(uc)
        match (table_name_part):
            case 'Type':
                return self.db.session.query(Skillinfo_DoubleDutch_Type).filter_by(id=uc).first() is not None
            case 'Turner':
                return self.db.session.query(Skillinfo_DoubleDutch_Turner).filter_by(id=uc).first() is not None
            case 'Skill':
                return self.db.session.query(Skillinfo_DoubleDutch_Skill).filter_by(id=uc).first() is not None
            case _:
                raise ValueError(f"tablenamepart {table_name_part} does not exist")
    
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


    ##########
    # Skills #
    ##########
    def add_skill(self, videoId: int, disciplineConfig: dict, skillinfo: dict, start: int, end: int) -> int:
        """Let the service be responsible for good values in the dicts"""
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(start)
        ValueHelper.check_raise_frameNr(end)

        assert self.db.session.query(VideoInfoDB).filter_by(id=videoId).count() > 0, f"VideoId {videoId} does not exist"        
        assert self.db.session.query(Skillinfo_DoubleDutch_Type).filter_by(id=skillinfo["Type"]).count() > 0, f"Skillinfo_DoubleDutch_Type {skillinfo["Type"]} does not exist"
        assert self.db.session.query(Skillinfo_DoubleDutch_Turner).filter_by(id=skillinfo["Turner1"]).count() > 0, f"Skillinfo_DoubleDutch_Turner {skillinfo["Turner1"]} does not exist"
        assert self.db.session.query(Skillinfo_DoubleDutch_Turner).filter_by(id=skillinfo["Turner2"]).count() > 0, f"Skillinfo_DoubleDutch_Turner {skillinfo["Turner2"]} does not exist"
        assert self.db.session.query(Skillinfo_DoubleDutch_Skill).filter_by(id=skillinfo["Skill"]).count() > 0, f"Skillinfo_DoubleDutch_Skill {skillinfo["Skill"]} does not exist"

        skill = Skillinfo_DoubleDutch(
            videoId = videoId,
            frameStart = start,
            frameEnd = end,
            type = skillinfo["Type"],
            rotations = skillinfo["Rotations"],
            turner1 = skillinfo["Turner1"],
            turner2 = skillinfo["Turner2"],
            skill = skillinfo["Skill"],
            hands = skillinfo["Hands"],
            feet = skillinfo["Feet"],
            turntable = skillinfo["Turntable"],
            bodyRotations = skillinfo["BodyRotations"],
            backwards = skillinfo["Backwards"],
            sloppy = skillinfo["Sloppy"],
            hard2see = skillinfo["Hard2see"],
            fault = skillinfo["Fault"]
        )

        self.db.session.add(skill)
        self.db.session.commit()
        return skill.id
    
    def update_skill(self, id: int, videoId: int, disciplineConfig: dict, skillinfo: dict, start: int, end: int) -> int:
        """Let the service be responsible for good values in the dicts"""
        ValueHelper.check_raise_id(id)
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(start)
        ValueHelper.check_raise_frameNr(end)

        skill = self.db.session.query(Skillinfo_DoubleDutch).filter_by(id=id).first()
        assert skill is not None, f"Skill {id} does not exist"
        assert self.db.session.query(VideoInfoDB).filter_by(id=videoId).count() > 0, f"VideoId {videoId} does not exist"        
        assert self.db.session.query(Skillinfo_DoubleDutch_Type).filter_by(id=skillinfo["Type"]).count() > 0, f"Skillinfo_DoubleDutch_Type {skillinfo["Type"]} does not exist"
        assert self.db.session.query(Skillinfo_DoubleDutch_Turner).filter_by(id=skillinfo["Turner1"]).count() > 0, f"Skillinfo_DoubleDutch_Turner {skillinfo["Turner1"]} does not exist"
        assert self.db.session.query(Skillinfo_DoubleDutch_Turner).filter_by(id=skillinfo["Turner2"]).count() > 0, f"Skillinfo_DoubleDutch_Turner {skillinfo["Turner2"]} does not exist"
        assert self.db.session.query(Skillinfo_DoubleDutch_Skill).filter_by(id=skillinfo["Skill"]).count() > 0, f"Skillinfo_DoubleDutch_Skill {skillinfo["Skill"]} does not exist"

        skill.frameStart = start
        skill.frameEnd = end
        skill.type = skillinfo["Type"]
        skill.rotations = skillinfo["Rotations"]
        skill.turner1 = skillinfo["Turner1"]
        skill.turner2 = skillinfo["Turner2"]
        skill.skill = skillinfo["Skill"]
        skill.hands = skillinfo["Hands"]
        skill.feet = skillinfo["Feet"]
        skill.turntable = skillinfo["Turntable"]
        skill.bodyRotations = skillinfo["BodyRotations"]
        skill.backwards = skillinfo["Backwards"]
        skill.sloppy = skillinfo["Sloppy"]
        skill.hard2see = skillinfo["Hard2see"]
        skill.fault = skillinfo["Fault"]


        self.db.session.commit()
    
    def get_skills(self, videoId: int) -> List[Skill]:
        skillsDB = self.db.session.query(Skillinfo_DoubleDutch).filter_by(videoId=videoId).all()
        return MapToDomain.map_skills(skillsDB)
    
    def get_skilloptions(self, skilltype: str, tablepart: str, include_levels=False) -> dict[int, str]:
        if not include_levels:
            match (tablepart):
                case 'Type':
                    return { so.id : so.name for so in self.db.session.query(Skillinfo_DoubleDutch_Type).all()}
                case 'Turner':
                    return { so.id : so.name for so in self.db.session.query(Skillinfo_DoubleDutch_Turner).all()}
                case 'Skill':
                    return { so.id : so.name for so in self.db.session.query(Skillinfo_DoubleDutch_Skill).all()}
                case _:
                    raise ValueError(f"{tablepart} does not exist")
        else:
            match (tablepart):
                case 'Type':
                    return { so.id : so.name for so in self.db.session.query(Skillinfo_DoubleDutch_Type).all()}
                case 'Turner':
                    return { so.id : { 
                        "name" : so.name, 
                        "dd" : so.level_dd, 
                        "cw" : so.level_cw,
                        "dd_requires_both" : so.dd_requires_both,
                        "cw_requires_both" : so.cw_requires_both,
                    } for so in self.db.session.query(Skillinfo_DoubleDutch_Turner).all()}
                case 'Skill':
                    return { so.id : { 
                        "name" : so.name, 
                        "dd" : so.level_dd, 
                        "cw" : so.level_cw,
                    } for so in self.db.session.query(Skillinfo_DoubleDutch_Skill).all()}
                case _:
                    raise ValueError(f"{tablepart} does not exist")

        return None

    def get_previous_skill(self, videoId: int, frameEnd: int) -> tuple[dict, str, int]:
        """Returns prev_skillinfo, prev_skillname, base_level"""
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(frameEnd)
        DDskillDB = self.db.session.query(Skillinfo_DoubleDutch).filter(Skillinfo_DoubleDutch.videoId==videoId).filter(Skillinfo_DoubleDutch.frameEnd <= frameEnd).order_by(desc(Skillinfo_DoubleDutch.frameEnd)).first()
        if DDskillDB is None:
            return None, None, 0
        skillDB = self.db.session.query(Skillinfo_DoubleDutch_Skill).filter_by(id=DDskillDB.skill).first()
        return MapToDomain.map_skills([DDskillDB])[0], skillDB.name, 0 if skillDB.level_dd == "/" else int(str.split(skillDB.level_dd, '-')[-1])

    def remove_skill(self, disciplineconfig: dict, videoId, start: int, end: int):
        ValueHelper.check_raise_id(videoId)
        ValueHelper.check_raise_frameNr(start)
        ValueHelper.check_raise_frameNr(end)

        skillDB = self.db.session.query(Skillinfo_DoubleDutch).filter_by(frameStart=start, frameEnd=end).first()
        self.db.session.delete(skillDB)
        self.db.session.commit()
    
