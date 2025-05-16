import os
import json
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.frameinfo import FrameInfo
from domain.skill import Skill
from helpers.ValueHelper import ValueHelper
from helpers.ConfigHelper import get_discipline_DoubleDutch_config
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from services.jobService import JobService
from typing import List

STORAGE_DIR = os.getenv("STORAGE_DIR")
FOLDER_VIDEORESULTS = os.getenv("FOLDER_VIDEORESULTS")
SUPPORTED_VIDEO_FORMATS = [
    'webm',
    'mp4'
    'jpg',
    'png',
] # Temporarily media formats

LEVEL_TO_SCORE_MAP = {
    0 : 0,
    1 : 1.5,
    2 : 2.2,
    3 : 3.3,
    4 : 4.9,
    5 : 7.3,
    6 : 11,
    7 : 11,
    8 : 11,
}

class VideoService:
    """Provides the video information of videos"""
    PROPERTIES = [
        "VideoRepo",
        "FolderRepo",
        "StorageFolder",
        "jobService",
    ]
    def __init__(self, storage_folder: str):
        ValueHelper.check_raise_string(storage_folder)
        self.VideoRepo = VideoRepository(db=db)
        self.FolderRepo = FolderRepository(db=db)
        self.jobService = JobService(STORAGE_DIR)

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
        assert frameEnd - frameStart > 4, f"Skill must be long enough"
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
        assert frameEnd - frameStart > 4, f"Skill must be long enough"
        if videoinfo.has_skill_overlap(frameStart, frameEnd, skillId=id):
            raise ValueError(f"Skill has overlap with another skill, {frameStart} -> {frameEnd}")

        ValueHelper.check_raise_skillinfo_values(config, skillinfo, repo=self.VideoRepo)

        self.VideoRepo.update_skill(id=id, videoId=videoinfo.Id, disciplineConfig=config, skillinfo=skillinfo, start=frameStart, end=frameEnd)
        video = self.VideoRepo.get(videoinfo.Id)
        for s in self.VideoRepo.get_skills(video.Id):
            video.add_skill(s)
        return video

    
    def remove_skill(self, disciplineconfig: dict, videoinfo: VideoInfo, frameStart: int, frameEnd: int) -> VideoInfo:
        skill = videoinfo.get_skill(frameStart, frameEnd)
        print("remove", skill)
        self.VideoRepo.remove_skill(disciplineconfig, videoinfo.Id, skill.FrameStart, skill.FrameEnd)
        videoinfo.remove_skill(skill)
        return videoinfo
    
    def calculate_skill_level(self, disciplineconfig: dict, skillinfo: dict, frameStart: int, videoId: int) -> int:
        ValueHelper.check_raise_skillinfo_values(config=disciplineconfig, skillinfo=skillinfo, repo=self.VideoRepo)
        ValueHelper.check_raise_frameNr(frameStart)

        options_type = self.get_skilloptions(skilltype=disciplineconfig["Tablename"], tablepart="Type", include_levels=True)
        options_skill = self.get_skilloptions(skilltype=disciplineconfig["Tablename"], tablepart="Skill", include_levels=True)
        options_turner = self.get_skilloptions(skilltype=disciplineconfig["Tablename"], tablepart="Turner", include_levels=True)

        if skillinfo["Fault"]:
            skillinfo["Fault"] = False
            return f"F {self.calculate_skill_level(disciplineconfig=disciplineconfig, skillinfo=skillinfo, frameStart=frameStart, videoId=videoId)}"
        if skillinfo["Sloppy"]:
            skillinfo["Sloppy"] = False
            return f"Onafgewerkte L{self.calculate_skill_level(disciplineconfig=disciplineconfig, skillinfo=skillinfo, frameStart=frameStart, videoId=videoId)}"

        match (options_type[skillinfo["Type"]]):
            case 'Double Dutch':
                return self.calculate_level_double_dutch(skillinfo, options_type, options_skill, options_turner, frameStart, videoId)
            case 'Single Dutch':
                return self.calculate_level_single_dutch(skillinfo, options_type, options_skill, options_turner, frameStart, videoId)
            case 'Irish Dutch':
                return self.calculate_level_double_dutch(skillinfo, options_type, options_skill, options_turner, frameStart, videoId)
            case 'Chinese Wheel':
                return self.calculate_level_chinese_wheel(skillinfo, options_type, options_skill, options_turner, frameStart, videoId)
            case 'Transition':
                return 1
            case 'Snapperlike':
                return self.calculate_level_snapperlike(skillinfo, options_type, options_skill, options_turner, frameStart, videoId)
            case _:
                raise ValueError(f"unknown option {options_type[skillinfo["Type"]]}")

    def calculate_level_double_dutch(self, skillinfo: dict, otype, oskill, oturner, frameStart: int, videoId: int) -> List[int]:
        base_skill_levels = str.split(oskill[skillinfo["Skill"]]["dd"], sep="-")
        additional_levels = 0
        skillname = oskill[skillinfo["Skill"]]["name"]
        
        if skillinfo["Rotations"] == 0:
            return 0

        # Return 0 or 0.5 if no skill or footwork
        match (base_skill_levels):
            case ['0']:
                return 0
            case ['0.5']:
                return 0.5
            case ['/']: # Return from power
                prev_skillinfo, prev_skillname, base_level = self.VideoRepo.get_previous_skill(videoId=videoId, frameEnd=frameStart)
                print(type(base_level), base_level)
                base_skill_levels = [base_level + 1 if prev_skillname == "frog" and prev_skillinfo.SkillInfo["Hands"] == 1 else base_level]
                base_skill_levels = base_skill_levels if prev_skillname != 'stut' else [2]
                # if consequetive_possibility and prev_base_skill_level == 3 and skillname != 'frog': # 1h frog  (high frog has base skill level 2)
                #     additional_levels += 1
            case _:
                base_skill_levels = [int(bs) for bs in base_skill_levels]
                print("base skill levels", base_skill_levels)
                prev_skillinfo, prev_skillname, base_level = self.VideoRepo.get_previous_skill(videoId=videoId, frameEnd=frameStart)

                # high frog?
                print("prev", prev_skillname)
                if skillname == 'frog' and skillinfo["Feet"] == 2 and prev_skillname not in ["pushup", "split", "crab", "swift", "SPAGAAT", "buddy-bounce", "rol2kip"]:
                    print("+1 highfrog")
                    additional_levels += 1
                
                # stut?
                if skillname == 'frog' and skillinfo["Feet"] == 2 and prev_skillname == "rol2kip":
                    print("+1 stut")
                    additional_levels += 1
                
                # one handed frog?
                if skillname == 'frog' and skillinfo["Hands"] == 1:
                    print("+1 one handed frog")
                    additional_levels += 1
                
                # consequetive?
                if skillname == 'frog' and prev_skillname == 'frog':
                    print("+1 consequetive")
                    additional_levels += 1
                
                # 1h frog -> other skill
                if prev_skillname == 'frog' and prev_skillinfo and prev_skillinfo.SkillInfo["Hands"] == 1:
                    print("+1 from 1h frog -> other skill")
                    additional_levels += 1

                # Turntable
                if skillname == prev_skillname and prev_skillinfo and skillinfo["Hands"] == prev_skillinfo.SkillInfo["Hands"]:
                    print(f"+{skillinfo["Turntable"]} turntable")
                    additional_levels += skillinfo["Turntable"]

                # Air skills
                if skillname in ["rad", "rondat", "handspring"] and skillinfo["Hands"] == 0:
                    base_skill_levels = [4]
                
        match (skillinfo["Rotations"]):
            case 2:
                print('+1 double under')
                additional_levels += 1
            case 3 | 4:
                print('+2 triple/quad')
                additional_levels += 2
            case 5 | 6:
                print('+3 quint/hex')
                additional_levels += 3
            case 7 | 8:
                print('+4 sept/oct')
                additional_levels += 4

        turnername1 = oturner[skillinfo["Turner1"]]["name"]
        turnername2 = oturner[skillinfo["Turner2"]]["name"]
        print(f"turnername", turnername1)
        # TODO : fix consequetive turns (i.e. keep turning in an EB or cross)
        # print(f"idk what this is {"noting" if prev_skillinfo is None else oturner[prev_skillinfo.SkillInfo["Turner1"]]["name"]}")
        if not (turnername1 in ["cross", "crougercross", "inverse toad"] and prev_skillinfo is not None and oturner[prev_skillinfo.SkillInfo["Turner1"]]["name"] in ["cross", "crougercross", "inverse toad"] and prev_skillinfo.SkillInfo["Rotations"] < 3):
            extra_level = oturner[skillinfo['Turner1']]['dd']
            print(f"+{extra_level} {oturner[skillinfo['Turner1']]['name']} turner")
            additional_levels += extra_level
            
        if not (turnername2 in ["cross", "crougercross", "inverse toad"] and prev_skillinfo is not None and oturner[prev_skillinfo.SkillInfo["Turner2"]]["name"] in ["cross", "crougercross", "inverse toad"] and prev_skillinfo.SkillInfo["Rotations"] < 3):
            extra_level = oturner[skillinfo['Turner2']]['dd']
            print(f"+{extra_level} {oturner[skillinfo['Turner2']]['name']} turner")
            additional_levels += extra_level
            
        if skillinfo["BodyRotations"] > 0 and skillname in ["crab", "pushup"]:
            print(f"+{skillinfo["BodyRotations"]} body rotations (crab/push)")
            additional_levels += skillinfo["BodyRotations"] // 2

        level_total = []
        for baselevel in base_skill_levels:
            level_total.append(int(baselevel) + additional_levels)
       
        print("level total", level_total)
        return level_total

    def calculate_level_single_dutch(self, skillinfo: dict, otype, oskill, oturner, frameStart: int, videoId: int):
        base_skill_levels = str.split(oskill[skillinfo["Skill"]]["dd"], sep="-")
        additional_levels = 0
        skillname = oskill[skillinfo["Skill"]]["name"]

        # Return 0 or 0.5 if no skill or footwork
        match (base_skill_levels):
            case ['0']:
                return 0
            case ['0.5']:
                return 0.5
            case ['/']: # Return from power
                prev_skillinfo, prev_skillname, base_level = self.VideoRepo.get_previous_skill(videoId=videoId, frameEnd=frameStar)
                print(type(base_level), base_level)
                base_skill_levels = [base_level + 1 if prev_skillname == "frog" and prev_skillinfo.SkillInfo["Hands"] == 1 else base_level]
                # if consequetive_possibility and prev_base_skill_level == 3 and skillname != 'frog': # 1h frog  (high frog has base skill level 2)
                #     additional_levels += 1
            case _:
                base_skill_levels = [int(bs) for bs in base_skill_levels]
                print("base skill levels", base_skill_levels)
                prev_skillinfo, prev_skillname, base_level = self.VideoRepo.get_previous_skill(videoId=videoId, frameEnd=frameStart)

                # high frog?
                if skillname == 'frog' and skillinfo["Feet"] == 2 and prev_skillname not in ["pushup", "split", "crab", "swift", "SPAGAAT", "buddy-bounce", "rol2kip"]:
                    print("+1 highfrog")
                    additional_levels += 1
                
                if skillname == 'frog' and skillinfo["Feet"] == 2 and prev_skillname == "rol2kip":
                    print("+1 stut")
                    additional_levels += 1
                                
                # one handed frog?
                if skillname == 'frog' and skillinfo["Hands"] == 1:
                    print("+1 one handed frog")
                    additional_levels += 1
                
                # consequetive?
                if skillname == 'frog' and prev_skillname == 'frog':
                    print("+1 consequetive")
                    additional_levels += 1
                
                # 1h frog -> other skill
                if prev_skillname == 'frog' and prev_skillinfo and prev_skillinfo.SkillInfo["Hands"] == 1:
                    print("+1 from 1h frog -> other skill")
                    additional_levels += 1

                # Turntable
                if skillname == prev_skillname and prev_skillinfo and skillinfo["Hands"] == prev_skillinfo.SkillInfo["Hands"]:
                    print(f"+{skillinfo["Turntable"]} turntable")
                    additional_levels += skillinfo["Turntable"]

                # Air skills
                if skillname in ["rad", "rondat", "handspring"] and skillinfo["Hands"] == 0:
                    base_skill_levels = [4]
                
        if skillinfo["Rotations"] > 1:
            print(f'+{skillinfo["Rotations"]-1} multiple under')
            additional_levels += skillinfo["Rotations"] - 1
            
        if skillinfo["BodyRotations"] > 0 and skillname in ["crab", "pushup"]:
            print(f"+{skillinfo["BodyRotations"]} body rotations (crab/push)")
            additional_levels += skillinfo["BodyRotations"] // 2

        level_total = []
        for baselevel in base_skill_levels:
            level_total.append(int(baselevel) + additional_levels)
       
        print("level total", level_total)
        return level_total

    def calculate_level_snapperlike(self, skillinfo: dict, otype, oskill, oturner, frameStart: int, videoId: int):
        skillname = oskill[skillinfo["Skill"]]["name"]
        if skillname in ["roll", "rad", "rondat", "handspring", "stut"]:
            # Air skills
            if skillname in ["rad", "rondat", "handspring"] and skillinfo["Hands"] == 0:
                return [4]
            return int(oskill[skillinfo["Skill"]]["dd"][0])
        return 0
    
    def calculate_level_chinese_wheel(self, skillinfo: dict, otype, oskill, oturner, frameStart: int, videoId: int):
        base_skill_levels = str.split(oskill[skillinfo["Skill"]]["dd"], sep="-")
        additional_levels = 0
        skillname = oskill[skillinfo["Skill"]]["name"]

        # Return 0 or 0.5 if no skill or footwork
        match (base_skill_levels):
            case ['0']:
                return 0
            case ['0.5']:
                return 0.5
            case ['/']: # Return from power
                prev_skillinfo, prev_skillname, base_level = self.VideoRepo.get_previous_skill(videoId=videoId, frameEnd=frameStart)
                print(type(base_level), base_level)
                base_skill_levels = [base_level + 1 if prev_skillname == "frog" and prev_skillinfo.SkillInfo["Hands"] == 1 else base_level]
            case _:
                base_skill_levels = [int(bs) for bs in base_skill_levels]
                print("base skill levels", base_skill_levels)
                prev_skillinfo, prev_skillname, base_level = self.VideoRepo.get_previous_skill(videoId=videoId, frameEnd=frameStart)

                # high frog?
                if skillname == 'frog' and skillinfo["Feet"] == 2 and prev_skillname not in ["pushup", "split", "crab", "swift", "SPAGAAT", "buddy-bounce", "rol2kip"]:
                    print("+1 highfrog")
                    additional_levels += 1
                
                if skillname == 'frog' and skillinfo["Feet"] == 2 and prev_skillname == "rol2kip":
                    print("+1 stut")
                    additional_levels += 1
                
                # one handed frog?
                if skillname == 'frog' and skillinfo["Hands"] == 1:
                    print("+1 one handed frog")
                    additional_levels += 1
                
                # consequetive?
                if skillname == 'frog' and prev_skillname == 'frog':
                    print("+1 consequetive")
                    additional_levels += 1
                
                # 1h frog -> other skill
                if prev_skillname == 'frog' and prev_skillinfo and prev_skillinfo.SkillInfo["Hands"] == 1:
                    print("+1 from 1h frog -> other skill")
                    additional_levels += 1

                # Turntable
                if skillname == prev_skillname and prev_skillinfo and prev_skillinfo.SkillInfo["Type"] == skillinfo["Type"] and skillinfo["Hands"] == prev_skillinfo.SkillInfo["Hands"]:
                    print(f"+{skillinfo["Turntable"]} turntable")
                    additional_levels += skillinfo["Turntable"]

                # Air skills
                if skillname in ["rad", "rondat", "handspring"] and skillinfo["Hands"] == 0:
                    base_skill_levels = [4]
                
        if skillinfo["Rotations"] > 1:
            print(f'+{skillinfo["Rotations"]-1} multiple under')
            additional_levels += skillinfo["Rotations"] - 1
        elif skillinfo["Rotations"] == 0:
            return 0

        turnername1 = oturner[skillinfo["Turner1"]]["name"]
        turnername2 = oturner[skillinfo["Turner2"]]["name"]
        print(f"turnername", turnername1)

        # TODO : fix consequetive turns (i.e. keep turning in an EB or cross)
        # TODO : fix cross CW, make use of DB property requires both
        if turnername1 == "cross" and turnername2 == "cross":
            print(f"+1 for 2 crosses")
            extra_level = oturner[skillinfo['Turner2']]['cw']
            additional_levels += extra_level
        
        if turnername1 != "cross":
            extra_level = oturner[skillinfo['Turner1']]['cw']
            print(f"+{extra_level} {oturner[skillinfo['Turner1']]['name']} turner")
            additional_levels += extra_level
            
        if turnername2 != "cross":
            extra_level = oturner[skillinfo['Turner2']]['cw']
            print(f"+{extra_level} {oturner[skillinfo['Turner2']]['name']} turner")
            additional_levels += extra_level
            
        if skillinfo["BodyRotations"] > 0 and skillname in ["crab", "pushup"]:
            print(f"+{skillinfo["BodyRotations"]} body rotations (crab/push)")
            additional_levels += skillinfo["BodyRotations"] // 2

        level_total = []
        for baselevel in base_skill_levels:
            level_total.append(int(baselevel) + additional_levels)
       
        print("level total", level_total)
        return level_total
    

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
    
    def get_videoId(self, name: str = None, folder: Folder = None) -> int:
        """Get videoId based on name"""
        ValueHelper.check_raise_string_only_abc123_extentions(name)
        if folder is None or not isinstance(folder, Folder):
            raise ValueError(f"When no id, but name is given, folder also needs to be given; got {folder}")
        return self.VideoRepo.get_videoId(name=name, folder=folder)

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
                self.VideoRepo.remove_frameInfo(videoId=id, frameNr=f_nr.FrameNr, frameinfo=f_nr)
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
    
    def get_skilloptions(self, skilltype: str, tablepart: str, include_levels=False):
        assert skilltype == "DoubleDutch", f"Currently only DoubleDutch supported, got {skilltype}"
        assert tablepart in ["Type", "Turner", "Skill"], f"Only tables Type, turner & skills are created, got {tablepart}"
        return self.VideoRepo.get_skilloptions(skilltype, tablepart, include_levels=include_levels)

    
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

    def update_skills_completed(self, video: VideoInfo, completed: bool):
        if video is None or not isinstance(video, VideoInfo):
            raise ValueError(f"frameInfo is not {VideoInfo}, got {video}")
        if not isinstance(completed, bool):
            raise ValueError(f"Completed must be a boolean {completed}")
        if not self.VideoRepo.exists(video.Id):
            raise ValueError(f"Video does not exist, {video.Id}")
        self.VideoRepo.update_skills_completed(video.Id, completed)
        return "ok"


    # TODO : nice to have
    def upload(self):
        raise NotImplementedError("Nice to have, end of journey")

    def video_has_predictions(self, videoId: int, model: str):
        return os.path.exists(
            os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{videoId}_skills_{model}.json")
        )

    def __calculate_diff_score(self, videoId: int, model: str):
        freq_table = {l: 0 for l in range(9)}
        
        filepath = os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{videoId}_skills_{model}.json")
        with open(filepath, 'r') as f:
            predicted_skills = json.load(f)
        
        config = get_discipline_DoubleDutch_config()
        levels = [ 
            self.calculate_skill_level(
                disciplineconfig=config,
                skillinfo= {k: v['y_pred'] + 1 if config[k][0] == 'Categorical' else v['y_pred'] for k, v in predicted_skills[frameStart].items()},
                frameStart=int(frameStart),
                videoId=videoId
            ) for frameStart in
            predicted_skills.keys()
        ]
        levels = [lvl if not isinstance(lvl, list) else lvl[0] for lvl in levels]
        
        score = 0
        for lvl in levels:
            freq_table[lvl] += 1
            score += LEVEL_TO_SCORE_MAP[lvl]

        return freq_table, score

    def get_score_comparison(self, videoIds: List[int]):
        allowed_models = ['HAR_MViT']
        print("@"*80)
        print(videoIds)
        scores = {
            "total" : { "judges": 0, "HAR_MViT": 0}
        }
        

        for model in allowed_models:
            for videoId in videoIds:
                if model not in allowed_models:
                    return f"Model {model} not allowed", 404
                
                scores[videoId] = {}
                scores[videoId]["videoId"] = videoId
                scores[videoId]["judges"] = self.get(id=videoId).JudgeDiffScore
                if self.video_has_predictions(videoId=videoId, model=model):
                    # TODO : add re-calculate after x days or when a new model has been trained
                    freq, score = self.__calculate_diff_score(videoId=videoId, model=model)
                    scores[videoId][model] = round(score, 2)
                    scores[videoId][f"{model}_freq"] = freq
                    
                    if scores[videoId]["judges"]:
                        scores[videoId][f"{model}_difference"] = round(100 * (scores[videoId][model] - scores[videoId]["judges"]) / scores[videoId]["judges"], 2)
                        scores["total"]["judges"] += scores[videoId]["judges"]
                        scores["total"][f"{model}"] += round(score, 2)

                elif not self.jobService.video_has_pending_job(videoId=videoId, model=model):
                    self.jobService.launch_job_predict_skills(step='FULL', model=model, videoId=videoId)
                    scores[videoId][model] = "Created"
                else:
                    scores[videoId][model] = "Waiting"
            scores["total"]["judges"] = round(scores["total"]["judges"], 2)
            scores["total"][f"{model}_difference"] = round(100 * (scores["total"][model] - scores["total"]["judges"]) / scores["total"]["judges"], 2)

        return scores