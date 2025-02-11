import os
from dotenv import load_dotenv
from flask import send_file, Response, request
from flask_restful import Resource, current_app
from domain.frameinfo import FrameInfo
from domain.skill import Skill
from domain.videoinfo import VideoInfo
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper
from helpers.ConfigHelper import get_discipline_DoubleDutch_config

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

class OptionRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)

    def get(self, skilltype: str, tableinfo: str):
        if skilltype != "DoubleDutch":
            raise ValueError(f"Skilltype is not DoubleDutch")
        return self.videoService.get_skilloptions(skilltype, tableinfo)
class SkillRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return str(ve), 404
        return [s.to_dict() for s in self.videoService.get_skills(videoId)]
    
    def post(self, videoId: int):
        data = request.get_json()
        
        # Extract the required fields from the body
        # frameNr = data.get('frameNr')
        skillinfo = data.get('skillinfo')
        frameStart = data.get('frameStart')
        frameEnd = data.get('frameEnd')
        try:
            ValueHelper.check_raise_id(videoId)
            ValueHelper.check_raise_frameNr(frameStart)
            ValueHelper.check_raise_frameNr(frameEnd)
        except ValueError as ve:
            return str(ve), 404
        
        video = self.videoService.get(videoId)
        skill = self.videoService.add_skill(
            videoinfo=video,
            frameStart=frameStart,
            frameEnd=frameEnd,
            skillinfo=skillinfo
        )
        video.add_skill(skill)

        return video.to_dict(), 200
    
    def put(self, videoId: int):
        data = request.get_json()
        
        skillId = data.get("Id")
        skillinfo = data.get('Skillinfo')
        frameStart = data.get('FrameStart')
        frameEnd = data.get('FrameEnd')
        try:
            ValueHelper.check_raise_id(videoId)
            ValueHelper.check_raise_frameNr(frameStart)
            ValueHelper.check_raise_frameNr(frameEnd)
        except ValueError as ve:
            return str(ve), 404
        
        video = self.videoService.get(videoId)
        return self.videoService.update_skill(
            id=skillId,
            videoinfo=video,
            frameStart=frameStart,
            frameEnd=frameEnd,
            skillinfo=skillinfo
        ).to_dict(), 200
    
    def delete(self, videoId: int):
        data = request.get_json()
        start = data.get('FrameStart')
        end = data.get('FrameEnd')

        try:
            ValueHelper.check_raise_id(videoId)
            ValueHelper.check_raise_frameNr(start)
            ValueHelper.check_raise_frameNr(end)
        except ValueError as ve:
            return str(ve), 404
        video = self.videoService.get(videoId)
        return self.videoService.remove_skill(
            disciplineconfig=get_discipline_DoubleDutch_config(),
            videoinfo=video,
            frameStart=start,
            frameEnd=end,
        ).to_dict(), 200

class SkillLevel(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)

    def post(self):
        data = request.get_json()
        skillinfo = data.get('skillinfo')
        frameStart = data.get('frameStart')
        videoId = data.get('videoId')
        ValueHelper.check_raise_frameNr(frameStart)
        config = get_discipline_DoubleDutch_config()
        return self.videoService.calculate_skill_level(config, skillinfo=skillinfo, frameStart=frameStart, videoId=videoId)
