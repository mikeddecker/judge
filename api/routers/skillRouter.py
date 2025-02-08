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

class FrameRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self, videoId: int):
        try:
            ValueHelper.check_raise_id(videoId)
        except ValueError as ve:
            return str(ve), 404
        print(f"Getting skills for video {videoId}")
        return [s.to_dict() for s in self.videoService.get_skills(videoId)]
    
    def post(self, videoId: int):
        data = request.get_json()
        
        print(f"Adding skill for video {videoId} wit info: {data}")
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
    
    def delete(self, videoId: int):
        data = request.get_json()
        start = data.get('FrameStart')
        end = data.get('FrameEnd')

        print(f"Deleting skill from video {videoId} from {start} to {end}")
        try:
            ValueHelper.check_raise_id(videoId)
            ValueHelper.check_raise_frameNr(start)
            ValueHelper.check_raise_frameNr(end)
        except ValueError as ve:
            return str(ve), 404
        video = self.videoService.get(videoId)
        self.videoService.remove_skill(
            disciplineconfig=get_discipline_DoubleDutch_config(),
            videoinfo=video,
            frameStart=start,
            frameEnd=end,
        )
        video.remove_skill(video.get_skill(start=start, end=end))
        return video.to_dict(), 200

