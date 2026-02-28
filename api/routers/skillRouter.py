from flask import request
from flask_restful import Resource
from helpers.ConfigHelper import get_discipline_DoubleDutch_config
from helpers.ValueHelper import ValueHelper
from services.folderService import FolderService
from services.videoService import VideoService
from uuid import UUID

class SkillLabelingCompletedRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)

    def post(self, videoId: UUID):
        data = request.get_json()
        completed = data.get('completed')
        try:
            ValueHelper.check_raise_uuid(videoId)
            video = self.videoService.get(videoId)
            self.videoService.update_skills_completed(video=video, completed=completed)
            return "done" , 200
        except ValueError as ve:
            return str(ve), 404

class SkillRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)

    def get(self, videoId: UUID):
        try:
            ValueHelper.check_raise_uuid(videoId)
        except ValueError as ve:
            return str(ve), 404
        return [s.to_dict() for s in self.videoService.get_skills(videoId)]

    def post(self, videoId: UUID):
        data = request.get_json()

        # Extract the required fields from the body
        # frameNr = data.get('frameNr')
        skillinfo = data.get('Skillinfo')
        frameStart = data.get('FrameStart')
        frameEnd = data.get('FrameEnd')
        try:
            ValueHelper.check_raise_uuid(videoId)
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

    def put(self, videoId: UUID):
        data = request.get_json()

        skillId = UUID(data.get("Id"))
        skillinfo = data.get('Skillinfo')
        frameStart = data.get('FrameStart')
        frameEnd = data.get('FrameEnd')
        try:
            ValueHelper.check_raise_uuid(videoId)
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

    def delete(self, videoId: UUID):
        data = request.get_json()
        start = data.get('FrameStart')
        end = data.get('FrameEnd')

        try:
            ValueHelper.check_raise_uuid(videoId)
            ValueHelper.check_raise_frameNr(start)
            ValueHelper.check_raise_frameNr(end)
        except ValueError as ve:
            return str(ve), 404
        video = self.videoService.get(videoId)
        return self.videoService.remove_skill(
            videoinfo=video,
            frameStart=start,
            frameEnd=end,
        ).to_dict(), 200

class SkillCount(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)

    def get(self) -> int:
        return self.videoService.get_skill_count()

class SkillLevel(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)

    def post(self):
        data = request.get_json()
        skillinfo = data.get('skillinfo')
        previous_skillinfo = data.get('prevSkillinfo')
        previous_skillname = data.get('prevSkillname')
        frameStart = data.get('frameStart')
        videoId = data.get('videoId')
        ValueHelper.check_raise_frameNr(frameStart)
        config = get_discipline_DoubleDutch_config()
        return self.videoService.calculate_skill_level(config, skillinfo=skillinfo, previous_skillinfo=previous_skillinfo, prev_skillname=previous_skillname, frameStart=frameStart, videoId=videoId)

class DiffScoreComparison(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)

    def get(self):
        data = request.get_json()
        videoIds = data.get('videoIds')
        # models = data.get('models')
        print("VideoIds", videoIds)
        # print("Models", models)

        # TODO : validate models, linke somewhere else

        # frameStart = data.get('frameStart')
        # ValueHelper.check_raise_frameNr(frameStart)
        # config = get_discipline_DoubleDutch_config()
        # return self.videoService.calculate_skill_level(config, skillinfo=skillinfo, frameStart=frameStart, videoId=videoId)

        return [], 200

