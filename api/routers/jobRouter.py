from flask import request
from flask_restful import Resource
from services.folderService import FolderService
from services.videoService import VideoService
from services.jobService import JobService

class JobTrainRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.jobService = JobService()
        super().__init__(**kwargs)
    
    def get(self):
        self.jobService.re_train_and_predict()
        return "Ok", 200

class JobPredictVideo(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.jobService = JobService()
        super().__init__(**kwargs)
    
    def get(self, videoId:int):
        # TODO : modify
        self.jobService.launch_job_predict_skills(
            step='FULL',
            model='MViT',
            videoId=videoId
        )
        return "Ok", 200

    def post(self):
        data = request.get_json()

        # TODO : prepare
        return "Ok", 200