from config import RECIPES
from domain.job import Job
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

class JobOptionsRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.jobService = JobService()
        super().__init__(**kwargs)
    
    def get(self, step):
        match (step):
            case 'LOCALIZE':
                return { k: vars(sns) for k,sns in RECIPES.items() }
                # TODO : add best model training
        return "Ok", 200

class JobLaunchRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        self.jobService = JobService()
        super().__init__(**kwargs)
    
    # def get(self, videoId:int):
    #     # TODO : modify
    #     self.jobService.launch_job_predict_skills(
    #         step='FULL',
    #         model='MViT',
    #         videoId=videoId
    #     )
    #     return "Ok", 200

    def post(self):
        data = request.get_json()

        job = Job(
            type = data.get('type'),
            step = data.get('step'),
            job_arguments = { 
                "videoId": data.get('videoId'), 
                "model": data.get('model'), 
                "weights": data.get('weights'),
                "save_mp4": data.get('save_mp4'),
                "model_kwargs" : data.get('model_kwargs'),
            },
        )

        self.jobService.launch_job(job)

        return "Job launched", 200