import os
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.frameinfo import FrameInfo
from domain.skill import Skill
from domain.job import Job
from helpers.ValueHelper import ValueHelper
from helpers.ConfigHelper import get_discipline_DoubleDutch_config
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.jobRepo import JobRepository
from typing import List

VISION_MODELS = ['HAR_MViT']

class JobService:
    """Provides the video information of videos"""
    PROPERTIES = [
        "VideoRepo",
        "FolderRepo",
        "JobRepo",
        "StorageFolder",
    ]
    def __init__(self, storage_folder: str):
        ValueHelper.check_raise_string(storage_folder)
        self.VideoRepo = VideoRepository(db=db)
        self.FolderRepo = FolderRepository(db=db)
        self.JobRepo = JobRepository(db=db)


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

    def __add(self, job: Job) -> None:
        """Adds the given job in the database
        TRAIN, PREDICT - L, S, R, FULL

        """
        assert isinstance(job, Job)
        if self.JobRepo.exists(job):
            raise ValueError("Job exists")
        else:
            self.JobRepo.add(job)
        
    def count(self) -> int:
        return self.JobRepo.count()
    
    def get(self) -> List[Job]:
        """Get video with the corresponding Id"""
        return self.JobRepo.get_all()
    
    def video_has_pending_job(self, videoId: int, model: str, step: str = 'FULL'):
        return self.JobRepo.exists_by_job_content(
            Job(type='PREDICT', step=step, job_arguments={'model': model, 'videoId':videoId})
        )
    
    def launch_job_predict_skills(self, step:str, model: str, videoId: int):
        job = Job(
            type = 'PREDICT',
            step = step,
            status = 'Created',
            job_arguments = { "videoId": videoId, "model": model, "save_mp4": True },
        )
        if not self.JobRepo.exists_by_job_content(job):
            self.__add(job)

    def re_train_and_predict(self):
        trainjob = Job(
            type='TRAIN',
            step = 'FULL',
            status= 'Created',
            job_arguments = { "model": VISION_MODELS[0] }
        )
        if not self.JobRepo.exists_by_job_content(trainjob):
            self.__add(trainjob)


        videoIds = [1285, 1315, 1178, 1408, 2283, 2285, 2289, 2288, 2296, 2309, 2568,2569,2570,2571,2572,2573,2574,2575,2576,2577,2578,2579,2580,2581,2582,2583,2584,2585,2586,2587,2588,2589]
        for videoId in videoIds:
            predictJob = Job(
                type = 'PREDICT',
                step = 'SEGMENT_SKILL',
                status = 'Created',
                job_arguments = { "videoId": videoId, "model": VISION_MODELS[0], "save_mp4": True },
            )
            if not self.JobRepo.exists_by_job_content(predictJob):
                self.__add(predictJob)