from .videoService import VideoService # TODO : move to repo
from .jobService import JobService # TODO : move to repo
from helpers.ConfigHelper import PYTORCH_MODELS_SKILLS, get_discipline_DoubleDutch_config
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.statsRepo import StatsRepository
from typing import List
from config import LEVEL_TO_SCORE_MAP

class StatsService:
    PROPERTIES = [
        "FolderRepo",
        "VideoRepo",
        "StatsRepo",
        "videoService",
        "jobService",
    ]
    def __init__(self, videoService:VideoService):
        self.FolderRepo = FolderRepository(db=db)
        self.VideoRepo = VideoRepository(db=db)
        self.StatsRepo = StatsRepository(db=db)
        self.videoService = videoService
        self.jobService = JobService()

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name in self.PROPERTIES:
                raise AttributeError(f"Cannot modify {name} once it's set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def general(self) -> dict:
        return {
            'videos' : self.videoService.count()
        }

    def localization(self):
        return {
            'boxcounts': {
                'total' : self.StatsRepo.localize_box_counts(),
                'daily' : self.StatsRepo.localize_box_counts_daily(),
            },
            'framecounts': {
                'total' : self.StatsRepo.localize_frame_counts(),
            },
            'labelinfo_per_video' : self.StatsRepo.get_localize_labelinfo_per_video(),
        }

    def segmentation(self):
        return {}

    def recognition(self):
        layercomposition_names = self.StatsRepo.layercomposition_names()
        return {
            'layercomposition_names': layercomposition_names,
            'layercomposition_counts': self.StatsRepo.layer_composition_counts(),
            'layer_counts' : {
                'total': self.StatsRepo.layer_counts(),
                **{ lcn: self.StatsRepo.layer_counts(lcn) for lcn in layercomposition_names }
            },
            'skills_counts': {
                'total' : self.StatsRepo.skill_counts(),
                'daily' : self.StatsRepo.skill_counts_daily(),
            },
            'prop_value_frequencies': self.StatsRepo.layer_value_counts(),
        }

