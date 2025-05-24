import os
import time
import torch
from dotenv import load_dotenv
from flask import send_file, request
from flask_restful import Resource
from services.folderService import FolderService
from services.videoService import VideoService
from services.statsService import StatsService
from helpers.ValueHelper import ValueHelper

PYTORCH_MODELS_SKILLS = {
    "HAR_SA_Conv3D" : None,
    "HAR_MViT" : None,
    "HAR_MViT_extra_dense" : None
}


load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

class StatsRouter(Resource):
    def __init__(self, **kwargs):
        self.statsService = StatsService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self):
        start = time.time()
        videoIds = [int(i) for i in request.args.getlist('videoIds[]')]
        stat = request.args.get('stat')
        return {
            'localization': self.statsService.getLocalizeResults(selectedModel='TODO'),
            'segmentation' : {
                'mse-val' : 0.06,
                'mse-test' : 0.065,
                'test-iou-real-over-predict' : 0.7,
                'test-iou-predict-over-real' : 0.71,
                'test-avg-splitpoint-distance' : 7.31,
                'test-avg-highest-splitpoint-distance' : 182,
                'test-highest-splitpoint-distance': 1502,
                'val-iou-real-over-predict' : 0.7,
                'val-iou-predict-over-real' : 0.71,
                'val-avg-splitpoint-distance' : 7.31,
                'val-avg-highest-splitpoint-distance' : 182,
                'val-highest-splitpoint-distance': 1502,
                'videos-train' : 44,
                'videos-val' : 7,
                'videos-test': 3,
                'total-frames': 7800,
                'best-model' : 'MViT',
                'train-time' : 6852.3
            },
            'recognition' : self.statsService.getRecognitionResults(selectedModel='HAR_MViT'),
            'scores' : self.videoService.get_score_comparison(videoIds=videoIds),
            'time' : time.time() - start,
        }, 200

