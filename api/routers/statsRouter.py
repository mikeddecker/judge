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

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

class StatsRouter(Resource):
    def __init__(self, **kwargs):
        self.statsService = StatsService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self):
        videoIds = [int(i) for i in request.args.getlist('videoIds[]')]
        stat = request.args.get('stat')

        match stat:
            case 'localize':
                return self.statsService.getLocalizeResults(selectedModel='TODO'), 200
            case 'segmentation':
                return {}, 200
            case 'recognition':
                return self.statsService.getRecognitionResults(selectedModel='HAR_MViT'), 200
            case 'judge':
                return self.videoService.get_score_comparison(videoIds=videoIds), 200
            case _:
                return f'Forbidden {stat}', 404


