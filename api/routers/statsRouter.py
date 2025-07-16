from flask import request
from flask_restful import Resource
from services.videoService import VideoService
from services.statsService import StatsService

class StatsRouter(Resource):
    def __init__(self, **kwargs):
        self.videoService = VideoService()
        self.statsService = StatsService(self.videoService)
        super().__init__(**kwargs)
    
    def get(self):
        videoIds = [int(i) for i in request.args.getlist('videoIds[]')]
        stat = request.args.get('stat')

        match stat:
            case 'general':
                return self.statsService.getGeneralStats(), 200
            case 'localize':
                return self.statsService.getLocalizeResults(), 200
            case 'segmentation':
                return {}, 200
            case 'recognition':
                return self.statsService.getRecognitionResults(selectedModel='MViT'), 200
            case 'judge':
                return self.statsService.get_score_comparison(videoIds=videoIds), 200
            case _:
                return f'Forbidden {stat}', 404


