from flask import request
from flask_restful import Resource
from services.videoService import VideoService
from services.statsService import StatsService
from services.resultsService import ResultsService

class ResultsRouter(Resource):
    def __init__(self, **kwargs):
        self.videoService = VideoService()
        self.statsService = StatsService(self.videoService)
        self.resultsService = ResultsService(self.videoService)
        super().__init__(**kwargs)

    def get(self):
        step = request.args.get('step')

        match step:
            case 'general':
                return self.resultsService.general(), 200
            case 'localization':
                return self.resultsService.localization(), 200
            case 'segmentation':
                return self.resultsService.segmentation(), 200
            case 'recognition':
                return self.resultsService.recognition(), 200
            case 'judge':
                return self.resultsService.judge([]), 200
            case _:
                return f'Forbidden step: {step}', 404

