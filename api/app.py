import os

from config import ENVS
from flask import Flask, request, jsonify, current_app, g
from flask_cors import CORS
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from helpers.ValueHelper import ValueHelper
from repository.db import db
from routers.folderRouter import FolderRouter
from routers.videoRouter import VideoRouter, VideoRouterCropped, VideoImageRouter, VideoInfoRouter, VideoPredictionRouter, VideoPredictionRouter_GetLocalizePredictions, VideoPredictionRouter_HasLocalizePredictions
from routers.frameRouter import FrameRouter, FrameLabelTypeRouter
from routers.jobRouter import JobTrainRouter, JobLaunchRouter, JobOptionsRouter
from routers.storageRouter import StorageRouter, OrphanDeleterRouter
from routers.skillRouter import SkillRouter, OptionRouter, SkillLevel, SkillLabelingCompletedRouter, DiffScoreComparison
from routers.downloadRouter import DownloadRouter
from routers.mlLayerRouter import MLLayerRouter, MLLayerTypesRouter, MLLayerCompositionRouter
from routers.statsRouter import StatsRouter
from routers.tagRouter import TagRouter, TagGroupRouter
from services.videoService import VideoService

DATABASE_URL = os.getenv('DATABASE_URL')

# db = SQLAlchemy()
migrate = Migrate()

def create_app(config_object="config.Config"):
    app = Flask(__name__)
    CORS(app)
    
    # Load configuration from config file or environment variable
    app.config.from_object(config_object)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    return app

app = create_app()
CORS(app)
api = Api(app)

# use api.add_resource to add the paths
api.add_resource(FolderRouter, '/folders', '/folders/<int:folderId>')
api.add_resource(VideoRouter, '/video/<int:videoId>')
api.add_resource(VideoRouterCropped, '/video/<int:videoId>/cropped')
api.add_resource(VideoInfoRouter, '/video/<int:videoId>/info')
api.add_resource(VideoImageRouter, '/video/<int:videoId>/image')
api.add_resource(VideoPredictionRouter, '/video/<int:videoId>/predictions')
api.add_resource(VideoPredictionRouter_HasLocalizePredictions, '/video/<int:videoId>/predictions/hasLocalizePredictions')
api.add_resource(VideoPredictionRouter_GetLocalizePredictions, '/video/<int:videoId>/predictions/getLocalizePredictions')
api.add_resource(TagRouter, '/tags')
api.add_resource(TagGroupRouter, '/tagGroups')
api.add_resource(MLLayerRouter, '/layers')
api.add_resource(MLLayerTypesRouter, '/layers/types')
api.add_resource(MLLayerCompositionRouter, '/layercompositions')
api.add_resource(FrameRouter, '/video/<int:videoId>/frameNr/<int:frameNr>')
api.add_resource(FrameLabelTypeRouter, '/frameLabelTypes')
api.add_resource(OptionRouter, '/skilloptions/<skilltype>/<tableinfo>')
api.add_resource(SkillRouter, '/skill/<int:videoId>')
api.add_resource(SkillLevel, '/skilllevel')
api.add_resource(DiffScoreComparison, '/diff-score-comparison')
api.add_resource(SkillLabelingCompletedRouter, '/skillcompleted/<int:videoId>')
api.add_resource(StorageRouter, '/discover')
api.add_resource(OrphanDeleterRouter, '/discover/deleteOrphans')
api.add_resource(DownloadRouter, '/download')
api.add_resource(JobTrainRouter, '/job/retrain')
api.add_resource(JobLaunchRouter, '/job')
api.add_resource(JobOptionsRouter, '/job/options/<step>')

api.add_resource(StatsRouter, '/stats')

# Check if .env folders are filled in
ValueHelper.check_raise_string(ENVS.DIRS.VIDEOS)
ValueHelper.check_raise_string(ENVS.DIRS.GENERATED)
ValueHelper.check_raise_string(ENVS.DIRS.GENERATED_VIDEODATA)
ValueHelper.check_raise_string(ENVS.DIRS.WEIGHTS)
ValueHelper.check_raise_string(ENVS.DIRS.YOLO_LABELS)

os.makedirs(ENVS.DIRS.GENERATED, exist_ok=True)
os.makedirs(ENVS.DIRS.GENERATED_VIDEODATA, exist_ok=True)
os.makedirs(ENVS.DIRS.WEIGHTS, exist_ok=True)
os.makedirs(ENVS.DIRS.YOLO_LABELS, exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'labels'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'val'), exist_ok=True)

if __name__ == '__main__':
    with app.app_context():
        videoservice = VideoService()
        videoservice.initiate()
        del videoservice
    app.run(port=5555, debug=True)

