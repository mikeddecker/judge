import os

from flask import Flask, request, jsonify, current_app, g
from flask_cors import CORS
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from repository.db import db
from routers.folderRouter import FolderRouter
from routers.videoRouter import VideoRouter, VideoRouterCropped, VideoImageRouter, VideoInfoRouter, VideoPredictionRouter
from routers.frameRouter import FrameRouter
from routers.storageRouter import StorageRouter, OrphanDeleterRouter
from routers.skillRouter import SkillRouter, OptionRouter, SkillLevel, SkillLabelingCompletedRouter, DiffScoreComparison
from routers.downloadRouter import DownloadRouter
from routers.statsRouter import StatsRouter

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

api.add_resource(FrameRouter, '/video/<int:videoId>/frameNr/<int:frameNr>')
api.add_resource(OptionRouter, '/skilloptions/<skilltype>/<tableinfo>')
api.add_resource(SkillRouter, '/skill/<int:videoId>')
api.add_resource(SkillLevel, '/skilllevel')
api.add_resource(DiffScoreComparison, '/diff-score-comparison')
api.add_resource(SkillLabelingCompletedRouter, '/skillcompleted/<int:videoId>')
api.add_resource(StorageRouter, '/discover')
api.add_resource(OrphanDeleterRouter, '/discover/deleteOrphans')
api.add_resource(DownloadRouter, '/download')



api.add_resource(StatsRouter, '/stats')

if __name__ == '__main__':
    app.run(port=5555, debug=True)
