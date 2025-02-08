import os

from flask import Flask, request, jsonify, current_app, g
from flask_cors import CORS
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from repository.db import db
from routers.folderRouter import FolderRouter
from routers.videoRouter import VideoRouter, VideoImageRouter, VideoInfoRouter
from routers.frameRouter import FrameRouter
from routers.storageRouter import StorageRouter, OrphanDeleterRouter
from routers.skillRouter import SkillRouter, OptionRouter
from routers.downloadRouter import DownloadRouter

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
api.add_resource(VideoInfoRouter, '/video/<int:videoId>/info')
api.add_resource(VideoImageRouter, '/video/<int:videoId>/image')

api.add_resource(FrameRouter, '/video/<int:videoId>/frameNr/<int:frameNr>')
api.add_resource(OptionRouter, '/skilloptions/<skilltype>/<tableinfo>')
api.add_resource(SkillRouter, '/skill/<int:videoId>')
api.add_resource(StorageRouter, '/discover')
api.add_resource(OrphanDeleterRouter, '/discover/deleteOrphans')
api.add_resource(DownloadRouter, '/download')

if __name__ == '__main__':
    app.run(port=5555, debug=True)
