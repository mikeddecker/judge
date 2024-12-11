import os

from flask import Flask, request, jsonify, current_app, g
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from repository.db import db
from routers.folderRouter import FolderRouter

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
    
    from repository.models import Folder as FolderDB, Video as VideoDB

    return app

app = create_app()
api = Api(app)

# use api.add_resource to add the paths
api.add_resource(FolderRouter, '/folders', '/folders/<int:user_id>')

if __name__ == '__main__':
    app.run(port=5555, debug=True)