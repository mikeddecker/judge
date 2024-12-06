from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate

from repository.models import Folder, Video, db
import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

app = Flask(__name__)
# applies CORS headers to all routes, enabling resources to be accessed
CORS(app)
print(DATABASE_URL)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.json.compact = False

migrate = Migrate(app, db)
db.init_app(app)

api = Api(app)

from routers.browseRouter import FolderRouter
# use api.add_resource to add the paths
api.add_resource(FolderRouter, '/folders', '/folders/<int:user_id>')

if __name__ == '__main__':
    app.run(port=5555, debug=True)