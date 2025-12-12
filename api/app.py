import atexit
import os
import signal
import sys
import subprocess

from config import ENVS
from datetime import datetime
from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from flask_cors import CORS
from flask_migrate import Migrate, upgrade
from helpers.ValueHelper import ValueHelper
from repository.db import db
from routers.folderRouter import FolderRouter
from routers.healthRouter import HealthRouter
from routers.videoRouter import VideoRouter, VideoRouterCropped, VideoImageRouter, VideoInfoRouter, VideoPredictionRouter, VideoPredictionRouter_GetLocalizePredictions, VideoPredictionRouter_HasLocalizePredictions
from routers.frameRouter import FrameRouter, FrameLabelTypeRouter
from routers.jobRouter import JobTrainRouter, JobLaunchRouter, JobOptionsRouter
from routers.storageRouter import StorageRouter, OrphanDeleterRouter
from routers.skillRouter import SkillRouter, SkillLevel, SkillLabelingCompletedRouter, DiffScoreComparison
from routers.downloadRouter import DownloadRouter
from routers.mlLayerRouter import MLLayerRouter, MLLayerTypesRouter, MLLayerCompositionRouter, MLLayerCompositionMovePropertyRouter, MLLayerCompositionPropertyAttributeRouter
from routers.statsRouter import StatsRouter
from routers.tagRouter import TagRouter, TagGroupRouter
from services.videoService import VideoService
from typing import cast

MYSQL_ROOT_PASSWORD : str = cast(str, os.getenv("MYSQL_ROOT_PASSWORD"))
MYSQL_HOST : str = cast(str, os.getenv("MYSQL_HOST"))
MYSQL_DATABASE : str = cast(str, os.getenv("MYSQL_DATABASE"))
MYSQL_DOCKER_PORT : str = cast(str, os.getenv("MYSQL_DOCKER_PORT"))
MYSQL_BACKUP : str = cast(str, os.getenv("MYSQL_BACKUP"))
assert MYSQL_ROOT_PASSWORD is not None, f"Fill in the MYSQL_ROOT_PASSWORD variable in the .env file, located in the api folder."
assert MYSQL_HOST is not None, f"Fill in the MYSQL_HOST variable in the .env file, located in the api folder."
assert MYSQL_DATABASE is not None, f"Fill in the MYSQL_DATABASE variable in the .env file, located in the api folder."
assert MYSQL_DOCKER_PORT is not None, f"Fill in the MYSQL_DOCKER_PORT variable in the .env file, located in the api folder."
assert MYSQL_BACKUP is not None, f"Fill in the MYSQL_BACKUP variable in the .env file, located in the api folder."

API_DOCKER_PORT : str = cast(str, os.getenv("API_DOCKER_PORT", "5555"))

migrate = Migrate()

def restore_latest_mysql_backup(backup_dir: str = MYSQL_BACKUP):
    """
    Restore the latest MySQL backup from a directory.
    """
    if not os.path.exists(backup_dir):
        raise FileNotFoundError(f"Backup directory does not exist: {backup_dir}")

    # Find the latest .sql backup file
    sql_files = [f for f in os.listdir(backup_dir) if f.endswith(".sql")]
    if not sql_files:
        print(f"⚠️ No database backup found to restore from")
        return
    
    # Sort by modification time descending, pick the newest
    sql_files.sort(key=lambda f: os.path.getmtime(os.path.join(backup_dir, f)), reverse=True)
    latest_backup = os.path.join(backup_dir, sql_files[0])

    print(f"⏳ Restoring latest backup: {latest_backup}")

    # Use env var to avoid password on command line
    env = os.environ.copy()
    env["MYSQL_PWD"] = MYSQL_ROOT_PASSWORD

    restore_cmd = [
        "mysql",
        "-u", "root",
        "-h", MYSQL_HOST,
        "-P", str(MYSQL_DOCKER_PORT),
        MYSQL_DATABASE,
    ]

    # Run restore
    with open(latest_backup, "r") as f:
        subprocess.run(restore_cmd, stdin=f, check=True, env=env)

    print(f"✅ Database restored from: {latest_backup}")

def create_app(config_object:str="config.Config"):
    app = Flask(__name__)
    CORS(app)
    
    # Load configuration from config file or environment variable
    app.config.from_object(config_object)
    
    restore_latest_mysql_backup()
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        upgrade()
    
    return app

app = create_app()
CORS(app)
api : Api = Api(app)

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
api.add_resource(MLLayerCompositionMovePropertyRouter, '/layers/move')
api.add_resource(MLLayerCompositionRouter, '/layercompositions')
api.add_resource(MLLayerCompositionPropertyAttributeRouter, '/layercompositions/attribute')
api.add_resource(FrameRouter, '/video/<int:videoId>/frameNr/<int:frameNr>')
api.add_resource(FrameLabelTypeRouter, '/frameLabelTypes')
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
api.add_resource(HealthRouter, '/health')

# Check if .env folders are filled in
ValueHelper.check_raise_string(ENVS.DIRS.VIDEOS)
ValueHelper.check_raise_string(ENVS.DIRS.GENERATED)
ValueHelper.check_raise_string(ENVS.DIRS.GENERATED_VIDEODATA)
ValueHelper.check_raise_string(ENVS.DIRS.WEIGHTS.YOLO)
ValueHelper.check_raise_string(ENVS.DIRS.WEIGHTS.SKILLS)
ValueHelper.check_raise_string(ENVS.DIRS.YOLO_LABELS)

os.makedirs(ENVS.DIRS.GENERATED, exist_ok=True)
os.makedirs(ENVS.DIRS.GENERATED_VIDEODATA, exist_ok=True)
os.makedirs(ENVS.DIRS.WEIGHTS.YOLO, exist_ok=True)
os.makedirs(ENVS.DIRS.WEIGHTS.SKILLS, exist_ok=True)
os.makedirs(ENVS.DIRS.YOLO_LABELS, exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'labels'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(ENVS.DIRS.YOLO_LABELS, 'images', 'val'), exist_ok=True)

def backup_mysql_db(backup_dir: str = MYSQL_BACKUP):
    """Create a MySQL database backup using mysqldump"""
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    backup_file = os.path.join(backup_dir, f"{MYSQL_DATABASE}_{timestamp}.sql")

    # Use env var to avoid "Using a password on the command line is insecure"
    env = os.environ.copy()
    env["MYSQL_PWD"] = MYSQL_ROOT_PASSWORD

    dump_cmd = [
        "mysqldump",
        "-u", "root",
        "-h", MYSQL_HOST,
        "-P", str(MYSQL_DOCKER_PORT),
        MYSQL_DATABASE,
    ]

    with open(backup_file, "w") as f:
        # Write CREATE DATABASE + USE at the top of the backup file
        f.write(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;\n")
        f.write(f"USE `{MYSQL_DATABASE}`;\n\n")

        subprocess.run(dump_cmd, stdout=f, check=True, env=env)

    print(f"✅ Database backup created: {backup_file}")

_shutdown_called = False
def shutdown_handler(*args):
    global _shutdown_called
    if _shutdown_called:
        return
    _shutdown_called = True
    print("⚠️ Shutting down Flask app, creating backup...")
    backup_mysql_db()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler) # Handle Ctrl+C and kill
signal.signal(signal.SIGTERM, shutdown_handler) # Handle Ctrl+C and kill
atexit.register(shutdown_handler) # Also run on interpreter exit

if __name__ == '__main__':
    with app.app_context():
        videoservice = VideoService()
        videoservice.initiate()
        del videoservice
    app.run(host="0.0.0.0", port=API_DOCKER_PORT, debug=True, use_reloader=False)

