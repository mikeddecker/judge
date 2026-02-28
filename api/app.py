import atexit
import os
import signal
import sys
import subprocess

from apscheduler.schedulers.background import BackgroundScheduler
from config import ENVS
from datetime import datetime, date
from domain.tag import Tag
from flask import Flask, jsonify
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from flask_restful import Api
from flask_migrate import Migrate, upgrade
from flask_session import Session
from flask_talisman import Talisman
from helpers.ValueHelper import ValueHelper
from repository.db import db
from routers.folderRouter import FolderRouter
from routers.healthRouter import HealthRouter
from routers.videoRouter import VideoRouter, VideoRouterCropped, VideoImageRouter, VideoInfoRouter, VideoPredictionRouter, VideoPredictionRouter_GetLocalizePredictions, VideoPredictionRouter_HasLocalizePredictions
from routers.frameRouter import FrameRouter, FrameLabelTypeRouter
from routers.jobRouter import JobTrainRouter, JobLaunchRouter, JobOptionsRouter
from routers.storageRouter import StorageRouter, OrphanDeleterRouter
from routers.skillRouter import SkillRouter, SkillLevel, SkillLabelingCompletedRouter, DiffScoreComparison, SkillCount
from routers.downloadRouter import DownloadRouter
from routers.mlLayerRouter import MLLayerRouter, MLLayerTypesRouter, MLLayerCompositionRouter, MLLayerCompositionMoveLayerRouter, MLLayerCompositionAttributeRouter
from routers.resultsRouter import ResultsRouter
from routers.statsRouter import StatsRouter
from routers.tagRouter import TagRouter, TagGroupRouter
from routers.accountRouter import AccountRegisterRouter, AccountLoginRouter, AccountMFAVerifyRouter, AccountLogoutRouter, AccountMeRouter, AccountForgotPasswordRouter, AccountResetPasswordRouter, AccountEnableMFARouter
from services.videoService import VideoService
from typing import cast
from werkzeug.routing import BaseConverter
import uuid

class CustomJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, uuid.UUID):
            return str(obj)

        return super().default(obj)

    def encode(self, obj):
        if isinstance(obj, dict):
            obj = {str(k) if isinstance(k, uuid.UUID) else k: v for k, v in obj.items()}
        return super().encode(obj)

    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, dict):
            obj = {str(k) if isinstance(k, uuid.UUID) else k: v for k, v in obj.items()}
        return super().iterencode(obj, _one_shot)

class UUIDConverter(BaseConverter):
    # To use <uuid:videoId> in the routes
    def to_python(self, value):
        return uuid.UUID(value)  # string -> UUID object on the way in

    def to_url(self, value):
        return str(value)  # UUID object -> string for url_for()

MYSQL_ROOT_PASSWORD : str = cast(str, os.getenv("MYSQL_ROOT_PASSWORD"))
MYSQL_HOST : str = cast(str, os.getenv("MYSQL_HOST"))
MYSQL_DATABASE : str = cast(str, os.getenv("MYSQL_DATABASE"))
MYSQL_DOCKER_PORT : int = cast(int, os.getenv("MYSQL_DOCKER_PORT"))
MYSQL_BACKUP : str = cast(str, os.getenv("MYSQL_BACKUP"))
assert MYSQL_ROOT_PASSWORD is not None, f"Fill in the MYSQL_ROOT_PASSWORD variable in the .env file, located in the api folder."
assert MYSQL_HOST is not None, f"Fill in the MYSQL_HOST variable in the .env file, located in the api folder."
assert MYSQL_DATABASE is not None, f"Fill in the MYSQL_DATABASE variable in the .env file, located in the api folder."
assert MYSQL_DOCKER_PORT is not None, f"Fill in the MYSQL_DOCKER_PORT variable in the .env file, located in the api folder."
assert MYSQL_BACKUP is not None, f"Fill in the MYSQL_BACKUP variable in the .env file, located in the api folder."

API_DOCKER_PORT : str = cast(str, os.getenv("API_DOCKER_PORT", "5555"))
print(f"🔧 Starting API on port: {API_DOCKER_PORT}")

migrate = Migrate()

def restore_latest_mysql_backup(backup_dir: str = MYSQL_BACKUP):
    """
    Restore the latest MySQL backup from a directory.
    """
    if not os.path.exists(backup_dir):
        print(f"⚠️ Backup directory does not exist: {backup_dir}, no backup restored.")
        return

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

    views_sql = os.path.join(os.getcwd(), 'repository', 'views.sql')
    print(f"⏳ Running {views_sql}")
    with open(views_sql, "r") as f:
        subprocess.run(restore_cmd, stdin=f, check=True, env=env)
    print(f"✅ {views_sql} applied successfully")

def is_running_manual_migrations():
    # Detect if the current process is running flask db migrate/upgrade
    # This is yes when inside the container flask db upgrade is executed
    return any(arg in sys.argv for arg in ("db", "migrate", "upgrade"))

def create_app(config_object:str="config.Config"):
    app = Flask(__name__)
    CORS(app)

    # Load configuration from config file or environment variable
    app.config.from_object(config_object)

    # Set session secret key for secure session management
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-min-32-chars')
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)

    # Force HTTPS in production only
    if os.getenv('FLASK_ENV') == 'production':
        Talisman(app, force_https=True, strict_transport_security=True, strict_transport_security_max_age=63072000)

    if not app.config.get('TESTING', False) and not is_running_manual_migrations() and os.path.exists('migrations'):
        # TODO : restore only if db empty or not at last db version or after GUI/cmd
        # print("⏳ Restoring latest MySQL backup before starting the app...")
        # restore_latest_mysql_backup()
        pass

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    if os.path.exists('migrations'):
        with app.app_context():
            upgrade()

    return app

app = create_app()
app.json = CustomJSONProvider(app)
app.url_map.converters['uuid'] = UUIDConverter
CORS(app)
api : Api = Api(app)

app.config['RESTFUL_JSON'] = {
    'default': app.json.default
}

# use api.add_resource to add the paths
api.add_resource(FolderRouter, '/folders', '/folders/<uuid:folderId>')
api.add_resource(VideoRouter, '/video/<uuid:videoId>')
api.add_resource(VideoRouterCropped, '/video/<uuid:videoId>/cropped')
api.add_resource(VideoInfoRouter, '/video/<uuid:videoId>/info')
api.add_resource(VideoImageRouter, '/video/<uuid:videoId>/image')
api.add_resource(VideoPredictionRouter, '/video/<uuid:videoId>/predictions')
api.add_resource(VideoPredictionRouter_HasLocalizePredictions, '/video/<uuid:videoId>/predictions/hasLocalizePredictions')
api.add_resource(VideoPredictionRouter_GetLocalizePredictions, '/video/<uuid:videoId>/predictions/getLocalizePredictions')
api.add_resource(TagRouter, '/tags')
api.add_resource(TagGroupRouter, '/tagGroups')
api.add_resource(MLLayerRouter, '/layers')
api.add_resource(MLLayerTypesRouter, '/layers/types')
api.add_resource(MLLayerCompositionMoveLayerRouter, '/layers/move')
api.add_resource(MLLayerCompositionRouter, '/layercompositions')
api.add_resource(MLLayerCompositionAttributeRouter, '/layercompositions/attribute')
api.add_resource(FrameRouter, '/video/<uuid:videoId>/frameNr/<int:frameNr>')
api.add_resource(FrameLabelTypeRouter, '/frameLabelTypes')
api.add_resource(SkillRouter, '/skill/<uuid:videoId>')
api.add_resource(SkillLevel, '/skilllevel')
api.add_resource(SkillCount, '/skill/count')
api.add_resource(DiffScoreComparison, '/diff-score-comparison')
api.add_resource(SkillLabelingCompletedRouter, '/skillcompleted/<uuid:videoId>')
api.add_resource(StorageRouter, '/discover')
api.add_resource(OrphanDeleterRouter, '/discover/deleteOrphans')
api.add_resource(DownloadRouter, '/download')
api.add_resource(JobTrainRouter, '/job/retrain')
api.add_resource(JobLaunchRouter, '/job')
api.add_resource(JobOptionsRouter, '/job/options/<step>')

api.add_resource(StatsRouter, '/stats')
api.add_resource(ResultsRouter, '/results')
api.add_resource(HealthRouter, '/health')

# Account authentication routes
api.add_resource(AccountRegisterRouter, '/auth/register')
api.add_resource(AccountLoginRouter, '/auth/login')
api.add_resource(AccountMFAVerifyRouter, '/auth/mfa/verify')
api.add_resource(AccountLogoutRouter, '/auth/logout')
api.add_resource(AccountMeRouter, '/auth/me')
api.add_resource(AccountForgotPasswordRouter, '/auth/forgot-password')
api.add_resource(AccountResetPasswordRouter, '/auth/reset-password')
api.add_resource(AccountEnableMFARouter, '/auth/enable-mfa')

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
    try:
        pass
        # backup_mysql_db()
    except Exception as e:
        print(e)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler) # Handle Ctrl+C and kill
signal.signal(signal.SIGTERM, shutdown_handler) # Handle Ctrl+C and kill
atexit.register(shutdown_handler) # Also run on interpreter exit
scheduler = BackgroundScheduler()
scheduler.add_job(backup_mysql_db, "interval", hours=1)
scheduler.start()

if __name__ == '__main__':
    with app.app_context():
        videoservice = VideoService()
        videoservice.initiate()
        del videoservice
    app.run(host="0.0.0.0", port=API_DOCKER_PORT, debug=True, use_reloader=False)

