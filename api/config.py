import os

from dotenv import load_dotenv
from types import SimpleNamespace
from helpers.helpers import load_json_file

load_dotenv()

class Config:
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv(f"DATABASE_URL")  # For development or testing purposes

class TestConfig:
    TESTING = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv(f"DATABASE_URL_TEST")  # For development or testing purposes

ENVS = SimpleNamespace(
    DATABASE = SimpleNamespace(
        MYSQLDB_DATABASE = os.getenv('MYSQLDB_DATABASE'),
        MYSQLDB_DATABASE_TEST = os.getenv('MYSQLDB_DATABASE_TEST'),
        MYSQLDB_ROOT_PASSWORD = os.getenv('MYSQLDB_ROOT_PASSWORD'),
        MYSQLDB_USERNAME = os.getenv('MYSQLDB_USERNAME'),
        MYSQLDB_LOCAL_PORT = os.getenv('MYSQLDB_LOCAL_PORT'),
        MYSQLDB_DOCKER_PORT = os.getenv('MYSQLDB_DOCKER_PORT'),
        HOST = os.getenv('HOST'),
    ),
    DIRS = SimpleNamespace(
        VIDEOS = os.getenv("STORAGE_DIR_VIDEOS"),
        GENERATED = os.getenv("STORAGE_DIR_GENERATED_DATA"),
        GENERATED_VIDEODATA = os.path.join(os.getenv("STORAGE_DIR_GENERATED_DATA"), 'videodata'),
        WEIGHTS = os.path.join(os.getenv("STORAGE_DIR_GENERATED_DATA"), 'weights'),
        YOLO_LABELS = os.path.join(os.getenv("STORAGE_DIR_GENERATED_DATA"), 'labels-ultralytics-yolo'),
    ),
    SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS"),
    SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS"),
)

RECIPES = {
    step: {
        recipename: SimpleNamespace(**kwargs) 
        for recipename, kwargs in step_recipes.items()
    }
    for step, step_recipes in load_json_file('../recipes.json').items()
}

JOB_TYPES = ['TRAIN', 'PREDICT']
JOB_STEPS = ['LOCALIZE', 'SEGMENT', 'RECOGNIZE', 'FULL']
LAYER_TYPES = ['boolean', 'categorical', 'numerical']