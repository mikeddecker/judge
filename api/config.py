import os

from dotenv import load_dotenv
from types import SimpleNamespace
from helpers.helpers import load_json_file

load_dotenv()

class Config:
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv(f"DATABASE_URL")  # For development or testing purposes
    print(f"Using database URL: {SQLALCHEMY_DATABASE_URI}")

class TestConfig:
    TESTING = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv(f"DATABASE_URL_TEST")  # For development or testing purposes

ENVS = SimpleNamespace(
    DATABASE = SimpleNamespace(
        MYSQL_DATABASE = os.getenv('MYSQL_DATABASE'),
        MYSQL_DATABASE_TEST = os.getenv('MYSQL_DATABASE_TEST'),
        MYSQL_ROOT_PASSWORD = os.getenv('MYSQL_ROOT_PASSWORD'),
        MYSQL_USERNAME = os.getenv('MYSQL_USERNAME'),
        MYSQL_LOCAL_PORT = os.getenv('MYSQL_LOCAL_PORT'),
        MYSQL_DOCKER_PORT = os.getenv('MYSQL_DOCKER_PORT'),
        MYSQL_HOST = os.getenv('MYSQL_HOST'),
    ),
    DIRS = SimpleNamespace(
        VIDEOS = os.getenv("STORAGE_DIR_VIDEOS"),
        GENERATED = os.getenv("STORAGE_DIR_GENERATED_DATA"),
        GENERATED_VIDEODATA = os.path.join(os.getenv("STORAGE_DIR_GENERATED_DATA"), 'videodata'),
        WEIGHTS = SimpleNamespace(
            YOLO = os.path.join(os.getenv("STORAGE_DIR_GENERATED_DATA"), 'weights', 'yolo'),
            SKILLS = os.path.join(os.getenv("STORAGE_DIR_GENERATED_DATA"), 'weights', 'skills'),
        ),
        YOLO_LABELS = os.path.join(os.getenv("STORAGE_DIR_GENERATED_DATA"), 'labels-ultralytics-yolo'),
    ),
    SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS"),
    SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS"),
)

RECIPES = {
    step: {
        recipename: SimpleNamespace(**kwargs, name=recipename) 
        for recipename, kwargs in step_recipes.items()
    }
    for step, step_recipes in load_json_file('/machine_learning_recipes.json').items()
}

JOB_TYPES = ['TRAIN', 'PREDICT']
JOB_STEPS = ['LOCALIZE', 'SEGMENT', 'SKILL', 'FULL']
LAYER_TYPES = ['boolean', 'categorical', 'numerical']

STAGES = ['GeneralProperties', 'StartProperties', 'EndProperties', 'StageProperties']
STAGE_MAP = {
    'GeneralProperties' : None,
    'StartProperties' : 0,
    'EndProperties' : -1
}

