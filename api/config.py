import os

from dotenv import load_dotenv
from types import SimpleNamespace
from helpers.helpers import load_json_file

load_dotenv()
MYSQL_USERNAME = os.getenv('MYSQL_USERNAME')
MYSQL_ROOT_PASSWORD = os.getenv('MYSQL_ROOT_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_DOCKER_PORT = os.getenv('MYSQL_DOCKER_PORT')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

# Distinction will be made in container / call method
DATABASE_URL = f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_ROOT_PASSWORD}@{MYSQL_HOST}:{MYSQL_DOCKER_PORT}/{MYSQL_DATABASE}"

class Config:
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    
    # Session configuration
    SESSION_COOKIE_SECURE = os.getenv('FLASK_ENV') == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 86400 * 7  # 7 days in seconds (604800)
    SESSION_REFRESH_EACH_REQUEST = True
    
    print(f"Using database URL: {SQLALCHEMY_DATABASE_URI}")

class TestConfig:
    TESTING = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    
    # Session configuration for testing
    SESSION_COOKIE_SECURE = False  # Always False for testing (HTTP)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour for testing
    SESSION_REFRESH_EACH_REQUEST = True
    
    print(f"Using database URL: {SQLALCHEMY_DATABASE_URI}")

ENVS = SimpleNamespace(
    DATABASE = SimpleNamespace(
        MYSQL_DATABASE = os.getenv('MYSQL_DATABASE'),
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

_RECIPES_PATH = os.getenv('ML_RECIPES_PATH', '/machine_learning_recipes.json')
_RECIPES_FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'machine_learning_recipes.json')
_raw_recipes = load_json_file(_RECIPES_PATH) or load_json_file(_RECIPES_FALLBACK) or {}

RECIPES = {
    step: {
        recipename: SimpleNamespace(**kwargs, name=recipename) 
        for recipename, kwargs in step_recipes.items()
    }
    for step, step_recipes in _raw_recipes.items()
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

LEVEL_TO_SCORE_MAP = {
    0 : 0,
    1 : 1.5,
    2 : 2.2,
    3 : 3.3,
    4 : 4.9,
    5 : 7.3,
    6 : 11,
    7 : 11,
    8 : 11,
}

