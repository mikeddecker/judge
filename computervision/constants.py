import os

from dotenv import load_dotenv
from base_utils import load_json_file
from models.SA_Conv3D_pytorch_1 import get_model as SA_Conv3D_pytorch_1
from models.MViT_pytorch import get_model as MViT_pytorch, MViT
from models.MViT_extra_dense_pytorch import get_model as MViT_extra_dense, MViT_Dense
from models.Resnet import get_get_model as Resnet_get_model
from models.SwinT_pytorch import get_get_model as get_SwinT
from types import SimpleNamespace

load_dotenv()

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

KERAS_MODELS = {}
PYTORCH_MODELS_SKILLS = {
    "SA_Conv3D" : SA_Conv3D_pytorch_1,
    "MViT" : MViT_pytorch,
    "MViT_extra_dense" : MViT_extra_dense,
    "Resnet_R3D" : Resnet_get_model('R3D'),
    "Resnet_MC3" : Resnet_get_model('MC3'),
    "Resnet_R2plus1" : Resnet_get_model('R2plus1'),
    "SwinT_t" : get_SwinT('t'),
    "SwinT_s" : get_SwinT('s'),
}

PYTORCH_MODELS_SKILLS_TEST = {
    "MViT" : MViT,
    "MViT_extra_dense" : MViT_Dense,
    # "SA_Conv3D" : SA_Conv3D_pytorch_1,
    # "Resnet_R3D" : Resnet_get_model('R3D'),
    # "Resnet_MC3" : Resnet_get_model('MC3'),
    # "Resnet_R2plus1" : Resnet_get_model('R2plus1'),
    # "SwinT_t" : get_SwinT('t'),
    # "SwinT_s" : get_SwinT('s'),
}

DIM = 224
RECIPES = {
    step: {
        recipename: SimpleNamespace(**kwargs, name=recipename) 
        for recipename, kwargs in step_recipes.items()
    }
    for step, step_recipes in load_json_file('recipes.json').items()
}

JOB_TYPES = ['TRAIN', 'PREDICT']
JOB_STEPS = ['LOCALIZE', 'SEGMENT', 'SKILL', 'FULL']
SPEEDMODES = ['quick', 'selective', 'all']
LAYER_TYPES = ['boolean', 'categorical', 'numerical']

STAGES = ['GeneralProperties', 'StartProperties', 'EndProperties', 'StageProperties']
STAGE_MAP = {
    'GeneralProperties' : None,
    'StartProperties' : 0,
    'EndProperties' : -1
}

