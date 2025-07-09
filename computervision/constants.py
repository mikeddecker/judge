import os

from dotenv import load_dotenv
from models.SA_Conv3D_pytorch_1 import get_model as SA_Conv3D_pytorch_1
from models.MViT_pytorch import get_model as MViT_pytorch
from models.MViT_extra_dense_pytorch import get_model as MViT_extra_dense
from models.Resnet import get_get_model as Resnet_get_model
from models.SwinT_pytorch import get_get_model as get_SwinT
from types import SimpleNamespace

load_dotenv()

ENVS = SimpleNamespace(
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

# TODO : modelparams -> extend this so more general and more different settings can be created
DIM = 224 # TODO : transition to RECIPES
RECIPES = SimpleNamespace(
    MViT = SimpleNamespace(
        DIM = 224
    )
)
