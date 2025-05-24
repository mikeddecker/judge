from models.HAR_SA_Conv3D_pytorch_1 import get_model as HAR_SA_Conv3D_pytorch_1
from models.HAR_MViT_pytorch import get_model as HAR_MViT_pytorch
from models.HAR_MViT_extra_dense_pytorch import get_model as HAR_MViT_extra_dense

KERAS_MODELS = {}
PYTORCH_MODELS_SKILLS = {
    "HAR_SA_Conv3D" : HAR_SA_Conv3D_pytorch_1,
    "HAR_MViT" : HAR_MViT_pytorch,
    "HAR_MViT" : HAR_MViT_extra_dense,
}


