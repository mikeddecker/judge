from models.HAR_SA_Conv3D_pytorch_1 import get_model as HAR_SA_Conv3D_pytorch_1
from models.HAR_MViT_pytorch import get_model as HAR_MViT_pytorch
from models.HAR_MViT_extra_dense_pytorch import get_model as HAR_MViT_extra_dense
from models.HAR_Resnet import get_get_model as HAR_Resnet_get_model
from models.HAR_SwinT_pytorch import get_get_model as HAR_get_SwinT

KERAS_MODELS = {}
PYTORCH_MODELS_SKILLS = {
    "HAR_SA_Conv3D" : HAR_SA_Conv3D_pytorch_1,
    "HAR_MViT" : HAR_MViT_pytorch,
    "HAR_MViT_extra_dense" : HAR_MViT_extra_dense,
    "HAR_Resnet_R3D" : HAR_Resnet_get_model('R3D'),
    "HAR_Resnet_MC3" : HAR_Resnet_get_model('MC3'),
    "HAR_Resnet_R2plus1" : HAR_Resnet_get_model('R2plus1'),
    "HAR_SwinT_t" : HAR_get_SwinT('t'),
    "HAR_SwinT_s" : HAR_get_SwinT('s'),
}

