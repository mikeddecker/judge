from models.SA_Conv3D_pytorch_1 import get_model as SA_Conv3D_pytorch_1
from models.MViT_pytorch import get_model as MViT_pytorch
from models.MViT_extra_dense_pytorch import get_model as MViT_extra_dense
from models.Resnet import get_get_model as Resnet_get_model
from models.SwinT_pytorch import get_get_model as get_SwinT

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

