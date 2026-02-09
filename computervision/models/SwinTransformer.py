import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from types import SimpleNamespace
from models.JudgeVisionModel import JudgeVisionModel
from models.torch_output_layers import create_pytorch_segmentation_output_layers, forward_skill_output_layers, forward_segmentation_output_layers

class SwinTransformer(JudgeVisionModel):
    def __init__(self, head: nn.Module, recipe: SimpleNamespace):
        match recipe.variant:
            case 't':
                swin = models.video.swin3d_t(weights='DEFAULT')
            case 's':
                swin = models.video.swin3d_s(weights='DEFAULT')
            case 'b':
                swin = models.video.swin3d_b(weights='DEFAULT')
        swin.head = torch.nn.Identity()  # This removes the top layer

        super(SwinTransformer, self).__init__(
            backbone=swin,
            head=head
        )

    @staticmethod
    def get_output_feature_dim(recipe: SimpleNamespace) -> int:
        backbone = models.video.swin3d_s(weights='DEFAULT')
        return JudgeVisionModel.get_output_feature_dim(
            recipe=recipe,
            backbone=backbone
        )
