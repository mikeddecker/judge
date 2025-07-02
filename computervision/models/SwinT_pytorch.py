import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torchvision.models as models
import numpy as np
import pandas as pd
from models.torch_output_layers import create_pytorch_skill_output_layers, create_pytorch_segmentation_output_layers, forward_skill_output_layers, forward_segmentation_output_layers

class SwinTransformer(nn.Module):
    def __init__(self, skill_or_segment:str, modelinfo:dict, df_table_counts:pd.DataFrame, variant:str):
        super(SwinTransformer, self).__init__()
        self.modelinfo = modelinfo
        self.df_table_counts = df_table_counts
        self.isSkillModel = skill_or_segment == "skills"
        
        input_shape = (3, modelinfo['timesteps'], modelinfo['dim'], modelinfo['dim'])
        match variant:
            case 't':
                self.swint = models.video.swin3d_t(weights='DEFAULT')
            case 's':
                self.swint = models.video.swin3d_s(weights='DEFAULT')
            case 'b':
                self.swint = models.video.swin3d_b(weights='DEFAULT')

        self.swint = self.swint.to('cuda').eval()

        for param in self.swint.parameters():
            param.requires_grad = False

        self.swint.head = torch.nn.Identity()  # This removes the top layer
        self.flatten = nn.Flatten()
        self.LastNNeurons = self._get_model_output(input_shape)
        
        if self.isSkillModel:
            self.output_layers = create_pytorch_skill_output_layers(lastNNeurons=self.LastNNeurons, balancedType=modelinfo['balancedType'], df_table_counts = self.df_table_counts)
        else:
            self.output_layer = create_pytorch_segmentation_output_layers(lastNNeurons=self.LastNNeurons, timesteps=modelinfo['timesteps'])

        
    def _get_model_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape).to('cuda')
            output = self.swint(input)
            output = self.flatten(output)
            return output.shape[1]
    
    def forward(self, x):
        # Input shape: (batch_size, channels, timesteps, height, width)
        x = self.swint(x)
        x = self.flatten(x)
        
        if self.isSkillModel:
            return forward_skill_output_layers(features=x, output_layers=self.output_layers)
        else:
            return forward_segmentation_output_layers(features=x, output_layer=self.output_layer)

def get_model(skill_or_segment:str, modelinfo, df_table_counts: pd.DataFrame, variant:str):
    """Build the swintransformer variant in PyTorch"""
    return SwinTransformer(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts, variant=variant)

def get_get_model(variant:str):
    """Returns the expected variant"""
    match variant:
        case 't':
            return lambda skill_or_segment, modelinfo, df_table_counts: get_model(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts, variant='t')
        case 's':
            return lambda skill_or_segment, modelinfo, df_table_counts: get_model(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts, variant='s')
        case 'b':
            return lambda skill_or_segment, modelinfo, df_table_counts: get_model(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts, variant='b')
        case _:
            raise ValueError(f"Got invalid resnet variant {variant}")