import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torchvision.models as models
import numpy as np
import pandas as pd
from models.torch_output_layers import create_pytorch_skill_output_layers, create_pytorch_segmentation_output_layers, forward_skill_output_layers, forward_segmentation_output_layers
    

class R3D(nn.Module):
    """Based on resnet: https://docs.pytorch.org/vision/main/models/video_resnet.html"""
    def __init__(self, skill_or_segment:str, modelinfo:dict, df_table_counts:pd.DataFrame, variant:str):
        super(R3D, self).__init__()
        self.modelinfo = modelinfo
        self.df_table_counts = df_table_counts
        self.isSkillModel = skill_or_segment == "skills"
        
        input_shape = (3, modelinfo['timesteps'], modelinfo['dim'], modelinfo['dim'])

        match variant:
            case 'R3D':
                self.model = models.video.resnet.r3d_18(weights='DEFAULT')
            case 'MC3':
                self.model = models.video.resnet.mc3_18(weights='DEFAULT')
            case 'R2plus1':
                self.model = models.video.resnet.r2plus1d_18(weights='DEFAULT')
            case _:
                raise ValueError(f"Got invalid resnet variant {variant}")
        
        self.model = self.model.to('cuda').eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = torch.nn.Identity()  # This removes the top layer, .fc?
        self.flatten = nn.Flatten()
        self.LastNNeurons = self._get_model_output(input_shape)
        
        if self.isSkillModel:
            self.output_layers = create_pytorch_skill_output_layers(lastNNeurons=self.LastNNeurons, balancedType=modelinfo['balancedType'], df_table_counts = self.df_table_counts)
        else:
            self.output_layer = create_pytorch_segmentation_output_layers(lastNNeurons=self.LastNNeurons, timesteps=modelinfo['timesteps'])

        
    def _get_model_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape).to('cuda')
            output = self.model(input)
            output = self.flatten(output)
            return output.shape[1]
    
    def forward(self, x):
        # Input shape: (batch_size, timesteps, channels, height, width)
        x = self.model(x)
        x = self.flatten(x)
        
        if self.isSkillModel:
            return forward_skill_output_layers(features=x, output_layers=self.output_layers)
        else:
            return forward_segmentation_output_layers(features=x, output_layer=self.output_layer)

def get_model_r3d(skill_or_segment:str, modelinfo, df_table_counts: pd.DataFrame):
    """Build the r3d resnet video variant in PyTorch"""
    return R3D(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts, variant='R3D')

def get_model_mc3(skill_or_segment:str, modelinfo, df_table_counts: pd.DataFrame):
    """Build the mc3 resnet video variant in PyTorch"""
    return R3D(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts, variant='MC3')

def get_model_r2plus1(skill_or_segment:str, modelinfo, df_table_counts: pd.DataFrame):
    """Build the r2plus1d resnet video variant in PyTorch"""
    return R3D(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts, variant='R2plus1')

def get_get_model(variant:str):
    """Returns the expected variant"""
    match variant:
        case 'R3D':
            return lambda skill_or_segment, modelinfo, df_table_counts: get_model_r3d(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts)
        case 'MC3':
            return lambda skill_or_segment, modelinfo, df_table_counts: get_model_mc3(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts)
        case 'R2plus1':
            return lambda skill_or_segment, modelinfo, df_table_counts: get_model_r2plus1(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts)
        case _:
            raise ValueError(f"Got invalid resnet variant {variant}")