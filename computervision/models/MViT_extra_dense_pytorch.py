import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torchvision.models as models
import numpy as np
import pandas as pd
from models.torch_output_layers import create_pytorch_skill_output_layers, create_pytorch_segmentation_output_layers, forward_skill_output_layers, forward_segmentation_output_layers

import sys
sys.path.append('..')
from api.helpers.ConfigHelper import get_discipline_DoubleDutch_config
    

class MViT(nn.Module):
    def __init__(self, skill_or_segment:str, modelinfo:dict, df_table_counts:pd.DataFrame):
        super(MViT, self).__init__()
        self.modelinfo = modelinfo
        self.df_table_counts = df_table_counts
        self.isSkillModel = skill_or_segment == "skills"
        
        input_shape = (3, modelinfo['timesteps'], modelinfo['dim'], modelinfo['dim'])
        self.mvit = models.video.mvit_v1_b(weights='DEFAULT')
        self.mvit = self.mvit.to('cuda').eval()

        for param in self.mvit.parameters():
            param.requires_grad = False

        self.mvit.head = torch.nn.Identity()  # This removes the top layer
        self.LastNNeurons = 384
        self.flatten = nn.Flatten()
        self.features = nn.Linear(self._get_mvit_output(input_shape), self.LastNNeurons)
        
        if self.isSkillModel:
            self.output_layers = create_pytorch_skill_output_layers(lastNNeurons=self.LastNNeurons, balancedType=modelinfo['balancedType'], df_table_counts = self.df_table_counts) # TODO : make dynamic
        else:
            self.output_layer = create_pytorch_segmentation_output_layers(lastNNeurons=self.LastNNeurons, timesteps=modelinfo['timesteps'])

        
    def _get_mvit_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape).to('cuda')
            output = self.mvit(input)
            output = self.flatten(output)
            return output.shape[1]
    
    def forward(self, x):
        # Input shape: (batch_size, channels, timesteps, height, width)
        x = self.mvit(x)
        x = self.flatten(x)
        features = F.relu(self.features(x))
        
        if self.isSkillModel:
            return forward_skill_output_layers(features=features, output_layers=self.output_layers)
        else:
            return forward_segmentation_output_layers(features=features, output_layer=self.output_layer)

def get_model(skill_or_segment:str, modelinfo, df_table_counts: pd.DataFrame):
    """Build a Self-Attention ConvLSTM model in PyTorch"""
    return MViT(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts)