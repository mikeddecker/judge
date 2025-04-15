import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torchvision.models as models
import numpy as np
import pandas as pd
from models.torch_output_layers import create_pytorch_skill_output_layers

import sys
sys.path.append('..')
from api.helpers.ConfigHelper import get_discipline_DoubleDutch_config
    

class MViT(nn.Module):
    def __init__(self, modelinfo, df_table_counts):
        super(MViT, self).__init__()
        self.modelinfo = modelinfo
        self.df_table_counts = df_table_counts
        
        input_shape = (3, modelinfo['timesteps'], modelinfo['dim'], modelinfo['dim'])
        self.mvit = models.video.mvit_v1_b(weights='DEFAULT')
        self.mvit = self.mvit.to('cuda').eval()

        for param in self.mvit.parameters():
            param.requires_grad = False

        self.mvit.head = torch.nn.Identity()  # This removes the top layer
        self.LastNNeurons = 384
        self.flatten = nn.Flatten()
        self.features = nn.Linear(self._get_mvit_output(input_shape), self.LastNNeurons)
        
        self.output_layers = create_pytorch_skill_output_layers(lastNNeurons=self.LastNNeurons, balancedType='jump_return_push_frog_other', df_table_counts = self.df_table_counts) # TODO : make dynamic

        self.head = create_pytorch_skill_output_layers(lastNNeurons=self.LastNNeurons, balancedType='jump_return_push_frog_other', df_table_counts = self.df_table_counts) # TODO : make dynamic
        
    def _get_mvit_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape).to('cuda')
            output = self.mvit(input)
            output = self.flatten(output)
            return output.shape[1]
    
    def forward(self, x):
        # Input shape: (batch_size, timesteps, channels, height, width)
        x = self.mvit(x)
        x = self.flatten(x)
        features = F.relu(self.features(x))
        
        outputs = {}
        for key, layer in self.output_layers.items():
            if key in ['Skill', 'Turner1', 'Turner2', 'Type']:
                outputs[key] = layer(features)
            else:  # Regression outputs
                outputs[key] = torch.sigmoid(layer(features))
        
        return outputs

def get_model(modelinfo, df_table_counts: pd.DataFrame):
    """Build a Self-Attention ConvLSTM model in PyTorch"""
    return MViT(modelinfo, df_table_counts)