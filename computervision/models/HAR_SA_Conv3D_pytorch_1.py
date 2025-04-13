import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from api.helpers.ConfigHelper import get_discipline_DoubleDutch_config

class SelfAttention(nn.Module):
    """3D version of self-attention that handles temporal dimension"""
    def __init__(self, channels, kernel_size=1):
        super(SelfAttention, self).__init__()
        self.channels = channels
        
        # Use Conv3d instead of Conv2d
        self.query = nn.Conv3d(channels, channels, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.key = nn.Conv3d(channels, channels, kernel_size=kernel_size,
                            padding=kernel_size//2, bias=False)
        self.value = nn.Conv3d(channels, channels, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=False)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x shape: [batch, channels, timesteps, height, width]
        batch_size, C, T, H, W = x.size()
        
        # Project to query, key, value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        attn_scores = torch.einsum('bcthw,bcthw->bthw', q, k)
        attn_scores = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bthw,bcthw->bcthw', attn_scores, v)
        
        # Add skip connection
        out = self.gamma * out + x
        return out
    

class SAConv3D(nn.Module):
    def __init__(self, modelinfo, df_table_counts):
        super(SAConv3D, self).__init__()
        self.modelinfo = modelinfo
        self.df_table_counts = df_table_counts
        
        input_shape = (3,modelinfo['timesteps'], modelinfo['dim'], modelinfo['dim'])
        kernel_size = 3
        filters = 32
        
        # Conv3D layers
        self.conv1 = nn.Conv3d(3, filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv3d(filters, int(filters * 1.5), kernel_size=kernel_size, padding=kernel_size//2)
        self.attn1 = SelfAttention(int(filters * 1.5))
        
        self.conv3 = nn.Conv3d(int(filters * 1.5), int(filters * 2), kernel_size=kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv3d(int(filters * 2), int(filters * 2), kernel_size=kernel_size, padding=kernel_size//2)
        self.attn2 = SelfAttention(int(filters * 2))
        
        self.conv5 = nn.Conv3d(int(filters * 2), int(filters * 3), kernel_size=kernel_size, 
                              stride=(1, 2, 2), padding=kernel_size//2)
        self.conv6 = nn.Conv3d(int(filters * 3), int(filters * 4), kernel_size=kernel_size, padding=kernel_size//2)
        self.attn3 = SelfAttention(int(filters * 4))
        
        self.conv7 = nn.Conv3d(int(filters * 4), int(filters * 4), kernel_size=kernel_size, padding=kernel_size//2)
        self.conv8 = nn.Conv3d(int(filters * 4), int(filters * 4), kernel_size=kernel_size, padding=kernel_size//2)
        self.attn4 = SelfAttention(int(filters * 4))
        
        self.conv9 = nn.Conv3d(int(filters * 4), int(filters * 6), kernel_size=kernel_size, stride=2, padding=kernel_size//2)
        self.conv10 = nn.Conv3d(int(filters * 6), int(filters * 6), kernel_size=kernel_size, padding=kernel_size//2)
        self.conv11 = nn.Conv3d(int(filters * 6), int(filters * 8), kernel_size=kernel_size, 
                               stride=(1, 2, 2), padding=kernel_size//2)
        self.conv12 = nn.Conv3d(int(filters * 8), int(filters * 8), kernel_size=1, stride=2)
        
        # Flatten and dense layers
        self.flatten = nn.Flatten()
        self.LastNNeurons = 256
        self.features = nn.Linear(self._get_conv_output(input_shape), self.LastNNeurons)
        
        # Output layers
        self._create_output_layers(balancedType='jump_return_push_frog_other') # TODO : make dynamic
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.conv1(input)
            output = self.conv2(output)
            output = self.attn1(output)
            output = self.conv3(output)
            output = self.conv4(output)
            output = self.attn2(output)
            output = self.conv5(output)
            output = self.conv6(output)
            output = self.attn3(output)
            output = self.conv7(output)
            output = self.conv8(output)
            output = self.attn4(output)
            output = self.conv9(output)
            output = self.conv10(output)
            output = self.conv11(output)
            output = self.conv12(output)
            output = self.flatten(output)
            return output.shape[1]
        
    def _create_output_layers(self, balancedType=None):
        dd_config = get_discipline_DoubleDutch_config()
        self.output_layers = nn.ModuleDict()
        
        for key, value in dd_config.items():
            if key == "Tablename":
                continue
            if value[0] == "Categorical":
                columnname = "skill"
                if key == 'Skill':
                    columnname = 'skills'
                elif key in ['Turner1', 'Turner2']:
                    columnname = "turners"
                elif key == 'Type':
                    columnname = 'types'
                
                classes = int(self.df_table_counts.iloc[0][columnname])
                self.output_layers[key] = nn.Linear(self.LastNNeurons, classes)
            else:
                self.output_layers[key] = nn.Linear(self.LastNNeurons, 1)
        
        if balancedType == 'jump_return_push_frog_other':
            self.output_layers['Skill'] = nn.Linear(self.LastNNeurons, 5)
    
    def forward(self, x):
        # Input shape: (batch_size, timesteps, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attn1(x)
        
        x = F.relu(self.conv3(x))
        skip = x
        x = F.relu(self.conv4(x))
        x = self.attn2(x) + skip
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        skip = x
        x = self.attn3(x) + skip
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.attn4(x) + skip
        
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        
        x = self.flatten(x)
        features = F.softmax(self.features(x), dim=1)
        
        # Outputs
        outputs = {}
        for key, layer in self.output_layers.items():
            if key in ['Skill', 'Turner1', 'Turner2', 'Type']:  # Categorical outputs
                outputs[key] = F.softmax(layer(features), dim=1)
            else:  # Regression outputs
                outputs[key] = torch.sigmoid(layer(features))
        
        return outputs

def get_model(modelinfo, df_table_counts: pd.DataFrame):
    """Build a Self-Attention ConvLSTM model in PyTorch"""
    return SAConv3D(modelinfo, df_table_counts)