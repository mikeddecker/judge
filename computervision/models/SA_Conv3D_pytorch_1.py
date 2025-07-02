import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import numpy as np
import pandas as pd
from models.torch_output_layers import create_pytorch_skill_output_layers, create_pytorch_segmentation_output_layers, forward_skill_output_layers, forward_segmentation_output_layers

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
        # Input shape: (batch_size, channels, timesteps, height, width)
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
    def __init__(self, skill_or_segment:str, modelinfo:dict, df_table_counts:pd.DataFrame):
        super(SAConv3D, self).__init__()
        self.modelinfo = modelinfo
        self.df_table_counts = df_table_counts
        self.isSkillModel = skill_or_segment == "skills"
        
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
        
        self.flatten = nn.Flatten()
        self.LastNNeurons = self._get_conv_output(input_shape)

        if self.isSkillModel:
            self.output_layers = create_pytorch_skill_output_layers(lastNNeurons=self.LastNNeurons, balancedType=modelinfo['balancedType'], df_table_counts = self.df_table_counts)
        else:
            self.output_layer = create_pytorch_segmentation_output_layers(lastNNeurons=self.LastNNeurons, timesteps=modelinfo['timesteps'])
      
    def _get_conv_output(self, shape):
        # Input shape: (batch_size, channels, timesteps, height, width)
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
        
        if self.isSkillModel:
            return forward_skill_output_layers(features=x, output_layers=self.output_layers)
        else:
            return forward_segmentation_output_layers(features=x, output_layer=self.output_layer)

def get_model(skill_or_segment:str, modelinfo, df_table_counts: pd.DataFrame):
    """Build a Self-Attention ConvLSTM model in PyTorch"""
    return SAConv3D(skill_or_segment=skill_or_segment, modelinfo=modelinfo, df_table_counts=df_table_counts)