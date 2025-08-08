import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torchvision.models as models
import numpy as np
import pandas as pd
from models.torch_output_layers import create_pytorch_segmentation_output_layers, forward_skill_output_layers, forward_segmentation_output_layers
from types import SimpleNamespace

class MViT(nn.Module):
    def __init__(self, head: nn.Module, recipe: SimpleNamespace):
        super(MViT, self).__init__()
        
        self.mvit = models.video.mvit_v1_b(weights='DEFAULT')
        self.mvit = self.mvit.to('cuda').eval()

        for param in self.mvit.parameters():
            param.requires_grad = False

        self.mvit.head = torch.nn.Identity()  # This removes the top layer
        self.flatten = nn.Flatten()
        self.head = head
    
    def forward(self, x):
        # Input shape: (batch_size, channels, timesteps, height, width)
        x = self.mvit(x)
        x = self.flatten(x)
        return self.head(x)
    
    @staticmethod
    def get_output_feature_dim(recipe: SimpleNamespace) -> int:
        input_shape = (3, recipe.timesteps, recipe.dim, recipe.dim)
        mvit = models.video.mvit_v1_b(weights=recipe.default_weights).to('cuda').eval()
        mvit.head = torch.nn.Identity()

        with torch.no_grad():
            input = torch.rand(1, *input_shape).to('cuda')
            output = mvit(input)
            output = output.flatten(start_dim=1)
            return output.shape[1]

def get_model(head: nn.Module, recipe: SimpleNamespace):
    """Build an MViT model in PyTorch"""
    return MViT(head=head, recipe=recipe)

