import torch
import torch.nn.functional as F
import sys

def create_pytorch_segmentation_output_layers(lastNNeurons:int, timesteps:int):
    return torch.nn.Linear(lastNNeurons, timesteps)

def forward_skill_output_layers(features: torch.tensor, output_layers: dict[str, torch.nn.Module]):
    outputs = {}
    for key, layer in output_layers.items():
        if key in ['Skill', 'Turner1', 'Turner2', 'Type']:
            outputs[key] = layer(features)
        else:  # Regression outputs
            outputs[key] = torch.sigmoid(layer(features))
    
    return outputs

def forward_segmentation_output_layers(features: torch.tensor, output_layer: torch.nn.Module):
    return output_layer(features)