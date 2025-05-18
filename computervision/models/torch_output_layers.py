import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from api.helpers.ConfigHelper import get_discipline_DoubleDutch_config


def create_pytorch_skill_output_layers(lastNNeurons, balancedType, df_table_counts):
    dd_config = get_discipline_DoubleDutch_config()
    output_layers = torch.nn.ModuleDict()
    
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
            
            classes = int(df_table_counts.iloc[0][columnname] + 1) # Because of MysqlIndex starts from 1
            output_layers[key] = torch.nn.Linear(lastNNeurons, classes)
        else:
            output_layers[key] = torch.nn.Linear(lastNNeurons, 1)
    
    if balancedType == 'jump_return_push_frog_other':
        output_layers['Skill'] = torch.nn.Linear(lastNNeurons, 5)
    
    return output_layers

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