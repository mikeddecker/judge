import torch
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
            
            classes = int(df_table_counts.iloc[0][columnname])
            output_layers[key] = torch.nn.Linear(lastNNeurons, classes)
        else:
            output_layers[key] = torch.nn.Linear(lastNNeurons, 1)
    
    if balancedType == 'jump_return_push_frog_other':
        output_layers['Skill'] = torch.nn.Linear(lastNNeurons, 5)
    
    return output_layers