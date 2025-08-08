import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import map_stageNr, mapped_stage_is_not_stageProperties
from constants import STAGES
from pprint import pprint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OutputHeadRecognition(nn.Module):
    """Template by ChatGPT 2025-08-07, altered to incorporate stages"""
    def __init__(self, input_neurons, df_layers, df_composition, max_instances_per_role):
        super().__init__()

        self.output_layers = nn.ModuleDict()
        self.df_layers = df_layers 
        self.df_composition = df_composition 
        self.max_instances_per_role = max_instances_per_role

        print(f"max instances per role")
        print(max_instances_per_role)

        # Precompute categorical mappings: {propertyId: {valueId: class_idx}}
        self.categorical_valueId_to_idx = {}
        self.categorical_idx_to_valueId = {}
        for prop_id, group_df in df_layers[df_layers['type'] == 'categorical'].groupby('propertyId'):
            self.categorical_valueId_to_idx[int(prop_id)] = {
                int(row['valueId']): idx + 1   # +1 so 0 is reserved for "absent"
                for idx, row in group_df.reset_index().iterrows()
            }
            self.categorical_idx_to_valueId[int(prop_id)] = {
                idx + 1: int(row['valueId'])   # +1 so 0 is reserved for "absent"
                for idx, row in group_df.reset_index().iterrows()
            }
            self.categorical_valueId_to_idx[prop_id] = { **self.categorical_valueId_to_idx[prop_id], 0:0 }
            self.categorical_idx_to_valueId[prop_id] = { **self.categorical_idx_to_valueId[prop_id], 0:0 }
        print(f"property maps")
        print(self.categorical_idx_to_valueId)
        print(self.categorical_valueId_to_idx)

        # TODO : update, use the same layer for each property id
        for index, row in df_composition.iterrows():
            composition_name = row['compositionName']
            mapped_stage = map_stageNr(row['stage'])
            property_row = df_layers[df_layers['propertyId'] == row['propertyId']].iloc[0]
            prop_name = row['name']
            prop_type = property_row['type']

            for i in range(max_instances_per_role[composition_name]):
                output_head = '_'.join([composition_name, str(i), mapped_stage, prop_name])
                if prop_type == "categorical":
                    num_classes = df_layers[df_layers['propertyId'] == row['propertyId']]['value'].nunique()
                    self.output_layers[output_head] = nn.Linear(input_neurons, num_classes + 1) # Account for 0 class
                elif prop_type == "boolean":
                    self.output_layers[output_head] = nn.Linear(input_neurons, 1)
                elif prop_type == "numerical":
                    self.output_layers[output_head] = nn.Linear(input_neurons, 1)  # You may normalize this value
                
    def forward(self, x):
        """
        x: [B, input_neurons] (global-pooled embedding from MViT)

        Returns a FLAT dict:
        { 'Role.idx.Stage.PropName': tensor([B, ...]), ... }
        """
        output = {}
        for output_head_name, output_head_layer in self.output_layers.items():
            output[output_head_name] = output_head_layer(x)
        return output

    def label_to_tensor(self, label_dict, flip_instances=False):
        """
        label_dict: {
            "Turner": [{ "GeneralProperties": {"Hands": 2, "Feet": 1, ...} }, {...}, ...],
            "Jumper": [...]
            ...
        }
        Return:
            target_tensor: same nested dict structure with torch tensors
            mask_tensor: same structure but indicating valid fields
        """
        target, mask = {}, {}

        for index, row in self.df_composition.iterrows():
            composition_name = row['compositionName']
            mapped_stage = map_stageNr(row['stage'])
            property_row = self.df_layers[self.df_layers['propertyId'] == row['propertyId']].iloc[0] # TODO : create dict: composition.prop_name -> type
            prop_name = row['name']
            prop_type = property_row['type']
            prop_id = row['propertyId']

            instance_indexes = range(self.max_instances_per_role[composition_name])
            instance_indexes = reversed(instance_indexes) if flip_instances else instance_indexes
            for i in instance_indexes:
                output_head = '_'.join([composition_name, str(i), mapped_stage, prop_name])

                is_stage = not mapped_stage_is_not_stageProperties(mapped_stage)
                requires_gradient = True if \
                    composition_name in label_dict.keys() \
                    and i < len(label_dict[composition_name]) \
                    and ( \
                            ( \
                                is_stage \
                                and mapped_stage in label_dict[composition_name][i]['StageProperties'].keys() \
                                and prop_name in label_dict[composition_name][i]['StageProperties'][mapped_stage].keys() \
                            )  or ( \
                                not is_stage \
                                and mapped_stage in label_dict[composition_name][i].keys() \
                                and prop_name in label_dict[composition_name][i][mapped_stage].keys() \
                            ) \
                        ) \
                    else False

                # TODO : fix value of categorical: guess now is: propValueId and not index of ... + 1
                if requires_gradient:
                    value = int(label_dict[composition_name][i]['StageProperties'][mapped_stage][prop_name] if is_stage else label_dict[composition_name][i][mapped_stage][prop_name])
                    if prop_type == 'categorical':
                        target[output_head] = torch.tensor(int(self.categorical_valueId_to_idx[prop_id][value]), device=device)
                    elif prop_type == 'boolean':
                        target[output_head] = torch.tensor(int(value), device=device)
                    elif prop_type == 'numerical':
                        target[output_head] = torch.tensor(float(value), dtype=torch.float32, device=device)
                    mask[output_head] = torch.tensor(True, device=device)
                else :
                    target[output_head] = torch.tensor(0.0, device=device) # Dummy
                    mask[output_head] = torch.tensor(True, device=device)

        return target, mask

    @staticmethod
    def compute_loss(preds, targets, masks):
        """
        preds: dict[str, Tensor] where each tensor is [B, C] or [B, 1]
        targets: dict[str, Tensor] where each tensor is [B, 1]
        masks: dict[str, Tensor] where each tensor is [B] boolean or 0/1 mask
        """
        total_loss = 0.0
        total_count = 0

        try:
            
            for key, pred_val in preds.items():
                target_val = targets[key]
                mask_val = masks[key]  # shape [B]

                # Filter only valid samples
                valid_idx = mask_val.bool().nonzero(as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue

                pred_valid = pred_val[valid_idx]
                target_valid = target_val[valid_idx]

                if pred_valid.ndim > 1 and pred_valid.shape[1] > 1:
                    # Categorical
                    loss = F.cross_entropy(pred_valid, target_valid.long())
                else:
                    # Boolean or numerical
                    loss = F.mse_loss(pred_valid.squeeze(), target_valid.float())

                total_loss += loss
                total_count += 1

            if total_count == 0:
                return torch.tensor(0.0, device=device)

        except Exception as e:
            print()
            print(f"compute loss pred", key, pred_valid)
            print(f"compute loss targ", key, target_valid)

            raise e
        return total_loss / total_count

