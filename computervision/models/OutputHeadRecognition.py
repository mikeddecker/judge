import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import metrics.NumericStepMetrics as nsm

from helpers import map_stageNr, mapped_stage_is_not_stageProperties
from pprint import pprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OutputHeadRecognition(nn.Module):
    """Template by ChatGPT 2025-08-07, altered to incorporate stages"""
    def __init__(self, input_neurons: int, df_layers, df_composition, max_instances_per_role):
        super().__init__()

        self.output_layers = nn.ModuleDict()
        self.df_layers = df_layers 
        self.df_composition = df_composition 
        self.max_instances_per_role = max_instances_per_role

        print(f"max instances per role")
        print(max_instances_per_role)

        self.init_categorical_mappings()
        self.init_layers(input_neurons)
        self.init_metrics()

    def init_categorical_mappings(self):
        # Precompute categorical mappings: {propertyId: {valueId: class_idx}}
        self.categorical_valueId_to_idx = {}
        self.categorical_idx_to_valueId = {}
        for prop_id, group_df in self.df_layers[self.df_layers['type'] == 'categorical'].groupby('propertyId'):
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

    def init_layers(self, input_neurons: int):
        self.layers: dict[str, nn.Module] = {} # Key = prop_name, Value = Layer
        for index, row in self.df_composition.iterrows():
            composition_name = row['compositionName']
            mapped_stage = map_stageNr(row['stage'])
            property_row = self.df_layers[self.df_layers['propertyId'] == row['propertyId']].iloc[0]
            prop_name = row['name']
            prop_type = property_row['type']

            for i in range(self.max_instances_per_role[composition_name]):
                output_head = '_'.join([composition_name, str(i), mapped_stage, prop_name])
                if prop_name in self.layers.keys():
                    self.output_layers[output_head] = self.layers[prop_name]
                elif prop_type == "categorical":
                    num_classes = self.df_layers[self.df_layers['propertyId'] == row['propertyId']]['value'].nunique()
                    layer = nn.Linear(input_neurons, num_classes + 1) # Account for 0 class
                    self.layers[prop_name] = layer
                    self.output_layers[output_head] = layer
                elif prop_type == "boolean":
                    layer = nn.Linear(input_neurons, 1)
                    self.layers[prop_name] = layer
                    self.output_layers[output_head] = layer
                elif prop_type == "numerical":
                    layer = nn.Linear(input_neurons, 1)
                    self.layers[prop_name] = layer
                    self.output_layers[output_head] = layer

    def reset_metrics(self):
        for metric_type, prop_name_metric in self.metrics.items():
            for prop_name, metric in prop_name_metric.items():
                metric.reset()
    
    def init_metrics(self, average='macro'):
        self.metrics: dict[str, dict[str, torchmetrics.Metric]] = {
            'precision': {}, # prop_name: Metric
            'recall': {},
            'f1': {},
            'acc': {},
            'confusion': {},
        }
        self.reset_metrics()
        for index, row in self.df_composition.iterrows():
            composition_name = row['compositionName']
            mapped_stage = map_stageNr(row['stage'])
            property_row = self.df_layers[self.df_layers['propertyId'] == row['propertyId']].iloc[0]
            prop_name = row['name']
            prop_type = property_row['type']

            if self.max_instances_per_role[composition_name] > 0:
                if prop_name in self.metrics['acc'].keys():
                    continue
                elif prop_type == "categorical":
                    num_classes = self.df_layers[self.df_layers['propertyId'] == row['propertyId']]['value'].nunique()
                    num_classes += 1
                    self.metrics['precision'][prop_name] = torchmetrics.Precision(task="multiclass", average=average, num_classes=num_classes).to(device)
                    self.metrics['recall'][prop_name]    = torchmetrics.Recall(task="multiclass", average=average, num_classes=num_classes).to(device)
                    self.metrics['f1'][prop_name]        = torchmetrics.F1Score(task="multiclass", average=average, num_classes=num_classes).to(device)
                    self.metrics['acc'][prop_name]       = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
                    self.metrics['confusion'][prop_name] = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
                elif prop_type == "boolean":
                    self.metrics['precision'][prop_name] = torchmetrics.Precision(task="binary").to(device)
                    self.metrics['recall'][prop_name]    = torchmetrics.Recall(task="binary").to(device)
                    self.metrics['f1'][prop_name]        = torchmetrics.F1Score(task="binary").to(device)
                    self.metrics['acc'][prop_name]       = torchmetrics.Accuracy(task="binary").to(device)
                    self.metrics['confusion'][prop_name] = torchmetrics.ConfusionMatrix(task="binary").to(device)
                elif prop_type == "numerical":
                    step = float(property_row['step'])
                    step = 0.1 if step < 0.1 else step
                    min = float(property_row['min'])
                    max = float(property_row['max'])
                    self.metrics['precision'][prop_name] = nsm.NumericStepPrecision(step=step).to(device)
                    self.metrics['recall'][prop_name]    = nsm.NumericStepRecall(step=step).to(device)
                    self.metrics['f1'][prop_name]        = nsm.NumericStepF1Score(step=step).to(device)
                    self.metrics['acc'][prop_name]       = nsm.NumericStepAccuracy(step=step).to(device)
                    self.metrics['confusion'][prop_name] = nsm.NumericStepConfusionMatrix(step=step, min=min, max=max).to(device)

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

    def compute_loss(self, preds, targets, masks):
        """
        preds: dict[str, Tensor] where each tensor is [B, C] or [B, 1]
        targets: dict[str, Tensor] where each tensor is [B, 1]
        masks: dict[str, Tensor] where each tensor is [B] boolean or 0/1 mask
        """
        total_loss = 0.0
        total_count = 0

        try:
            for output_head, pred_val in preds.items():
                target_val = targets[output_head]
                mask_val = masks[output_head]  # shape [B]

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
                    loss = F.mse_loss(pred_valid.squeeze(dim=1), target_valid.float())

                total_loss += loss
                total_count += 1

            if total_count == 0:
                return torch.tensor(0.0, device=device)

        except Exception as e:
            print()
            print(f"compute exception loss pred", output_head, pred_valid)
            print(f"compute exception loss targ", output_head, target_valid)

            raise e
        return total_loss / total_count

    def update_metrics(self, preds: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], masks):
        """Return current f1_average"""
        for output_head, pred_val in preds.items():
            target_val = targets[output_head]
            mask_val = masks[output_head]
            output_head_splits = output_head.split('_')

            if len(output_head_splits) == 1:
                # compositionName: count
                raise NotImplementedError(f"Counts prediction not yet implemented")
            elif len(output_head_splits) != 4:
                raise ValueError(f"Something went wrong with output_head:", output_head)
            composition_name = output_head_splits[0]
            prop_name = output_head_splits[3]

            valid_idx = mask_val.bool().nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue

            pred_valid = pred_val[valid_idx].squeeze(dim=1)
            target_valid = target_val[valid_idx]
            # target_valid = target_valid if target_valid.ndim == 1 else target_valid.squeeze(dim=1)

            self.metrics['precision'][prop_name].update(pred_valid, target_valid)
            self.metrics['recall'][prop_name].update(pred_valid, target_valid)
            self.metrics['f1'][prop_name].update(pred_valid, target_valid)
            self.metrics['acc'][prop_name].update(pred_valid, target_valid)
            self.metrics['confusion'][prop_name].update(pred_valid, target_valid)

        return np.mean([self.metrics['f1'][pn].compute().item() for pn in self.metrics['f1'].keys()])
    
    def compute_metrics(self):
        """
        Computes precision, recall, f1, and accuracy for each output_head in preds using torchmetrics.
        """
        return {
            type: {
                propname: metric.compute().item() if metric.compute().ndim == 0 else metric.compute().tolist()
                for propname, metric in propname_metric.items()
            } for type, propname_metric in self.metrics.items()
        }

