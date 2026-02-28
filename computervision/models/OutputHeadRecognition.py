import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
import metrics.NumericStepMetrics as nsm

from pprint import pprint

from constants import PYTORCH_MODELS_SKILLS
from collections import defaultdict
from helpers import map_stageNr, mapped_stage_is_not_stageProperties, weighted_mse_loss
from helpers import confusion_accuracy, get_numeric_metric_average, get_confusion_average
from managers.RepoGeneral import REPO_GENERAL
from types import SimpleNamespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OutputHeadRecognition(nn.Module):
    """The output head for recognizing skills"""
    def __init__(self, recipe: SimpleNamespace):
        super().__init__()

        self.confusion_heads : dict[list[str]] = {}
        self.df_layers : pd.DataFrame = REPO_GENERAL.get_skill_layers()
        self.df_composition : pd.DataFrame = REPO_GENERAL.get_skill_compositions()
        self.max_instances_per_composition : pd.Series = REPO_GENERAL.get_max_instances_per_role(self.df_composition)
        self.output_layers = nn.ModuleDict()

        backbone_output_neurons = PYTORCH_MODELS_SKILLS[recipe.model].get_output_feature_dim(recipe)
        prop_counts = REPO_GENERAL.get_skill_prop_counts()

        print(f"max instances per role")
        print(self.max_instances_per_composition)

        self.init_categorical_mappings()
        self.init_layers(backbone_output_neurons, prop_counts)
        self.init_metrics()

    def init_categorical_mappings(self):
        # Precompute categorical mappings: {layerId: {valueId: class_idx}}
        self.categorical_valueId_to_idx = {}
        self.categorical_idx_to_valueId = {}
        for prop_id, group_df in self.df_layers[self.df_layers['type'] == 'categorical'].groupby('layerId'):
            self.categorical_valueId_to_idx[prop_id] = {
                int(row['valueId']): idx + 1   # +1 so 0 is reserved for "absent"
                for idx, row in group_df.reset_index().iterrows()
            }
            self.categorical_idx_to_valueId[prop_id] = {
                idx + 1: int(row['valueId'])   # +1 so 0 is reserved for "absent"
                for idx, row in group_df.reset_index().iterrows()
            }
            self.categorical_valueId_to_idx[prop_id] = { **self.categorical_valueId_to_idx[prop_id], 0:0 }
            self.categorical_idx_to_valueId[prop_id] = { **self.categorical_idx_to_valueId[prop_id], 0:0 }
        print(f"init_categorical_mappings maps")
        print(self.categorical_idx_to_valueId)
        print(self.categorical_valueId_to_idx)

    def init_layers(self, input_neurons: int, prop_counts: defaultdict[defaultdict[int]]):
        self.layers: dict[str, nn.Module] = {} # Key = layerName, Value = Layer
        self.loss_fns: dict[str, callable] = {}
        weight_alpha = 1.5

        # Add layers for composition counts e.g.:
        # DDSwitch      1
        # Jumper        2
        # SingleRope    4
        # Turner        4
        # 1, 2, 4, 4 neurons respectively
        for composition, max_instances in self.max_instances_per_composition.items():
            output_head = f"composition_{composition}"
            self.output_layers[output_head] = nn.Linear(input_neurons, max_instances + 1)
            self.loss_fns[output_head] = nn.CrossEntropyLoss().to(device)

        print("self.df_composition - init layers")
        print(self.df_composition)

        # Add output layer heads for each layer e.g.:
        # DDSwitch_0_StageProperties_Hands
        # Jumper_1_StageProperties_Feet
        # SingleRope_0_StageProperties_Backwards
        for index, row in self.df_composition.iterrows():
            composition_name = row['compositionName']
            mapped_stage = map_stageNr(row['stage'])
            layer_row = self.df_layers[self.df_layers['layerId'] == row['layerId']].iloc[0]
            layerName = row['name']
            prop_type = layer_row['type']

            for i in range(self.max_instances_per_composition[composition_name]):
                output_head = '_'.join([composition_name, str(i), mapped_stage, layerName])
                if layerName in self.layers.keys():
                    self.output_layers[output_head] = self.layers[layerName]
                elif prop_type == "categorical":
                    num_classes = self.df_layers[self.df_layers['layerId'] == row['layerId']]['value'].nunique()
                    layer = nn.Linear(input_neurons, num_classes + 1) # Account for 0 class
                    self.layers[layerName] = layer
                    self.output_layers[output_head] = layer

                    counts = torch.Tensor([prop_counts[layerName].get(self.categorical_idx_to_valueId[row['layerId']][k], 0) for k in range(num_classes+1)]).to(device)
                    max_count = counts.max()
                    counts = counts ** weight_alpha
                    counts[counts == 0] = max_count
                    weights = (max_count ** weight_alpha) / counts
                    weights = weights / weights.mean()
                    print(layerName, weights)
                    self.loss_fns[layerName] = torch.nn.CrossEntropyLoss(weights).to(device)
                elif prop_type == "boolean":
                    layer = nn.Linear(input_neurons, 1)
                    self.layers[layerName] = layer
                    self.output_layers[output_head] = layer

                    counts = torch.Tensor([prop_counts[layerName].get(k, 0) for k in [False, True]]).to(device)
                    max_count = counts.max()
                    counts = counts ** weight_alpha
                    counts[counts == 0] = max_count
                    weights = (max_count ** weight_alpha) / counts
                    weights = weights / weights.mean()
                    self.loss_fns[layerName] = lambda input, target: weighted_mse_loss(input=input, target=target, weight=weights)
                elif prop_type == "numerical":
                    layer = nn.Linear(input_neurons, 1)
                    self.layers[layerName] = layer
                    self.output_layers[output_head] = layer

                    step = layer_row['step'] if layer_row['step'] > 0.1 else 0.1
                    counts = torch.Tensor([prop_counts[layerName].get(k * step, 0) for k in range(round(layer_row['min'] / step), round(layer_row['max'] / step) + 1)]).to(device)
                    max_count = counts.max()
                    counts = counts ** weight_alpha
                    counts[counts == 0] = max_count
                    weights = (max_count ** weight_alpha) / counts
                    weights = weights / weights.mean()
                    print(layerName, weights)
                    self.loss_fns[layerName] = lambda input, target: weighted_mse_loss(input=input, target=target, weight=weights, step=step)

    def reset_metrics(self):
        for metric_type, layerName_metric in self.metrics.items():
            for layerName, metric in layerName_metric.items():
                metric.reset()

    def init_metrics(self, average='macro'):
        self.metrics: dict[str, dict[str, torchmetrics.Metric]] = {
            'precision': {}, # layerName: Metric
            'recall': {},
            'f1': {},
            'acc': {},
            'confusion': {},
        }
        self.reset_metrics()
        for composition_name, max_amount in self.max_instances_per_composition.items():
            # Composition name = DDSwitch, Jumper, SingleRope, Turner... # TODO : enforce unique!
            num_classes = max_amount + 1
            self.metrics['precision'][composition_name] = torchmetrics.Precision(task="multiclass", average=average, num_classes=num_classes).to(device)
            self.metrics['recall'][composition_name]    = torchmetrics.Recall(task="multiclass", average=average, num_classes=num_classes).to(device)
            self.metrics['f1'][composition_name]        = torchmetrics.F1Score(task="multiclass", average=average, num_classes=num_classes).to(device)
            self.metrics['acc'][composition_name]       = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
            self.metrics['confusion'][composition_name] = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
            self.confusion_heads[composition_name] = list(range(num_classes))

        for index, row in self.df_composition.iterrows():
            composition_name = row['compositionName']
            mapped_stage = map_stageNr(row['stage'])
            layer_row = self.df_layers[self.df_layers['layerId'] == row['layerId']].iloc[0]
            layerName = row['name']
            prop_type = layer_row['type']

            if self.max_instances_per_composition[composition_name] > 0:
                if layerName in self.metrics['acc'].keys():
                    continue
                elif prop_type == "categorical":
                    categorical_values : pd.DataFrame = self.df_layers[self.df_layers['layerId'] == row['layerId']]['value']
                    num_classes = categorical_values.nunique()
                    num_classes += 1
                    self.metrics['precision'][layerName] = torchmetrics.Precision(task="multiclass", average=average, num_classes=num_classes).to(device)
                    self.metrics['recall'][layerName]    = torchmetrics.Recall(task="multiclass", average=average, num_classes=num_classes).to(device)
                    self.metrics['f1'][layerName]        = torchmetrics.F1Score(task="multiclass", average=average, num_classes=num_classes).to(device)
                    self.metrics['acc'][layerName]       = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
                    self.metrics['confusion'][layerName] = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
                    self.confusion_heads[layerName] = categorical_values.values.tolist()
                    self.confusion_heads[layerName].insert(0, None)
                elif prop_type == "boolean":
                    self.metrics['precision'][layerName] = torchmetrics.Precision(task="binary").to(device)
                    self.metrics['recall'][layerName]    = torchmetrics.Recall(task="binary").to(device)
                    self.metrics['f1'][layerName]        = torchmetrics.F1Score(task="binary").to(device)
                    self.metrics['acc'][layerName]       = torchmetrics.Accuracy(task="binary").to(device)
                    self.metrics['confusion'][layerName] = torchmetrics.ConfusionMatrix(task="binary").to(device)
                    self.confusion_heads[layerName] = [False, True]
                elif prop_type == "numerical":
                    step = float(layer_row['step'])
                    step = 0.1 if step < 0.1 else step
                    min = float(layer_row['min'])
                    max = float(layer_row['max'])
                    epsilon = step / 8 # Float round errors otherwise.
                    numerical_values = [min + step * i for i in range(int(1 + (max - min + epsilon) // step))]
                    numerical_values = [round(x, 2) for x in numerical_values]
                    self.confusion_heads[layerName] = numerical_values
                    self.metrics['precision'][layerName] = nsm.NumericStepPrecision(layerName=layerName, step=step).to(device)
                    self.metrics['recall'][layerName]    = nsm.NumericStepRecall(layerName=layerName, step=step).to(device)
                    self.metrics['f1'][layerName]        = nsm.NumericStepF1Score(layerName=layerName, step=step).to(device)
                    self.metrics['acc'][layerName]       = nsm.NumericStepAccuracy(layerName=layerName, step=step).to(device)
                    self.metrics['confusion'][layerName] = nsm.NumericStepConfusionMatrix(layerName=layerName, step=step, min=min, max=max).to(device)
                print(layerName, self.confusion_heads[layerName])

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

        try:

            for index, row in self.df_composition.iterrows():
                composition_name = row['compositionName']
                composition_head = f"composition_{composition_name}"
                target[composition_head] = torch.tensor(len(label_dict[composition_name]) if composition_name in label_dict.keys() else 0, device=device)
                mask[composition_head] = torch.tensor(True, device=device)
                mapped_stage = map_stageNr(row['stage'])
                layer_row = self.df_layers[self.df_layers['layerId'] == row['layerId']].iloc[0] # TODO : create dict: composition.layerName -> type
                layerName = row['name']
                prop_type = layer_row['type']
                prop_id = row['layerId']

                instance_indexes = range(self.max_instances_per_composition[composition_name])
                instance_indexes = reversed(instance_indexes) if flip_instances else instance_indexes
                for i in instance_indexes:
                    output_head = '_'.join([composition_name, str(i), mapped_stage, layerName])

                    is_stage = not mapped_stage_is_not_stageProperties(mapped_stage)
                    requires_gradient = True if \
                        composition_name in label_dict.keys() \
                        and i < len(label_dict[composition_name]) \
                        and ( \
                                ( \
                                    is_stage \
                                    and 'StageProperties' in label_dict[composition_name][i] \
                                    and mapped_stage in label_dict[composition_name][i]['StageProperties'].keys() \
                                    and layerName in label_dict[composition_name][i]['StageProperties'][mapped_stage].keys() \
                                )  or ( \
                                    not is_stage \
                                    and mapped_stage in label_dict[composition_name][i].keys() \
                                    and layerName in label_dict[composition_name][i][mapped_stage].keys() \
                                ) \
                            ) \
                        else False

                    # TODO : fix value of categorical: guess now is: propValueId and not index of ... + 1
                    if requires_gradient:
                        value = label_dict[composition_name][i]['StageProperties'][mapped_stage][layerName] if is_stage else label_dict[composition_name][i][mapped_stage][layerName]
                        try:
                            if prop_type == 'categorical':
                                target[output_head] = torch.tensor(int(self.categorical_valueId_to_idx[prop_id][int(value)]), device=device)
                            elif prop_type == 'boolean':
                                target[output_head] = torch.tensor(int(value), device=device)
                            elif prop_type == 'numerical':
                                target[output_head] = torch.tensor(float(value), dtype=torch.float32, device=device)
                            mask[output_head] = torch.tensor(True, device=device)
                        except:
                            print(f"Error for outputhead: {output_head}, prop_id: {prop_id}, prop_type: {prop_type}, layerName: {layerName}, value: {value}")
                            if prop_type == 'categorical':
                                print(f"{self.categorical_valueId_to_idx[prop_id]}")
                            raise
                    else :
                        target[output_head] = torch.tensor(0.0, device=device) # Dummy
                        mask[output_head] = torch.tensor(False, device=device)
        except:
            print("Label to error", "labeldict")
            print(label_dict)
            if 'output_head' in locals():
                print('Output head:', output_head)
            raise
        return target, mask

    def compute_loss(self, preds, targets, masks, skillId=None):
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

                if "composition_" in output_head:
                    # Composition count prediction
                    target_val = targets[output_head].long()  # shape [B]
                    loss = self.loss_fns[output_head](pred_val.squeeze(dim=1), target_val)
                else:
                    layerName = output_head.split('_')[-1]
                    if pred_valid.ndim > 1 and pred_valid.shape[1] > 1:
                        # Categorical
                        loss = self.loss_fns[layerName](pred_valid, target_valid.long())
                    else:
                        # Boolean or numerical
                        loss = self.loss_fns[layerName](pred_valid.squeeze(dim=1), target_valid.float())

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

            if len(output_head_splits) != 4 and len(output_head_splits) != 2:
                raise ValueError(f"Something went wrong with output_head:", output_head)

            composition_name = output_head_splits[0]
            layerName = output_head_splits[1] if "composition_" in output_head else output_head_splits[3]

            valid_idx = mask_val.bool().nonzero(as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue

            pred_valid = pred_val[valid_idx].squeeze(dim=1)
            target_valid = target_val[valid_idx]
            # target_valid = target_valid if target_valid.ndim == 1 else target_valid.squeeze(dim=1)

            self.metrics['precision'][layerName].update(pred_valid, target_valid)
            self.metrics['recall'][layerName].update(pred_valid, target_valid)
            self.metrics['f1'][layerName].update(pred_valid, target_valid)
            self.metrics['acc'][layerName].update(pred_valid, target_valid)
            self.metrics['confusion'][layerName].update(pred_valid, target_valid)

    def compute_metrics(self):
        """
        Computes precision, recall, f1, confusion and accuracy for each output_head in preds using torchmetrics.

        Returns: {
            "metric_per_layer": {'acc': {'prop1': list[float]|float, ...}, 'f1': {...}'},
            "metric_avg_of_layers": {'acc': float, 'f1': float, ...},
            "confusion_heads": self.confusion_heads,
        }

        """
        # Metric types: acc, f1, precision...
        prop_metrics = {
            metric_type: {
                propname: (
                    metric.compute().item()
                    if metric.compute().ndim == 0
                    else metric.compute().tolist()
                ) for propname, metric in propname_metric.items()
            } for metric_type, propname_metric in self.metrics.items()
        }

        avg_metrics = {}
        total_average = defaultdict(lambda: [])

        for metric_type, values in prop_metrics.items():
            avg = get_confusion_average(values) if metric_type == "confusion" else get_numeric_metric_average(values)
            avg_metrics[metric_type] = avg

        pprint('Average metrics')
        # pprint(prop_metrics['f1'])
        pprint(avg_metrics)

        return {
            'metric_per_layer': prop_metrics,
            'metric_avg_of_layers': avg_metrics,
            'confusion_heads': self.confusion_heads,
        }

