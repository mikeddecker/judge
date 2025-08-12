import torch
from torchmetrics import Metric, ConfusionMatrix

class NumericStepAccuracy(Metric):
    """
    Computes accuracy for numerical predictions after rounding to the nearest multiple of `step`.
    Works with continuous predictions in regression-like tasks.
    """

    def __init__(self, step: float = 1.0, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.step = step

        # Metric state: accumulates counts across batches
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: Tensor [B] or [B, 1] - continuous predictions
        targets: Tensor [B] or [B, 1] - continuous targets
        """
        preds = preds if preds.ndim == 1 else preds.squeeze(dim=1)
        targets = targets
        
        pred_rounded = torch.round(preds / self.step) * self.step
        target_rounded = torch.round(targets / self.step) * self.step

        correct = (pred_rounded == target_rounded).sum()
        total = target_rounded.numel()

        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)

class NumericStepPrecision(Metric):
    def __init__(self, step: float = 1.0, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.step = step
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds if preds.ndim == 1 else preds.squeeze(dim=1)
        targets = targets if preds.ndim == 1 else targets.squeeze(dim=1)
        
        pred_rounded = torch.round(preds / self.step) * self.step
        target_rounded = torch.round(targets / self.step) * self.step

        matches = pred_rounded == target_rounded
        self.tp += matches.sum()
        self.fp += (~matches).sum()

    def compute(self):
        denom = self.tp + self.fp
        return self.tp.float() / denom if denom > 0 else torch.tensor(0.0)

class NumericStepRecall(Metric):
    def __init__(self, step: float = 1.0, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.step = step
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds if preds.ndim == 1 else preds.squeeze(dim=1)
        targets = targets if preds.ndim == 1 else targets.squeeze(dim=1)
        pred_rounded = torch.round(preds / self.step) * self.step
        target_rounded = torch.round(targets / self.step) * self.step

        matches = pred_rounded == target_rounded
        self.tp += matches.sum()
        self.fn += (~matches).sum()

    def compute(self):
        denom = self.tp + self.fn
        return self.tp.float() / denom if denom > 0 else torch.tensor(0.0)

class NumericStepF1Score(Metric):
    def __init__(self, step: float = 1.0, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.step = step
        self.add_state("tp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds if preds.ndim == 1 else preds.squeeze(dim=1)
        targets = targets if preds.ndim == 1 else targets.squeeze(dim=1)

        pred_rounded = torch.round(preds / self.step) * self.step
        target_rounded = torch.round(targets / self.step) * self.step

        matches = pred_rounded == target_rounded
        self.tp += matches.sum()
        self.fp += (~matches).sum()
        self.fn += (~matches).sum()

    def compute(self):
        precision = self.tp.float() / (self.tp + self.fp) if (self.tp + self.fp) > 0 else torch.tensor(0.0)
        recall = self.tp.float() / (self.tp + self.fn) if (self.tp + self.fn) > 0 else torch.tensor(0.0)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)

class NumericStepConfusionMatrix(Metric):
    def __init__(self, step: float = 1.0, min: float = 0.0, max: float = 2.0, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.step = step
        self.num_classes = round((max - min) / step) + 1
        self.confusion_matrix = ConfusionMatrix('multiclass', num_classes=self.num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds if preds.ndim == 1 else preds.squeeze(dim=1)
        targets = targets if preds.ndim == 1 else targets.squeeze(dim=1)

        pred_rounded = torch.round(preds / self.step) * self.step
        target_rounded = torch.round(targets / self.step) * self.step

        self.confusion_matrix.update(pred_rounded, target_rounded)

    def compute(self):
        return self.confusion_matrix.compute()
    
    def to(self, *args, **kwargs):
        self.confusion_matrix.to(*args, **kwargs)
        super().to(*args, **kwargs)
        return self

class NumericStepWeightedMSE(Metric):
    def __init__(self, weights: torch.Tensor, step: float = 1.0, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.step = step
        self.weights = weights

        self.register_buffer("weights", weights.float())

        # Metric state: accumulates loss across batches
        self.add_state("sum_loss", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds if preds.ndim == 1 else preds.squeeze(dim=1)
        targets = targets if preds.ndim == 1 else targets.squeeze(dim=1)

        pred_rounded = torch.round(preds / self.step) * self.step
        target_rounded = torch.round(targets / self.step) * self.step

        weights = self.weights[target_rounded.long()]
        loss_per_sample = weights * (pred_rounded - target_rounded) ** 2
        self.sum_loss += loss_per_sample.sum()
        self.total += target_rounded.numel()

    def compute(self):
        return self.sum_loss / self.total

