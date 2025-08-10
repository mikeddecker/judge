import torch
from torchmetrics import Metric

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

