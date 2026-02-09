import torch
import torch.nn as nn
import torchvision.models as models
from types import SimpleNamespace

class JudgeVisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, freeze=True):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.flatten = nn.Flatten(start_dim=1) # Dim 0 is batch

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

    def extract_features(self, x):
        x = self.backbone(x)
        return self.flatten(x)

    def forward(self, x):
        features = self.extract_features(x)
        return self.head(features)

    @staticmethod
    def get_output_feature_dim(recipe: SimpleNamespace, backbone: nn.Module) -> int:
        input_shape = (3, recipe.timesteps, recipe.dim, recipe.dim)

        # remove classifier if exists
        if hasattr(backbone, "head"):
            backbone.head = nn.Identity()
        if hasattr(backbone, "fc"):
            backbone.fc = nn.Identity()

        backbone = backbone.eval()
        device = next(backbone.parameters()).device

        with torch.no_grad():
            x = torch.rand(1, *input_shape).to(device)
            out = backbone(x)
            out = torch.flatten(out, 1)
            return out.shape[1]
