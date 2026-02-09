import torch
import torch.nn as nn
import torchvision.models as models
from types import SimpleNamespace
from models.JudgeVisionModel import JudgeVisionModel

class MViT(JudgeVisionModel):
    def __init__(self, head: nn.Module, recipe: SimpleNamespace):
        mvit = models.video.mvit_v1_b(weights="DEFAULT")
        mvit.head = nn.Identity()

        super(MViT, self).__init__(
            backbone=mvit,
            head=head,
        )
    
    @staticmethod
    def get_output_feature_dim(recipe: SimpleNamespace) -> int:
        mvit = models.video.mvit_v1_b(weights='DEFAULT')
        return JudgeVisionModel.get_output_feature_dim(
            recipe=recipe,
            backbone=mvit
        )

class MViT_Dense(JudgeVisionModel):
    def __init__(self, head: nn.Module, recipe: SimpleNamespace):
        backbone = models.video.mvit_v1_b(weights="DEFAULT")
        backbone.head = nn.Identity()

        in_dim = JudgeVisionModel.get_output_feature_dim(recipe, backbone)
        out_dim = in_dim * 3 // 4

        super().__init__(backbone=backbone, head=head)

        self.features = nn.Linear(in_dim, out_dim)

    def extract_features(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.features(x)
        return x

    @staticmethod
    def get_output_feature_dim(recipe: SimpleNamespace) -> int:
        backbone = models.video.mvit_v1_b(weights="DEFAULT")
        in_dim = JudgeVisionModel.get_output_feature_dim(recipe, backbone)
        return in_dim * 3 // 4

# Example template for custom classes + see below class
class MiniMViTBackbone(nn.Module):
    def __init__(self, embed_dim=96, depth=4, num_heads=4):
        super().__init__()

        # 1️⃣ Spatiotemporal patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(3, 7, 7),
            stride=(2, 4, 4),
            padding=(1, 3, 3)
        )

        # 2️⃣ Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 3️⃣ Feature projection
        self.proj = nn.Linear(embed_dim, embed_dim)

        # 4️⃣ Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, C, T, H, W)

        x = self.patch_embed(x)          # (B, D, T', H', W')
        B, D, T, H, W = x.shape

        x = x.view(B, D, T * H * W)      # flatten space-time
        x = x.transpose(1, 2)            # (B, tokens, D)

        x = self.blocks(x)               # transformer

        x = self.proj(x)

        x = x.transpose(1, 2)            # (B, D, tokens)
        x = self.pool(x)                 # (B, D, 1)
        x = x.squeeze(-1)                # (B, D)

        return x

class MiniMViT(JudgeVisionModel):
    def __init__(self, head: nn.Module):
        backbone = MiniMViTBackbone(
            embed_dim=128,
            depth=5,
            num_heads=4
        )

        super().__init__(backbone=backbone, head=head)

    def extract_features(self, x):
        return self.backbone(x)  # already a vector
