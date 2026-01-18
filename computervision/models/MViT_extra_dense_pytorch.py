import torch
import torch.nn as nn
import torchvision.models as models
from models.MViT_pytorch import MViT
from types import SimpleNamespace

class MViT_Dense(nn.Module):
    def __init__(self, head: nn.Module, recipe: SimpleNamespace):
        super(MViT_Dense, self).__init__()
        
        self.mvit = models.video.mvit_v1_b(weights='DEFAULT')
        self.mvit = self.mvit.to('cuda').eval()

        for param in self.mvit.parameters():
            param.requires_grad = False

        self.mvit.head = torch.nn.Identity()  # This removes the top layer
        self.flatten = nn.Flatten()
        self.LastNNeuronsMViT = MViT.get_output_feature_dim(recipe)
        self.LastNNeurons = self.LastNNeuronsMViT * 3 // 4
        self.features = nn.Linear(self.LastNNeuronsMViT, self.LastNNeurons)

        self.head = head
    
    def forward(self, x):
        # Input shape: (batch_size, channels, timesteps, height, width)
        x = self.mvit(x)
        x = self.flatten(x)
        x = self.features(x)
        return self.head(x)
    
    @staticmethod
    def get_output_feature_dim(recipe: SimpleNamespace) -> int:
        input_shape = (3, recipe.timesteps, recipe.dim, recipe.dim)
        weights = "DEFAULT" if not hasattr(recipe, "use_existing_weights") else recipe.use_existing_weights
        mvit = models.video.mvit_v1_b(weights=weights).to('cuda').eval()
        mvit.head = torch.nn.Identity()

        with torch.no_grad():
            input = torch.rand(1, *input_shape).to('cuda')
            output = mvit(input)
            output = output.flatten(start_dim=1)
            return output.shape[1] * 3 // 4
