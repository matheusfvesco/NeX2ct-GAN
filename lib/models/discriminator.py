import torch
import torch.nn as nn
from lib.models.basic_blocks import ClassifierBlock
from lib.models.convnext import Convnext3dBlock


class Discriminator(nn.Module):
    def __init__(self, features: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder = []

        dim = 256
        for in_channels, out_channels in zip(features, features[1:]):
            # TODO: check stride, 2 doesn't run due to memory, while 3 and 4 are feasable
            # should use ks = stride
            encoder.append(
                ClassifierBlock(in_channels, out_channels, 2, 2, Convnext3dBlock)
            )
            dim = (dim - 2) // 2 + 1  # For kernel=2, stride=2, padding=0
        
        out_dim = (dim**3) * features[-1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_dim, 2)
        )
        self.encoder = nn.Sequential(*tuple(encoder))

    def forward(self, x):
        x: torch.Tensor = self.encoder(x)
        x = self.head(x)
        return x