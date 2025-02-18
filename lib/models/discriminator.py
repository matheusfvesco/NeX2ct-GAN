import torch
import torch.nn as nn
from lib.models.basic_blocks import ClassifierBlock
from lib.models.convnext import Convnext3dBlock


class Discriminator(nn.Module):
    def __init__(self, features: list[int], device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder = []

        for in_channels, out_channels in zip(features, features[1:]):
            # TODO: check stride, 2 doesn't run due to memory, while 3 and 4 are feasable
            # should use ks = stride
            encoder.append(
                ClassifierBlock(in_channels, out_channels, 2, 2, Convnext3dBlock).to(device)
            )
        
        self.encoder = nn.Sequential(*tuple(encoder)).to(device)
        self.device = device

    def forward(self, x):
        x: torch.Tensor = self.encoder(x)
        x = x.flatten(1)
        #assert x.shape is None, f"{x.shape}"
        _, features = x.size() 
        head = nn.Sequential(
            #nn.Linear(features, features),
            nn.Linear(features, 2)
        ).to(self.device)
        x = head(x)
        return x