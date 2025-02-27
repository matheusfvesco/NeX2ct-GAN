import torch
import torch.nn as nn

class GanModel(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
    
    def forward(self, x1, x2):
        x = self.generator(x1, x2)
        x = self.discriminator(x)
        return x