import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T

class Normalize(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        return x

class Resize(nn.Module):
    def __init__(self, size: tuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(size, tuple):
            raise ValueError(f"Size must be a tuple, but found {type(size)}")
        if len(size) < 2 or len(size) > 3:
            raise ValueError(f"Size must be a tuple with 2 or 3 values, but found {len(size)} values")
        self.interpolation = "bicubic"
        if len(size) == 3:
            self.interpolation = "trilinear"
        self.size = size
    
    def forward(self, x: torch.Tensor):
        x = F.interpolate(x,self.size,mode=self.interpolation)
        return x