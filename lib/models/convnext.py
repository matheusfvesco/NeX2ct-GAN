import torch.nn as nn
from .basic_blocks import create_depthwise, create_depthwise_3d

# depthwise 7x7 conv (in_channel) > layer norm > 1x1 (4*in_channel) > GELU > 1x1 (in_channel)
class ConvnextBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_dw = create_depthwise(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3) # pad preserves spatial dimension
        self.norm = nn.InstanceNorm2d(in_channels) # would be layer norm, but instance norm is better for image gen
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4,kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1)
        self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        sequential = nn.Sequential(
            self.conv_dw,
            self.norm,
            self.conv1,
            self.act,
            self.conv2
        )
        x = sequential(x)
        x = x + shortcut
        return x

# probably not use due to computational cost
class Convnext3dBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_dw = create_depthwise_3d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3) # pad preserves spatial dimension
        self.norm = nn.InstanceNorm3d(in_channels) # would be layer norm, but instance norm is better for image gen
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels*4,kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1)
        self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        sequential = nn.Sequential(
            self.conv_dw,
            self.norm,
            self.conv1,
            self.act,
            self.conv2
        )
        x = sequential(x)
        x = x + shortcut
        return x
