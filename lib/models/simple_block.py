import torch.nn as nn

# depthwise 7x7 conv (in_channel) > layer norm > 1x1 (4*in_channel) > GELU > 1x1 (in_channel)
class SimpleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(in_channels) # would be layer norm, but instance norm is better for image gen
        self.act = nn.ReLU()
        self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        sequential = nn.Sequential(
            self.conv1,
            self.norm,
            self.act,
        )
        x = sequential(x)
        x = x + shortcut
        return x

# probably not use due to computational cost
class Simple3dBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels,kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(in_channels) # would be layer norm, but instance norm is better for image gen
        self.act = nn.ReLU()
        self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        sequential = nn.Sequential(
            self.conv1,
            self.norm,
            self.act,
        )
        x = sequential(x)
        x = x + shortcut
        return x
