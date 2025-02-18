import torch.nn as nn
import torch


def create_depthwise(
    in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1
):
    groups = in_channels
    if out_channels % in_channels != 0:
        raise ValueError("out_channels needs to be a multiple of in_channels")
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def create_depthwise_3d(
    in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1
):
    groups = in_channels
    if out_channels % in_channels != 0:
        raise ValueError("out_channels needs to be a multiple of in_channels")
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


class Downsample(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1
    ):
        super().__init__()
        self.pool = nn.Identity()
        self.conv = nn.Identity()

        # spatial downsample
        if kernel_size > 1 or stride > 1:
            self.pool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        # channel downsample
        if in_channels != out_channels:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


# skip connection 2d->3d
class Skip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.norm2d = nn.InstanceNorm2d(out_channels)
        self.conv3d = nn.Conv3d(out_channels, out_channels, 1, 1)
        self.norm3d = nn.InstanceNorm3d(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv2d(x)
        x = self.norm2d(x)
        b, c, h, w = x.size()
        x = x.unsqueeze(-1)
        x = x.expand((b, c, h, w, w))  # expand to a cube
        x = self.conv3d(x)  # learn features for 3d
        x = self.norm3d(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, block: nn.Module
    ):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.block = block(out_channels)

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1(x)  # reduce spatial
        x = self.norm2(x)
        x = self.block(x)  # extract features
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, block: nn.Module
    ):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride
        )  # most likely ks=2, stride=2
        self.block = block(out_channels)

    def forward(self, x):
        x = self.conv1(x)  # reduce spatial
        x = self.block(x)  # extract features
        return x


class ClassifierBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, block: nn.Module
    ):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride
        )  # 2x2 s=2
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.block = block(out_channels)

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1(x)  # reduce spatial
        x = self.norm2(x)
        x = self.block(x)  # extract features
        return x
