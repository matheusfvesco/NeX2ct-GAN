import pytest
import torch
from lib.models.basic_blocks import Downsample, EncoderBlock, DecoderBlock, ClassifierBlock
from lib.models.convnext import ConvnextBlock, Convnext3dBlock



@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,padding,dilation",[
    (3, 64, 1, 1, 0, 1),
    (3, 4, 1, 1, 0, 1),
    (64, 256, 1, 1, 0, 1),
    (10, 50, 1, 1, 0, 1),
    (50, 10, 1, 1, 0, 1),
    (15, 3, 1, 1, 0, 1),
])
def test_downsample_channels(in_channels, out_channels, kernel_size, stride, padding, dilation):
    assert in_channels != out_channels
    x = torch.randn((5, in_channels, 256, 256))
    downsample = Downsample(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    out = downsample(x)

    assert not isinstance(downsample.conv,torch.nn.Identity)
    assert isinstance(downsample.pool,torch.nn.Identity)
    assert out.shape[1] == out_channels
    assert out.shape[2] == x.shape[2]
    assert out.shape[3] == x.shape[3]

@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,padding,dilation",[
    (10, 10, 5, 1, 0, 1),
    (3, 3, 7, 1, 0, 1),
    (3, 3, 7, 1, 1, 1),
    (256, 256, 3, 2, 0, 1),
    (256, 256, 3, 2, 0, 1),
    (256, 256, 5, 2, 1, 1),
])
def test_downsample_spatial(in_channels, out_channels, kernel_size, stride, padding, dilation):
    assert in_channels == out_channels
    x = torch.randn((5, in_channels, 256, 256))
    downsample = Downsample(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    out = downsample(x)

    assert isinstance(downsample.conv,torch.nn.Identity)
    assert not isinstance(downsample.pool,torch.nn.Identity)
    assert out.shape[1] == out_channels
    assert out.shape[1] == x.shape[1]
    assert out.shape[2] != x.shape[2]
    assert out.shape[3] != x.shape[3]


@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,padding,dilation", [
    (3, 64, 5, 1, 0, 1),
    (3, 64, 7, 1, 0, 1),
    (3, 64, 3, 1, 0, 1),
    (64, 3, 3, 1, 0, 1),
    (256, 64, 3, 1, 0, 1),
])
def test_downsample_spatial_and_channels(in_channels, out_channels, kernel_size, stride, padding, dilation):
    assert in_channels != out_channels
    x = torch.randn((5, in_channels, 256, 256))
    downsample = Downsample(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    out = downsample(x)

    assert not isinstance(downsample.conv,torch.nn.Identity)
    assert not isinstance(downsample.pool,torch.nn.Identity)
    assert out.shape[1] == out_channels
    assert out.shape[1] != x.shape[1]
    assert out.shape[2] != x.shape[2]
    assert out.shape[3] != x.shape[3]

@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,padding,dilation", [
    (3, 3, 1, 1, 0, 1),
    (64, 64, 1, 1, 0, 1),
    (50, 50, 1, 1, 0, 1),
    (30, 30, 1, 1, 0, 1),
    (256, 256, 1, 1, 0, 1),
])
def test_downsample_identity(in_channels, out_channels, kernel_size, stride, padding, dilation):
    x = torch.randn((5, in_channels, 256, 256))
    downsample = Downsample(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    out = downsample(x)
    assert isinstance(downsample.conv,torch.nn.Identity)
    assert isinstance(downsample.pool,torch.nn.Identity)
    assert out.shape[1] == out_channels
    assert out.shape[1] == x.shape[1]
    assert out.shape[2] == x.shape[2]
    assert out.shape[3] == x.shape[3]

@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride", [
    #(in_channels, out_channels, kernel_size, stride, block: nn.Module)
    (1, 8, 3, 2),
    (64, 96, 1, 1),
    (50, 50, 2, 2),
])
def test_encoder(in_channels, out_channels, kernel_size, stride):
    x = torch.randn((5, in_channels, 256, 256))
    encoder = EncoderBlock(
in_channels, out_channels, kernel_size, stride, block=ConvnextBlock
    )
    out = encoder(x)
    assert out.size(1) == out_channels

    expected_height = (256 - kernel_size) // stride + 1
    expected_width = (256 - kernel_size) // stride + 1

    assert out.size(2) == expected_height
    assert out.size(3) == expected_width


@pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride", [
    #(in_channels, out_channels, kernel_size, stride, block: nn.Module)
    (1, 8, 3, 2),
    #(1, 16, 1, 1),
    #(1, 32, 2, 2),
])
def test_classifier(in_channels, out_channels, kernel_size, stride):
    x = torch.randn((5, in_channels, 256, 256, 256))
    x = x.to("cuda")
    classifier = ClassifierBlock(
in_channels, out_channels, kernel_size, stride, block=Convnext3dBlock
    )
    classifier = classifier.to("cuda")
    out = classifier(x)
    assert out.size(1) == out_channels

    expected_height = (256 - kernel_size) // stride + 1
    expected_width = (256 - kernel_size) // stride + 1

    assert out.size(2) == expected_height
    assert out.size(3) == expected_width