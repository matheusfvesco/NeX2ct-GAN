import pytest
import torch
from lib.models.simple_block import SimpleBlock, Simple3dBlock
from time import sleep

@pytest.mark.parametrize("in_channels,size",[
    (3, 200),
    (8, 200),
    (16, 100),
    (32, 100),
    (64, 100),
    (64, 50),
])
def test_convnext_block(in_channels, size):
    x = torch.randn((5, in_channels, size, size))
    x = x.to("cuda")
    convnext = SimpleBlock(
        in_channels=in_channels,
    )
    convnext = convnext.to("cuda")
    out = convnext(x)

    assert out.shape == x.shape

@pytest.mark.parametrize("in_channels,size",[
    (3, 50),
    (5, 50),
    (3, 100),
    (5, 100), # computationally intensive
])
def test_convnext_3d_block(in_channels, size):
    x = torch.randn((5, in_channels, size, size, size))
    x = x.to("cuda")

    convnext = Simple3dBlock(
        in_channels=in_channels,
    )
    convnext = convnext.to("cuda")

    out = convnext(x)

    assert out.shape == x.shape