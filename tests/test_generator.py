import pytest
import torch
from lib.models.generator import Generator
from time import sleep


@pytest.mark.parametrize("features",[
    [8, 32, 64, 96],
    [2, 3, 4, 5],
])
def test_convnext_block(features):
    x = torch.randn((5, 1, 256, 256))
    x = x.to("cuda")
    generator = Generator(
        features=features,
    )
    generator = generator.to("cuda")
    out = generator(x)

    assert len(out.shape) == 5
