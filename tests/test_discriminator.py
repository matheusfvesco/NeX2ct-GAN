import pytest
import torch
import torch.nn as nn
from lib.models.discriminator import Discriminator
from time import sleep

@pytest.mark.parametrize("features",[
    [1, 2, 3],
    [1, 8, 16],
    [1, 8, 32],
])
def test_discriminator(features):
    x = torch.randn((2, 1, 256, 256, 256))
    x = x.to("cuda")
    generator = Discriminator(
        features=features,
        device="cuda"
    )
    generator = generator.to("cuda")
    out: torch.Tensor = generator(x)

    sleep(15)