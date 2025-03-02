import torch
import torch.nn as nn
import pytest
from lib.models.generator import Generator
from lib.models.discriminator import Discriminator
from lib.models.gan import GanModel

@pytest.mark.parametrize("gen_features,disc_features",[
    ([1, 2, 3, 4, 5],[1, 2, 3]),
    ([1, 8, 16, 32],[1, 8, 16]),
    ([1, 8, 32, 96],[1, 8, 32]),
])
def test_discriminator(gen_features, disc_features):
    x1 = torch.randn((1, 1, 256, 256)).to("cuda")
    x2 = torch.randn((1, 1, 256, 256)).to("cuda")
    gen = Generator(gen_features, permute_order=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), device="cuda")
    disc = Discriminator(disc_features, device="cuda")
    gan = GanModel(gen,disc)

    out = gan(x1, x2)

    b_in, c, h, w = x1.size()
    b_out, classes = out.size()
    assert b_in == b_out
    assert classes == 2
    
