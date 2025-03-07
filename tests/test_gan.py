import torch
import torch.nn as nn
import pytest
from lib.models.generator import ConvnextGenerator
from lib.models.discriminator import Discriminator
from lib.models.gan import GanModel
from lib.config import Config


@pytest.mark.parametrize(
    "gen_features,disc_features",
    [
        ([1, 2, 3, 4, 5], [1, 2, 3]),
        ([1, 8, 16, 32], [1, 8, 16]),
        ([1, 8, 32, 96], [1, 8, 32]),
    ],
)
def test_gan(gen_features, disc_features):
    x1 = torch.randn((1, 1, 256, 256)).to("cuda")
    x2 = torch.randn((1, 1, 256, 256)).to("cuda")
    opts = Config("experiments/base.yaml")
    gen = ConvnextGenerator(
        gen_features, permute_order=([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    ).to("cuda")
    disc = Discriminator(disc_features).to("cuda")
    gan = GanModel(gen, disc, opts)

    out = gan(x1, x2)
    out_d = gan.discriminator(out)

    b_in, c, h, w = x1.size()
    b_out, classes = out_d.size()
    assert b_in == b_out
    assert classes == 1

    b, c, h, w, d = out.size()
    assert h == w == d
    assert h == 256

    
