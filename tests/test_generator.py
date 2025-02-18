import pytest
import torch
from lib.models.generator import Generator
from lib.utils.viz import ArrayDisplayUtility
from time import sleep


@pytest.mark.parametrize("features",[
    [1, 2, 3, 4, 5],
    [1, 8, 16, 32],
    [1, 8, 32, 64, 96],
])
def test_generator(features):
    x1 = torch.randn((1, 1, 256, 256))
    x2 = torch.randn((1, 1, 256, 256))
    x1 = x1.to("cuda")
    x2 = x2.to("cuda")
    generator = Generator(
        features=features,
        device="cuda"
    )
    generator = generator.to("cuda")
    out: torch.Tensor = generator(x1, x2)


    assert len(out.shape) == 5
    b1, c1, h1, w1 = x1.size()
    b2, c2, h2, w2, d2 = out.size()
    #sleep(15)


    assert b1 == b2
    assert c1 == c2
    assert h1 == h2
    assert w1 == w2
    assert h2 == w2 == d2


    b1, c1, h1, w1 = x2.size()
    assert b1 == b2
    assert c1 == c2
    assert h1 == h2
    assert w1 == w2
    assert h2 == w2 == d2

