import pytest
import torch
import torch.nn as nn
from lib.models.generator import Generator
from lib.models.basic_blocks import EncoderBlock
from lib.models.convnext import ConvnextBlock
from lib.utils.viz import ArrayDisplayUtility
from time import sleep
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader


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

def test_encoder():
    features_encoder = [1, 8, 16, 32]
    encoder = []
    for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
        encoder.append(
            EncoderBlock(in_channels, out_channels, 2, 2, ConvnextBlock).to("cuda")
        )
    model = nn.Sequential(
        *tuple(encoder)
    )
    model = nn.Sequential(
        model,
        nn.Flatten(),
        nn.LazyLinear(28),
        nn.Linear(28, 10)
    )
    model.to("cuda")

    transforms = T.Compose([
        T.ToTensor(),
        T.Resize((256, 256), antialias=True),
    ])
    dataset = MNIST("~/", download=True, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=64)
    lr = 1e-3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr)
    losses = []
    for epoch in range(3):
        running_loss = 0.0
        for img, labels in dataloader:
            img = img.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        losses.append(running_loss)
    
    assert losses[-1] < losses[0]
        
