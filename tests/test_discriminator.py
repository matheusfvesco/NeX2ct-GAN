import pytest
import torch
import torch.nn as nn
from lib.models.discriminator import Discriminator
from lib.models.basic_blocks import ClassifierBlock
from lib.models.convnext import Convnext3dBlock
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torch.utils.data import DataLoader
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
    )
    generator = generator.to("cuda")
    out: torch.Tensor = generator(x)

    _, c = out.size()
    assert c == 1

def test_encoder():
    features_encoder = [1, 8, 16]
    encoder = []
    for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
        encoder.append(
            ClassifierBlock(in_channels, out_channels, 2, 2, Convnext3dBlock)
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
        #T.Resize((256, 256), antialias=True),
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
            img: torch.Tensor = img.to("cuda")
            b, c, h, w = img.size()
            img = img.unsqueeze(-1)
            img = img.expand((b, c, h, w, w))
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        losses.append(running_loss)
    
    assert losses[-1] < losses[0]

