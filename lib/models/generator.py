import torch
from torch import nn
from lib.models.basic_blocks import EncoderBlock, DecoderBlock, Skip
from lib.models.convnext import ConvnextBlock, Convnext3dBlock


class Generator(nn.Module):
    def __init__(self, features: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        features_encoder = [1] + features
        encoder = []
        for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
            encoder.append(
                EncoderBlock(in_channels, out_channels, 2, 2, ConvnextBlock)
            )
        
        features_decoder = list(reversed(features_encoder))
        decoder = []
        for in_channels, out_channels in zip(features_decoder, features_decoder[1:]):
            decoder.append(
                DecoderBlock(in_channels, out_channels, 2, 2, Convnext3dBlock)
            )
        

        self.encoder = nn.Sequential(*tuple(encoder))
        print(self.encoder)
        self.decoder = nn.Sequential(*tuple(decoder))
        self.linker = Skip(features[-1], features[-1])
    
    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        x = self.linker(x)
        x = self.decoder(x)
        return x
