import torch
from torch import nn
from lib.models.basic_blocks import EncoderBlock, DecoderBlock, Skip
from lib.models.convnext import ConvnextBlock, Convnext3dBlock


class Generator(nn.Module):
    def __init__(self, features: list[int], device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        features_encoder = features
        self.device = device
        encoder1 = []
        encoder2 = []
        skips1 = []
        skips2 = []
        # [1, 8, 32, 64, 96]
        # (1, 8), (8, 32), (32, 64), (64, 96)
        for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
            encoder1.append(
                EncoderBlock(in_channels, out_channels, 2, 2, ConvnextBlock).to(device)
            )
            skips1.append(Skip(out_channels, out_channels))
            encoder2.append(
                EncoderBlock(in_channels, out_channels, 2, 2, ConvnextBlock).to(device)
            )
            # (1, 8) -> (8, 1)
            # (8, 32) -> (32, 8)
            # (32, 64) -> (64, 32)
            # (64, 96) -> (96, 64)
            skips2.append(Skip(out_channels, out_channels).to(device))
        
        # [1, 8, 32, 64, 96] -> [96, 64, 32, 8, 1]
        features_decoder = list(reversed(features_encoder))
        decoder1 = []
        decoder2 = []
        # [96, 64, 32, 8, 1]
        # (96, 64), (64, 32), (32, 8), (8, 1)
        for in_channels, out_channels in zip(features_decoder, features_decoder[1:]):
            decoder1.append(
                DecoderBlock(in_channels*2, out_channels, 2, 2, Convnext3dBlock).to(device) # times 2 cause unet concatenation (96+96), (8+8)
            )
            decoder2.append(
                DecoderBlock(in_channels*2, out_channels, 2, 2, Convnext3dBlock).to(device)
            )
        
        merger = []
        for in_channels, out_channels in zip(features_decoder[1:], features_decoder[2:]):
            merger.append(
                DecoderBlock(in_channels, out_channels, 2, 2, Convnext3dBlock).to(device)
            )
        

        #self.encoder1 = nn.Sequential(*tuple(encoder1))
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.skips1 = skips1
        self.skips2 = skips2
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.merger = merger
        #self.decoder1 = nn.Sequential(*tuple(decoder1))
        self.linker = Skip(features[-1], features[-1]).to(device)
        self.f_act = nn.Tanh()
    


    def forward(self, x1, x2):
        # runs encoder and store intermediary
        skips1 = []
        for layer in self.encoder1:
            layer = layer.to(self.device)
            x1 = layer(x1)
            skips1.append(x1)
        
        # runs linker and store results
        skips = []
        for x, layer in zip(skips1, self.skips1):
            layer = layer.to(self.device)
            x = layer(x)
            skips.append(x)
        skips1 = skips
        
        # last x is result of the first encoder-decoder link
        x1 = skips1[-1]
        net1_out = []
        # concat encoder outputs and runs decoder block
        for x, layer in zip(reversed(skips1), self.decoder1):
            layer = layer.to(self.device)
            x1 = torch.concat((x1,  x), dim=1)
            x1 = layer(x1)
            net1_out.append(x1)
        
        # ===========
        # NETWORK 2
        # ===========
        skips2 = []
        for layer in self.encoder2:
            layer = layer
            x2 = layer(x2)
            skips2.append(x2)
        
        # runs linker and store results
        skips = []
        for x, layer in zip(skips2, self.skips2):
            layer = layer.to(self.device)
            x = layer(x)
            skips.append(x)
        skips2 = skips
        
        # last x is result of the first encoder-decoder link
        x2 = skips2[-1]
        net2_out = []
        # concat encoder outputs and runs decoder block
        for x, layer in zip(reversed(skips2), self.decoder2):
            layer = layer.to(self.device)
            x2 = torch.concat((x2,  x), dim=1)
            x2 = layer(x2)
            net2_out.append(x2)
        
        x_out = None
        for x1, x2, layer in zip(net1_out, net2_out, self.merger):
            x2 = x2.permute((0, 1, 3, 2, 4))
            #assert x2.size()[1] == 200000, f"{[(x1.shape, x2.shape) for x1, x2 in zip(net1_out, net2_out)]}"
            x = x1 + x2
            if x_out is None:
                x_out = x
            else:
                x_out: torch.Tensor = x_out + x
                x_out = x_out / 2 # mean
            x_out = layer(x_out)
        
        x_out = self.f_act(x_out)

        return x_out
