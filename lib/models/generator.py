import torch
from torch import nn
from lib.models.basic_blocks import EncoderBlock, DecoderBlock, Skip, Linker


class DoubleCnnGenerator(nn.Module):
    def __init__(
        self,
        features: list[int],
        permute_order: tuple,
        blocks: tuple[nn.Module, nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        encoder_block, decoder_block = blocks
        features_encoder = features
        encoder1 = []
        encoder2 = []
        skips1 = []
        skips2 = []
        # [1, 8, 32, 64, 96]
        # (1, 8), (8, 32), (32, 64), (64, 96)
        for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
            encoder1.append(
                EncoderBlock(in_channels, out_channels, 2, 2, encoder_block)
            )
            skips1.append(Skip(out_channels, out_channels))
            encoder2.append(
                EncoderBlock(in_channels, out_channels, 2, 2, encoder_block)
            )
            # (1, 8) -> (8, 1)
            # (8, 32) -> (32, 8)
            # (32, 64) -> (64, 32)
            # (64, 96) -> (96, 64)
            skips2.append(Skip(out_channels, out_channels))

        # [1, 8, 32, 64, 96] -> [96, 64, 32, 8, 1]
        features_decoder = list(reversed(features_encoder))
        decoder1 = []
        decoder2 = []
        # [96, 64, 32, 8, 1]
        # (96, 64), (64, 32), (32, 8), (8, 1)
        for in_channels, out_channels in zip(features_decoder, features_decoder[1:]):
            decoder1.append(
                DecoderBlock(
                    in_channels * 2, out_channels, 2, 2, decoder_block
                )  # times 2 cause unet concatenation (96+96), (8+8)
            )
            decoder2.append(
                DecoderBlock(in_channels * 2, out_channels, 2, 2, decoder_block)
            )

        merger = []
        for in_channels, out_channels in zip(
            features_decoder[1:], features_decoder[2:]
        ):
            merger.append(DecoderBlock(in_channels, out_channels, 2, 2, decoder_block))

        self.encoder1 = nn.ModuleList(encoder1)
        self.encoder2 = nn.ModuleList(encoder2)
        self.skips1 = nn.ModuleList(skips1)
        self.skips2 = nn.ModuleList(skips2)
        self.decoder1 = nn.ModuleList(decoder1)
        self.decoder2 = nn.ModuleList(decoder2)
        self.merger = nn.ModuleList(merger)
        self.f_act = nn.Tanh()
        self.permute1, self.permute2 = permute_order

    def forward(self, x1, x2):
        # runs encoder and store intermediary
        skips1 = []
        for layer in self.encoder1:
            layer = layer
            x1 = layer(x1)
            skips1.append(x1)

        # runs linker and store results
        skips = []
        for x, layer in zip(skips1, self.skips1):
            layer = layer
            x = layer(x)
            skips.append(x)
        skips1 = skips

        # last x is result of the first encoder-decoder link
        x1 = skips1[-1]
        net1_out = []
        # concat encoder outputs and runs decoder block
        for x, layer in zip(reversed(skips1), self.decoder1):
            layer = layer
            x1 = torch.concat((x1, x), dim=1)
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
            layer = layer
            x = layer(x)
            skips.append(x)
        skips2 = skips

        # last x is result of the first encoder-decoder link
        x2 = skips2[-1]
        net2_out = []
        # concat encoder outputs and runs decoder block
        for x, layer in zip(reversed(skips2), self.decoder2):
            layer = layer
            x2 = torch.concat((x2, x), dim=1)
            x2 = layer(x2)
            net2_out.append(x2)

        # merges input list
        x_out = None
        for x1, x2, layer in zip(net1_out, net2_out, self.merger):
            # x2 = x2.permute((0, 1, 3, 2, 4))
            x1 = x1.permute(*tuple(self.permute1))
            x2 = x2.permute(*tuple(self.permute2))
            # assert x2.size()[1] == 200000, f"{[(x1.shape, x2.shape) for x1, x2 in zip(net1_out, net2_out)]}"
            x = x1 + x2
            if x_out is None:
                x_out = x
            else:
                x_out: torch.Tensor = x_out + x
                x_out = x_out / 2  # mean
            x_out = layer(x_out)

        x_out = self.f_act(x_out)

        return x_out, net1_out[-1], net2_out[-1]


class DoubleLinkerGenerator(nn.Module):
    def __init__(
        self,
        features: list[int],
        permute_order: tuple,
        blocks: tuple[nn.Module, nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        encoder_block, decoder_block = blocks
        features_encoder = features
        encoder1 = []
        encoder2 = []
        skips1 = []
        skips2 = []
        # [1, 8, 32, 64, 96]
        # (1, 8), (8, 32), (32, 64), (64, 96)
        dim = 256
        for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
            encoder1.append(
                EncoderBlock(in_channels, out_channels, 2, 2, encoder_block)
            )
            skips1.append(Skip(out_channels, out_channels))
            encoder2.append(
                EncoderBlock(in_channels, out_channels, 2, 2, encoder_block)
            )
            # (1, 8) -> (8, 1)
            # (8, 32) -> (32, 8)
            # (32, 64) -> (64, 32)
            # (64, 96) -> (96, 64)
            dim = (dim - 2) // 2 + 1
            skips2.append(Skip(out_channels, out_channels))

        # replaces last connection for a linear linker
        skips1[-1] = Linker(out_channels, out_channels, dim)
        skips2[-1] = Linker(out_channels, out_channels, dim)

        # [1, 8, 32, 64, 96] -> [96, 64, 32, 8, 1]
        features_decoder = list(reversed(features_encoder))
        decoder1 = []
        decoder2 = []
        # [96, 64, 32, 8, 1]
        # (96, 64), (64, 32), (32, 8), (8, 1)
        for in_channels, out_channels in zip(features_decoder, features_decoder[1:]):
            decoder1.append(
                DecoderBlock(
                    in_channels * 2, out_channels, 2, 2, decoder_block
                )  # times 2 cause unet concatenation (96+96), (8+8)
            )
            decoder2.append(
                DecoderBlock(in_channels * 2, out_channels, 2, 2, decoder_block)
            )

        merger = []
        for in_channels, out_channels in zip(
            features_decoder[1:], features_decoder[2:]
        ):
            merger.append(DecoderBlock(in_channels, out_channels, 2, 2, decoder_block))

        self.encoder1 = nn.ModuleList(encoder1)
        self.encoder2 = nn.ModuleList(encoder2)
        self.skips1 = nn.ModuleList(skips1)
        self.skips2 = nn.ModuleList(skips2)
        self.decoder1 = nn.ModuleList(decoder1)
        self.decoder2 = nn.ModuleList(decoder2)
        self.merger = nn.ModuleList(merger)
        self.f_act = nn.Tanh()
        self.permute1, self.permute2 = permute_order

    def forward(self, x1, x2):
        # runs encoder and store intermediary
        skips1 = []
        for layer in self.encoder1:
            layer = layer
            x1 = layer(x1)
            skips1.append(x1)

        # runs linker and store results
        skips = []
        for x, layer in zip(skips1, self.skips1):
            layer = layer
            x = layer(x)
            skips.append(x)
        skips1 = skips

        # last x is result of the first encoder-decoder link
        x1 = skips1[-1]
        net1_out = []
        # concat encoder outputs and runs decoder block
        for x, layer in zip(reversed(skips1), self.decoder1):
            layer = layer
            x1 = torch.concat((x1, x), dim=1)
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
            layer = layer
            x = layer(x)
            skips.append(x)
        skips2 = skips

        # last x is result of the first encoder-decoder link
        x2 = skips2[-1]
        net2_out = []
        # concat encoder outputs and runs decoder block
        for x, layer in zip(reversed(skips2), self.decoder2):
            layer = layer
            x2 = torch.concat((x2, x), dim=1)
            x2 = layer(x2)
            net2_out.append(x2)

        # merges input list
        x_out = None
        for x1, x2, layer in zip(net1_out, net2_out, self.merger):
            # x2 = x2.permute((0, 1, 3, 2, 4))
            x1 = x1.permute(*tuple(self.permute1))
            x2 = x2.permute(*tuple(self.permute2))
            # assert x2.size()[1] == 200000, f"{[(x1.shape, x2.shape) for x1, x2 in zip(net1_out, net2_out)]}"
            x = x1 + x2
            if x_out is None:
                x_out = x
            else:
                x_out: torch.Tensor = x_out + x
                x_out = x_out / 2  # mean
            x_out = layer(x_out)

        x_out = self.f_act(x_out)

        return x_out, net1_out[-1], net2_out[-1]


class SingleLinkerGenerator(nn.Module):
    def __init__(
        self,
        features: list[int],
        permute_order: tuple,
        blocks: tuple[nn.Module, nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        encoder_block, decoder_block = blocks
        features_encoder = features
        encoder1 = []
        skips1 = []
        # [1, 8, 32, 64, 96]
        # (1, 8), (8, 32), (32, 64), (64, 96)
        dim = 256
        for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
            encoder1.append(
                EncoderBlock(in_channels, out_channels, 2, 2, encoder_block)
            )
            skips1.append(Skip(out_channels, out_channels))
            # (1, 8) -> (8, 1)
            # (8, 32) -> (32, 8)
            # (32, 64) -> (64, 32)
            # (64, 96) -> (96, 64)
            dim = (dim - 2) // 2 + 1

        # replaces last connection for a linear linker
        skips1[-1] = Linker(out_channels, out_channels, dim)

        # [1, 8, 32, 64, 96] -> [96, 64, 32, 8, 1]
        features_decoder = list(reversed(features_encoder))
        decoder1 = []
        # [96, 64, 32, 8, 1]
        # (96, 64), (64, 32), (32, 8), (8, 1)
        for in_channels, out_channels in zip(features_decoder, features_decoder[1:]):
            decoder1.append(
                DecoderBlock(
                    in_channels * 2, out_channels, 2, 2, decoder_block
                )  # times 2 cause unet concatenation (96+96), (8+8)
            )

        self.encoder1 = nn.ModuleList(encoder1)
        self.skips1 = nn.ModuleList(skips1)
        self.decoder1 = nn.ModuleList(decoder1)
        self.f_act = nn.Tanh()
        self.permute1, self.permute2 = permute_order

    def forward(self, x1, x2):
        # runs encoder and store intermediary
        skips1 = []
        for layer in self.encoder1:
            layer = layer
            x1 = layer(x1)
            skips1.append(x1)

        # runs linker and store results
        skips = []
        for x, layer in zip(skips1, self.skips1):
            layer = layer
            x = layer(x)
            skips.append(x)
        skips1 = skips

        # last x is result of the first encoder-decoder link
        x1 = skips1[-1]
        net1_out = []
        # concat encoder outputs and runs decoder block
        for x, layer in zip(reversed(skips1), self.decoder1):
            layer = layer
            x1 = torch.concat((x1, x), dim=1)
            x1 = layer(x1)
            net1_out.append(x1)

        x_out = self.f_act(x1)

        return x_out, net1_out[-1], net1_out[-1]


class SingleLinkerNonUnetGenerator(nn.Module):
    def __init__(
        self,
        features: list[int],
        permute_order: tuple,
        blocks: tuple[nn.Module, nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        encoder_block, decoder_block = blocks
        features_encoder = features
        encoder1 = []
        # [1, 8, 32, 64, 96]
        # (1, 8), (8, 32), (32, 64), (64, 96)
        dim = 256
        for in_channels, out_channels in zip(features_encoder, features_encoder[1:]):
            encoder1.append(
                EncoderBlock(in_channels, out_channels, 2, 2, encoder_block)
            )
            # (1, 8) -> (8, 1)
            # (8, 32) -8i> (32, 8)
            # (32, 64) -> (64, 32)
            # (64, 96) -> (96, 64)
            dim = (dim - 2) // 2 + 1
        self.linker = Linker(out_channels, out_channels, dim)

        # replaces last connection for a linear linker

        # [1, 8, 32, 64, 96] -> [96, 64, 32, 8, 1]
        features_decoder = list(reversed(features_encoder))
        decoder1 = []
        # [96, 64, 32, 8, 1]
        # (96, 64), (64, 32), (32, 8), (8, 1)
        for in_channels, out_channels in zip(features_decoder, features_decoder[1:]):
            decoder1.append(
                DecoderBlock(
                    in_channels, out_channels, 2, 2, decoder_block
                )  # times 2 cause unet concatenation (96+96), (8+8)
            )
        
        print(out_channels)
        

        self.encoder = nn.Sequential(*encoder1)
        self.decoder = nn.Sequential(*decoder1)

        self.f_act = nn.Tanh()
        self.permute1, self.permute2 = permute_order

    def forward(self, x1, x2):
        # runs encoder and store intermediary
        
        x_out = self.encoder(x1)
        x_out = self.linker(x_out)
        x_out = self.decoder(x_out)
        x_out = self.f_act(x_out)


        return x_out, x_out, x_out
