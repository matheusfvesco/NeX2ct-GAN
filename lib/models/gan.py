import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
from lib.losses import GANLoss
from lib.config import Config


class GanModel(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opts: Config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

        device = opts.device
        if device == "xla":
            device = xm.xla_device(0)
        self.gan_loss = GANLoss(device=opts.device)

        self.branch_loss = nn.MSELoss()
        self.identity_loss = nn.MSELoss()

        self.g_optim = torch.optim.AdamW(
            self.generator.parameters(), lr=opts.generator.lr
        )
        self.d_optim = torch.optim.AdamW(
            self.discriminator.parameters(), lr=opts.discriminator.lr
        )

    def train_batch(self, x1, x2, g_truth):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

        # forward g
        fake_cts, b1, b2 = self.generator(x1, x2)

        # forward d (fake)
        fake_d = self.discriminator(fake_cts.detach())
        loss_d = self.gan_loss(fake_d, False)  # is_real = False

        # forward d (real)
        real_d = self.discriminator(g_truth)
        loss_d += self.gan_loss(real_d, True)  # is_real = True

        # backward d
        loss_d.backward()
        self.d_optim.step()

        # backward g
        loss_g = self.discriminator(fake_cts)
        loss_g = self.gan_loss(loss_g, False)

        # branch loss
        b_loss = self.branch_loss(b1, g_truth)
        b_loss += self.branch_loss(b2, g_truth)
        b_loss = b_loss / 2

        loss_g.backward()
        self.g_optim.step()

        return loss_g, loss_d

    def forward(self, x1, x2):
        x, x1, x2 = self.generator(x1, x2)
        return x
