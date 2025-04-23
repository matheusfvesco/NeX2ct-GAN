import torch
import torch.nn as nn
from lib.losses import GANLoss
from lib.config import Config
from lib.factories import optim_factory


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
        self.gan_loss = GANLoss(device=opts.device)

        self.branch_loss = nn.MSELoss()
        self.identity_loss = nn.MSELoss()

        optim = optim_factory(opts.train.optimizer)
        self.g_optim = optim(
            self.generator.parameters(), lr=opts.generator.lr
        )
        self.d_optim = optim(
            self.discriminator.parameters(), lr=opts.discriminator.lr
        )

        self.view1_permute = opts.generator.view1_permute
        self.view2_permute = opts.generator.view2_permute

        self.opts = opts
    
    def train_discriminator_batch(self, x1, x2, g_truth):
        self.discriminator.zero_grad()

        # forward g
        fake_cts, b1, b2 = self.generator(x1, x2)

        assert fake_cts.min() >= -1
        assert fake_cts.max() <= 1

        # forward d (fake)
        fake_d = self.discriminator(fake_cts.detach())
        loss_d = self.gan_loss(fake_d, False)  # is_real = False

        # forward d (real)
        real_d = self.discriminator(g_truth)
        loss_d += self.gan_loss(real_d, True)  # is_real = True

        # backward d
        loss_d.backward()
        self.d_optim.step()

        return loss_d



    def train_batch(self, x1: torch.Tensor, x2: torch.Tensor, g_truth: torch.Tensor):
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
        total_g_loss = loss_g

        # identity loss (fake vs g_truth)
        if self.opts.generator.identity_loss:
            identity_loss = self.identity_loss(fake_cts, g_truth)
            total_g_loss += identity_loss

        # branch loss
        if self.opts.generator.branch_loss:
            permute1 = tuple(torch.argsort(torch.Tensor(self.view1_permute)))
            permute2 = tuple(torch.argsort(torch.Tensor(self.view2_permute)))
            b_loss = self.branch_loss(b1, g_truth.permute(permute1))
            b_loss += self.branch_loss(b2, g_truth.permute(permute2))
            b_loss = b_loss / 2
            total_g_loss += b_loss

        total_g_loss.backward()
        self.g_optim.step()

        return loss_g, loss_d

    def forward(self, x1, x2):
        x, x1, x2 = self.generator(x1, x2)
        return x
