import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from lib.datasets import CTFactory
from lib.config import Config
from lib.models.gan import GanModel
from lib.transforms import Resize, Normalize
from lib.models.generator import ConvnextGenerator
from lib.models.discriminator import Discriminator
from lib.utils.utils import collate_fn
from lib.logging import setup_logger
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torchvision.utils import make_grid


def main(opts: Config):
    logger = setup_logger(
        "train",
        opts.experiment_name,
        opts.comment,
        "-".join([p for p in opts.experiment_name.split()]),
    )

    summary = SummaryWriter(comment=opts.experiment_name)

    ckpt_path = Path("models/gan")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_path / ("_".join([p for p in opts.experiment_name.split()]) + "_latest")
    best_path = ckpt_path / ("_".join([p for p in opts.experiment_name.split()]) + "_best")


    transforms = T.Compose([
        Resize((256, 256)),
        Normalize()
    ])

    transforms_ct = T.Compose([
        Resize((256, 256, 256)),
        Normalize()
    ])

    train, test = CTFactory(
        dataset_type=opts.dataset,
        data_root=opts.data_root,
        set_root=opts.dataset_set_root,
        transforms=transforms,
        ct_transforms=transforms_ct,
    )

    train_loader = DataLoader(train, batch_size=opts.batch_size,collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=opts.batch_size,collate_fn=collate_fn)

    device = opts.device

    generator = ConvnextGenerator(opts.generator.features,(opts.generator.view1_permute, opts.generator.view2_permute)).to(device)
    discriminator = Discriminator(opts.discriminator.features).to(device)
    gan_model = GanModel(generator, discriminator, opts)

    logger.info(f"training for {opts.epochs} epochs")
    best_loss = None
    for i in range(opts.epochs):
        logger.info(f"Epoch {i}: ")
        total_loss = 0.0
        for idx, (cts, xr1, xr2, path) in enumerate(train_loader):
            cts = cts.to(device)
            xr1 = xr1.to(device)
            xr2 = xr2.to(device)
            loss_g, loss_d = gan_model.train_batch(xr1, xr2, cts)
            total_loss += loss_g + loss_d

        # saves latest
        torch.save({
            "discriminator_w": gan_model.discriminator.state_dict(),
            "generator_2": gan_model.generator.state_dict(),
            "epochs": i,
            "loss_train": total_loss
        }, latest_path)
        
        total_loss = total_loss / (idx+1)

        logger.info(f"Total loss: {total_loss}")
        if best_loss is None:
            best_loss = total_loss

        if total_loss < best_loss:
            logger.info(f"New best: {total_loss} | previous best: {best_loss}")
            best_loss = total_loss
            torch.save({
                "discriminator_w": gan_model.discriminator.state_dict(),
                "generator_2": gan_model.generator.state_dict(),
                "epochs": i,
                "loss_train": total_loss
            }, best_path)
        
        if not i % 5 == 0:
            continue

        # fakes
        axials = []
        coronals = []
        saggitals = []
        # reals
        axials_r = []
        coronals_r = []
        saggitals_r = []
        for idx, (cts, xr1, xr2, path) in enumerate(test_loader):
            fakes = gan_model(xr1, xr2)
            a = []
            c = []
            s = []

            ar = []
            cr = []
            sr = []
            for fake, g_truth in zip(fakes, cts):
                # now HWD
                fake = fake.squeeze()
                g_truth = g_truth.squeeze()
                fake_mid = fake.shape[-1] // 2
                real_mid = g_truth.shape[-1] // 2

                # Appends fake images for batch with 3 channels
                a.append(torch.stack([fake[fake_mid, :, :], fake[fake_mid, :, :], fake[fake_mid, :, :]]))
                c.append(torch.stack([fake[:, fake_mid, :], fake[:, fake_mid, :], fake[:, fake_mid, :]]))
                s.append(torch.stack([fake[:, :, fake_mid], fake[:, :, fake_mid], fake[:, :, fake_mid]]))

                # Append real images for the batch with 3 channels
                ar.append(torch.stack([g_truth[real_mid, :, :], g_truth[real_mid, :, :], g_truth[real_mid, :, :]]))
                cr.append(torch.stack([g_truth[:, real_mid, :], g_truth[:, real_mid, :], g_truth[:, real_mid, :]]))
                sr.append(torch.stack([g_truth[:, :, real_mid], g_truth[:, :, real_mid], g_truth[:, :, real_mid]]))
            # sum lists
            axials += a
            coronals += c
            saggitals_r += s

            axials_r += ar
            coronals_r += cr
            saggitals_r += sr
        axials = make_grid(torch.stack(axials))
        coronals = make_grid(torch.stack(coronals))
        saggitals = make_grid(torch.stack(saggitals))

        axials_r = make_grid(torch.stack(axials_r))
        coronals_r = make_grid(torch.stack(coronals_r))
        saggitals_r = make_grid(torch.stack(coronals_r))

        summary.add_image("axials_fake", axials, i)
        summary.add_image("coronals_fake", coronals, i)
        summary.add_image("saggitals_fake", saggitals, i)

        summary.add_image("axials_real", axials_r, i)
        summary.add_image("coronals_real", coronals_r, i)
        summary.add_image("saggitals_real", saggitals_r, i)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str)

    args = parser.parse_args()
    config = Config(args.experiment_path)
    main(config)
