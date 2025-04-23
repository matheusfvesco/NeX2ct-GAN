import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from lib.factories import CTFactory
from lib.config import Config
from lib.models.gan import GanModel
from lib.transforms import Resize, Normalize
from lib.models.generator import DoubleCnnGenerator
from lib.discriminator import Discriminator
from lib.utils.utils import collate_fn
from lib.logging import setup_logger
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torchvision.utils import make_grid, save_image
from lib.factories import block_factory, generator_factory
from tqdm import tqdm
import os


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
    latest_path = ckpt_path / ("_".join([p for p in opts.experiment_name.split()]))
    latest_path.mkdir(parents=True, exist_ok=True)
    latest_path = latest_path / "latest.ckpt"

    best_path = ckpt_path / ("_".join([p for p in opts.experiment_name.split()]))
    best_path.mkdir(parents=True, exist_ok=True)
    best_path = best_path / "best.ckpt"

    continue_training = opts.continue_t and os.path.exists(latest_path)

    data_save_path = Path("output/training-plot") / "-".join(
        [p for p in opts.experiment_name.split()]
    )
    image_save_path = data_save_path / "gan-images"
    image_save_path.mkdir(parents=True, exist_ok=True)

    transforms = T.Compose([Resize((256, 256)), Normalize()])

    transforms_ct = T.Compose([Resize((256, 256, 256)), Normalize()])

    train, test = CTFactory(
        dataset_type=opts.dataset,
        data_root=opts.data_root,
        set_root=opts.dataset_set_root,
        transforms=transforms,
        ct_transforms=transforms_ct,
    )

    train_loader = DataLoader(train, batch_size=opts.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=opts.batch_size, collate_fn=collate_fn)

    device = opts.device

    gen_blocks = block_factory(opts.generator.arch)
    _, disc_block = block_factory(opts.discriminator.arch)

    generator = generator_factory(opts.generator.type)
    generator = generator(
        opts.generator.features,
        (opts.generator.view1_permute, opts.generator.view2_permute),
        blocks=gen_blocks,
    ).to(device)
    discriminator = Discriminator(opts.discriminator.features, block=disc_block).to(
        device
    )

    best_loss = None
    epochs = 0
    if continue_training:
        logger.info("Found existing checkpoint, continuing training from it...")
        data = torch.load(latest_path)
        epochs = data["epochs"]
        best_loss = data["best_loss"]
        generator.load_state_dict(data["generator_w"])
        discriminator.load_state_dict(data["discriminator_w"])
        del data

    gan_model = GanModel(generator, discriminator, opts)
    with open(data_save_path / "model_arch.txt", "w") as f:
        f.write(f"{gan_model.generator}\n{gan_model.discriminator}")

    logger.info(f"training for {opts.epochs} epochs")
    main_pbar = tqdm(range(epochs, opts.epochs))
    for i in main_pbar:
        main_pbar.set_description(f"Epoch {i}")
        logger.info(f"Epoch {i}: ")
        total_loss = 0.0
        total_loss_g = 0.0
        total_loss_d = 0.0

        if opts.disc_warmup:
            # pretrains discriminator
            if i == 0:
                logger.info("Training discriminator for n epochs")
                pretrain_pbar = tqdm(range(1))
                for j in pretrain_pbar:
                    pretrain_pbar.set_description(f"Pretrain epoch {j}")
                    epoch_pbar = tqdm(train_loader)
                    for idx, (cts, xr1, xr2, path) in enumerate(epoch_pbar):
                        cts = cts.to(device)
                        xr1 = xr1.to(device)
                        xr2 = xr2.to(device)

                        assert cts.max() == 1
                        assert cts.min() == -1

                        assert xr1.max() == 1
                        assert xr1.min() == -1
                        assert xr2.max() == 1
                        assert xr2.min() == -1

                        loss_d = gan_model.train_discriminator_batch(xr1, xr2, cts)
                        epoch_pbar.set_description(f"loss_d: {loss_d}")
                        summary.add_scalar("epoch_loss_d", loss_d, global_step=idx)
                        total_loss_d += loss_d
                    logger.info(f"Loss_d epoch {j}: {total_loss_d / idx}")
                pretrain_pbar.close()
            # saves latest
            torch.save(
                {
                    "discriminator_w": gan_model.discriminator.state_dict(),
                    "generator_w": gan_model.generator.state_dict(),
                    "epochs": i,
                    "loss_train": total_loss,
                    "best_loss": best_loss,
                },
                latest_path,
            )

        total_loss_d = 0.0

        epoch_pbar = tqdm(train_loader)
        for idx, (cts, xr1, xr2, path) in enumerate(epoch_pbar):
            cts = cts.to(device)
            xr1 = xr1.to(device)
            xr2 = xr2.to(device)

            assert cts.max() == 1
            assert cts.min() == -1

            assert xr1.max() == 1
            assert xr1.min() == -1
            assert xr2.max() == 1
            assert xr2.min() == -1

            loss_g, loss_d = gan_model.train_batch(xr1, xr2, cts)
            summary.add_scalar("epoch_loss_g", loss_g, global_step=idx)
            summary.add_scalar("epoch_loss_d", loss_d, global_step=idx)
            summary.add_scalar("epoch_loss", loss_g + loss_d, global_step=idx)
            total_loss += loss_g + loss_d
            total_loss_g += loss_g
            total_loss_d += loss_d
            epoch_pbar.set_description(
                f"loss_g: {loss_g} | loss_d: {loss_d} | total: {loss_g + loss_d}"
            )

        # saves latest
        torch.save(
            {
                "discriminator_w": gan_model.discriminator.state_dict(),
                "generator_w": gan_model.generator.state_dict(),
                "epochs": i,
                "loss_train": total_loss,
                "best_loss": best_loss,
            },
            latest_path,
        )

        total_loss = total_loss / (idx + 1)
        total_loss_g = total_loss_g / (idx + 1)
        total_loss_d = total_loss_d / (idx + 1)
        summary.add_scalar("loss_g", total_loss_g, global_step=i)
        summary.add_scalar("loss_d", total_loss_d, global_step=i)
        summary.add_scalar("loss_total", total_loss, global_step=i)

        logger.info(f"Loss_g: {total_loss_g} | Loss_d: {total_loss_d}")
        logger.info(f"Total loss: {total_loss}")
        if best_loss is None:
            best_loss = total_loss + 1

        if total_loss < best_loss:
            logger.info(f"New best: {total_loss} | previous best: {best_loss}")
            best_loss = total_loss
            torch.save(
                {
                    "discriminator_w": gan_model.discriminator.state_dict(),
                    "generator_w": gan_model.generator.state_dict(),
                    "epochs": i,
                    "loss_train": total_loss,
                    "best_loss": best_loss,
                },
                best_path,
            )

        if not i % 5 == 0:
            continue

        with torch.no_grad():
            # fakes
            axials = []
            coronals = []
            saggitals = []
            # reals
            axials_r = []
            coronals_r = []
            saggitals_r = []
            for idx, (cts, xr1, xr2, path) in enumerate(test_loader):
                xr1 = xr1.to(device)
                xr2 = xr2.to(device)
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
                    a.append(
                        torch.stack(
                            [
                                fake[fake_mid, :, :],
                                fake[fake_mid, :, :],
                                fake[fake_mid, :, :],
                            ]
                        )
                    )
                    c.append(
                        torch.stack(
                            [
                                fake[:, fake_mid, :],
                                fake[:, fake_mid, :],
                                fake[:, fake_mid, :],
                            ]
                        )
                    )
                    s.append(
                        torch.stack(
                            [
                                fake[:, :, fake_mid],
                                fake[:, :, fake_mid],
                                fake[:, :, fake_mid],
                            ]
                        )
                    )

                    # Append real images for the batch with 3 channels
                    sr.append(
                        torch.stack(
                            [
                                g_truth[real_mid, :, :],
                                g_truth[real_mid, :, :],
                                g_truth[real_mid, :, :],
                            ]
                        )
                    )
                    cr.append(
                        torch.stack(
                            [
                                g_truth[:, real_mid, :],
                                g_truth[:, real_mid, :],
                                g_truth[:, real_mid, :],
                            ]
                        )
                    )
                    ar.append(
                        torch.stack(
                            [
                                g_truth[:, :, real_mid],
                                g_truth[:, :, real_mid],
                                g_truth[:, :, real_mid],
                            ]
                        )
                    )
                # sum lists
                axials += a
                coronals += c
                saggitals += s

                axials_r += ar
                coronals_r += cr
                saggitals_r += sr

            # sums 1 and divide by 2 to transform range [-1, 1] to [0, 1]
            axials = make_grid((torch.stack(axials) + 1) / 2)
            coronals = make_grid((torch.stack(coronals) + 1) / 2)
            saggitals = make_grid((torch.stack(saggitals) + 1) / 2)

            axials_r = make_grid((torch.stack(axials_r) + 1) / 2)
            coronals_r = make_grid((torch.stack(coronals_r) + 1) / 2)
            saggitals_r = make_grid((torch.stack(saggitals_r) + 1) / 2)

            summary.add_image("axials_fake", axials, i)
            summary.add_image("coronals_fake", coronals, i)
            summary.add_image("saggitals_fake", saggitals, i)

            summary.add_image("axials_real", axials_r, i)
            summary.add_image("coronals_real", coronals_r, i)
            summary.add_image("saggitals_real", saggitals_r, i)

            save_image(axials, image_save_path / f"axials_fake_{i}.jpeg")
            save_image(coronals, image_save_path / f"coronals_fake_{i}.jpeg")
            save_image(saggitals, image_save_path / f"saggitals_fake_{i}.jpeg")

            save_image(axials_r, image_save_path / f"axials_real_{i}.jpeg")
            save_image(coronals_r, image_save_path / f"coronals_real_{i}.jpeg")
            save_image(saggitals_r, image_save_path / f"saggitals_real_{i}.jpeg")

    i = opts.epochs + 1
    # saves train set results at the end of training
    with torch.no_grad():
        # fakes
        axials = []
        coronals = []
        saggitals = []
        # reals
        axials_r = []
        coronals_r = []
        saggitals_r = []
        for idx, (cts, xr1, xr2, path) in enumerate(train_loader):
            xr1 = xr1.to(device)
            xr2 = xr2.to(device)
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
                a.append(
                    torch.stack(
                        [
                            fake[fake_mid, :, :],
                            fake[fake_mid, :, :],
                            fake[fake_mid, :, :],
                        ]
                    )
                )
                c.append(
                    torch.stack(
                        [
                            fake[:, fake_mid, :],
                            fake[:, fake_mid, :],
                            fake[:, fake_mid, :],
                        ]
                    )
                )
                s.append(
                    torch.stack(
                        [
                            fake[:, :, fake_mid],
                            fake[:, :, fake_mid],
                            fake[:, :, fake_mid],
                        ]
                    )
                )

                # Append real images for the batch with 3 channels
                sr.append(
                    torch.stack(
                        [
                            g_truth[real_mid, :, :],
                            g_truth[real_mid, :, :],
                            g_truth[real_mid, :, :],
                        ]
                    )
                )
                cr.append(
                    torch.stack(
                        [
                            g_truth[:, real_mid, :],
                            g_truth[:, real_mid, :],
                            g_truth[:, real_mid, :],
                        ]
                    )
                )
                ar.append(
                    torch.stack(
                        [
                            g_truth[:, :, real_mid],
                            g_truth[:, :, real_mid],
                            g_truth[:, :, real_mid],
                        ]
                    )
                )
            # sum lists
            axials += a
            coronals += c
            saggitals += s

            axials_r += ar
            coronals_r += cr
            saggitals_r += sr

        # sums 1 and divide by 2 to transform range [-1, 1] to [0, 1]
        axials = make_grid((torch.stack(axials) + 1) / 2)
        coronals = make_grid((torch.stack(coronals) + 1) / 2)
        saggitals = make_grid((torch.stack(saggitals) + 1) / 2)

        axials_r = make_grid((torch.stack(axials_r) + 1) / 2)
        coronals_r = make_grid((torch.stack(coronals_r) + 1) / 2)
        saggitals_r = make_grid((torch.stack(saggitals_r) + 1) / 2)

        summary.add_image("axials_fake", axials, i)
        summary.add_image("coronals_fake", coronals, i)
        summary.add_image("saggitals_fake", saggitals, i)

        summary.add_image("axials_real", axials_r, i)
        summary.add_image("coronals_real", coronals_r, i)
        summary.add_image("saggitals_real", saggitals_r, i)

        save_image(axials, image_save_path / f"axials_fake_{i}.jpeg")
        save_image(coronals, image_save_path / f"coronals_fake_{i}.jpeg")
        save_image(saggitals, image_save_path / f"saggitals_fake_{i}.jpeg")

        save_image(axials_r, image_save_path / f"axials_real_{i}.jpeg")
        save_image(coronals_r, image_save_path / f"coronals_real_{i}.jpeg")
        save_image(saggitals_r, image_save_path / f"saggitals_real_{i}.jpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str)

    args = parser.parse_args()
    config = Config(args.experiment_path)
    main(config)
