from lib.datasets import RealCT, SyntheticCT
from pathlib import Path
from lib.models.convnext import ConvnextBlock, Convnext3dBlock
from lib.models.simple_block import SimpleBlock, Simple3dBlock
from lib.models.generator import (
    DoubleCnnGenerator,
    DoubleLinkerGenerator,
    SingleLinkerGenerator,
    SingleLinkerNonUnetGenerator
)
import torch


def block_factory(block_name):
    if block_name == "convnext":
        return ConvnextBlock, Convnext3dBlock
    if block_name == "simple":
        return SimpleBlock, Simple3dBlock
    raise ValueError(
        f"Expected block to be 'simple' or 'convnext', but received block {block_name}"
    )


def generator_factory(gen_name):
    if gen_name == "double_cnn":
        return DoubleCnnGenerator
    if gen_name == "double_linker":
        return DoubleLinkerGenerator
    if gen_name == "single_linker":
        return SingleLinkerGenerator
    if gen_name == "single_linker_non_unet":
        return SingleLinkerNonUnetGenerator


def CTFactory(dataset_type, data_root, set_root, transforms: None, ct_transforms: None):
    data_root = Path(data_root)
    set_root = Path(set_root)
    if dataset_type == "real":
        train = RealCT(
            data_root=data_root,
            set_file=set_root / "train_real.txt",
            transforms=transforms,
            ct_transforms=ct_transforms,
        )
        test = RealCT(
            data_root=data_root,
            set_file=set_root / "test_real.txt",
            transforms=transforms,
            ct_transforms=ct_transforms,
        )
        return train, test

    if dataset_type == "synthetic":
        train = SyntheticCT(
            data_root=data_root,
            set_file=set_root / "train_synthetic.txt",
            transforms=transforms,
            ct_transforms=ct_transforms,
        )
        test = SyntheticCT(
            data_root=data_root,
            set_file=set_root / "test_synthetic.txt",
            transforms=transforms,
            ct_transforms=ct_transforms,
        )
        return train, test
    if dataset_type == "same":
        # uses same dataset for both train and test.
        # useful when quickly testing the model
        train = SyntheticCT(
            data_root=data_root,
            set_file=set_root / "test_synthetic.txt",
            transforms=transforms,
            ct_transforms=ct_transforms,
        )
        test = SyntheticCT(
            data_root=data_root,
            set_file=set_root / "test_synthetic.txt",
            transforms=transforms,
            ct_transforms=ct_transforms,
        )
        return train, test

def optim_factory(optim):
    if optim == "adam":
        return torch.optim.Adam
    if optim == "adamw":
        return torch.optim.AdamW
    raise ValueError(f"Optim must be one of adam or adamw, but instead got {optim}")