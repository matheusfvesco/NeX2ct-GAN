from lib.datasets import RealCT, SyntheticCT
import pytest
from lib.transforms import Normalize

@pytest.mark.parametrize(
    ("data_root", "set"),
    [
        ("output/preprocess/GAN-dataset-complete/real/real_test", "splits/real_test.txt"),
        ("output/preprocess/GAN-dataset-complete/real/real_train", "splits/real_train.txt"),
    ],
)
def test_real_dataset(data_root, set):
    dataset = RealCT(
        data_root=data_root,
        set_file=set,
        transforms=Normalize(),
        ct_transforms=Normalize()
    )
    for ct, xr1, xr2, path in dataset:
        c, h, w, d = ct.size()
        assert c == 1
        assert h == w
        assert w == d
        assert h == 256
        assert ct.max() == 1.0
        assert ct.min() == 0.0

        c, h, w = xr1.size()
        assert c == 1
        assert h == w
        assert h == 256
        assert xr1.max() == 1.0
        assert xr2.min() == 0.0

        c, h, w = xr2.size()
        assert c == 1
        assert h == w
        assert h == 256
        assert xr2.max() == 1.0
        assert xr2.min() == 0.0

@pytest.mark.parametrize(
    ("data_root", "set"),
    [
        ("output/preprocess/GAN-dataset-complete/synthetic/synthetic_test", "splits/synthetic_test.txt"),
        ("output/preprocess/GAN-dataset-complete/synthetic/synthetic_train", "splits/synthetic_train.txt"),
    ],
)
def test_synthetic_dataset(data_root, set):
    dataset = SyntheticCT(
        data_root=data_root,
        set_file=set,
        transforms=Normalize(),
        ct_transforms=Normalize()
    )
    for ct, xr1, xr2, path in dataset:
        c, h, w, d = ct.size()
        assert c == 1
        assert h == w
        assert h == 256
        assert ct.max() == 1.0
        assert ct.min() == 0.0

        c, h, w = xr1.size()
        assert c == 1
        assert h == w
        assert h == 256
        assert xr1.max() == 1.0
        assert xr2.min() == 0.0

        c, h, w = xr2.size()
        assert c == 1
        assert h == w
        assert h == 256
        assert xr2.max() == 1.0
        assert xr2.min() == 0.0
