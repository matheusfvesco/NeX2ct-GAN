from lib.datasets import RealCT, SyntheticCT
import pytest

@pytest.mark.parametrize("set",["splits/test_real.txt", "splits/train_real.txt"])
def test_real_dataset(set):
    dataset = RealCT(
        data_root="preprocess/GAN-dataset",
        set_file=set,
    )
    for ct, xr1, xr2, path in dataset:
        h, w, d = ct.size()
        assert h == w
        assert h == 512

@pytest.mark.parametrize("set",["splits/test_synthetic.txt", "splits/train_synthetic.txt"])
def test_real_dataset(set):
    dataset = SyntheticCT(
        data_root="preprocess/GAN-dataset",
        set_file=set,
    )
    for ct, xr1, xr2, path in dataset:
        h, w, d = ct.size()
        assert h == w
        assert h == 512