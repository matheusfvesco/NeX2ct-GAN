"""
Divides into train and test sets. Ensures synthetic test is defined by real_test + n% synthetic
"""

from lib.io import HDF5FileHandler
import random
from pathlib import Path
import argparse

def main(args):
    random.seed(42)
    #SAVE_PATH = Path("splits")
    SAVE_PATH = Path(args.save_path)
    #DATA_ROOT = Path("preprocess/GAN-dataset")
    DATA_ROOT = Path(args.root_path)

    #test_size = 0.05
    test_size = args.test_size


    paired = []
    complete = []
    for filepath in sorted(list(DATA_ROOT.rglob("*.h5"))):
        file = HDF5FileHandler(filepath)
        keys = list(file.file.keys())
        complete.append(filepath.name)
        if "real_xray1" in keys and "real_xray2" in keys:
            paired.append(filepath.name)

    paired = set(paired)
    complete = set(complete)

    synthetic_only = complete.difference(paired)

    paired_list = sorted(list(paired))
    random.shuffle(paired_list)
    test_real_size = int(test_size * len(paired_list))
    test_real = set(paired_list[:test_real_size])
    train_real = set(paired_list[test_real_size:])

    synthetic_list = sorted(list(synthetic_only))
    random.shuffle(synthetic_list)
    test_synthetic_size = int(test_size * len(synthetic_list))
    test_synthetic_part = set(synthetic_list[:test_synthetic_size])

    # test synthetic = test_real + n% of synthetic samples
    test_synthetic = test_real.union(test_synthetic_part)


    train_synthetic = complete - test_synthetic

    assert test_real.issubset(test_synthetic)


    with open(SAVE_PATH/"train_real.txt", "w") as f:
        samples = sorted(list(train_real))
        lines = [f"{line}\n" for line in samples]
        f.writelines(lines)

    with open(SAVE_PATH/"test_real.txt", "w") as f:
        samples = sorted(list(test_real))
        lines = [f"{line}\n" for line in samples]
        f.writelines(lines)

    with open(SAVE_PATH/"train_synthetic.txt", "w") as f:
        samples = sorted(list(train_synthetic))
        lines = [f"{line}\n" for line in samples]
        f.writelines(lines)

    with open(SAVE_PATH/"test_synthetic.txt", "w") as f:
        samples = sorted(list(test_synthetic))
        lines = [f"{line}\n" for line in samples]
        f.writelines(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path")
    parser.add_argument("save_path")
    parser.add_argument("test_size",type=float)
    args = parser.parse_args()
    main(args)