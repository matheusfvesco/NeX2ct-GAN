"""Divide samples into train and test sets using only folder names.

This script no longer inspects HDF5 internal keys. It collects patient/sample
folder names by finding all ``*.h5`` files under ``root_path`` and using the
parent folder name as the sample identifier.

Usage (positional args):
    root_path save_path test_size prefix

Outputs two files written to ``save_path``:
    {prefix}_train.txt
    {prefix}_test.txt

The splitting is reproducible: it seeds Python's ``random`` with 42, shuffles
the (sorted) list of samples, selects the first N as test (N = floor(test_size * n)),
and writes sorted lists to the output files.
"""

import random
from pathlib import Path
import argparse


def main(args):
    # Reproducible shuffle
    random.seed(42)

    SAVE_PATH = Path(args.save_path)
    DATA_ROOT = Path(args.root_path)
    test_size = args.test_size
    prefix = args.prefix

    # Collect sample names from parent folder names of any .h5 files
    samples = []
    for filepath in sorted(list(DATA_ROOT.rglob("*.h5"))):
        samples.append(filepath.parent.name)

    # Unique sample names
    samples = sorted(set(samples))

    # Deterministic shuffle then split
    random.shuffle(samples)
    test_count = int(test_size * len(samples))
    test_set = set(samples[:test_count])
    train_set = set(samples[test_count:])

    # Sanity: disjoint and cover all
    assert test_set.isdisjoint(train_set)
    assert test_set.union(train_set) == set(samples)

    # Write outputs (sorted in output files)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with open(SAVE_PATH / f"{prefix}_train.txt", "w") as f:
        out = sorted(list(train_set))
        lines = [f"{line}\n" for line in out]
        f.writelines(lines)

    with open(SAVE_PATH / f"{prefix}_test.txt", "w") as f:
        out = sorted(list(test_set))
        lines = [f"{line}\n" for line in out]
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path")
    parser.add_argument("save_path")
    parser.add_argument(
        "prefix", help="dataset prefix for output files, e.g. 'synthetic'"
    )
    parser.add_argument("test_size", type=float)
    args = parser.parse_args()
    main(args)
