"""
Implements pytorch Datasets for h5 files
"""

import torch
from torch.utils.data import Dataset
import h5py
from lib.io import HDF5FileHandler
from pathlib import Path
import numpy as np


class BaseDataset(
    Dataset
):  # Base Dataset implementing basic info and other stuff for h5
    def __init__(self, data_root, transforms=None, ct_transforms=None):
        super().__init__()
        self.data_root = Path(data_root)
        self.paths = list(self.data_root.rglob("*.h5"))
        self.data = []
        self.transforms = transforms
        self.ct_transforms = ct_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pass


class Real(BaseDataset):
    EXPECTED_KEYS = ["frontal_xray", "lateral_xray"]

    def __init__(self, data_root, transforms=None, ct_transforms=None):
        super().__init__(data_root, transforms, ct_transforms)
        for path in self.paths:
            file = h5py.File(path)
            keys = list(file.keys())
            for key in self.EXPECTED_KEYS:
                if key in keys:
                    self.data.append(path)
                    break

    def __getitem__(self, index):
        file = h5py.File(self.data[index])
        ct = torch.Tensor(np.array(file["ct"]))
        frontal = None
        lateral = None
        if "frontal_xray" in list(file.keys()):
            frontal = torch.Tensor(np.array(file["frontal_xray"]))
        if "lateral_xray" in list(file.keys()):
            lateral = torch.Tensor(np.array(file["lateral_xray"]))
        return ct, frontal, lateral


class Synthetic(BaseDataset):
    EXPECTED_KEYS = ["frontal_drr", "lateral_drr"]

    def __init__(self, data_root, transforms=None, ct_transforms=None):
        super().__init__(data_root, transforms, ct_transforms)
        for path in self.paths:
            file = h5py.File(path)
            keys = list(file.keys())
            for key in self.EXPECTED_KEYS:
                if key in keys:
                    self.data.append(path)
                    break

    def __getitem__(self, index):
        file = h5py.File(self.data[index])
        return file


class Mixed(BaseDataset):
    pass


class RealXRayClassifier(BaseDataset):
    EXPECTED_KEYS = ["frontal_xray", "lateral_xray"]
    classes = ["frontal_xray", "lateral_xray"]

    def __init__(self, data_root, transforms=None, ct_transforms=None):
        super().__init__(data_root, transforms, ct_transforms)
        # takes only patients with both real xray views
        for path in self.paths:
            file = h5py.File(path)
            keys = list(file.keys())
            if self.EXPECTED_KEYS[0] not in keys:
                continue
            if self.EXPECTED_KEYS[1] not in keys:
                continue
            self.data.append((path, "frontal_xray"))
            self.data.append((path, "lateral_xray"))

    def __getitem__(self, index):
        path, key = self.data[index]
        file = HDF5FileHandler(path)

        # shape should be (N, M)
        xray = torch.Tensor(file.retrieve_array(key))

        if self.transforms:
            xray = self.transforms(xray)
        # shape should be (3, 200, 200)
        # Normalize(mean=[x, y, z]) creates channels
        mapping = {"frontal_xray": 0, "lateral_xray": 1}
        target = torch.tensor(mapping[key])
        return xray, target


class SyntheticXRayClassifier(BaseDataset):
    EXPECTED_KEYS = ["frontal_drr", "lateral_drr"]
    classes = ["frontal_drr", "lateral_drr"]

    def __init__(self, data_root, transforms=None, ct_transforms=None):
        super().__init__(data_root, transforms, ct_transforms)
        # takes only patients with both real xray views
        for path in self.paths:
            file = h5py.File(path)
            keys = list(file.keys())
            if self.EXPECTED_KEYS[0] not in keys:
                continue
            if self.EXPECTED_KEYS[1] not in keys:
                continue
            self.data.append((path, "frontal_drr"))
            self.data.append((path, "lateral_drr"))

    def __getitem__(self, index):
        path, key = self.data[index]
        file = HDF5FileHandler(path)

        # shape should be (N, M)
        xray = torch.Tensor(file.retrieve_array(key))

        if self.transforms:
            xray = self.transforms(xray)
        # shape should be (3, 200, 200)
        # Normalize(mean=[x, y, z]) creates channels
        mapping = {"frontal_drr": 0, "lateral_drr": 1}
        target = torch.tensor(mapping[key])
        return xray, target


class MixedXRayClassifier(BaseDataset):
    EXPECTED_KEYS = ["frontal_drr", "lateral_drr"]
    classes = ["frontal_drr", "lateral_drr"]

    def __init__(self, data_root, transforms=None, ct_transforms=None):
        super().__init__(data_root, transforms, ct_transforms)
        # takes only patients with both real xray views
        for path in self.paths:
            file = h5py.File(path)
            keys = list(file.keys())
            if self.EXPECTED_KEYS[0] not in keys:
                continue
            if self.EXPECTED_KEYS[1] not in keys:
                continue
            self.data.append((path, "frontal_drr"))
            self.data.append((path, "lateral_drr"))

    def __getitem__(self, index):
        path, key = self.data[index]
        file = HDF5FileHandler(path)
        keys = list(file.file.keys())
        if key.replace("drr", "xray") in keys:
            key = key.replace("drr", "xray")

        # shape should be (N, M)
        xray = torch.Tensor(file.retrieve_array(key))

        if self.transforms:
            xray = self.transforms(xray)
        # shape should be (3, 200, 200)
        # Normalize(mean=[x, y, z]) creates channels
        mapping = {
            "frontal_drr": 0,
            "lateral_drr": 1,
            "frontal_xray": 0,
            "lateral_xray": 1,
        }
        target = torch.tensor(mapping[key])
        return xray, target


class RealCT(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        set_file: str | Path,
        transforms=None,
        ct_transforms=None,
        strict_coverage=True,
    ):
        super().__init__(data_root, transforms, ct_transforms)
        self.set_file_path = Path(set_file)

        with open(self.set_file_path) as f:
            lines = f.readlines()

        selected_set = [line.rstrip("\r\n") for line in lines]
        self.data = _split_paths(self.data_root, self.set_file_path, selected_set, strict_coverage)
        for path in self.data:
            with HDF5FileHandler(path, mode="read_only") as stored:
                stored.validate_required_keys("real")

    def __getitem__(self, index):
        path = self.data[index]

        with HDF5FileHandler(path, mode="read_only") as file:
            ct = torch.Tensor(file.retrieve_array("ct"))
            xr1 = torch.Tensor(file.retrieve_array("real_xray1"))
            xr2 = torch.Tensor(file.retrieve_array("real_xray2"))

        # 1 channel
        ct = ct.unsqueeze(0)
        xr1 = xr1.unsqueeze(0)
        xr2 = xr2.unsqueeze(0)

        if self.ct_transforms:
            ct = self.ct_transforms(ct)

        if self.transforms:
            xr1 = self.transforms(xr1)
            xr2 = self.transforms(xr2)

        return ct, xr1, xr2, path


class SyntheticCT(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        set_file: str | Path,
        transforms=None,
        ct_transforms=None,
        strict_coverage=True,
    ):
        super().__init__(data_root, transforms, ct_transforms)
        self.set_file_path = Path(set_file)

        with open(self.set_file_path) as f:
            lines = f.readlines()

        selected_set = [line.rstrip("\r\n") for line in lines]
        self.data = _split_paths(self.data_root, self.set_file_path, selected_set, strict_coverage)
        for path in self.data:
            with HDF5FileHandler(path, mode="read_only") as stored:
                stored.validate_required_keys("synthetic")

    def __getitem__(self, index):
        path = self.data[index]

        with HDF5FileHandler(path, mode="read_only") as file:
            ct = torch.Tensor(file.retrieve_array("ct"))
            xr1 = torch.Tensor(file.retrieve_array("xray1"))
            xr2 = torch.Tensor(file.retrieve_array("xray2"))

        # 1 channel and batch = 1
        ct = ct.unsqueeze(0).unsqueeze(0)
        ct = ct
        xr1 = xr1.unsqueeze(0).unsqueeze(0)
        xr1 = xr1
        xr2 = xr2.unsqueeze(0).unsqueeze(0)
        xr2 = xr2

        if self.ct_transforms:
            ct = self.ct_transforms(ct)

        if self.transforms:
            xr1 = self.transforms(xr1)
            xr2 = self.transforms(xr2)

        ct = ct.squeeze().unsqueeze(0)
        xr1 = xr1.squeeze().unsqueeze(0)
        xr2 = xr2.squeeze().unsqueeze(0)

        return ct, xr1, xr2, path


def _split_paths(root, split_file, expected_ids, strict):
    """Support existing flat variant roots and the new per-split output layout."""
    variant, partition = split_file.stem.rsplit("_", 1)
    scoped = root / variant / split_file.stem
    if not scoped.is_dir():
        scoped = root / split_file.stem
    if scoped.is_dir():
        return _resolve_split_paths(sorted(scoped.rglob("ct_xray_data.h5")), expected_ids, strict)
    other_partition = "test" if partition == "train" else "train"
    counterpart = split_file.with_name(f"{variant}_{other_partition}.txt")
    permitted = set()
    if counterpart.is_file():
        permitted = set(counterpart.read_text(encoding="utf-8").splitlines())
    return _resolve_split_paths(
        sorted(root.rglob("ct_xray_data.h5")), expected_ids, strict,
        permitted_extra_ids=permitted,
    )


def _resolve_split_paths(paths, expected_ids, strict=True, *, permitted_extra_ids=()):
    """Resolve HDF5 by patient directory and preserve the literal split order."""
    malformed = [patient_id for patient_id in expected_ids if not patient_id or patient_id.strip() != patient_id]
    if malformed:
        raise ValueError(f"Split contains empty or whitespace-padded IDs: {malformed}")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Split contains duplicate patient IDs")
    by_id = {}
    duplicates = {}
    for path in paths:
        patient_id = path.parent.name
        if patient_id in by_id:
            duplicates.setdefault(patient_id, [str(by_id[patient_id])]).append(str(path))
        else:
            by_id[patient_id] = path
    expected = set(expected_ids)
    found = set(by_id)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected - set(permitted_extra_ids))
    if strict and (missing or unexpected or duplicates):
        raise ValueError(
            "HDF5/split coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )
    return [by_id[patient_id] for patient_id in expected_ids if patient_id in by_id]
