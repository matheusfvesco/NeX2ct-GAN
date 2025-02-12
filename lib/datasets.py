"""
Implements pytorch Datasets for h5 files
"""
import torch
from torch.utils.data import Dataset
import h5py
from lib.io import HDF5FileHandler
from pathlib import Path
import numpy as np


class BaseDataset(Dataset):  # Base Dataset implementing basic info and other stuff for h5
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
        mapping = {
            "frontal_xray": 0,
            "lateral_xray": 1
        }
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
        mapping = {
            "frontal_drr": 0,
            "lateral_drr": 1
        }
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