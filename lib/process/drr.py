"""
Functions for generating DRRs
"""

from __future__ import annotations

import numpy as np
import torch
from diffdrr.drr import DRR
import cv2
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.dicom import CTScanDirectory
from lib.process.preprocess import CTProcessor


from lib.utils.utils import degrees_to_radians

RESIZE_SIZE = 256


class DRRGenerator:
    def __init__(
        self,
        ctdirectory: CTScanDirectory,
        custom_process_order: list[str] = None,
        **kwargs,
    ):
        self.dicom = ctdirectory.dicom

        self.spacing = np.array(list(ctdirectory.spacing))
        if not custom_process_order:
            #self.volume = ctdirectory.get_array()
            processor = CTProcessor(array=ctdirectory.get_raw_array(**kwargs), scan=ctdirectory)
            self.volume = processor.process()
            self.spacing = processor.spacing
        else:
            if not kwargs:  # if no kwargs provided, allows the use of cache
                array = ctdirectory.get_raw_array(**kwargs)
            else:  # if kwargs provided, bypass the cache, reads it from disk again and process directly from raw array
                array = ctdirectory._get_raw_array(**kwargs)
            # pass full directory so custom processing functions can
            # inspect scan-level metadata if required
            self.volume = CTProcessor(array=array, scan=ctdirectory)._process_compose(
                custom_process_order
            )
        # TODO: replace spacing being taken from ctdirectory to instead use the new spacing from the CTProcessor 
        self.frontal_view = self.generate_drr_from_string("frontal")
        self.lateral_view = self.generate_drr_from_string("lateral")

    def show_drr_by_string(self, view_type: str = "frontal", window_name: str = ""):
        if not isinstance(view_type, str):
            raise ValueError("view_type must be a string.")

        if view_type == "frontal":
            array = self.frontal_view
        elif view_type == "lateral":
            array = self.lateral_view
        else:
            raise ValueError('view_type must be either "frontal" or "lateral".')

        self._show_array(array=array, window_name=window_name)

    def _show_array(self, array: np.ndarray, window_name: str = ""):
        if not window_name or not isinstance(window_name, str):
            window_name = "DRR"

        cv2.imshow(window_name, array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write_drr_by_string(self, path: str | Path, view_type: str = "frontal"):
        path = Path(path)

        if not isinstance(view_type, str):
            raise ValueError("view_type must be a string.")

        if view_type == "frontal":
            array = self.frontal_view
        elif view_type == "lateral":
            array = self.lateral_view
        else:
            raise ValueError('view_type must be either "frontal" or "lateral".')

        self._write_drr(array=array, path=path)

    def _write_drr(self, array: np.ndarray, path: str | Path):
        path = Path(path)
        if not path.suffix:
            raise ValueError("Provided path is not a file.")

        path.parent.mkdir(parents=True, exist_ok=True)

        self._write_array_to_disk(array=array, path=path)

    @staticmethod
    def _write_array_to_disk(array: np.ndarray, path: Path):
        cv2.imwrite(str(path), array)

    @staticmethod
    def _generate_raw_drr(
        volume: np.ndarray,
        spacing,
        height: int,
        patch_size: int,
        theta: float,
        phi: float,
        gamma: float,
        sdr: float,
    ) -> torch.tensor:
        """
        Returns a reconstructed DRR from an axial view CT with any config.

        Args:
            volume (np.ndarray): The CT volume.
            spacing: The pixel spacing of the dcm.
            theta (float): The theta(x-y plane) angle in radians.
            phi (float): The phi(-z) angle in radians.
            gamma (float): The gamma(-z) angle in radians.
            sdr (float): The source-detector distance.

        Returns:
            torch.tensor: A tensor representing the DRR.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Get parameters for the detector
        bx, by, bz = torch.tensor(volume.shape) * torch.tensor(spacing) / 2
        detector_kwargs = {
            "sdr": sdr,
            "theta": theta,
            "phi": phi,
            "gamma": gamma,
            "bx": bx,
            "by": by,
            "bz": bz,
        }

        # for future diffdrr versions:
        # 3D numpy to tensor
        # tensor = torch.from_numpy(volume)
        # tio.ScalarImage expects a 4D (C, W, H, D) tensor
        # tensor = torch.unsqueeze(tensor, 0)
        # subject = tio.Subject(t1=tio.ScalarImage(tensor=tensor))

        # Make the DRR
        drr = DRR(
            volume, spacing, sdr=sdr, height=height, patch_size=patch_size, delx=4.0
        ).to(device)

        rotations = torch.tensor([[theta, phi, gamma]], device=device)
        translations = torch.tensor([[bx, by, bz]], device=device)

        img: torch.Tensor = drr(rotations, translations)

        return img

    def generate_drr(self) -> torch.tensor:
        """
        Returns a reconstructed DRR from an axial view CT using default configs.

        Args:
            volume (np.ndarray): The CT volume.
            spacing: The pixel spacing of the dcm.

        Returns:
            torch.tensor: A tensor representing the DRR.
        """
        # Default configs
        theta = degrees_to_radians(180)  # (8 / 8) * torch.pi
        phi = degrees_to_radians(0)  # 0
        sdr = 300.0
        gamma = degrees_to_radians(90)  # torch.pi / 2

        # Generate DRR with default configs
        img = self._generate_raw_drr(self.volume, self.spacing, theta, phi, gamma, sdr)

        return img

    def generate_drr_from_string(self, view_type: str) -> np.ndarray:
        if view_type == "frontal":
            theta = degrees_to_radians(180)  # (8 / 8) * torch.pi
            phi = degrees_to_radians(0)  # 0
            gamma = degrees_to_radians(90)  # torch.pi / 2
        elif view_type == "lateral":
            theta = degrees_to_radians(270)  # (8 / 8) * torch.pi
            phi = degrees_to_radians(0)  # 0
            gamma = degrees_to_radians(90)  # torch.pi / 2
        else:
            raise ValueError(
                """Invalid value for view_label, should be 
                            'frontal' or 'lateral'"""
            )

        sdr = 300.0
        height = 200
        patch_size = None

        img = self._generate_raw_drr(
            volume=self.volume,
            spacing=self.spacing,
            sdr=sdr,
            theta=theta,
            phi=phi,
            gamma=gamma,
            height=height,
            patch_size=patch_size,
        )

        img = img.squeeze().cpu().numpy()
        img = cv2.resize(img, dsize=(RESIZE_SIZE, RESIZE_SIZE))
        img = cv2.normalize(
            img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return img
