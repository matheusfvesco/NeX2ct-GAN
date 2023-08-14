"""
Functions for generating DRRs
"""

import numpy as np
import torch
from diffdrr.drr import DRR

from lib.utils.utils import degrees_to_radians


def gen_drr_raw(
    volume: np.ndarray, spacing, theta: float, phi: float, gamma: float, sdr: float
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

    # Make the DRR
    drr = DRR(volume, spacing, height=200, delx=4.0).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    img = drr.project(**detector_kwargs)
    return img


def gen_default_drr(volume: np.ndarray, spacing) -> torch.tensor:
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
    img = gen_drr_raw(volume, spacing, theta, phi, gamma, sdr)
    return img
