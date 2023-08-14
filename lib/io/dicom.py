"""
Functions for reading different types of dicom data
"""
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pydicom


def load_first_dcm(pathdir: Path) -> pydicom.Dataset:
    """
    Load the first DICOM file in a directory.

    Args:
        pathdir (Path): The directory path containing the DICOM files.

    Returns:
        pydicom.Dataset: The first DICOM file in the directory (sorted alphabetically by filename).
    """
    pathdir = Path(pathdir)
    dcmfiles = sorted(list(pathdir.glob("*.dcm")))
    dcm = pydicom.dcmread(dcmfiles[0])
    return dcm


def load_middle_dcm(pathdir: Path) -> pydicom.Dataset:
    """
    Load the middle DICOM file in a directory.

    Args:
        pathdir (Path): The directory path containing the DICOM files.

    Returns:
        pydicom.Dataset: The middle DICOM file in the directory (sorted alphabetically by filename).
    """
    pathdir = Path(pathdir)
    dcmfiles = sorted(list(pathdir.glob("*.dcm")))
    dcm = pydicom.dcmread(dcmfiles[len(dcmfiles) // 2])
    return dcm


def load_last_dcm(pathdir: Path) -> pydicom.Dataset:
    """
    Load the last DICOM file in a directory.

    Args:
        pathdir (Path): The directory path containing the DICOM files.

    Returns:
        pydicom.Dataset: The last DICOM file in the directory (sorted alphabetically by filename).
    """
    pathdir = Path(pathdir)
    dcmfiles = sorted(list(pathdir.glob("*.dcm")))
    dcm = pydicom.dcmread(dcmfiles[-1])
    return dcm


def load_slices(pathdir: Path) -> Tuple[np.ndarray, List[float], pydicom.Dataset]:
    """
    Read a directory of DICOM files and return the volume and voxel spacings.

    Args:
        pathdir (Path): The path to a directory containing the DICOM files.

    Returns:
        Tuple[np.ndarray, List[float], pydicom.Dataset]: A tuple containing the volume as a numpy array, the voxel spacings as a list of floats, and a pydicom Dataset object for the middle slice of the volume.
    """
    pathdir = Path(pathdir)

    # Get all DICOM files in the directory and sort them
    dcmfiles = sorted(list(pathdir.glob("*.dcm")))

    # Read the first DICOM file to get the image dimensions and pixel spacing
    ds = pydicom.dcmread(dcmfiles[0])
    nx, ny = ds.pixel_array.shape
    nz = len(dcmfiles)
    del_x, del_y = map(float, ds.PixelSpacing)

    # Create an empty 3D numpy array to store the volume
    volume = np.zeros((nx, ny, nz), dtype=np.float32)

    # Loop through all DICOM files and add them to the volume
    del_zs = []
    for idx, dcm in enumerate(dcmfiles):
        ds = pydicom.dcmread(dcm)
        volume[:, :, idx] = ds.pixel_array
        del_zs.append(ds.ImagePositionPatient[2])

    # Calculate the z-spacing
    del_zs = np.diff(del_zs)
    del_z = float(np.abs(np.unique(del_zs)[0]))

    # Create a list of voxel spacings
    spacing = [del_x, del_y, del_z]

    # Load middle slice dcm
    dcm = pydicom.dcmread(dcmfiles[len(dcmfiles) // 2])

    return volume, spacing, dcm


def read_xr(pathdir: Union[str, Path]) -> np.ndarray:
    """
    Reads an x-ray image from the specified path and returns it as a numpy array.

    Args:
        pathdir (Union[str, Path]): The path to the directory containing the x-ray image.

    Returns:
        np.ndarray: The x-ray image as a numpy array.
    """
    # TODO: check if dir has more than one .dcm file
    # Get the first DICOM file in the directory
    dicom_file = next(Path(pathdir).rglob("*.dcm"))

    # Read the DICOM file and return the pixel array
    ds = pydicom.dcmread(str(dicom_file))
    return ds.pixel_array
