"""
Dicom utilities, like checking tags
"""

import os
from pathlib import Path
from typing import Optional, Union

import pydicom
from pydicom import FileDataset
from pydicom.dataelem import DataElement
from pydicom.multival import MultiValue
from pydicom.valuerep import DSfloat

from lib.io.dicom import load_middle_dcm


def get_tag(file_name: str, tag: Union[str, int]) -> DataElement:
    """Retrieves the value of a specific DICOM tag from a DICOM file.

    Args:
        file_name (str): The name of the DICOM file.
        tag (Union[str, int]): The tag to retrieve. Can be specified as a string or an integer.

    Returns:
        DataElement: The value of the specified tag.
    """
    ds = pydicom.dcmread(file_name)
    element = ds[tag]
    return element


def get_image_plane(dcm: FileDataset, verbose: bool = False) -> Optional[str]:
    """Returns a string with the plane listed on the DICOM tags.

    Args:
        dcm (FileDataset): The DICOM file.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        Optional[str]: The plane of the image. Returns 'cor' if the image is on the coronal plane,
        'sag' if the image is on the saggital plane, 'ax' if the image is on the axial plane,
        and 'scout' if the image is a scout image. Returns None if orientation couldn't be found.
    """

    # gets ImageOrientationPatient parameter(list)
    orientation = dcm.ImageOrientationPatient

    # converts orien metadata to hardcoded if statements format
    orientation = [str(int(item)) for item in orientation]

    if orientation == ["0", "1", "0", "0", "0", "0"]:  # exception
        return "sag"

    if orientation == ["0", "1", "0", "0", "0", "-1"]:
        return "sag"

    if orientation == ["1", "0", "0", "0", "0", "0"]:
        return "cor"

    if orientation == ["1", "0", "0", "0", "0", "-1"]:
        return "cor"

    if orientation == ["1", "0", "0", "0", "1", "0"]:
        return "ax"

    if orientation == ["0", "0", "0", "0", "0", "0"]:
        return "ax"

    if orientation == ["0", "-1", "0", "0", "0", "-1"]:  # scout
        return "scout"

    if verbose:
        print(dcm.PatientName)
        print(orientation)

    return None


def get_image_modality(dcm: FileDataset, verbose: bool = False) -> Optional[str]:
    if verbose:
        print(dcm.Modality)
    return dcm.Modality


def get_folder_modality(path: Path, verbose: bool = False) -> Optional[str]:
    path = Path(path)
    dcm = load_middle_dcm(path)
    modality = get_image_modality(dcm)

    if verbose:
        print(path)

    return modality


def get_folder_plane(path: Path, verbose: bool = False) -> Optional[str]:
    """Takes a path, searches for .dcm files in the path, and returns the image orientation of the first .dcm file.

    Args:
        path (Path): The path to search for .dcm files.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        Optional[str]: The plane of the image. Returns 'cor' if the image is on the coronal plane,
        'sag' if the image is on the saggital plane, 'ax' if the image is on the axial plane,
        and 'scout' if the image is a scout image. Returns None if orientation couldn't be found.
    """
    dcm = load_middle_dcm(path)
    plane = get_image_plane(dcm, verbose)
    if plane == None:
        if verbose:
            print(path)

    return plane


def get_image_window(dcm: FileDataset, verbose: bool = False) -> Optional[str]:
    """Returns the window type of a DICOM image.

    Args:
        dcm (FileDataset): The DICOM file.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        Optional[str]: The window type of the image. Returns 'lung' if the window is a lung acquisition,
        'standard' if the window is a regular thorax CT, and None otherwise.
    """
    window_center = dcm.WindowCenter
    window_width = dcm.WindowWidth

    if verbose:
        print(window_center, window_width)
        print(type(window_center), type(window_width))

    if isinstance(window_center, DSfloat):
        return "lung"

    if isinstance(window_center, MultiValue):
        return "standard"

    return None


def get_folder_window(path: Path, verbose: bool = False) -> Optional[str]:
    """Takes a path, searches for .dcm files in the path, and returns the window type of the first .dcm file.

    Args:
        path (Path): The path to search for .dcm files.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        Optional[str]: The window type of the image. Returns 'lung' if the window center is a DSfloat,
        'standard' if the window center is a MultiValue, and None otherwise.
    """
    dcm = load_middle_dcm(path)
    window = get_image_window(dcm, verbose)
    if window == None:
        if verbose:
            print(path)

    return window


def is_ct(dcm: FileDataset) -> bool:
    """Determines if a given DICOM file represents a CT image.

    Args:
        dcm (FileDataset): A pydicom FileDataset object representing a DICOM file.

    Returns:
        bool: True if the DICOM file represents a CT image, False otherwise.
    """
    # Check if the Modality tag is set to 'CT'
    return dcm.Modality == "CT"


def is_lung_window(dcm: FileDataset) -> bool:
    """Determines if a given DICOM file has a lung window.

    Args:
        dcm (FileDataset): A pydicom FileDataset object representing a DICOM file.

    Returns:
        bool: True if the DICOM file has a lung window, False otherwise.
    """
    # Use the get_image_window function to determine the window type of the image
    window = get_image_window(dcm)

    # Return True if the window is 'lung', False otherwise
    return window == "lung"


def is_standard_window(dcm: FileDataset) -> bool:
    """Determines if a given DICOM file has a standard window.

    Args:
        dcm (FileDataset): A pydicom FileDataset object representing a DICOM file.

    Returns:
        bool: True if the DICOM file has a standard window, False otherwise.
    """
    # Use the get_image_window function to determine the window type of the image
    window = get_image_window(dcm)

    # Return True if the window is 'standard', False otherwise
    return window == "standard"


def is_axial_ct(dcm: FileDataset) -> bool:
    """
    Determines if a given DICOM file represents an axial CT image.

    Args:
        dcm (FileDataset): A pydicom FileDataset object representing a DICOM file.

    Returns:
        bool: True if the DICOM file represents an axial CT image, False otherwise.
    """
    # Use the get_image_plane function to determine the plane of the image
    plane = get_image_plane(dcm)

    # Return True if the plane is 'ax', False otherwise
    return plane == "ax" and is_ct(dcm)


def is_standard_axial_ct(dcm: FileDataset) -> bool:
    """Determines if a given DICOM file represents a standard axial CT image.

    Args:
        dcm (FileDataset): A pydicom FileDataset object representing a DICOM file.

    Returns:
        bool: True if the DICOM file represents a standard axial CT image, False otherwise.
    """
    # Return True if the plane is 'ax' and the window is 'standard', False otherwise
    return is_standard_window(dcm) and is_axial_ct(dcm)


def is_lung_axial_ct(dcm: FileDataset) -> bool:
    """Determines if a given DICOM file represents a lung axial CT image.

    Args:
        dcm (FileDataset): A pydicom FileDataset object representing a DICOM file.

    Returns:
        bool: True if the DICOM file represents a lung axial CT image, False otherwise.
    """
    # Return True if the plane is 'ax' and the window is 'lung', False otherwise
    return is_lung_window(dcm) and is_axial_ct(dcm)
