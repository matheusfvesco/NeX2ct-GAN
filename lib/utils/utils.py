"""
More general utilities, like string formatting
"""

import math
import os
from pathlib import Path
from typing import Dict, Optional, Union


def remove_file_extension(file_name: str) -> str:
    """Removes the file extension from a given file name.

    Args:
        file_name (str): The name of the file, including its extension.

    Returns:
        str: The file name without its extension.
    """
    return os.path.splitext(file_name)[0]


def remove_ricord_prefix(ricord_string: str) -> str:
    """Removes the Ricord dataset prefix from a given string.

    Args:
        ricord_string (str): A string in the format "MIDRC-RICORD-1C-patient-id" or "MIDRC-RICORD-1A-patient-id" or "patient-id".

    Returns:
        str: The string without the "MIDRC-RICORD-1#-" prefix or the string unchanged.
    """
    if ricord_string.startswith("MIDRC-RICORD-1"):
        return ricord_string[16:]
    else:
        return ricord_string


def is_same_patient(patient_name_1: str, patient_name_2: str) -> bool:
    """Checks if two raw patient names are equal. Useful when checking if patient is in both RICORD datasets.

    Args:
        patient_name_1 (str): The first raw patient name in the format "MIDRC-RICORD-1C-patient-id" or "MIDRC-RICORD-1A-patient-id" or "patient-id".
        patient_name_2 (str): The second raw patient name in the same format as `patient_name_1`.

    Returns:
        bool: True if the two raw patient names are equal after removing the prefix, False otherwise.
    """
    return remove_ricord_prefix(patient_name_1) == remove_ricord_prefix(patient_name_2)


def degrees_to_radians(angle_degrees: Union[int, float]) -> float:
    """
    Converts an angle from degrees to radians.

    Args:
        angle_degrees (Union[int, float]): The angle in degrees.

    Returns:
        float: The equivalent angle in radians.
    """
    return math.radians(angle_degrees)


def is_paired_patient(
    data_dict: Dict,
):  # checks if dictionary contains ct and xr values, or only one of them. Returns true if contains both.
    pass


def get_patient_id_from_path(path: str) -> Optional[str]:
    """
    Extracts and returns the patient ID from a given path, not removing the dataset prefix.

    Args:
        path (str): The path to extract the patient ID from.

    Returns:
        Optional[str]: The extracted patient ID, or None if not found.
    """
    path = Path(path)

    for part in path.parts:
        if "MIDRC-RICORD-1A-" in part or "MIDRC-RICORD-1C-" in part:
            return part
        if "LIDC-IDRI-" in part:
            return part

    return None
