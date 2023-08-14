"""
Crawls through the data and returns lists or dictionaries
for dicom

Finds pairs
"""
import os
from pathlib import Path

from lib.utils.utils import (
    get_patient_id_from_path,
    is_same_patient,
    remove_file_extension,
    remove_ricord_prefix,
)
from lib.utils.dicom import get_folder_modality
from pprint import pprint

from typing import Dict


def group_cts_patient(
    path, verbose=False
):  # Returns dict with patient name being the key, with all the CTs for that patient
    path = Path(path)

    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        # ignores main folder and empty dirs
        if filenames == []:
            continue
        if filenames == ["LICENSE"]:
            continue

        # gets patient id from path and removes dataset prefix
        patient_id = remove_ricord_prefix(get_patient_id_from_path(dirpath))

        if patient_id not in out:  # if first ct for patient, creates key with list
            out[patient_id] = []
            out[patient_id].append(dirpath)
        else:  # if not, appends to existing list
            out[patient_id].append(dirpath)

    if verbose:
        pprint(out)
    return out


def group_xrs_patient(
    path, verbose=False
):  # Returns dict with patient name being the key, with all the CTs for that patient
    path = Path(path)

    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        # ignores main folder and empty dirs
        if filenames == []:
            continue
        if filenames == ["LICENSE"]:
            continue

        # gets patient id from path and removes dataset prefix
        patient_id = remove_ricord_prefix(get_patient_id_from_path(dirpath))

        if patient_id not in out:  # if first ct for patient, creates key with list
            out[patient_id] = []
            out[patient_id].append(dirpath)
        else:  # if not, appends to existing list
            out[patient_id].append(dirpath)

    if verbose:
        pprint(out)
    return out


def group_lidc_patients(path, verbose=False):
    path = Path(path)

    out = {}
    for dirpath, dirnames, filenames in os.walk(path):
        # ignores main folder and empty dirs
        if filenames == []:
            continue
        if filenames == ["LICENSE"]:
            continue

        patient_id = get_patient_id_from_path(dirpath)

        if patient_id not in out:  # if first ct for patient, creates key with list
            out[patient_id] = {"cts": [], "xrs": []}
            if get_folder_modality(dirpath) != "CT":
                out[patient_id]["xrs"].append(dirpath)
            else:
                out[patient_id]["cts"].append(dirpath)
        else:  # if not, appends to existing list
            if get_folder_modality(dirpath) != "CT":
                out[patient_id]["xrs"].append(dirpath)
            else:
                out[patient_id]["cts"].append(dirpath)
    if verbose:
        pprint(out)
    return out


def filter_paired(original: Dict):
    """
    Returns original dict and a dict with only paired data
    """
    out = {}  # should have 290 patients with paired data for lidc and 64 for ricord
    for key in original:
        if original[key]["cts"] != [] and original[key]["xrs"] != []:
            out[key] = original[key]
    return original, out


def filter_cts():  # For a given patient, drops any CTs that don't follow the filter. Returns original and filtered
    pass


def filter_xrs():  # For a given patient, drops any XRs that don't follow the filter
    pass


def merge_patient_exams(
    ct_dirs, xr_dirs
):  # Takes the two dicts and merges CTs and XRs data for that patient, returning dict with all patients and their exams
    out = {}
    for patient in ct_dirs:
        out[patient] = {"cts": ct_dirs[patient], "xrs": []}
    for patient in xr_dirs:
        if patient in out:
            out[patient]["xrs"] = xr_dirs[patient]
        else:
            out[patient] = {"cts": [], "xrs": xr_dirs[patient]}
    """out = {}
    for key in ct_dirs:
        if key in xr_dirs:
            out[key] = {"cts": ct_dirs[key], "xrs": xr_dirs[key]}
        else:
            out[key] = {"cts": ct_dirs[key], "xrs": []}
    for key in xr_dirs:
        if key in xr_dirs:
            continue
        else:
            out[key] = {"cts": [], "xrs": xr_dirs[key]}"""

    # pprint(out)
    # print(len(out))
    return out


def filter_paired_ricord(original: Dict):
    """
    Returns original dict and a dict with only paired data
    """
    out = {}
    for key in original:
        if original[key]["cts"] != [] and original[key]["xrs"] != []:
            out[key] = original[key]
    return original, out
