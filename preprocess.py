"""
Preprocesses and filters the datasets of CT scans and X-rays.

This script filters a dataset of patients with associated CT scans and/or X-rays,
and writes summary statistics and individual patient data to disk in CSV and HDF5 formats.
"""

from lib.dicom import (
    CTScanDirectory,
    SingleViewXRayDirectory,
    MultiviewXRayDirectory,
)
from lib.io import PatientDataStorer
from lib.data.dicom import DicomDataset
from lib.logging import setup_logger
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import csv

# Setup logger
logger = setup_logger(
    logger_name="preprocess",
    experiment_name="creating gan dataset",
    filename_comment="preprocess",
)

DATA_ROOT = Path("data")
SAVE_ROOT = Path("output/preprocess/GAN-dataset")
DATASETS = ["MIDRC-RICORD-1A", "MIDRC-RICORD-1C", "LIDC-IDRI"]

SAVE_ROOT.mkdir(parents=True, exist_ok=True)

dataset = DicomDataset(root_dir=DATA_ROOT, accepted_datasets=DATASETS, logger=logger)

dataset.load()

# Prints and writes Patient and basic data availability for each patient on disk
# before filtering them
counts = defaultdict(int)
data_summary = []
logger.info("Acquiring per patient statistics...")

for patient in dataset:
    ct_scans: list[CTScanDirectory] = dataset[patient]["cts"]
    singleview_dirs: list[SingleViewXRayDirectory] = dataset[patient]["xrays"]
    multiview_dirs: list[MultiviewXRayDirectory] = dataset[patient]["multiview"]
    data_summary.append(
        {
            "patient": patient,
            "total_number_of_cts": len(ct_scans),  # CTs
            "number_of_valid_cts": len([x for x in ct_scans if x.is_valid_scan]),
            "number_of_valid_axial_cts": len(
                [x for x in ct_scans if x.is_valid_axial_scan]
            ),
            "number_of_axial_cts": len([x for x in ct_scans if x.is_axial]),
            # singleviews
            "total_number_of_singleview_xrays": len(singleview_dirs),
            "number_of_frontal_singleview_xrays": len(
                [x for x in singleview_dirs if x.is_frontal]
            ),
            "number_of_lateral_singleview_xrays": len(
                [x for x in singleview_dirs if x.is_lateral]
            ),
            # multiviews
            "total_number_of_multiview_xrays": len(multiview_dirs),
            "number_of_both_views_available_multiview_xrays": len(
                [x for x in multiview_dirs if x.has_both_views]
            ),
            "number_of_frontal_multiview_xrays": len(
                [x for x in multiview_dirs if x.has_frontal]
            ),
            "number_of_lateral_multiview_xrays": len(
                [x for x in multiview_dirs if x.has_lateral]
            ),
            "is_paired_patient": True
            if ct_scans and (singleview_dirs or multiview_dirs)
            else False,
        }
    )
logger.info("Finished!\n\n")

logger.info(
    f'Writing per patient summary to {SAVE_ROOT.parent / "raw_data_summary.csv"}...'
)
with open(SAVE_ROOT.parent / "raw_data_summary.csv", "w") as f:
    writer = csv.DictWriter(f=f, fieldnames=data_summary[0].keys(), delimiter=";")
    writer.writeheader()
    writer.writerows(data_summary)
logger.info("Success!\n\n")

logger.debug("Counting full dataset statistics for printing...")
for patient in dataset:
    counts["total_patients"] += 1
    ct_scans: list[CTScanDirectory] = dataset[patient]["cts"]
    singleview_dirs: list[SingleViewXRayDirectory] = dataset[patient]["xrays"]
    multiview_dirs: list[MultiviewXRayDirectory] = dataset[patient]["multiview"]

    if ct_scans and (singleview_dirs or multiview_dirs):
        counts["paired_patients"] += 1

    if singleview_dirs or multiview_dirs:
        counts["at_least_one_xray"] += 1
        if singleview_dirs and multiview_dirs:
            counts["has_singleview_and_multiview"] += 1

        if len(singleview_dirs) > 1:
            counts["more_than_one_singleview"] += 1

        if len(multiview_dirs) > 1:
            counts["more_than_one_multiview"] += 1

    if ct_scans:
        counts["at_least_one_ct"] += 1
        if len(ct_scans) > 1:
            counts["more_than_one_CT"] += 1

    if ct_scans and "LIDC-IDRI" in ct_scans[0].dataset:
        counts["LIDC_IDRI_CT_scans"] += 1

    if ct_scans and "LIDC-IDRI" not in ct_scans[0].dataset:
        counts["RICORD_CT_scans"] += 1

    if (
        ct_scans
        and (singleview_dirs or multiview_dirs)
        and "LIDC-IDRI" in ct_scans[0].dataset
    ):
        counts["LIDC_IDRI_paired_patients"] += 1

    if (
        ct_scans
        and (singleview_dirs or multiview_dirs)
        and "LIDC-IDRI" not in ct_scans[0].dataset
    ):
        counts["RICORD_paired_patients"] += 1
logger.debug("Finished!\n\n")

logger.debug("Counts:")
for count in counts:
    logger.debug(f"{count}: {counts[count]}")
logger.debug("Finished!\n\n")

# filters dataset
dataset.filter_dataset()

# Prints and writes Patient and basic data availability for each patient on disk
# after filtering them.
counts = defaultdict(int)
data_summary = []

logger.info("Acquiring per patient statistics (after filter)...")
for patient in dataset:
    ct_scans: list[CTScanDirectory] = dataset[patient]["cts"]
    singleview_dirs: list[SingleViewXRayDirectory] = dataset[patient]["xrays"]
    multiview_dirs: list[MultiviewXRayDirectory] = dataset[patient]["multiview"]
    data_summary.append(
        {
            "patient": patient,
            "total_number_of_cts": len(ct_scans),  # CTs
            "number_of_valid_cts": len([x for x in ct_scans if x.is_valid_scan]),
            "number_of_valid_axial_cts": len(
                [x for x in ct_scans if x.is_valid_axial_scan]
            ),
            "number_of_axial_cts": len([x for x in ct_scans if x.is_axial]),
            # singleviews
            "total_number_of_singleview_xrays": len(singleview_dirs),
            "number_of_frontal_singleview_xrays": len(
                [x for x in singleview_dirs if x.is_frontal]
            ),
            "number_of_lateral_singleview_xrays": len(
                [x for x in singleview_dirs if x.is_lateral]
            ),
            # multiviews
            "total_number_of_multiview_xrays": len(multiview_dirs),
            "number_of_both_views_available_multiview_xrays": len(
                [x for x in multiview_dirs if x.has_both_views]
            ),
            "number_of_frontal_multiview_xrays": len(
                [x for x in multiview_dirs if x.has_frontal]
            ),
            "number_of_lateral_multiview_xrays": len(
                [x for x in multiview_dirs if x.has_lateral]
            ),
            "is_paired_patient": True
            if ct_scans and (singleview_dirs or multiview_dirs)
            else False,
        }
    )
logger.info("Finished!\n\n")

logger.info(
    f'Writing per patient summary to {SAVE_ROOT.parent / "filtered_data_summary.csv"}...'
)
with open(SAVE_ROOT.parent / "filtered_data_summary.csv", "w") as f:
    writer = csv.DictWriter(f=f, fieldnames=data_summary[0].keys(), delimiter=";")
    writer.writeheader()
    writer.writerows(data_summary)
logger.info("Finished!\n\n")

logger.debug("Counting full dataset statistics for printing (after filter)...")
for patient in dataset:
    counts["total_patients"] += 1
    ct_scans: list[CTScanDirectory] = dataset[patient]["cts"]
    singleview_dirs: list[SingleViewXRayDirectory] = dataset[patient]["xrays"]
    multiview_dirs: list[MultiviewXRayDirectory] = dataset[patient]["multiview"]

    if ct_scans and (singleview_dirs or multiview_dirs):
        counts["paired_patients"] += 1

    if singleview_dirs or multiview_dirs:
        counts["at_least_one_xray"] += 1
        if singleview_dirs and multiview_dirs:
            counts["has_singleview_and_multiview"] += 1

        if len(singleview_dirs) > 1:
            counts["more_than_one_singleview"] += 1

        if len(multiview_dirs) > 1:
            counts["more_than_one_multiview"] += 1

    if ct_scans:
        counts["at_least_one_ct"] += 1
        if len(ct_scans) > 1:
            counts["more_than_one_CT"] += 1

    if ct_scans and "LIDC-IDRI" in ct_scans[0].dataset:
        counts["LIDC_IDRI_CT_scans"] += 1

    if ct_scans and "LIDC-IDRI" not in ct_scans[0].dataset:
        counts["RICORD_CT_scans"] += 1

    if (
        ct_scans
        and (singleview_dirs or multiview_dirs)
        and "LIDC-IDRI" in ct_scans[0].dataset
    ):
        counts["LIDC_IDRI_paired_patients"] += 1

    if (
        ct_scans
        and (singleview_dirs or multiview_dirs)
        and "LIDC-IDRI" not in ct_scans[0].dataset
    ):
        counts["RICORD_paired_patients"] += 1
logger.debug("Finished!\n\n")

logger.debug("Counts:")
for count in counts:
    logger.debug(f"{count}: {counts[count]}")
logger.debug("Finished!\n\n")

writing_progress_bar = tqdm(total=len(list(dataset.keys())))

logger.info("Writing dataset to disk in hdf5 format...")
for patient in dataset:
    logger.debug(f"Processing {patient}...")
    writing_progress_bar.set_description(
        f"Writing {patient}..."
    )  # update the progress bar description to show which patient we are currently processing
    writing_progress_bar.update()
    if not dataset[patient]["cts"]:
        logger.debug("No CT scans found for this patient. Skipping...")
        continue

    patient_output_file = SAVE_ROOT / (patient + ".hdf5")
    if (
        patient_output_file.exists()
    ):  # if the file already exists, skip it to avoid overwriting
        logger.debug(f"'{patient_output_file}' already exists. Skipping...")
        continue
    logger.debug("Writing patient data to disk...")
    PatientDataStorer(
        patient_id=patient, patient_data_dictionary=dataset[patient]
    ).store_to_file(dirpath=SAVE_ROOT, compression=9)
    logger.debug("Finished writing patient data to disk.")

    logger.debug("Clearing cache...")
    ct_scans: list[CTScanDirectory] = dataset[patient]["cts"]
    for ct_scan in ct_scans:
        ct_scan.cache.clear()
    singleview_dirs: list[SingleViewXRayDirectory] = dataset[patient]["xrays"]
    for singleview_dir in singleview_dirs:
        singleview_dir.cache.clear()
    multiview_dirs: list[MultiviewXRayDirectory] = dataset[patient]["multiview"]
    for multiview_dir in multiview_dirs:
        multiview_dir.cache.clear()
    logger.debug(f"Finished processing {patient}!\n")
logger.info("Finished!\n\n")

writing_progress_bar.close()

# TODO access the hdf5 and ensure the stats remain the same, like
# total number of paired patients.
counts = defaultdict(int)
logger.debug("Accessing h5 dataset and validating statistics...")
for patient in dataset:
    counts["total_patients"] += 1
    ct_scans: list[CTScanDirectory] = dataset[patient]["cts"]
    singleview_dirs: list[SingleViewXRayDirectory] = dataset[patient]["xrays"]
    multiview_dirs: list[MultiviewXRayDirectory] = dataset[patient]["multiview"]

    if ct_scans and (singleview_dirs or multiview_dirs):
        counts["paired_patients"] += 1

    if singleview_dirs or multiview_dirs:
        counts["at_least_one_xray"] += 1

    if ct_scans:
        counts["at_least_one_ct"] += 1

    if ct_scans and "LIDC-IDRI" in ct_scans[0].dataset:
        counts["LIDC_IDRI_CT_scans"] += 1

    if ct_scans and "LIDC-IDRI" not in ct_scans[0].dataset:
        counts["RICORD_CT_scans"] += 1

    if (
        ct_scans
        and (singleview_dirs or multiview_dirs)
        and "LIDC-IDRI" in ct_scans[0].dataset
    ):
        counts["LIDC_IDRI_paired_patients"] += 1

    if (
        ct_scans
        and (singleview_dirs or multiview_dirs)
        and "LIDC-IDRI" not in ct_scans[0].dataset
    ):
        counts["RICORD_paired_patients"] += 1
logger.debug("Finished!\n\n")

logger.debug("Counts:")
for count in counts:
    logger.debug(f"{count}: {counts[count]}")
logger.debug("Finished!\n\n")

logger.info("Preprocessing finished!")
