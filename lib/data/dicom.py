from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict
from lib.dicom import (
    DicomDirectory,
    CTScanDirectory,
    SingleViewXRayDirectory,
    MultiviewXRayDirectory,
)
from lib.logging import setup_logger
import os
import csv
import json
from lib.data.cohort import Cohort
from lib.data.audit import DicomAudit
from lib.data.inventory import annotated_view_paths, read_authorized_index, record_directory_result
from lib.utils.dicom import XRayJsonClassifier


class DicomDataset:
    def __init__(
        self,
        root_dir: str | Path,
        accepted_datasets: list[str] = [
            "MIDRC-RICORD-1A",
            "MIDRC-RICORD-1C",
            "LIDC-IDRI",
        ],
        logger=None,
        cohort: Cohort | None = None,
        audit: DicomAudit | None = None,
    ) -> None:
        self.root_dirpath = Path(root_dir)
        self.accepted_datasets = accepted_datasets
        self.dataset = None
        self.logger = logger if logger is not None else setup_logger("preprocess")
        self.cohort = cohort
        self.audit = audit if audit is not None else DicomAudit()
        self.patient_selection_annotations = {}

    def __iter__(self):
        return iter(self.dataset.keys())

    def __getitem__(self, key):
        return self.dataset[key]

    def __setitem__(self, key, value):
        self.dataset[key] = value

    def __delitem__(self, key):
        del self.dataset[key]

    def keys(self):
        return self.dataset.keys()

    @staticmethod
    def _create_default_patient_dict():
        return {"xrays": [], "multiview": [], "cts": []}

    def load(self):
        """
        Loads the DICOM dataset into memory.
        """
        self.logger.info("Creating dataset dictionary...")
        self.dataset = defaultdict(self._create_default_patient_dict)
        if self.cohort is not None:
            self._load_authorized_series()
            return

        datasets_bar = tqdm(sorted(self.accepted_datasets))
        for dataset_name in datasets_bar:
            datasets_bar.set_description(dataset_name)
            dataset_path = self.root_dirpath / dataset_name
            self.logger.debug(f"including data inside '{dataset_path}'...")
            if not dataset_path.exists():
                self.logger.warning("Dataset path does not exist: '%s'", dataset_path)
                continue
            directories = sorted((Path(dirpath), filenames) for dirpath, _, filenames in os.walk(dataset_path))
            single_dataset_bar = tqdm(directories)
            for directory_path, filenames in single_dataset_bar:
                dirpath = str(directory_path)
                single_dataset_bar.set_description(dataset_name)
                dcm_paths = sorted(
                    path for path in directory_path.iterdir()
                    if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}
                )
                if not dcm_paths:
                    continue
                patient_id, file_rows = None, []
                try:
                    directory = DicomDirectory(dirpath)
                    patient_id = patient_id if self.cohort is not None else directory.dataset_agnostic_patient_id
                    record_directory_result(self.audit, file_rows, dirpath, patient_id, directory=directory)
                    if directory.is_multiview:
                        self[patient_id]["multiview"].append(directory)
                    if directory.is_ct:
                        if directory.is_valid_scan and directory.is_axial:
                            self.dataset[patient_id]["cts"].append(directory)
                    if directory.is_singleview:
                        self[patient_id]["xrays"].append(directory)
                except Exception as e:
                    self.logger.warning(
                        f"Error processing directory '{dirpath}' with {len(dcm_paths)} DICOM files: {e}"
                    )
                    record_directory_result(self.audit, file_rows, dirpath, patient_id, error=e)
                    continue
            single_dataset_bar.close()
        self.logger.info("Success!\n\n")
        datasets_bar.close()

    def _load_authorized_series(self):
        """Filter using the collection index before opening candidate DICOMs."""
        for name in sorted(self.accepted_datasets):
            index_root = self.root_dirpath / name
            for series in read_authorized_index(index_root, self.cohort,
                                                self.patient_selection_annotations, self.audit):
                self._load_indexed_series(series)

    def _load_indexed_series(self, series):
        paths, rows, metadata = series["paths"], series["file_rows"], series["metadata"]
        patient_id = metadata["patient_id"]
        if metadata["modality"] != "CT":
            paths = annotated_view_paths(paths, XRayJsonClassifier())
            selected = set(paths)
            for row in rows:
                if Path(row["path"]) not in selected:
                    row.update(selection_status="skipped", processing_status="skipped",
                               selection_reason="path_not_selected_for_view_classification")
            rows = [row for row in rows if Path(row["path"]) in selected]
        if not paths:
            self.audit.add_series(**metadata, file_paths=[], eligibility_status="ineligible",
                                  reason="no_indexed_dicom_or_eligible_view_path")
            return
        try:
            directory = DicomDirectory(paths[0].parent, dicom_paths=paths)
            header = directory.header
            actual_id, status = self.cohort.resolve_patient_id(header.get("PatientID"))
            if status != "matched" or actual_id != patient_id:
                raise ValueError(f"Representative PatientID differs from collection index for {patient_id}")
            for field, key in (("Modality", "modality"), ("StudyInstanceUID", "study_instance_uid"),
                               ("SeriesInstanceUID", "series_instance_uid")):
                if metadata.get(key) and str(header.get(field)) != metadata[key]:
                    raise ValueError(f"Representative {field} differs from collection index for {patient_id}")
            record_directory_result(self.audit, rows, paths[0].parent, patient_id, directory=directory)
            if directory.is_ct:
                if directory.is_valid_axial_scan:
                    self.dataset[patient_id]["cts"].append(directory)
            elif directory.is_multiview:
                self.dataset[patient_id]["multiview"].append(directory)
            else:
                self.dataset[patient_id]["xrays"].append(directory)
        except Exception as exc:
            self.logger.warning("Error classifying series for '%s': %s", patient_id, exc)
            record_directory_result(self.audit, rows, paths[0].parent, patient_id, error=exc)

    def filter_dataset(self, allowed_patient_ids=None):
        """
        Sorts the datasets based on their number of slices and removes invalid CTs
        """
        if not self.dataset:
            raise ValueError("The dataset haven't been loaded yet")
        self.logger.info("Filtering invalid CTs and sorting by the amount of slices...")

        if allowed_patient_ids is None:
            if self.cohort is None:
                # Preserve the existing figure-script API outside the split-protected flow.
                with (self.root_dirpath / "patient-selection-annotation.json").open(encoding="utf-8") as stream:
                    selection_annotations = json.load(stream)
                allowed_patient_ids = {patient for patient, value in selection_annotations.items() if value == "selected"}
            else:
                allowed_patient_ids = set(self.cohort.all_ids)
        else:
            allowed_patient_ids = set(allowed_patient_ids)
        dataset_bar = tqdm(sorted(self.keys()))
        for patient in dataset_bar:
            dataset_bar.set_description(patient)
            if (
                not self[patient]
                or self[patient] == self._create_default_patient_dict()
            ):
                raise ValueError(f"The dataset {patient} is empty")
            if patient not in allowed_patient_ids:
                del self[patient]
                continue
            ct_scans: list[CTScanDirectory] = self[patient]["cts"]
            singleview_dirs: list[SingleViewXRayDirectory] = self[patient]["xrays"]
            multiview_dirs: list[MultiviewXRayDirectory] = self[patient]["multiview"]

            if singleview_dirs:
                self[patient]["xrays"] = singleview_dirs = sorted(
                    singleview_dirs, key=lambda x: len(x), reverse=True
                )
                self[patient]["xrays"] = singleview_dirs = [
                    singleview for singleview in singleview_dirs
                ]

            if ct_scans:
                self[patient]["cts"] = ct_scans = sorted(
                    ct_scans, key=lambda x: len(x), reverse=True
                )
                self[patient]["cts"] = ct_scans = [
                    ct_scan for ct_scan in ct_scans if ct_scan.is_valid_scan
                ]

            if multiview_dirs:
                self[patient]["multiview"] = multiview_dirs = sorted(
                    multiview_dirs, key=lambda x: len(x), reverse=True
                )
                self[patient]["multiview"] = multiview_dirs = [
                    multiview for multiview in multiview_dirs
                ]
        self.logger.info("Success!\n\n")
        dataset_bar.close()

    def clear_cache(self):
        for patient in self:
            for key in self[patient]:
                for dir in self[patient][key]:
                    dir.cache.clear()

    def write_summary(self, save_path="preprocess_data_summary.csv"):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data_summary = []
        for patient in self:
            ct_scans: list[CTScanDirectory] = self[patient]["cts"]
            singleview_dirs: list[SingleViewXRayDirectory] = self[patient]["xrays"]
            multiview_dirs: list[MultiviewXRayDirectory] = self[patient]["multiview"]
            data_summary.append(
                {
                    "patient": patient,
                    "total_number_of_cts": len(ct_scans),  # CTs
                    "number_of_valid_cts": len(
                        [x for x in ct_scans if x.is_valid_scan]
                    ),
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

        with open(save_path, "w") as f:
            writer = csv.DictWriter(
                f=f, fieldnames=data_summary[0].keys(), delimiter=";"
            )
            writer.writeheader()
            writer.writerows(data_summary)

    def write_counts(self, save_path="modality_counts.csv"):
        counts = defaultdict(int)
        for patient in self:
            counts["total_patients"] += 1
            ct_scans: list[CTScanDirectory] = self[patient]["cts"]
            singleview_dirs: list[SingleViewXRayDirectory] = self[patient]["xrays"]
            multiview_dirs: list[MultiviewXRayDirectory] = self[patient]["multiview"]

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

        # Save counts to CSV file
        with open(save_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Count Type", "Count"])
            for count_type, count in counts.items():
                writer.writerow([count_type, count])

    def get_demo_subset_names(
        self, seed=42, sample_size_porcentage=10, ensure_upside_down_cases=True
    ):
        """
        Returns a list of patient ids that have at least one paired CT scan.
        Use it to iterate over the dataset object and do any kind of transformation
        """
        if ensure_upside_down_cases:
            # found using the logic of CtScanDirectory._is_reversed and hard coded for run efficiency
            upside_down_cases = [
                "419639-000831",
                "419639-001155",
                "419639-001336",
                "419639-001533",
                "LIDC-IDRI-0052",
                "LIDC-IDRI-0067",
                "LIDC-IDRI-0082",
                "LIDC-IDRI-0083",
                "LIDC-IDRI-0091",
                "LIDC-IDRI-0099",
                "LIDC-IDRI-0109",
                "LIDC-IDRI-0114",
                "LIDC-IDRI-0123",
                "LIDC-IDRI-0125",
                "LIDC-IDRI-0129",
                "LIDC-IDRI-0131",
                "LIDC-IDRI-0133",
                "LIDC-IDRI-0155",
                "LIDC-IDRI-0161",
                "LIDC-IDRI-0164",
                "LIDC-IDRI-0168",
                "LIDC-IDRI-0169",
                "LIDC-IDRI-0170",
                "LIDC-IDRI-0176",
                "LIDC-IDRI-0180",
                "LIDC-IDRI-0192",
                "LIDC-IDRI-0199",
                "LIDC-IDRI-0207",
                "LIDC-IDRI-0211",
                "LIDC-IDRI-0220",
                "LIDC-IDRI-0223",
                "LIDC-IDRI-0249",
                "LIDC-IDRI-0264",
                "LIDC-IDRI-0266",
                "LIDC-IDRI-0278",
                "LIDC-IDRI-0280",
                "LIDC-IDRI-0288",
                "LIDC-IDRI-0399",
                "LIDC-IDRI-0470",
                "LIDC-IDRI-0485",
                "LIDC-IDRI-0715",
                "LIDC-IDRI-0760",
                "LIDC-IDRI-0818",
                "LIDC-IDRI-0879",
                "LIDC-IDRI-1010",
                "SITE2-000037",
                "SITE2-000105",
            ]
        sample_size_decimal = int(sample_size_porcentage) / 100
        random.seed(seed)
        # finds patients with CTs and xrays
        paired_patients = [
            patient
            for patient in self
            if self[patient]["cts"]
            and (self[patient]["xrays"] or self[patient]["multiview"])
        ]
        # filters only patients with CTs scans
        patients_with_cts = [
            patient
            for patient in self
            if patient not in paired_patients and self[patient]["cts"]
        ]
        if ensure_upside_down_cases:
            upside_down_cases = [
                patient
                for patient in upside_down_cases
                if patient not in paired_patients and patient not in patients_with_cts
            ]

        # Randomly select N% of patients from patients_with_cts and paired_patients
        num_to_select_cts = int(sample_size_decimal * len(patients_with_cts))
        num_to_select_paired = int(sample_size_decimal * len(paired_patients))
        if ensure_upside_down_cases:
            num_to_select_upside_down = int(
                sample_size_decimal * len(upside_down_cases)
            )

        random_patients_cts = random.sample(patients_with_cts, num_to_select_cts)
        random_patients_paired = random.sample(paired_patients, num_to_select_paired)
        if ensure_upside_down_cases:
            random_patients_upside_down = random.sample(
                upside_down_cases, num_to_select_upside_down
            )

        selected_patients = random_patients_cts + random_patients_paired
        if ensure_upside_down_cases:
            selected_patients += random_patients_upside_down

        selected_patients = sorted(selected_patients)
        return selected_patients
