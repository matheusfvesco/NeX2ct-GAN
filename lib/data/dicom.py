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
from lib.data.audit import DicomAudit
import os
import csv


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
    ) -> None:
        self.root_dirpath = Path(root_dir)
        self.accepted_datasets = accepted_datasets
        self.dataset = None
        self.logger = logger if logger is not None else setup_logger("preprocess")
        self.audit = DicomAudit()
        self.selection_annotations = {}
        self._audit_root = self.root_dirpath.absolute()

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
        self.audit = DicomAudit()
        self.dataset = defaultdict(self._create_default_patient_dict)

        datasets_bar = tqdm(self.accepted_datasets)
        for dataset_name in datasets_bar:
            datasets_bar.set_description(dataset_name)
            dataset_path = self.root_dirpath / dataset_name
            self.logger.debug(f"including data inside '{dataset_path}'...")
            single_dataset_bar = tqdm(list(os.walk(dataset_path)))
            for dirpath, dirnames, filenames in single_dataset_bar:
                single_dataset_bar.set_description(f"{dataset_name}")
                if not filenames:
                    continue
                dcm_paths = list(Path(dirpath).glob("*.dcm"))
                if not dcm_paths:
                    continue
                #self.logger.warning(
                #    f"Begiinning processing directory '{dirpath}' with {len(dcm_paths)} DICOM files"
                #)
                patient_id = None
                try:
                    directory = DicomDirectory(dirpath)
                    directory.source_root = self._audit_root
                    patient_id: str = directory.dataset_agnostic_patient_id
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
                    failure = {
                        "stage": "directory_classification",
                        "exception_type": type(e).__name__,
                        "message": str(e),
                    }
                    self.audit.add_series(
                        path=str(Path(dirpath).absolute()),
                        patient_id=patient_id,
                        expected_patient=patient_id is not None,
                        dataset=dataset_name,
                        file_paths=[],
                        file_count=len(dcm_paths),
                        files_not_verified=len(dcm_paths),
                        read_status="read_failed",
                        metadata_status="not_verified",
                        eligibility_status="blocked",
                        selection_status="not_evaluated",
                        reason="directory_classification_failed",
                        errors=[failure],
                    )
                    continue
                # Observational failures must never enter the classification/exclusion path.
                try:
                    self._record_directory_inventory(directory, dirpath, dataset_name, dcm_paths, patient_id)
                except Exception as exc:
                    self.audit.errors.append({
                        "stage": "metadata_capture", "path": str(dirpath),
                        "patient_id": patient_id, "exception_type": type(exc).__name__,
                        "message": str(exc),
                    })
                    self.logger.warning(f"Audit metadata unavailable for '{dirpath}': {exc}")
            single_dataset_bar.close()
        self.logger.info("Success!\n\n")
        datasets_bar.close()

    def _record_directory_inventory(self, directory, dirpath, dataset_name, dcm_paths, patient_id):
        header = directory.header
        representative_path = str(Path(directory.representative_path).absolute())
        is_ct = str(header.get("Modality") or "").upper() == "CT"
        from lib.data.temporal import observe_ct_series_header, observe_dataset
        if is_ct:
            paths = []
            observed = {}
            observation = observe_ct_series_header(header, Path(dirpath).absolute())
        else:
            observed = {row["path"]: row for row in getattr(directory, "observed_sources", [])}
            paths = list(dict.fromkeys([representative_path, *observed]))
            observation = observe_dataset(header, representative_path)
        file_count = len(directory._paths_to_dicom_files)
        retained = any(item is directory for items in self.dataset.get(patient_id, {}).values() for item in items)
        series = self.audit.add_series(
            path=str(Path(dirpath).absolute()), patient_id=patient_id, expected_patient=True,
            root_id="dicom_root", root_path=str(self._audit_root),
            relative_path=str(Path(dirpath).absolute().relative_to(self._audit_root)),
            dataset=dataset_name, collection=str(directory.dataset),
            role="ct_series" if is_ct else "radiographic_series",
            modality=str(header.get("Modality") or "").upper(),
            study_instance_uid=observation.get("study_instance_uid"),
            series_instance_uid=observation.get("series_instance_uid"),
            representative_file_path=None if is_ct else representative_path,
            file_paths=paths, file_count=file_count,
            file_paths_scope="series_record" if is_ct else "observed_images_only",
            instance_count_observed=file_count if is_ct else None,
            metadata_scope=observation.get("temporal_scope", "representative_image_header"),
            eligibility_status="eligible" if retained else "ineligible",
            selection_status="not_evaluated" if retained else "excluded",
            reason=None if retained else "legacy_technical_criteria",
        )
        series["source_metadata"] = observation
        series.update(read_status="read_ok", metadata_status="observed")
        if is_ct:
            series["series_summary"] = {
                "status": "not_montaged_during_normal_processing",
                "instance_count_observed": file_count,
                "instance_count_used": None,
                "instance_count_discarded": None,
                "discard_reason_counts": {},
                "assembly_rule": None,
            }
        for path in paths:
            row = self.audit.add_file(
                path=path, relative_path=str(Path(path).relative_to(self._audit_root)),
                dataset=dataset_name, collection=str(directory.dataset),
                modality=str(header.get("Modality") or "").upper(), patient_id=patient_id,
                expected_patient=True, study_instance_uid=None, series_instance_uid=None,
                sop_instance_uid=None, read_status="indexed", eligibility_status=series["eligibility_status"],
                selection_status=series["selection_status"], metadata_status="not_verified",
                selection_reason=series["reason"],
            )
            source = observed.get(path)
            if source is not None:
                row.update(
                    read_status="read_ok", study_instance_uid=source["study_instance_uid"],
                    series_instance_uid=source["series_instance_uid"], sop_instance_uid=source["sop_instance_uid"],
                    metadata_status="observed", source_metadata=source,
                    temporal=source["temporal"], field_status=source["field_status"],
                    respiratory_evidence=source["respiratory_evidence"],
                )
        series["file_paths"] = paths
        patient_row = self.audit.patients_by_id.get(patient_id)
        if patient_row is None:
            self.audit.add_patient(patient_id=patient_id, dataset=dataset_name,
                                   collections=[str(directory.dataset)], selection_status="not_evaluated")
        elif str(directory.dataset) not in patient_row.setdefault("collections", []):
            patient_row["collections"].append(str(directory.dataset))

    def filter_dataset(self):
        """
        Sorts the datasets based on their number of slices and removes invalid CTs
        """
        if not self.dataset:
            raise ValueError("The dataset haven't been loaded yet")
        self.logger.info("Filtering invalid CTs and sorting by the amount of slices...")

        annotation_path = Path("data/patient-selection-annotation.json")
        from lib.data.cohort import read_original_annotations, verify_patient_membership
        selection_annotations, self.selection_annotation_sha256 = read_original_annotations(annotation_path)
        self.selection_annotations = selection_annotations
        self.selection_annotation_path = str(annotation_path.resolve())
        discovered_patient_ids = tuple(self.keys())
        dataset_bar = tqdm(sorted(self.keys()))
        for patient in dataset_bar:
            dataset_bar.set_description(patient)
            if (
                not self[patient]
                or self[patient] == self._create_default_patient_dict()
            ):
                raise ValueError(f"The dataset {patient} is empty")
            if patient not in selection_annotations.keys():
                self._record_patient_selection(patient, "not_selected", "missing_selection_annotation")
                del self[patient]
                continue
            if selection_annotations[patient] != "selected":
                self._record_patient_selection(patient, "excluded", "patient_selection_annotation")
                del self[patient]
                continue
            self._record_patient_selection(patient, "selected", None)
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

        self.selection_preservation = verify_patient_membership(
            discovered_patient_ids, tuple(self.keys()), selection_annotations
        )
        self.selection_preservation["reference_sha256"] = self.selection_annotation_sha256

    def _record_patient_selection(self, patient_id, status, reason):
        row = self.audit.patients_by_id.get(patient_id)
        if row is not None:
            row.update(selection_status=status, selection_reason=reason)
        for row in self.audit.series_by_patient[patient_id] + self.audit.files_by_patient[patient_id]:
            row["patient_selection_status"] = status
            if status != "selected":
                row.update(selection_status=status, selection_reason=reason)

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
