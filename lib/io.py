"""
Functions for saving and exporting files into the disc
"""

from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import time
import json
import os
import shutil
import tempfile

from typing import TYPE_CHECKING
from lib.process.preprocess import CTProcessor

if TYPE_CHECKING:
    from lib.dicom import (
        SingleViewXRayDirectory,
        MultiviewXRayDirectory,
        CTScanDirectory,
    )


class HDF5FileHandler:
    def __init__(
        self, filepath: str | Path, mode: str = "auto", compression: str | int = "lzf"
    ) -> None:
        """
        compression:
            (String or int) Compression strategy. Legal values are 'gzip', 'szip', 'lzf'. If an integer in range(10), this indicates gzip compression level. Otherwise, an integer indicates the number of a dynamically loaded compression filter.
        """
        self.filepath = Path(filepath)
        if not self.filepath.suffix == ".h5":
            raise ValueError("The file must be an HDF5 file (.h5)")
        if not self.filepath.suffix:
            raise ValueError("The path must have an extension")
        if not isinstance(compression, (str, int)):
            raise ValueError("Compression must be a string or an integer")

        # Create parent folders if they don't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.compression = compression

        if self.mode not in ["auto", "read_only", "overwrite"]:
            raise ValueError("Mode must be either 'auto', 'read_only' or 'overwrite'")

        # If the file exists, open it in read mode, otherwise create a new one and open it in write mode
        if self.mode == "overwrite":
            self._open_in_mode("write")
        elif self.mode == "auto" and not self.exists:  # if doesn't exist, create
            self._open_in_mode("write")
        elif self.mode == "auto" and self.exists:  # if exists, append
            self._open_in_mode("append")
        elif self.mode == "read_only" and self.exists:  # if read only
            self._open_in_mode("read")
        else:
            raise ValueError(
                "If the file exists, mode must be either 'auto', or 'edit'"
            )

    @property
    def is_read_only(self):
        return self.file.mode == "r"

    @property
    def is_writable(self):
        return "a" in self.file.mode or "w" in self.file.mode or "+" in self.file.mode

    @property
    def exists(self):
        return self.filepath.exists()

    def _open_in_mode(self, mode: str):
        # print(mode)
        if mode == "read":
            self.file = h5py.File(self.filepath, "r")
        elif mode == "write":
            self.file = h5py.File(self.filepath, "w")
        elif mode == "append":
            self.file = h5py.File(self.filepath, "a")
        else:
            raise ValueError("Mode must be either 'read', 'write', or 'append'")

    def validate_required_keys(self):
        required_keys = ["ct", "xray1", "xray2"]
        if "frontal_drr" in self.file and "lateral_drr" in self.file:
            required_keys = ["ct", "frontal_drr", "lateral_drr"]
        for required_key in required_keys:
            if required_key not in self.file:
                raise ValueError(f"The required key '{required_key}' is missing")
        metadata_keys = ("metadata_json", "provenance_json")
        present = [key in self.file for key in metadata_keys]
        if any(present) and not all(present):
            raise ValueError("HDF5 metadata and provenance must be present together")
        if all(present):
            for key in metadata_keys:
                value = json.loads(self.file[key].asstr()[()])
                if not isinstance(value, dict) or value.get("schema_version") != 1:
                    raise ValueError(f"Invalid HDF5 {key} schema")

    def close(self) -> None:
        self.file.close()

    def store_array(self, key: str, array: np.ndarray, force: bool = False):
        if not self.is_writable and not force:
            raise ValueError(
                "The file is opened in read only mode, use 'force=True' to open it anyway"
            )
        if not self.is_writable and force:
            self.close()
            if self.exists:
                self._open_in_mode("append")
            else:
                self._open_in_mode("write")
        if key in self.file:
            del self.file[key]
        self.file.create_dataset(key, data=array, compression=self.compression)

    def store_dict(self, key: str, dict: dict, force: bool = False):
        if not self.is_writable and not force:
            raise ValueError(
                "The file is opened in read only mode, use 'force=True' to open it anyway"
            )
        if not self.is_writable and force:
            self.close()
            if self.exists:
                self._open_in_mode("append")
            else:
                self._open_in_mode("write")
        if key in self.file:
            del self.file[key]

        # Create a group for the top-level key
        group = self.file.create_group(key)

        # recursively creates entires for each dict key and value
        self._store_nested_dict(group, dict)

    def _store_nested_dict(self, hdf5_group, data):
        for sub_key, value in data.items():
            if isinstance(value, dict):
                subgroup = hdf5_group.create_group(sub_key)
                self._store_nested_dict(subgroup, value)
            else:
                # Store non-dict items as datasets
                hdf5_group.create_dataset(sub_key, data=value)

    def read_dict(self, key: str) -> dict:
        if key not in self.file:
            raise KeyError(f"No data found for key '{key}'")
        group = self.file[key]
        return self._load_nested_dict(group)

    def _load_nested_dict(self, hdf5_group):
        result = {}
        for sub_key, item in hdf5_group.items():
            if isinstance(item, h5py.Group):
                # Recursively load subgroups
                result[sub_key] = self._load_nested_dict(item)
            else:
                # Load datasets directly
                out = item[()]
                if isinstance(out, bytes):
                    out = out.decode("utf-8")
                result[sub_key] = out  # Retrieve the dataset value
        return result

    def retrieve_array(self, key: str) -> np.ndarray:
        return np.array(self.file[key])


# should also take in the logger
class PatientDataStorer:
    def __init__(
        self, patient_id: str, patient_data_dictionary: dict, logger=None
    ) -> None:
        self.patient_id = patient_id
        self.data_dictionary = patient_data_dictionary
        self.logger = logger
        if not self.data_dictionary["cts"]:
            raise ValueError("The provided dictionary does not contain any CT scans")
        self._order_data_dictionary()

    def _order_data_dictionary(self):
        """
        Reorders the data dictionary based on the datetime delta from
        the CT scan of the patient data dictionary. This is meant
        to get the x-rays with the least difference in time from the CT scan.
        """
        ct_scans: list[CTScanDirectory] = self.data_dictionary["cts"]
        singleview_dirs: list[SingleViewXRayDirectory] = self.data_dictionary["xrays"]
        multiview_dirs: list[MultiviewXRayDirectory] = self.data_dictionary["multiview"]

        if ct_scans:
            self.data_dictionary["cts"] = ct_scans = [
                ct_scan for ct_scan in ct_scans if ct_scan.is_valid_axial_scan
            ]
            self.data_dictionary["cts"] = ct_scans = sorted(
                ct_scans, key=lambda x: len(x), reverse=True
            )

            # used_ct_scan = ct_scans[0].acquisition_datetime

        # gets the x-ray with acquisition time closest to the CT scan or the
        # newest x-ray
        if singleview_dirs:
            if ct_scans:
                self.data_dictionary["xrays"] = singleview_dirs = sorted(
                    singleview_dirs,
                    key=lambda x: abs(
                        x.acquisition_datetime - ct_scans[0].acquisition_datetime
                    ),
                )
            else:
                self.data_dictionary["xrays"] = singleview_dirs = sorted(
                    singleview_dirs, key=lambda x: x.acquisition_datetime
                )
            self.data_dictionary["xrays"] = singleview_dirs = [
                singleview for singleview in singleview_dirs
            ]

        # gets the x-ray with acquisition time closest to the CT scan or the
        # newest x-ray
        if multiview_dirs:
            if ct_scans:
                self.data_dictionary["multiview"] = multiview_dirs = sorted(
                    multiview_dirs,
                    key=lambda x: abs(
                        x.acquisition_datetime - ct_scans[0].acquisition_datetime
                    ),
                )
            else:
                self.data_dictionary["multiview"] = multiview_dirs = sorted(
                    multiview_dirs, key=lambda x: x.acquisition_datetime
                )
            self.data_dictionary["multiview"] = multiview_dirs = [
                multiview for multiview in multiview_dirs
            ]

    def store_to_file(self, dirpath: str | Path, compression: str | int = "lzf"):
        """
        compression:
            (String or int) Compression strategy. Legal values are 'gzip', 'szip', 'lzf'. If an integer in range(10), this indicates gzip compression level. Otherwise, an integer indicates the number of a dynamically loaded compression filter.
        """
        age = None
        dirpath = Path(dirpath)
        ct_scans: list[CTScanDirectory] = self.data_dictionary["cts"]
        singleview_dirs: list[SingleViewXRayDirectory] = self.data_dictionary["xrays"]
        multiview_dirs: list[MultiviewXRayDirectory] = self.data_dictionary["multiview"]

        ct_scan = ct_scans[0]
        audit = getattr(self, "audit", None)
        if audit is not None:
            audit.select_ct_series(self.patient_id, ct_scan.path_directory)
        # preloads processed array
        start_time = time.time()
        ct_scan.get_array()
        if self.logger:
            self.logger.debug(
                f"Preloading processed CT array took {time.time() - start_time:.2f} seconds"
            )
        # logs starting to generate DRRs
        start_time = time.time()
        drr_generator = ct_scan.drr_generator
        # logs end of DRR generation and the time taken
        frontal_drr = drr_generator.frontal_view
        lateral_drr = drr_generator.lateral_view
        if self.logger:
            self.logger.debug(
                f"DRR generation took {time.time() - start_time:.2f} seconds"
            )
        #del drr_generator  # free up memory from DRR generator

        # creates lists containing frontal and lateral views of the x-rays
        singleview_frontals = [
            singleview
            for singleview in singleview_dirs
            if singleview.view_type == "frontal"
        ]
        singleview_laterals = [
            singleview
            for singleview in singleview_dirs
            if singleview.view_type == "lateral"
        ]

        frontal_xray = None
        lateral_xray = None
        frontal_view = None
        lateral_view = None
        observed_xray_sources = []

        if singleview_dirs or multiview_dirs:
            pass

        if singleview_frontals:
            frontal_view = singleview_frontals[0].get_view()
            frontal_xray = frontal_view.get_array()
            observed_xray_sources.append(getattr(frontal_view, "source_metadata", None))
            if age is None or age == "":
                age = singleview_frontals[0].dicom.dicom.get("PatientAge")
        if singleview_laterals:
            lateral_view = singleview_laterals[0].get_view()
            lateral_xray = lateral_view.get_array()
            observed_xray_sources.append(getattr(lateral_view, "source_metadata", None))
            if age is None or age == "":
                age = singleview_laterals[0].dicom.dicom.get("PatientAge")
        # decides to use either multiview dir or singleview dir for xrays
        # prefers using multiview if any of the directories have both views
        if multiview_dirs:
            multiviews_with_both_views = [
                multiview for multiview in multiview_dirs if multiview.has_both_views
            ]
            # print(multiviews_with_both_views)
            if multiviews_with_both_views:
                candidate_frontal = multiviews_with_both_views[0].get_by_view_type("frontal")
                candidate_lateral = multiviews_with_both_views[0].get_by_view_type("lateral")
                if candidate_frontal is not None:
                    frontal_view = candidate_frontal
                    frontal_xray = frontal_view.get_array()
                    observed_xray_sources.append(getattr(frontal_view, "source_metadata", None))
                    if age is None or age == "":
                        age = frontal_view.dicom.get("PatientAge")
                if candidate_lateral is not None:
                    lateral_view = candidate_lateral
                    lateral_xray = lateral_view.get_array()
                    observed_xray_sources.append(getattr(lateral_view, "source_metadata", None))
                    if age is None or age == "":
                        age = lateral_view.dicom.get("PatientAge")
            else:
                # handle the case where there are no directories with both views
                candidate_frontal = multiview_dirs[0].get_by_view_type(
                    "frontal"
                )  # may be None
                candidate_lateral = multiview_dirs[0].get_by_view_type(
                    "lateral"
                )  # may be None
                if candidate_frontal:
                    frontal_view = candidate_frontal
                    frontal_xray = frontal_view.get_array()
                    observed_xray_sources.append(getattr(frontal_view, "source_metadata", None))
                    if age is None or age == "":
                        age = frontal_view.dicom.get("PatientAge")
                if candidate_lateral:
                    lateral_view = candidate_lateral
                    lateral_xray = lateral_view.get_array()
                    observed_xray_sources.append(getattr(lateral_view, "source_metadata", None))
                    if age is None or age == "":
                        age = lateral_view.dicom.get("PatientAge")

        hdf5_path = dirpath / self.patient_id / "ct_xray_data.h5"

        # get the raw CT array (preserves original pixel values)
        # and convert it to Hounsfield Units (HU) without applying
        # the full `process()` pipeline (which would rescale/normalize)
        # logs time taken to get raw array
        start_time = time.time()
        ct_raw = ct_scan.get_array()
        if self.logger:
            self.logger.debug(
                f"Getting raw CT array took {time.time() - start_time:.2f} seconds"
            )
        # logs time taken to get raw array
        # print("[Step 1 – Raw array] min:", ct_raw.min(), "max:", ct_raw.max())

        # Step 2 – Initialise processor (no processing yet)
        # supply the entire scan; the processor will grab the
        # representative slice internally
        ct_processor = CTProcessor(ct_raw, ct_scan)
        # print("[Step 2 – Processor init] min:", ct_processor.array.min(), "max:", ct_processor.array.max())

        # Step 3 – Convert raw values to Hounsfield Units (HU)
        #ct_processor._raw_array_to_hu()
        # print("[Step 3 – Raw→HU] min:", ct_processor.array.min(), "max:", ct_processor.array.max())

        # Step 4 - segment body
        # logs time taken to segment step
        #start_time = time.time()
        #ct_processor._segment_body()
        #if self.logger:
        #    self.logger.debug(
        #        f"Segmenting body took {time.time() - start_time:.2f} seconds"
        #    )
        # finishes logging time taken to segment step

        # Step 5 – Transform to LIDC format (clipping and shifting)
        # logs time taken to transform to LIDC format
        #start_time = time.time()
        #ct_processor._transform_to_lidc_format()
        #if self.logger:
        #    self.logger.debug(
        #        f"Transforming to LIDC format took {time.time() - start_time:.2f} seconds"
        #    )
        # finishes logging time taken to transform to LIDC format
        # print("[Step 4 – LIDC transformation] min:", ct_processor.array.min(), "max:", ct_processor.array.max())

        # Step 6 – Transpose axes
        ct_processor._transpose()
        # print("[Step 5 – Transpose] min:", ct_processor.array.min(), "max:", ct_processor.array.max())

        # Step 7 – Resize (kept same size here)
        ct_processor._resize()
        # print("[Step 6 – Resize] min:", ct_processor.array.min(), "max:", ct_processor.array.max())

        # Final volume ready for downstream use
        ct_volume = ct_processor.array
        # print("[Final] CT volume shape:", ct_volume.shape, "min:", ct_volume.min(), "max:", ct_volume.max())

        metadata = {}
        dcm = ct_scan.header
        if age is None or age == "":
            age = dcm.get("PatientAge")

        age = age if age is not None else "000Y"
        age = age if age else "000Y"
        metadata["age"] = age.replace("Y", "")
        metadata["slice_thickness"] = str(dcm.get("SliceThickness"))
        metadata["acquisition_date"] = str(dcm.get("AcquisitionDate"))
        metadata["spacing"] = ct_scan.spacing

        source_records = self._build_source_records(ct_scan, dcm, frontal_view, lateral_view)
        sample_id = f"{source_records['ct']['collection'] or 'unknown'}::{self.patient_id}"
        metadata_observations = {
            "schema_version": 1,
            "run_id": getattr(self, "run_id", None),
            "sample_id": sample_id,
            "patient_id": self.patient_id,
            "collection": source_records["ct"]["collection"],
            "ct": source_records["ct"],
            "frontal": source_records["frontal"],
            "lateral": source_records["lateral"],
        }
        provenance = {
            "schema_version": 1,
            "run_id": getattr(self, "run_id", None),
            "sample_id": sample_id,
            "patient_id": self.patient_id,
            "collection": source_records["ct"]["collection"],
            "selection_status": "selected",
            "selection_context": getattr(self, "selection_context", None),
            "selected": {
                role: record.get("path") if isinstance(record, dict) else None
                for role, record in source_records.items()
            },
            "artifacts": {
                "ct_xray_data.h5": "pending",
                "real_xray1": "pending" if frontal_xray is not None else "not_generated",
                "real_xray2": "pending" if lateral_xray is not None else "not_generated",
            },
        }

        # logs time taken to store data in h5 file
        start_time = time.time()
        arrays = {"ct": ct_volume, "xray1": frontal_drr, "xray2": lateral_drr}
        if frontal_xray is not None:
            arrays["real_xray1"] = frontal_xray
        if lateral_xray is not None:
            arrays["real_xray2"] = lateral_xray
        self._publish_product(hdf5_path, arrays, metadata,
                              (metadata_observations, provenance), compression)
        audit = getattr(self, "audit", None)
        if audit is not None:
            audit.samples.append({
                **metadata_observations, "provenance": provenance,
                "output_path": str(hdf5_path), "production_status": "generated",
            })
            try:
                self._record_audit_output(audit, source_records, observed_xray_sources, hdf5_path)
            except Exception as exc:
                audit.errors.append({"stage": "output_audit", "patient_id": self.patient_id,
                                     "message": str(exc), "exception_type": type(exc).__name__})
        if self.logger:
            self.logger.debug(
                f"Storing data to h5 file took {time.time() - start_time:.2f} seconds"
            )
        # finishes logging time taken to store data in h5 file

    def _write_json_metadata(self, h5f, metadata, provenance):
        json_default = lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value)
        for key, payload in (("metadata_json", metadata), ("provenance_json", provenance)):
            if payload is None:
                continue
            if key in h5f.file:
                del h5f.file[key]
            h5f.file.create_dataset(
                key,
                data=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

    def _publish_product(self, path, arrays, legacy_metadata, documents, compression):
        """Publish a complete product once; failures leave the previous file intact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".ct_xray_", suffix=".h5", dir=path.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            if path.exists():
                shutil.copy2(path, temporary)
            else:
                temporary.unlink()
            stored = HDF5FileHandler(temporary, mode="auto", compression=compression)
            try:
                for key, array in arrays.items():
                    stored.store_array(key, array)
                stored.store_dict("metadata", legacy_metadata)
                metadata, provenance = documents
                provenance["artifacts"] = {path.name: "generated", **{
                    key: "generated" if key in arrays else "not_generated"
                    for key in ("real_xray1", "real_xray2")
                }}
                self._write_json_metadata(stored, metadata, provenance)
            finally:
                stored.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _build_source_records(self, ct_scan, dcm, frontal_view, lateral_view):
        ct_root = getattr(ct_scan, "source_root", None)
        ct_path = Path(ct_scan.path_directory)
        from lib.data.temporal import observe_ct_series_header

        ct_record = observe_ct_series_header(ct_scan.header, ct_path.absolute())
        ct_record.update(self._source_reference(ct_path, ct_root))
        ct_record["collection"] = getattr(ct_scan, "dataset", None)
        ct_record["series_summary"] = getattr(ct_scan, "series_summary", {
            "status": "not_observed_during_montage",
        })
        records = {"ct": ct_record}
        for role, image in (("frontal", frontal_view), ("lateral", lateral_view)):
            record = getattr(image, "source_metadata", None)
            if record is not None:
                root = getattr(image, "source_root", None)
                record.update(self._source_reference(record["path"], root))
            records[role] = record
        return records

    @staticmethod
    def _source_reference(path, root):
        path = Path(path).absolute()
        root = Path(root).absolute() if root is not None else None
        inside = root is not None and path.is_relative_to(root)
        return {"path": str(path), "root_id": "dicom_root" if root is not None else None,
                "root_path": str(root) if root is not None else None,
                "relative_path": str(path.relative_to(root)) if inside else None,
                "reference_status": "relative" if inside else "absolute_only"}

    def _record_audit_output(self, audit, source_records, observed_xray_sources, hdf5_path):
        selected_sources = {}
        final_paths = set()
        source_roles = {}
        observed_xray_by_path = {
            row["path"]: row for row in observed_xray_sources
            if isinstance(row, dict) and row.get("path")
        }
        for role in ("frontal", "lateral"):
            record = source_records[role]
            if record and record.get("path"):
                selected_sources[record["path"]] = record
                source_roles[record["path"]] = role
                final_paths.add(record["path"])
        observed_sources = {**observed_xray_by_path, **selected_sources}
        series_by_directory = {row.get("path"): row for row in audit.series_by_patient[self.patient_id]}
        recorded_paths = {directory: set(row["file_paths"]) for directory, row in series_by_directory.items()}
        ct_record = source_records["ct"]
        ct_series = series_by_directory.get(ct_record.get("path"))
        if ct_series is not None:
            ct_series.update(
                selection_status="selected", processing_status="processed",
                selection_reason=None,
                metadata_status="observed", role="ct_series", source_metadata=ct_record,
                series_summary=ct_record["series_summary"],
                sample_id=f"{ct_record.get('collection') or 'unknown'}::{self.patient_id}",
            )
        for path, source in observed_sources.items():
            directory = str(Path(path).parent)
            series = series_by_directory.get(directory)
            if series is not None and path not in recorded_paths[directory]:
                series["file_paths"].append(path)
                recorded_paths[directory].add(path)
            if path not in audit.files_by_path:
                audit.add_file(
                    path=path, patient_id=self.patient_id,
                    collection=series.get("collection") if series else source_records["ct"]["collection"],
                    modality=series.get("modality") if series else None,
                    root_id=source.get("root_id"), relative_path=source.get("relative_path"),
                    patient_selection_status="selected",
                )
        for row in audit.files_by_patient[self.patient_id]:
            path = row.get("path")
            source = selected_sources.get(path) or observed_xray_by_path.get(path)
            if source is not None:
                selected = path in final_paths
                row.update(read_status="read_ok", metadata_status="observed", source_metadata=source,
                           role=source_roles.get(path, "clinical_candidate"),
                           study_instance_uid=source.get("study_instance_uid"),
                           series_instance_uid=source.get("series_instance_uid"),
                           sop_instance_uid=source.get("sop_instance_uid"),
                           processing_status="processed" if selected else "not_applicable",
                           selection_status="selected" if selected else "not_selected",
                           selection_reason=source.get("discard_reason") or (None if selected else "candidate_replaced_or_not_used_by_existing_processing_order"))
                if selected:
                    row["sample_id"] = f"{source_records['ct']['collection'] or 'unknown'}::{self.patient_id}"
            elif row.get("patient_selection_status") == "selected" and row.get("selection_status") == "not_evaluated":
                row.update(selection_status="not_selected", selection_reason="candidate_not_chosen_by_existing_processing_order")
        for row in audit.series_by_patient[self.patient_id]:
            if row is ct_series:
                continue
            if final_paths.intersection(row.get("file_paths", [])):
                row.update(selection_status="selected", processing_status="processed")
            elif row.get("eligibility_status") not in {"ineligible", "blocked"}:
                row.update(selection_status="not_selected", processing_status="not_applicable",
                           selection_reason="series_not_used_by_existing_processing_order")
        row = audit.patients_by_id.get(self.patient_id)
        if row is not None:
            row.update(processing_status="processed", output_path=str(hdf5_path))

    def retrieve_from_file(self):
        h5f = HDF5FileHandler(self.patient_id, mode="read")
        data_dictionary = {}
        for key in h5f.file:
            data_dictionary[key] = h5f.retrieve_array(key)
        h5f.close()
        return data_dictionary

    def append_to_file(self, new_data):
        h5f = HDF5FileHandler(self.patient_id, mode="append")
        for key, value in new_data.items():
            h5f.store_array(key, value)
        h5f.close()

    def validate_file(self):
        h5f = HDF5FileHandler(self.patient_id, mode="read")
        h5f.validate_required_keys()
        h5f.close()
