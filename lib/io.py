"""
Functions for saving and exporting files into the disc
"""

from __future__ import annotations

from pathlib import Path
import json
import h5py
import numpy as np
import time
import os

from typing import TYPE_CHECKING
from lib.process.preprocess import CTProcessor
from lib.data.temporal import acquisition_distance

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
        if mode != "read_only":
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

    def validate_required_keys(self, variant: str | None = None):
        required_keys = ["ct", "xray1", "xray2"]
        if variant == "real":
            required_keys.extend(["real_xray1", "real_xray2"])
        elif variant not in {None, "synthetic", "mixed"}:
            raise ValueError("variant must be one of real, synthetic, mixed, or None")
        for required_key in required_keys:
            if required_key not in self.file:
                raise ValueError(f"The required key '{required_key}' is missing")
            dataset = self.file[required_key]
            dimensions = 3 if required_key == "ct" else 2
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != dimensions:
                raise ValueError(f"The required key '{required_key}' must be a {dimensions}D array")
            if any(size == 0 for size in dataset.shape):
                raise ValueError(f"The required key '{required_key}' has an empty dimension")

    def store_json(self, key: str, value: dict, force: bool = False):
        if not self.is_writable and not force:
            raise ValueError("The file is opened in read only mode")
        if not self.is_writable and force:
            self.close()
            self._open_in_mode("append" if self.exists else "write")
        if key in self.file:
            del self.file[key]
        dtype = h5py.string_dtype(encoding="utf-8")
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda item: item.isoformat() if hasattr(item, "isoformat") else str(item),
        )
        self.file.create_dataset(key, data=payload, dtype=dtype)

    def read_json(self, key: str) -> dict:
        value = self.file[key][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

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
        self,
        patient_id: str,
        patient_data_dictionary: dict,
        logger=None,
        selection_plan: dict | None = None,
    ) -> None:
        self.patient_id = patient_id
        self.data_dictionary = patient_data_dictionary
        self.logger = logger
        self.selection_plan = selection_plan
        if not self.data_dictionary["cts"]:
            raise ValueError("The provided dictionary does not contain any CT scans")
        if self.selection_plan is None:
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
                ct_scans, key=lambda x: (-len(x), str(x.path_directory))
            )

            # used_ct_scan = ct_scans[0].acquisition_datetime

        # gets the x-ray with acquisition time closest to the CT scan or the
        # newest x-ray
        if singleview_dirs:
            if ct_scans:
                self.data_dictionary["xrays"] = singleview_dirs = sorted(
                    singleview_dirs,
                    key=lambda x: (acquisition_distance(x.acquisition_time, ct_scans[0].acquisition_time), str(x.path_directory)),
                )
            else:
                self.data_dictionary["xrays"] = singleview_dirs = sorted(
                    singleview_dirs, key=lambda x: (x.acquisition_datetime is None, str(x.path_directory))
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
                    key=lambda x: (acquisition_distance(x.acquisition_time, ct_scans[0].acquisition_time), str(x.path_directory)),
                )
            else:
                self.data_dictionary["multiview"] = multiview_dirs = sorted(
                    multiview_dirs, key=lambda x: (x.acquisition_datetime is None, str(x.path_directory))
                )
            self.data_dictionary["multiview"] = multiview_dirs = [
                multiview for multiview in multiview_dirs
            ]

    def _get_drr_views(self, ct_scan):
        cached = ct_scan.cache.get("storer_drr_views") if self.selection_plan else None
        if cached is not None:
            return cached
        generator = ct_scan.drr_generator
        views = (generator.frontal_view, generator.lateral_view)
        if self.selection_plan:
            ct_scan.cache.add("storer_drr_views", views)
        return views

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

        ct_scan = self.selection_plan["ct"] if self.selection_plan else ct_scans[0]
        # preloads processed array
        start_time = time.time()
        ct_scan.get_array()
        if self.logger:
            self.logger.debug(
                f"Preloading processed CT array took {time.time() - start_time:.2f} seconds"
            )
        # logs starting to generate DRRs
        start_time = time.time()
        frontal_drr, lateral_drr = self._get_drr_views(ct_scan)
        if self.logger:
            self.logger.debug(
                f"DRR generation took {time.time() - start_time:.2f} seconds"
            )

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

        if singleview_dirs or multiview_dirs:
            pass

        if self.selection_plan is not None:
            frontal_view = self.selection_plan.get("frontal")
            lateral_view = self.selection_plan.get("lateral")
            if frontal_view is not None:
                frontal_xray = frontal_view.get_array()
                if age is None or age == "":
                    age = frontal_view.header.get("PatientAge")
            if lateral_view is not None:
                lateral_xray = lateral_view.get_array()
                if age is None or age == "":
                    age = lateral_view.header.get("PatientAge")
        elif singleview_frontals:
            frontal_xray = singleview_frontals[0].get_view().get_array()
            if age is None or age == "":
                age = singleview_frontals[0].header.get("PatientAge")
        if singleview_laterals and self.selection_plan is None:
            lateral_xray = singleview_laterals[0].get_view().get_array()
            if age is None or age == "":
                age = singleview_laterals[0].header.get("PatientAge")
        # decides to use either multiview dir or singleview dir for xrays
        # prefers using multiview if any of the directories have both views
        if multiview_dirs and self.selection_plan is None:
            multiviews_with_both_views = [
                multiview for multiview in multiview_dirs if multiview.has_both_views
            ]
            # print(multiviews_with_both_views)
            if multiviews_with_both_views:
                frontal_view = multiviews_with_both_views[0].get_by_view_type("frontal")
                lateral_view = multiviews_with_both_views[0].get_by_view_type("lateral")
                if frontal_view is not None:
                    frontal_xray = frontal_view.get_array()
                    if age is None or age == "":
                        age = frontal_view.header.get("PatientAge")
                if lateral_view is not None:
                    lateral_xray = lateral_view.get_array()
                    if age is None or age == "":
                        age = lateral_view.header.get("PatientAge")
            else:
                # handle the case where there are no directories with both views
                frontal_view = multiview_dirs[0].get_by_view_type(
                    "frontal"
                )  # may be None
                lateral_view = multiview_dirs[0].get_by_view_type(
                    "lateral"
                )  # may be None
                if frontal_view:
                    frontal_xray = frontal_view.get_array()
                    if age is None or age == "":
                        age = frontal_view.header.get("PatientAge")
                if lateral_view:
                    lateral_xray = lateral_view.get_array()
                    if age is None or age == "":
                        age = lateral_view.header.get("PatientAge")

        output_path = dirpath / self.patient_id / "ct_xray_data.h5"
        temporary_path = output_path.with_name(f"{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}")

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
        metadata["acquisition_date"] = str(dcm.get("AcquisitionDate", ""))
        metadata["spacing"] = ct_scan.spacing
        legacy_metadata = dict(metadata)
        metadata["schema_version"] = 1
        metadata["drr_temporal_status"] = "not_applicable_same_acquisition"
        destination = str(self.selection_plan.get("destination", "")) if self.selection_plan else ""
        mixed_real_views = bool(
            destination.startswith("mixed_")
            and frontal_xray is not None
            and lateral_xray is not None
        )
        metadata["training_xray_source"] = "real_xray" if mixed_real_views or destination.startswith("real_") else "drr"
        if self.selection_plan:
            metadata["selection_policy"] = self.selection_plan.get("policy")
            metadata["selection_status"] = self.selection_plan.get("status")
            metadata["intervals"] = self.selection_plan.get("intervals", {})
            metadata["memberships"] = self.selection_plan.get("memberships", [])
            metadata["destination"] = self.selection_plan.get("destination")
            metadata["respiratory_evidence"] = self.selection_plan.get("respiratory_evidence", {})
            metadata["selection_fingerprint"] = self.selection_plan.get("selection_fingerprint")
            metadata["processing_contract_version"] = self.selection_plan.get("processing_contract_version")

        # logs time taken to store data in h5 file
        start_time = time.time()
        self._publish_arrays(
            output_path, temporary_path, compression,
            {"ct": ct_volume,
             "xray1": frontal_xray if mixed_real_views else frontal_drr,
             "xray2": lateral_xray if mixed_real_views else lateral_drr,
             "real_xray1": frontal_xray, "real_xray2": lateral_xray},
            legacy_metadata, metadata,
        )
        if self.logger:
            self.logger.debug(
                f"Storing data to h5 file took {time.time() - start_time:.2f} seconds"
            )
        # finishes logging time taken to store data in h5 file

    def _publish_arrays(self, output_path, temporary_path, compression, arrays, legacy_metadata, metadata):
        """Keep the existing array pipeline; isolate transactional storage from selection."""
        plan = self.selection_plan or {}
        provenance = {
            "schema_version": 1,
            "patient_id": self.patient_id,
            "selection_policy": plan.get("policy", "legacy_largest_valid"),
            "memberships": plan.get("memberships", []),
            "destination": plan.get("destination"),
            "selection_fingerprint": plan.get("selection_fingerprint"),
            "processing_contract_version": plan.get("processing_contract_version"),
            "selected": plan.get("selected", {}),
            "written_ct_instances": getattr(plan.get("ct"), "source_instances", []),
            "training_xray_source": metadata["training_xray_source"],
            "drr": {"derived_from_role": "ct", "temporal_interval_status": "not_applicable_same_acquisition"},
        }
        destination = plan.get("destination", "")
        variant = destination.rsplit("_", 1)[0] or None
        try:
            with HDF5FileHandler(temporary_path, mode="overwrite", compression=compression) as stored:
                for key, array in arrays.items():
                    if array is not None:
                        stored.store_array(key, array)
                stored.store_dict("metadata", legacy_metadata)
                stored.store_json("metadata_json", metadata)
                stored.store_json("provenance_json", provenance)
            with HDF5FileHandler(temporary_path, mode="read_only") as stored:
                stored.validate_required_keys(variant)
                for key in ("metadata_json", "provenance_json"):
                    stored.read_json(key)
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def retrieve_from_file(self):
        h5f = HDF5FileHandler(self.patient_id, mode="read_only")
        data_dictionary = {}
        for key in h5f.file:
            data_dictionary[key] = h5f.retrieve_array(key)
        h5f.close()
        return data_dictionary

    def append_to_file(self, new_data):
        h5f = HDF5FileHandler(self.patient_id, mode="auto")
        for key, value in new_data.items():
            h5f.store_array(key, value)
        h5f.close()

    def validate_file(self):
        h5f = HDF5FileHandler(self.patient_id, mode="read_only")
        h5f.validate_required_keys()
        h5f.close()
