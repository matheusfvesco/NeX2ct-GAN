"""
Functions for saving and exporting files into the disc
"""

from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.dicom import (
        SingleViewXRayDirectory,
        MultiviewXRayDirectory,
        CTScanDirectory,
    )


class HDF5FileHandler:
    def __init__(
        self, filepath: str | Path, mode: str = "auto", compression: str | int = 0
    ) -> None:
        """
        compression:
            (String or int) Compression strategy. Legal values are 'gzip', 'szip', 'lzf'. If an integer in range(10), this indicates gzip compression level. Otherwise, an integer indicates the number of a dynamically loaded compression filter.
        """
        self.filepath = Path(filepath)
        if not self.filepath.suffix == ".hdf5":
            raise ValueError("The file must be an HDF5 file (.hdf5)")
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
        required_keys = ["ct", "frontal_drr", "lateral_drr"]
        for required_key in required_keys:
            if required_key not in self.file:
                raise ValueError(f"The required key '{required_key}' is missing")

    def close(self) -> None:
        self.file.close()

    def store_array(self, key: str, array: np.ndarray, force=False):
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
        self.file.create_dataset(key, data=array)

    def retrieve_array(self, key: str) -> np.ndarray:
        return np.array(self.file[key])


class PatientDataStorer:
    def __init__(self, patient_id: str, patient_data_dictionary: dict) -> None:
        self.patient_id = patient_id
        self.data_dictionary = patient_data_dictionary
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

    def store_to_file(self, dirpath: str | Path, compression: str | int = 0):
        """
        compression:
            (String or int) Compression strategy. Legal values are 'gzip', 'szip', 'lzf'. If an integer in range(10), this indicates gzip compression level. Otherwise, an integer indicates the number of a dynamically loaded compression filter.
        """
        dirpath = Path(dirpath)
        ct_scans: list[CTScanDirectory] = self.data_dictionary["cts"]
        singleview_dirs: list[SingleViewXRayDirectory] = self.data_dictionary["xrays"]
        multiview_dirs: list[MultiviewXRayDirectory] = self.data_dictionary["multiview"]

        ct_scan = ct_scans[0]
        ct_volume = ct_scan.get_array()
        drr_generator = ct_scan.drr_generator
        frontal_drr = drr_generator.frontal_view
        lateral_drr = drr_generator.lateral_view

        # creates lists containing frontal and lateral views of the x-rays
        singlewview_frontals = [
            singleview
            for singleview in singleview_dirs
            if singleview.view_type == "frontal"
        ]
        singlewview_laterals = [
            singleview
            for singleview in singleview_dirs
            if singleview.view_type == "lateral"
        ]

        frontal_xray = None
        lateral_xray = None

        if singleview_dirs or multiview_dirs:
            pass

        if singlewview_frontals:
            frontal_xray = singlewview_frontals[0].get_view().get_array()
        if singlewview_laterals:
            lateral_xray = singlewview_laterals[0].get_view().get_array()
        # decides to use either multiview dir or singleview dir for xrays
        # prefers using multiview if any of the directories have both views
        if multiview_dirs:
            multiviews_with_both_views = [
                multiview for multiview in multiview_dirs if multiview.has_both_views
            ]
            # print(multiviews_with_both_views)
            if multiviews_with_both_views:
                frontal_view = multiviews_with_both_views[0].get_by_view_type("frontal")
                lateral_view = multiviews_with_both_views[0].get_by_view_type("lateral")
                if frontal_view is not None:
                    frontal_xray = frontal_view.get_array()
                if lateral_view is not None:
                    lateral_xray = lateral_view.get_array()
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
                if lateral_view:
                    lateral_xray = lateral_view.get_array()

        h5f = HDF5FileHandler(
            filepath=dirpath / (self.patient_id + ".hdf5"), compression=compression
        )
        h5f.store_array("ct", ct_volume)
        h5f.store_array("frontal_drr", frontal_drr)
        h5f.store_array("lateral_drr", lateral_drr)
        if frontal_xray is not None:
            h5f.store_array("frontal_xray", frontal_xray)
        if lateral_xray is not None:
            h5f.store_array("lateral_xray", lateral_xray)

        h5f.close()

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
