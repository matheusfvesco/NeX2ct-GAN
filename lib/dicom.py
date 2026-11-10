from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import pydicom
from pydicom import FileDataset
from pydicom.multival import MultiValue
from pydicom.valuerep import DSfloat
from collections import defaultdict
import nibabel as nib

import numpy as np


# from scipy import ndimage # for applying masks
# from skimage import morphology # for applying masks
from functools import partial, wraps
from datetime import date, datetime
from abc import ABC

from lib.utils.viz import ArrayDisplayUtility
from lib.utils.utils import get_multival_by_index, Cache
from lib.process.preprocess import CTProcessor, XRayProcessor
from lib.process.drr import DRRGenerator
from lib.utils.dicom import XRayJsonClassifier
from lib.data.temporal import comparable_datetime, parse_acquisition_time
from lib.data.cohort import normalize_patient_id


def _parse_acquisition_datetime(dataset) -> dict:
    """Compatibility wrapper; optional acquisition metadata never rejects an image."""
    return parse_acquisition_time(dataset)


def get_middle_dicom_filedataset(func, *, header_only=False):
    """Pass a directory's representative object, or a retained metadata-only header."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        dicom = self.header if header_only else self.dicom
        return func(self, dicom, *args, **kwargs)

    return wrapper


# loads the file dataset if it is not already loaded
def get_dicom_filedataset(func):
    """If dicom FileDataset is already loaded in object(during init), uses it.
    Otherwise, loads from disk and passes it to function
    Class meant for File classes"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        dicom = (
            pydicom.dcmread(self.dicom_path)
            if not self.is_dicom_cached
            else self.cache.get("dicom")
        )
        return func(self, dicom, *args, **kwargs)

    return wrapper


get_dicom_header = partial(get_middle_dicom_filedataset, header_only=True)


class BaseDicomFile(ABC):
    """
    Gets information from a single .dcm file and makes it easy to access.
    """

    def __init__(self, dicom_path):
        super().__init__()
        self.dicom_path = Path(dicom_path)
        if not self.is_valid_path:
            raise FileNotFoundError("File is not a valid dicom")
        self.cache = Cache()

        self._load_dicom(stop_before_pixels=True)
        self._load_basic_tags()  # loads the tags into attributes
        self.is_ct: bool = self.modality == "CT"
        self.is_xray: bool = self.modality != "CT"

    @property
    def is_valid_path(self):
        accepted_extensions = [".dcm", ".dicom"]
        if not self.dicom_path.is_file():
            return False
        if self.dicom_path.suffix not in accepted_extensions:
            return False
        return True

    @property
    def is_dicom_cached(self):
        return self.cache.contains("dicom")

    @property
    def is_array_cached(self):
        return self.cache.contains("array")

    @property
    def is_raw_array_cached(self):
        return self.cache.contains("raw_array")

    @property
    def dicom(self) -> FileDataset:
        if "dicom" in self.cache:
            return self.cache["dicom"]
        else:
            return pydicom.dcmread(self.dicom_path)

    @property
    def header(self) -> FileDataset:
        """Metadata retained from construction; never reload PixelData for tags."""
        return self._header

    @property
    def raw_visualizer(self) -> ArrayDisplayUtility:
        """
        Returns a visualizer utility for the unprocessed array
        """
        return ArrayDisplayUtility(self.get_raw_array())

    @property
    def visualizer(self) -> ArrayDisplayUtility:
        """
        Returns a visualizer utility for the processed array
        """
        return ArrayDisplayUtility(self.get_array())

    def __str__(self):
        return f"{self.__class__.__name__}(dicom_path={self.dicom_path}, patient_id={self.patient_id})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.dicom_path == other.dicom_path
                and self.patient_id == other.patient_id
            )
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.dicom_path}')"

    @get_dicom_header
    def get_modality(self, dicom: FileDataset) -> str:
        return dicom.Modality

    @get_dicom_filedataset
    def get_raw_array(self, dicom: FileDataset) -> np.ndarray:
        """This is a test"""
        return dicom.pixel_array

    @get_dicom_filedataset
    def get_array(self, dicom: FileDataset) -> np.ndarray:
        processed_array = self._preprocess_array(dicom=dicom)

        return processed_array

    def _load_basic_tags(self):
        # ['AcquisitionDate', 'AcquisitionTime', 'BodyPartExamined', 'Modality',
        # 'PatientID', 'WindowCenter', 'WindowWidth', 'RescaleSlope', 'RescaleIntercept',
        # 'PhotometricInterpretation', 'BitsAllocated', 'BitsStored', 'Rows', 'Columns',
        # '(0013, 1010)']
        self.acquisition_time: dict = self.get_acquisition_time()
        self.acquisition_datetime: date | datetime | None = self.acquisition_time["parsed"]
        self.body_part_examined: str = self.get_body_part_examined()
        self.modality: str = self.get_modality()
        self.patient_id: str = self.get_patient_id()
        self.dataset_agnostic_patient_id: str = self.get_dataset_agnostic_patient_id()
        self.window_center, self.window_width = self.get_windowing_information()
        self.slope, self.intercept = self.get_rescale_information()
        self.photometric = self.get_photometric_interpretation()
        self.bits_allocated, self.bits_stored = self.get_bits_information()
        self.rows, self.columns = self.get_image_dimensions()
        self.dataset = self.get_dataset()

    @get_dicom_header
    def get_patient_id(self, dicom: FileDataset) -> str:
        return dicom.get("PatientID")

    @get_dicom_header
    def get_dataset_agnostic_patient_id(self, dicom: FileDataset) -> str:
        patient_id = str(dicom.PatientID)

        if "RICORD-1C" in patient_id:
            dataset_agnostic_patient_id = patient_id.replace("MIDRC-RICORD-1C-", "")
        elif "RICORD-1A" in patient_id:
            dataset_agnostic_patient_id = patient_id.replace("MIDRC-RICORD-1A-", "")
        else:
            dataset_agnostic_patient_id = patient_id

        return dataset_agnostic_patient_id

    @get_dicom_header
    def get_modality(self, dicom: FileDataset) -> str:
        return dicom.get("Modality")

    @get_dicom_header
    def get_acquisition_time(self, dicom: FileDataset) -> dict:
        return _parse_acquisition_datetime(dicom)

    @get_dicom_header
    def get_acquisition_datetime(self, dicom: FileDataset) -> date | datetime | None:
        """Compatibility accessor; missing/invalid timestamps no longer reject a DICOM."""
        return _parse_acquisition_datetime(dicom)["parsed"]

    @get_dicom_header
    def get_body_part_examined(self, dicom: FileDataset) -> str:
        return dicom.get("BodyPartExamined", None)

    @get_dicom_header
    def get_windowing_information(self, dicom: FileDataset) -> tuple[int, int]:
        window_center = get_multival_by_index(dicom.get("WindowCenter"), index=0)
        window_center = (
            int(window_center) if window_center is not None else window_center
        )
        window_width = get_multival_by_index(dicom.get("WindowWidth"), index=0)
        window_center = int(window_width) if window_width is not None else window_width
        return window_center, window_width

    @get_dicom_header
    def get_rescale_information(self, dicom: FileDataset) -> tuple[int, int]:
        return dicom.get("RescaleSlope"), dicom.get("RescaleIntercept")

    @get_dicom_header
    def get_photometric_interpretation(self, dicom: FileDataset) -> str:
        return dicom.get("PhotometricInterpretation")

    @get_dicom_header
    def get_bits_information(self, dicom: FileDataset) -> str:
        return dicom.get("BitsAllocated"), dicom.get("BitsStored")

    @get_dicom_header
    def get_image_dimensions(self, dicom: FileDataset) -> str:
        return dicom.get("Rows"), dicom.get("Columns")

    @get_dicom_header
    def get_dataset(self, dicom: FileDataset) -> str:
        element = dicom.get((0x0013, 0x1010))
        return element.value if element is not None else None

    @get_dicom_filedataset
    def _get_dicom(self, dicom: FileDataset):
        return dicom

    def _load_dicom(self, stop_before_pixels=False):
        dataset = pydicom.dcmread(self.dicom_path, stop_before_pixels=stop_before_pixels)
        if stop_before_pixels:
            self._header = dataset
        self.cache.add("dicom", dataset)
        return True

    def _unload_dicom(self):
        self.cache.remove("dicom")
        return True


class GenericDicomFile(BaseDicomFile):
    """Interface class for the abstract class"""

    def __init__(self, dicom_path):
        super().__init__(dicom_path)

    def _preprocess_array(self, dicom: FileDataset) -> np.ndarray:
        pass


class XRayView(BaseDicomFile):
    """Represents a single X-Ray image.
    Gets information from a single .dcm file and makes it easy to access.
    """

    def __init__(self, dicom_file: GenericDicomFile) -> None:
        for attr in vars(dicom_file):
            setattr(self, attr, getattr(dicom_file, attr))
        if self.modality == "CT":
            raise TypeError("""Dicom file is a CT slice, not a X-Ray""")
        self._load_modality_tags()
        self.is_original: bool = self.image_type == "ORIGINAL"
        self.is_derived: bool = self.image_type == "DERIVED"
        self.is_cr: bool = self.modality == "CR"
        self.is_dx: bool = self.modality == "DX"
        self._unload_dicom()

    @property
    def is_frontal(self):
        return self.view_type == "frontal"

    @property
    def is_lateral(self):
        return self.view_type == "lateral"

    @property
    def is_valid(self):
        """
        True if it is a CR, or if it is a DX and is the original acquisition
        """
        return self.is_cr or (self.is_dx and self.is_original)

    def _load_modality_tags(self):
        # actual dicom view position tag
        self.view_position = self._get_view_position()
        # frontal, lateral, etc.
        self.view_type = self.get_view_type()
        # original, derived, etc.
        self.image_type = self._get_image_type()

    def get_array(self) -> np.ndarray:
        """
        Returns the preprocessed array of the X Ray View
        """
        processed_array = XRayProcessor(self.get_raw_array(), self).process()

        return processed_array

    @get_dicom_header
    def _get_windowing_information(
        self, dicom: FileDataset, index: int = 0
    ) -> tuple[int, int, str]:
        window_center = int(
            get_multival_by_index(dicom.get("WindowCenter"), index=index)
        )
        window_width = int(get_multival_by_index(dicom.get("WindowWidth"), index=index))
        photometric = str(dicom.get("PhotometricInterpretation"))
        return window_center, window_width, photometric

    def get_view_type(self):
        try:
            classifier = XRayJsonClassifier()
            class_name = classifier.classify(self.dicom_path)
            self.view_type_source = "manual_annotation"
            return class_name
        except (FileNotFoundError, KeyError, ValueError):
            position = str(self.view_position or "").strip().upper()
            if position in {"AP", "PA", "AP SUPINE", "PA ERECT"}:
                self.view_type_source = "dicom_view_position"
                return "frontal"
            if position in {"LAT", "LATERAL", "LL", "RL"}:
                self.view_type_source = "dicom_view_position"
                return "lateral"
            self.view_type_source = "unresolved"
            return None

    @get_dicom_header
    def _get_view_position(self, dicom: FileDataset) -> str:
        return dicom.get("ViewPosition")

    @get_dicom_header
    def _get_image_type(self, dicom: FileDataset) -> str:
        image_type = dicom.get("ImageType")  # this is a list of two itens.
        return image_type[0] if image_type else None


class CTSlice(BaseDicomFile):
    """Represents a single X-Ray image.
    Gets information from a single .dcm file and makes it easy to access.
    """

    def __init__(self, dicom_file: GenericDicomFile) -> None:
        for attr in vars(dicom_file):
            setattr(self, attr, getattr(dicom_file, attr))
        if self.modality != "CT":
            raise TypeError("""Dicom file is a X-Ray, not a CT slice""")
        self._load_modality_tags()

        # del self.dicom  # deletes dicom to avoid high memory usage
        self._unload_dicom()

    @property
    def x(self) -> int:
        return self.image_position_patient[0]

    @property
    def y(self) -> int:
        return self.image_position_patient[1]

    @property
    def z(self) -> int:
        return self.image_position_patient[2]

    @property
    def voxel_spacing_x(self):
        return self.pixel_spacing[0]

    @property
    def voxel_spacing_y(self):
        return self.pixel_spacing[1]

    def _load_modality_tags(self):
        self.image_position_patient: tuple[int, int, int] = (
            self.get_image_position_patient()
        )
        self.image_orientation_patient = self._get_image_orientation_patient()
        self.pixel_spacing: tuple[float, float] = self._get_pixel_spacing()

    @get_dicom_header
    def get_image_position_patient(self, dicom: FileDataset) -> tuple:
        return tuple(dicom.ImagePositionPatient)  # return (x, y, z)

    @get_dicom_header
    def _get_windowing_information(
        self, dicom: FileDataset, index: int = 0
    ) -> tuple[int, int, int, int]:
        self.window_center = int(
            get_multival_by_index(dicom.get("WindowCenter"), index)
        )
        self.window_width = int(get_multival_by_index(dicom.get("WindowWidth"), index))
        self.intercept = int(
            get_multival_by_index(dicom.get("RescaleIntercept"), index)
        )
        self.slope = int(get_multival_by_index(dicom.get("RescaleSlope"), index))
        return self.window_center, self.window_width, self.intercept, self.slope

    @get_dicom_header
    def _get_raw_windowing_information(
        self, dicom: FileDataset
    ) -> tuple[int, int, int, int]:
        window_center = dicom.get("WindowCenter")
        window_width = dicom.get("WindowWidth")
        intercept = dicom.get("RescaleIntercept")
        slope = dicom.get("RescaleSlope")
        return window_center, window_width, intercept, slope

    @property
    @get_dicom_header
    def has_windowing_information(self, dicom: FileDataset):
        tags = ["WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope"]
        return all(dicom.get(tag) is not None for tag in tags)
        # check if all these tags are not None and return either True or False

    @get_dicom_header
    def _get_image_orientation_patient(self, dicom: FileDataset):
        return dicom.get("ImageOrientationPatient", None)

    @get_dicom_header
    def _get_pixel_spacing(self, dicom: FileDataset) -> tuple[float, float]:
        return tuple(map(float, dicom.PixelSpacing))

    def get_array(self) -> np.ndarray:
        # processed_array = self._preprocess_array(dicom=dicom)

        return self.get_raw_array()


class DicomFileFactory(CTSlice, XRayView):
    """
    Base class that returns specific class related to the exam inside the directory
    Should return CTScan, SingleviewXRay or MultiviewXRay
    """

    def __new__(cls, dicom_path: Path) -> CTSlice | XRayView:
        dicom_directory = GenericDicomFile(dicom_path=dicom_path)

        if dicom_directory.is_ct:
            return CTSlice(dicom_directory)
        elif dicom_directory.is_xray:
            return XRayView(dicom_directory)
        else:
            raise ValueError(
                """Directory is invalid. It is an incomplete CT Scan or not a dicom directory"""
            )


class DicomFile:
    """Interface for the DicomFileFactory. Use this to create a specific
    DicomFile class from an unknown file."""

    def __new__(cls, dicom_path: Path) -> DicomFileFactory:
        return DicomFileFactory(dicom_path)


class BaseDicomDir(ABC):
    def __init__(self, directory_path, *, dicom_paths=None):
        super().__init__()
        accepted_extensions = [".dcm", ".dicom"]
        self.path_directory = Path(directory_path)
        self._paths_to_dicom_files = [
            file_path
            for extension in accepted_extensions
            for file_path in self.path_directory.glob(f"*{extension}")
        ]
        if dicom_paths is not None:
            self._paths_to_dicom_files = [Path(path) for path in dicom_paths]

        if not self.is_valid_path:
            raise ValueError(
                """Dicom directory is invalid. It is not a Directory or doesn't contain dicom files."""
            )
        self.cache = Cache()
        self._load_dicom()  # loads middle dcm into the cache
        self._header = self.dicom.header
        self.modality = self._get_modality()
        self.is_ct = True if self.modality == "CT" else False
        self.is_multiview = True if not self.is_ct and len(self) > 1 else False
        self.is_singleview = True if not self.is_ct and len(self) == 1 else False

        self.is_complete_ct = True if self.is_ct and len(self) > 20 else False

        if self.is_ct:
            self.dir_type = "CT"
        elif self.is_multiview:
            self.dir_type = "multiview"
        elif self.is_singleview:
            self.dir_type = "singleview"
        else:
            raise ValueError("Dicom Directory doesn't belong to any dir type")

        self.dataset: str = self._get_dataset_name()
        self.acquisition_time: dict = self._get_acquisition_time()
        self.acquisition_datetime: date | datetime | None = self._get_acquisition_datetime()
        self.dataset_agnostic_patient_id: str = self._get_dataset_agnostic_patient_id()

    @property
    def is_valid_path(self):
        if not self.path_directory.is_dir():
            return False
        if not self._paths_to_dicom_files:
            return False
        return True

    @property
    def is_dicom_cached(self):
        return self.cache.contains("dicom")

    @property
    def dicom(self) -> BaseDicomFile:
        if "dicom" in self.cache:
            return self.cache["dicom"]
        else:
            return self.get_middle_dcm()

    @property
    def header(self) -> FileDataset:
        return self._header

    @property
    def is_array_cached(self):
        return self.cache.contains("array")

    @property
    def is_raw_array_cached(self):
        return self.cache.contains("raw_array")

    # Magic methods

    def __len__(self) -> int:
        return len(self._paths_to_dicom_files)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise ValueError("This class doesn't support slicing")
        return DicomFile(self._paths_to_dicom_files[index])

    def __str__(self):
        return f"{self.__class__.__name__}(directory_path={self.path_directory}, dir_type={self.dir_type})"

    def __repr__(self):
        return f"{self.__class__.__name__}(directory_path={self.path_directory!r}, dir_type={self.dir_type!r})"

    def __iter__(self):
        """
        Returns a DicomFile for each of the dicoms inside the directory
        """
        # Return a generator that yields DICOM files loaded by the ThreadPoolExecutor
        for path in self._paths_to_dicom_files:
            yield DicomFile(path)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.path_directory == other.path_directory
        return False

    # High level methods

    def get_first_dcm(self) -> DicomFile:
        return self._get_first_dcm()

    def get_middle_dcm(self) -> DicomFile:
        return self._get_middle_dcm()

    def get_last_dcm(self) -> DicomFile:
        return self._get_last_dcm()

    @property
    def raw_visualizer(self) -> ArrayDisplayUtility:
        """
        Returns a visualizer utility for the unprocessed array
        """
        return ArrayDisplayUtility(self.get_raw_array())

    @property
    def visualizer(self) -> ArrayDisplayUtility:
        """
        Returns a visualizer utility for the processed array
        """
        return ArrayDisplayUtility(self.get_array())

    # Data loading methods

    def _get_first_dcm(self) -> DicomFile:
        return DicomFile(self._paths_to_dicom_files[0])

    def _get_middle_dcm(self) -> DicomFile:
        return DicomFile(self._paths_to_dicom_files[self.__len__() // 2])

    def _get_last_dcm(self) -> DicomFile:
        return DicomFile(self._paths_to_dicom_files[-1])

    def _unload_dicom(self) -> bool:
        self.cache.remove("dicom")
        return True

    def _load_dicom(self) -> bool:
        self.cache.add("dicom", self.get_middle_dcm())
        return True

    # Processing after loading data

    def get_all_dicoms(self, ordered_by: str = "") -> list[BaseDicomFile]:
        """
        Returns a list containing all the individual views.

        :param ordered_by: The key by which the DICOMs should be ordered.
          If empty, the DICOMs are returned unordered.
        :type ordered_by: str, optional
        :return: A list of all individual views, either ordered or unordered.
        :rtype: list[DicomFile]
        """
        if ordered_by:
            return self._get_all_dicoms_ordered(key=ordered_by)
        else:
            return self._get_all_dicoms_unordered()

    def _get_all_dicoms_unordered(self) -> list:
        return self._get_all_dcms_unordered()

    def _get_all_dcms_unordered(self) -> list:
        start_time = time.time()  # Record the start time

        def create_dicom_file(path):
            return DicomFile(path)

        # Use ThreadPoolExecutor to create DicomFile objects in parallel
        with ThreadPoolExecutor() as executor:
            dicom_files = list(
                executor.map(create_dicom_file, self._paths_to_dicom_files)
            )

        end_time = time.time()  # Record the end time

        execution_time = end_time - start_time  # Calculate the execution time
        print(
            f"Execution time load all dicoms: {execution_time} seconds"
        )  # Print the execution time

        return dicom_files

    def _get_array(self):
        dicoms = self.get_all_dicoms()

        for dicom in dicoms:
            dicom.get_raw_array()

    # consider using the .get() method
    def _get_all_dicoms_ordered(self, key: str) -> list:
        return sorted(self._get_all_dcms_unordered(), key=lambda dcm: dcm[key])

    @get_middle_dicom_filedataset
    def _get_acquisition_time(self, dicom: BaseDicomFile) -> dict:
        return dicom.acquisition_time

    @get_middle_dicom_filedataset
    def _get_acquisition_datetime(self, dicom: BaseDicomFile) -> date | datetime | None:
        return dicom.acquisition_datetime

    @get_middle_dicom_filedataset
    def _get_modality(self, dicom: BaseDicomFile):
        return dicom.modality

    @get_middle_dicom_filedataset
    def _get_dataset_name(self, dicom: BaseDicomFile):
        return str(dicom.dataset)

    @get_middle_dicom_filedataset
    def _get_dataset_agnostic_patient_id(self, dicom: BaseDicomFile):
        return dicom.dataset_agnostic_patient_id

    def get_most_recent(self):
        return self._ordered_dated_dicoms()[-1]

    def get_oldest(self):
        return self._ordered_dated_dicoms()[0]

    def _ordered_dated_dicoms(self):
        dated = []
        for dicom in self.get_all_dicoms():
            record = dicom.acquisition_time
            timestamp = comparable_datetime({"acquisition_datetime": record["parsed"], "time_status": record["status"]})
            if timestamp is not None:
                dated.append((timestamp, record["precision"], dicom))
        profiles = {(precision, timestamp.utcoffset() is not None) for timestamp, precision, _ in dated}
        if len(profiles) != 1:
            raise ValueError("Cannot order acquisitions with missing or incomparable timestamps")
        return [dicom for _, _, dicom in sorted(dated, key=lambda row: (row[0], str(row[2].dicom_path)))]


class GenericDicomDirectory(BaseDicomDir):
    """Interface for the BaseDicomDir."""

    def __init__(self, directory_path, *, dicom_paths=None) -> None:
        super().__init__(directory_path=directory_path, dicom_paths=dicom_paths)


class CTScanDirectory(BaseDicomDir):
    """Represents a Directory that contains multiple CT Scan Slices"""

    def __init__(self, dicom_directory: GenericDicomDirectory) -> None:
        # super().__init__(dicom_dir.path)

        # initializes function with same attributes as DicomDir
        # Should save some IO by not reading middle dcm again
        for attr in vars(dicom_directory):
            setattr(self, attr, getattr(dicom_directory, attr))

        self.plane: str = self._get_image_plane()

        self.window_type: bool = self._get_scan_window_type()

        self._unload_dicom()

    def get_array(self) -> np.ndarray:
        if "array" in self.cache:
            return self.cache.get("array")

        array = self._get_array()
        self.cache.add(key="array", value=array)
        return self.cache.get("array")

    def get_raw_array(self, **kwargs) -> np.ndarray:
        if "raw_array" in self.cache:
            return self.cache.get("raw_array")

        raw_array = self._get_raw_array(**kwargs)
        self.cache.add(key="raw_array", value=raw_array)
        return self.cache.get("raw_array")

    @property
    def drr_generator(self) -> DRRGenerator:
        return DRRGenerator(self)

    @property
    def is_axial(self) -> bool:
        return self.plane == "ax"

    @property
    def is_coronal(self) -> bool:
        return self.plane == "cor"

    @property
    def is_saggital(self) -> bool:
        return self.plane == "sag"

    @property
    def is_lung_acquisition(self) -> bool:
        return self.window_type == "lung"

    @property
    def is_standard_acquisition(self) -> bool:
        return self.window_type == "standard"

    @property
    def is_valid_scan(self) -> bool:
        """
        A Scan will be considered valid if it contains plane
        (ax, sag, cor) information, a window_type (standard or lung)
        and if it has at least 20 slices
        """
        return bool(self.plane and self.window_type and len(self) > 20)

    @property
    def is_valid_axial_scan(self) -> bool:
        """
        A Scan will be considered valid if it is a valid scan
        (contains plane (ax, sag, cor) information, a window_type
        (standard or lung) and if it has at least 20 slices) and if it
        is an axial acquisition
        """
        return bool(self.is_valid_scan and self.is_axial)

    @property
    def spacing(self) -> tuple[float, float, float]:
        return self._get_voxel_spacing()

    @property
    def _is_reversed(self) -> bool:
        """
        Returns True if the CTFolder is likely a reversed case. Useful for the demos.
        HIGHLY INNEFICIENT, USE FOR DEBUGGING PURPOSES
        """
        last: CTSlice = self.get_last_dcm()
        middle: CTSlice = self.get_middle_dcm()
        # first = self.get_first_dcm()
        dicoms = [middle, last]
        z_position_sorted = sorted(dicoms, key=lambda dicom: dicom.z, reverse=True)
        name_sorted = sorted(
            dicoms, key=lambda dicom: dicom.dicom_path.stem, reverse=False
        )
        return z_position_sorted != name_sorted

    @property
    @get_middle_dicom_filedataset
    def affine(self, dicom: CTSlice) -> np.ndarray:
        """
        Generates a 4x4 affine matrix for the CT volume.
        """
        # Get all dicoms and sort them by z position descending
        # This matches the order in _get_raw_array when use_z_ordering=True and reverse_order=False
        dicoms = list(self)
        dicoms_sorted = sorted(dicoms, key=lambda d: d.z, reverse=True)

        if not dicoms_sorted:
            return np.eye(4)

        first = dicoms_sorted[0]
        last = dicoms_sorted[-1]

        n = len(dicoms_sorted)

        # ImageOrientationPatient is a list of 6 floats: [i1, i2, i3, i4, i5, i6]
        # i1, i2, i3 are the row direction cosine
        # i4, i5, i6 are the column direction cosine
        orientation = dicom.image_orientation_patient
        if orientation is None:
            orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        else:
            orientation = [float(x) for x in orientation]

        i1, i2, i3, i4, i5, i6 = orientation

        # Pixel spacing is (row_spacing, col_spacing)
        dr, dc = dicom.pixel_spacing

        # ImagePositionPatient is (x, y, z)
        t1 = [float(x) for x in first.image_position_patient]
        tn = [float(x) for x in last.image_position_patient]

        affine = np.zeros((4, 4))

        # Column 0: row direction cosine * row spacing
        affine[0, 0] = i4 * dr
        affine[1, 0] = i5 * dr
        affine[2, 0] = i6 * dr

        # Column 1: column direction cosine * column spacing
        affine[0, 1] = i1 * dc
        affine[1, 1] = i2 * dc
        affine[2, 1] = i3 * dc

        # Column 2: slice step
        if n > 1:
            affine[0, 2] = (tn[0] - t1[0]) / (n - 1)
            affine[1, 2] = (tn[1] - t1[1]) / (n - 1)
            affine[2, 2] = (tn[2] - t1[2]) / (n - 1)
        else:
            # Fallback for single slice
            affine[0, 2] = 0.0
            affine[1, 2] = 0.0
            affine[2, 2] = 1.0

        # Column 3: origin
        affine[0, 3] = t1[0]
        affine[1, 3] = t1[1]
        affine[2, 3] = t1[2]
        affine[3, 3] = 1.0

        # Convert from DICOM LPS+ to NIfTI RAS+
        affine[0, :] = -affine[0, :]
        affine[1, :] = -affine[1, :]

        return affine

    @get_middle_dicom_filedataset
    def _get_windowing_information(self, dicom: CTSlice, index: int = 0):
        return dicom._get_windowing_information(index=index)

    def _record_source_instances(self, files, dicoms):
        """Use already-loaded instances for provenance and patient/series checks."""
        self.source_instances = []
        for path, dicom in zip(files, dicoms):
            if normalize_patient_id(dicom.get("PatientID")) != normalize_patient_id(self.header.get("PatientID")):
                raise ValueError(f"CT instance PatientID differs from its selected series: {path}")
            expected_series = self.header.get("SeriesInstanceUID")
            if expected_series and dicom.get("SeriesInstanceUID") != expected_series:
                raise ValueError(f"CT instance SeriesInstanceUID differs from its selected series: {path}")
            self.source_instances.append({
                "path": str(path), "study_instance_uid": dicom.get("StudyInstanceUID"),
                "series_instance_uid": dicom.get("SeriesInstanceUID"),
                "sop_instance_uid": dicom.get("SOPInstanceUID"),
                "used_for_volume": int(dicom.get("Rows")) == 512 and int(dicom.get("Columns")) == 512,
            })

    def _get_raw_array(
        self, use_z_ordering=True, reverse_order=False, **kwargs
    ) -> np.ndarray:
        if use_z_ordering:  # usual step, doesn't order files before loading
            files = self._paths_to_dicom_files
            dicoms = [pydicom.dcmread(path) for path in files]
        else:
            files = sorted(
                [path for path in self._paths_to_dicom_files],
                key=lambda x: x.name,
                reverse=reverse_order,  # should usually order in "0, 1, 2", and if reversed, "2, 1, 0"
            )
            dicoms = [pydicom.dcmread(path) for path in files]

        self._record_source_instances(files, dicoms)

        # filters arrays to only arrays with shape 512x512
        dicoms = [
            dicom
            for dicom in dicoms
            if int(dicom.get("Rows")) == 512 and int(dicom.get("Columns")) == 512
        ]

        # sorts slices by z location relative to patient
        if use_z_ordering:
            dicoms = sorted(
                dicoms,
                key=lambda x: tuple(x.get("ImagePositionPatient"))[2],
                # this should usually order in "2, 1, 0" and if reversed "0, 1, 2"
                reverse=not reverse_order,
            )

        arrays = [dicom.pixel_array for dicom in dicoms]

        array = np.stack(arrays, axis=-1)
        return array

    def _get_array(self) -> np.ndarray:
        array = self._get_raw_array()
        # array = self._preprocess_array(array=array, shift_0=shift_0)
        # pass the entire scan directory instead of just a slice
        array = CTProcessor(array=array, scan=self).process()
        return array

    def _get_processor(self, shift_0: bool = True, **kwargs) -> CTProcessor:
        array = self._get_raw_array(**kwargs)
        # array = self._preprocess_array(array=array, shift_0=shift_0)
        # the processor now expects the scan rather than a single slice
        processor = CTProcessor(array=array, scan=self)
        return processor

    @get_middle_dicom_filedataset
    def _get_voxel_spacing(self, dicom: CTSlice) -> tuple[float, float, float]:
        # Initialize an empty list to store the z positions
        z_positions = []

        # Loop through all DICOM files and add the z position to the list
        for ctslice in self:
            z_positions.append(ctslice.z)

        # Sort the z positions
        z_positions.sort(reverse=True)

        # Calculate the voxel spacings in the x and y directions
        voxel_spacing_x, voxel_spacing_y = dicom.voxel_spacing_x, dicom.voxel_spacing_y

        # Calculate the differences between consecutive z positions
        distance_slices_z = np.diff(z_positions)

        # Calculate the voxel spacing in the z direction (slice thickness)
        voxel_spacing_z = float(np.abs(np.unique(distance_slices_z)[0]))

        # Return the voxel spacings
        spacing = (voxel_spacing_x, voxel_spacing_y, voxel_spacing_z)

        return spacing

    @property
    @get_middle_dicom_filedataset
    def has_windowing_information(self, dicom: CTSlice):
        return dicom.has_windowing_information

    @get_middle_dicom_filedataset
    def _get_image_plane(self, ctslice: CTSlice) -> str:
        """Returns a string with the plane listed on the DICOM tags.

        Args:
            dcm (FileDataset): The DICOM file.
            verbose (bool, optional): If True, prints additional information. Defaults to False.

        Returns:
            str: The plane of the image. Returns 'cor' if the image is on the coronal plane,
            'sag' if the image is on the saggital plane, 'ax' if the image is on the axial plane,
            and 'scout' if the image is a scout image. Raises an error if orientation couldn't be found.
        """

        # gets ImageOrientationPatient parameter(list)
        orientation = ctslice.image_orientation_patient

        # converts orien metadata to hardcoded if statements format
        orientation = [int(item) for item in orientation]

        if orientation == [0, 1, 0, 0, 0, 0]:  # exception
            return "sag"

        if orientation == [0, 1, 0, 0, 0, -1]:
            return "sag"

        if orientation == [1, 0, 0, 0, 0, 0]:
            return "cor"

        if orientation == [1, 0, 0, 0, 0, -1]:
            return "cor"

        if orientation == [1, 0, 0, 0, 1, 0]:
            return "ax"

        if orientation == [0, 0, 0, 0, 0, 0]:
            return "ax"

        if orientation == [
            -1,
            0,
            0,
            0,
            -1,
            0,
        ]:  # exception patient LIDC-IDRI-0415
            return "ax"

        if orientation == [0, -1, 0, 0, 0, -1]:  # scout
            return "scout"

        raise ValueError(
            f"""
        Orientation couldn't be found for patient {ctslice.dataset_agnostic_patient_id}.
        Orientation: {orientation}
        """
        )

    @get_middle_dicom_filedataset
    def _get_scan_window_type(self, dicom: CTSlice) -> str:
        """Returns the window type of a DICOM image.

        Args:
            dcm (FileDataset): The DICOM file.
            verbose (bool, optional): If True, prints additional information. Defaults to False.

        Returns:
            Optional[str]: The window type of the image. Returns 'lung' if the window is a lung acquisition,
            'standard' if the window is a regular thorax CT, and None otherwise.
        """
        window_center, window_width, _, _ = dicom._get_raw_windowing_information()

        if isinstance(window_center, DSfloat):
            return "lung"

        if isinstance(window_center, MultiValue):
            return "standard"

        raise ValueError(
            f"""Window type couldn't be found for patient {dicom.patient_id}.
        Window center: {window_center}, type: {type(window_center)}
        Window width(not used for this func): {window_width}, type: {type(window_width)}

        Expected: DSfloat if lung window, and MultiValue if standard window
        """
        )


class SingleViewXRayDirectory(BaseDicomDir):
    """Represents a Directory that contains only one XRayView"""

    def __init__(self, dicom_dir: GenericDicomDirectory):
        # initializes function with same attributes as DicomDir
        # Should save some IO by not reading middle dcm again
        for attr in vars(dicom_dir):
            setattr(self, attr, getattr(dicom_dir, attr))

        self.cache.add("views_by_path", {self._paths_to_dicom_files[0]: self.dicom})
        self._unload_dicom()

    @property
    def is_frontal(self):
        return self.view_type == "frontal"

    @property
    def is_lateral(self):
        return self.view_type == "lateral"

    @property
    def view_position(self) -> str:
        """
        Getter for 'view_position' attribute.

        This property represents the viewing position of the image in a DICOM file,
        such as "", "", etc., based on the value obtained from '_get_view_position()' method.

        Returns:
            str: The view position of the DICOM image.
        """
        return self[0].view_position

    @property
    def view_type(self) -> str:
        """
        Getter for 'view_type' attribute.

        This property represents the type of viewing of the DICOM image,
        such as "frontal", "lateral", etc., based on the value obtained from 'get_view_type()' method.
        This uses the manually annotated json.

        Returns:
            str: The view type of the DICOM image.
        """
        return self.get_view().view_type

    @property
    def image_type(self) -> str:
        """
        Getter for 'image_type' attribute.

        This property represents the type of the DICOM image,
        such as "original", "derived", etc., based on the value obtained from '_get_image_type()' method.

        Returns:
            str: The image type of the DICOM image.
        """
        return self[0].image_type

    def get_view(self):
        views = self.cache.get("views_by_path", {})
        return views[self._paths_to_dicom_files[0]] if views else self[0]


class MultiviewXRayDirectory(BaseDicomDir):
    """Represents a Directory that contains multiple XRayViews"""

    def __init__(self, dicom_dir: GenericDicomDirectory):
        # initializes function with same attributes as DicomDir
        # Should save some IO by not reading middle dcm again
        for attr in vars(dicom_dir):
            setattr(self, attr, getattr(dicom_dir, attr))

        self._generate_view_types_dictionaries()

        self._unload_dicom()

    @property
    def has_frontal(self):
        return self.check_has_view("frontal")

    @property
    def has_lateral(self):
        return self.check_has_view("lateral")

    @property
    def has_both_views(self):
        return self.has_frontal and self.has_lateral

    def _generate_view_types_dictionaries(self):
        """
        Generates two dictionaries: one where each key is a path to a
        .dcm file and each value its respective view_type, and another
        where each key is a view_type ('frontal' or 'lateral') and its
        value is a list of the filepaths that refer to that view_type
        """
        representative = self.dicom
        views = {
            path: representative if path == representative.dicom_path else DicomFile(path)
            for path in self._paths_to_dicom_files
        }
        self.cache.add("views_by_path", views)
        self.filenames_to_view_type = {
            path: view.view_type for path, view in views.items()
            if view.view_type and not (view.is_dx and view.is_derived)
        }

        # creates the inverse of the filenames_to_view_types dictionary,
        # so each key(view_type) returns a list of the filenames that
        # refer to a dicom that contains that specific view_type
        self.view_type_to_filename = defaultdict(list)
        for key, value in self.filenames_to_view_type.items():
            self.view_type_to_filename[value].append(key)

    def check_has_view(self, type: str):
        return bool(self.view_type_to_filename[type])

    def get_by_view_type(self, type: str) -> XRayView | None:
        # dcms = self._get_all_dcms_unordered()
        if type == "frontal" and not self.has_frontal:
            return None

        if type == "lateral" and not self.has_lateral:
            return None

        cached_views = self.cache.get("views_by_path")
        for view in (cached_views.values() if cached_views is not None else self):
            if view.view_type != type:
                continue
            if view.is_dx:
                if view.is_derived:  # if the image is not the original acquisition...
                    continue
            return view
        raise ValueError("MultiviewXRayDirectory doesn't contain the desired view")


class DicomDirectoryFactory(
    CTScanDirectory, SingleViewXRayDirectory, MultiviewXRayDirectory
):
    """
    Base class that returns specific class related to the exam inside the directory
    Should return CTScan, SingleviewXRay or MultiviewXRay
    """

    def __new__(
        cls, directory_path: str, *, dicom_paths=None
    ) -> CTScanDirectory | SingleViewXRayDirectory | MultiviewXRayDirectory:
        dicom_directory = GenericDicomDirectory(directory_path, dicom_paths=dicom_paths)

        if dicom_directory.is_ct:
            return CTScanDirectory(dicom_directory)
        elif dicom_directory.is_singleview:
            return SingleViewXRayDirectory(dicom_directory)
        elif dicom_directory.is_multiview:
            return MultiviewXRayDirectory(dicom_directory)
        else:
            raise ValueError(
                """Directory is invalid. It is an incomplete CT Scan or not a dicom directory"""
            )


class DicomDirectory:
    """Interface for the DicomDirectoryFactory. Use this to create a specific
    DicomDirectory class from an unknown file."""

    def __new__(cls, directory_path: str, *, dicom_paths=None) -> DicomDirectoryFactory:
        return DicomDirectoryFactory(directory_path, dicom_paths=dicom_paths)


class XRayFactory:
    def __new__(cls, dicom_directory: GenericDicomDirectory):
        instance = dicom_directory
        # instance = super().__new__(cls)
        # instance.__init__(dicom_dir)
        if instance.is_multiview:
            return MultiviewXRayDirectory(instance)
        else:
            return SingleViewXRayDirectory(instance)
