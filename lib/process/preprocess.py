from __future__ import annotations

import numpy as np
from abc import ABC
import cv2


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.dicom import BaseDicomFile, CTSlice, XRayView


class BaseProcessor(ABC):
    def __init__(
        self,
        array: np.ndarray,
        dicom: BaseDicomFile | None = None,
    ) -> None:
        array = array.astype(np.float32)
        self.raw_array = array  # keeps the original array intact
        self.array = array.copy()  # creates copy to store the changes
        self.dicom = dicom

    @property
    def is_array_all_positive(self):
        """Checks wheter there are values less than zero on the
        processed array
        """
        return np.min(self.array) > 0

    @property
    def is_raw(self):
        """Dynamically evaluates if the array is still raw, by
        comparing it to the raw_array. If they are equal, it assumes
        the array is still raw."""
        return np.array_equal(self.array, self.raw_array)

    @property
    def is_processed(self):
        """Dynamically evaluates if the array has been processed, by
        comparing it to the raw_array. If they are not equal, it assumes
        processing has been done."""
        return not self.is_raw

    # base tags

    @property
    def acquisition_datetime(self):
        return self.dicom.acquisition_datetime

    @property
    def body_part_examined(self):
        return self.dicom.body_part_examined

    @property
    def modality(self):
        return self.dicom.modality

    @property
    def patient_id(self):
        return self.dicom.patient_id

    @property
    def dataset_agnostic_patient_id(self):
        return self.dicom.dataset_agnostic_patient_id

    @property
    def window_center(self):
        return self.dicom.window_center

    @property
    def window_width(self):
        return self.dicom.window_width

    @property
    def slope(self):
        return self.dicom.slope

    @property
    def intercept(self):
        return self.dicom.intercept

    @property
    def photometric(self):
        return self.dicom.photometric

    @property
    def bits_allocated(self):
        return self.dicom.bits_allocated

    @property
    def bits_stored(self):
        return self.dicom.bits_stored

    @property
    def rows(self):
        return self.dicom.rows

    @property
    def columns(self):
        return self.dicom.columns

    @property
    def dataset(self):
        return self.dicom.dataset

    def reset_array(self):
        self.array = self.raw_array.copy()


class XRayProcessor(BaseProcessor):
    def __init__(
        self,
        array: np.ndarray,
        dicom: XRayView | None = None,
    ) -> None:
        """
        Initialize an instance of XRayProcessor.
        Args:
            array (np.ndarray): The input 2D image array.
            dicom (FileDataset, optional): The corresponding DICOM file metadata. Defaults to None.
        """
        super().__init__(array=array, dicom=dicom)
        self.dicom: XRayView = self.dicom

    @property
    def view_position(self):
        """
        Prefer using view_type
        """
        return self.dicom.view_position

    @property
    def view_type(self):
        return self.dicom.view_type

    def process(self) -> np.ndarray:
        """
        Process the input image array using the following steps (in order):
            1. Apply photometric transformation.
            2. Apply windowing transformation.

        Returns:
            np.ndarray: The processed image array.
        """
        self._apply_photometric()
        # self._apply_window_to_array()
        # self._set_max_to_min()
        self._normalize()

        return self.array

    def _process_compose(self, transforms: list[str]):
        """
        For testing different pipelines. Use for debugging only.
        Inspired by the pytorch transforms.Compose.
        Applies the transforms in any order, based on the provided list
        of 'transforms'.
        Args:
            transforms (list[str]): The list of transformations to apply.

        Returns:
            np.ndarray: The processed image array.
        """
        # applies the processing in any order
        for transform in transforms:
            self._process_factory(transform)
        return self.array

    def _process_factory(self, name: str):
        """
        Applies any of the following methods to the input image array:
            1. apply_window_to_array
            2. apply_photometric
            3. apply_clahe
            4. apply_equalizehist
        Args:
            name (str): The name of the transformation method to be applied.
        Raises:
            ValueError: If 'name' is not recognized as a valid transformation method.
        """
        # applies any of the following methods
        transform_methods = {
            "apply_window_to_array": self._apply_window_to_array,
            "apply_photometric": self._apply_photometric,
            "apply_clahe": self._apply_clahe,
            "apply_equalizehist": self._apply_equalizehist,
            # "apply_laplace": self._subtract_minimum,
            "normalize": self._normalize,
        }

        if name in transform_methods:
            transform_methods[name]()
        else:
            raise ValueError(f"Transform '{name}' not recognized.")

    def _normalize(self):
        self.array = self.array - np.min(self.array)
        self.array = (255 / (np.max(self.array) - np.min(self.array))) * (
            self.array - np.min(self.array)
        )

    def _apply_window_to_array(self):
        """
        Applies the specified windowing to the input image array.

        Args:
            None

        Returns:
            np.ndarray: The processed image array with the applied windowing transformation.
        """
        # self._get_windowing_information()

        if self.window_center is not None and self.window_width is not None:
            lower_bound = self.window_center - (self.window_width / 2)
            upper_bound = self.window_center + (self.window_width / 2)

            self.array = np.clip(self.array, lower_bound, upper_bound)
            self.array = np.interp(self.array, [lower_bound, upper_bound], [0, 255])
        else:
            raise ValueError(
                "Window center and width are not set. Cannot apply windowing."
            )

    def _apply_photometric(self):
        """
        Applies any necessary photometric correction to the input image array and returns the modified array.
        This method should only modify the array if the photometric interpretation is "MONOCRHOME1". Otherwise, it should return the input array unmodified.

        Args:
            None

        Returns:
            np.ndarray: The processed image array with the applied windowing transformation.
        """
        # self._get_photometric_information()

        if self.photometric == "MONOCHROME1":
            try:
                inverted_image = np.invert(self.array)
            except:
                # inverted_image = np.linalg.inv(self.array)
                inverted_image = np.max(self.array) - self.array

            # Find the min and max of the original image
            original_min = np.min(self.array)
            original_max = np.max(self.array)

            # Normalize the inverted image to the original range (a, b)
            a, b = original_min, original_max
            c, d = np.min(inverted_image), np.max(inverted_image)
            normalized_image = (inverted_image - c) * ((b - a) / (d - c)) + a

            # Convert to the same dtype as the original image
            normalized_image = normalized_image.astype(self.array.dtype)

            self.array = normalized_image

    def _set_max_to_min(self):
        """
        Tries to remove the lead text from the image
        """
        self.array[self.array == self.array.max()] = self.array.min()

    def _apply_equalizehist(self):
        """
        Applies histogram equalization to the input image array and returns the modified array.

        Args:
            None

        Returns:
            np.ndarray: The processed image array with the applied windowing transformation.
        """
        # Implement this method

    def _apply_clahe(self):
        """
        Applies contrast-limited adaptive histogram equalization (CLAHE) to the input image array and returns the modified array.

        Args:
            None

        Returns:
            np.ndarray: The processed image array with the applied windowing transformation.
        """
        # Implement this method

    def _get_photometric_information(self):
        self.photometric = self.dicom.get("PhotometricInterpretation")


class CTProcessor(BaseProcessor):
    def __init__(
        self,
        array: np.ndarray,
        dicom: CTSlice,
    ) -> None:
        super().__init__(array=array, dicom=dicom)
        self.dicom: CTSlice = self.dicom

    @property
    def image_position_patient(self):
        return self.dicom.image_position_patient

    @property
    def pixel_spacing(self):
        return self.dicom.pixel_spacing

    def process(self) -> np.ndarray:
        self._set_min_to_max()
        self._raw_array_to_hu()
        self._rescale()
        self._set_max_to_min()
        self._normalize()
        #self._resize()

        return self.array

    def _process_compose(self, transforms: list[str]):
        """
        For testing different pipelines. Use for debugging only.
        Inspired by the pytorch transforms.Compose.
        Applies the transforms in any order, based on the provided list
        of 'transforms'.
        """
        # applies the processing in any order
        for transform in transforms:
            self._process_factory(transform)
        return self.array

    def _process_factory(self, name: str):
        transform_methods = {
            "set_min_to_zero": self._set_min_to_zero,
            "raw_array_to_hu": self._raw_array_to_hu,
            "rescale": self._rescale,
            "set_negative_values_to_zero": self._set_negative_values_to_zero,
            "subtract_minimum": self._subtract_minimum,
            "normalize": self._normalize,
            "set_min_to_max": self._set_min_to_max,
            "set_max_to_min": self._set_max_to_min,
        }

        if name in transform_methods:
            transform_methods[name]()
        else:
            raise ValueError(f"Transform '{name}' not recognized.")

    def _set_min_to_zero(self):
        """Fixes the round crop value so it stays within HU range."""
        self.array[self.array == self.array.min()] = 0

    def _set_min_to_max(self):
        """Fixes the round crop value so it stays within HU range.
        Sets the minimum value to the max+1
        Must also use set_max_to_min after a rescale operation.

        set_min_to_max -> rescale -> set_max_to_min
        """
        self.array[self.array == self.array.min()] = self.array.max() + 1

    def _set_max_to_min(self):
        """Fixes the round crop value so it stays within HU range.
        Sets the maximum value to the minimum
        Must also use set_max_to_min after a rescale operation.

        set_min_to_max -> rescale -> set_max_to_min
        """
        self.array[self.array == self.array.max()] = self.array.min()

    def _raw_array_to_hu(self):
        """
        Converts raw values array to HU scale
        """
        # Convert array to float32 if it's not already a float type to avoid data type issues
        if not np.issubdtype(self.array.dtype, np.floating):
            self.array = self.array.astype(np.float32)
        # same as: self.array = self.array * self.slope + self.intercept
        # Perform in-place operations to reduce memory overhead
        np.multiply(self.array, self.slope, out=self.array)  # Multiplication in-place
        np.add(self.array, self.intercept, out=self.array)  # Addition in-place

    def _set_negative_values_to_zero(self):
        """Fixes the round crop value so it stays within HU range."""
        self.array[self.array < 0] = 0

    def _rescale(self):
        """
        Shifts the HU array so the minimum HU value is 0, optimized for
        in-place modification.
        This corrects the "Shadow box" artifact from diffdrr
        see: https://github.com/eigenvivek/DiffDRR/issues/100
        """
        min_value = self.array.min()
        if min_value < -1024:
            # Perform the operation only where necessary
            np.maximum(self.array, -1024, out=self.array)
        self.array += 1024  # This operation is in-place by default

    def _subtract_minimum(self):
        """DEPRECATED
        Shifts the range so it starts at 0, no matter the minimum
        value

        """
        self.array -= np.min(self.array)

    def _normalize(self):
        # Convert the array to a float type to prevent data type overflow/underflow issues
        self.array = self.array.astype("float64")

        # Find the minimum and maximum values in the array
        min_val = self.array.min()
        max_val = self.array.max()

        # Perform min-max normalization
        self.array = (self.array - min_val) / (max_val - min_val) * 255.0

    def _apply_windowing(self):
        """Applies windowing based on the stored window center and width."""

        if self.window_center is not None and self.window_width is not None:
            lower_bound = self.window_center - (self.window_width / 2)
            upper_bound = self.window_center + (self.window_width / 2)

            self.array = np.clip(self.array, lower_bound, upper_bound)
            self.array = np.interp(self.array, [lower_bound, upper_bound], [0, 255])
        else:
            raise ValueError(
                "Window center and width are not set. Cannot apply windowing."
            )
    
    def _resize(self):
        """
        Resizes de number of slices to 100 slices. If there are more slices,
        resize using one of the axis. If there are less than 100 slices, add
        padding.
        """
        # resizes to 100 slices
        slcs = [cv2.resize(slc,(512, 512)) for slc in self.array] # was (100,512)
        self.array = np.stack(slcs)
        if self.array.shape[-1] != 512:
            raise ValueError("Shape is different than 100")
        return

        # resizes to 100 slices if > 100 slices, else add padding
        if self.array.shape[-1] > 100:
            # resizes
            slcs = [cv2.resize(slc,(100, 512)) for slc in self.array]
            self.array = np.stack(slcs)
        else:
            # adds padding
            diff = 100 - self.array.shape[-1]
            top = (diff // 2) + (diff % 2)
            bottom = diff // 2
            for i in range(top):
                self.array = np.insert(self.array,-1, np.zeros((512,512)),axis=2)
            for i in range(bottom):
                self.array = np.insert(self.array,0, np.zeros((512,512)),axis=2)
        if self.array.shape[-1] != 100:
            raise ValueError("Shape is different than 100")
