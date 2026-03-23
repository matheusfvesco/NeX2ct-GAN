from __future__ import annotations

import numpy as np
from abc import ABC
import cv2
import torch
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

RESIZE_SIZE = 256

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # avoid import-time circular dependency; only imported for type checking
    from lib.dicom import BaseDicomFile, CTSlice, XRayView, CTScanDirectory


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

        self._resize()

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
        self.array = self.array.astype(np.float64)
        self.array = self.array - np.min(self.array)
        self.array = (1 / (np.max(self.array) - np.min(self.array))) * (
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

    def _resize(self):
        self.array = cv2.resize(self.array, (RESIZE_SIZE, RESIZE_SIZE))


class CTProcessor(BaseProcessor):
    def __init__(
        self,
        array: np.ndarray,
        scan: "CTScanDirectory",
    ) -> None:
        """Initialize a CT processor from a full scan directory.

        The previous API accepted a single ``CTSlice`` instance;
        the new contract expects the caller to provide the entire
        :class:`~lib.dicom.CTScanDirectory`.  For backwards
        compatibility we transparently keep ``self.dicom`` pointing to
        the middle slice of the scan (``scan.dicom``), but the full
        directory is stored in ``self.scan`` so that higher‑level code
        can access spacing, orientation, etc. across the volume.
        """
        # ``BaseProcessor`` still wants a ``dicom`` argument, so pass
        # the representative slice from the scan.
        super().__init__(array=array, dicom=scan.dicom)

        # store both objects as requested by the new design
        self.scan: "CTScanDirectory" = scan
        # keep the familiar attribute for existing methods
        self.dicom: CTSlice = scan.dicom

        self.pixel_spacing = self.dicom.pixel_spacing
        self.spacing = self.scan.spacing

    @property
    def image_position_patient(self):
        # image_position_patient is a property of the slice; we still
        # forward to ``self.dicom`` for convenience
        return self.dicom.image_position_patient

    def process(self) -> np.ndarray:
        # self._set_min_to_max()
        self._raw_array_to_hu()
        self._segment_body()

        # 2. geographic standardization: make spacing isotropic
        self._resample()

        # 3. dimensionality -- resize/crop to fixed input size
        self._crop()

        # shifts everything by +1024
        #self._rescale()
        self._transform_to_lidc_format()
        # self._set_max_to_min()
        #self._normalize()
        # self._resize()

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

    def _segment_body(self):
        """
        Run TotalSegmentator to generate a body mask and apply it to
        ``self.array``.

        The segmentation is executed on the current volume (which should be
        in HU units after ``_raw_array_to_hu``).  Voxels outside of the body
        are set to a constant minimum HU value (typically -1024) so that
        subsequent rescaling/normalization treats them as air/background.

        This helper mirrors the steps demonstrated in the notebooks
        (see ``total_seg.ipynb``) and is invoked from :meth:`process`.
        """
        # import lazily to avoid pulling heavy dependencies unless used



        # create a temporary NIfTI image with identity affine; spacing
        # information is not required for the body task.
        # affine = np.eye(4)
        affine = self.scan.affine
        nifti_img = nib.Nifti1Image(self.array, affine)

        # perform segmentation
        mask_img = totalsegmentator(nifti_img, task="body", ml=True, quiet=True, device="gpu")
        mask = mask_img.get_fdata().astype(bool)

        # apply mask: outside body -> minimum HU (air)
        min_hu = -1024
        self.array = np.where(mask, self.array, min_hu)

    def _transform_to_lidc_format(self):
        """
        Transforms the CT volume to the LIDC format:
        1. Clips lower bound at -1024 (HU) to handle air-background variance.
        2. Applies an offset of +1024 to shift Air to ~0.
        3. Clips the final range to [0, 2500].
        """
        # Ensure we are in float32 for calculations if not already
        #if not np.issubdtype(self.array.dtype, np.floating):
        #    self.array = self.array.astype(np.float32)

        # 1. Start from HU units (assumed already converted)
        # 2. Clip lower bound at -1024 to handle scanners with -2048 background
        # and upper bound to handle extreme metal artifacts before offset.
        # We use 20000 as a safe upper bound for initial clipping.
        # Perform in-place operations to reduce memory overhead
        # np.clip(self.array, -1024, 20000, out=self.array)

        # 3. Apply Offset
        # self.array += 1024

        # 4. Final Clip for [0, 2500] range
        np.clip(self.array, -1024, 5500, out=self.array)
        self.array += 1024

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

        # By default, apply windowing as a clip in HU space and do NOT
        # rescale to 0-255. Use `rescale=True` if you need the output
        # to be mapped to the [0,255] range (keeps backward-compat).
        def _apply(rescale: bool = False):
            if self.window_center is not None and self.window_width is not None:
                lower_bound = self.window_center - (self.window_width / 2)
                upper_bound = self.window_center + (self.window_width / 2)

                # Clip values to the requested HU window
                self.array = np.clip(self.array, lower_bound, upper_bound)

                # Optionally rescale clipped values to 0-255
                if rescale:
                    # Use interpolation to map [lower_bound, upper_bound] -> [0,255]
                    self.array = np.interp(
                        self.array, [lower_bound, upper_bound], [0, 255]
                    )
            else:
                # If no windowing information is present, do nothing. This
                # keeps callers simpler (no exceptions) and preserves HU.
                return

        # expose a simple API: call _apply() without args to clip in HU space
        _apply(rescale=False)

    def _resize(self):
        """
        Resizes de number of slices to 100 slices. If there are more slices,
        resize using one of the axis. If there are less than 100 slices, add
        padding.
        """
        # legacy helper left in place for backwards compatibility; new
        # pipelines should use `_crop` which internally handles both
        # upsampling and center‑cropping.
        arr = torch.Tensor(self.array)
        arr = arr.unsqueeze(0).unsqueeze(0)
        arr = torch.nn.functional.interpolate(
            arr, size=(RESIZE_SIZE, RESIZE_SIZE, RESIZE_SIZE), mode="trilinear"
        )
        arr = arr.squeeze()
        self.array = arr.numpy()
        for shape in self.array.shape:
            if shape == RESIZE_SIZE:
                continue
            ValueError(
                f"Shape ({self.array.shape}) is different than ({RESIZE_SIZE},{RESIZE_SIZE},{RESIZE_SIZE})"
            )
        return

        # resizes to 100 slices
        slcs = [cv2.resize(slc, (512, 512)) for slc in self.array]  # was (100,512)
        self.array = np.stack(slcs)
        if self.array.shape[-1] != 512:
            raise ValueError("Shape is different than 100")
        return

        # resizes to 100 slices if > 100 slices, else add padding
        if self.array.shape[-1] > 100:
            # resizes
            slcs = [cv2.resize(slc, (100, 512)) for slc in self.array]
            self.array = np.stack(slcs)
        else:
            # adds padding
            diff = 100 - self.array.shape[-1]
            top = (diff // 2) + (diff % 2)
            bottom = diff // 2
            for i in range(top):
                self.array = np.insert(self.array, -1, np.zeros((512, 512)), axis=2)
            for i in range(bottom):
                self.array = np.insert(self.array, 0, np.zeros((512, 512)), axis=2)
        if self.array.shape[-1] != 100:
            raise ValueError("Shape is different than 100")

    def _transpose(self):
        arr = self.array
        self.array = np.transpose(arr, axes=(2, 0, 1))
        return

    def _resample(self, target_spacing: float = 1.0) -> None:
        """
        Resample the current volume so that voxel spacing becomes isotropic
        (default 1 mm).

        The method computes zoom factors using the original spacing stored
        in ``self.scan`` and applies a trilinear interpolation via
        ``torch.nn.functional.interpolate`` (which is significantly faster
        than ``scipy.ndimage.zoom`` for 3D volumes). After resampling, the
        volume is written back to ``self.array`` and the ``spacing``
        attribute is updated so that later code is aware of the new spacing.
        """
        original_spacing = np.array(list(self.scan.spacing), dtype=float)
        if original_spacing.size != 3:
            raise ValueError("expected 3 spacing values for CT volume")

        # compute zoom that converts spacing -> target_spacing.
        zoom_factors = original_spacing / float(target_spacing)
        if not np.all(zoom_factors > 0):
            raise ValueError("computed non-positive zoom factor")

        # Compute target size exactly like ndimage.zoom does
        target_size = [
            int(np.round(s * z)) for s, z in zip(self.array.shape, zoom_factors)
        ]

        # perform the interpolation
        # torch expects (N, C, D, H, W) for 3D interpolation
        tensor = (
            torch.from_numpy(np.ascontiguousarray(self.array)).unsqueeze(0).unsqueeze(0)
        )

        # align_corners=True matches the behavior of scipy.ndimage.zoom
        resampled_tensor = torch.nn.functional.interpolate(
            tensor, size=target_size, mode="trilinear", align_corners=True
        )

        self.array = resampled_tensor.squeeze().numpy()

        # update stored spacing information for downstream consumers
        self.spacing = np.array([target_spacing, target_spacing, target_spacing])

    def _crop(self, target_size: int = 320) -> None:
        """
        Adjust the volume to satisfy dimensionality requirements:

        * Case A (all dimensions \>= ``target_size``): centre‑crop to
          ``target_size`` in every axis.
        * Case B (any dimension < ``target_size``): first upsample the
          volume until its smallest axis reaches **256** voxels, then
          centre‑crop the remaining dimensions that exceed
          ``target_size``.  The intent is to avoid zero‑padding while
          limiting unnatural up‑sampling.

        The final shape will therefore either be ``target_size`` in every
        axis or, in the second case, contain one axis between 256 and
        ``target_size`` with the others clipped to ``target_size``.
        """
        # lazy import because SciPy may not be present in minimal installs
        from scipy import ndimage

        arr = self.array
        shape = np.array(arr.shape, dtype=int)

        if np.all(shape >= target_size):
            # simple centre crop
            start = (shape - target_size) // 2
            slices = tuple(slice(start[i], start[i] + target_size) for i in range(3))
            self.array = arr[slices]
            return

        # Case B: upsample so that the smallest axis is at least 256
        min_dim = shape.min()
        if min_dim < 256:
            scale = 256.0 / float(min_dim)
            arr = ndimage.zoom(arr, zoom=scale, order=3)
            shape = np.array(arr.shape, dtype=int)

        # centre crop any dimension that exceeds the target size
        slices = []
        for size in shape:
            if size > target_size:
                start = (size - target_size) // 2
                slices.append(slice(start, start + target_size))
            else:
                slices.append(slice(0, size))
        self.array = arr[tuple(slices)]
