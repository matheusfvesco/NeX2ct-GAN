"""
Functions for visualizing results, plots and dicom data
"""

from concurrent.futures import ThreadPoolExecutor


import numpy as np
import torch

import imageio
import cv2

from pathlib import Path

from lib.utils.utils import Cache


class ArrayDisplayUtility:
    def __init__(self, array: np.ndarray | torch.Tensor) -> None:
        if isinstance(array, torch.Tensor):
            array = self._to_numpy(array)
        self.array = array if isinstance(array, np.ndarray) else np.array(array)
        self.shape = array.shape
        if self.requires_squeezing:
            raise ValueError(
                """Tensor is probably a batch of tensors.
                             Make sure to use a for loop to iterate over
                             each item in the batch"""
            )
        if self.is_3d:
            self.array = np.transpose(self.array, (2, 0, 1))
            self.shape = self.array.shape
        # Initialize cache for computed properties
        self.cache = Cache()

    @property
    def min(self):
        if "min" not in self.cache:
            self.cache["min"] = self.array.min()
        return self.cache["min"]

    @property
    def max(self):
        if "max" not in self.cache:
            self.cache["max"] = self.array.max()
        return self.cache["max"]

    @property
    def range(self):
        return (self.min, self.max)

    @property
    def mean(self):
        if "mean" not in self.cache:
            self.cache["mean"] = self.array.mean()
        return self.cache["mean"]

    @property
    def median(self):
        if "median" not in self.cache:
            self.cache["median"] = np.median(self.array)
        return self.cache["median"]

    @property
    def variance(self):
        if "variance" not in self.cache:
            self.cache["variance"] = self.array.var()
        return self.cache["variance"]

    @property
    def std_dev(self):
        if "std_dev" not in self.cache:
            self.cache["std_dev"] = self.array.std()
        return self.cache["std_dev"]

    @property
    def precision(self):
        # Since the dtype is unlikely to change, it might not need caching
        return self.array.dtype

    def display(self, iteractive=True):
        if self.is_3d:
            self._display_3d(iteractive=iteractive)
        else:
            self._display_2d()

    def write_as_image(
        self,
        path: str | Path,
        complete_array: bool = True,
        extension: str = ".jpeg",
        thumbnail: int | tuple[int, int] | None = None,
    ):
        """
        Writes both 2D Arrays and 3D Arrays to disk.
        path: The path to the output file. If the it contains a file
            extension and it is .gif, each slice will be writen as a
            gif in the specified path. If path has a file extension,
            only the middle slice will be written on the specified path.
            If no file extension is provided, all slices will be writen
            inside the provided directory, using the file extension
            defined by the "extension" parameter.
        complete_array
        """
        path = self._validate_path(path=path)
        if thumbnail is not None:
            thumbnail = self._process_thumbnail_size(size=thumbnail)
        if self.is_3d:
            self._write_3d_to_disk(
                path=path,
                complete_array=complete_array,
                extension=extension,
                thumbnail=thumbnail,
            )
        else:
            self._write_2d_to_disk(path=path, thumbnail=thumbnail)

    def display_histogram(self):
        pass

    def write_histogram_to_disk(self, path: str | Path):
        path = self._validate_path(path=path)
        pass

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array)

    @property
    def normalized_array(self) -> np.ndarray:
        """
        Returns a normalized copy of self.array in the 0-255 range, computing it only once.
        """
        if "normalized_array" not in self.cache:
            min_val = self.array.min()
            max_val = self.array.max()
            # Avoid division by zero in case max_val == min_val
            range_val = max_val - min_val if max_val != min_val else 1
            # Normalize array to 0-255 range
            normalized = ((self.array - min_val) / range_val) * 255
            self.cache["normalized_array"] = normalized.astype(np.uint8)
        return self.cache["normalized_array"]

    @property
    def is_3d(self) -> bool:
        return len(self.shape) == 3

    @property
    def requires_squeezing(self) -> bool:
        """
        Checks if provided array was a 4D tensor or tensor with higher
        dimentionality, to ensure it wasn't a batch of 3D tensors
        """
        return len(self.shape) > 3

    def _display_2d(self):
        array = self.normalized_array  # 2D array (black and white)
        cv2.imshow("Image", array)  # Display the image
        cv2.waitKey(0)  # Wait for a key press to close the image window
        cv2.destroyAllWindows()  # Close all OpenCV windows

    def _display_3d(self, iteractive: bool = True):
        pass

    def _is_file(self, path: Path) -> bool:
        return path.suffix != ""

    def _validate_path(self, path: str | Path) -> Path:
        """
        Ensures path is a Path object and creates parent directories.
        """
        path = Path(path)
        if self._is_file(path):  # checks if it has file extension
            parent = path.parent  # get parent folder
        else:
            parent = path

        parent.mkdir(parents=True, exist_ok=True)  # create parent folders

        return path

    def _path_is_gif(self, path: Path) -> bool:
        return path.suffix == ".gif"

    def _write_2d_to_disk(self, path: Path, thumbnail: tuple[int, int] | None):
        """
        Write a 2D array to disk at the specified path
        """
        array = self.normalized_array

        # Convert the 2D array to an 8-bit grayscale image
        img_2d = array.astype(np.uint8)

        if thumbnail is not None:
            img_2d = self._resize_2d_array(array=img_2d, thumbnail=thumbnail)

        # Write the image to a file
        cv2.imwrite(str(path), img_2d)

    def _write_3d_to_disk(
        self,
        path: Path,
        complete_array: bool,
        extension: str,
        thumbnail: tuple[int, int],
    ):
        write_gif = self._path_is_gif(path=path)
        is_file = self._is_file(path=path)
        if complete_array and write_gif and is_file:
            self._write_images_as_gif(path=path, thumbnail=thumbnail)
        elif complete_array and not write_gif and not is_file:
            self._write_images_in_folder(
                path=path, extension=extension, thumbnail=thumbnail
            )
        elif not complete_array and not write_gif and is_file:
            self._write_single_slice_to_disk(path=path, thumbnail=thumbnail)
        else:
            raise ValueError(
                f"""complete_array must be a bool, and path
                             must be a directory or an image file. If
                             path is a gif file, complete_array must be
                             true.
                             complete_array={complete_array}
                             path={path}"""
            )

    def _write_images_as_gif(self, path: Path, thumbnail: tuple[int, int] | None):
        """
        Writes 3D array as a gif to disk
        """
        array = self.normalized_array
        skip_slices = 5  # value to skip every 2 slices

        # Convert the 3D array to a list of 2D images, taking every 2nd slice
        images = [
            ct_slice.astype(np.uint8)
            for i, ct_slice in enumerate(array)
            if i % skip_slices == 0
        ]

        if thumbnail:
            images = [
                self._resize_2d_array(array=ct_slice, thumbnail=thumbnail)
                for ct_slice in images
            ]

        # Write the images to a GIF file, adjusting the duration based on the number of images
        imageio.mimsave(path, images, "GIF", duration=2.5 / len(images), loop=0)

    def _write_images_in_folder(
        self, path: Path, extension: str, thumbnail: tuple[int, int] | None
    ):
        """
        Writes 3D array as a series of images inside the specified directory using multiple threads.
        If a thumbnail size is provided, each image is resized to this size before being saved.
        """
        array = self.normalized_array
        # Assign the instance method to a local variable
        resize_2d_array = self._resize_2d_array

        def write_image(idx, ct_slice: np.ndarray):
            # Convert the 2D image to an 8-bit grayscale image
            ct_slice = ct_slice.astype(np.uint8)

            # Resize the image if a thumbnail size is provided
            if thumbnail is not None:
                ct_slice = resize_2d_array(ct_slice, thumbnail)

            # Write the image to a file
            img_path = path / f"slice_{idx}{extension}"
            cv2.imwrite(str(img_path), ct_slice)

        # Use ThreadPoolExecutor to write images in parallel
        with ThreadPoolExecutor() as executor:
            # Submit all image writing tasks to the executor
            futures = [
                executor.submit(write_image, idx, ct_slice)
                for idx, ct_slice in enumerate(array)
            ]

            # Optionally, wait for all tasks to complete and handle exceptions
            for future in futures:
                try:
                    future.result()  # This will raise exceptions from the worker threads if any
                except Exception as e:
                    print(f"Exception occurred during image writing: {e}")

    def _write_single_slice_to_disk(
        self, path: Path, thumbnail: tuple[int, int] | None, index: int | str = "middle"
    ):
        """
        Write a single slice of the 3D array to disk on the specified
        """
        array = self.normalized_array

        # Determine which slice to write
        if index == "middle":
            index = len(array) // 2
        elif isinstance(index, int):
            if index < 0 or index >= len(array):
                raise ValueError("Index out of range")

        # Convert the 2D image to an 8-bit grayscale image
        img_2d = array[index].astype(np.uint8)

        if thumbnail is not None:
            img_2d = self._resize_2d_array(array=img_2d, thumbnail=thumbnail)

        # Write the image to a file
        cv2.imwrite(str(path), img_2d)

    def _process_thumbnail_size(self, size: int | tuple[int, int]) -> tuple[int, int]:
        if not isinstance(size, tuple) and not isinstance(size, int):
            raise ValueError("Thumbnail size must be int or tuple[int, int]")

        if isinstance(size, tuple) and len(size) != 2:
            raise ValueError(
                f"Thumbnail size must be int or tuple[int, int], but len(size) == {len(size)}"
            )

        if isinstance(size, int):
            size = (size, size)

        if not isinstance(size, tuple):
            raise ValueError("Thumbnail size must be int or tuple[int, int]")
        return size

    def _resize_2d_array(
        self, array: np.ndarray, thumbnail: tuple[int, int]
    ) -> np.ndarray:
        resized_array = cv2.resize(array, thumbnail)
        return resized_array
