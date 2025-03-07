"""
More general utilities, like string formatting
"""

import math
import torch
from pathlib import Path

from pydicom.multival import MultiValue
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import polars as pl


def get_multival_by_index(value: MultiValue, index: int = 0) -> float:
    if not value:  #
        return value
    if isinstance(value, MultiValue):
        return float(value[index])
    else:
        return float(value)


def degrees_to_radians(angle_degrees: int | float) -> float:
    """
    Converts an angle from degrees to radians.

    Args:
        angle_degrees (int | float): The angle in degrees.

    Returns:
        float: The equivalent angle in radians.
    """
    return math.radians(angle_degrees)


class Cache:
    """
    A simple in-memory caching system. It provides methods to add, get, remove and check for
    existence of items in the cache.
    """

    def __init__(self):
        self._cache = {}

    def __str__(self):
        """
        Returns a human-readable representation of the cache.
        """
        return str(self._cache)

    def __repr__(self):
        """
        Returns an official string representation of the cache for debugging.
        """
        return f"Cache({str(self._cache)})"

    def __len__(self):
        """
        Returns the number of items in the cache.
        """
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """
        Checks if a key exists in the cache.

        Args:
            key (str): The key to check in the cache.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return self.contains(key=key)

    def __getitem__(self, key: str) -> object:
        return self.get(key)

    def __setitem__(self, key, value):
        self.add(key=key, value=value)

    def add(self, key: str, value: object) -> None:
        """
        Adds an item to the cache with a specific key.

        Args:
            key (str): The key to associate with the item in the cache.
            value (object): The item to add to the cache.

        Raises:
            ValueError: If either the key or value is None.
        """
        if key is None or value is None:
            raise ValueError("Both key and value must be provided.")
        self._cache[key] = value

    def get(self, key: str, default: object | None = None) -> object:
        """
        Retrieves an item from the cache by key.

        Args:
            key (str): The key of the item to retrieve.
            default (object | None): The default to be returned if key is not stored

        Returns:
            object: The cached item, or None if not found.

        Raises:
            KeyError: If the key is not found in the cache.
        """
        try:
            return self._cache[key]
        except KeyError:
            return default

    def remove(self, key: str) -> None:
        """
        Removes an item from the cache by key.

        Args:
            key (str): The key of the item to remove.

        Raises:
            KeyError: If the key is not found in the cache.
        """
        try:
            del self._cache[key]
        except KeyError:
            raise KeyError("Key not found in cache.")

    def contains(self, key: str) -> bool:
        """
        Checks if a key exists in the cache.

        Args:
            key (str): The key to check in the cache.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self._cache

    def get_key(self, obj: object) -> tuple[bool, str]:
        """
        Checks if an object exists in the cache and returns its key.

        Args:
            obj (object): The object to check in the cache.

        Returns:
            str or None: If the object is found in the cache, return its unique
                identifier, else `None` indicating that the object does not exist in
                the cache.
        """
        for key in self._cache:
            if self._cache[key] == obj:
                return key
        return None

    def clear(self) -> None:
        """
        Clears all items from the cache.
        """
        self._cache = {}


class UniquePathGenerator:
    """
    Class that prevents the overwrite of existing subfiles and subdirectories
    in a directory. Provides an easy to use method (.generate()) that creates
    the unique path based on the specified method.
    """

    ACCEPTED_METHODS = ["naive-increment", "timestamp", "continue-increment"]

    def __init__(
        self,
        rootdir: Path | str = None,
        method: str = "increment",
        datetime_format: str = "%Y-%m-%d_%H:%M:%S:%f",
        in_suffix: bool = True,
        sep: str = "-",
        create_root_dir: bool = False,
        rootdir_do_not_exists_ok: bool = False,
    ) -> None:
        """
        Args:
            rootdir (Path | str): The directory to store subdirectories or subfiles.
            method (str): The method to generate unique names, must be one of "increment" or "timestamp".
            datetime_format (str): Format string for `datetime.now()`, default is "%Y%m%dT%H%M%S".
            in_suffix (bool): If True, the unique identifier will be placed after the stem and before the suffix.
                            If False, it will be placed before the stem and after the separator.
            sep (str): The string used as a separator between parts of the filename.
            ensure_rootdir_exists (bool): Whether to create the root directory if it does not exist.

        Raises:
            ValueError: If `method` is not one of the accepted methods, or if `rootdir` does not exist and
                        `ensure_rootdir_exists` is False.
        """
        # ensures it is a Path object
        self.rootdir = Path(rootdir)
        # creates or raises error depending on desired behavior.
        if create_root_dir:
            self.rootdir.mkdir(parents=True, exist_ok=True)
        elif not self.rootdir.exists() and not rootdir_do_not_exists_ok:
            raise ValueError("rootdir does not exist")

        # checks if method is within the accepted methods
        if method not in self.ACCEPTED_METHODS:
            raise ValueError(f"Method must be one of {self.ACCEPTED_METHODS}")
        self.method = method

        # stores the strftime string for formatting time.now()
        self.datetime_format = datetime_format

        # defines the string template
        if in_suffix:
            self.name_template = "{stem}{sep}{unique}{suffix}"
        else:
            self.name_template = "{unique}{sep}{stem}{suffix}"

        # defines the string separator
        self.sep = sep

    def _path_exists(self, subpath: str) -> bool:
        return (self.rootdir / subpath).exists()

    @property
    def now(self) -> str:
        """
        Returns the current time in the strftime format specified on
        class creation.
        """
        return datetime.now().strftime(self.datetime_format)

    def generate(self, original_name: str) -> Path:
        """
        Generates a unique name (directory or filename) inside the rootdir,
        following the specified format. Returns a tuple

        Args:
            self: The instance of the class (usually included automatically by Python).
            parameter1 (type): Description of parameter1. Must be such and such.

        Returns:
            type: Description of what the method returns. If it doesn't return anything, omit this section.
        """
        original_name: Path = Path(original_name)

        template_mapping = {
            "stem": original_name.stem,
            "suffix": original_name.suffix,
            "sep": self.sep,
        }

        if self.method == "naive-increment":
            unique_name = self._build_naive_increment(template_mapping=template_mapping)
        elif self.method == "timestamp":
            unique_name = self._build_time_name(template_mapping=template_mapping)

        return self.rootdir, unique_name, self.rootdir / unique_name

    def _build_naive_increment(self, template_mapping: str) -> str:
        counter = 0
        while True:
            new_name: str = self.name_template.format(
                unique=str(counter), **template_mapping
            )
            full_path = self.rootdir / new_name
            if not full_path.exists():
                return new_name
            counter += 1

    def _build_time_name(self, template_mapping: str) -> str:
        timestamp = self.now
        new_name: str = self.name_template.format(
            unique=str(timestamp), **template_mapping
        )
        full_path = self.rootdir / new_name
        if not full_path.exists():
            return new_name
        raise ValueError("Name already exists")

    # TODO: Check this
    def _get_latest_increment(self):
        files = self.rootdir.glob("*")
        if not files:
            return None, None  # Return None values if the list is empty
        # Use a regular expression to find all numbers in each filename
        numbers = []
        for file in files:
            match = re.search(r"\d+", file)
            if match:
                numbers.append(int(match.group()))
            # Sort the numbers and get the maximum (latest) one
            if numbers:
                latest_number = max(numbers)
        if not numbers:
            return None, None
        # Find the corresponding filename with the highest numerical value
        for file in files:
            match = re.search(r"\d+", file)
        if match and int(match.group()) == latest_number:
            return file, latest_number
        return None, None  # Return None values if no valid number is found

    def _build_continue_increment(self):
        # Get the latest incremental filename and its corresponding number
        file, number = self._get_latest_increment()
        if not file:
            return None  # Return None values if no valid number is found
        new_number: int = number + 1
        new_name: str = self._build_name_template(file=file, number=number).format(
            unique=str(new_number)
        )
        full_path = self.rootdir / new_name
        if not full_path.exists():
            return new_name  # Return the new name and number if it exists
        else:
            # Raise an exception if the file already exists
            raise ValueError("Name already exists")

    def _build_name_template(self, file: str, number: int):
        name_template = file.replace(str(number), "{unique}")
        return name_template

class SingleDataStorer:
    def __init__(self, max_points: int = 0):
        self.data: list = []

    def store(self, item):
        if isinstance(item, torch.Tensor):
            item = item.clone().to("cpu").detach().numpy()
            item = float(item)
        self.data.append(item)

    def plot(self, xlabel, ylabel, title, style = "darkgrid", figsize = (10,10)):
        """
        Uses seaborn styles to create a plot of the stored data.
        
        Parameters:
        style (str): The seaborn style to use for plotting. Can bo one of: 'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'

        Returns:
        plt (matplotlib.pyplot): The plot object.
        """
        sns.set_style(style)  # Set seaborn style
        plt.figure(figsize=figsize)
        
        sns.lineplot(x=range(len(self.data)), y=self.data, marker='o')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        return plt

    def pop(self, index: int =-1):
        return self.data.pop(index)
    
    def mean(self):
        return torch.tensor(self.data).mean()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

class DataStorer:
    def __init__(self):
        self.data: dict[str, SingleDataStorer] = {}

    def store(self, tag, item):
        if tag not in self.data:
            self.data[tag] = SingleDataStorer()
        self.data[tag].store(item)
    
    def plot(self, xlabel, ylabel, title, tags=None, style="darkgrid", figsize=(10, 10)):
        """
        Plots the data stored in specified columns (tags).
        
        Parameters:
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the plot.
        tags (list): List of tags to plot. If None, plots all available tags.
        style (str): Seaborn style to use for plotting.
        figsize (tuple): Size of the plot figure.
        
        Returns:
        plt (matplotlib.pyplot): The plot object.
        """
        sns.set_style(style)
        plt.figure(figsize=figsize)

        if tags is None:
            tags = list(self.data.keys())
        
        # Plot each selected tag's data
        for tag in tags:
            sns.lineplot(x=range(len(self.data[tag].data)), y=self.data[tag].data, marker='o', label=tag)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        
        return plt

    def write_csv(self, path: str):
        """
        Writes the stored data to a CSV file.
        
        Parameters:
        path (str): Path to save the CSV file.
        """
        # converts data from dict of SingleDataStorer to dict of lists
        data = {tag: self.data[tag].data for tag in self.data}

        # Create a DataFrame from the stored data
        df = pl.DataFrame(data)
        # Write the DataFrame to a CSV file
        df.write_csv(path, separator="\t")

def collate_fn(batch):
    cts = torch.stack([b[0] for b in batch])
    xr1 = torch.stack([b[1] for b in batch])
    xr2 = torch.stack([b[2] for b in batch])
    paths = [b[3] for b in batch]

    return cts, xr1, xr2, paths