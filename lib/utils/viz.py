"""
Functions for visualizing results, plots and dicom data
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider


def show_ct(volume: Union[np.ndarray, torch.Tensor], interactive: bool = True) -> None:
    """
    Plots a numpy array or torch tensor representing a CT volume. If interactive is True (default), enables scrolling through the slices and starts on the first slice. Otherwise, simply shows the middle slice.

    Args:
        volume (Union[np.ndarray, torch.Tensor]): The CT volume to be plotted, as a numpy array or torch tensor.
        interactive (bool): If True, enables scrolling through the slices and starts on the first slice. Defaults to True.

    Returns:
        None
    """
    # Convert the volume to a numpy array if it is a torch tensor
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()

    # Get the number of slices in the volume
    num_slices = volume.shape[2]

    # Create a function to update the displayed slice
    def update_slice(val: int) -> None:
        ax.imshow(volume[:, :, int(val)], cmap="gray")

    # Create a new figure and axis for plotting
    fig, ax = plt.subplots()

    if interactive:
        # Plot the first slice of the volume
        ax.imshow(volume[:, :, 0], cmap="gray")

        # Create a slider to control the displayed slice
        ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
        slider = Slider(ax_slider, "Slice", 0, num_slices - 1, valinit=0, valstep=1)

        # Update the displayed slice when the slider is changed
        slider.on_changed(update_slice)
    else:
        # Plot the middle slice of the volume
        ax.imshow(volume[:, :, num_slices // 2], cmap="gray")

    plt.show()


def show_xr(image: Union[np.ndarray, torch.Tensor]) -> None:
    """
    Plots a single numpy array or torch tensor using matplotlib.

    Args:
        image (Union[np.ndarray, torch.Tensor]): The image to be plotted, as a numpy array or torch tensor.

    Returns:
        None
    """
    # Convert the image to a numpy array if it is a torch tensor
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    # Plot the image using matplotlib
    plt.imshow(image, cmap="gray")
    plt.show()


def plot_xrs(
    arrays: List[Tuple[Union[np.ndarray, torch.Tensor], str]],
    title: Optional[str] = None,
) -> None:
    """
    Plots a list of numpy.ndarrays or PyTorch tensors as subplots in a single figure.

    Args:
        arrays (List[Tuple[Union[np.ndarray, torch.Tensor], str]]): A list of tuples containing an array or tensor to plot and a string to use as the x label for the corresponding subplot.
        title (Optional[str], optional): The title for the figure. Defaults to None.

    Returns:
        None
    """
    # Determine the number of subplots
    n = len(arrays)

    # Create a figure with n subplots
    fig, axs = plt.subplots(n)

    # Plot each array/tensor in a separate subplot
    for idx, (array, xlabel) in enumerate(arrays):
        axs[idx].plot(array if isinstance(array, np.ndarray) else array.numpy())

        # Set the x label for the subplot if provided
        if xlabel:
            axs[idx].set_xlabel(xlabel)

    # Set the title for the figure if provided
    if title:
        fig.suptitle(title)

    # Show the figure
    plt.show()
