from typing import List
from typing import Sized
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import cv2
from numpy import ndarray
from torch import Tensor

from .array.normalize import normalize_range
from .array.normalize import normalize_sum
from .array.numpy import to_numpy

__all__ = ['to_heatmap', 'overlay_images', 'plt_remove_margin']


def to_heatmap(
    arr: Union[ndarray, Tensor],
    normalized=False,
    cmap=cv2.COLORMAP_JET
) -> Union[ndarray, Tensor]:
    """ convert arr to a heatmap with specified cmap.

    :param arr: a 2D matrix
    :param normalized: if False, arr will be normalized to [0, 1] at first
    :param cmap: the color map, see cv2.COLORMAP
    :return: if arr is a torch tensor, a C x H x W tensor will be returned.
        Otherwise, an H x W x C numpy array is returned
    """
    tmp = to_numpy(arr)
    if normalized:
        tmp = (np.clip(tmp, 0, 1) * 255).astype(np.uint8)
    else:
        tmp = (normalize_range(tmp) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(tmp, cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32)
    if isinstance(arr, Tensor):
        heatmap = torch.tensor(np.transpose(heatmap, [2, 0, 1]))
    return heatmap / 255.


def overlay_images(
    images: List[Union[ndarray, Tensor]],
    weights: Union[float, List[float], Tuple[float]]
) -> Union[ndarray, Tensor]:
    """ Overlay images together with the specified weights.

    :param images: the images to overlay together
    :param weights: the weight of each image, if weights is a float number, each
        image will have the same weight. However, the total weight is always 1.
    :return: numpy array or pytorch tensor, the same as the first image
    """
    if isinstance(weights, float):
        weights = [float] * len(images)
    elif isinstance(weights, Sized):
        assert len(weights) == len(images), \
            "images and weights should have the same number of elements"
    else:
        raise ValueError("weights should be float, List or Tuple")

    out = to_numpy(images[0])
    weights = normalize_sum(np.array(weights), axis=0)
    out *= weights[0]

    for i in range(1, len(images)):
        out += to_numpy(images[i]) * weights[i]
    out = np.clip(out, a_min=0, a_max=1)

    if isinstance(images[0], Tensor):
        out = torch.tensor(out)

    return out


def plt_remove_margin():
    """ remove the margin of the current matplotlib figure axis
    """
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
