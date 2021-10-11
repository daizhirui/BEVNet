from typing import Union

import numpy as np
from numpy import ndarray
from torch import Tensor

__all__ = [
    'to_numpy',
    'unfold',
    'cdf'
]


def to_numpy(arr: Union[ndarray, Tensor], np_type=np.float32) -> ndarray:
    """ convert `arr` to numpy array

    :param arr: `numpy.ndarray` or `torch.Tensor`
    :param np_type: data type
    :return: numpy.ndarray
    """
    if isinstance(arr, ndarray):
        return arr.astype(np_type)
    elif isinstance(arr, Tensor):
        return arr.data.cpu().numpy().astype(np_type)
    else:
        return np.array(arr, dtype=np_type)


def unfold(a, axis, size, step):
    """Numpy version of ``torch.Tensor.unfold``.

    Parameters
    ----------
    a: numpy.ndarray
        Numpy array of shape (N1, .., Ni, .., Nk)
    axis: int
        The axis in which unfolding happens.
    size: int
        The size of each slice that is unfolded.
    step: int
        The step between each slice.

    Returns
    -------
    unfolded_a: numpy.array
        Unfolded version of ``a``, whose shape is (N1, .., M, .., Nk, size),
        where ``M = int((axis_size - size) / step + 1)``.
    """
    idx = np.arange(0, a.shape[axis] - size + 1, step)
    shape = np.ones(a.ndim, dtype=int)
    shape[axis] = idx.size
    idx = idx.reshape(shape)

    out = [np.take_along_axis(a, idx, axis)]
    for i in range(1, size):
        out.append(np.take_along_axis(a, idx + i, axis))

    out = np.stack(out, -1)
    return out


def cdf(x, bins=50):
    hist, bin_edges = np.histogram(x, bins=bins, density=True)
    y = np.empty_like(bin_edges)
    y[0] = 0
    y[1:] = np.cumsum(np.diff(bin_edges) * hist)
    return bin_edges, y
