#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torchvision as tv

from plenoptic.tools.signal import raised_cosine

__all__ = [
    'GaussianPyramid',
    'compute_window',
    'compute_log_window'
]


class GaussianPyramid(nn.Module):
    def __init__(self,
                 kernel_size: list[int] = [5, 5],
                 sigma: list[float] = [0.1, 2.0],
                 n_levels: int = 3):
        """
        A PyTorch implementation of the Gaussian Pyramid.

        :param kernel_size: The kernel size of the Gaussian blur.
        :param sigma: Standard deviation of the Gaussian blur.
        :param n_levels: How many levels/layers to compute of the image.
        """
        super(GaussianPyramid, self).__init__()
        self.n_levels = n_levels
        self.sigma = sigma
        self.gauss = tv.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        # nn.Upsample is deprecated, so we use interpolate
        self.upsample = nn.functional.interpolate

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsamples the input tensor.
        :param x: Input tensor, should be in shape batch x channels x height x width.
        :return: The downsampled tensor
        """
        assert x.ndim == 4, f'Tensor should have 4 dimensions but has {x.ndim}.'
        return x[:, :, ::2, ::2]

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shifts the values by 128, to ensure that we don't have any negative values,
        then clamps the values into RGB range of [0, 255].
        :param x: Input tensor, should be in shape batch x channels x height x width.
        :return: The clamped tensor.
        """
        x += 128
        return torch.clamp(x, 0, 255)

    def forward(self,
                img: torch.Tensor,
                max_levels: int | None = None,
                order_coarse_to_fine: bool = False) -> list[torch.Tensor]:
        """
        Forward pass of the Gaussian Pyramid.
        :param img: input images, should be in shape batch x channels x height x width.
        :param max_levels: if specified, number of layers to compute of the image, else falls back to class attribute.
        :param order_coarse_to_fine: if true, layer order in list will be coarse to fine.
        :return: a list of tensors with their respective scale
        """
        max_levels = max_levels or self.n_levels
        current = img
        pyr = []
        for level in range(max_levels):
            filtered = self.gauss(current)
            down = self.downsample(filtered)
            up = self.upsample(down, size=current.size()[2:])
            diff = current - up
            pyr.append(self.clamp(diff))
            current = down

        return pyr[::-1] if order_coarse_to_fine else pyr


def compute_r():
    ...


def compute_angle():
    ...


def compute_window(center: int,
                   width: int,
                   *,
                   t_width: list[int] | None = None,
                   interval: list[int] | None = None,
                   wrapping: bool = True,
                   min: int = 0,
                   max: int = 1) -> list[np.ndarray]:
    """
    Creates a flat windowing function with raised cosine transition boundaries to
    evaluate.

    :param center: center of the flat portion of the function
    :param width: width of the flat portion of the function
    :param t_width: width of the raised cosine transition region
    :param interval: the range of values over which the function should be created
    :param wrapping: whether to wrap if it exceeds the range
    :param min: minimum value of the window
    :param max: maximum value of the window
    :return: range of evaluated function and evaluation
    """
    t_width = t_width or [1, 1]
    interval = interval or [center - width / 2 - t_width[0], center + width / 2 + t_width[1]]
    min = min or 0
    max = max or 1

    minmax = abs(interval[-1] - interval[0])
    X1, Y1 = raised_cosine(t_width[0], center - width / 2 - t_width[0] / 2, (min, max))
    X2, Y2 = raised_cosine(t_width[1], center - width / 2 - t_width[1] / 2, (min, max))
    Y2 = -Y2 + (1 + min)
    x = np.concatenate((X1, X2))
    y = np.concatenate((Y1, Y2))

    if wrapping:
        new_range = np.arange(interval[0] - minmax, interval[1] + minmax, 1e-3)
        interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
        y_interp = interp(new_range)
        y_interp[np.isnan(y_interp)] = 0

        foo = np.where(x > interval[0])[0]
        condition = np.where(x > interval[1])[0]
        foo_end_length = len(condition)
        y_foo = y_interp[condition]
        foo = foo[-foo_end_length:]
        y_interp[foo] = y_foo

        foo = np.where(x > interval[0])[0]
        duh = y_interp[x > interval[1]]
        duh_cond = np.where(duh != 0)[0]
        foo = foo[:len(duh_cond)]
        y_interp[foo] = duh[duh_cond]

        ind = np.where((x >= interval[0]) & (x <= interval[1]))[0]
        xout = x[ind]
        yout = y_interp[ind]

    else:
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)(y)
        yout[np.isnan(yout)] = 0
        xout = np.arange(interval[0], interval[1], 1e-3)

        return [xout, yout]


def compute_log_window(center: int,
                       width: int,
                       *,
                       t_width: list[int] | None = None,
                       interval: list[int] | None = None,
                       wrapping: bool = True) -> list[np.ndarray]:
    """
    Creates a flat windowing function with raised cosine transition boundaries to
    evaluate in the log domain.

    :param center: center of the flat portion of the function
    :param width: width of the flat portion of the function
    :param t_width: width of the raised cosine transition region
    :param interval: the range of values over which the function should be created
    :param wrapping: whether to wrap if it exceeds the range
    :return: range of evaluated function and evaluation
    """
    t_width = t_width or [1, 1]
    interval = interval or [center - width / 2 - t_width[0], center + width / 2 + t_width[1]]

    minmax = abs(interval[-1] - interval[0])
    X1, Y1 = raised_cosine(t_width[0], center - width / 2 - t_width[0] / 2)
    X2, Y2 = raised_cosine(t_width[1], center - width / 2 - t_width[1] / 2)
    Y2 = -Y2 + 1
    zero_indices = np.where(Y2 == 0)[0]
    substitute = Y2[-len(zero_indices)]
    Y2[zero_indices] = substitute

    Y1 = np.log2(Y1)
    Y2 = np.log2(Y2)
    x = np.concatenate((X1, X2))
    y = np.concatenate((Y1, Y2))

    if wrapping:
        # Define the new range for interpolation
        new_range = np.arange(interval[0] - minmax, interval[1] + minmax, 0.001)

        # Create the interpolation function
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)

        # Apply the interpolation function to the new range
        y_interp = interp_func(new_range)
        y_interp[np.isnan(y_interp)] = 0
        x = new_range

        foo = np.where(x < interval[1])[0]
        y_sub = y_interp[np.where(x < interval[0])[0]]
        foo_end_length = len(y_sub)
        y_foo = y_sub
        foo = foo[-foo_end_length:]
        y_interp[foo] = y_foo

        foo = np.where(x > interval[0])[0]
        duh = y_interp[x > interval[1]]
        duh_cond = np.where(duh != 0)[0]
        foo = foo[:len(duh_cond)]
        y_interp[foo] = duh[duh_cond]

        ind = np.where((x >= interval[0]) & (x <= interval[1]))[0]
        xout = x[ind]
        yout = y_interp[ind]

    else:
        # Define the new range for interpolation
        new_range = np.arange(interval[0], minmax, 0.001)

        # Create the interpolation function
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)

        # Apply the interpolation function to the new range
        yout = interp_func(new_range)
        xout = new_range

    return [xout, yout]
