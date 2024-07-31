"""
Windowing functions
"""
from abc import ABC, abstractmethod
from enum import Enum

from plenoptic.simulate.canonical_computations import SteerablePyramidFreq

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import GaussianPyramid
from .config import Config

__all__ = [
    'RadialWindow',
    'SquareWindow',
    'WindowType',
]

class WindowFunction(nn.Module, ABC):
    @classmethod
    def create_from_config(cls, config: Config) -> "Self":
        ...

    @abstractmethod
    @staticmethod
    def create_mask():
        ...


class RadialWindow(WindowFunction):
    def create_from_config(cls, config: Config) -> "Self":
        ...

    @staticmethod
    def create_mask():
        ...


class SquareWindow(WindowFunction):
    def create_from_config(cls, config: Config) -> "Self":
        ...

    @staticmethod
    def create_mask():
        ...


# TODO for later, replace strings with window classes
class WindowType(Enum):
    radial = RadialWindow.create_from_config
    square = SquareWindow.create_from_config

def generate_image_masks(config: Config) -> "ImageMasks":
    pyr = SteerablePyramidFreq(image_shape=config.image_shape,
                               height=config.number_of_scales,
                               order=config.number_of_orientations)

    # get scales
    scales = pyr.scales


    # construct empty masks
    for i in range(1, config.number_of_scales+1):
        # m.scale{i}.maskMat = zeros(m.scale{1}.nMasks,scales(i,1),scales(i,2));
        # m.scale{i}.maskNorm = zeros(scales(i,1)*scales(i,2),m.scale{1}.nMasks);
        ...


    # compute windows
    windows = config.window_type.value(config)

    # more gaussian windows over empty mask
    gpyr = GaussianPyramid()

    # combine gaussian and masked windows

    # store indices should look like this
    # if Nsc == 4
    #     m.bandToMaskScale = [1 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5];
    # end
    # if Nsc == 5
    #     m.bandToMaskScale = [1 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6];
    # end
    band2masks = [i for _ in range(4) for i in range(1, config.number_of_scales-1)]
    band2masks.insert(1, 0)
    band2masks.append(config.number_of_scales)
