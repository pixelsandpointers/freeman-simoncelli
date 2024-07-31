#!/usr/bin/env python3
from enum import Enum
from typing import NamedTuple
from .window import WindowType

__all__ = [
    'Config'
]

"""
Options are:
debugging
Na: number of positions in V1 = 7
Nsc: number of scales in V1 = 4
Nor: number of orientations in V1 = 4

outputPath
nIters: number of iterations for synthesis = 50
windowType: radial or square
aspect: circumferential aspect ration = 2
scale: size / eccentricity = 0.5
overlap: overlap between windows = 0.5
centerRadPerc: center radial perc = 0.025
origin = []

Usage:
======

% load original image
oim = double(imread('example-im-512x512.png'));

% set options
opts = metamerOpts(oim,'windowType=square','nSquares=[3 1]');

% make windows
m = mkImMasks(opts) => generates the window functions

% plot windows
plotWindows(m,opts);

% do metamer analysis on original (measure statistics)
params = metamerAnalysis(oim,m,opts);

% do metamer synthesis (generate new image matched for statistics)
res = metamerSynthesis(params,size(oim),m,opts);


%% METAMER DEMO (will take a few min per iteration)
%
% This version uses windows that tile the image in
% polar angle and log eccentricity, with parameters
% used to generate metamers in Freeman & Simoncelli

% load original image
oim = double(imread('example-im-512x512.png'));

% set options
opts = metamerOpts(oim,'windowType=radial','scale=0.5','aspect=2');

% make windows
m = mkImMasks(opts);

% plot windows
plotWindows(m,opts);

% do metamer analysis on original (measure statistics)
params = metamerAnalysis(oim,m,opts);

% do metamer synthesis (generate new image matched for statistics)
res = metamerSynthesis(params,size(oim),m,opts);
"""

# TODO for later, CLI for window exploration
class Parser:
    ...


class Config(NamedTuple):
    image_shape: list[int] = [512, 512]
    number_of_positions: int = 7
    number_of_scales: int = 4
    number_of_orientations: int = 4
    window_type: WindowType = WindowType.square
    circumferential_aspect: int = 1
    scale: float = 0.5
    window_overlap: float = 0.5
    center_rad_perc: float = 0.025
    origin: list = []
