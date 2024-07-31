#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision as tv

__all__ = [
    'GaussianPyramid'
]

class GaussianPyramid(nn.Module):
    def __init__(self,
                 kernel_size: list[int] = [5, 5],
                 sigma: list[float] = [0.1, 2.0],
                 n_levels: int = 3):
        super(GaussianPyramid, self).__init__()
        self.n_levels = n_levels
        self.sigma = sigma
        self.gauss = tv.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.upsample = nn.functional.interpolate

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f'Tensor should have 4 dimensions but has {x.ndim}.'
        return x[:, :, ::2, ::2]

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        x += 128
        return torch.clamp(x, 0, 255)

    def forward(self, img: torch.Tensor, max_levels: int | None = None) -> list[torch.Tensor]:
        max_levels = max_levels or self.n_levels
        current = img
        pyr = []
        for level in range(max_levels):
            filtered = self.gauss(current)
            down = self.downsample(filtered)
            up = self.upsample(down, size=current.size()[2:])
            diff = current-up
            pyr.append(self.clamp(diff))
            current = down
        return pyr
