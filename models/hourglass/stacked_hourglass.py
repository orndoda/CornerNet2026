import torch
import torch.nn as nn
from typing import Callable, Literal
from torch.nn import functional as F
from .hourglass_module import HourglassModule

# Your import style preserved
from .residual_module import *

class DownSample(nn.Module):
    def __init__(self, in_channels: int=3, normalization: Callable = nn.BatchNorm2d):
        super(DownSample, self).__init__()

        self.normalization = normalization

        self.downsample = nn.Sequential(
            # 640 → 320
            nn.Identity(in_channels),
            nn.GELU(),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),

            # 320 -> 64
            self.normalization(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=5, padding=0),
        )

    def forward(self, x):
        return self.downsample(x)

class StackedHourglass(nn.Module):
    def __init__(self, in_channels: int=3, normalization: Callable=nn.BatchNorm2d,
                 inner_module: int=1,
                 interpolation_mode: Literal["bilinear","bicubic"]="bilinear",
        ):
        super(StackedHourglass, self).__init__()

        self.normalization = normalization
        self.inner_module = inner_module
        self.interpolation_mode = interpolation_mode
        self.in_channels = in_channels

        self.downsample = DownSample(self.normalization)

        self.hourglass_stack = nn.Sequential(
            HourglassModule(self.normalization, self.inner_module),
            HourglassModule(self.normalization, self.inner_module),
            HourglassModule(self.normalization, self.inner_module),
            HourglassModule(self.normalization, self.inner_module),
            HourglassModule(self.normalization, self.inner_module),
        )

    def forward(self, x):
        x = F.interpolate(x,
                          size=(640, 640),
                          mode=self.interpolation_mode, # Use 'bilinear' for smooth results
                          align_corners=False)

        x = self.downsample(x)
        return self.hourglass_stack(x)