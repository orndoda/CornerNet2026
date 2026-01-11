import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Literal
from .blocks import *

class Hourglass(nn.Module):
    def __init__(self,
                 normalization: Callable[[int], nn.Module] = nn.BatchNorm2d,
                 activation: Callable[[], nn.Module] = nn.ReLU,
                 pre_activation: bool = False,
                 upsample_mode: Literal["nearest", "deconv"] = "nearest",
                 interpolation_mode: Literal["bilinear", "bicubic"] = "bilinear"):
        super(Hourglass, self).__init__()

        self.normalization = normalization
        self.activation = activation
        self.pre_activation = pre_activation
        self.upsample_mode = upsample_mode
        self.interpolation_mode = interpolation_mode

        self.intake = HardDown(3, 256,
                               self.normalization, self.activation, self.pre_activation)

        self.hourglass1 = HourglassBlock(self.normalization, self.activation,
                                         self.pre_activation, self.upsample_mode)
        self.hourglass2 = HourglassBlock(self.normalization, self.activation,
                                         self.pre_activation, self.upsample_mode)

        self.conv_hourglass1_out = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
        )
        self.conv_hourglass1_in = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
        )

    def forward(self, x: torch.Tensor, train: bool=True) -> torch.Tensor:
        x = F.interpolate(x,
                          size=(512, 512),
                          mode=self.interpolation_mode,  # Use 'bilinear' for smooth results
                          align_corners=False)
        print(x.shape)
        x = self.intake(x)
        hourglass1_out = self.hourglass1(x)

        out1 = F.relu(self.conv_hourglass1_out(hourglass1_out)+self.conv_hourglass1_in(x))
        out = self.hourglass2(out1)
        if train:
            return out, hourglass1_out

        return out