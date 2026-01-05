import torch
import torch.nn as nn
from typing import Callable, Literal
from torch.nn import functional as F
from .unet_module import UnetModule
from .residual_module import ResidualModule

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

class IntermediateResidualModule(nn.Module):
    def __init__(self, normalization: Callable = nn.BatchNorm2d, intermdiate_super: bool = True):
        super(IntermediateResidualModule, self).__init__()
        self.normalization = normalization
        self.intermdiate_super = intermdiate_super

        self.res1 = ResidualModule(64, 32, 64, normalization=normalization)
        self.res2 = ResidualModule(64, 32, 64, normalization=normalization)
        if self.intermdiate_super:
            self.res3 = ResidualModule(64, 32, 64, normalization=normalization)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)

        if self.intermdiate_super:
            return x, self.res3(x)

        return x

class StackedUnet(nn.Module):
    def __init__(self, in_channels: int=3, normalization: Callable=nn.BatchNorm2d,
                 inner_module: int=1,
                 interpolation_mode: Literal["bilinear","bicubic"]="bilinear",
                 intermediate_super: bool = True,
        ):
        super(StackedUnet, self).__init__()

        self.normalization = normalization
        self.inner_module = inner_module
        self.interpolation_mode = interpolation_mode
        self.in_channels = in_channels
        self.intermediate_super = intermediate_super

        self.downsample = DownSample(self.normalization)

        self.unet_stack1 = UnetModule(self.normalization, self.inner_module)
        self.inter1 = IntermediateResidualModule(self.normalization, self.intermediate_super)

        self.unet_stack2 = UnetModule(self.normalization, self.inner_module)
        self.inter2 = IntermediateResidualModule(self.normalization, self.intermediate_super)

        self.unet_stack3 = UnetModule(self.normalization, self.inner_module)
        self.inter3 = IntermediateResidualModule(self.normalization, self.intermediate_super)

    def forward(self, x):
        x = F.interpolate(x,
                          size=(640, 640),
                          mode=self.interpolation_mode, # Use 'bilinear' for smooth results
                          align_corners=False)

        x = self.downsample(x)

        x = self.unet_stack1(x)
        if self.intermediate_super:
            x, out1 = self.inter1(x)
        else:
            x = self.inter1(x)

        x = self.unet_stack2(x)
        if self.intermediate_super:
            x, out2 = self.inter2(x)
        else:
            x = self.inter2(x)

        x = self.unet_stack3(x)
        if self.intermediate_super:
            x, out3 = self.inter3(x)
        else:
            x = self.inter3(x)

        if self.intermediate_super:
            return x, [out1, out2, out3]

        return x, []