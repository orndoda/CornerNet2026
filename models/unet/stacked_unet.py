import torch
import torch.nn as nn
from typing import Callable, Literal
from torch.nn import functional as F
from .unet_module import UnetModule
from .residual_module import ResidualModule
from models.utils import DownSample

# Your import style preserved
from .residual_module import *

class IntermediateResidualModule(nn.Module):
    def __init__(self, normalization: Callable = nn.BatchNorm2d):
        super(IntermediateResidualModule, self).__init__()
        self.normalization = normalization

        self.res1 = ResidualModule(64, 32, 64, normalization=normalization)
        self.res2 = ResidualModule(64, 32, 64, normalization=normalization)

        self.res3 = ResidualModule(64, 32, 64, normalization=normalization)

    def forward(self, x):
        x = self.res1(x)
        inter_out = x
        x = self.res2(x)

        return x + self.res3(inter_out), inter_out

class StackedUnet(nn.Module):
    def __init__(self, in_channels: int=3, normalization: Callable=nn.BatchNorm2d,
                 inner_module: int=1,
                 interpolation_mode: Literal["bilinear","bicubic"]="bilinear",
        ):
        super(StackedUnet, self).__init__()

        self.normalization = normalization
        self.inner_module = inner_module
        self.interpolation_mode = interpolation_mode
        self.in_channels = in_channels

        self.downsample = DownSample(self.normalization)

        self.unet_stack1 = UnetModule(self.normalization, self.inner_module)
        self.inter1 = IntermediateResidualModule(self.normalization)

        self.unet_stack2 = UnetModule(self.normalization, self.inner_module)
        self.inter2 = IntermediateResidualModule(self.normalization)

        self.unet_stack3 = UnetModule(self.normalization, self.inner_module)
        self.inter3 = IntermediateResidualModule(self.normalization)

    def forward(self, x):
        x = F.interpolate(x,
                          size=(512, 512),
                          mode=self.interpolation_mode, # Use 'bilinear' for smooth results
                          align_corners=False)

        x = self.downsample(x)

        u1 = self.unet_stack1(x)
        x, out1 = self.inter1(u1)
        x = u1 + x

        u2 = self.unet_stack1(x)
        x, out2 = self.inter1(u2)
        x = u2 + x

        u3 = self.unet_stack1(x)
        x, out3 = self.inter1(u3)
        x = u3 + x

        return x, [out1, out2, out3]