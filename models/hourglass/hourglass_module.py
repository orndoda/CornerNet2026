import torch
from torch.nn import nn
from torch.testing._internal.common_quantization import InnerModule
from torchvision.models.optical_flow.raft import ResidualBlock
from typing import Callable, Literal

from .residual_module import *

def get_inner_layer(inner_module: Literal["single", "double", "triple"]):
    if inner_module == "single":
        InnerModule = ResidualBlock
    elif inner_module == "double":
        InnerModule = DoubleResInnerModule
    elif inner_module == "triple":
        InnerModule = TripleResInnerModule
    else:
        raise ValueError("Inner module must be either 'single', 'double', or 'triple'")

    return InnerModule

class DoubleResInnerModule(nn.Module):
    def __init__(self, C_in:int=256, C_mid:int=128, C_out:int=256,
                     normalization: Callable=nn.BatchNorm2d):
        super(DoubleResInnerModule, self).__init__()

        self.block = nn.Sequential(
            ResidualBlock(C_in, C_mid, C_out, normalization)
            ,ResidualBlock(C_in, C_mid, C_out, normalization)
        )

    def forward(self, x):
        return self.block(x)

class TripleResInnerModule(nn.Module):
    def __init__(self, C_in:int=256, C_mid:int=128, C_out:int=256,
                 normalization: Callable=nn.BatchNorm2d):
        super(TripleResInnerModule, self).__init__()

        self.block = nn.Sequential(
            ResidualBlock(C_in, C_mid, C_out, normalization)
            ,ResidualBlock(C_in, C_mid, C_out, normalization)
            ,ResidualBlock(C_in, C_mid, C_out, normalization)
        )

    def forward(self, x):
        return self.block(x)

class HourglassModuleDown(nn.Module):
    def __init__(self, C_in:int=256, C_mid:int=128, C_out:int=256,
                 normalization: Callable=nn.BatchNorm2d,
                 inner_module: Literal["single", "double", "triple"] = "single"):
        super(HourglassModuleDown, self).__init__()

        self.InnerModule = get_inner_layer(inner_module)

        self.block1 = self.InnerModule(C_in, C_mid, C_out, normalization)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = self.InnerModule(C_in, C_mid, C_out, normalization)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = self.InnerModule(C_in, C_mid, C_out, normalization)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = self.InnerModule(C_in, C_mid, C_out, normalization)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x.contiguous(memory_format=torch.channels_last)

        x_32 = self.pool1(self.block1(x))
        x_16 = self.pool2(self.block2(x_32))
        x_8 = self.pool3(self.block3(x_16))
        x_4 = self.pool4(self.block4(x_8))
        return x, x_32, x_16, x_8, x_4

class HourglassModuleMiddle(nn.Module):
    def __init__(self, C_in:int=256, C_mid:int=128, C_out:int=256,
                 normalization: Callable=nn.BatchNorm2d,
                 inner_module: Literal["single", "double", "triple"] = "single"):
        super(HourglassModuleMiddle, self).__init__()

        self.InnerModule = get_inner_layer(inner_module)

        self.block = nn.Sequential(
            InnerModule(C_in, C_mid, C_out, normalization)
            ,InnerModule(C_in, C_mid, C_out, normalization)
            ,InnerModule(C_in, C_mid, C_out, normalization)
        )

    def forward(self, x):
        return self.block(x)

class HourglassModule(nn.Module):
    def __init__(self, C_in:int=256, C_mid:int=128, C_out:int=256,
                 normalization: Callable=nn.BatchNorm2d,
                 inner_module: Literal["single", "double", "triple"] = "single",
                 up_sample_method: Literal["nearest", "bilinear", "bicubic"] = "bilinear"):
        super(HourglassModule, self).__init__()

        self.down = HourglassModuleDown(C_in, C_mid, C_out, normalization, inner_module=inner_module)
        self.middle = HourglassModuleMiddle(C_in, C_mid, C_out, normalization, inner_module=inner_module)
        self.InnerModule = get_inner_layer(inner_module)
        self.up_sample_method = up_sample_method
        self.upsample = nn.Upsample(scale_factor=2, mode=self.up_sample_method)

        self.block1 = self.InnerModule(C_in, C_mid, C_out, normalization)
        self.block2 = self.InnerModule(C_in, C_mid, C_out, normalization)
        self.block3 = self.InnerModule(C_in, C_mid, C_out, normalization)
        self.block4 = self.InnerModule(C_in, C_mid, C_out, normalization)

    def forward(self, x):
        x_64, x_32, x_16, x_8, x_4 = self.down(x)
        mid = self.middle(x_4)

        up = self.upsample(mid)+x_8
        up = self.block1(up)

        up = self.upsample(up)+x_16
        up = self.block2(up)

        up = self.upsample(up)+x_32
        up = self.block3(up)

        up = self.upsample(up)+x_64
        up = self.block4(up)
        return up