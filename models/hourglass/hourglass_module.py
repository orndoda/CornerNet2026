import torch
import torch.nn as nn
from typing import Callable, Literal, Tuple

# Your import style preserved
from .residual_module import *

def get_inner_layer(inner_module: Literal["single", "double", "triple"]) -> Callable:
    """
    Factory function to retrieve the appropriate inner residual module.

    Parameters
    ----------
    inner_module : Literal["single", "double", "triple"]
        Specifies the depth of the residual inner module.

    Returns
    -------
    Callable
        A residual module class.
    """
    if inner_module == "single":
        InnerModule = ResidualModule
    elif inner_module == "double":
        InnerModule = DoubleResInnerModule
    elif inner_module == "triple":
        InnerModule = TripleResInnerModule
    else:
        raise ValueError("Inner module must be either 'single', 'double', or 'triple'")
    return InnerModule

class DoubleResInnerModule(nn.Module):
    """
    Two stacked residual modules.
    """

    def __init__(
            self,
            C_in: int = 256,
            C_mid: int = 128,
            C_out: int = 256,
            normalization: Callable = nn.BatchNorm2d,
    ):
        super(DoubleResInnerModule, self).__init__()

        self.block = nn.Sequential(
            ResidualModule(C_in, C_mid, C_out, normalization),
            ResidualModule(C_out, C_mid, C_out, normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        return self.block(x)

class TripleResInnerModule(nn.Module):
    """
    Three stacked residual modules.
    """

    def __init__(
            self,
            C_in: int = 256,
            C_mid: int = 128,
            C_out: int = 256,
            normalization: Callable = nn.BatchNorm2d,
    ):
        super(TripleResInnerModule, self).__init__()

        self.block = nn.Sequential(
            ResidualModule(C_in, C_mid, C_out, normalization),
            ResidualModule(C_out, C_mid, C_out, normalization),
            ResidualModule(C_out, C_mid, C_out, normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        return self.block(x)

class HourglassModuleDown(nn.Module):
    """
    Downsampling path of the hourglass module.
    """

    def __init__(
            self,
            C_in: int = 256,
            C_mid: int = 128,
            C_out: int = 256,
            normalization: Callable = nn.BatchNorm2d,
            inner_module: Literal["single", "double", "triple"] = "single",
    ):
        super(HourglassModuleDown, self).__init__()

        InnerModule = get_inner_layer(inner_module)

        self.block1 = InnerModule(C_in, C_mid, C_out, normalization)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = InnerModule(C_out, C_mid, 2 * C_out, normalization)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = InnerModule(2 * C_out, C_mid, 4 * C_out, normalization)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = InnerModule(4 * C_out, C_mid, 8 * C_out, normalization)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the downsampling path.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Feature maps at multiple resolutions.
        """
        x = x.contiguous(memory_format=torch.channels_last)

        x_64 = x
        x_32 = self.pool1(self.block1(x_64))
        x_16 = self.pool2(self.block2(x_32))
        x_8 = self.pool3(self.block3(x_16))
        x_4 = self.pool4(self.block4(x_8))

        return x_64, x_32, x_16, x_8, x_4

class HourglassModuleMiddle(nn.Module):
    """
    Bottleneck (middle) of the hourglass module.
    """

    def __init__(
            self,
            C_in: int = 256,
            C_mid: int = 128,
            C_out: int = 256,
            normalization: Callable = nn.BatchNorm2d,
            inner_module: Literal["single", "double", "triple"] = "single",
    ):
        super(HourglassModuleMiddle, self).__init__()

        InnerModule = get_inner_layer(inner_module)

        self.block = nn.Sequential(
            InnerModule(8 * C_out, C_mid, 8 * C_out, normalization),
            InnerModule(8 * C_out, C_mid, 8 * C_out, normalization),
            InnerModule(8 * C_out, C_mid, 8 * C_out, normalization),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        return self.block(x)

class HourglassModuleUp(nn.Module):
    """
    Upsampling path of the hourglass module.
    """

    def __init__(
            self,
            C_out: int = 256,
            C_mid: int = 128,
            normalization: Callable = nn.BatchNorm2d,
            inner_module: Literal["single", "double", "triple"] = "single",
            up_sample_method: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
    ):
        super(HourglassModuleUp, self).__init__()

        InnerModule = get_inner_layer(inner_module)

        if up_sample_method == "nearest":
            self.upsample = nn.Upsample(scale_factor=2, mode=up_sample_method)
        else:
            self.upsample = nn.Upsample(
                scale_factor=2, mode=up_sample_method, align_corners=False
            )

        self.proj8_to4 = nn.Conv2d(8 * C_out, 4 * C_out, kernel_size=1)
        self.proj4_to2 = nn.Conv2d(4 * C_out, 2 * C_out, kernel_size=1)
        self.proj2_to1 = nn.Conv2d(2 * C_out, C_out, kernel_size=1)
        self.proj1_toin = nn.Conv2d(C_out, C_out, kernel_size=1)

        self.block1 = InnerModule(4 * C_out, C_mid, 4 * C_out, normalization)
        self.block2 = InnerModule(2 * C_out, C_mid, 2 * C_out, normalization)
        self.block3 = InnerModule(C_out, C_mid, C_out, normalization)
        self.block4 = InnerModule(C_out, C_mid, C_out, normalization)

    def forward(
            self,
            mid: torch.Tensor,
            x_8: torch.Tensor,
            x_16: torch.Tensor,
            x_32: torch.Tensor,
            x_64: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the upsampling path.

        Parameters
        ----------
        mid : torch.Tensor
            Bottleneck feature map.
        x_8, x_16, x_32, x_64 : torch.Tensor
            Skip connection feature maps.

        Returns
        -------
        torch.Tensor
            Reconstructed high-resolution feature map.
        """
        up = self.upsample(mid)
        up = self.proj8_to4(up) + x_8
        up = self.block1(up)

        up = self.upsample(up)
        up = self.proj4_to2(up) + x_16
        up = self.block2(up)

        up = self.upsample(up)
        up = self.proj2_to1(up) + x_32
        up = self.block3(up)

        up = self.upsample(up)
        up = self.proj1_toin(up) + x_64
        up = self.block4(up)

        return up

class HourglassModule(nn.Module):
    """
    Full hourglass module combining downsampling, bottleneck, and upsampling paths.
    """

    def __init__(
            self,
            C_in: int = 256,
            C_mid: int = 128,
            C_out: int = 256,
            normalization: Callable = nn.BatchNorm2d,
            inner_module: Literal["single", "double", "triple"] = "single",
            up_sample_method: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
    ):
        super(HourglassModule, self).__init__()

        self.down = HourglassModuleDown(
            C_in, C_mid, C_out, normalization, inner_module=inner_module
        )
        self.middle = HourglassModuleMiddle(
            C_in, C_mid, C_out, normalization, inner_module=inner_module
        )
        self.up = HourglassModuleUp(
            C_out=C_out,
            C_mid=C_mid,
            normalization=normalization,
            inner_module=inner_module,
            up_sample_method=up_sample_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full hourglass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        x_64, x_32, x_16, x_8, x_4 = self.down(x)
        mid = self.middle(x_4)
        up = self.up(mid, x_8, x_16, x_32, x_64)
        return up