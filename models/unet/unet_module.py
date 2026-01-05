import torch
import torch.nn as nn
from typing import Tuple

# Your import style preserved
from .residual_module import *

class UnetModuleDown(nn.Module):
    """
    Downsampling path of the unet module.
    """

    def __init__(
            self,
            normalization: Callable = nn.BatchNorm2d,
            inner_module: int=1,
    ):
        super(UnetModuleDown, self).__init__()

        self.block1 = nn.Sequential(
            *[ResidualModule(64, 32, 64, normalization) for _ in range(inner_module)]
        )
        self.pool1 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.block2 = nn.Sequential(
            *[ResidualModule(128, 64, 128, normalization) for _ in range(inner_module)]
        )
        self.pool2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)

        self.block3 = nn.Sequential(
            *[ResidualModule(256, 128, 256, normalization) for _ in range(inner_module)]
        )
        self.pool3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            *[ResidualModule(512, 256, 512, normalization) for _ in range(inner_module)]
        )
        self.pool4 = nn.Conv2d(512, 1024, kernel_size=2, stride=2)

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
        x_64 = x
        x_32 = self.pool1(self.block1(x_64))
        x_16 = self.pool2(self.block2(x_32))
        x_8 = self.pool3(self.block3(x_16))
        x_4 = self.pool4(self.block4(x_8))

        return x_64, x_32, x_16, x_8, x_4

class UnetModuleMiddle(nn.Module):
    """
    Bottleneck (middle) of the unet module.
    """

    def __init__(
            self,
            normalization: Callable = nn.BatchNorm2d,
            inner_module: int=1,
    ):
        super(UnetModuleMiddle, self).__init__()

        self.block = nn.Sequential(
            nn.Sequential(
                *[ResidualModule(1024, 512, 1024, normalization) for _ in range(3*inner_module)]
            )
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

class UnetModuleUp(nn.Module):
    """
    Upsampling path of the unet module.
    """

    def __init__(
            self,
            normalization: Callable = nn.BatchNorm2d,
            inner_module: int=1,
    ):
        super(UnetModuleUp, self).__init__()

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.block1 = nn.Sequential(
            *[ResidualModule(512, 256, 512, normalization) for _ in range(inner_module)]
        )

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            *[ResidualModule(256, 128, 256, normalization) for _ in range(inner_module)]
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.block3 = nn.Sequential(
            *[ResidualModule(128, 64, 128, normalization) for _ in range(inner_module)]
        )

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.block4 = nn.Sequential(
            *[ResidualModule(64, 32, 64, normalization) for _ in range(inner_module)]
        )

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
        up = self.up1(mid) + x_8
        up = self.block1(up)

        up = self.up2(up) + x_16
        up = self.block2(up)

        up = self.up3(up) + x_32
        up = self.block3(up)

        up = self.up4(up) + x_64
        up = self.block4(up)

        return up

class UnetModule(nn.Module):
    def __init__(self,
                 normalization: Callable=nn.BatchNorm2d,
                 inner_module: int=1,
        ):
        super(UnetModule, self).__init__()

        self.normalization = normalization
        self.inner_module = inner_module

        self.down = UnetModuleDown(self.normalization, self.inner_module)
        self.middle = UnetModuleMiddle(self.normalization, self.inner_module)
        self.up = UnetModuleUp(self.normalization, self.inner_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_64, x_32, x_16, x_8, x_4 = self.down(x)
        mid = self.middle(x_4)
        up = self.up(mid, x_8, x_16, x_32, x_64)
        return up