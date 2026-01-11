import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Literal

class ResidualBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_mid: int,
        C_out: int,
        stride: int = 1,
        normalization: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: Callable[[], nn.Module] = nn.ReLU,
        pre_activation: bool = False,   # default = Hourglass-104 style (post-activation)
    ):
        super(ResidualBlock, self).__init__()

        self.pre_activation = pre_activation

        if not pre_activation:
            self.main = nn.Sequential(
                nn.Conv2d(C_in, C_mid, 1, bias=False),
                normalization(C_mid),
                activation(),

                nn.Conv2d(C_mid, C_mid, 3, stride=stride, padding=1, bias=False),
                normalization(C_mid),
                activation(),

                nn.Conv2d(C_mid, C_out, 1, bias=False),
                normalization(C_out),
            )
        else:
            self.main = nn.Sequential(
                normalization(C_in),
                activation(),
                nn.Conv2d(C_in, C_mid, 1, bias=False),

                normalization(C_mid),
                activation(),
                nn.Conv2d(C_mid, C_mid, 3, stride=stride, padding=1, bias=False),

                normalization(C_mid),
                activation(),
                nn.Conv2d(C_mid, C_out, 1, bias=False),

                normalization(C_out),
            )

        if C_in != C_out or stride != 1:
            self.skip = nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

        self.final_act = activation()


    def forward(self, x):
        out = self.main(x)
        skip = self.skip(x)
        return self.final_act(out + skip)

class HardDown(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        normalization: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: Callable[[], nn.Module] = nn.ReLU,
        pre_activation: bool = False,
    ):
        super(HardDown, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(C_in, C_out//2, kernel_size=7, stride=2, padding=3),
            ResidualBlock(C_out//2, C_out//4, C_out, 2, normalization, activation, pre_activation),
        )

    def forward(self, x):
        return self.block(x)

class DownBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        normalization: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: Callable[[], nn.Module] = nn.ReLU,
        pre_activation: bool = False,
    ):
        super(DownBlock, self).__init__()

        self.block = nn.Sequential(
            ResidualBlock(C_in, C_in//2, C_out, 2, normalization, activation, pre_activation),
            ResidualBlock(C_out, C_out//2, C_out, 1, normalization, activation, pre_activation),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        normalization: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: Callable[[], nn.Module] = nn.ReLU,
        pre_activation: bool = False,
        upsample_mode: Literal["nearest", "deconv"] = "nearest"
    ):
        super().__init__()

        if upsample_mode == "nearest":
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(C_in, C_in, kernel_size=1, bias=False),
            )
        elif upsample_mode == "deconv":
            self.upsample = nn.ConvTranspose2d(
                C_in, C_in, kernel_size=2, stride=2, bias=False
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        self.block = nn.Sequential(
            ResidualBlock(
                C_in, C_in // 2, C_out,
                stride=1,
                normalization=normalization,
                activation=activation,
                pre_activation=pre_activation,
            ),
            ResidualBlock(
                C_out, C_out // 2, C_out,
                stride=1,
                normalization=normalization,
                activation=activation,
                pre_activation=pre_activation,
            ),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        return self.block(x) + skip