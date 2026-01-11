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

class HourglassBlock(nn.Module):
    def __init__(
        self,
        normalization: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: Callable[[], nn.Module] = nn.ReLU,
        pre_activation: bool = False,
        upsample_mode: Literal["nearest", "deconv"] = "nearest",
    ):
        super(HourglassBlock, self).__init__()

        # Channel progression
        C0 = 256
        C1 = 384
        C2 = 384
        C3 = 384
        C4 = 512

        # -----------------------------
        # DOWN PATH
        # -----------------------------
        self.down1 = DownBlock(C0, C1, normalization, activation, pre_activation)
        self.down2 = DownBlock(C1, C2, normalization, activation, pre_activation)
        self.down3 = DownBlock(C2, C3, normalization, activation, pre_activation)
        self.down4 = DownBlock(C3, C4, normalization, activation, pre_activation)

        # -----------------------------
        # BOTTOM (4 residual blocks)
        # -----------------------------
        self.bottom = nn.Sequential(
            ResidualBlock(C4, C4 // 2, C4, 1, normalization, activation, pre_activation),
            ResidualBlock(C4, C4 // 2, C4, 1, normalization, activation, pre_activation),
            ResidualBlock(C4, C4 // 2, C4, 1, normalization, activation, pre_activation),
            ResidualBlock(C4, C4 // 2, C4, 1, normalization, activation, pre_activation),
        )

        # -----------------------------
        # SKIP PROJECTIONS
        # -----------------------------
        # Project each down feature to the channels expected by the corresponding UpBlock
        self.skip1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C1, C0, 1, bias=False),
        )
        self.skip2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C2, C1, 1, bias=False),
        )
        self.skip3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C3, C2, 1, bias=False),
        )
        self.skip4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C4, C3, 1, bias=False),
        )
        # -----------------------------
        # UP PATH
        # -----------------------------
        self.up4 = UpBlock(C4, C3, normalization, activation, pre_activation, upsample_mode)
        self.up3 = UpBlock(C3, C2, normalization, activation, pre_activation, upsample_mode)
        self.up2 = UpBlock(C2, C1, normalization, activation, pre_activation, upsample_mode)
        self.up1 = UpBlock(C1, C0, normalization, activation, pre_activation, upsample_mode)

    def forward(self, x):
        # Down path
        d1 = self.down1(x)   # C1 (384)
        d2 = self.down2(d1)  # C2 (384)
        d3 = self.down3(d2)  # C3 (384)
        d4 = self.down4(d3)  # C4 (512)

        # Bottom
        b = self.bottom(d4)  # C4 (512)

        # Up path (note corrected skip tensors)
        u4 = self.up4(b,  self.skip4(d4))  # -> C3 (384)
        u3 = self.up3(u4, self.skip3(d3))  # -> C2 (384)
        u2 = self.up2(u3, self.skip2(d2))  # -> C1 (384)
        u1 = self.up1(u2, self.skip1(d1))  # -> C0 (256)

        return u1