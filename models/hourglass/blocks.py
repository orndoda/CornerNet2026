import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


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
        super().__init__()

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