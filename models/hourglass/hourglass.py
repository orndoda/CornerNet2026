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
                 upsample_mode: Literal["nearest", "deconv"] = "nearest"):
        super(Hourglass, self).__init__()

        self.normalization = normalization
        self.activation = activation
        self.pre_activation = pre_activation

        self.intake = HardDown(3, 256,
                               self.normalization, self.activation, self.pre_activation)

        self.down1 = DownBlock(256, 256,
                               self.normalization, self.activation, self.pre_activation)
        self.down2 = DownBlock(256, 384,
                               self.normalization, self.activation, self.pre_activation)
        self.down3 = DownBlock(384, 384,
                               self.normalization, self.activation, self.pre_activation)
        self.down4 = DownBlock(384, 384,
                               self.normalization, self.activation, self.pre_activation)
        self.bottom = nn.Sequential(
            DownBlock(384, 512,
                      self.normalization, self.activation, self.pre_activation),
            ResidualBlock(512, 256, 512, 1,
                           self.normalization, self.activation, self.pre_activation),
            ResidualBlock(512, 256, 512, 1,
                            self.normalization, self.activation, self.pre_activation),
        )

        self.up1 = UpBlock(512, 512,
                           self.normalization, self.activation, self.pre_activation)