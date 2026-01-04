import torch
import torch.nn as nn
from .utils import *
from typing import Callable

class ResidualModule(nn.Module):
    def __init__(self, C_in:int=256, C_mid:int=128, C_out:int=256, normalization: Callable=nn.BatchNorm2d):
        super(ResidualModule, self).__init__()

        self.block = nn.Sequential(
            normalization(C_in)
            ,nn.GELU()
            ,nn.Conv2d(C_in, C_mid, kernel_size=1)

            ,normalization(C_mid)
            ,nn.GELU()
            ,nn.Conv2d(C_mid, C_mid, kernel_size=3, padding="same")

            ,normalization(C_mid)
            ,nn.GELU()
            ,nn.Conv2d(C_mid, C_out, kernel_size=1)
        )

    def forward(self, x):
        return x+self.block(x)