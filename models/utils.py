import torch
import torch.nn as nn
from typing import Callable

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class DownSample(nn.Module):
    def __init__(self, in_channels: int=3, normalization: Callable = nn.BatchNorm2d):
        super(DownSample, self).__init__()

        self.normalization = normalization

        self.downsample = nn.Sequential(
            #512 -> 256
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.Identity(),
            nn.GELU(),

            #256 -> 128
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            self.normalization(32),
            nn.GELU(),

            #128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            self.normalization(64),
            nn.GELU(),
        )

    def forward(self, x):
        return self.downsample(x)