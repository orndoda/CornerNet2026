import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, C_f: int, D: int, stride: int):
        super(ResidualBlock, self).__init__()

        self.C_f = C_f
        self.D = D
        self.stride = stride

        self.conv1 = nn.Conv2d(C_f, D, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Conv2d(C_f, D, kernel_size=1, stride=stride, bias=False)

        self.norm = nn.BatchNorm2d(D)
        self.relu = nn.ReLU()

        self.block = nn.Sequential(
            self.conv1,
            self.norm,
            self.relu,
            self.conv2,
            self.norm,
            self.relu,
        )

    def forward(self, x):

        residual = self.shortcut(x) if self.stride>1 or self.D != self.C_f else x

        out = self.block(x)
        return out + residual