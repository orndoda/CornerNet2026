import torch
import torch.nn as nn

class ResidualModule(nn.Module):
    def __init__(self, C_in=256, C_mid=128, C_out=256):
        super(ResidualModule, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(C_in)
            ,nn.GELU()
            ,nn.Conv2d(C_in, C_mid, kernel_size=1, padding=1)

            ,nn.BatchNorm2d(C_mid)
            ,nn.GELU()
            ,nn.Conv2d(C_mid, C_mid, kernel_size=3, padding=1)

            ,nn.BatchNorm2d(C_mid)
            ,nn.GELU()
            ,nn.Conv2d(C_mid, C_out, kernel_size=1, padding=1)
        )

    def forward(self, x):
        return x+self.block(x)