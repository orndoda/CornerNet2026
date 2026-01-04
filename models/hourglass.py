import torch
import torch.nn as nn

class ResidualModule(nn.Module):
    def __init__(self):
        super(ResidualModule, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=1)
            ,nn.GELU()
            ,nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            ,nn.GELU()
            ,nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=1)
        )

    def forward(self, x):
        return x+self.block(x)