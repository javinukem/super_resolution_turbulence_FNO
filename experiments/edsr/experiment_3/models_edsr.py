# https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
import torch.nn as nn
import math
import sys
import os
import torch
sys.path.append(os.path.abspath(".."))  # This brings 'src' into the path
from src.utils.pixel_shuffle3d import PixelShuffle3d


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=kernel_size, stride=stride, padding="same"
        )
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=kernel_size, stride=stride, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv = self.conv1(x)
        x_conv = self.relu(x_conv)
        x_conv = self.conv2(x_conv)
        x = x_conv + x

        return x


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1):
        super(ResBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, stride=stride, padding="same"
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, stride=stride, padding="same"
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv = self.conv1(x)
        x_conv = self.relu(x_conv)
        x_conv = self.conv2(x_conv)
        x = x_conv + x

        return x


class Upsampler(nn.Sequential):
    def __init__(
        self,
        scale: int = 4,
        n_feats: int = 5,
        batch_norm: bool = False,
        activation_f: str = False,
    ):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(
                    nn.Conv3d(
                        in_channels=n_feats,
                        out_channels=8 * n_feats,
                        kernel_size=3,
                        padding=1,
                    )
                )
                m.append(PixelShuffle3d(2))
                if batch_norm:
                    m.append(nn.BatchNorm3d(n_feats))
                if activation_f == "relu":
                    m.append(nn.ReLU(True))
                elif activation_f == "prelu":
                    m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(
        self,
        input_channels: int = 5,
        n_resblocks: int = 32,
        n_feats: int = 128,
        kernel_size: int = 3,
        scale: int = 4,
        activation_f: str = False,
    ):
        super().__init__()
        m_head = [nn.Conv3d(input_channels, n_feats, kernel_size, padding="same")]

        # define body module
        m_body = [ResBlock(n_feats, kernel_size) for _ in range(n_resblocks)]
        m_body.append(nn.Conv3d(n_feats, n_feats, kernel_size, padding="same"))

        # define tail module
        m_tail = [
            Upsampler(scale, n_feats, activation_f=activation_f),
            nn.Conv3d(n_feats, input_channels, kernel_size, padding="same"),
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        # ensure pressure and density are positives
        x[:, 0] = torch.exp(x[:, 0])
        x[:, 4] = torch.exp(x[:, 4])

        return x
