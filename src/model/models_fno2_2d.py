import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.models_edsr import ResBlock2d
from src.model.fno_layer import SpectralConv2d
from typing import Union, List


# this layer comes from https://arxiv.org/pdf/2208.05424
class SoftmaxConstraint(nn.Module):
    def __init__(self, exp_factor=1):
        super(SoftmaxConstraint, self).__init__()

        self.exp_factor = exp_factor

    def forward(self, x, y, upsample_factor):
        # x: (n_batch, in_channel, size_x, size_y)
        # y: (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y)

        y = torch.exp(y * self.exp_factor)
        # (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y)
        avg_y = F.avg_pool2d(y, kernel_size=upsample_factor)
        # (n_batch, in_channel, size_x, size_y)
        out = y * torch.kron(
            x / avg_y, torch.ones((upsample_factor, upsample_factor)).cuda()
        )
        # (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y)

        return out


class FNOBlock(nn.Module):
    """
    Spectral convolution + MLP of the input
    from the paper: fourier neural operator for parametric partial differential equations
    https://openreview.net/pdf?id=c8P9NQVtmnO
    modification from:
    https://arxiv.org/abs/2111.13802
    (shared weights added by myself lol)
    """

    def __init__(
        self,
        n_channels: int,
        modes: Union[int, List[int]],
        shifting_modes: int,
    ):
        super(FNOBlock, self).__init__()

        self.n_channels = n_channels
        self.modes = modes
        self.shifting_modes = shifting_modes

        self.conv = SpectralConv2d(
            self.n_channels,
            self.n_channels,
            self.modes,
            self.modes,
            self.shifting_modes,
        )

        # basically an mlp
        self.w = nn.Conv2d(self.n_channels, self.n_channels, 1)

    def forward(self, x):
        x1 = self.conv(x, 1)
        x2 = self.w(x)
        x3 = x1 + x2

        return F.relu(x3)


class FNO_2(nn.Module):
    def __init__(
        self,
        in_channel=1,
        n_channels=64,
        n_residual_blocks=4,
        n_operator_blocks=2,
        modes=18,  # number of modes to cut in the fno
        shifting_modes: int = 0,  # shifting of the modes
        apply_constraint: bool = False,
        last_layer_kernel: int = 3,
    ):
        super(FNO_2, self).__init__()

        self.in_channel = in_channel
        self.n_channels = n_channels
        self.n_residual_blocks = n_residual_blocks
        self.n_operator_blocks = n_operator_blocks
        self.modes = modes
        self.shifting_modes = shifting_modes
        self.apply_constraint = apply_constraint
        self.last_layer_kernel = last_layer_kernel
        # First Conv Layer (lift the data onto the n_channel space)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.in_channel, self.n_channels, kernel_size=3, stride=1, padding=1
            )
        )

        # Residual Blocks (look for the features)
        self.res_blocks = nn.ModuleList()
        for _ in range(self.n_residual_blocks):
            self.res_blocks.append(ResBlock2d(self.n_channels))

        # apply the fno blocks to look for the edges and more delicated parts (still not sure about this nevertheless)
        # FNO Blocks
        self.fno_blocks = nn.ModuleList()
        for _ in range(self.n_operator_blocks):
            self.fno_blocks.append(
                FNOBlock(
                    self.n_channels,
                    self.modes,
                    self.shifting_modes,
                )
            )

        # Projecting back onto the original space
        self.tail = nn.Sequential(
            nn.Conv2d(
                self.n_channels, self.in_channel, self.last_layer_kernel, padding="same"
            )
        )
        # Constraint Layer
        if self.apply_constraint:
            self.constraint = SoftmaxConstraint()

        self.tail_relu = nn.ReLU()

    def forward(self, x, upsample_factor):
        # x: (n_batch, n_channels, x, y)

        # residual convolutional neural network
        x1 = self.conv1(x)
        x2 = x1
        for layer in self.res_blocks:
            x2 = layer(x2)
        # (n_batch, n_channels, x, y)
        out = x1 + x2
        # interpolation
        out = torch.nn.functional.interpolate(
            out, scale_factor=upsample_factor, mode="bilinear", align_corners=False
        )
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y)

        # fourier neural operators
        for layer in self.fno_blocks:
            out = layer(out)
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y)

        out = self.tail(out)

        if self.apply_constraint:
            out = self.constraint(x, out, upsample_factor)

        out[:, 0] = torch.exp(out[:, 0])
        out[:, 3] = torch.exp(out[:, 3])

        return out
