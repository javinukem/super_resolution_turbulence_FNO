# This file contains network modules to build downscaling FNO model.
# Author: Qidong Yang
# Date: 2022-08-26
# git: https://github.com/qy707/DSFNO
# paper: "Fourier Neural Operators for Arbitrary Resolution Climate Data Downscaling".
import torch
import torch.nn as nn
import torch.nn.functional as F


# this layer comes from https://arxiv.org/pdf/2208.05424
class SoftmaxConstraint(nn.Module):
    def __init__(self):
        super(SoftmaxConstraint, self).__init__()

    def forward(self, lr, sr, upsample_factor):
        # x: (n_batch, in_channel, size_x, size_y, size_z)
        # y: (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y, upsample_factor * size_z)

        avg_sr = F.avg_pool3d(sr, kernel_size=upsample_factor)
        # (n_batch, in_channel, size_x, size_y, size_z)

        tile = torch.ones(
            (upsample_factor, upsample_factor, upsample_factor), device=sr.device
        )
        ratio = avg_sr / (lr.float().clamp(min=1e-3))
        scale_upsampled = torch.kron(ratio, tile)

        out = sr * scale_upsampled
        # (n_batch, in_channel, upsample_factor * size_x, upsample_factor * size_y, upsample_factor * size_z)

        return out


##### Model Modules #####


class SpectralConv3d(nn.Module):
    """
    3D implementation of a spectral convolution,
    does a fft transform, convolves in that space and then brings it back,
    keep in mind the limitation with the number of modes,
    by ignoring high number modes we intent to get rid of the noise
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        shifting_modes: int,
    ):
        super(SpectralConv3d, self).__init__()

        # 3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        # in_channels: the number of input channels
        # out_channels: the number of output channels
        # modes1: the number of modes used for dimension 1, at most floor(N/2) + 1
        # modes2: the number of modes used for dimension 2, at most floor(N/2) + 1
        # shifting_modes: how many k to shift the modes from 0
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (2 * in_channels)) ** (1.0 / 2.0)  # initialization scale
        self.weights1 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                    dtype=torch.cfloat,
                )
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                    dtype=torch.cfloat,
                )
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                    dtype=torch.cfloat,
                )
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                    dtype=torch.cfloat,
                )
            )
        )
        self.shifting_modes = shifting_modes

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (n_batch, in_channels, x, y), (in_channels, out_channels, x, y) -> (n_batch, out_channels, x, y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, upsample_factor):
        # x: input function (n_batch, in_channels, n_dim1, n_dim2)
        # upsample_factor: upsample factor for n_dim1 and n_dim2
        # dim1: scaled n_dim1 i.e. n_dim1 * upsample_factor
        # dim2: scaled n_dim2 i.e. n_dim2 * upsample_factor
        # dim3: scaled n_dim3
        # assert self.modes1 < dim1 // 2 and self.modes2 < dim2 // 2

        batch_size = x.shape[0]
        dim1 = int(upsample_factor * x.shape[2])
        dim2 = int(upsample_factor * x.shape[3])
        dim3 = int(upsample_factor * x.shape[4])

        # Compute Fourier coeffcients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # (n_batch, in_channels, n_dim1, n_dim2//2 + 1)

        shifting_modes = int(min(0, self.shifting_modes))
        modes1_use = int(min(self.modes1, dim1 // 2, x.shape[2] // 2)) + shifting_modes
        modes2_use = int(min(self.modes2, dim2 // 2, x.shape[3] // 2)) + shifting_modes
        modes3_use = (
            int(min(self.modes3, dim3 // 2 + 1, x.shape[4] // 2 + 1)) + shifting_modes
        )

        # Initialize output in frequency domain
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            dim1,
            dim2,
            dim3 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        region_1 = (
            slice(None),
            slice(None),
            slice(shifting_modes, modes1_use),
            slice(shifting_modes, modes2_use),
            slice(shifting_modes, modes3_use),
        )
        region_2 = (
            slice(None),
            slice(None),
            slice(-modes1_use, -shifting_modes),
            slice(shifting_modes, modes2_use),
            slice(shifting_modes, modes3_use),
        )
        region_3 = (
            slice(None),
            slice(None),
            slice(shifting_modes, modes1_use),
            slice(-modes2_use, -shifting_modes),
            slice(shifting_modes, modes3_use),
        )
        region_4 = (
            slice(None),
            slice(None),
            slice(-modes1_use, -shifting_modes),
            slice(-modes2_use, -shifting_modes),
            slice(shifting_modes, modes3_use),
        )

        # Region 1: low-low-low frequencies
        out_ft[region_1] = self.compl_mul3d(
            x_ft[region_1],
            self.weights1[region_1],
        )

        # Region 2: high-low-low frequencies
        out_ft[region_2] = self.compl_mul3d(
            x_ft[region_2],
            self.weights2[region_2],
        )

        # Region 3: low-high-low frequencies
        out_ft[region_3] = self.compl_mul3d(
            x_ft[region_3],
            self.weights3[region_3],
        )

        # Region 4: high-high-low frequencies
        out_ft[region_4] = self.compl_mul3d(
            x_ft[region_4],
            self.weights4[region_4],
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(dim1, dim2, dim3), dim=[-3, -2, -1]) * (
            upsample_factor**3
        )

        # Shape: (batch, out_channels, dim1, dim2, dim3)
        return x


class OperatorBlock(nn.Module):
    """
    Spectral convolution + MLP of the input
    from the paper: fourier neural operator for parametric partial differential equations
    https://openreview.net/pdf?id=c8P9NQVtmnO
    """

    def __init__(
        self,
        n_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        shifting_modes: int,
    ):
        super(OperatorBlock, self).__init__()

        self.n_channels = n_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.shifting_modes = shifting_modes

        self.conv = SpectralConv3d(
            self.n_channels,
            self.n_channels,
            self.modes1,
            self.modes2,
            self.modes3,
            self.shifting_modes,
        )

        # basically an mlp
        self.w1 = nn.Conv3d(self.n_channels, self.n_channels, 1)
        self.w2 = nn.Conv3d(self.n_channels, self.n_channels, 1)

    def forward(self, x):
        x1 = self.conv(x, 1)
        x1 = self.w1(x1)
        x1 = F.relu(x1)
        x1 = self.w2(x1)
        x1 = F.relu(x1)
        x2 = x1 + x

        return x2


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()

        self.n_channels = n_channels

        self.conv1 = nn.Conv3d(
            self.n_channels,
            self.n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv3d(
            self.n_channels,
            self.n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + residual

        return out


##### Downscaling Model #####


class FNO_1(nn.Module):
    def __init__(
        self,
        in_channel=1,
        n_channels=64,
        n_residual_blocks=4,
        n_operator_blocks=2,
        modes=18,
        shifting_modes: int = 0,
        apply_constraint=False,
    ):
        super(FNO_1, self).__init__()

        self.in_channel = in_channel
        self.n_channels = n_channels
        self.n_residual_blocks = n_residual_blocks
        self.n_operator_blocks = n_operator_blocks
        self.modes = modes
        self.shifting_modes = shifting_modes
        self.apply_constraint = apply_constraint

        # First Conv Layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                self.in_channel, self.n_channels, kernel_size=3, stride=1, padding=1
            )
        )

        # Residual Blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(self.n_residual_blocks):
            self.res_blocks.append(ResidualBlock(self.n_channels))

        # FNO Blocks
        self.fno_blocks = nn.ModuleList()
        for _ in range(self.n_operator_blocks):
            self.fno_blocks.append(
                OperatorBlock(
                    self.n_channels, self.modes, self.modes, self.shifting_modes, True
                )
            )

        self.tail = nn.Sequential(
            nn.Conv3d(self.n_channels, self.in_channel, 3, padding="same")
        )
        # Constraint Layer
        if self.apply_constraint:
            self.constraint = SoftmaxConstraint()

        self.tail_relu = nn.ReLU()

    def forward(self, x, upsample_factor):
        # x: (n_batch, n_channels, x, y, size_z)

        # residual convolutional neural network
        x1 = self.conv1(x)
        x2 = x1
        for layer in self.res_blocks:
            x2 = layer(x2)
        # (n_batch, n_channels, x, y, size_z)
        out = x1 + x2
        # interpolation
        out = torch.nn.functional.interpolate(
            out, scale_factor=upsample_factor, mode="trilinear", align_corners=False
        )
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y)

        # fourier neural operators
        for layer in self.fno_blocks:
            out = layer(out)
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y, upsample_factor * size_z)

        out = self.tail(out)

        if self.apply_constraint:
            out = self.constraint(x, out, upsample_factor)

        out[:, 0] = self.tail_relu(out[:, 0])
        out[:, 4] = self.tail_relu(out[:, 4])

        return out
