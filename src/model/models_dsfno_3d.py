# This file contains network modules to build downscaling FNO model.
# Author: Qidong Yang
# Date: 2022-08-26
# git: https://github.com/qy707/DSFNO
# paper: "Fourier Neural Operators for Arbitrary Resolution Climate Data Downscaling".
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

##### Constraint Layer #####


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
        shifting_modes: int = 0,
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

        # Region 1: low-low-low frequencies
        out_ft[
            :,
            :,
            shifting_modes:modes1_use,
            shifting_modes:modes2_use,
            shifting_modes:modes3_use,
        ] = self.compl_mul3d(
            x_ft[
                :,
                :,
                shifting_modes:modes1_use,
                shifting_modes:modes2_use,
                shifting_modes:modes3_use,
            ],
            self.weights1[
                :,
                :,
                shifting_modes:modes1_use,
                shifting_modes:modes2_use,
                shifting_modes:modes3_use,
            ],
        )

        # Region 2: high-low-low frequencies
        out_ft[
            :,
            :,
            -modes1_use:-shifting_modes,
            shifting_modes:modes2_use,
            shifting_modes:modes3_use,
        ] = self.compl_mul3d(
            x_ft[
                :,
                :,
                -modes1_use:-shifting_modes,
                shifting_modes:modes2_use,
                shifting_modes:modes3_use,
            ],
            self.weights2[
                :,
                :,
                -modes1_use:-shifting_modes,
                shifting_modes:modes2_use,
                shifting_modes:modes3_use,
            ],
        )

        # Region 3: low-high-low frequencies
        out_ft[
            :,
            :,
            shifting_modes:modes1_use,
            -modes2_use:-shifting_modes,
            shifting_modes:modes3_use,
        ] = self.compl_mul3d(
            x_ft[
                :,
                :,
                shifting_modes:modes1_use,
                -modes2_use:-shifting_modes,
                shifting_modes:modes3_use,
            ],
            self.weights3[
                :,
                :,
                shifting_modes:modes1_use,
                -modes2_use:-shifting_modes,
                shifting_modes:modes3_use,
            ],
        )

        # Region 4: high-high-low frequencies
        out_ft[
            :,
            :,
            -modes1_use:-shifting_modes,
            -modes2_use:-shifting_modes,
            shifting_modes:modes3_use,
        ] = self.compl_mul3d(
            x_ft[
                :,
                :,
                -modes1_use:-shifting_modes,
                -modes2_use:-shifting_modes,
                shifting_modes:modes3_use,
            ],
            self.weights4[
                :,
                :,
                -modes1_use:-shifting_modes,
                -modes2_use:-shifting_modes,
                shifting_modes:modes3_use,
            ],
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
        activation: bool = True,
    ):
        super(OperatorBlock, self).__init__()

        self.n_channels = n_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.activation = activation

        self.conv = SpectralConv3d(
            self.n_channels, self.n_channels, self.modes1, self.modes2, self.modes3
        )

        # basically an mlp
        self.w = nn.Conv3d(self.n_channels, self.n_channels, 1)

    def forward(self, x):
        x1 = self.conv(x, 1)
        x2 = self.w(x)
        x = x1 + x2

        if self.activation:
            x = F.gelu(x)

        return x


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
            bias=False,
        )
        self.conv2 = nn.Conv3d(
            self.n_channels,
            self.n_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
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


class DSFNO(nn.Module):
    def __init__(
        self,
        in_channel=1,
        n_channels=64,
        n_residual_blocks=4,
        n_operator_blocks=2,
        modes=18,
        apply_constraint=False,
    ):
        super(DSFNO, self).__init__()

        self.in_channel = in_channel
        self.n_channels = n_channels
        self.n_residual_blocks = n_residual_blocks
        self.n_operator_blocks = n_operator_blocks
        self.modes = modes
        self.apply_constraint = apply_constraint

        # First Conv Layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                self.in_channel, self.n_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
        )

        # Residual Blocks
        self.res_blocks = nn.ModuleList()
        for i in range(self.n_residual_blocks):
            self.res_blocks.append(ResidualBlock(self.n_channels))

        # Second Conv Layer
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
        )

        # FNO Blocks
        self.fno_blocks = nn.ModuleList()
        for i in range(self.n_operator_blocks - 1):
            self.fno_blocks.append(
                OperatorBlock(self.n_channels, self.modes, self.modes, True)
            )
        self.fno_blocks.append(
            OperatorBlock(self.n_channels, self.modes, self.modes, False)
        )

        # Channel Linear Layers
        self.fc1 = nn.Linear(self.n_channels, 128)
        self.fc2 = nn.Linear(128, self.in_channel)

        # Constraint Layer
        if self.apply_constraint:
            self.constraint = SoftmaxConstraint()

    def checkpointed_layers(self, x):
        out = self.fc1(x)
        out = F.gelu(out)
        out = self.fc2(out)
        return out

    def forward(self, x, upsample_factor):
        # x: (n_batch, n_channels, x, y, size_z)

        # residual convolutional neural network
        out = self.conv1(x)
        # (n_batch, n_channels, x, y, z)

        for layer in self.res_blocks:
            out = layer(out)
        # (n_batch, n_channels, x, y, size_z)

        out = self.conv2(out)
        # (n_batch, n_channels, x, y, size_z)

        # interpolation
        out = torch.nn.functional.interpolate(
            out, scale_factor=upsample_factor, mode="trilinear", align_corners=False
        )
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y)

        # fourier neural operators
        for layer in self.fno_blocks:
            out = layer(out)
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y, upsample_factor * size_z)

        out = out.permute(0, 2, 3, 4, 1)

        out = checkpoint.checkpoint(self.checkpointed_layers, out, use_reentrant=False)

        out = out.permute(0, 4, 1, 2, 3)

        if self.apply_constraint:
            out = self.constraint(x, out, upsample_factor)

        return out
