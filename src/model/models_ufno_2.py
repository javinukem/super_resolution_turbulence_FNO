import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.models_edsr import ResBlock
from src.model.fno_layer import SpectralConv3d
from typing import Union, List


# Interpolation modes that accept the ``align_corners`` argument.
_ALIGN_OK = {"linear", "bilinear", "bicubic", "trilinear"}


def _upsample_3d(x: torch.Tensor, scale_factor: int, mode: str) -> torch.Tensor:
    """3-D upsample that omits ``align_corners`` for modes that reject it.

    ``nearest`` / ``area`` / ``nearest-exact`` raise if ``align_corners`` is
    passed, so this helper mirrors the per-mode branch used by the grid
    experiments and keeps the ``trilinear`` default bit-identical.
    """
    if mode in _ALIGN_OK:
        return F.interpolate(
            x, scale_factor=scale_factor, mode=mode, align_corners=False
        )
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


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


# PyTorch port of the U-Net path from
# https://github.com/RuneRost/RT-JAX/blob/main/architectures/ufno_3d.py
# Runs in parallel with the spectral conv and the 1x1 bypass conv inside each
# UFNO block. The encoder downsamples by a factor of 8 (three stride-2 convs);
# the input spatial size therefore must be divisible by 8.
class Conv3dBlock(nn.Module):
    def __init__(self, channels: int, stride: int, dropout_rate: float):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(
            num_groups=1, num_channels=channels, eps=1e-6, affine=True
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        return x


class Deconv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True
        )

    def forward(self, x):
        x = self.deconv(x)
        return F.leaky_relu(x, negative_slope=0.1)


class UNet3d(nn.Module):
    """3-D U-Net used as the parallel path inside a UFNO block.

    Mirrors the structure of ``U_net`` in the RT-JAX reference: three
    stride-2 downsampling convs (each followed by a stride-1 additive conv),
    mirrored transposed-conv upsampling with skip-concatenation, and a final
    3x3 conv that projects the concatenated input back to ``channels``.
    """

    def __init__(self, channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = Conv3dBlock(channels, stride=2, dropout_rate=dropout_rate)
        self.conv2 = Conv3dBlock(channels, stride=2, dropout_rate=dropout_rate)
        self.conv2_add = Conv3dBlock(channels, stride=1, dropout_rate=dropout_rate)
        self.conv3 = Conv3dBlock(channels, stride=2, dropout_rate=dropout_rate)
        self.conv3_add = Conv3dBlock(channels, stride=1, dropout_rate=dropout_rate)

        self.deconv2 = Deconv3dBlock(channels, channels)
        self.deconv1 = Deconv3dBlock(channels * 2, channels)
        self.deconv0 = Deconv3dBlock(channels * 2, channels)

        self.output_layer = nn.Conv3d(
            channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_add(self.conv2(out_conv1))
        out_conv3 = self.conv3_add(self.conv3(out_conv2))

        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat([out_conv2, out_deconv2], dim=1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat([out_conv1, out_deconv1], dim=1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat([x, out_deconv0], dim=1)
        return self.output_layer(concat0)


class UFNOBlock(nn.Module):
    """
    Spectral convolution + 1x1 MLP + U-Net path.
    from the paper: fourier neural operator for parametric partial differential
    equations
    https://openreview.net/pdf?id=c8P9NQVtmnO
    U-Net augmentation from:
    https://github.com/gegewen/ufno/ (and the RT-JAX port
    https://github.com/RuneRost/RT-JAX/blob/main/architectures/ufno_3d.py)
    """

    def __init__(
        self,
        n_channels: int,
        modes: Union[int, List[int]],
        shifting_modes: int,
        dropout_rate: float = 0.0,
    ):
        super(UFNOBlock, self).__init__()

        self.n_channels = n_channels
        self.modes = modes
        self.shifting_modes = shifting_modes

        self.conv = SpectralConv3d(
            self.n_channels,
            self.n_channels,
            self.modes,
            self.modes,
            self.modes,
            self.shifting_modes,
        )

        # basically an mlp
        self.w = nn.Conv3d(self.n_channels, self.n_channels, 1)

        # u-net path (the UFNO addition)
        self.unet = UNet3d(self.n_channels, dropout_rate=dropout_rate)

    def forward(self, x):
        x1 = self.conv(x, 1)
        x2 = self.w(x)
        x3 = self.unet(x)
        x3 = x1 + x2 + x3

        return F.relu(x3)


class UFNO_2(nn.Module):
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
        interpolation_mode: str = "trilinear",
        skip_connection: bool = False,
        skip_connection_interpolation_mode: str = "nearest",
        apply_positivity_relu: bool = True,
        dropout_rate: float = 0.0,
    ):
        super(UFNO_2, self).__init__()

        self.in_channel = in_channel
        self.n_channels = n_channels
        self.n_residual_blocks = n_residual_blocks
        self.n_operator_blocks = n_operator_blocks
        self.modes = modes
        self.shifting_modes = shifting_modes
        self.apply_constraint = apply_constraint
        self.last_layer_kernel = last_layer_kernel
        self.interpolation_mode = interpolation_mode
        self.skip_connection = skip_connection
        self.apply_positivity_relu = apply_positivity_relu
        self.skip_connection_interpolation_mode = skip_connection_interpolation_mode
        self.dropout_rate = dropout_rate
        # First Conv Layer (lift the data onto the n_channel space)
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                self.in_channel, self.n_channels, kernel_size=3, stride=1, padding=1
            )
        )

        # Residual Blocks (look for the features)
        self.res_blocks = nn.ModuleList()
        for _ in range(self.n_residual_blocks):
            self.res_blocks.append(ResBlock(self.n_channels))

        # UFNO Blocks: spectral conv + 1x1 MLP + U-Net path
        self.fno_blocks = nn.ModuleList()
        for _ in range(self.n_operator_blocks):
            self.fno_blocks.append(
                UFNOBlock(
                    self.n_channels,
                    self.modes,
                    self.shifting_modes,
                    dropout_rate=self.dropout_rate,
                )
            )

        # Projecting back onto the original space
        self.tail = nn.Sequential(
            nn.Conv3d(
                self.n_channels, self.in_channel, self.last_layer_kernel, padding="same"
            )
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
        latent = x1 + x2  # LR latent, kept for the optional skip connection

        # interpolation (LR -> HR)
        out = _upsample_3d(latent, upsample_factor, self.interpolation_mode)
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y)

        # fourier neural operators (with u-net path)
        for layer in self.fno_blocks:
            out = layer(out)
        # (n_batch, n_channels, upsample_factor * size_x, upsample_factor * size_y, upsample_factor * size_z)

        if self.skip_connection:
            # Upsample the LR latent (NOT the already-HR ``out``) and add it
            # as a residual.  Interpolating ``out`` re-applied ``upsample_factor``
            # to an already-HR tensor and crashed with a shape mismatch.
            skip = _upsample_3d(
                latent, upsample_factor, self.skip_connection_interpolation_mode
            )
            out = out + skip

        out = self.tail(out)

        if self.apply_constraint:
            out = self.constraint(x, out, upsample_factor)

        if self.apply_positivity_relu:
            out[:, 0] = self.tail_relu(out[:, 0])
            out[:, 4] = self.tail_relu(out[:, 4])

        return out
