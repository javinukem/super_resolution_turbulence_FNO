# https://github.com/sanghyun-son/EDSR-PyTorch/tree/master
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Optional

sys.path.append(os.path.abspath(".."))  # This brings 'src' into the path
from src.utils.pixel_shuffle3d import PixelShuffle3d

# Interpolation modes that accept the ``align_corners`` argument (mirrors
# models_fno_2.py so the skip-connection upsample behaves consistently).
_ALIGN_OK = {"linear", "bilinear", "bicubic", "trilinear"}


def _upsample_3d(x, scale_factor, mode):
    if mode in _ALIGN_OK:
        return F.interpolate(
            x, scale_factor=scale_factor, mode=mode, align_corners=False
        )
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


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
        activation_f: Optional[str] = None,
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
        apply_positivity_relu: bool = True,
        skip_connection: bool = False,
        skip_connection_interpolation_mode: str = "trilinear",
    ):
        super().__init__()
        self.apply_positivity_relu = apply_positivity_relu
        self.skip_connection = skip_connection
        self.skip_connection_interpolation_mode = skip_connection_interpolation_mode
        self._skip_scale = scale
        m_head = [nn.Conv3d(input_channels, n_feats, kernel_size, padding="same")]

        # define body module
        m_body = [ResBlock(n_feats, kernel_size) for _ in range(n_resblocks)]
        m_body.append(nn.Conv3d(n_feats, n_feats, kernel_size, padding="same"))

        # define tail module
        m_tail = [
            Upsampler(scale, n_feats, activation_f=activation_f),
            nn.Conv3d(n_feats, input_channels, kernel_size, padding="same"),
        ]
        self.tail_relu = nn.ReLU()
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        if self.skip_connection:
            # Mirror FNO_2: upsample the LR latent (post-body residual, at
            # ``n_feats`` channels) and add it to the HR latent BEFORE the
            # final channel projection. Indexing ``self.tail`` keeps the
            # module names (and thus the saved state_dict keys) unchanged, so
            # existing non-skip weights remain loadable.
            skip = _upsample_3d(
                res, self._skip_scale, self.skip_connection_interpolation_mode
            )
            out = self.tail[0](res) + skip  # Upsampler (n_feats LR -> n_feats HR)
            out = self.tail[1](out)  # Conv3d (n_feats HR -> input_channels HR)
        else:
            out = self.tail(res)
        # ensure pressure and density are positives
        if self.apply_positivity_relu:
            out[:, 0] = self.tail_relu(out[:, 0])
            out[:, 4] = self.tail_relu(out[:, 4])

        return out
