"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""

import torch
import torch.nn as nn
from typing import Union, List


class FactorizedSpectralConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, List[int]],
        shifting_modes: int = 0,
        share_weights: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = (
            modes if isinstance(modes, (list, tuple)) else [modes, modes, modes]
        )
        self.shifting_modes = shifting_modes
        self.share_weights = share_weights

        if self.share_weights:
            self.weights = nn.Parameter(
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes[0],  # same for all
                    dtype=torch.cfloat,
                )
            )
        else:
            self.weights = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(
                            self.in_channels,
                            self.out_channels,
                            self.modes[i],
                            dtype=torch.cfloat,
                        )
                    )
                    for i in range(3)
                ]
            )

    def forward(self, x, upsample_factor):
        batch_size, _, d1, d2, d3 = x.shape
        dims = [d1, d2, d3]
        up_dims = [int(upsample_factor * d) for d in dims]

        axes = [-3, -2, -1]
        einsum_strs = ["bixyz,iox->boxyz", "bixyz,ioy->boxyz", "bixyz,ioz->boxyz"]
        results = []

        for i, axis in enumerate(axes):
            # Forward FFT along axis
            x_ft = torch.fft.rfft(x, dim=axis, norm="ortho")

            # Output shape in Fourier domain
            out_shape = [batch_size, self.out_channels, *up_dims]
            out_shape[axis] = up_dims[i] // 2 + 1
            out_ft = torch.zeros(
                *out_shape,
                dtype=torch.cfloat,
                device=x.device,
            )

            # Select correct weight
            weight = self.weights if self.share_weights else self.weights[i]

            # Fill Fourier output via einsum
            slices = [slice(None)] * 5
            slices[2 + i] = slice(
                self.shifting_modes, self.modes[i] + self.shifting_modes
            )
            out_ft[tuple(slices)] = torch.einsum(
                einsum_strs[i],
                x_ft[tuple(slices)],
                weight,
            )

            # Inverse FFT along axis
            x_rec = torch.fft.irfft(out_ft, n=up_dims[i], dim=axis, norm="ortho")
            results.append(x_rec)

        # Combine outputs from all axes
        return sum(results)
