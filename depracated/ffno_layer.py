import torch
import torch.nn as nn


class FactorizedSpectralConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        shifting_modes,
        share_weights: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = (
            modes if isinstance(modes, (list, tuple)) else [modes, modes, modes]
        )
        self.share_weights = share_weights
        if share_weights:
            self.weights = nn.Parameter(
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes[0],
                    dtype=torch.cfloat,
                )
            )
        else:
            self.weights1 = nn.Parameter(
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes[0],
                    dtype=torch.cfloat,
                )
            )
            self.weights2 = nn.Parameter(
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes[1],
                    dtype=torch.cfloat,
                )
            )

            self.weights3 = nn.Parameter(
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes[2],
                    dtype=torch.cfloat,
                )
            )

        self.shifting_modes = shifting_modes
        self.modes += self.shifting_modes

    def forward(self, x, upsample_factor):
        batch_size = x.shape[0]
        dim1 = int(upsample_factor * x.shape[2])
        dim2 = int(upsample_factor * x.shape[3])
        dim3 = int(upsample_factor * x.shape[4])

        # # # Dimension Z # # #
        x_ftz = torch.fft.rfft(x, dim=-1, norm="ortho")
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size, grid_size // 2 + 1]

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            dim1,
            dim2,
            dim3 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        # out_ft.shape == [batch_size, out_dim, grid_size, grid_size, grid_size // 2 + 1]
        if self.share_weights:
            out_ft[:, :, :, :, self.shifting_modes : self.modes[2]] = torch.einsum(
                "bixyz,ioz->boxyz",
                x_ftz[:, :, :, :, self.shifting_modes : self.modes[2]],
                self.weights,
            )
        else:
            out_ft[:, :, :, :, self.shifting_modes : self.modes[2]] = torch.einsum(
                "bixyz,ioz->boxyz",
                x_ftz[:, :, :, :, self.shifting_modes : self.modes[2]],
                self.weights3,
            )

        xz = torch.fft.irfft(out_ft, n=dim3, dim=-1, norm="ortho")
        # x.shape == [batch_size, in_dim, grid_size, grid_size, grid_size]

        # # # Dimension Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm="ortho")
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size, grid_size // 2 + 1]

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            dim1,
            dim2 // 2 + 1,
            dim3,
            dtype=torch.cfloat,
            device=x.device,
        )
        # out_ft.shape == [batch_size, out_dim, grid_size, grid_size, grid_size // 2 + 1]
        if self.share_weights:
            out_ft[:, :, :, self.shifting_modes : self.modes[1], :] = torch.einsum(
                "bixyz,ioy->boxyz",
                x_fty[:, :, :, self.shifting_modes : self.modes[1], :],
                self.weights,
            )
        else:
            out_ft[:, :, :, self.shifting_modes : self.modes[1], :] = torch.einsum(
                "bixyz,ioy->boxyz",
                x_fty[:, :, :, self.shifting_modes : self.modes[1], :],
                self.weights2,
            )

        xy = torch.fft.irfft(out_ft, n=dim2, dim=-2, norm="ortho")
        # x.shape == [batch_size, in_dim, grid_size, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm="ortho")
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size, grid_size // 2 + 1]

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            dim1 // 2 + 1,
            dim2,
            dim3,
            dtype=torch.cfloat,
            device=x.device,
        )
        # out_ft.shape == [batch_size, out_dim, grid_size, grid_size, grid_size // 2 + 1]
        if self.share_weights:
            out_ft[:, :, self.shifting_modes : self.modes[0], :, :] = torch.einsum(
                "bixyz,iox->boxyz",
                x_ftx[:, :, self.shifting_modes : self.modes[0], :, :],
                self.weights,
            )
        else:
            out_ft[:, :, self.shifting_modes : self.modes[0], :, :] = torch.einsum(
                "bixyz,iox->boxyz",
                x_ftx[:, :, self.shifting_modes : self.modes[0], :, :],
                self.weights1,
            )

        xx = torch.fft.irfft(out_ft, n=dim1, dim=-3, norm="ortho")
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy + xz

        return x
