import torch
import torch.nn as nn


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
        # weights1: (+kx, +ky)
        # weights2: (-kx, +ky)
        # weights3: (+kx, -ky)
        # weights4: (-kx, -ky)
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
        self.shifting_modes = max(0, shifting_modes)

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (n_batch, in_channels, x, y), (in_channels, out_channels, x, y) -> (n_batch, out_channels, x, y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, upsample_factor=1):
        # x: input function (n_batch, in_channels, n_dim1, n_dim2)
        # upsample_factor: upsample factor for n_dim1 and n_dim2
        # dim1: scaled n_dim1 i.e. n_dim1 * upsample_factor
        # dim2: scaled n_dim2 i.e. n_dim2 * upsample_factor
        # dim3: scaled n_dim3
        # assert self.modes1 < dim1 // 2 and self.modes2 < dim2 // 2
        assert self.modes1 + self.shifting_modes <= x.shape[-3] // 2, (
            "Too many modes + shift"
        )

        batch_size = x.shape[0]
        dim1 = int(x.shape[2])
        dim2 = int(x.shape[3])
        dim3 = int(x.shape[4])

        # Compute Fourier coeffcients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # (n_batch, in_channels, n_dim1, n_dim2//2 + 1)
        m1 = min(self.modes1, x_ft.shape[-3] // 2)
        m2 = min(self.modes2, x_ft.shape[-2] // 2)
        m3 = min(self.modes3, x_ft.shape[-1])

        # Define the slices clearly
        kx_pos = slice(self.shifting_modes, m1 + self.shifting_modes)
        ky_pos = slice(self.shifting_modes, m2 + self.shifting_modes)
        kz_pos = slice(self.shifting_modes, m3 + self.shifting_modes)

        kx_neg = slice(
            -m1 - self.shifting_modes,
            -self.shifting_modes if self.shifting_modes > 0 else None,
        )
        ky_neg = slice(
            -m2 - self.shifting_modes,
            -self.shifting_modes if self.shifting_modes > 0 else None,
        )

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            dim1,
            dim2,
            dim3 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Quadrant 1: (+kx, +ky)
        out_ft[..., kx_pos, ky_pos, kz_pos] = self.compl_mul3d(
            x_ft[..., kx_pos, ky_pos, kz_pos], self.weights1[..., :m1, :m2, :m3]
        )

        # Quadrant 2: (-kx, +ky)
        out_ft[..., kx_neg, ky_pos, kz_pos] = self.compl_mul3d(
            x_ft[..., kx_neg, ky_pos, kz_pos], self.weights2[..., :m1, :m2, :m3]
        )

        # Quadrant 3: (+kx, -ky)
        out_ft[..., kx_pos, ky_neg, kz_pos] = self.compl_mul3d(
            x_ft[..., kx_pos, ky_neg, kz_pos], self.weights3[..., :m1, :m2, :m3]
        )

        # Quadrant 4: (-kx, -ky)
        out_ft[..., kx_neg, ky_neg, kz_pos] = self.compl_mul3d(
            x_ft[..., kx_neg, ky_neg, kz_pos], self.weights4[..., :m1, :m2, :m3]
        )

        # Return to physical space
        x = torch.fft.irfftn(
            out_ft, s=(dim1, dim2, dim3), dim=[-3, -2, -1]
        )  # * (upsample_factor**3)

        # Shape: (batch, out_channels, dim1, dim2, dim3)
        return x


class SpectralConv2d(nn.Module):
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
        shifting_modes: int,
        use_scaling: bool = False,
    ):
        super(SpectralConv2d, self).__init__()

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
        self.scale = (
            (1 / (2 * in_channels)) ** (1.0 / 2.0) if use_scaling else 1.0
        )  # initialization scale
        self.weights1 = nn.Parameter(
            self.scale
            * (
                torch.randn(
                    self.in_channels,
                    self.out_channels,
                    self.modes1,
                    self.modes2,
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
                    dtype=torch.cfloat,
                )
            )
        )
        self.shifting_modes = shifting_modes

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (n_batch, in_channels, x, y), (in_channels, out_channels, x, y) -> (n_batch, out_channels, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

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

        # Compute Fourier coeffcients
        x_ft = torch.fft.rfft2(x)
        # (n_batch, in_channels, n_dim1, n_dim2//2 + 1)

        shifting_modes = int(min(0, self.shifting_modes))
        modes1_use = int(min(self.modes1 + shifting_modes, dim1 // 2, x.shape[2] // 2))
        modes2_use = int(min(self.modes2 + shifting_modes, dim2 // 2, x.shape[3] // 2))

        # Initialize output in frequency domain
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            dim1,
            dim2 // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Region 1: low-low
        out_ft[
            :,
            :,
            shifting_modes:modes1_use,
            shifting_modes:modes2_use,
        ] = self.compl_mul2d(
            x_ft[
                :,
                :,
                shifting_modes:modes1_use,
                shifting_modes:modes2_use,
            ],
            self.weights1[
                :,
                :,
                shifting_modes:modes1_use,
                shifting_modes:modes2_use,
            ],
        )

        # Region 2: high-low frequencies
        out_ft[
            :,
            :,
            -modes1_use:-shifting_modes,
            shifting_modes:modes2_use,
        ] = self.compl_mul2d(
            x_ft[
                :,
                :,
                -modes1_use:-shifting_modes,
                shifting_modes:modes2_use,
            ],
            self.weights2[
                :,
                :,
                -modes1_use:-shifting_modes,
                shifting_modes:modes2_use,
            ],
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(dim1, dim2))
        # Shape: (batch, out_channels, dim1, dim2, dim3)
        return x
