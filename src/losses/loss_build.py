import torch.nn as nn
import torch
from src.losses.spectral_mse import SpectralLoss


class GeneralLoss(nn.Module):
    """Weighted combination of MSE, velocity-spectral and L1 losses."""

    def __init__(
        self,
        mse_weight: float = 0,
        spectral_weight: float = 0,
        l1_weight: float = 0,
        spectral_v_indices: tuple = (1, 2, 3),
        spectral_pool: int = 2,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.spectral_weight = spectral_weight
        self.l1_weight = l1_weight
        self.spectral_v_indices = spectral_v_indices
        self.spectral_pool = spectral_pool
        if min(mse_weight, spectral_weight, l1_weight) < 0:
            raise ValueError("Negative values for loss weights inputted")
        if mse_weight == 0 and spectral_weight == 0 and l1_weight == 0:
            raise ValueError("All loss weights are zero")
        if mse_weight != 0:
            self.mse = nn.MSELoss()
        if l1_weight != 0:
            self.l1_loss = nn.L1Loss()
        if spectral_weight != 0:
            self.spectral_loss = SpectralLoss(spectral_pool, *spectral_v_indices)

    def loss_name(self):
        names = []
        if self.mse_weight != 0:
            names.append("mse")
        if self.spectral_weight != 0:
            names.append("spectral")
        if self.l1_weight != 0:
            names.append("l1")
        name = "_".join(names)
        return name

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        individual_losses = {}
        total_loss = pred.new_zeros(())
        if self.mse_weight != 0:
            mse_value = self.mse(pred, target)
            total_loss = total_loss + mse_value * self.mse_weight
            individual_losses["mse"] = mse_value.item()
        if self.l1_weight != 0:
            l1_value = self.l1_loss(pred, target)
            total_loss = total_loss + l1_value * self.l1_weight
            individual_losses["l1"] = l1_value.item()
        if self.spectral_weight != 0:
            spectral_value, _ = self.spectral_loss(pred, target)
            total_loss = total_loss + spectral_value * self.spectral_weight
            individual_losses["spectral"] = spectral_value.item()
        individual_losses["total"] = total_loss.item()

        return total_loss, individual_losses
