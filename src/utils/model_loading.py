"""
Model Builder
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Model construction
# =====================================================================


class _TrilinearBaseline(nn.Module):
    """Trilinear (3-D bicubic analog) upsampling baseline."""

    def forward(self, x: torch.Tensor, upsample_factor: int = 4) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=upsample_factor,
            mode="trilinear",
            align_corners=False,
        )


def _instantiate(entry: dict) -> nn.Module:
    mtype = entry["model_type"]
    params = dict(entry.get("model_params") or {})
    apply_relu = entry.get("apply_positivity_relu", True)
    if mtype in ("sfno", "cfno"):
        from src.model.models_fno_2 import FNO_2

        return FNO_2(**params, apply_positivity_relu=apply_relu)
    if mtype == "edsr":
        from src.model.models_edsr import EDSR

        return EDSR(**params, apply_positivity_relu=apply_relu)
    if mtype in ("usfno", "ufno"):
        from src.model.models_ufno_2 import UFNO_2

        return UFNO_2(**params, apply_positivity_relu=apply_relu)
    if mtype == "trilinear":
        return _TrilinearBaseline()
    raise ValueError(f"Unknown model_type: {mtype}")


def build_model(
    entry: dict,
    device: torch.device,
    norm_stats: dict | None = None,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """
    Build a model from a manifest entry and return a uniform run callable.

    The returned callable has signature run(lr, upsample_factor) -> sr and
    handles: optional LR normalization, the cfno-vs-edsr forward branch
    (cfno/trilinear take upsample_factor, edsr does not), and optional SR
    denormalization. The underlying nn.Module is accessible via
    run.model.
    """
    model = _instantiate(entry).to(device)
    weights = entry.get("weights")
    if weights is not None:
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True)
        )
    model.eval()

    use_norm = entry.get("use_norm", False)
    supports_variable_scale = entry.get("supports_variable_scale", False)

    def run(lr: torch.Tensor, upsample_factor: int = 4) -> torch.Tensor:
        with torch.no_grad():
            if use_norm and norm_stats is not None:
                ml = norm_stats["mean_lr"][None, :, None, None, None]
                sl = norm_stats["std_lr"][None, :, None, None, None]
                mh = norm_stats["mean_hr"][None, :, None, None, None]
                sh = norm_stats["std_hr"][None, :, None, None, None]
                lr_n = (lr - ml) / sl
                if supports_variable_scale:
                    sr_n = model(lr_n, upsample_factor=upsample_factor)
                else:
                    sr_n = model(lr_n)
                return sr_n * sh + mh
            if supports_variable_scale:
                return model(lr, upsample_factor=upsample_factor)
            return model(lr)

    run.model = model
    return run
