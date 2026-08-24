"""Checkpoint-free spatial payloads and RealWonder RGB preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def coordinate_identity_payload(
    height: int = 60,
    width: int = 104,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return normalized X, Y, and asymmetric identity-pattern channels."""

    y = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pattern = 0.55 * xx + 0.25 * yy.square() + 0.20 * torch.sin(
        2.0 * torch.pi * (1.7 * xx + 0.6 * yy)
    ).mul(0.5).add(0.5)
    return torch.stack([xx, yy, pattern.clamp(0.0, 1.0)])


def load_realwonder_rgb_crop(path: str | Path) -> torch.Tensor:
    """Apply RealWonder's exact 512 -> 832 resize and centered 480 crop."""

    image = Image.open(path).convert("RGB")
    if image.size != (512, 512):
        raise ValueError(f"expected a 512x512 RealWonder frame, got {image.size}")
    image = image.resize((832, 832), resample=Image.Resampling.BILINEAR)
    image = image.crop((0, 176, 832, 656))
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def rgb_crop_to_spatial_grid(
    cropped_rgb: torch.Tensor,
    *,
    height: int = 60,
    width: int = 104,
) -> torch.Tensor:
    """Area-pool a `[3,480,832]` visual proxy to a spatial payload grid."""

    cropped_rgb = torch.as_tensor(cropped_rgb, dtype=torch.float32)
    if cropped_rgb.shape != (3, 480, 832):
        raise ValueError(
            f"cropped_rgb must have shape [3, 480, 832], got {tuple(cropped_rgb.shape)}"
        )
    return F.interpolate(
        cropped_rgb.unsqueeze(0), size=(height, width), mode="area"
    )[0]


def load_realwonder_rgb_grid(
    path: str | Path,
    *,
    height: int = 60,
    width: int = 104,
) -> torch.Tensor:
    return rgb_crop_to_spatial_grid(
        load_realwonder_rgb_crop(path), height=height, width=width
    )


def load_realwonder_rgb_grid_sequence(
    paths: Sequence[str | Path],
    *,
    height: int = 60,
    width: int = 104,
) -> torch.Tensor:
    if not paths:
        raise ValueError("at least one RGB frame path is required")
    return torch.stack(
        [load_realwonder_rgb_grid(path, height=height, width=width) for path in paths]
    )


def point_support_mask(
    target_uv: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Rasterize nearest-cell point support without feature transport."""

    target_uv = torch.as_tensor(target_uv, dtype=torch.float32)
    target_valid = torch.as_tensor(target_valid, dtype=torch.bool)
    if target_uv.ndim != 3 or target_uv.shape[-1] != 2:
        raise ValueError("target_uv must have shape [T, N, 2]")
    if target_valid.shape != target_uv.shape[:2]:
        raise ValueError("target_valid must have shape [T, N]")
    finite = torch.isfinite(target_uv).all(dim=-1)
    in_bounds = (
        finite
        & (target_uv[..., 0] >= 0)
        & (target_uv[..., 0] < width)
        & (target_uv[..., 1] >= 0)
        & (target_uv[..., 1] < height)
    )
    cells = torch.floor(target_uv + 0.5).to(torch.long)
    in_bounds &= (
        (cells[..., 0] >= 0)
        & (cells[..., 0] < width)
        & (cells[..., 1] >= 0)
        & (cells[..., 1] < height)
    )
    valid = target_valid & in_bounds
    masks = []
    for frame_index in range(target_uv.shape[0]):
        mask = torch.zeros(height * width, dtype=torch.bool, device=target_uv.device)
        selected = cells[frame_index, valid[frame_index]]
        if selected.numel():
            linear = selected[:, 1] * width + selected[:, 0]
            mask[linear] = True
        masks.append(mask.reshape(1, height, width))
    return torch.stack(masks)

