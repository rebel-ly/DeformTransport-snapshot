"""Validated boundary for injecting precomputed transport into RealWonder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch


TransportCondition = Literal["correct", "shuffled", "flow"]

_LATENT_KEYS = {
    "correct": "correct_fused_latent",
    "shuffled": "shuffled_fused_latent",
    "flow": "flow_fused_latent",
}


def load_precomputed_transport_latent(
    artifact_path: str | Path,
    *,
    mode: TransportCondition,
    reference_latent: torch.Tensor,
) -> torch.Tensor:
    """Load a fused latent and match the existing RealWonder ``sim_latent``.

    The artifact is produced by ``run_wan_vae_transport_probe.py``. Requiring
    exact agreement with the freshly encoded reference prevents an artifact
    from a different case, frame count, resolution, or latent layout from
    silently entering the diffusion pipeline.
    """

    if mode not in _LATENT_KEYS:
        raise ValueError(f"unsupported transport mode: {mode}")
    if reference_latent.ndim != 5:
        raise ValueError("reference_latent must have shape [B,T,C,H,W]")

    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("transport artifact must contain a dictionary")

    key = _LATENT_KEYS[mode]
    selected = state.get(key)
    if not isinstance(selected, torch.Tensor):
        raise ValueError(f"transport artifact is missing tensor {key!r}")
    if selected.ndim != 5 or tuple(selected.shape) != tuple(reference_latent.shape):
        raise ValueError(
            f"{key} shape {tuple(selected.shape)} does not match freshly encoded "
            f"sim_latent {tuple(reference_latent.shape)}"
        )
    if not selected.dtype.is_floating_point or not bool(torch.isfinite(selected).all()):
        raise ValueError(f"{key} must be a finite floating-point tensor")

    frame_count, height, width = selected.shape[1], selected.shape[3], selected.shape[4]
    mask = state.get("transport_mask")
    count = state.get("contribution_count")
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (
        frame_count,
        1,
        height,
        width,
    ):
        raise ValueError("transport_mask does not match the selected latent")
    if mask.dtype != torch.bool:
        raise ValueError("transport_mask must be boolean")
    if not isinstance(count, torch.Tensor) or tuple(count.shape) != tuple(mask.shape):
        raise ValueError("contribution_count does not match transport_mask")
    if count.dtype.is_floating_point or bool((count < 0).any()):
        raise ValueError("contribution_count must contain nonnegative integers")
    if not torch.equal(mask, count > 0):
        raise ValueError("transport_mask must equal contribution_count > 0")

    return selected.to(
        device=reference_latent.device,
        dtype=reference_latent.dtype,
    ).contiguous()
