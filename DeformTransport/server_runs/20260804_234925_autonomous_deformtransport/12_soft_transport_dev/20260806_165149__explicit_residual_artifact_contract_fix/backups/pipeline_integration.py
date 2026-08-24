"""Validated boundary for injecting precomputed transport into RealWonder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch


TransportCondition = Literal["correct", "shuffled", "flow", "blend"]

_LATENT_KEYS = {
    "correct": "correct_fused_latent",
    "shuffled": "shuffled_fused_latent",
    "flow": "flow_fused_latent",
    "blend": "blend_fused_latent",
}

_RAW_TRANSPORT_KEYS = {
    "correct": "correct_transported_latent",
    "shuffled": "shuffled_transported_latent",
}


def load_precomputed_transport_residual(
    artifact_path: str | Path,
    *,
    mode: TransportCondition,
    reference_latent: torch.Tensor,
) -> torch.Tensor:
    """Load an artifact-local masked transport residual.

    The residual is computed entirely inside the artifact:

        transported_latent - artifact_target_latent

    It is then moved to the device and dtype of the freshly encoded
    RealWonder reference latent. The artifact target is deliberately not
    assumed to be numerically identical to the runtime VAE encoding.
    """

    if mode not in _RAW_TRANSPORT_KEYS:
        raise ValueError(
            "residual transport currently supports only "
            f"'correct' and 'shuffled', received: {mode}"
        )

    if reference_latent.ndim != 5:
        raise ValueError(
            "reference_latent must have shape [B,T,C,H,W]"
        )

    path = Path(artifact_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, dict):
        raise ValueError(
            "transport artifact must contain a dictionary"
        )

    raw_key = _RAW_TRANSPORT_KEYS[mode]

    transported = state.get(raw_key)
    target = state.get("target_latent")

    for tensor, label in (
        (transported, raw_key),
        (target, "target_latent"),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"transport artifact is missing tensor {label!r}"
            )

        if tensor.ndim != 5:
            raise ValueError(
                f"{label} must have shape [B,T,C,H,W]"
            )

        if tuple(tensor.shape) != tuple(
            reference_latent.shape
        ):
            raise ValueError(
                f"{label} shape {tuple(tensor.shape)} does not "
                "match freshly encoded sim_latent "
                f"{tuple(reference_latent.shape)}"
            )

        if (
            not tensor.dtype.is_floating_point
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"{label} must be a finite floating-point tensor"
            )

    if tuple(transported.shape) != tuple(target.shape):
        raise ValueError(
            "transported latent and target latent shapes differ"
        )

    frame_count = transported.shape[1]
    height = transported.shape[3]
    width = transported.shape[4]

    mask = state.get("transport_mask")
    count = state.get("contribution_count")

    expected_mask_shape = (
        frame_count,
        1,
        height,
        width,
    )

    if (
        not isinstance(mask, torch.Tensor)
        or tuple(mask.shape) != expected_mask_shape
    ):
        raise ValueError(
            "transport_mask does not match transported latent"
        )

    if mask.dtype != torch.bool:
        raise ValueError(
            "transport_mask must be boolean"
        )

    if (
        not isinstance(count, torch.Tensor)
        or tuple(count.shape) != tuple(mask.shape)
    ):
        raise ValueError(
            "contribution_count does not match transport_mask"
        )

    if (
        count.dtype.is_floating_point
        or bool((count < 0).any())
    ):
        raise ValueError(
            "contribution_count must contain "
            "nonnegative integers"
        )

    if not torch.equal(mask, count > 0):
        raise ValueError(
            "transport_mask must equal contribution_count > 0"
        )

    residual = (
        transported.to(torch.float32)
        - target.to(torch.float32)
    )

    mask_5d = (
        mask.unsqueeze(0)
        .expand_as(residual)
    )

    residual = torch.where(
        mask_5d,
        residual,
        torch.zeros_like(residual),
    )

    if not bool(torch.isfinite(residual).all()):
        raise ValueError(
            "computed transport residual contains NaN or Inf"
        )

    return residual.to(
        device=reference_latent.device,
        dtype=reference_latent.dtype,
    ).contiguous()


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
