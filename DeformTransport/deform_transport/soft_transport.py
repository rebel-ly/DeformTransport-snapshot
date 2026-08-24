"""Normalized bilinear material-point feature transport.

This module is intentionally separate from hard_transport.py so the verified
nearest-cell implementation remains available as a fixed baseline and ablation.
"""

from __future__ import annotations

from typing import Literal

import torch

from deform_transport.hard_transport import make_objectwise_feature_permutation


TransportMode = Literal["correct", "shuffled"]


def _as_device_tensor(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if type(tensor) is not torch.Tensor:
        tensor = tensor.as_subclass(torch.Tensor)
    return tensor.to(device=device, dtype=dtype).contiguous()


def _validate_continuous_uv(
    uv: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return validity for finite continuous XY coordinates inside the grid."""

    uv_float = uv.to(torch.float32)
    return (
        torch.isfinite(uv_float).all(dim=-1)
        & (uv_float[..., 0] >= 0)
        & (uv_float[..., 0] < width)
        & (uv_float[..., 1] >= 0)
        & (uv_float[..., 1] < height)
    )


def _bilinear_neighbors(
    uv: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four XY cells, bilinear weights, and per-neighbor validity.

    Args:
        uv: Continuous XY coordinates with shape [..., 2].

    Returns:
        cells: [..., 4, 2] integer XY cells.
        weights: [..., 4] bilinear weights.
        valid: [..., 4] indicating neighbors inside the grid.

    Neighbor order:
        0: top-left
        1: top-right
        2: bottom-left
        3: bottom-right
    """

    uv_float = uv.to(torch.float32)
    x = uv_float[..., 0]
    y = uv_float[..., 1]

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    cells = torch.stack(
        [
            torch.stack([x0, y0], dim=-1),
            torch.stack([x1, y0], dim=-1),
            torch.stack([x0, y1], dim=-1),
            torch.stack([x1, y1], dim=-1),
        ],
        dim=-2,
    ).to(torch.long)

    weights = torch.stack(
        [
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ],
        dim=-1,
    )

    valid = (
        (cells[..., 0] >= 0)
        & (cells[..., 0] < width)
        & (cells[..., 1] >= 0)
        & (cells[..., 1] < height)
        & torch.isfinite(weights)
        & (weights > 0)
    )

    return cells, weights, valid


def _sample_source_bilinear(
    source_grid: torch.Tensor,
    source_uv: torch.Tensor,
    eligible_source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one feature per source point using normalized bilinear sampling."""

    channel_count, height, width = source_grid.shape
    device = source_grid.device

    cells, weights, neighbor_valid = _bilinear_neighbors(
        source_uv,
        height=height,
        width=width,
    )

    point_count = source_uv.shape[0]
    numerator = torch.zeros(
        (point_count, channel_count),
        dtype=source_grid.dtype,
        device=device,
    )
    denominator = torch.zeros(
        point_count,
        dtype=torch.float32,
        device=device,
    )

    for neighbor_index in range(4):
        valid = eligible_source & neighbor_valid[:, neighbor_index]
        if not bool(valid.any()):
            continue

        selected_cells = cells[valid, neighbor_index]
        selected_weights = weights[valid, neighbor_index]

        sampled = source_grid[
            :,
            selected_cells[:, 1],
            selected_cells[:, 0],
        ].T

        numerator[valid] += (
            sampled * selected_weights.to(source_grid.dtype).unsqueeze(1)
        )
        denominator[valid] += selected_weights

    source_sample_valid = eligible_source & (denominator > 0)

    point_features = torch.zeros(
        (point_count, channel_count),
        dtype=source_grid.dtype,
        device=device,
    )

    if bool(source_sample_valid.any()):
        point_features[source_sample_valid] = (
            numerator[source_sample_valid]
            / denominator[source_sample_valid]
            .to(source_grid.dtype)
            .unsqueeze(1)
        )

    return point_features, source_sample_valid


def soft_point_transport(
    source_grid: torch.Tensor,
    source_uv: torch.Tensor,
    target_uv: torch.Tensor,
    source_visible: torch.Tensor,
    source_valid: torch.Tensor,
    target_valid: torch.Tensor,
    point_id: torch.Tensor,
    *,
    object_id: torch.Tensor | None = None,
    mode: TransportMode = "correct",
    seed: int = 0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor | str | int | float]:
    """Transport point features using normalized bilinear forward splatting.

    Source features are bilinearly sampled from the initial latent grid.
    Each valid target point contributes to up to four neighboring target cells.
    Contributions are normalized by their accumulated spatial weights.

    This first implementation deliberately excludes depth, occlusion confidence,
    learned fusion, and denoising-time injection so that bilinear assignment can
    be evaluated as an isolated design variable.
    """

    if mode not in ("correct", "shuffled"):
        raise ValueError(f"unsupported transport mode: {mode}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    source_grid = torch.as_tensor(source_grid)

    if source_grid.ndim != 3 or not source_grid.dtype.is_floating_point:
        raise ValueError(
            "source_grid must be floating point with shape [C, H, W]"
        )

    channel_count, height, width = source_grid.shape
    device = source_grid.device

    source_uv = _as_device_tensor(
        source_uv,
        device=device,
        dtype=torch.float32,
    )
    target_uv = _as_device_tensor(
        target_uv,
        device=device,
        dtype=torch.float32,
    )
    source_visible = _as_device_tensor(
        source_visible,
        device=device,
        dtype=torch.bool,
    )
    source_valid = _as_device_tensor(
        source_valid,
        device=device,
        dtype=torch.bool,
    )
    target_valid = _as_device_tensor(
        target_valid,
        device=device,
        dtype=torch.bool,
    )
    point_id = _as_device_tensor(
        point_id,
        device=device,
        dtype=torch.long,
    )

    if source_uv.ndim != 2 or source_uv.shape[-1] != 2:
        raise ValueError("source_uv must have shape [N, 2]")

    point_count = source_uv.shape[0]

    if target_uv.ndim != 3 or target_uv.shape[1:] != (point_count, 2):
        raise ValueError("target_uv must have shape [T, N, 2]")

    frame_count = target_uv.shape[0]

    if source_visible.shape != (point_count,):
        raise ValueError("source_visible must have shape [N]")
    if source_valid.shape != (point_count,):
        raise ValueError("source_valid must have shape [N]")
    if target_valid.shape != (frame_count, point_count):
        raise ValueError("target_valid must have shape [T, N]")
    if point_id.shape != (point_count,):
        raise ValueError("point_id must have shape [N]")
    if torch.unique(point_id).numel() != point_count:
        raise ValueError("point_id must contain N unique identities")

    if object_id is None:
        object_id = torch.zeros(
            point_count,
            dtype=torch.long,
            device=device,
        )
    else:
        object_id = _as_device_tensor(
            object_id,
            device=device,
            dtype=torch.long,
        )
        if object_id.shape != (point_count,):
            raise ValueError("object_id must have shape [N]")

    source_continuous_valid = _validate_continuous_uv(
        source_uv,
        height=height,
        width=width,
    )

    eligible_source = (
        source_visible
        & source_valid
        & source_continuous_valid
    )

    correct_point_features, source_sample_valid = _sample_source_bilinear(
        source_grid,
        source_uv,
        eligible_source,
    )

    if mode == "shuffled":
        permutation = make_objectwise_feature_permutation(
            source_sample_valid.detach().cpu(),
            object_id.detach().cpu(),
            seed=seed,
        ).to(device)
        point_features = correct_point_features[permutation]
    else:
        permutation = torch.arange(
            point_count,
            dtype=torch.long,
            device=device,
        )
        point_features = correct_point_features

    target_continuous_valid = _validate_continuous_uv(
        target_uv,
        height=height,
        width=width,
    )

    valid_point_mask = (
        source_sample_valid.unsqueeze(0)
        & target_valid
        & target_continuous_valid
    )

    target_cells, target_weights, target_neighbor_valid = (
        _bilinear_neighbors(
            target_uv,
            height=height,
            width=width,
        )
    )

    flat_cell_count = height * width

    transported_frames = []
    mask_frames = []
    count_frames = []
    weight_frames = []

    for frame_index in range(frame_count):
        feature_sum = torch.zeros(
            (channel_count, flat_cell_count),
            dtype=source_grid.dtype,
            device=device,
        )
        weight_sum = torch.zeros(
            (1, flat_cell_count),
            dtype=torch.float32,
            device=device,
        )
        contribution_count = torch.zeros(
            (1, flat_cell_count),
            dtype=torch.long,
            device=device,
        )

        for neighbor_index in range(4):
            valid = (
                valid_point_mask[frame_index]
                & target_neighbor_valid[frame_index, :, neighbor_index]
            )

            if not bool(valid.any()):
                continue

            cells = target_cells[
                frame_index,
                valid,
                neighbor_index,
            ]
            weights = target_weights[
                frame_index,
                valid,
                neighbor_index,
            ]

            linear_index = cells[:, 1] * width + cells[:, 0]

            weighted_features = (
                point_features[valid].T
                * weights.to(source_grid.dtype).unsqueeze(0)
            )

            feature_sum.scatter_add_(
                1,
                linear_index.unsqueeze(0).expand(channel_count, -1),
                weighted_features,
            )

            weight_sum.scatter_add_(
                1,
                linear_index.unsqueeze(0),
                weights.unsqueeze(0),
            )

            contribution_count.scatter_add_(
                1,
                linear_index.unsqueeze(0),
                torch.ones(
                    (1, linear_index.numel()),
                    dtype=torch.long,
                    device=device,
                ),
            )

        transport_mask = weight_sum > eps

        normalized = feature_sum / weight_sum.clamp_min(eps).to(
            source_grid.dtype
        )

        normalized = torch.where(
            transport_mask.expand_as(normalized),
            normalized,
            torch.zeros_like(normalized),
        )

        transported_frames.append(
            normalized.reshape(channel_count, height, width)
        )
        mask_frames.append(
            transport_mask.reshape(1, height, width)
        )
        count_frames.append(
            contribution_count.reshape(1, height, width)
        )
        weight_frames.append(
            weight_sum.reshape(1, height, width)
        )

    result: dict[str, torch.Tensor | str | int | float] = {
        "mode": mode,
        "seed": int(seed),
        "eps": float(eps),
        "point_features": point_features,
        "transported_grid": torch.stack(transported_frames),
        "transport_mask": torch.stack(mask_frames),
        "contribution_count": torch.stack(count_frames),
        "transport_weight": torch.stack(weight_frames),
        "valid_point_mask": valid_point_mask,
        "source_point_mask": source_sample_valid,
        "permutation": permutation,
    }

    if not bool(torch.isfinite(result["point_features"]).all()):
        raise RuntimeError("soft transport produced non-finite point features")
    if not bool(torch.isfinite(result["transported_grid"]).all()):
        raise RuntimeError("soft transport produced NaN or Inf")
    if not bool(torch.isfinite(result["transport_weight"]).all()):
        raise RuntimeError("soft transport produced non-finite weights")

    return result
