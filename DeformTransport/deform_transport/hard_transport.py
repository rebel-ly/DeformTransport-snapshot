"""Checkpoint-free nearest-cell point-feature transport."""

from __future__ import annotations

from typing import Literal

import torch


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


def _nearest_cells(
    uv: torch.Tensor, *, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest integer XY cells and an unclamped in-bounds mask."""

    uv_float = uv.to(torch.float32)
    finite = torch.isfinite(uv_float).all(dim=-1)
    continuous_in_bounds = (
        finite
        & (uv_float[..., 0] >= 0)
        & (uv_float[..., 0] < width)
        & (uv_float[..., 1] >= 0)
        & (uv_float[..., 1] < height)
    )
    cells = torch.floor(uv_float + 0.5).to(torch.long)
    rounded_in_bounds = (
        (cells[..., 0] >= 0)
        & (cells[..., 0] < width)
        & (cells[..., 1] >= 0)
        & (cells[..., 1] < height)
    )
    return cells, continuous_in_bounds & rounded_in_bounds


def make_objectwise_feature_permutation(
    eligible_source_points: torch.Tensor,
    object_id: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Map each point row to a shuffled source-feature row within its object."""

    eligible = torch.as_tensor(eligible_source_points, dtype=torch.bool, device="cpu")
    objects = torch.as_tensor(object_id, dtype=torch.long, device="cpu")
    if eligible.ndim != 1 or objects.shape != eligible.shape:
        raise ValueError("eligible_source_points and object_id must have shape [N]")
    permutation = torch.arange(eligible.numel(), dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for current_object in torch.unique(objects, sorted=True):
        indices = torch.nonzero(
            eligible & (objects == current_object), as_tuple=False
        ).flatten()
        if indices.numel() > 1:
            order = torch.randperm(indices.numel(), generator=generator)
            permutation[indices] = indices[order]
    return permutation


def hard_point_transport(
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
) -> dict[str, torch.Tensor | str | int]:
    """Carry nearest-cell source features along stable point identities.

    Coordinates are XY in the spatial grid. Out-of-frame points are filtered
    before integer indexing and are never clamped to an image boundary.
    """

    if mode not in ("correct", "shuffled"):
        raise ValueError(f"unsupported transport mode: {mode}")
    source_grid = torch.as_tensor(source_grid)
    if source_grid.ndim != 3 or not source_grid.dtype.is_floating_point:
        raise ValueError("source_grid must be floating point with shape [C, H, W]")
    channel_count, height, width = source_grid.shape
    device = source_grid.device
    source_uv = _as_device_tensor(source_uv, device=device)
    target_uv = _as_device_tensor(target_uv, device=device)
    source_visible = _as_device_tensor(
        source_visible, device=device, dtype=torch.bool
    )
    source_valid = _as_device_tensor(source_valid, device=device, dtype=torch.bool)
    target_valid = _as_device_tensor(target_valid, device=device, dtype=torch.bool)
    point_id = _as_device_tensor(point_id, device=device, dtype=torch.long)

    if source_uv.ndim != 2 or source_uv.shape[-1] != 2:
        raise ValueError("source_uv must have shape [N, 2]")
    point_count = source_uv.shape[0]
    if target_uv.ndim != 3 or target_uv.shape[1:] != (point_count, 2):
        raise ValueError("target_uv must have shape [T, N, 2]")
    frame_count = target_uv.shape[0]
    if source_visible.shape != (point_count,) or source_valid.shape != (point_count,):
        raise ValueError("source_visible and source_valid must have shape [N]")
    if target_valid.shape != (frame_count, point_count):
        raise ValueError("target_valid must have shape [T, N]")
    if point_id.shape != (point_count,) or torch.unique(point_id).numel() != point_count:
        raise ValueError("point_id must contain N unique identities")
    if object_id is None:
        object_id = torch.zeros(point_count, dtype=torch.long, device=device)
    else:
        object_id = _as_device_tensor(object_id, device=device, dtype=torch.long)
        if object_id.shape != (point_count,):
            raise ValueError("object_id must have shape [N]")

    source_cells, source_in_bounds = _nearest_cells(
        source_uv, height=height, width=width
    )
    eligible_source = source_visible & source_valid & source_in_bounds
    correct_point_features = torch.zeros(
        (point_count, channel_count), dtype=source_grid.dtype, device=device
    )
    if bool(eligible_source.any()):
        selected_cells = source_cells[eligible_source]
        correct_point_features[eligible_source] = source_grid[
            :, selected_cells[:, 1], selected_cells[:, 0]
        ].T

    if mode == "shuffled":
        permutation = make_objectwise_feature_permutation(
            eligible_source.detach().cpu(), object_id.detach().cpu(), seed=seed
        ).to(device)
        point_features = correct_point_features[permutation]
    else:
        permutation = torch.arange(point_count, dtype=torch.long, device=device)
        point_features = correct_point_features

    target_cells, target_in_bounds = _nearest_cells(
        target_uv, height=height, width=width
    )
    valid_point_mask = (
        eligible_source.unsqueeze(0) & target_valid & target_in_bounds
    )
    accumulated_frames = []
    count_frames = []
    mask_frames = []
    flat_cell_count = height * width
    for frame_index in range(frame_count):
        valid = valid_point_mask[frame_index]
        accumulated = torch.zeros(
            (channel_count, flat_cell_count),
            dtype=source_grid.dtype,
            device=device,
        )
        counts = torch.zeros((1, flat_cell_count), dtype=torch.long, device=device)
        if bool(valid.any()):
            cells = target_cells[frame_index, valid]
            linear_index = cells[:, 1] * width + cells[:, 0]
            accumulated.scatter_add_(
                1,
                linear_index.unsqueeze(0).expand(channel_count, -1),
                point_features[valid].T,
            )
            counts.scatter_add_(
                1,
                linear_index.unsqueeze(0),
                torch.ones((1, linear_index.numel()), dtype=torch.long, device=device),
            )
        mask = counts > 0
        normalized = accumulated / counts.clamp_min(1).to(source_grid.dtype)
        accumulated_frames.append(normalized.reshape(channel_count, height, width))
        count_frames.append(counts.reshape(1, height, width))
        mask_frames.append(mask.reshape(1, height, width))

    result: dict[str, torch.Tensor | str | int] = {
        "mode": mode,
        "seed": int(seed),
        "point_features": point_features,
        "transported_grid": torch.stack(accumulated_frames),
        "transport_mask": torch.stack(mask_frames),
        "contribution_count": torch.stack(count_frames),
        "valid_point_mask": valid_point_mask,
        "source_point_mask": eligible_source,
        "permutation": permutation,
    }
    if not bool(torch.isfinite(result["transported_grid"]).all()):
        raise RuntimeError("hard transport produced NaN or Inf")
    return result
