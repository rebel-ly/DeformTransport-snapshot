"""Target-validity contracts for material-point latent transport.

This module keeps projection validity and raster visibility as separate
semantic fields.

projection_only:
    target_valid = projection_valid

source_and_future_visible:
    target_valid = projection_valid AND aligned_visible

Source visibility remains the responsibility of soft_point_transport().
"""

from __future__ import annotations

from typing import Any, Literal

import torch


TransportValidityMode = Literal[
    "projection_only",
    "source_and_future_visible",
]


def select_target_validity(
    *,
    state: dict[str, Any],
    latent_indices: torch.Tensor,
    visibility_state: dict[str, Any] | None,
    mode: TransportValidityMode,
) -> dict[str, torch.Tensor | None]:
    """Select the target-validity tensor for the requested contract."""

    if mode not in (
        "projection_only",
        "source_and_future_visible",
    ):
        raise ValueError(
            f"unsupported transport validity mode: {mode}"
        )

    required_state = {
        "projection_valid",
        "source_visible",
        "point_id",
    }

    missing_state = sorted(
        required_state - set(state)
    )

    if missing_state:
        raise ValueError(
            "transport-ready artifact is missing validity fields: "
            f"{missing_state}"
        )

    indices = torch.as_tensor(
        latent_indices,
        dtype=torch.long,
        device="cpu",
    ).contiguous()

    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError(
            "latent_indices must be a non-empty one-dimensional tensor"
        )

    projection_valid = torch.as_tensor(
        state["projection_valid"],
        dtype=torch.bool,
        device="cpu",
    ).contiguous()

    source_visible = torch.as_tensor(
        state["source_visible"],
        dtype=torch.bool,
        device="cpu",
    ).contiguous()

    point_id = torch.as_tensor(
        state["point_id"],
        dtype=torch.long,
        device="cpu",
    ).contiguous()

    if projection_valid.ndim != 2:
        raise ValueError(
            "projection_valid must have shape [pixel_frames, points]"
        )

    frame_count, point_count = projection_valid.shape

    if source_visible.shape != (point_count,):
        raise ValueError(
            "source_visible shape does not match projection_valid"
        )

    if point_id.shape != (point_count,):
        raise ValueError(
            "point_id shape does not match projection_valid"
        )

    if int(indices.min()) < 0 or int(indices.max()) >= frame_count:
        raise ValueError(
            "latent_indices exceed projection_valid frame range"
        )

    projection_target_valid = projection_valid.index_select(
        0,
        indices,
    ).contiguous()

    if mode == "projection_only":
        return {
            "projection_target_valid": projection_target_valid,
            "selected_target_valid": projection_target_valid.clone(),
            "selected_future_visible": None,
        }

    if visibility_state is None:
        raise ValueError(
            "source_and_future_visible mode requires a visibility artifact"
        )

    if "aligned_visible" not in visibility_state:
        raise ValueError(
            "visibility artifact is missing aligned_visible"
        )

    aligned_visible = torch.as_tensor(
        visibility_state["aligned_visible"],
        dtype=torch.bool,
        device="cpu",
    ).contiguous()

    if aligned_visible.shape != projection_valid.shape:
        raise ValueError(
            "aligned_visible shape does not match projection_valid: "
            f"aligned_visible={tuple(aligned_visible.shape)}, "
            f"projection_valid={tuple(projection_valid.shape)}"
        )

    if not torch.equal(
        aligned_visible[0],
        source_visible,
    ):
        raise ValueError(
            "aligned_visible frame 0 does not reproduce source_visible"
        )

    if bool(
        (
            aligned_visible
            & ~projection_valid
        ).any()
    ):
        raise ValueError(
            "aligned_visible must be a subset of projection_valid"
        )

    if "point_id" in visibility_state:
        visibility_point_id = torch.as_tensor(
            visibility_state["point_id"],
            dtype=torch.long,
            device="cpu",
        )

        if not torch.equal(
            visibility_point_id,
            point_id,
        ):
            raise ValueError(
                "visibility point_id does not match transport point_id"
            )

    for temporal_key in (
        "frame_ids",
        "simulation_steps",
    ):
        if (
            temporal_key in state
            and temporal_key in visibility_state
        ):
            state_value = torch.as_tensor(
                state[temporal_key],
                device="cpu",
            )
            visibility_value = torch.as_tensor(
                visibility_state[temporal_key],
                device="cpu",
            )

            if not torch.equal(
                state_value,
                visibility_value,
            ):
                raise ValueError(
                    f"visibility {temporal_key} does not match "
                    "transport-ready artifact"
                )

    selected_future_visible = aligned_visible.index_select(
        0,
        indices,
    ).contiguous()

    selected_target_valid = (
        projection_target_valid
        & selected_future_visible
    ).contiguous()

    return {
        "projection_target_valid": projection_target_valid,
        "selected_target_valid": selected_target_valid,
        "selected_future_visible": selected_future_visible,
    }
