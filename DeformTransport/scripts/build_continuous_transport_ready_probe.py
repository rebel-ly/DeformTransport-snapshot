"""Add continuous latent coordinates to an existing transport-ready artifact.

This probe preserves the original format-version-1 contract and all legacy
integer coordinates. It adds continuous coordinates derived from the already
stored float32 video coordinates.

The input artifact is never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from deform_transport.transport_ready import validate_transport_ready


EXTENSION_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_plain_tensor(
    state: Mapping[str, Any],
    key: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = state.get(key)

    if type(value) is not torch.Tensor:
        raise ValueError(f"{key} must be a plain torch.Tensor")

    if value.dtype != dtype:
        raise ValueError(
            f"{key} must have dtype {dtype}, got {value.dtype}"
        )

    if value.device.type != "cpu":
        raise ValueError(f"{key} must be stored on CPU")

    return value


def validate_continuous_extension(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    validate_transport_ready(state)

    source_video = _require_plain_tensor(
        state,
        "source_points_2d_video",
        dtype=torch.float32,
    )
    target_video = _require_plain_tensor(
        state,
        "points_2d_video",
        dtype=torch.float32,
    )
    source_discrete = _require_plain_tensor(
        state,
        "source_points_2d_latent",
        dtype=torch.long,
    )
    target_discrete = _require_plain_tensor(
        state,
        "points_2d_latent",
        dtype=torch.long,
    )
    source_continuous = _require_plain_tensor(
        state,
        "source_points_2d_latent_continuous",
        dtype=torch.float32,
    )
    target_continuous = _require_plain_tensor(
        state,
        "points_2d_latent_continuous",
        dtype=torch.float32,
    )

    if source_continuous.shape != source_discrete.shape:
        raise ValueError(
            "source continuous and discrete coordinates must have "
            "identical shapes"
        )

    if target_continuous.shape != target_discrete.shape:
        raise ValueError(
            "target continuous and discrete coordinates must have "
            "identical shapes"
        )

    if not bool(torch.isfinite(source_continuous).all()):
        raise ValueError("source continuous coordinates contain NaN or Inf")

    if not bool(torch.isfinite(target_continuous).all()):
        raise ValueError("target continuous coordinates contain NaN or Inf")

    video_height = int(state["video_height"])
    video_width = int(state["video_width"])
    latent_height = int(state["latent_height"])
    latent_width = int(state["latent_width"])

    scale = torch.tensor(
        [
            latent_width / video_width,
            latent_height / video_height,
        ],
        dtype=torch.float32,
    )

    expected_source = source_video * scale
    expected_target = target_video * scale

    if not torch.equal(source_continuous, expected_source):
        raise ValueError(
            "source continuous coordinates do not match video-to-latent scale"
        )

    if not torch.equal(target_continuous, expected_target):
        raise ValueError(
            "target continuous coordinates do not match video-to-latent scale"
        )

    recovered_source = torch.floor(
        source_continuous
    ).to(torch.long)

    recovered_target = torch.floor(
        target_continuous
    ).to(torch.long)

    if not torch.equal(recovered_source, source_discrete):
        raise ValueError(
            "floor(source continuous) does not recover legacy coordinates"
        )

    if not torch.equal(recovered_target, target_discrete):
        raise ValueError(
            "floor(target continuous) does not recover legacy coordinates"
        )

    source_fraction = torch.abs(
        source_continuous
        - torch.floor(source_continuous)
    )
    target_fraction = torch.abs(
        target_continuous
        - torch.floor(target_continuous)
    )

    source_noninteger = (
        source_fraction > 1e-6
    ).any(dim=-1)

    target_noninteger = (
        target_fraction > 1e-6
    ).any(dim=-1)

    if not bool(source_noninteger.any()):
        raise ValueError(
            "source continuous coordinates contain no fractional positions"
        )

    if not bool(target_noninteger.any()):
        raise ValueError(
            "target continuous coordinates contain no fractional positions"
        )

    source_valid = state["source_valid"]
    target_valid = state["projection_valid"]

    valid_source_coordinates = source_continuous[source_valid]
    valid_target_coordinates = target_continuous[target_valid]

    if valid_source_coordinates.numel():
        source_in_bounds = (
            (valid_source_coordinates[:, 0] >= 0)
            & (valid_source_coordinates[:, 0] < latent_width)
            & (valid_source_coordinates[:, 1] >= 0)
            & (valid_source_coordinates[:, 1] < latent_height)
        )

        if not bool(source_in_bounds.all()):
            raise ValueError(
                "valid source continuous coordinates are out of bounds"
            )

    if valid_target_coordinates.numel():
        target_in_bounds = (
            (valid_target_coordinates[:, 0] >= 0)
            & (valid_target_coordinates[:, 0] < latent_width)
            & (valid_target_coordinates[:, 1] >= 0)
            & (valid_target_coordinates[:, 1] < latent_height)
        )

        if not bool(target_in_bounds.all()):
            raise ValueError(
                "valid target continuous coordinates are out of bounds"
            )

    return {
        "extension_version": int(
            state["continuous_coordinate_extension_version"]
        ),
        "scale_xy": scale.tolist(),
        "source_shape": list(source_continuous.shape),
        "target_shape": list(target_continuous.shape),
        "source_noninteger_points": int(source_noninteger.sum()),
        "target_noninteger_points": int(target_noninteger.sum()),
        "source_floor_recovers_legacy": True,
        "target_floor_recovers_legacy": True,
        "source_fraction_mean": float(source_fraction.mean()),
        "target_fraction_mean": float(target_fraction.mean()),
        "source_fraction_max": float(source_fraction.max()),
        "target_fraction_max": float(target_fraction.max()),
    }


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_path}"
        )

    state = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    validate_transport_ready(state)

    output_state = copy.deepcopy(state)

    scale = torch.tensor(
        [
            int(state["latent_width"]) / int(state["video_width"]),
            int(state["latent_height"]) / int(state["video_height"]),
        ],
        dtype=torch.float32,
    )

    output_state["source_points_2d_latent_continuous"] = (
        state["source_points_2d_video"].to(torch.float32)
        * scale
    ).contiguous()

    output_state["points_2d_latent_continuous"] = (
        state["points_2d_video"].to(torch.float32)
        * scale
    ).contiguous()

    output_state["continuous_coordinate_extension_version"] = (
        EXTENSION_VERSION
    )

    coordinate_system = dict(
        output_state.get("coordinate_system", {})
    )

    coordinate_system["points_2d_latent"] = (
        "wan_spatial_cell_xy_floor_legacy"
    )
    coordinate_system[
        "points_2d_latent_continuous"
    ] = "wan_spatial_xy_continuous_legacy_scale"

    output_state["coordinate_system"] = coordinate_system

    output_state["continuous_coordinate_mapping"] = {
        "source": "source_points_2d_video * scale_xy",
        "target": "points_2d_video * scale_xy",
        "scale_xy": scale.tolist(),
        "legacy_recovery": "floor(continuous) == legacy_discrete",
        "pixel_center_offset": False,
        "align_corners": None,
    }

    summary_before_save = validate_continuous_extension(
        output_state
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=False,
    )

    torch.save(output_state, output_path)

    loaded = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    summary_after_load = validate_continuous_extension(
        loaded
    )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "legacy_format_version_preserved": int(
            loaded["format_version"]
        ),
        "summary_before_save": summary_before_save,
        "summary_after_load": summary_after_load,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
