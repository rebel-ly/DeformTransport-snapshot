"""Read-only audit for material-identity evaluation contracts.

This script does not modify any artifact and does not run a GPU model.
It inventories trajectory coordinates, temporal mappings, validity fields,
and the existing generated videos needed for MIC/TAE/Warp metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport-ready",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--confidence-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--final-sim",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {
            "type": type(value).__name__,
            "repr": repr(value)[:300],
        }

    result: dict[str, Any] = {
        "type": "Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }

    if value.numel() == 0:
        return result

    if value.dtype == torch.bool:
        result["true_values"] = int(value.sum())
        result["false_values"] = int(
            value.numel() - value.sum()
        )
        return result

    if value.dtype.is_floating_point:
        float_value = value.to(torch.float32)

        result.update(
            {
                "finite": bool(
                    torch.isfinite(float_value).all()
                ),
                "min": float(float_value.min()),
                "max": float(float_value.max()),
                "mean": float(float_value.mean()),
                "std": float(float_value.std()),
                "nonzero_values": int(
                    torch.count_nonzero(float_value)
                ),
            }
        )
    else:
        result.update(
            {
                "min": int(value.min()),
                "max": int(value.max()),
                "nonzero_values": int(
                    torch.count_nonzero(value)
                ),
            }
        )

    return result


def coordinate_report(
    tensor: torch.Tensor | None,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        return {
            "available": False,
        }

    if tensor.shape[-1] != 2:
        return {
            "available": True,
            "valid_xy_shape": False,
            "inventory": inventory(tensor),
        }

    value = tensor.to(torch.float32)

    x = value[..., 0]
    y = value[..., 1]

    finite = torch.isfinite(value).all(dim=-1)

    in_bounds = (
        finite
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    fractional = (
        torch.abs(value - torch.round(value))
        > 1e-6
    ).any(dim=-1)

    return {
        "available": True,
        "valid_xy_shape": True,
        "inventory": inventory(tensor),
        "finite_points": int(finite.sum()),
        "in_bounds_points": int(in_bounds.sum()),
        "fractional_points": int(
            fractional.sum()
        ),
        "total_points": int(finite.numel()),
        "in_bounds_ratio": float(
            in_bounds.to(torch.float32).mean()
        ),
    }


def video_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
        }

    reader = imageio.get_reader(path)

    try:
        frame_count = reader.count_frames()
        first = reader.get_data(0)

        middle_index = max(0, frame_count // 2)
        middle = reader.get_data(middle_index)
        final = reader.get_data(frame_count - 1)

        return {
            "path": str(path),
            "exists": True,
            "frame_count": int(frame_count),
            "first_shape": list(first.shape),
            "middle_shape": list(middle.shape),
            "final_shape": list(final.shape),
            "dtype": str(first.dtype),
        }
    finally:
        reader.close()


def main() -> None:
    args = parse_args()

    transport_path = args.transport_ready.resolve()
    artifact_path = args.confidence_artifact.resolve()
    final_sim = args.final_sim.resolve()
    output_report = args.output_report.resolve()

    if not transport_path.is_file():
        raise FileNotFoundError(transport_path)

    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    if not final_sim.is_dir():
        raise FileNotFoundError(final_sim)

    if output_report.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_report}"
        )

    transport = torch.load(
        transport_path,
        map_location="cpu",
        weights_only=True,
    )

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(transport, dict):
        raise ValueError(
            "transport-ready artifact is not a dictionary"
        )

    if not isinstance(artifact, dict):
        raise ValueError(
            "confidence artifact is not a dictionary"
        )

    exact_coordinate_keys = (
        "source_points_2d_render",
        "source_points_2d_video",
        "source_points_2d_latent",
        "source_points_2d_latent_continuous",
        "points_2d_render",
        "points_2d_video",
        "points_2d_latent",
        "points_2d_latent_continuous",
    )

    validity_keys = sorted(
        key
        for key in transport
        if any(
            word in key.lower()
            for word in (
                "valid",
                "visible",
                "visibility",
                "occlusion",
                "mask",
            )
        )
    )

    identity_keys = sorted(
        key
        for key in transport
        if any(
            word in key.lower()
            for word in (
                "point_id",
                "particle",
                "material",
                "track",
            )
        )
    )

    temporal_keys = sorted(
        key
        for key in transport
        if any(
            word in key.lower()
            for word in (
                "frame",
                "time",
                "step",
            )
        )
    )

    latent_indices = artifact.get(
        "latent_frame_indices"
    )

    points_video = transport.get(
        "points_2d_video"
    )

    selected_points_video = None

    if (
        isinstance(points_video, torch.Tensor)
        and isinstance(
            latent_indices,
            torch.Tensor,
        )
        and points_video.ndim >= 3
    ):
        index = latent_indices.to(
            torch.long
        )

        if bool(
            (index >= 0).all()
            and (index < points_video.shape[0]).all()
        ):
            selected_points_video = (
                points_video[index]
            )

    frames_dir = final_sim / "frames"

    frame_files = (
        sorted(
            path
            for path in frames_dir.iterdir()
            if path.is_file()
        )
        if frames_dir.is_dir()
        else []
    )

    video_reports = {
        Path(path).stem: video_report(
            Path(path).resolve()
        )
        for path in args.videos
    }

    report = {
        "inputs": {
            "transport_ready": str(
                transport_path
            ),
            "confidence_artifact": str(
                artifact_path
            ),
            "final_sim": str(final_sim),
        },
        "transport_top_level_keys": sorted(
            transport.keys()
        ),
        "confidence_artifact_keys": sorted(
            artifact.keys()
        ),
        "coordinate_fields": {
            key: inventory(
                transport.get(key)
            )
            if key in transport
            else {
                "available": False,
            }
            for key in exact_coordinate_keys
        },
        "coordinate_bounds": {
            "source_video": coordinate_report(
                transport.get(
                    "source_points_2d_video"
                ),
                width=832,
                height=480,
            ),
            "future_video": coordinate_report(
                transport.get(
                    "points_2d_video"
                ),
                width=832,
                height=480,
            ),
            "source_latent_continuous":
                coordinate_report(
                    transport.get(
                        "source_points_2d_latent_continuous"
                    ),
                    width=104,
                    height=60,
                ),
            "future_latent_continuous":
                coordinate_report(
                    transport.get(
                        "points_2d_latent_continuous"
                    ),
                    width=104,
                    height=60,
                ),
            "selected_future_video":
                coordinate_report(
                    selected_points_video,
                    width=832,
                    height=480,
                ),
        },
        "validity_candidate_keys": {
            key: inventory(
                transport[key]
            )
            for key in validity_keys
        },
        "identity_candidate_keys": {
            key: inventory(
                transport[key]
            )
            for key in identity_keys
        },
        "temporal_candidate_keys": {
            key: inventory(
                transport[key]
            )
            for key in temporal_keys
        },
        "artifact_temporal_contract": {
            "latent_frame_indices":
                inventory(latent_indices),
            "latent_frame_indices_values": (
                latent_indices.tolist()
                if isinstance(
                    latent_indices,
                    torch.Tensor,
                )
                and latent_indices.numel() <= 100
                else None
            ),
            "transport_mask": inventory(
                artifact.get(
                    "transport_mask"
                )
            ),
            "confidence_map": inventory(
                artifact.get(
                    "confidence_map"
                )
            ),
            "confidence_ratio": inventory(
                artifact.get(
                    "confidence_ratio"
                )
            ),
        },
        "final_sim_contract": {
            "config_exists": (
                final_sim / "config.yaml"
            ).is_file(),
            "noise_exists": (
                final_sim / "noises.npy"
            ).is_file(),
            "first_frame_exists": (
                final_sim
                / "resized_input_image.png"
            ).is_file(),
            "frame_directory_exists":
                frames_dir.is_dir(),
            "frame_file_count": len(
                frame_files
            ),
            "first_frame_file": (
                str(frame_files[0])
                if frame_files
                else None
            ),
            "final_frame_file": (
                str(frame_files[-1])
                if frame_files
                else None
            ),
        },
        "generated_videos": video_reports,
        "metric_readiness": {
            "has_source_video_coordinates":
                isinstance(
                    transport.get(
                        "source_points_2d_video"
                    ),
                    torch.Tensor,
                ),
            "has_future_video_coordinates":
                isinstance(
                    transport.get(
                        "points_2d_video"
                    ),
                    torch.Tensor,
                ),
            "has_source_continuous_latent_coordinates":
                isinstance(
                    transport.get(
                        "source_points_2d_latent_continuous"
                    ),
                    torch.Tensor,
                ),
            "has_future_continuous_latent_coordinates":
                isinstance(
                    transport.get(
                        "points_2d_latent_continuous"
                    ),
                    torch.Tensor,
                ),
            "has_any_visibility_or_validity_field":
                bool(validity_keys),
            "has_material_identity_field":
                bool(identity_keys),
            "has_21_latent_indices": bool(
                isinstance(
                    latent_indices,
                    torch.Tensor,
                )
                and latent_indices.numel()
                == 21
            ),
            "all_videos_have_81_frames":
                all(
                    item.get(
                        "frame_count"
                    ) == 81
                    for item in video_reports.values()
                ),
            "all_videos_are_480x832": all(
                item.get(
                    "first_shape"
                ) == [480, 832, 3]
                for item in video_reports.values()
            ),
        },
        "interpretation_boundary": (
            "This is a read-only contract audit. "
            "It does not compute or validate MIC, TAE, "
            "or Warp Consistency scores."
        ),
    }

    output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
