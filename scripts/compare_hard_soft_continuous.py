"""Compare legacy hard transport with continuous-coordinate soft transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deform_transport.hard_transport import hard_point_transport
from deform_transport.soft_transport import soft_point_transport
from deform_transport.transport_ready import validate_transport_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport-ready",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=4,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state = torch.load(
        args.transport_ready,
        map_location="cpu",
        weights_only=False,
    )

    validate_transport_ready(state)

    required_extension_keys = {
        "source_points_2d_latent_continuous",
        "points_2d_latent_continuous",
        "continuous_coordinate_extension_version",
    }

    missing = sorted(required_extension_keys - set(state))

    if missing:
        raise ValueError(
            f"continuous coordinate extension is missing: {missing}"
        )

    all_target_discrete = state[
        "points_2d_latent"
    ].to(torch.float32)

    all_target_continuous = state[
        "points_2d_latent_continuous"
    ].to(torch.float32)

    indices = torch.arange(
        0,
        all_target_discrete.shape[0],
        args.temporal_stride,
        dtype=torch.long,
    )

    if indices[-1].item() != all_target_discrete.shape[0] - 1:
        indices = torch.cat(
            [
                indices,
                torch.tensor(
                    [all_target_discrete.shape[0] - 1],
                    dtype=torch.long,
                ),
            ]
        )

    source_discrete = state[
        "source_points_2d_latent"
    ].to(torch.float32)

    source_continuous = state[
        "source_points_2d_latent_continuous"
    ].to(torch.float32)

    target_discrete = all_target_discrete[indices]
    target_continuous = all_target_continuous[indices]

    source_visible = state["source_visible"].to(torch.bool)
    source_valid = state["source_valid"].to(torch.bool)
    target_valid = state["projection_valid"][indices].to(torch.bool)
    point_id = state["point_id"].to(torch.long)
    object_id = state["object_id"].to(torch.long)

    height = int(state["latent_height"])
    width = int(state["latent_width"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260805)

    source_grid = torch.randn(
        (4, height, width),
        generator=generator,
        dtype=torch.float32,
    )

    common = {
        "source_grid": source_grid,
        "source_visible": source_visible,
        "source_valid": source_valid,
        "target_valid": target_valid,
        "point_id": point_id,
        "object_id": object_id,
        "seed": 0,
    }

    hard = hard_point_transport(
        source_uv=source_discrete,
        target_uv=target_discrete,
        mode="correct",
        **common,
    )

    soft_correct = soft_point_transport(
        source_uv=source_continuous,
        target_uv=target_continuous,
        mode="correct",
        **common,
    )

    soft_shuffled = soft_point_transport(
        source_uv=source_continuous,
        target_uv=target_continuous,
        mode="shuffled",
        **common,
    )

    hard_grid = hard["transported_grid"]
    soft_grid = soft_correct["transported_grid"]

    union_mask = (
        hard["transport_mask"]
        | soft_correct["transport_mask"]
    )

    expanded_union = union_mask.expand_as(hard_grid)

    absolute_difference = torch.abs(
        hard_grid - soft_grid
    )

    per_frame = []

    for frame_index in range(indices.numel()):
        hard_mask = hard["transport_mask"][frame_index, 0]
        soft_mask = soft_correct[
            "transport_mask"
        ][frame_index, 0]

        intersection = hard_mask & soft_mask
        union = hard_mask | soft_mask

        hard_cells = int(hard_mask.sum())
        soft_cells = int(soft_mask.sum())

        per_frame.append(
            {
                "latent_slot": frame_index,
                "pixel_frame_index": int(indices[frame_index]),
                "hard_support_cells": hard_cells,
                "soft_support_cells": soft_cells,
                "soft_over_hard_ratio": (
                    float(soft_cells / hard_cells)
                    if hard_cells
                    else None
                ),
                "mask_iou": (
                    float(
                        intersection.sum().to(torch.float32)
                        / union.sum().to(torch.float32)
                    )
                    if bool(union.any())
                    else 1.0
                ),
                "valid_points_hard": int(
                    hard["valid_point_mask"][frame_index].sum()
                ),
                "valid_points_soft": int(
                    soft_correct[
                        "valid_point_mask"
                    ][frame_index].sum()
                ),
            }
        )

    weights = soft_correct["transport_weight"]
    positive_weights = weights[
        soft_correct["transport_mask"]
    ]

    noninteger_weight_entries = int(
        (
            torch.abs(
                positive_weights
                - torch.round(positive_weights)
            )
            > 1e-6
        ).sum()
    )

    source_fractional_points = int(
        (
            torch.abs(
                source_continuous
                - torch.floor(source_continuous)
            )
            > 1e-6
        ).any(dim=-1).sum()
    )

    target_fractional_points = int(
        (
            torch.abs(
                target_continuous
                - torch.floor(target_continuous)
            )
            > 1e-6
        ).any(dim=-1).sum()
    )

    checks = {
        "hard_finite": bool(
            torch.isfinite(hard_grid).all()
        ),
        "soft_correct_finite": bool(
            torch.isfinite(soft_grid).all()
        ),
        "soft_shuffled_finite": bool(
            torch.isfinite(
                soft_shuffled["transported_grid"]
            ).all()
        ),
        "hard_soft_valid_points_equal": bool(
            torch.equal(
                hard["valid_point_mask"],
                soft_correct["valid_point_mask"],
            )
        ),
        "hard_soft_output_not_exactly_equal": bool(
            not torch.equal(hard_grid, soft_grid)
        ),
        "soft_correct_shuffled_same_mask": bool(
            torch.equal(
                soft_correct["transport_mask"],
                soft_shuffled["transport_mask"],
            )
        ),
        "soft_correct_shuffled_same_count": bool(
            torch.equal(
                soft_correct["contribution_count"],
                soft_shuffled["contribution_count"],
            )
        ),
        "soft_correct_shuffled_same_weight": bool(
            torch.allclose(
                soft_correct["transport_weight"],
                soft_shuffled["transport_weight"],
            )
        ),
        "soft_correct_shuffled_same_valid_points": bool(
            torch.equal(
                soft_correct["valid_point_mask"],
                soft_shuffled["valid_point_mask"],
            )
        ),
        "soft_correct_shuffled_output_different": bool(
            not torch.equal(
                soft_correct["transported_grid"],
                soft_shuffled["transported_grid"],
            )
        ),
        "source_has_fractional_coordinates": (
            source_fractional_points > 0
        ),
        "target_has_fractional_coordinates": (
            target_fractional_points > 0
        ),
        "soft_has_noninteger_accumulated_weights": (
            noninteger_weight_entries > 0
        ),
    }

    summary = {
        "artifact": str(
            args.transport_ready.resolve()
        ),
        "selected_pixel_frames": indices.tolist(),
        "source_fractional_points": source_fractional_points,
        "target_fractional_points": target_fractional_points,
        "noninteger_positive_weight_entries": (
            noninteger_weight_entries
        ),
        "hard_soft_output_max_abs_difference": float(
            absolute_difference.max()
        ),
        "hard_soft_output_mean_abs_difference_on_union": (
            float(
                absolute_difference[
                    expanded_union
                ].mean()
            )
            if bool(expanded_union.any())
            else 0.0
        ),
        "mean_soft_over_hard_ratio": float(
            sum(
                row["soft_over_hard_ratio"]
                for row in per_frame
                if row["soft_over_hard_ratio"] is not None
            )
            / len(per_frame)
        ),
        "minimum_mask_iou": float(
            min(row["mask_iou"] for row in per_frame)
        ),
        "soft_weight_quantiles": (
            torch.quantile(
                positive_weights.to(torch.float32),
                torch.tensor(
                    [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
                ),
            ).tolist()
            if positive_weights.numel()
            else []
        ),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "per_frame": per_frame,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=False,
    )

    args.output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    if not summary["all_checks_pass"]:
        raise RuntimeError(
            "continuous hard-versus-soft validation failed"
        )


if __name__ == "__main__":
    main()
