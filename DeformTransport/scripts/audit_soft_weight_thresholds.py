"""Audit confidence thresholds for continuous-coordinate soft transport.

The audit compares the legacy hard support with the continuous bilinear support.
It does not modify or generate any latent artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from deform_transport.hard_transport import hard_point_transport
from deform_transport.soft_transport import soft_point_transport
from deform_transport.transport_ready import validate_transport_ready


DEFAULT_THRESHOLDS = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport-ready",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=4,
    )
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def main() -> None:
    args = parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite output directory: {args.output_dir}"
        )

    args.output_dir.mkdir(parents=True)

    state = torch.load(
        args.transport_ready,
        map_location="cpu",
        weights_only=False,
    )
    validate_transport_ready(state)

    required = {
        "source_points_2d_latent_continuous",
        "points_2d_latent_continuous",
    }

    missing = sorted(required - set(state))
    if missing:
        raise ValueError(
            f"continuous coordinate fields are missing: {missing}"
        )

    target_all_discrete = state[
        "points_2d_latent"
    ].to(torch.float32)

    target_all_continuous = state[
        "points_2d_latent_continuous"
    ].to(torch.float32)

    indices = torch.arange(
        0,
        target_all_discrete.shape[0],
        args.temporal_stride,
        dtype=torch.long,
    )

    if indices[-1].item() != target_all_discrete.shape[0] - 1:
        indices = torch.cat(
            [
                indices,
                torch.tensor(
                    [target_all_discrete.shape[0] - 1],
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

    target_discrete = target_all_discrete[indices]
    target_continuous = target_all_continuous[indices]

    source_visible = state["source_visible"].to(torch.bool)
    source_valid = state["source_valid"].to(torch.bool)
    target_valid = state["projection_valid"][indices].to(torch.bool)
    point_id = state["point_id"].to(torch.long)
    object_id = state["object_id"].to(torch.long)

    height = int(state["latent_height"])
    width = int(state["latent_width"])

    # Feature values are irrelevant for this support/weight audit.
    source_grid = torch.ones(
        (1, height, width),
        dtype=torch.float32,
    )

    common = {
        "source_grid": source_grid,
        "source_visible": source_visible,
        "source_valid": source_valid,
        "target_valid": target_valid,
        "point_id": point_id,
        "object_id": object_id,
        "mode": "correct",
        "seed": 0,
    }

    hard = hard_point_transport(
        source_uv=source_discrete,
        target_uv=target_discrete,
        **common,
    )

    soft = soft_point_transport(
        source_uv=source_continuous,
        target_uv=target_continuous,
        **common,
    )

    if not torch.equal(
        hard["valid_point_mask"],
        soft["valid_point_mask"],
    ):
        raise RuntimeError(
            "hard and soft transports used different valid point sets"
        )

    hard_mask = hard["transport_mask"][:, 0]
    soft_mask = soft["transport_mask"][:, 0]
    weights = soft["transport_weight"][:, 0]

    soft_only_mask = soft_mask & ~hard_mask
    overlap_mask = soft_mask & hard_mask

    soft_only_weights = weights[soft_only_mask]
    overlap_weights = weights[overlap_mask]

    if soft_only_weights.numel() == 0:
        raise RuntimeError("soft transport produced no soft-only support")

    quantile_levels = torch.tensor(
        [
            0.0,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.0,
        ],
        dtype=torch.float32,
    )

    global_summary = {
        "transport_ready": str(args.transport_ready.resolve()),
        "selected_pixel_frames": indices.tolist(),
        "hard_support_cells": int(hard_mask.sum()),
        "soft_support_cells": int(soft_mask.sum()),
        "soft_only_cells": int(soft_only_mask.sum()),
        "overlap_cells": int(overlap_mask.sum()),
        "soft_over_hard_ratio": safe_ratio(
            int(soft_mask.sum()),
            int(hard_mask.sum()),
        ),
        "soft_only_weight_sum": float(
            soft_only_weights.sum()
        ),
        "soft_only_weight_mean": float(
            soft_only_weights.mean()
        ),
        "soft_only_weight_quantiles": torch.quantile(
            soft_only_weights.to(torch.float32),
            quantile_levels,
        ).tolist(),
        "overlap_weight_quantiles": (
            torch.quantile(
                overlap_weights.to(torch.float32),
                quantile_levels,
            ).tolist()
            if overlap_weights.numel()
            else []
        ),
        "quantile_levels": quantile_levels.tolist(),
    }

    threshold_rows = []
    per_frame_rows = []

    full_soft_only_weight = float(soft_only_weights.sum())
    full_soft_only_cells = int(soft_only_mask.sum())

    for threshold in DEFAULT_THRESHOLDS:
        threshold_mask = weights >= threshold

        # Preserve all legacy Hard cells. Threshold only controls the new fringe.
        gated_mask = hard_mask | (
            soft_only_mask & threshold_mask
        )

        retained_soft_only = soft_only_mask & threshold_mask
        retained_soft_only_weights = weights[retained_soft_only]

        retained_weight_sum = float(
            retained_soft_only_weights.sum()
        )

        retained_soft_only_count = int(
            retained_soft_only.sum()
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "gated_support_cells": int(gated_mask.sum()),
                "gated_over_hard_ratio": safe_ratio(
                    int(gated_mask.sum()),
                    int(hard_mask.sum()),
                ),
                "retained_soft_only_cells": retained_soft_only_count,
                "retained_soft_only_cell_ratio": safe_ratio(
                    retained_soft_only_count,
                    full_soft_only_cells,
                ),
                "retained_soft_only_weight_sum": retained_weight_sum,
                "retained_soft_only_weight_ratio": safe_ratio(
                    retained_weight_sum,
                    full_soft_only_weight,
                ),
                "removed_soft_only_cells": (
                    full_soft_only_cells
                    - retained_soft_only_count
                ),
            }
        )

        for frame_slot in range(indices.numel()):
            frame_hard = hard_mask[frame_slot]
            frame_soft_only = soft_only_mask[frame_slot]
            frame_weights = weights[frame_slot]

            frame_retained = (
                frame_soft_only
                & (frame_weights >= threshold)
            )

            frame_gated = frame_hard | frame_retained

            frame_soft_only_weight_sum = float(
                frame_weights[frame_soft_only].sum()
            )

            per_frame_rows.append(
                {
                    "threshold": threshold,
                    "latent_slot": frame_slot,
                    "pixel_frame_index": int(indices[frame_slot]),
                    "hard_cells": int(frame_hard.sum()),
                    "full_soft_only_cells": int(
                        frame_soft_only.sum()
                    ),
                    "retained_soft_only_cells": int(
                        frame_retained.sum()
                    ),
                    "gated_support_cells": int(
                        frame_gated.sum()
                    ),
                    "gated_over_hard_ratio": safe_ratio(
                        int(frame_gated.sum()),
                        int(frame_hard.sum()),
                    ),
                    "retained_soft_only_weight_ratio": safe_ratio(
                        float(frame_weights[frame_retained].sum()),
                        frame_soft_only_weight_sum,
                    ),
                }
            )

    # Engineering selection rule:
    # choose the lowest threshold that keeps >=95% of soft-only weight
    # while limiting mean support expansion to <=8%.
    recommended = None

    for row in threshold_rows:
        if (
            row["retained_soft_only_weight_ratio"] >= 0.95
            and row["gated_over_hard_ratio"] <= 1.08
        ):
            recommended = row
            break

    summary = {
        "global": global_summary,
        "thresholds": threshold_rows,
        "selection_rule": {
            "minimum_retained_soft_only_weight_ratio": 0.95,
            "maximum_gated_over_hard_ratio": 1.08,
            "recommended_threshold": (
                recommended["threshold"]
                if recommended is not None
                else None
            ),
            "rule_satisfied": recommended is not None,
        },
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (
        args.output_dir / "threshold_summary.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(threshold_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    with (
        args.output_dir / "per_frame_thresholds.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(per_frame_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
