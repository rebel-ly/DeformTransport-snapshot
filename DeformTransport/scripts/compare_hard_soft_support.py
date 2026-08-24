"""Compare hard and bilinear soft transport support on transport-ready data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from deform_transport.hard_transport import hard_point_transport
from deform_transport.soft_transport import soft_point_transport
from deform_transport.transport_ready import validate_transport_ready


def parse_args():
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    state = torch.load(
        args.transport_ready,
        map_location="cpu",
        weights_only=False,
    )
    validate_transport_ready(state)

    source_uv = state["source_points_2d_latent"].to(torch.float32)
    all_target_uv = state["points_2d_latent"].to(torch.float32)

    indices = torch.arange(
        0,
        all_target_uv.shape[0],
        args.temporal_stride,
        dtype=torch.long,
    )

    if indices[-1].item() != all_target_uv.shape[0] - 1:
        indices = torch.cat(
            [
                indices,
                torch.tensor(
                    [all_target_uv.shape[0] - 1],
                    dtype=torch.long,
                ),
            ]
        )

    target_uv = all_target_uv[indices]

    source_visible = state["source_visible"].to(torch.bool)
    source_valid = state["source_valid"].to(torch.bool)
    target_valid = state["projection_valid"][indices].to(torch.bool)
    point_id = state["point_id"].to(torch.long)
    object_id = state["object_id"].to(torch.long)

    height = 60
    width = 104

    y = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    source_grid = (100.0 * y + x).unsqueeze(0)

    hard = hard_point_transport(
        source_grid=source_grid,
        source_uv=source_uv,
        target_uv=target_uv,
        source_visible=source_visible,
        source_valid=source_valid,
        target_valid=target_valid,
        point_id=point_id,
        object_id=object_id,
        mode="correct",
        seed=0,
    )

    soft = soft_point_transport(
        source_grid=source_grid,
        source_uv=source_uv,
        target_uv=target_uv,
        source_visible=source_visible,
        source_valid=source_valid,
        target_valid=target_valid,
        point_id=point_id,
        object_id=object_id,
        mode="correct",
        seed=0,
    )

    rows = []

    for slot in range(target_uv.shape[0]):
        hard_mask = hard["transport_mask"][slot, 0]
        soft_mask = soft["transport_mask"][slot, 0]
        weights = soft["transport_weight"][slot, 0]

        intersection = hard_mask & soft_mask
        union = hard_mask | soft_mask
        soft_only = soft_mask & ~hard_mask

        hard_cells = int(hard_mask.sum())
        soft_cells = int(soft_mask.sum())
        soft_only_cells = int(soft_only.sum())

        rows.append(
            {
                "latent_slot": slot,
                "pixel_frame_index": int(indices[slot]),
                "valid_points_hard": int(
                    hard["valid_point_mask"][slot].sum()
                ),
                "valid_points_soft": int(
                    soft["valid_point_mask"][slot].sum()
                ),
                "hard_support_cells": hard_cells,
                "soft_support_cells": soft_cells,
                "soft_over_hard_ratio": (
                    float(soft_cells / hard_cells)
                    if hard_cells > 0
                    else None
                ),
                "mask_iou": (
                    float(intersection.sum() / union.sum())
                    if bool(union.any())
                    else 1.0
                ),
                "soft_only_cells": soft_only_cells,
                "soft_only_weight_mean": (
                    float(weights[soft_only].mean())
                    if bool(soft_only.any())
                    else 0.0
                ),
                "soft_only_weight_below_025": (
                    int((weights[soft_only] < 0.25).sum())
                    if bool(soft_only.any())
                    else 0
                ),
                "soft_only_weight_below_050": (
                    int((weights[soft_only] < 0.50).sum())
                    if bool(soft_only.any())
                    else 0
                ),
                "soft_weight_mean_on_support": (
                    float(weights[soft_mask].mean())
                    if bool(soft_mask.any())
                    else 0.0
                ),
                "soft_weight_max": float(weights.max()),
            }
        )

    csv_path = args.output_dir / "per_frame_support.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    ratios = [
        row["soft_over_hard_ratio"]
        for row in rows
        if row["soft_over_hard_ratio"] is not None
    ]

    summary = {
        "transport_ready": str(args.transport_ready.resolve()),
        "latent_indices": indices.tolist(),
        "latent_slot_count": len(rows),
        "point_count": int(point_id.numel()),
        "source_visible_count": int(source_visible.sum()),
        "mean_soft_over_hard_ratio": float(
            sum(ratios) / len(ratios)
        ),
        "max_soft_over_hard_ratio": float(max(ratios)),
        "min_mask_iou": float(
            min(row["mask_iou"] for row in rows)
        ),
        "valid_point_counts_equal": all(
            row["valid_points_hard"] == row["valid_points_soft"]
            for row in rows
        ),
        "hard_finite": bool(
            torch.isfinite(hard["transported_grid"]).all()
        ),
        "soft_finite": bool(
            torch.isfinite(soft["transported_grid"]).all()
        ),
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
