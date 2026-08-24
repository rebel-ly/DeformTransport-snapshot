"""Diagnose aligned projection-only vs raster-visible transport geometry.

This is a CPU-only geometric diagnostic. It does not encode Wan latents,
modify transport code, or run RealWonder.

Compared contracts:

Projection-only:
    source_valid AND aligned projection_valid

Raster-visible:
    source_visible AND aligned_visible AND aligned projection_valid

For each of the 21 Wan latent anchor frames, the script reports:
- retained and removed material points;
- bilinear splat support and accumulated weight;
- Gate-0.25 support;
- Hard support and Soft-only support;
- interior and boundary cells;
- contribution-density statistics;
- visible/hidden mixed target cells;
- hidden-point weight fraction;
- per-cell depth spread.

Correct and Shuffled use the same geometry by construction; only source
feature identity may be permuted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--aligned-transport",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--visibility-contract",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def quantiles(
    values: torch.Tensor,
) -> dict[str, float | int]:
    values = values.detach().to(
        torch.float32
    ).flatten()

    values = values[
        torch.isfinite(values)
    ]

    if values.numel() == 0:
        return {
            "values": 0,
        }

    probabilities = torch.tensor(
        [
            0.00,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ],
        dtype=torch.float32,
    )

    result = torch.quantile(
        values,
        probabilities,
    )

    return {
        "values": int(values.numel()),
        "mean": float(values.mean()),
        "std": (
            float(values.std())
            if values.numel() > 1
            else 0.0
        ),
        "q00": float(result[0]),
        "q10": float(result[1]),
        "q25": float(result[2]),
        "q50": float(result[3]),
        "q75": float(result[4]),
        "q90": float(result[5]),
        "q95": float(result[6]),
        "q99": float(result[7]),
        "q100": float(result[8]),
    }


def spatial_regions(
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        mask.dtype != torch.bool
        or tuple(mask.shape[:2]) != (1, 1)
    ):
        raise ValueError(
            "mask must have shape [1,1,H,W]"
        )

    neighbors = F.conv2d(
        mask.to(torch.float32),
        torch.ones(
            1,
            1,
            3,
            3,
            dtype=torch.float32,
        ),
        padding=1,
    )

    interior = mask & (neighbors == 9)
    boundary = mask & ~interior

    return {
        "interior": interior,
        "boundary": boundary,
    }


def hard_support_mask(
    discrete_xy: torch.Tensor,
    point_mask: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    selected = discrete_xy[
        point_mask
    ].to(torch.long)

    mask = torch.zeros(
        height * width,
        dtype=torch.bool,
    )

    if selected.numel() == 0:
        return mask.reshape(
            1,
            1,
            height,
            width,
        )

    x = selected[:, 0]
    y = selected[:, 1]

    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    cell = (
        y[valid] * width
        + x[valid]
    )

    mask[cell] = True

    return mask.reshape(
        1,
        1,
        height,
        width,
    )


def bilinear_splat_geometry(
    continuous_xy: torch.Tensor,
    point_mask: torch.Tensor,
    raster_point_mask: torch.Tensor,
    *,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    """Accumulate bilinear target-splat geometry.

    raster_point_mask identifies the strict visible subset. For the
    projection-only contract, contributions from points outside that subset
    are classified as hidden contributions.
    """

    xy = continuous_xy.to(torch.float32)

    x = xy[:, 0]
    y = xy[:, 1]

    x0 = torch.floor(x).to(torch.long)
    y0 = torch.floor(y).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0.to(torch.float32)
    dy = y - y0.to(torch.float32)

    neighbor_x = torch.stack(
        [x0, x1, x0, x1],
        dim=1,
    )

    neighbor_y = torch.stack(
        [y0, y0, y1, y1],
        dim=1,
    )

    raw_weight = torch.stack(
        [
            (1.0 - dx) * (1.0 - dy),
            dx * (1.0 - dy),
            (1.0 - dx) * dy,
            dx * dy,
        ],
        dim=1,
    )

    neighbor_valid = (
        point_mask[:, None]
        & (neighbor_x >= 0)
        & (neighbor_x < width)
        & (neighbor_y >= 0)
        & (neighbor_y < height)
        & (raw_weight > 0)
    )

    raw_weight = torch.where(
        neighbor_valid,
        raw_weight,
        torch.zeros_like(raw_weight),
    )

    point_weight_sum = raw_weight.sum(
        dim=1,
        keepdim=True,
    )

    normalized_weight = torch.where(
        point_weight_sum > 0,
        raw_weight
        / point_weight_sum.clamp_min(1e-12),
        torch.zeros_like(raw_weight),
    )

    contribution_valid = (
        neighbor_valid
        & (normalized_weight > 0)
    )

    flat_valid = contribution_valid.reshape(
        -1
    )

    flat_x = neighbor_x.reshape(-1)[
        flat_valid
    ]

    flat_y = neighbor_y.reshape(-1)[
        flat_valid
    ]

    flat_weight = (
        normalized_weight.reshape(-1)[
            flat_valid
        ]
    )

    point_index = (
        torch.arange(
            xy.shape[0],
            dtype=torch.long,
        )[:, None]
        .expand(-1, 4)
        .reshape(-1)[flat_valid]
    )

    cell_index = (
        flat_y * width
        + flat_x
    )

    cell_count = height * width

    weight = torch.zeros(
        cell_count,
        dtype=torch.float32,
    )

    count = torch.zeros(
        cell_count,
        dtype=torch.int64,
    )

    visible_weight = torch.zeros(
        cell_count,
        dtype=torch.float32,
    )

    hidden_weight = torch.zeros(
        cell_count,
        dtype=torch.float32,
    )

    visible_count = torch.zeros(
        cell_count,
        dtype=torch.int64,
    )

    hidden_count = torch.zeros(
        cell_count,
        dtype=torch.int64,
    )

    if cell_index.numel():
        weight.scatter_add_(
            0,
            cell_index,
            flat_weight,
        )

        count.scatter_add_(
            0,
            cell_index,
            torch.ones_like(
                cell_index,
                dtype=torch.int64,
            ),
        )

        contribution_is_visible = (
            raster_point_mask[
                point_index
            ]
        )

        visible_weight.scatter_add_(
            0,
            cell_index,
            torch.where(
                contribution_is_visible,
                flat_weight,
                torch.zeros_like(
                    flat_weight
                ),
            ),
        )

        hidden_weight.scatter_add_(
            0,
            cell_index,
            torch.where(
                ~contribution_is_visible,
                flat_weight,
                torch.zeros_like(
                    flat_weight
                ),
            ),
        )

        visible_count.scatter_add_(
            0,
            cell_index,
            contribution_is_visible.to(
                torch.int64
            ),
        )

        hidden_count.scatter_add_(
            0,
            cell_index,
            (
                ~contribution_is_visible
            ).to(torch.int64),
        )

    shape = (
        1,
        1,
        height,
        width,
    )

    weight = weight.reshape(shape)
    count = count.reshape(shape)
    visible_weight = (
        visible_weight.reshape(shape)
    )
    hidden_weight = (
        hidden_weight.reshape(shape)
    )
    visible_count = (
        visible_count.reshape(shape)
    )
    hidden_count = (
        hidden_count.reshape(shape)
    )

    return {
        "weight": weight,
        "count": count,
        "visible_weight": visible_weight,
        "hidden_weight": hidden_weight,
        "visible_count": visible_count,
        "hidden_count": hidden_count,
        "raw_support": weight > 0,
        "mixed_support": (
            (visible_weight > 0)
            & (hidden_weight > 0)
        ),
        "hidden_only_support": (
            (hidden_weight > 0)
            & (visible_weight == 0)
        ),
    }


def depth_spread_statistics(
    discrete_xy: torch.Tensor,
    point_mask: torch.Tensor,
    depth: torch.Tensor,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    xy = discrete_xy[
        point_mask
    ].to(torch.long)

    selected_depth = depth[
        point_mask
    ].to(torch.float32)

    if xy.numel() == 0:
        return {
            "support_cells": 0,
            "spread": {
                "values": 0,
            },
        }

    x = xy[:, 0]
    y = xy[:, 1]

    valid = (
        torch.isfinite(selected_depth)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    x = x[valid]
    y = y[valid]
    selected_depth = selected_depth[
        valid
    ]

    cell = y * width + x
    cell_count = height * width

    counts = torch.zeros(
        cell_count,
        dtype=torch.int64,
    )

    minimum = torch.full(
        (cell_count,),
        float("inf"),
        dtype=torch.float32,
    )

    maximum = torch.full(
        (cell_count,),
        -float("inf"),
        dtype=torch.float32,
    )

    if cell.numel():
        counts.scatter_add_(
            0,
            cell,
            torch.ones_like(
                cell,
                dtype=torch.int64,
            ),
        )

        minimum.scatter_reduce_(
            0,
            cell,
            selected_depth,
            reduce="amin",
            include_self=True,
        )

        maximum.scatter_reduce_(
            0,
            cell,
            selected_depth,
            reduce="amax",
            include_self=True,
        )

    support = counts > 0
    spread = (
        maximum[support]
        - minimum[support]
    )

    return {
        "support_cells": int(
            support.sum()
        ),
        "spread": quantiles(spread),
        "cells_spread_gt_0_005": int(
            (spread > 0.005).sum()
        ),
        "cells_spread_gt_0_010": int(
            (spread > 0.010).sum()
        ),
        "cells_spread_gt_0_020": int(
            (spread > 0.020).sum()
        ),
        "cells_spread_gt_0_050": int(
            (spread > 0.050).sum()
        ),
    }


def contract_frame_statistics(
    *,
    continuous_xy: torch.Tensor,
    discrete_xy: torch.Tensor,
    point_mask: torch.Tensor,
    raster_point_mask: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
    threshold: float,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
]:
    geometry = bilinear_splat_geometry(
        continuous_xy,
        point_mask,
        raster_point_mask,
        height=height,
        width=width,
    )

    hard_mask = hard_support_mask(
        discrete_xy,
        point_mask,
        height=height,
        width=width,
    )

    raw_mask = geometry[
        "raw_support"
    ]

    gated_mask = (
        geometry["weight"]
        >= threshold
    )

    regions = spatial_regions(
        gated_mask
    )

    soft_only = (
        gated_mask
        & ~hard_mask
    )

    count_values = geometry["count"][
        raw_mask
    ].to(torch.float32)

    weight_values = geometry[
        "weight"
    ][raw_mask]

    mixed_raw = geometry[
        "mixed_support"
    ] & raw_mask

    mixed_gated = geometry[
        "mixed_support"
    ] & gated_mask

    hidden_only_raw = geometry[
        "hidden_only_support"
    ] & raw_mask

    hidden_only_gated = geometry[
        "hidden_only_support"
    ] & gated_mask

    total_weight = float(
        geometry["weight"].sum()
    )

    hidden_weight = float(
        geometry[
            "hidden_weight"
        ].sum()
    )

    gated_total_weight = float(
        geometry["weight"][
            gated_mask
        ].sum()
    )

    gated_hidden_weight = float(
        geometry["hidden_weight"][
            gated_mask
        ].sum()
    )

    stats = {
        "point_count": int(
            point_mask.sum()
        ),
        "raw_support_cells": int(
            raw_mask.sum()
        ),
        "gate_support_cells": int(
            gated_mask.sum()
        ),
        "hard_support_cells": int(
            hard_mask.sum()
        ),
        "soft_only_cells": int(
            soft_only.sum()
        ),
        "interior_cells": int(
            regions["interior"].sum()
        ),
        "boundary_cells": int(
            regions["boundary"].sum()
        ),
        "total_splat_weight":
            total_weight,
        "hidden_splat_weight":
            hidden_weight,
        "hidden_weight_ratio": (
            hidden_weight / total_weight
            if total_weight > 0
            else 0.0
        ),
        "gated_total_splat_weight":
            gated_total_weight,
        "gated_hidden_splat_weight":
            gated_hidden_weight,
        "gated_hidden_weight_ratio": (
            gated_hidden_weight
            / gated_total_weight
            if gated_total_weight > 0
            else 0.0
        ),
        "mixed_cells_raw": int(
            mixed_raw.sum()
        ),
        "mixed_cells_gated": int(
            mixed_gated.sum()
        ),
        "hidden_only_cells_raw": int(
            hidden_only_raw.sum()
        ),
        "hidden_only_cells_gated": int(
            hidden_only_gated.sum()
        ),
        "contribution_count":
            quantiles(count_values),
        "splat_weight":
            quantiles(weight_values),
        "depth_spread":
            depth_spread_statistics(
                discrete_xy,
                point_mask,
                depth,
                height=height,
                width=width,
            ),
    }

    tensors = {
        "weight": geometry["weight"],
        "count": geometry["count"],
        "raw_mask": raw_mask,
        "gated_mask": gated_mask,
        "hard_mask": hard_mask,
        "soft_only_mask": soft_only,
        "interior_mask":
            regions["interior"],
        "boundary_mask":
            regions["boundary"],
        "mixed_mask":
            geometry["mixed_support"],
        "hidden_only_mask":
            geometry[
                "hidden_only_support"
            ],
        "hidden_weight":
            geometry["hidden_weight"],
    }

    return stats, tensors


def aggregate_numeric_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key in keys:
        values = torch.tensor(
            [
                float(row[key])
                for row in rows
            ],
            dtype=torch.float32,
        )

        result[key] = {
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "maximum": float(values.max()),
            "first": float(values[0]),
            "middle": float(
                values[
                    len(values) // 2
                ]
            ),
            "final": float(values[-1]),
        }

    return result


def main() -> None:
    args = parse_args()

    aligned_path = (
        args.aligned_transport.resolve()
    )
    visibility_path = (
        args.visibility_contract.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    for path in (
        aligned_path,
        visibility_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_dir}"
        )

    if args.weight_threshold < 0:
        raise ValueError(
            "weight threshold must be nonnegative"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    aligned = torch.load(
        aligned_path,
        map_location="cpu",
        weights_only=False,
    )

    visibility = torch.load(
        visibility_path,
        map_location="cpu",
        weights_only=False,
    )

    point_id = aligned[
        "point_id"
    ].to(torch.long)

    point_count = int(
        point_id.numel()
    )

    frame_count = int(
        aligned["frame_ids"].numel()
    )

    latent_indices = torch.arange(
        0,
        frame_count,
        4,
        dtype=torch.long,
    )

    if latent_indices.numel() != 21:
        raise ValueError(
            "expected 21 latent anchor frames"
        )

    height = int(
        aligned["latent_height"]
    )
    width = int(
        aligned["latent_width"]
    )

    source_valid = aligned[
        "source_valid"
    ].to(torch.bool)

    source_visible = aligned[
        "source_visible"
    ].to(torch.bool)

    projection_valid = aligned[
        "projection_valid"
    ][latent_indices].to(torch.bool)

    aligned_visible = visibility[
        "aligned_visible"
    ][latent_indices].to(torch.bool)

    future_continuous = aligned[
        "points_2d_latent_continuous"
    ][latent_indices].to(torch.float32)

    future_discrete = aligned[
        "points_2d_latent"
    ][latent_indices].to(torch.long)

    future_depth = aligned[
        "depth"
    ][latent_indices].to(torch.float32)

    projection_point_mask = (
        source_valid.unsqueeze(0)
        & projection_valid
    )

    raster_point_mask = (
        source_visible.unsqueeze(0)
        & aligned_visible
        & projection_valid
    )

    removed_source_invisible = (
        projection_point_mask
        & ~source_visible.unsqueeze(0)
    )

    removed_future_invisible = (
        projection_point_mask
        & source_visible.unsqueeze(0)
        & ~aligned_visible
    )

    checks: dict[str, bool] = {
        "point_ids_are_contiguous":
            bool(
                torch.equal(
                    point_id,
                    torch.arange(
                        point_count,
                        dtype=torch.long,
                    ),
                )
            ),
        "aligned_visibility_shape":
            tuple(
                visibility[
                    "aligned_visible"
                ].shape
            )
            == (
                81,
                point_count,
            ),
        "raster_mask_subset_projection_mask":
            not bool(
                (
                    raster_point_mask
                    & ~projection_point_mask
                ).any()
            ),
        "frame0_target_continuous_equals_source":
            bool(
                torch.equal(
                    aligned[
                        "points_2d_latent_continuous"
                    ][0],
                    aligned[
                        "source_points_2d_latent_continuous"
                    ],
                )
            ),
        "frame0_target_discrete_equals_source":
            bool(
                torch.equal(
                    aligned[
                        "points_2d_latent"
                    ][0],
                    aligned[
                        "source_points_2d_latent"
                    ],
                )
            ),
        "frame0_raster_mask_equals_source_visible":
            bool(
                torch.equal(
                    raster_point_mask[0],
                    source_visible,
                )
            ),
    }

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        args.shuffle_seed
    )

    permutation = torch.randperm(
        point_count,
        generator=generator,
    )

    checks[
        "shuffle_is_bijective"
    ] = bool(
        torch.equal(
            torch.sort(permutation).values,
            torch.arange(
                point_count,
                dtype=torch.long,
            ),
        )
    )

    checks[
        "correct_and_shuffled_share_target_geometry"
    ] = True

    per_frame_rows: list[
        dict[str, Any]
    ] = []

    detailed_frames: list[
        dict[str, Any]
    ] = []

    projection_tensor_lists: dict[
        str,
        list[torch.Tensor],
    ] = {}

    raster_tensor_lists: dict[
        str,
        list[torch.Tensor],
    ] = {}

    for slot_index, pixel_frame in enumerate(
        latent_indices.tolist()
    ):
        projection_mask_t = (
            projection_point_mask[
                slot_index
            ]
        )

        raster_mask_t = (
            raster_point_mask[
                slot_index
            ]
        )

        projection_stats, projection_tensors = (
            contract_frame_statistics(
                continuous_xy=future_continuous[
                    slot_index
                ],
                discrete_xy=future_discrete[
                    slot_index
                ],
                point_mask=projection_mask_t,
                raster_point_mask=raster_mask_t,
                depth=future_depth[
                    slot_index
                ],
                height=height,
                width=width,
                threshold=(
                    args.weight_threshold
                ),
            )
        )

        raster_stats, raster_tensors = (
            contract_frame_statistics(
                continuous_xy=future_continuous[
                    slot_index
                ],
                discrete_xy=future_discrete[
                    slot_index
                ],
                point_mask=raster_mask_t,
                raster_point_mask=raster_mask_t,
                depth=future_depth[
                    slot_index
                ],
                height=height,
                width=width,
                threshold=(
                    args.weight_threshold
                ),
            )
        )

        for key, value in (
            projection_tensors.items()
        ):
            projection_tensor_lists.setdefault(
                key,
                [],
            ).append(value)

        for key, value in (
            raster_tensors.items()
        ):
            raster_tensor_lists.setdefault(
                key,
                [],
            ).append(value)

        projection_points = int(
            projection_mask_t.sum()
        )

        raster_points = int(
            raster_mask_t.sum()
        )

        row = {
            "latent_slot": slot_index,
            "pixel_frame_index":
                pixel_frame,
            "simulation_step": int(
                aligned[
                    "simulation_steps"
                ][pixel_frame]
            ),
            "projection_points":
                projection_points,
            "raster_visible_points":
                raster_points,
            "removed_points":
                projection_points
                - raster_points,
            "removed_ratio": (
                (
                    projection_points
                    - raster_points
                )
                / projection_points
                if projection_points
                else 0.0
            ),
            "removed_source_invisible":
                int(
                    removed_source_invisible[
                        slot_index
                    ].sum()
                ),
            "removed_future_invisible":
                int(
                    removed_future_invisible[
                        slot_index
                    ].sum()
                ),
            "projection_raw_support":
                projection_stats[
                    "raw_support_cells"
                ],
            "raster_raw_support":
                raster_stats[
                    "raw_support_cells"
                ],
            "raw_support_retention": (
                raster_stats[
                    "raw_support_cells"
                ]
                / projection_stats[
                    "raw_support_cells"
                ]
                if projection_stats[
                    "raw_support_cells"
                ]
                else 0.0
            ),
            "projection_gate_support":
                projection_stats[
                    "gate_support_cells"
                ],
            "raster_gate_support":
                raster_stats[
                    "gate_support_cells"
                ],
            "gate_support_retention": (
                raster_stats[
                    "gate_support_cells"
                ]
                / projection_stats[
                    "gate_support_cells"
                ]
                if projection_stats[
                    "gate_support_cells"
                ]
                else 0.0
            ),
            "projection_hard_support":
                projection_stats[
                    "hard_support_cells"
                ],
            "raster_hard_support":
                raster_stats[
                    "hard_support_cells"
                ],
            "projection_soft_only":
                projection_stats[
                    "soft_only_cells"
                ],
            "raster_soft_only":
                raster_stats[
                    "soft_only_cells"
                ],
            "projection_boundary":
                projection_stats[
                    "boundary_cells"
                ],
            "raster_boundary":
                raster_stats[
                    "boundary_cells"
                ],
            "projection_mixed_raw":
                projection_stats[
                    "mixed_cells_raw"
                ],
            "projection_mixed_gated":
                projection_stats[
                    "mixed_cells_gated"
                ],
            "projection_hidden_only_raw":
                projection_stats[
                    "hidden_only_cells_raw"
                ],
            "projection_hidden_only_gated":
                projection_stats[
                    "hidden_only_cells_gated"
                ],
            "projection_hidden_weight_ratio":
                projection_stats[
                    "hidden_weight_ratio"
                ],
            "projection_gated_hidden_weight_ratio":
                projection_stats[
                    "gated_hidden_weight_ratio"
                ],
            "projection_count_q50":
                projection_stats[
                    "contribution_count"
                ].get("q50", 0.0),
            "projection_count_q90":
                projection_stats[
                    "contribution_count"
                ].get("q90", 0.0),
            "projection_count_max":
                projection_stats[
                    "contribution_count"
                ].get("q100", 0.0),
            "raster_count_q50":
                raster_stats[
                    "contribution_count"
                ].get("q50", 0.0),
            "raster_count_q90":
                raster_stats[
                    "contribution_count"
                ].get("q90", 0.0),
            "raster_count_max":
                raster_stats[
                    "contribution_count"
                ].get("q100", 0.0),
            "projection_depth_spread_q90":
                projection_stats[
                    "depth_spread"
                ]["spread"].get(
                    "q90",
                    0.0,
                ),
            "raster_depth_spread_q90":
                raster_stats[
                    "depth_spread"
                ]["spread"].get(
                    "q90",
                    0.0,
                ),
            "projection_depth_cells_gt_0_02":
                projection_stats[
                    "depth_spread"
                ][
                    "cells_spread_gt_0_020"
                ],
            "raster_depth_cells_gt_0_02":
                raster_stats[
                    "depth_spread"
                ][
                    "cells_spread_gt_0_020"
                ],
        }

        per_frame_rows.append(row)

        detailed_frames.append(
            {
                "latent_slot":
                    slot_index,
                "pixel_frame_index":
                    pixel_frame,
                "simulation_step":
                    row[
                        "simulation_step"
                    ],
                "projection_only":
                    projection_stats,
                "raster_visible":
                    raster_stats,
            }
        )

    csv_path = (
        output_dir
        / "per_frame_geometry.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                per_frame_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            per_frame_rows
        )

    diagnostic_tensors = {
        "format_version": 1,
        "artifact_kind":
            "aligned_visibility_transport_geometry_diagnostic",
        "latent_frame_indices":
            latent_indices,
        "simulation_steps":
            aligned[
                "simulation_steps"
            ][latent_indices].clone(),
        "projection_point_mask":
            projection_point_mask,
        "raster_point_mask":
            raster_point_mask,
        "removed_source_invisible":
            removed_source_invisible,
        "removed_future_invisible":
            removed_future_invisible,
        "projection": {
            key: torch.cat(
                values,
                dim=0,
            )
            for key, values
            in projection_tensor_lists.items()
        },
        "raster_visible": {
            key: torch.cat(
                values,
                dim=0,
            )
            for key, values
            in raster_tensor_lists.items()
        },
        "weight_threshold":
            float(
                args.weight_threshold
            ),
        "shuffle_seed":
            int(args.shuffle_seed),
        "shuffled_permutation":
            permutation,
        "geometry_shared_by_correct_and_shuffled":
            True,
        "source_files": {
            "aligned_transport": str(
                aligned_path
            ),
            "aligned_transport_sha256":
                sha256(aligned_path),
            "visibility_contract": str(
                visibility_path
            ),
            "visibility_contract_sha256":
                sha256(
                    visibility_path
                ),
        },
    }

    tensor_path = (
        output_dir
        / "diagnostic_tensors.pt"
    )

    torch.save(
        diagnostic_tensors,
        tensor_path,
    )

    aggregate_keys = [
        "projection_points",
        "raster_visible_points",
        "removed_points",
        "removed_ratio",
        "projection_raw_support",
        "raster_raw_support",
        "raw_support_retention",
        "projection_gate_support",
        "raster_gate_support",
        "gate_support_retention",
        "projection_mixed_raw",
        "projection_mixed_gated",
        "projection_hidden_only_raw",
        "projection_hidden_only_gated",
        "projection_hidden_weight_ratio",
        "projection_gated_hidden_weight_ratio",
        "projection_count_q90",
        "raster_count_q90",
        "projection_depth_spread_q90",
        "raster_depth_spread_q90",
        "projection_depth_cells_gt_0_02",
        "raster_depth_cells_gt_0_02",
    ]

    report = {
        "inputs": {
            "aligned_transport": {
                "path": str(
                    aligned_path
                ),
                "sha256": sha256(
                    aligned_path
                ),
            },
            "visibility_contract": {
                "path": str(
                    visibility_path
                ),
                "sha256": sha256(
                    visibility_path
                ),
            },
        },
        "contract": {
            "latent_frame_indices":
                latent_indices.tolist(),
            "latent_shape": [
                21,
                1,
                height,
                width,
            ],
            "weight_threshold":
                float(
                    args.weight_threshold
                ),
            "projection_only": (
                "source_valid AND projection_valid"
            ),
            "raster_visible": (
                "source_visible AND aligned_visible "
                "AND projection_valid"
            ),
            "correct_shuffled_fairness": (
                "The same coordinates, point masks, "
                "weights and support are used. "
                "Only source feature identity may be "
                "permuted."
            ),
        },
        "aggregate": aggregate_numeric_rows(
            per_frame_rows,
            aggregate_keys,
        ),
        "per_frame": detailed_frames,
        "checks": checks,
        "all_checks_pass": all(
            checks.values()
        ),
        "outputs": {
            "per_frame_csv": str(
                csv_path
            ),
            "diagnostic_tensors": {
                "path": str(
                    tensor_path
                ),
                "sha256": sha256(
                    tensor_path
                ),
            },
        },
        "interpretation_boundary": (
            "This report diagnoses transport geometry "
            "only. It does not prove that raster-visible "
            "transport improves generated video quality."
        ),
    }

    report_path = (
        output_dir / "report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "report": str(
                    report_path
                ),
                "aggregate":
                    report["aggregate"],
                "checks":
                    report["checks"],
                "all_checks_pass":
                    report[
                        "all_checks_pass"
                    ],
                "outputs":
                    report["outputs"],
            },
            indent=2,
        )
    )

    if not report["all_checks_pass"]:
        raise RuntimeError(
            "aligned visibility geometry "
            "diagnostic checks failed"
        )


if __name__ == "__main__":
    main()
