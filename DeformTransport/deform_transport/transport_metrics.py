"""Metrics for checkpoint-free point-transport probes."""

from __future__ import annotations

import math

import torch


def masked_frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, float | int | None]]:
    """Compute per-frame image errors only on shared spatial support."""

    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32, device=prediction.device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=prediction.device)
    if prediction.ndim != 4:
        raise ValueError("prediction must have shape [T, C, H, W]")
    if target.shape != prediction.shape:
        raise ValueError("target must match prediction shape")
    if mask.shape != (prediction.shape[0], 1, *prediction.shape[2:]):
        raise ValueError("mask must have shape [T, 1, H, W]")
    metrics = []
    channel_count = prediction.shape[1]
    for frame_index in range(prediction.shape[0]):
        spatial_count = int(mask[frame_index].sum())
        if spatial_count == 0:
            metrics.append(
                {
                    "frame": frame_index,
                    "support_cells": 0,
                    "masked_l1": None,
                    "masked_mse": None,
                    "masked_psnr_db": None,
                }
            )
            continue
        expanded_mask = mask[frame_index].expand(channel_count, -1, -1)
        difference = prediction[frame_index] - target[frame_index]
        selected = difference[expanded_mask]
        l1 = float(selected.abs().mean())
        mse = float(selected.square().mean())
        psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)
        metrics.append(
            {
                "frame": frame_index,
                "support_cells": spatial_count,
                "masked_l1": l1,
                "masked_mse": mse,
                "masked_psnr_db": psnr,
            }
        )
    return metrics


def aggregate_frame_metrics(
    frame_metrics: list[dict[str, float | int | None]],
) -> dict[str, float | int | None]:
    valid = [item for item in frame_metrics if item["masked_l1"] is not None]
    if not valid:
        return {
            "valid_frames": 0,
            "mean_masked_l1": None,
            "mean_masked_mse": None,
            "mean_masked_psnr_db": None,
        }
    return {
        "valid_frames": len(valid),
        "mean_masked_l1": sum(float(item["masked_l1"]) for item in valid)
        / len(valid),
        "mean_masked_mse": sum(float(item["masked_mse"]) for item in valid)
        / len(valid),
        "mean_masked_psnr_db": sum(
            float(item["masked_psnr_db"]) for item in valid
        )
        / len(valid),
    }


def transport_coverage_metrics(result: dict) -> dict[str, float | int | list]:
    mask = torch.as_tensor(result["transport_mask"], dtype=torch.bool)
    counts = torch.as_tensor(result["contribution_count"], dtype=torch.long)
    valid_points = torch.as_tensor(result["valid_point_mask"], dtype=torch.bool)
    occupied_per_frame = mask.flatten(1).sum(dim=1)
    coverage_per_frame = mask.float().flatten(1).mean(dim=1)
    mean_counts = []
    max_counts = []
    for frame_index in range(mask.shape[0]):
        selected = counts[frame_index][mask[frame_index]]
        mean_counts.append(float(selected.float().mean()) if selected.numel() else 0.0)
        max_counts.append(int(selected.max()) if selected.numel() else 0)
    return {
        "coverage_ratio_per_frame": [float(value) for value in coverage_per_frame],
        "occupied_target_cells_per_frame": [
            int(value) for value in occupied_per_frame
        ],
        "valid_points_per_frame": [int(value) for value in valid_points.sum(dim=1)],
        "mean_contribution_count_per_occupied_cell": mean_counts,
        "max_contribution_count_per_frame": max_counts,
        "min_coverage_ratio": float(coverage_per_frame.min()),
        "max_coverage_ratio": float(coverage_per_frame.max()),
        "max_contribution_count": max(max_counts),
        "contains_nan_or_inf": not bool(
            torch.isfinite(torch.as_tensor(result["transported_grid"])).all()
        ),
    }


def compare_correct_and_shuffled(
    correct_metrics: list[dict[str, float | int | None]],
    shuffled_metrics: list[dict[str, float | int | None]],
    *,
    tolerance: float = 1e-8,
) -> dict[str, float | int | bool]:
    if len(correct_metrics) != len(shuffled_metrics):
        raise ValueError("correct and shuffled frame metrics must have equal length")
    comparable = []
    for correct, shuffled in zip(correct_metrics, shuffled_metrics):
        if correct["masked_l1"] is None or shuffled["masked_l1"] is None:
            continue
        comparable.append(
            (float(correct["masked_l1"]), float(shuffled["masked_l1"]))
        )
    correct_better = sum(c < s - tolerance for c, s in comparable)
    correct_not_worse = sum(c <= s + tolerance for c, s in comparable)
    mean_correct = sum(c for c, _ in comparable) / max(len(comparable), 1)
    mean_shuffled = sum(s for _, s in comparable) / max(len(comparable), 1)
    return {
        "comparable_frames": len(comparable),
        "correct_better_frames": correct_better,
        "correct_not_worse_frames": correct_not_worse,
        "correct_better_fraction": correct_better / max(len(comparable), 1),
        "correct_not_worse_fraction": correct_not_worse / max(len(comparable), 1),
        "mean_correct_masked_l1": mean_correct,
        "mean_shuffled_masked_l1": mean_shuffled,
        "mean_l1_improvement": mean_shuffled - mean_correct,
        "relative_l1_improvement": (
            (mean_shuffled - mean_correct) / mean_shuffled
            if mean_shuffled > 0
            else 0.0
        ),
        "overall_correct_better": mean_correct < mean_shuffled,
    }
