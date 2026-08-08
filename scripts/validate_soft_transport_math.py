"""Synthetic mathematical validation for normalized bilinear point transport."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deform_transport.soft_transport import soft_point_transport


def run_transport(
    source_grid: torch.Tensor,
    source_uv: torch.Tensor,
    target_uv: torch.Tensor,
    *,
    mode: str = "correct",
    seed: int = 0,
):
    point_count = source_uv.shape[0]
    frame_count = target_uv.shape[0]

    return soft_point_transport(
        source_grid=source_grid,
        source_uv=source_uv,
        target_uv=target_uv,
        source_visible=torch.ones(point_count, dtype=torch.bool),
        source_valid=torch.ones(point_count, dtype=torch.bool),
        target_valid=torch.ones(
            frame_count,
            point_count,
            dtype=torch.bool,
        ),
        point_id=torch.arange(point_count),
        mode=mode,
        seed=seed,
    )


def main() -> None:
    height = 8
    width = 10

    y = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(width, dtype=torch.float32).unsqueeze(0)

    linear_grid = torch.stack(
        [
            10.0 * y + x,
            3.0 * y - 2.0 * x,
        ],
        dim=0,
    )

    source_uv = torch.tensor(
        [
            [1.25, 1.50],
            [3.75, 2.25],
            [6.40, 4.60],
            [8.20, 5.10],
        ],
        dtype=torch.float32,
    )

    target_uv = torch.stack(
        [
            source_uv,
            source_uv + torch.tensor([0.20, 0.15]),
            source_uv + torch.tensor([0.45, 0.35]),
        ],
        dim=0,
    )

    correct = run_transport(
        linear_grid,
        source_uv,
        target_uv,
        mode="correct",
        seed=0,
    )

    shuffled = run_transport(
        linear_grid,
        source_uv,
        target_uv,
        mode="shuffled",
        seed=0,
    )

    constant_grid = torch.full(
        (2, height, width),
        3.25,
        dtype=torch.float32,
    )

    constant = run_transport(
        constant_grid,
        source_uv,
        target_uv,
    )

    constant_values = constant["transported_grid"][
        constant["transport_mask"].expand_as(
            constant["transported_grid"]
        )
    ]

    checks = {
        "correct_finite": bool(
            torch.isfinite(correct["transported_grid"]).all()
        ),
        "weight_finite": bool(
            torch.isfinite(correct["transport_weight"]).all()
        ),
        "mask_equals_positive_weight": bool(
            torch.equal(
                correct["transport_mask"],
                correct["transport_weight"] > correct["eps"],
            )
        ),
        "correct_shuffled_same_mask": bool(
            torch.equal(
                correct["transport_mask"],
                shuffled["transport_mask"],
            )
        ),
        "correct_shuffled_same_count": bool(
            torch.equal(
                correct["contribution_count"],
                shuffled["contribution_count"],
            )
        ),
        "correct_shuffled_same_weight": bool(
            torch.allclose(
                correct["transport_weight"],
                shuffled["transport_weight"],
            )
        ),
        "correct_shuffled_different_features": bool(
            not torch.equal(
                correct["transported_grid"],
                shuffled["transported_grid"],
            )
        ),
        "constant_field_preserved": bool(
            torch.allclose(
                constant_values,
                torch.full_like(constant_values, 3.25),
                atol=1e-6,
                rtol=0.0,
            )
        ),
    }

    summary = {
        "checks": checks,
        "frame_support_cells": [
            int(frame.sum())
            for frame in correct["transport_mask"]
        ],
        "frame_weight_sum": [
            float(frame.sum())
            for frame in correct["transport_weight"]
        ],
        "valid_point_count": [
            int(frame.sum())
            for frame in correct["valid_point_mask"]
        ],
        "all_checks_pass": all(checks.values()),
    }

    output = Path("soft_transport_math_validation.json")
    output.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    if not summary["all_checks_pass"]:
        raise RuntimeError("soft transport mathematical validation failed")


if __name__ == "__main__":
    main()
