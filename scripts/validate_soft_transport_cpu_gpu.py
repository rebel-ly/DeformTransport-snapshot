"""Validate CPU/CUDA parity for continuous bilinear point transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
        "--threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=4,
    )
    return parser.parse_args()


def tensor_difference(
    cpu: torch.Tensor,
    gpu: torch.Tensor,
) -> dict[str, Any]:
    gpu_cpu = gpu.detach().cpu()

    if cpu.dtype == torch.bool or not cpu.dtype.is_floating_point:
        return {
            "exact_equal": bool(
                torch.equal(cpu, gpu_cpu)
            )
        }

    difference = torch.abs(
        cpu.to(torch.float32)
        - gpu_cpu.to(torch.float32)
    )

    return {
        "exact_equal": bool(
            torch.equal(cpu, gpu_cpu)
        ),
        "allclose_atol_2e_4_rtol_2e_4": bool(
            torch.allclose(
                cpu.to(torch.float32),
                gpu_cpu.to(torch.float32),
                atol=2e-4,
                rtol=2e-4,
            )
        ),
        "max_abs_difference": float(
            difference.max()
        ),
        "mean_abs_difference": float(
            difference.mean()
        ),
    }


def run_transport(
    *,
    device: torch.device,
    state: dict,
    indices: torch.Tensor,
    source_grid_cpu: torch.Tensor,
    mode: str,
):
    source_grid = source_grid_cpu.to(device)

    common = {
        "source_grid": source_grid,
        "source_visible": state[
            "source_visible"
        ].to(device),
        "source_valid": state[
            "source_valid"
        ].to(device),
        "target_valid": state[
            "projection_valid"
        ][indices].to(device),
        "point_id": state["point_id"].to(device),
        "object_id": state["object_id"].to(device),
        "mode": mode,
        "seed": 0,
    }

    hard = hard_point_transport(
        source_uv=state[
            "source_points_2d_latent"
        ].to(torch.float32).to(device),
        target_uv=state[
            "points_2d_latent"
        ][indices].to(torch.float32).to(device),
        **common,
    )

    soft = soft_point_transport(
        source_uv=state[
            "source_points_2d_latent_continuous"
        ].to(device),
        target_uv=state[
            "points_2d_latent_continuous"
        ][indices].to(device),
        **common,
    )

    return hard, soft


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

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
            f"continuous fields are missing: {missing}"
        )

    frame_count = int(
        state["points_2d_latent"].shape[0]
    )

    indices = torch.arange(
        0,
        frame_count,
        args.temporal_stride,
        dtype=torch.long,
    )

    if indices[-1].item() != frame_count - 1:
        indices = torch.cat(
            [
                indices,
                torch.tensor(
                    [frame_count - 1],
                    dtype=torch.long,
                ),
            ]
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260805)

    source_grid_cpu = torch.randn(
        (
            4,
            int(state["latent_height"]),
            int(state["latent_width"]),
        ),
        generator=generator,
        dtype=torch.float32,
    )

    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")

    report = {
        "transport_ready": str(
            args.transport_ready.resolve()
        ),
        "cuda_device_name": torch.cuda.get_device_name(
            gpu_device
        ),
        "selected_frames": indices.tolist(),
        "threshold": args.threshold,
        "modes": {},
    }

    all_checks = []

    for mode in ("correct", "shuffled"):
        hard_cpu, soft_cpu = run_transport(
            device=cpu_device,
            state=state,
            indices=indices,
            source_grid_cpu=source_grid_cpu,
            mode=mode,
        )

        hard_gpu, soft_gpu = run_transport(
            device=gpu_device,
            state=state,
            indices=indices,
            source_grid_cpu=source_grid_cpu,
            mode=mode,
        )

        torch.cuda.synchronize()

        hard_mask_cpu = hard_cpu[
            "transport_mask"
        ]

        hard_mask_gpu = hard_gpu[
            "transport_mask"
        ].cpu()

        soft_mask_cpu = soft_cpu[
            "transport_mask"
        ]

        soft_mask_gpu = soft_gpu[
            "transport_mask"
        ].cpu()

        weight_cpu = soft_cpu[
            "transport_weight"
        ]

        weight_gpu = soft_gpu[
            "transport_weight"
        ].cpu()

        soft_only_cpu = (
            soft_mask_cpu
            & ~hard_mask_cpu
        )

        soft_only_gpu = (
            soft_mask_gpu
            & ~hard_mask_gpu
        )

        gated_cpu = (
            hard_mask_cpu
            | (
                soft_only_cpu
                & (weight_cpu >= args.threshold)
            )
        )

        gated_gpu = (
            hard_mask_gpu
            | (
                soft_only_gpu
                & (weight_gpu >= args.threshold)
            )
        )

        near_threshold = int(
            (
                torch.abs(
                    weight_cpu
                    - args.threshold
                )
                <= 2e-4
            ).sum()
        )

        mode_report = {
            "hard_grid": tensor_difference(
                hard_cpu["transported_grid"],
                hard_gpu["transported_grid"],
            ),
            "soft_grid": tensor_difference(
                soft_cpu["transported_grid"],
                soft_gpu["transported_grid"],
            ),
            "soft_weight": tensor_difference(
                weight_cpu,
                weight_gpu,
            ),
            "hard_mask_exact": bool(
                torch.equal(
                    hard_mask_cpu,
                    hard_mask_gpu,
                )
            ),
            "soft_mask_exact": bool(
                torch.equal(
                    soft_mask_cpu,
                    soft_mask_gpu,
                )
            ),
            "gated_mask_exact": bool(
                torch.equal(
                    gated_cpu,
                    gated_gpu,
                )
            ),
            "contribution_count_exact": bool(
                torch.equal(
                    soft_cpu["contribution_count"],
                    soft_gpu[
                        "contribution_count"
                    ].cpu(),
                )
            ),
            "valid_point_mask_exact": bool(
                torch.equal(
                    soft_cpu["valid_point_mask"],
                    soft_gpu[
                        "valid_point_mask"
                    ].cpu(),
                )
            ),
            "permutation_exact": bool(
                torch.equal(
                    soft_cpu["permutation"],
                    soft_gpu["permutation"].cpu(),
                )
            ),
            "cells_within_2e_4_of_threshold": (
                near_threshold
            ),
        }

        checks = [
            mode_report["hard_grid"][
                "allclose_atol_2e_4_rtol_2e_4"
            ],
            mode_report["soft_grid"][
                "allclose_atol_2e_4_rtol_2e_4"
            ],
            mode_report["soft_weight"][
                "allclose_atol_2e_4_rtol_2e_4"
            ],
            mode_report["hard_mask_exact"],
            mode_report["soft_mask_exact"],
            mode_report["gated_mask_exact"],
            mode_report["contribution_count_exact"],
            mode_report["valid_point_mask_exact"],
            mode_report["permutation_exact"],
        ]

        mode_report["all_checks_pass"] = all(
            checks
        )

        all_checks.extend(checks)
        report["modes"][mode] = mode_report

    report["all_checks_pass"] = all(all_checks)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=False,
    )

    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))

    if not report["all_checks_pass"]:
        raise RuntimeError(
            "CPU/CUDA soft transport parity failed"
        )


if __name__ == "__main__":
    main()
