"""Build a gated soft-transport Wan latent artifact.

The script reuses the exact source and target Wan latents stored by the
verified hard-transport VAE probe. It does not reload or re-encode the Wan VAE.

The original hard artifact and transport-ready artifact are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from deform_transport.hard_transport import hard_point_transport
from deform_transport.soft_transport import soft_point_transport
from deform_transport.transport_ready import validate_transport_ready
from deform_transport.visibility_contract import select_target_validity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport-ready",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--base-vae-artifact",
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
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--visibility-contract",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--transport-validity-mode",
        choices=(
            "projection_only",
            "source_and_future_visible",
        ),
        default="projection_only",
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


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(torch.float32)

    return {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
        "finite": bool(torch.isfinite(value).all()),
    }


def masked_mean_abs(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    expanded = mask.unsqueeze(0).expand_as(first)

    if not bool(expanded.any()):
        return 0.0

    return float(
        torch.abs(first - second)[expanded].mean()
    )


def main() -> None:
    args = parse_args()

    transport_path = args.transport_ready.resolve()
    base_path = args.base_vae_artifact.resolve()
    output_path = args.output.resolve()
    visibility_path = (
        args.visibility_contract.resolve()
        if args.visibility_contract is not None
        else None
    )

    if args.threshold < 0:
        raise ValueError("threshold must be nonnegative")

    if not transport_path.is_file():
        raise FileNotFoundError(transport_path)

    if not base_path.is_file():
        raise FileNotFoundError(base_path)

    if (
        args.transport_validity_mode
        == "source_and_future_visible"
        and visibility_path is None
    ):
        raise ValueError(
            "source_and_future_visible mode requires "
            "--visibility-contract"
        )

    if (
        visibility_path is not None
        and not visibility_path.is_file()
    ):
        raise FileNotFoundError(visibility_path)

    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    started = time.perf_counter()

    state = torch.load(
        transport_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_transport_ready(state)

    required_continuous = {
        "source_points_2d_latent_continuous",
        "points_2d_latent_continuous",
    }

    missing_continuous = sorted(
        required_continuous - set(state)
    )

    if missing_continuous:
        raise ValueError(
            "continuous transport-ready fields are missing: "
            f"{missing_continuous}"
        )

    base = torch.load(
        base_path,
        map_location="cpu",
        weights_only=True,
    )

    visibility_state = None

    if visibility_path is not None:
        visibility_state = torch.load(
            visibility_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(
            visibility_state,
            dict,
        ):
            raise ValueError(
                "visibility artifact must be a dictionary"
            )

    required_base = {
        "latent_frame_indices",
        "source_latent",
        "target_latent",
        "correct_fused_latent",
        "shuffled_fused_latent",
        "transport_mask",
        "contribution_count",
    }

    missing_base = sorted(required_base - set(base))

    if missing_base:
        raise ValueError(
            f"base VAE artifact is missing keys: {missing_base}"
        )

    latent_indices = base["latent_frame_indices"].to(
        dtype=torch.long,
        device="cpu",
    )

    source_latent_cpu = base["source_latent"]
    target_latent_cpu = base["target_latent"]

    if tuple(source_latent_cpu.shape) != (
        1,
        1,
        16,
        int(state["latent_height"]),
        int(state["latent_width"]),
    ):
        raise ValueError(
            f"unexpected source latent shape: "
            f"{tuple(source_latent_cpu.shape)}"
        )

    expected_target_shape = (
        1,
        int(latent_indices.numel()),
        16,
        int(state["latent_height"]),
        int(state["latent_width"]),
    )

    if tuple(target_latent_cpu.shape) != expected_target_shape:
        raise ValueError(
            f"unexpected target latent shape: "
            f"{tuple(target_latent_cpu.shape)}"
        )

    if int(latent_indices.min()) < 0:
        raise ValueError("latent frame indices must be nonnegative")

    if int(latent_indices.max()) >= state[
        "points_2d_latent"
    ].shape[0]:
        raise ValueError(
            "latent frame indices exceed transport trajectory"
        )

    validity = select_target_validity(
        state=state,
        latent_indices=latent_indices,
        visibility_state=visibility_state,
        mode=args.transport_validity_mode,
    )

    projection_target_valid_cpu = validity[
        "projection_target_valid"
    ]

    selected_target_valid_cpu = validity[
        "selected_target_valid"
    ]

    selected_future_visible_cpu = validity[
        "selected_future_visible"
    ]

    source_visible_cpu = state[
        "source_visible"
    ].to(
        dtype=torch.bool,
        device="cpu",
    )

    effective_projection_point_mask_cpu = (
        source_visible_cpu.unsqueeze(0)
        & projection_target_valid_cpu
    )

    effective_selected_point_mask_cpu = (
        source_visible_cpu.unsqueeze(0)
        & selected_target_valid_cpu
    )

    if bool(
        (
            effective_selected_point_mask_cpu
            & ~effective_projection_point_mask_cpu
        ).any()
    ):
        raise RuntimeError(
            "selected point contract is not a subset "
            "of projection-only contract"
        )

    device = torch.device("cuda:0")

    source_grid = (
        source_latent_cpu[0, 0]
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )

    target_latent = (
        target_latent_cpu
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )

    common = {
        "source_grid": source_grid,
        "source_visible": state[
            "source_visible"
        ].to(device),
        "source_valid": state[
            "source_valid"
        ].to(device),
        "target_valid": selected_target_valid_cpu.to(
            device
        ),
        "point_id": state["point_id"].to(device),
        "object_id": state["object_id"].to(device),
        "seed": args.seed,
    }

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    transport_started = time.perf_counter()

    projection_common = dict(common)
    projection_common["target_valid"] = (
        projection_target_valid_cpu.to(device)
    )

    hard_projection_reference = hard_point_transport(
        source_uv=state[
            "source_points_2d_latent"
        ].to(device),
        target_uv=state[
            "points_2d_latent"
        ][latent_indices].to(device),
        mode="correct",
        **projection_common,
    )

    if args.transport_validity_mode == "projection_only":
        hard_reference = hard_projection_reference
    else:
        hard_reference = hard_point_transport(
            source_uv=state[
                "source_points_2d_latent"
            ].to(device),
            target_uv=state[
                "points_2d_latent"
            ][latent_indices].to(device),
            mode="correct",
            **common,
        )

    soft_correct = soft_point_transport(
        source_uv=state[
            "source_points_2d_latent_continuous"
        ].to(device),
        target_uv=state[
            "points_2d_latent_continuous"
        ][latent_indices].to(device),
        mode="correct",
        **common,
    )

    soft_shuffled = soft_point_transport(
        source_uv=state[
            "source_points_2d_latent_continuous"
        ].to(device),
        target_uv=state[
            "points_2d_latent_continuous"
        ][latent_indices].to(device),
        mode="shuffled",
        **common,
    )

    torch.cuda.synchronize()

    transport_seconds = (
        time.perf_counter() - transport_started
    )

    base_mask = base["transport_mask"].to(device)
    base_count = base["contribution_count"].to(device)

    projection_hard_mask = hard_projection_reference[
        "transport_mask"
    ]
    projection_hard_count = hard_projection_reference[
        "contribution_count"
    ]

    hard_mask = hard_reference["transport_mask"]
    hard_count = hard_reference["contribution_count"]

    if not torch.equal(
        base_mask,
        projection_hard_mask,
    ):
        raise RuntimeError(
            "recomputed projection-only hard mask does not "
            "match base VAE artifact"
        )

    if not torch.equal(
        base_count,
        projection_hard_count,
    ):
        raise RuntimeError(
            "recomputed projection-only hard count does not "
            "match base VAE artifact"
        )

    if bool(
        (
            hard_mask
            & ~projection_hard_mask
        ).any()
    ):
        raise RuntimeError(
            "selected hard support is not a subset of "
            "projection-only hard support"
        )

    if not torch.equal(
        soft_correct["transport_mask"],
        soft_shuffled["transport_mask"],
    ):
        raise RuntimeError(
            "soft Correct and Shuffled masks differ"
        )

    if not torch.equal(
        soft_correct["contribution_count"],
        soft_shuffled["contribution_count"],
    ):
        raise RuntimeError(
            "soft Correct and Shuffled counts differ"
        )

    correct_transport_weight = soft_correct["transport_weight"]
    shuffled_transport_weight = soft_shuffled["transport_weight"]

    weight_absolute_difference = torch.abs(
        correct_transport_weight
        - shuffled_transport_weight
    )

    weight_max_abs_difference = float(
        weight_absolute_difference.max()
    )
    weight_mean_abs_difference = float(
        weight_absolute_difference.mean()
    )

    weights_allclose = bool(
        torch.allclose(
            correct_transport_weight,
            shuffled_transport_weight,
            atol=2e-4,
            rtol=2e-4,
        )
    )

    if not weights_allclose:
        raise RuntimeError(
            "soft Correct and Shuffled weights are not numerically "
            "equivalent: "
            f"max_abs_difference={weight_max_abs_difference}, "
            f"mean_abs_difference={weight_mean_abs_difference}"
        )

    if not torch.equal(
        soft_correct["valid_point_mask"],
        soft_shuffled["valid_point_mask"],
    ):
        raise RuntimeError(
            "soft Correct and Shuffled valid-point masks differ"
        )

    raw_soft_mask = soft_correct["transport_mask"]
    raw_soft_count = soft_correct["contribution_count"]
    # Geometry is shared between Correct and Shuffled. Use the Correct
    # accumulation as the canonical spatial weight for both variants.
    transport_weight = correct_transport_weight

    if bool((hard_mask & ~raw_soft_mask).any()):
        raise RuntimeError(
            "legacy Hard support is not a subset of Soft support"
        )

    soft_only_mask = raw_soft_mask & ~hard_mask

    gated_mask = (
        hard_mask
        | (
            soft_only_mask
            & (transport_weight >= args.threshold)
        )
    )

    # Diagnostic only: independently threshold the Shuffled accumulation.
    # Correct and Shuffled still use the shared canonical gated_mask below,
    # ensuring that feature identity is the only ablation variable.
    shuffled_independent_gated_mask = (
        hard_mask
        | (
            soft_only_mask
            & (
                shuffled_transport_weight
                >= args.threshold
            )
        )
    )

    independent_gated_mask_equal = bool(
        torch.equal(
            gated_mask,
            shuffled_independent_gated_mask,
        )
    )

    independent_gated_mask_difference_cells = int(
        (
            gated_mask
            != shuffled_independent_gated_mask
        ).sum()
    )

    gated_count = torch.where(
        gated_mask,
        raw_soft_count,
        torch.zeros_like(raw_soft_count),
    )

    if not torch.equal(
        gated_mask,
        gated_count > 0,
    ):
        raise RuntimeError(
            "gated mask/count contract is invalid"
        )

    raw_correct = (
        soft_correct["transported_grid"]
        .unsqueeze(0)
    )

    raw_shuffled = (
        soft_shuffled["transported_grid"]
        .unsqueeze(0)
    )

    broadcast_mask = gated_mask.unsqueeze(0)

    fused_correct = torch.where(
        broadcast_mask,
        raw_correct,
        target_latent,
    )

    fused_shuffled = torch.where(
        broadcast_mask,
        raw_shuffled,
        target_latent,
    )

    for tensor, name in (
        (raw_correct, "raw_correct"),
        (raw_shuffled, "raw_shuffled"),
        (fused_correct, "fused_correct"),
        (fused_shuffled, "fused_shuffled"),
        (transport_weight, "transport_weight"),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(
                f"{name} contains NaN or Inf"
            )

    output_state = {
        "format_version": 3,
        "artifact_kind": "wan_vae_soft_transport",
        "transport_method": (
            "continuous_bilinear_forward_splat_gate"
        ),
        "soft_weight_threshold": float(
            args.threshold
        ),
        "seed": int(args.seed),
        "transport_validity_mode": (
            args.transport_validity_mode
        ),
        "projection_target_valid": (
            projection_target_valid_cpu
            .detach()
            .cpu()
            .contiguous()
        ),
        "selected_target_valid": (
            selected_target_valid_cpu
            .detach()
            .cpu()
            .contiguous()
        ),
        "latent_frame_indices": (
            latent_indices.contiguous()
        ),
        "source_latent": (
            source_latent_cpu.detach().cpu().contiguous()
        ),
        "target_latent": (
            target_latent_cpu.detach().cpu().contiguous()
        ),
        # Compatibility keys used by pipeline_integration.py.
        "correct_transported_latent": (
            raw_correct.detach().cpu().contiguous()
        ),
        "shuffled_transported_latent": (
            raw_shuffled.detach().cpu().contiguous()
        ),
        "correct_fused_latent": (
            fused_correct.detach().cpu().contiguous()
        ),
        "shuffled_fused_latent": (
            fused_shuffled.detach().cpu().contiguous()
        ),
        "transport_mask": (
            gated_mask.detach().cpu().contiguous()
        ),
        "contribution_count": (
            gated_count.detach().cpu().contiguous()
        ),
        # Additional audit tensors.
        "transport_weight": (
            transport_weight.detach().cpu().contiguous()
        ),
        "raw_soft_transport_mask": (
            raw_soft_mask.detach().cpu().contiguous()
        ),
        "raw_soft_contribution_count": (
            raw_soft_count.detach().cpu().contiguous()
        ),
        "hard_reference_mask": (
            hard_mask.detach().cpu().contiguous()
        ),
        "hard_reference_contribution_count": (
            hard_count.detach().cpu().contiguous()
        ),
        "projection_hard_reference_mask": (
            projection_hard_mask
            .detach()
            .cpu()
            .contiguous()
        ),
        "projection_hard_reference_contribution_count": (
            projection_hard_count
            .detach()
            .cpu()
            .contiguous()
        ),
        "soft_only_mask": (
            soft_only_mask.detach().cpu().contiguous()
        ),
        "shuffled_permutation": (
            soft_shuffled["permutation"]
            .detach()
            .cpu()
            .contiguous()
        ),
        "source_files": {
            "transport_ready": str(transport_path),
            "transport_ready_sha256": sha256(
                transport_path
            ),
            "base_vae_artifact": str(base_path),
            "base_vae_artifact_sha256": sha256(
                base_path
            ),
            "visibility_contract": (
                str(visibility_path)
                if visibility_path is not None
                else None
            ),
            "visibility_contract_sha256": (
                sha256(visibility_path)
                if visibility_path is not None
                else None
            ),
        },
    }

    if selected_future_visible_cpu is not None:
        output_state["selected_future_visible"] = (
            selected_future_visible_cpu
            .detach()
            .cpu()
            .contiguous()
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(output_state, output_path)

    loaded = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True,
    )

    if not torch.equal(
        loaded["transport_mask"],
        loaded["contribution_count"] > 0,
    ):
        raise RuntimeError(
            "saved artifact broke the mask/count contract"
        )

    support_hard = int(hard_mask.sum())
    support_soft = int(raw_soft_mask.sum())
    support_gated = int(gated_mask.sum())

    report = {
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "transport_method": output_state[
            "transport_method"
        ],
        "threshold": float(args.threshold),
        "latent_frame_indices": (
            latent_indices.tolist()
        ),
        "validity": {
            "mode": args.transport_validity_mode,
            "visibility_contract": (
                str(visibility_path)
                if visibility_path is not None
                else None
            ),
            "projection_effective_points_per_slot": (
                effective_projection_point_mask_cpu
                .sum(dim=1)
                .tolist()
            ),
            "selected_effective_points_per_slot": (
                effective_selected_point_mask_cpu
                .sum(dim=1)
                .tolist()
            ),
            "removed_effective_points_per_slot": (
                (
                    effective_projection_point_mask_cpu
                    & ~effective_selected_point_mask_cpu
                )
                .sum(dim=1)
                .tolist()
            ),
            "selected_subset_projection": True,
            "frame0_effective_contract_equal": bool(
                torch.equal(
                    effective_projection_point_mask_cpu[0],
                    effective_selected_point_mask_cpu[0],
                )
            ),
        },
        "support": {
            "hard_cells": support_hard,
            "raw_soft_cells": support_soft,
            "gated_cells": support_gated,
            "raw_soft_over_hard": (
                support_soft / support_hard
            ),
            "gated_over_hard": (
                support_gated / support_hard
            ),
            "soft_only_cells": int(
                soft_only_mask.sum()
            ),
            "retained_soft_only_cells": int(
                (gated_mask & soft_only_mask).sum()
            ),
        },
        "fairness": {
            "masks_equal": True,
            "counts_equal": True,
            "weights_allclose": weights_allclose,
            "weight_allclose_atol": 2e-4,
            "weight_allclose_rtol": 2e-4,
            "weight_max_abs_difference": (
                weight_max_abs_difference
            ),
            "weight_mean_abs_difference": (
                weight_mean_abs_difference
            ),
            "independent_gated_mask_equal": (
                independent_gated_mask_equal
            ),
            "independent_gated_mask_difference_cells": (
                independent_gated_mask_difference_cells
            ),
            "shared_canonical_gated_mask": True,
            "valid_point_masks_equal": True,
            "permutations_differ": bool(
                not torch.equal(
                    soft_correct["permutation"],
                    soft_shuffled["permutation"],
                )
            ),
        },
        "compatibility": {
            "base_hard_mask_exact": True,
            "base_hard_count_exact": True,
            "base_projection_hard_mask_exact": True,
            "base_projection_hard_count_exact": True,
            "selected_hard_subset_projection_hard": bool(
                not (
                    hard_mask
                    & ~projection_hard_mask
                ).any()
            ),
            "transport_mask_equals_count_positive": True,
            "selected_key_shapes_match_target": (
                tuple(fused_correct.shape)
                == tuple(target_latent.shape)
                == tuple(fused_shuffled.shape)
            ),
        },
        "latent_differences": {
            "correct_vs_shuffled_mean_abs": float(
                torch.abs(
                    fused_correct - fused_shuffled
                ).mean()
            ),
            "soft_correct_vs_target_mean_abs": float(
                torch.abs(
                    fused_correct - target_latent
                ).mean()
            ),
            "soft_correct_vs_target_gated_mean_abs": (
                masked_mean_abs(
                    fused_correct,
                    target_latent,
                    gated_mask,
                )
            ),
            "soft_correct_vs_hard_correct_mean_abs": float(
                torch.abs(
                    fused_correct
                    - base[
                        "correct_fused_latent"
                    ].to(device)
                ).mean()
            ),
        },
        "statistics": {
            "target_latent": tensor_stats(
                target_latent
            ),
            "soft_correct_fused": tensor_stats(
                fused_correct
            ),
            "soft_shuffled_fused": tensor_stats(
                fused_shuffled
            ),
            "transport_weight": tensor_stats(
                transport_weight
            ),
        },
        "runtime_seconds": {
            "transport": transport_seconds,
            "total": (
                time.perf_counter() - started
            ),
        },
        "gpu_peak_memory_mib": {
            "allocated": float(
                torch.cuda.max_memory_allocated()
                / (1024**2)
            ),
            "reserved": float(
                torch.cuda.max_memory_reserved()
                / (1024**2)
            ),
        },
    }

    report_path = output_path.with_suffix(
        ".report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
