"""Build a confidence-weighted Soft transport residual artifact.

Confidence:
    ratio = transport_weight / contribution_count
    x = clamp((ratio - low) / (high - low), 0, 1)
    confidence = x^2 * (3 - 2*x)

Correct and Shuffled use exactly the same geometry-derived confidence map.
The source artifact is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from deform_transport.pipeline_integration import (
    load_precomputed_transport_residual,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--soft-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--confidence-low",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--confidence-high",
        type=float,
        default=0.25,
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


def statistics(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(torch.float32)
    finite = tensor[torch.isfinite(tensor)]

    if finite.numel() == 0:
        return {"values": 0}

    probabilities = torch.tensor(
        [0.00, 0.01, 0.05, 0.10, 0.25,
         0.50, 0.75, 0.90, 0.95, 0.99, 1.00],
        dtype=torch.float32,
    )

    quantile = torch.quantile(
        finite.flatten(),
        probabilities,
    )

    return {
        "values": int(finite.numel()),
        "mean": float(finite.mean()),
        "std": (
            float(finite.std())
            if finite.numel() > 1
            else 0.0
        ),
        "q00": float(quantile[0]),
        "q01": float(quantile[1]),
        "q05": float(quantile[2]),
        "q10": float(quantile[3]),
        "q25": float(quantile[4]),
        "q50": float(quantile[5]),
        "q75": float(quantile[6]),
        "q90": float(quantile[7]),
        "q95": float(quantile[8]),
        "q99": float(quantile[9]),
        "q100": float(quantile[10]),
    }


def region_statistics(
    value: torch.Tensor,
    regions: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result = {}

    for name, region in regions.items():
        selected = value[region]

        result[name] = {
            "cells": int(region.sum()),
            "statistics": statistics(selected),
        }

    return result


def main() -> None:
    args = parse_args()

    source_path = args.soft_artifact.resolve()
    output_dir = args.output_dir.resolve()

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_dir}"
        )

    if not (
        0.0
        <= args.confidence_low
        < args.confidence_high
    ):
        raise ValueError(
            "confidence thresholds must satisfy "
            "0 <= low < high"
        )

    state = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "latent_frame_indices",
        "source_latent",
        "target_latent",
        "correct_transported_latent",
        "shuffled_transported_latent",
        "transport_mask",
        "contribution_count",
        "transport_weight",
    }

    missing = sorted(required - set(state))

    if missing:
        raise ValueError(
            f"source artifact is missing keys: {missing}"
        )

    target = state["target_latent"].to(
        torch.float32
    )
    correct = state[
        "correct_transported_latent"
    ].to(torch.float32)
    shuffled = state[
        "shuffled_transported_latent"
    ].to(torch.float32)

    mask = state["transport_mask"]
    count = state["contribution_count"]
    weight = state["transport_weight"].to(
        torch.float32
    )

    if target.ndim != 5:
        raise ValueError(
            "target_latent must have shape [B,T,C,H,W]"
        )

    expected_map_shape = (
        target.shape[1],
        1,
        target.shape[3],
        target.shape[4],
    )

    for tensor, label in (
        (mask, "transport_mask"),
        (count, "contribution_count"),
        (weight, "transport_weight"),
    ):
        if tuple(tensor.shape) != expected_map_shape:
            raise ValueError(
                f"{label} shape mismatch: "
                f"{tuple(tensor.shape)}"
            )

    if mask.dtype != torch.bool:
        raise ValueError(
            "transport_mask must be boolean"
        )

    if not torch.equal(mask, count > 0):
        raise ValueError(
            "mask/count contract failed"
        )

    if not bool(torch.isfinite(weight).all()):
        raise ValueError(
            "transport_weight contains NaN or Inf"
        )

    if bool((weight < 0).any()):
        raise ValueError(
            "transport_weight must be nonnegative"
        )

    ratio = torch.where(
        count > 0,
        weight / count.to(torch.float32).clamp_min(1),
        torch.zeros_like(weight),
    )

    normalized = (
        (ratio - args.confidence_low)
        / (
            args.confidence_high
            - args.confidence_low
        )
    ).clamp(0.0, 1.0)

    confidence = (
        normalized
        * normalized
        * (3.0 - 2.0 * normalized)
    )

    confidence = torch.where(
        mask,
        confidence,
        torch.zeros_like(confidence),
    )

    confidence_5d = confidence.unsqueeze(0)

    correct_residual = correct - target
    shuffled_residual = shuffled - target

    weighted_correct_residual = (
        confidence_5d * correct_residual
    )

    weighted_shuffled_residual = (
        confidence_5d * shuffled_residual
    )

    weighted_correct = (
        target + weighted_correct_residual
    )

    weighted_shuffled = (
        target + weighted_shuffled_residual
    )

    mask_5d = mask.unsqueeze(0).expand_as(target)

    if bool(
        torch.count_nonzero(
            weighted_correct_residual[~mask_5d]
        )
    ):
        raise RuntimeError(
            "Correct residual is nonzero outside mask"
        )

    if bool(
        torch.count_nonzero(
            weighted_shuffled_residual[~mask_5d]
        )
    ):
        raise RuntimeError(
            "Shuffled residual is nonzero outside mask"
        )

    if not bool(
        torch.isfinite(weighted_correct).all()
        and torch.isfinite(weighted_shuffled).all()
    ):
        raise RuntimeError(
            "weighted latent contains NaN or Inf"
        )

    neighbor_count = F.conv2d(
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

    interior = mask & (neighbor_count == 9)
    boundary = mask & ~interior

    regions = {
        "support": mask,
        "interior": interior,
        "boundary": boundary,
    }

    soft_only = state.get("soft_only_mask")

    if (
        isinstance(soft_only, torch.Tensor)
        and tuple(soft_only.shape)
        == tuple(mask.shape)
    ):
        regions["soft_only"] = (
            soft_only.to(torch.bool) & mask
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_path = (
        output_dir
        / (
            "vae_latent_outputs_soft_confidence_"
            "wpc_smooth_l010_h025.pt"
        )
    )

    output_state = {
        "format_version": 4,
        "artifact_kind": (
            "wan_vae_soft_confidence_weighted_residual"
        ),
        "transport_method": state.get(
            "transport_method",
            "continuous_bilinear_forward_splat",
        ),
        "confidence_method": (
            "smoothstep_weight_per_count"
        ),
        "confidence_low": float(
            args.confidence_low
        ),
        "confidence_high": float(
            args.confidence_high
        ),
        "latent_frame_indices": state[
            "latent_frame_indices"
        ].contiguous(),
        "source_latent": state[
            "source_latent"
        ].contiguous(),
        "target_latent": target.contiguous(),
        "correct_transported_latent": (
            weighted_correct.contiguous()
        ),
        "shuffled_transported_latent": (
            weighted_shuffled.contiguous()
        ),
        # These make the artifact compatible with the
        # existing replacement loader as well, although
        # the intended use is inter-step residual loading.
        "correct_fused_latent": (
            weighted_correct.contiguous()
        ),
        "shuffled_fused_latent": (
            weighted_shuffled.contiguous()
        ),
        "transport_mask": mask.contiguous(),
        "contribution_count": count.contiguous(),
        "transport_weight": weight.contiguous(),
        "confidence_ratio": ratio.contiguous(),
        "confidence_map": confidence.contiguous(),
        "soft_only_mask": (
            state["soft_only_mask"].contiguous()
            if isinstance(
                state.get("soft_only_mask"),
                torch.Tensor,
            )
            else torch.zeros_like(mask)
        ),
        "source_files": {
            "soft_artifact": str(source_path),
            "soft_artifact_sha256": sha256(
                source_path
            ),
        },
    }

    torch.save(
        output_state,
        output_path,
    )

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
            "saved mask/count contract failed"
        )

    reference = loaded[
        "target_latent"
    ].to(torch.bfloat16)

    loaded_correct_residual = (
        load_precomputed_transport_residual(
            output_path,
            mode="correct",
            reference_latent=reference,
        )
    )

    loaded_shuffled_residual = (
        load_precomputed_transport_residual(
            output_path,
            mode="shuffled",
            reference_latent=reference,
        )
    )

    expected_correct = (
        weighted_correct_residual.to(
            torch.bfloat16
        )
    )

    expected_shuffled = (
        weighted_shuffled_residual.to(
            torch.bfloat16
        )
    )

    correct_loader_match = bool(
        torch.equal(
            loaded_correct_residual,
            expected_correct,
        )
    )

    shuffled_loader_match = bool(
        torch.equal(
            loaded_shuffled_residual,
            expected_shuffled,
        )
    )

    report = {
        "output_artifact": {
            "path": str(output_path),
            "sha256": sha256(output_path),
        },
        "source_artifact": {
            "path": str(source_path),
            "sha256": sha256(source_path),
        },
        "confidence": {
            "method": (
                "smoothstep_weight_per_count"
            ),
            "low": float(args.confidence_low),
            "high": float(args.confidence_high),
            "ratio_statistics":
                region_statistics(
                    ratio,
                    regions,
                ),
            "confidence_statistics":
                region_statistics(
                    confidence,
                    regions,
                ),
        },
        "residual": {
            "raw_correct_mean_abs": float(
                correct_residual.abs().mean()
            ),
            "weighted_correct_mean_abs": float(
                weighted_correct_residual.abs().mean()
            ),
            "raw_shuffled_mean_abs": float(
                shuffled_residual.abs().mean()
            ),
            "weighted_shuffled_mean_abs": float(
                weighted_shuffled_residual.abs().mean()
            ),
            "correct_vs_shuffled_mean_abs": float(
                (
                    weighted_correct
                    - weighted_shuffled
                ).abs().mean()
            ),
        },
        "checks": {
            "mask_count_contract": True,
            "confidence_finite": bool(
                torch.isfinite(confidence).all()
            ),
            "confidence_in_zero_one": bool(
                (confidence >= 0).all()
                and (confidence <= 1).all()
            ),
            "correct_zero_outside_mask": True,
            "shuffled_zero_outside_mask": True,
            "weighted_latents_finite": True,
            "correct_loader_match": (
                correct_loader_match
            ),
            "shuffled_loader_match": (
                shuffled_loader_match
            ),
            "correct_shuffled_different": bool(
                not torch.equal(
                    weighted_correct,
                    weighted_shuffled,
                )
            ),
        },
    }

    report["all_checks_pass"] = all(
        report["checks"].values()
    )

    report_path = output_dir / "report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))

    if not report["all_checks_pass"]:
        raise RuntimeError(
            "confidence artifact validation failed"
        )


if __name__ == "__main__":
    main()
