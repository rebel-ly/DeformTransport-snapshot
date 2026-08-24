"""Build controlled Soft-transport fusion ablation artifacts.

Candidate A:
    Soft transported values + legacy Hard support + full replacement.

Candidate B:
    Soft transported values + Gate-0.25 support + residual blend alpha=0.5.

No transport, VAE encoding, VAE decoding, or diffusion inference is rerun.
The source artifacts are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from deform_transport.pipeline_integration import (
    load_precomputed_transport_latent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hard-artifact",
        type=Path,
        required=True,
    )
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
        "--alpha",
        type=float,
        default=0.5,
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


def mean_abs(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    return float(
        torch.abs(
            first.to(torch.float32)
            - second.to(torch.float32)
        ).mean()
    )


def masked_mean_abs(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    expanded = mask.unsqueeze(0).expand_as(first)

    if not bool(expanded.any()):
        return 0.0

    return float(
        torch.abs(
            first.to(torch.float32)
            - second.to(torch.float32)
        )[expanded].mean()
    )


def tensor_stats(
    tensor: torch.Tensor,
) -> dict[str, Any]:
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


def validate_source_artifact(
    state: dict[str, Any],
    *,
    label: str,
) -> None:
    required = {
        "latent_frame_indices",
        "source_latent",
        "target_latent",
        "correct_transported_latent",
        "shuffled_transported_latent",
        "correct_fused_latent",
        "shuffled_fused_latent",
        "transport_mask",
        "contribution_count",
    }

    missing = sorted(required - set(state))

    if missing:
        raise ValueError(
            f"{label} artifact is missing keys: {missing}"
        )

    mask = state["transport_mask"]
    count = state["contribution_count"]
    target = state["target_latent"]

    if mask.dtype != torch.bool:
        raise ValueError(
            f"{label} transport_mask must be boolean"
        )

    if not torch.equal(mask, count > 0):
        raise ValueError(
            f"{label} mask/count contract failed"
        )

    for key in (
        "source_latent",
        "target_latent",
        "correct_transported_latent",
        "shuffled_transported_latent",
        "correct_fused_latent",
        "shuffled_fused_latent",
    ):
        tensor = state[key]

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"{label} {key} must be a tensor"
            )

        if not tensor.dtype.is_floating_point:
            raise ValueError(
                f"{label} {key} must be floating point"
            )

        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"{label} {key} contains NaN or Inf"
            )

    expected_shape = tuple(target.shape)

    for key in (
        "correct_transported_latent",
        "shuffled_transported_latent",
        "correct_fused_latent",
        "shuffled_fused_latent",
    ):
        if tuple(state[key].shape) != expected_shape:
            raise ValueError(
                f"{label} {key} shape mismatch"
            )


def save_candidate(
    *,
    path: Path,
    artifact_kind: str,
    fusion_mode: str,
    alpha: float,
    latent_indices: torch.Tensor,
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    raw_correct: torch.Tensor,
    raw_shuffled: torch.Tensor,
    fused_correct: torch.Tensor,
    fused_shuffled: torch.Tensor,
    mask: torch.Tensor,
    count: torch.Tensor,
    soft_only_mask: torch.Tensor,
    source_soft_only_mask: torch.Tensor,
    hard_path: Path,
    soft_path: Path,
    extra: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite: {path}"
        )

    if not torch.equal(mask, count > 0):
        raise RuntimeError(
            f"{artifact_kind} mask/count contract failed"
        )

    for tensor, label in (
        (fused_correct, "fused_correct"),
        (fused_shuffled, "fused_shuffled"),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(
                f"{artifact_kind} {label} is not finite"
            )

    state = {
        "format_version": 3,
        "artifact_kind": artifact_kind,
        "transport_method": (
            "continuous_bilinear_forward_splat"
        ),
        "fusion_mode": fusion_mode,
        "fusion_alpha": float(alpha),
        "latent_frame_indices": (
            latent_indices.detach().cpu().contiguous()
        ),
        "source_latent": (
            source_latent.detach().cpu().contiguous()
        ),
        "target_latent": (
            target_latent.detach().cpu().contiguous()
        ),
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
            mask.detach().cpu().contiguous()
        ),
        "contribution_count": (
            count.detach().cpu().contiguous()
        ),
        "soft_only_mask": (
            soft_only_mask.detach().cpu().contiguous()
        ),
        "source_soft_only_mask": (
            source_soft_only_mask.detach()
            .cpu()
            .contiguous()
        ),
        "source_files": {
            "hard_artifact": str(hard_path),
            "hard_artifact_sha256": sha256(hard_path),
            "soft_artifact": str(soft_path),
            "soft_artifact_sha256": sha256(soft_path),
        },
        "extra": extra,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(state, path)

    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not torch.equal(
        loaded["transport_mask"],
        loaded["contribution_count"] > 0,
    ):
        raise RuntimeError(
            f"{artifact_kind} saved mask/count contract failed"
        )

    reference = loaded["target_latent"]

    for mode in ("correct", "shuffled"):
        selected = load_precomputed_transport_latent(
            path,
            mode=mode,
            reference_latent=reference,
        )

        expected = loaded[f"{mode}_fused_latent"]

        if not torch.equal(selected, expected):
            raise RuntimeError(
                f"{artifact_kind} loader mismatch for {mode}"
            )


def main() -> None:
    args = parse_args()

    hard_path = args.hard_artifact.resolve()
    soft_path = args.soft_artifact.resolve()
    output_dir = args.output_dir.resolve()

    if not hard_path.is_file():
        raise FileNotFoundError(hard_path)

    if not soft_path.is_file():
        raise FileNotFoundError(soft_path)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite output directory: "
            f"{output_dir}"
        )

    if not 0.0 < args.alpha < 1.0:
        raise ValueError(
            "alpha must lie strictly between zero and one"
        )

    hard = torch.load(
        hard_path,
        map_location="cpu",
        weights_only=True,
    )

    soft = torch.load(
        soft_path,
        map_location="cpu",
        weights_only=True,
    )

    validate_source_artifact(
        hard,
        label="Hard",
    )
    validate_source_artifact(
        soft,
        label="Soft",
    )

    for key in (
        "latent_frame_indices",
        "source_latent",
        "target_latent",
    ):
        if not torch.equal(hard[key], soft[key]):
            raise RuntimeError(
                f"Hard and Soft {key} differ"
            )

    hard_mask = hard["transport_mask"]
    hard_count = hard["contribution_count"]

    soft_mask = soft["transport_mask"]
    soft_count = soft["contribution_count"]

    if bool((hard_mask & ~soft_mask).any()):
        raise RuntimeError(
            "Hard support is not a subset of Soft support"
        )

    target = hard["target_latent"].to(
        torch.float32
    )

    raw_correct = soft[
        "correct_transported_latent"
    ].to(torch.float32)

    raw_shuffled = soft[
        "shuffled_transported_latent"
    ].to(torch.float32)

    if tuple(raw_correct.shape) != tuple(target.shape):
        raise RuntimeError(
            "Soft Correct transported latent shape mismatch"
        )

    if tuple(raw_shuffled.shape) != tuple(target.shape):
        raise RuntimeError(
            "Soft Shuffled transported latent shape mismatch"
        )

    # ---------------------------------------------------------
    # Candidate A: Soft values, but only legacy Hard support.
    # ---------------------------------------------------------
    hard_broadcast = hard_mask.unsqueeze(0)

    hardmask_correct = torch.where(
        hard_broadcast,
        raw_correct,
        target,
    )

    hardmask_shuffled = torch.where(
        hard_broadcast,
        raw_shuffled,
        target,
    )

    hardmask_soft_only = torch.zeros_like(
        hard_mask
    )

    hardmask_path = (
        output_dir
        / "vae_latent_outputs_soft_hardmask_replace.pt"
    )

    save_candidate(
        path=hardmask_path,
        artifact_kind=(
            "wan_vae_soft_hardmask_replace"
        ),
        fusion_mode="masked_replace",
        alpha=1.0,
        latent_indices=hard["latent_frame_indices"],
        source_latent=hard["source_latent"],
        target_latent=target,
        raw_correct=raw_correct,
        raw_shuffled=raw_shuffled,
        fused_correct=hardmask_correct,
        fused_shuffled=hardmask_shuffled,
        mask=hard_mask,
        count=hard_count,
        soft_only_mask=hardmask_soft_only,
        source_soft_only_mask=soft["soft_only_mask"],
        hard_path=hard_path,
        soft_path=soft_path,
        extra={
            "support_source": "legacy_hard_mask",
            "purpose": (
                "isolate Soft interpolation from "
                "Soft support expansion"
            ),
        },
    )

    # ---------------------------------------------------------
    # Candidate B: Soft Gate-0.25 support, alpha residual blend.
    # ---------------------------------------------------------
    soft_broadcast = (
        soft_mask.unsqueeze(0)
        .to(dtype=target.dtype)
    )

    blend_correct = (
        target
        + args.alpha
        * soft_broadcast
        * (raw_correct - target)
    )

    blend_shuffled = (
        target
        + args.alpha
        * soft_broadcast
        * (raw_shuffled - target)
    )

    blend_path = (
        output_dir
        / (
            "vae_latent_outputs_"
            f"soft_gate025_blend_a{int(args.alpha * 100):03d}.pt"
        )
    )

    save_candidate(
        path=blend_path,
        artifact_kind=(
            "wan_vae_soft_gate025_residual_blend"
        ),
        fusion_mode="residual_blend",
        alpha=args.alpha,
        latent_indices=hard["latent_frame_indices"],
        source_latent=hard["source_latent"],
        target_latent=target,
        raw_correct=raw_correct,
        raw_shuffled=raw_shuffled,
        fused_correct=blend_correct,
        fused_shuffled=blend_shuffled,
        mask=soft_mask,
        count=soft_count,
        soft_only_mask=soft["soft_only_mask"],
        source_soft_only_mask=soft["soft_only_mask"],
        hard_path=hard_path,
        soft_path=soft_path,
        extra={
            "support_source": "soft_gate025_mask",
            "purpose": (
                "test whether partial residual injection "
                "preserves target detail"
            ),
        },
    )

    existing_hard_correct = hard[
        "correct_fused_latent"
    ].to(torch.float32)

    existing_soft_correct = soft[
        "correct_fused_latent"
    ].to(torch.float32)

    report = {
        "hard_artifact": str(hard_path),
        "soft_artifact": str(soft_path),
        "alpha": float(args.alpha),
        "support": {
            "hard_cells": int(hard_mask.sum()),
            "soft_gate025_cells": int(soft_mask.sum()),
            "soft_only_cells": int(
                (soft_mask & ~hard_mask).sum()
            ),
        },
        "candidate_outputs": {
            "soft_hardmask_replace": {
                "path": str(hardmask_path),
                "sha256": sha256(hardmask_path),
                "mask_cells": int(hard_mask.sum()),
                "correct_statistics": tensor_stats(
                    hardmask_correct
                ),
                "correct_vs_shuffled_mean_abs": (
                    mean_abs(
                        hardmask_correct,
                        hardmask_shuffled,
                    )
                ),
                "correct_vs_hard_correct_mean_abs": (
                    mean_abs(
                        hardmask_correct,
                        existing_hard_correct,
                    )
                ),
                "correct_vs_existing_soft_replace_mean_abs": (
                    mean_abs(
                        hardmask_correct,
                        existing_soft_correct,
                    )
                ),
                "correct_vs_target_on_mask": (
                    masked_mean_abs(
                        hardmask_correct,
                        target,
                        hard_mask,
                    )
                ),
                "loader_compatible": True,
            },
            "soft_gate025_blend_a050": {
                "path": str(blend_path),
                "sha256": sha256(blend_path),
                "mask_cells": int(soft_mask.sum()),
                "correct_statistics": tensor_stats(
                    blend_correct
                ),
                "correct_vs_shuffled_mean_abs": (
                    mean_abs(
                        blend_correct,
                        blend_shuffled,
                    )
                ),
                "correct_vs_hard_correct_mean_abs": (
                    mean_abs(
                        blend_correct,
                        existing_hard_correct,
                    )
                ),
                "correct_vs_existing_soft_replace_mean_abs": (
                    mean_abs(
                        blend_correct,
                        existing_soft_correct,
                    )
                ),
                "correct_vs_target_on_mask": (
                    masked_mean_abs(
                        blend_correct,
                        target,
                        soft_mask,
                    )
                ),
                "loader_compatible": True,
            },
        },
        "checks": {
            "source_latents_exactly_equal": True,
            "target_latents_exactly_equal": True,
            "latent_frame_indices_exactly_equal": True,
            "hard_support_subset_of_soft": True,
            "both_candidates_finite": bool(
                torch.isfinite(hardmask_correct).all()
                and torch.isfinite(hardmask_shuffled).all()
                and torch.isfinite(blend_correct).all()
                and torch.isfinite(blend_shuffled).all()
            ),
            "both_candidates_loader_compatible": True,
            "correct_shuffled_remain_different": bool(
                not torch.equal(
                    hardmask_correct,
                    hardmask_shuffled,
                )
                and not torch.equal(
                    blend_correct,
                    blend_shuffled,
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
            "fusion ablation artifact validation failed"
        )


if __name__ == "__main__":
    main()
