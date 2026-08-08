"""Decode Hard, Soft-replace, Soft-hardmask and Soft-blend artifacts.

No VAE encoding and no diffusion inference are performed. Every candidate
reuses exactly the same stored source latent, target latent and temporal map.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from deform_transport.wan_vae_codec import RealWonderWanVAECodec
from scripts.decode_compare_hard_soft_wan_artifacts import (
    EXPECTED_VAE_SHA256,
    make_comparison_grid,
    pixels_01,
    sha256,
    tensor_stats,
    uint8_frames,
    write_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hard-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--soft-replace-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--soft-hardmask-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--soft-blend-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
    )

    return parser.parse_args()


def require_artifact(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "latent_frame_indices",
        "source_latent",
        "target_latent",
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

    if mask.dtype != torch.bool:
        raise ValueError(
            f"{label} transport_mask must be boolean"
        )

    if not torch.equal(mask, count > 0):
        raise ValueError(
            f"{label} mask/count contract failed"
        )

    target_shape = tuple(
        state["target_latent"].shape
    )

    for key in (
        "correct_fused_latent",
        "shuffled_fused_latent",
    ):
        tensor = state[key]

        if tuple(tensor.shape) != target_shape:
            raise ValueError(
                f"{label} {key} shape mismatch"
            )

        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"{label} {key} contains NaN or Inf"
            )

    return state


def exact_common_input_check(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    label: str,
) -> None:
    for key in (
        "latent_frame_indices",
        "source_latent",
        "target_latent",
    ):
        if not torch.equal(
            reference[key],
            candidate[key],
        ):
            raise RuntimeError(
                f"{label} differs from Hard in {key}"
            )


def full_mae(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    return float(
        torch.abs(
            first.to(torch.float32)
            - second.to(torch.float32)
        ).mean()
    )


def save_grid_snapshots(
    frames,
    *,
    output_dir: Path,
    prefix: str,
) -> dict[str, str]:
    paths = {}

    for label, frame_index in (
        ("first", 0),
        ("middle", len(frames) // 2),
        ("final", len(frames) - 1),
    ):
        path = (
            output_dir
            / f"{prefix}_{label}.png"
        )

        Image.fromarray(
            frames[frame_index]
        ).save(path)

        paths[label] = str(path)

    return paths


def main() -> None:
    args = parse_args()

    hard_path = args.hard_artifact.resolve()
    soft_replace_path = (
        args.soft_replace_artifact.resolve()
    )
    soft_hardmask_path = (
        args.soft_hardmask_artifact.resolve()
    )
    soft_blend_path = (
        args.soft_blend_artifact.resolve()
    )
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_dir}"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    checkpoint_hash = sha256(checkpoint)

    if checkpoint_hash != EXPECTED_VAE_SHA256:
        raise ValueError(
            "Wan VAE SHA256 mismatch: "
            f"{checkpoint_hash}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    hard = require_artifact(
        hard_path,
        label="Hard",
    )

    soft_replace = require_artifact(
        soft_replace_path,
        label="Soft Replace",
    )

    soft_hardmask = require_artifact(
        soft_hardmask_path,
        label="Soft HardMask",
    )

    soft_blend = require_artifact(
        soft_blend_path,
        label="Soft Blend",
    )

    for candidate, label in (
        (soft_replace, "Soft Replace"),
        (soft_hardmask, "Soft HardMask"),
        (soft_blend, "Soft Blend"),
    ):
        exact_common_input_check(
            hard,
            candidate,
            label=label,
        )

    output_dir.mkdir(parents=True)

    latent_variants = {
        "original": hard["target_latent"],
        "hard_correct": hard[
            "correct_fused_latent"
        ],
        "hard_shuffled": hard[
            "shuffled_fused_latent"
        ],
        "soft_replace_correct": soft_replace[
            "correct_fused_latent"
        ],
        "soft_replace_shuffled": soft_replace[
            "shuffled_fused_latent"
        ],
        "soft_hardmask_correct": soft_hardmask[
            "correct_fused_latent"
        ],
        "soft_hardmask_shuffled": soft_hardmask[
            "shuffled_fused_latent"
        ],
        "soft_blend_correct": soft_blend[
            "correct_fused_latent"
        ],
        "soft_blend_shuffled": soft_blend[
            "shuffled_fused_latent"
        ],
    }

    expected_shape = tuple(
        hard["target_latent"].shape
    )

    for name, latent in latent_variants.items():
        if tuple(latent.shape) != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"{tuple(latent.shape)}"
            )

        if not bool(torch.isfinite(latent).all()):
            raise RuntimeError(
                f"{name} contains NaN or Inf"
            )

    device = torch.device("cuda:0")

    torch.cuda.reset_peak_memory_stats()

    total_started = time.perf_counter()

    load_started = time.perf_counter()

    codec = RealWonderWanVAECodec(
        checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )

    load_seconds = (
        time.perf_counter() - load_started
    )

    decoded = {}
    decode_seconds = {}

    for name, latent in latent_variants.items():
        torch.cuda.synchronize()
        started = time.perf_counter()

        result = codec.decode_latents(
            latent.to(
                device=device,
                dtype=torch.float32,
            )
        )

        torch.cuda.synchronize()

        decode_seconds[name] = (
            time.perf_counter() - started
        )

        decoded[name] = pixels_01(
            result.cpu()
        )

        del result
        codec.clear_cache()
        torch.cuda.empty_cache()

    videos = {
        name: uint8_frames(frames)
        for name, frames in decoded.items()
    }

    video_paths = {}

    for name, frames in videos.items():
        path = output_dir / f"{name}.mp4"

        write_video(
            path,
            frames,
            fps=args.fps,
        )

        video_paths[name] = str(path)

    quality_grid = make_comparison_grid(
        [
            (
                "Original coarse VAE",
                videos["original"],
            ),
            (
                "Hard Correct",
                videos["hard_correct"],
            ),
            (
                "Soft Replace",
                videos[
                    "soft_replace_correct"
                ],
            ),
            (
                "Soft HardMask",
                videos[
                    "soft_hardmask_correct"
                ],
            ),
            (
                "Soft Blend alpha=0.5",
                videos[
                    "soft_blend_correct"
                ],
            ),
        ]
    )

    quality_video_path = (
        output_dir
        / "quality_comparison.mp4"
    )

    write_video(
        quality_video_path,
        quality_grid,
        fps=args.fps,
    )

    quality_snapshots = save_grid_snapshots(
        quality_grid,
        output_dir=output_dir,
        prefix="quality",
    )

    identity_grid = make_comparison_grid(
        [
            (
                "Original coarse VAE",
                videos["original"],
            ),
            (
                "Hard Correct",
                videos["hard_correct"],
            ),
            (
                "Hard Shuffled",
                videos["hard_shuffled"],
            ),
            (
                "Blend Correct",
                videos[
                    "soft_blend_correct"
                ],
            ),
            (
                "Blend Shuffled",
                videos[
                    "soft_blend_shuffled"
                ],
            ),
        ]
    )

    identity_video_path = (
        output_dir
        / "identity_comparison.mp4"
    )

    write_video(
        identity_video_path,
        identity_grid,
        fps=args.fps,
    )

    identity_snapshots = save_grid_snapshots(
        identity_grid,
        output_dir=output_dir,
        prefix="identity",
    )

    pairwise_decoded_mae = {
        "hard_correct_vs_original": full_mae(
            decoded["hard_correct"],
            decoded["original"],
        ),
        "soft_replace_correct_vs_original":
            full_mae(
                decoded[
                    "soft_replace_correct"
                ],
                decoded["original"],
            ),
        "soft_hardmask_correct_vs_original":
            full_mae(
                decoded[
                    "soft_hardmask_correct"
                ],
                decoded["original"],
            ),
        "soft_blend_correct_vs_original":
            full_mae(
                decoded[
                    "soft_blend_correct"
                ],
                decoded["original"],
            ),
        "hard_correct_vs_soft_hardmask":
            full_mae(
                decoded["hard_correct"],
                decoded[
                    "soft_hardmask_correct"
                ],
            ),
        "soft_replace_vs_soft_hardmask":
            full_mae(
                decoded[
                    "soft_replace_correct"
                ],
                decoded[
                    "soft_hardmask_correct"
                ],
            ),
        "hard_correct_vs_hard_shuffled":
            full_mae(
                decoded["hard_correct"],
                decoded["hard_shuffled"],
            ),
        "blend_correct_vs_blend_shuffled":
            full_mae(
                decoded[
                    "soft_blend_correct"
                ],
                decoded[
                    "soft_blend_shuffled"
                ],
            ),
    }

    report = {
        "stage": (
            "soft_fusion_ablation_vae_decode"
        ),
        "inputs": {
            "hard": {
                "path": str(hard_path),
                "sha256": sha256(hard_path),
            },
            "soft_replace": {
                "path": str(soft_replace_path),
                "sha256": sha256(
                    soft_replace_path
                ),
            },
            "soft_hardmask": {
                "path": str(soft_hardmask_path),
                "sha256": sha256(
                    soft_hardmask_path
                ),
            },
            "soft_blend": {
                "path": str(soft_blend_path),
                "sha256": sha256(
                    soft_blend_path
                ),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
            },
        },
        "fairness": {
            "source_latents_exactly_equal": True,
            "target_latents_exactly_equal": True,
            "latent_frame_indices_exactly_equal": True,
            "hard_mask_cells": int(
                hard["transport_mask"].sum()
            ),
            "soft_replace_mask_cells": int(
                soft_replace[
                    "transport_mask"
                ].sum()
            ),
            "soft_hardmask_mask_cells": int(
                soft_hardmask[
                    "transport_mask"
                ].sum()
            ),
            "soft_blend_mask_cells": int(
                soft_blend[
                    "transport_mask"
                ].sum()
            ),
        },
        "latent_statistics": {
            name: tensor_stats(latent)
            for name, latent in latent_variants.items()
        },
        "decoded_statistics": {
            name: tensor_stats(frames)
            for name, frames in decoded.items()
        },
        "pairwise_decoded_full_frame_mae": (
            pairwise_decoded_mae
        ),
        "runtime_seconds": {
            "model_load": load_seconds,
            "decode": decode_seconds,
            "total": (
                time.perf_counter()
                - total_started
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
        "artifacts": {
            "individual_videos": video_paths,
            "quality_comparison": str(
                quality_video_path
            ),
            "quality_snapshots": (
                quality_snapshots
            ),
            "identity_comparison": str(
                identity_video_path
            ),
            "identity_snapshots": (
                identity_snapshots
            ),
        },
        "interpretation_boundary": (
            "Original is the coarse simulation VAE reconstruction, "
            "not real future-video ground truth."
        ),
    }

    report_path = output_dir / "report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "report": str(report_path),
                "fairness": report["fairness"],
                "pairwise_decoded_full_frame_mae":
                    pairwise_decoded_mae,
                "quality_comparison": str(
                    quality_video_path
                ),
                "identity_comparison": str(
                    identity_video_path
                ),
                "runtime_seconds": report[
                    "runtime_seconds"
                ],
                "gpu_peak_memory_mib": report[
                    "gpu_peak_memory_mib"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
