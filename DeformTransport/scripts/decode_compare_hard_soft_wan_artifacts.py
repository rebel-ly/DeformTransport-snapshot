"""Decode and compare Hard and Soft Wan latent transport artifacts.

The script performs no new VAE encoding and no diffusion generation. It decodes
the exact latent tensors already stored in the verified Hard and Soft artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from deform_transport.transport_payloads import (
    load_realwonder_rgb_crop,
)
from deform_transport.transport_ready import (
    validate_transport_ready,
)
from deform_transport.wan_vae_codec import (
    RealWonderWanVAECodec,
)


EXPECTED_VAE_SHA256 = (
    "38071ab59bd94681c686fa51d75a1968f"
    "64e470262043be31f7a094e442fd981"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transport-ready",
        type=Path,
        required=True,
    )
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def pixels_01(decoded: torch.Tensor) -> torch.Tensor:
    return (
        decoded[0]
        .mul(0.5)
        .add(0.5)
        .clamp(0.0, 1.0)
    )


def uint8_frames(frames: torch.Tensor) -> np.ndarray:
    return (
        frames.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def write_video(
    path: Path,
    frames: np.ndarray,
    *,
    fps: int,
) -> None:
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
    )

    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def load_coarse_pixels(
    paths: list[str],
) -> torch.Tensor:
    frames = [
        load_realwonder_rgb_crop(path)
        for path in paths
    ]

    return (
        torch.stack(frames)
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
        .mul(2.0)
        .sub(1.0)
    )


def expand_latent_mask_to_pixel_frames(
    mask: torch.Tensor,
    pixel_frames: int,
) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(
            "latent mask must have shape [Tz,1,H,W]"
        )

    expanded = [mask[:1]]

    for latent_index in range(1, mask.shape[0]):
        expanded.append(
            mask[
                latent_index : latent_index + 1
            ].repeat(4, 1, 1, 1)
        )

    return torch.cat(expanded, dim=0)[:pixel_frames]


def spatially_expand_mask(
    mask: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    return F.interpolate(
        mask.to(torch.float32),
        size=(height, width),
        mode="nearest",
    ).to(torch.bool)


def masked_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target shapes differ"
        )

    if mask.shape != (
        prediction.shape[0],
        1,
        prediction.shape[2],
        prediction.shape[3],
    ):
        raise ValueError(
            f"unexpected mask shape: {mask.shape}"
        )

    selected = mask.expand_as(prediction)

    if not bool(selected.any()):
        return {
            "selected_values": 0,
            "mae": None,
            "mse": None,
            "psnr_db": None,
        }

    difference = (
        prediction.to(torch.float32)
        - target.to(torch.float32)
    )[selected]

    mse = float(difference.square().mean())

    return {
        "selected_values": int(selected.sum()),
        "mae": float(difference.abs().mean()),
        "mse": mse,
        "psnr_db": (
            float(-10.0 * np.log10(mse))
            if mse > 0
            else None
        ),
    }


def latent_masked_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(
            "latent prediction and target shapes differ"
        )

    expanded = mask.unsqueeze(0).expand_as(
        prediction
    )

    if not bool(expanded.any()):
        return {
            "selected_values": 0,
            "mae": None,
            "mse": None,
        }

    difference = (
        prediction.to(torch.float32)
        - target.to(torch.float32)
    )[expanded]

    return {
        "selected_values": int(expanded.sum()),
        "mae": float(difference.abs().mean()),
        "mse": float(difference.square().mean()),
    }


def pairwise_error(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    return masked_error(
        first,
        second,
        mask,
    )


def tensor_stats(
    tensor: torch.Tensor,
) -> dict[str, Any]:
    value = tensor.detach().to(torch.float32)

    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
        "finite": bool(torch.isfinite(value).all()),
    }


def make_comparison_grid(
    labeled_frames: list[
        tuple[str, np.ndarray]
    ],
) -> np.ndarray:
    frame_count = len(labeled_frames[0][1])
    outputs = []

    panel_width = 320
    panel_height = 184
    label_height = 28

    for frame_index in range(frame_count):
        panels = []

        for label, frames in labeled_frames:
            image = Image.fromarray(
                frames[frame_index]
            ).resize(
                (panel_width, panel_height),
                resample=Image.Resampling.BILINEAR,
            )

            panel = Image.new(
                "RGB",
                (
                    panel_width,
                    panel_height + label_height,
                ),
                "black",
            )

            panel.paste(image, (0, label_height))

            ImageDraw.Draw(panel).text(
                (8, 7),
                label,
                fill="white",
            )

            panels.append(np.asarray(panel))

        blank = np.zeros_like(panels[0])

        row1 = np.concatenate(
            panels[:3],
            axis=1,
        )

        row2 = np.concatenate(
            [
                panels[3],
                panels[4],
                blank,
            ],
            axis=1,
        )

        outputs.append(
            np.concatenate(
                [row1, row2],
                axis=0,
            )
        )

    return np.stack(outputs)


def main() -> None:
    args = parse_args()

    transport_path = args.transport_ready.resolve()
    hard_path = args.hard_artifact.resolve()
    soft_path = args.soft_artifact.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    for path in (
        transport_path,
        hard_path,
        soft_path,
        checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint_hash = sha256(checkpoint)

    if checkpoint_hash != EXPECTED_VAE_SHA256:
        raise ValueError(
            "Wan VAE SHA256 mismatch: "
            f"{checkpoint_hash}"
        )

    output_dir.mkdir(parents=True)

    state = torch.load(
        transport_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_transport_ready(state)

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

    required_hard = {
        "latent_frame_indices",
        "source_latent",
        "target_latent",
        "correct_fused_latent",
        "shuffled_fused_latent",
        "transport_mask",
        "contribution_count",
    }

    required_soft = {
        "latent_frame_indices",
        "source_latent",
        "target_latent",
        "correct_fused_latent",
        "shuffled_fused_latent",
        "transport_mask",
        "contribution_count",
        "soft_only_mask",
    }

    missing_hard = sorted(
        required_hard - set(hard)
    )
    missing_soft = sorted(
        required_soft - set(soft)
    )

    if missing_hard:
        raise ValueError(
            f"Hard artifact missing: {missing_hard}"
        )

    if missing_soft:
        raise ValueError(
            f"Soft artifact missing: {missing_soft}"
        )

    if not torch.equal(
        hard["latent_frame_indices"],
        soft["latent_frame_indices"],
    ):
        raise RuntimeError(
            "Hard and Soft latent frame mappings differ"
        )

    if not torch.equal(
        hard["source_latent"],
        soft["source_latent"],
    ):
        raise RuntimeError(
            "Hard and Soft source latents differ"
        )

    if not torch.equal(
        hard["target_latent"],
        soft["target_latent"],
    ):
        raise RuntimeError(
            "Hard and Soft target latents differ"
        )

    if not torch.equal(
        hard["transport_mask"],
        hard["contribution_count"] > 0,
    ):
        raise RuntimeError(
            "Hard mask/count contract failed"
        )

    if not torch.equal(
        soft["transport_mask"],
        soft["contribution_count"] > 0,
    ):
        raise RuntimeError(
            "Soft mask/count contract failed"
        )

    hard_mask = hard["transport_mask"]
    soft_mask = soft["transport_mask"]

    if bool((hard_mask & ~soft_mask).any()):
        raise RuntimeError(
            "Hard support is not a subset of Soft support"
        )

    retained_soft_only = (
        soft_mask
        & soft["soft_only_mask"]
    )

    target_latent = hard["target_latent"]

    latent_variants = {
        "original": target_latent,
        "hard_correct": hard[
            "correct_fused_latent"
        ],
        "hard_shuffled": hard[
            "shuffled_fused_latent"
        ],
        "soft_correct": soft[
            "correct_fused_latent"
        ],
        "soft_shuffled": soft[
            "shuffled_fused_latent"
        ],
    }

    expected_shape = tuple(target_latent.shape)

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

    coarse_pixels = load_coarse_pixels(
        state["paths"]["coarse_rgb_frames"]
    )

    coarse_pixels_01 = (
        coarse_pixels[0]
        .permute(1, 0, 2, 3)
        .mul(0.5)
        .add(0.5)
        .clamp(0.0, 1.0)
    )

    pixel_frame_count = int(
        coarse_pixels.shape[2]
    )

    hard_pixel_mask = spatially_expand_mask(
        expand_latent_mask_to_pixel_frames(
            hard_mask,
            pixel_frame_count,
        ),
        height=480,
        width=832,
    )

    soft_pixel_mask = spatially_expand_mask(
        expand_latent_mask_to_pixel_frames(
            soft_mask,
            pixel_frame_count,
        ),
        height=480,
        width=832,
    )

    retained_soft_only_pixel_mask = (
        spatially_expand_mask(
            expand_latent_mask_to_pixel_frames(
                retained_soft_only,
                pixel_frame_count,
            ),
            height=480,
            width=832,
        )
    )

    full_pixel_mask = torch.ones(
        (
            pixel_frame_count,
            1,
            480,
            832,
        ),
        dtype=torch.bool,
    )

    device = torch.device("cuda:0")

    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()

    codec = RealWonderWanVAECodec(
        checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )

    load_seconds = (
        time.perf_counter() - started
    )

    decoded = {}
    decode_seconds = {}

    for name, latent in latent_variants.items():
        torch.cuda.synchronize()
        decode_started = time.perf_counter()

        result = codec.decode_latents(
            latent.to(
                device=device,
                dtype=torch.float32,
            )
        )

        torch.cuda.synchronize()

        decode_seconds[name] = (
            time.perf_counter()
            - decode_started
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

    output_paths = {}

    for name, frames in videos.items():
        path = output_dir / f"{name}.mp4"

        write_video(
            path,
            frames,
            fps=args.fps,
        )

        output_paths[name] = str(path)

    comparison = make_comparison_grid(
        [
            (
                "Original coarse VAE recon",
                videos["original"],
            ),
            (
                "Hard Correct",
                videos["hard_correct"],
            ),
            (
                "Soft Correct gate=0.25",
                videos["soft_correct"],
            ),
            (
                "Hard Shuffled",
                videos["hard_shuffled"],
            ),
            (
                "Soft Shuffled gate=0.25",
                videos["soft_shuffled"],
            ),
        ]
    )

    comparison_path = (
        output_dir
        / "hard_soft_comparison.mp4"
    )

    write_video(
        comparison_path,
        comparison,
        fps=args.fps,
    )

    for label, frame_index in (
        ("first", 0),
        ("middle", len(comparison) // 2),
        ("final", len(comparison) - 1),
    ):
        Image.fromarray(
            comparison[frame_index]
        ).save(
            output_dir
            / f"comparison_{label}.png"
        )

    pixel_proxy = {}

    for name, frames in decoded.items():
        pixel_proxy[name] = {
            "full_frame_vs_coarse": masked_error(
                frames,
                coarse_pixels_01,
                full_pixel_mask,
            ),
            "hard_support_vs_coarse": masked_error(
                frames,
                coarse_pixels_01,
                hard_pixel_mask,
            ),
            "soft_support_vs_coarse": masked_error(
                frames,
                coarse_pixels_01,
                soft_pixel_mask,
            ),
        }

    pixel_proxy["soft_correct"][
        "retained_soft_only_vs_coarse"
    ] = masked_error(
        decoded["soft_correct"],
        coarse_pixels_01,
        retained_soft_only_pixel_mask,
    )

    pixel_proxy["soft_shuffled"][
        "retained_soft_only_vs_coarse"
    ] = masked_error(
        decoded["soft_shuffled"],
        coarse_pixels_01,
        retained_soft_only_pixel_mask,
    )

    latent_proxy = {
        "hard_correct_on_hard_support":
            latent_masked_error(
                latent_variants["hard_correct"],
                target_latent,
                hard_mask,
            ),
        "soft_correct_on_hard_support":
            latent_masked_error(
                latent_variants["soft_correct"],
                target_latent,
                hard_mask,
            ),
        "hard_shuffled_on_hard_support":
            latent_masked_error(
                latent_variants["hard_shuffled"],
                target_latent,
                hard_mask,
            ),
        "soft_shuffled_on_hard_support":
            latent_masked_error(
                latent_variants["soft_shuffled"],
                target_latent,
                hard_mask,
            ),
        "soft_correct_on_retained_soft_only":
            latent_masked_error(
                latent_variants["soft_correct"],
                target_latent,
                retained_soft_only,
            ),
        "soft_shuffled_on_retained_soft_only":
            latent_masked_error(
                latent_variants["soft_shuffled"],
                target_latent,
                retained_soft_only,
            ),
    }

    pairwise = {
        "decoded_hard_correct_vs_soft_correct_full":
            pairwise_error(
                decoded["hard_correct"],
                decoded["soft_correct"],
                full_pixel_mask,
            ),
        "decoded_hard_correct_vs_soft_correct_hard_support":
            pairwise_error(
                decoded["hard_correct"],
                decoded["soft_correct"],
                hard_pixel_mask,
            ),
        "decoded_soft_correct_vs_soft_shuffled":
            pairwise_error(
                decoded["soft_correct"],
                decoded["soft_shuffled"],
                soft_pixel_mask,
            ),
        "decoded_hard_correct_vs_hard_shuffled":
            pairwise_error(
                decoded["hard_correct"],
                decoded["hard_shuffled"],
                hard_pixel_mask,
            ),
    }

    report = {
        "stage": "hard_soft_wan_vae_decode_comparison",
        "case": state["case_name"],
        "inputs": {
            "transport_ready": {
                "path": str(transport_path),
                "sha256": sha256(transport_path),
            },
            "hard_artifact": {
                "path": str(hard_path),
                "sha256": sha256(hard_path),
            },
            "soft_artifact": {
                "path": str(soft_path),
                "sha256": sha256(soft_path),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
            },
        },
        "fairness": {
            "latent_frame_indices_equal": True,
            "source_latents_exactly_equal": True,
            "target_latents_exactly_equal": True,
            "hard_support_subset_of_soft": True,
            "hard_mask_cells": int(
                hard_mask.sum()
            ),
            "soft_mask_cells": int(
                soft_mask.sum()
            ),
            "retained_soft_only_cells": int(
                retained_soft_only.sum()
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
        "latent_proxy_metrics": latent_proxy,
        "decoded_coarse_rgb_proxy_metrics": (
            pixel_proxy
        ),
        "pairwise_decoded_differences": pairwise,
        "runtime_seconds": {
            "model_load": load_seconds,
            "decode": decode_seconds,
            "total": (
                time.perf_counter()
                - started
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
            "videos": output_paths,
            "comparison_video": str(
                comparison_path
            ),
            "comparison_first": str(
                output_dir
                / "comparison_first.png"
            ),
            "comparison_middle": str(
                output_dir
                / "comparison_middle.png"
            ),
            "comparison_final": str(
                output_dir
                / "comparison_final.png"
            ),
        },
        "interpretation_boundary": (
            "The coarse RGB input and its VAE latent are geometry-aligned "
            "proxies, not real future-video ground truth. Lower proxy error "
            "does not by itself prove better future-video quality."
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
                "latent_proxy_metrics": latent_proxy,
                "pairwise_decoded_differences": pairwise,
                "runtime_seconds": report[
                    "runtime_seconds"
                ],
                "gpu_peak_memory_mib": report[
                    "gpu_peak_memory_mib"
                ],
                "comparison_video": str(
                    comparison_path
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
