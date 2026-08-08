"""Decode a precomputed Wan transport artifact without rebuilding transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.wan_vae_codec import RealWonderWanVAECodec  # noqa: E402


EXPECTED_VAE_SHA256 = (
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--artifact",
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
        "--evaluation-mask-artifact",
        type=Path,
        default=None,
        help=(
            "Optional artifact whose transport_mask is used as a common "
            "evaluation region across different transport contracts."
        ),
    )
    parser.add_argument(
        "--label",
        default="transport",
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


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()

    digest = hashlib.sha256()
    digest.update(value.tobytes())

    return digest.hexdigest()


def gpu_used_memory_mib() -> int | None:
    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        "",
    ).strip()

    physical_id = (
        visible.split(",", maxsplit=1)[0].strip()
        or "0"
    )

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={physical_id}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        return int(
            result.stdout.strip().splitlines()[0]
        )
    except (
        FileNotFoundError,
        IndexError,
        subprocess.SubprocessError,
        ValueError,
    ):
        return None


def timed_cuda(function):
    torch.cuda.synchronize()
    started = time.perf_counter()

    output = function()

    torch.cuda.synchronize()

    return output, time.perf_counter() - started


def pixels_01(decoded: torch.Tensor) -> torch.Tensor:
    if decoded.ndim != 5:
        raise ValueError(
            "decoded pixels must have shape [B,T,C,H,W]"
        )

    return (
        decoded[0]
        .mul(0.5)
        .add(0.5)
        .clamp(0.0, 1.0)
        .contiguous()
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


def labeled_three_panel(
    frames_by_label: list[
        tuple[str, np.ndarray]
    ],
) -> np.ndarray:
    frame_count = len(frames_by_label[0][1])

    if any(
        len(frames) != frame_count
        for _, frames in frames_by_label
    ):
        raise ValueError(
            "comparison videos have different frame counts"
        )

    outputs = []

    for frame_index in range(frame_count):
        panels = []

        for label, frames in frames_by_label:
            image = Image.fromarray(
                frames[frame_index]
            ).resize(
                (416, 240),
                resample=Image.Resampling.BILINEAR,
            )

            panel = Image.new(
                "RGB",
                (416, 272),
                "black",
            )

            panel.paste(image, (0, 32))

            ImageDraw.Draw(panel).text(
                (8, 9),
                label,
                fill="white",
            )

            panels.append(
                np.asarray(panel)
            )

        outputs.append(
            np.concatenate(
                panels,
                axis=1,
            )
        )

    return np.stack(outputs)


def expand_latent_mask(
    mask: torch.Tensor,
    pixel_frames: int,
) -> torch.Tensor:
    mask = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device="cpu",
    )

    if (
        mask.ndim != 4
        or mask.shape[1] != 1
    ):
        raise ValueError(
            "latent mask must have shape [Tz,1,H,W]"
        )

    expanded = [mask[:1]]

    for latent_index in range(
        1,
        mask.shape[0],
    ):
        expanded.append(
            mask[
                latent_index : latent_index + 1
            ].repeat(
                4,
                1,
                1,
                1,
            )
        )

    value = torch.cat(
        expanded,
        dim=0,
    )

    if value.shape[0] < pixel_frames:
        raise RuntimeError(
            "expanded latent mask has too few pixel frames"
        )

    return value[:pixel_frames]


def pixel_mask(
    latent_mask: torch.Tensor,
    *,
    pixel_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    expanded = expand_latent_mask(
        latent_mask,
        pixel_frames,
    )

    return F.interpolate(
        expanded.float(),
        size=(height, width),
        mode="nearest",
    ).bool()


def tensor_stats(
    tensor: torch.Tensor,
) -> dict[str, Any]:
    value = tensor.detach().float()

    return {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
        "finite": bool(
            torch.isfinite(value).all()
        ),
    }


def masked_frame_error_metrics(
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
            "pixel mask shape does not match frames"
        )

    frames = []

    for frame_index in range(
        prediction.shape[0]
    ):
        expanded = mask[
            frame_index
        ].expand(
            prediction.shape[1],
            -1,
            -1,
        )

        selected = (
            prediction[frame_index]
            - target[frame_index]
        )[expanded]

        if selected.numel() == 0:
            mae = 0.0
            mse = 0.0
            psnr = None
        else:
            mae = float(
                selected.abs().mean()
            )
            mse = float(
                selected.square().mean()
            )
            psnr = (
                float(
                    10.0
                    * torch.log10(
                        torch.tensor(
                            1.0 / mse
                        )
                    )
                )
                if mse > 0
                else None
            )

        frames.append(
            {
                "frame": frame_index,
                "support_pixels": int(
                    mask[frame_index].sum()
                ),
                "masked_l1": mae,
                "masked_mse": mse,
                "masked_psnr_db": psnr,
            }
        )

    valid_psnr = [
        item["masked_psnr_db"]
        for item in frames
        if item["masked_psnr_db"] is not None
    ]

    return {
        "frames": frames,
        "aggregate": {
            "valid_frames": len(frames),
            "mean_masked_l1": float(
                np.mean(
                    [
                        item["masked_l1"]
                        for item in frames
                    ]
                )
            ),
            "mean_masked_mse": float(
                np.mean(
                    [
                        item["masked_mse"]
                        for item in frames
                    ]
                )
            ),
            "mean_masked_psnr_db": (
                float(np.mean(valid_psnr))
                if valid_psnr
                else None
            ),
        },
    }


def grayscale(
    frames: torch.Tensor,
) -> torch.Tensor:
    weights = torch.tensor(
        [0.2989, 0.5870, 0.1140],
        dtype=frames.dtype,
        device=frames.device,
    ).view(1, 3, 1, 1)

    return (
        frames * weights
    ).sum(
        dim=1,
        keepdim=True,
    )


def sharpness_metrics(
    frames: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    gray = grayscale(
        frames.float()
    )

    laplacian_kernel = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)

    sobel_x_kernel = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)

    sobel_y_kernel = torch.tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)

    laplacian = F.conv2d(
        gray,
        laplacian_kernel,
        padding=1,
    )

    sobel_x = F.conv2d(
        gray,
        sobel_x_kernel,
        padding=1,
    )

    sobel_y = F.conv2d(
        gray,
        sobel_y_kernel,
        padding=1,
    )

    sobel = torch.sqrt(
        sobel_x.square()
        + sobel_y.square()
        + 1e-12
    )

    low_frequency = F.avg_pool2d(
        gray,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    high_frequency = (
        gray - low_frequency
    ).abs()

    per_frame = []

    for frame_index in range(
        frames.shape[0]
    ):
        selected = mask[
            frame_index,
            0,
        ]

        if not bool(selected.any()):
            lap_var = 0.0
            sobel_mean = 0.0
            high_frequency_mean = 0.0
        else:
            lap_values = laplacian[
                frame_index,
                0,
            ][selected]

            lap_var = float(
                lap_values.var(
                    unbiased=False
                )
            )

            sobel_mean = float(
                sobel[
                    frame_index,
                    0,
                ][selected].mean()
            )

            high_frequency_mean = float(
                high_frequency[
                    frame_index,
                    0,
                ][selected].mean()
            )

        per_frame.append(
            {
                "frame": frame_index,
                "laplacian_variance": lap_var,
                "sobel_mean": sobel_mean,
                "high_frequency_mean": (
                    high_frequency_mean
                ),
            }
        )

    return {
        "frames": per_frame,
        "aggregate": {
            "mean_laplacian_variance": float(
                np.mean(
                    [
                        item[
                            "laplacian_variance"
                        ]
                        for item in per_frame
                    ]
                )
            ),
            "mean_sobel": float(
                np.mean(
                    [
                        item["sobel_mean"]
                        for item in per_frame
                    ]
                )
            ),
            "mean_high_frequency": float(
                np.mean(
                    [
                        item[
                            "high_frequency_mean"
                        ]
                        for item in per_frame
                    ]
                )
            ),
        },
    }


def temporal_metrics(
    frames: torch.Tensor,
) -> dict[str, float]:
    first = (
        frames[1:] - frames[:-1]
    ).abs()

    second = (
        frames[2:]
        - 2.0 * frames[1:-1]
        + frames[:-2]
    ).abs()

    return {
        "mean_first_order_abs_difference": (
            float(first.mean())
            if first.numel()
            else 0.0
        ),
        "mean_second_order_abs_difference": (
            float(second.mean())
            if second.numel()
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()

    artifact_path = args.artifact.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    evaluation_artifact_path = (
        args.evaluation_mask_artifact.resolve()
        if args.evaluation_mask_artifact
        is not None
        else artifact_path
    )

    for path in (
        artifact_path,
        checkpoint_path,
        evaluation_artifact_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "refusing to use non-empty output directory: "
                f"{output_dir}"
            )
    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    checkpoint_hash = sha256(
        checkpoint_path
    )

    if (
        checkpoint_hash
        != EXPECTED_VAE_SHA256
    ):
        raise ValueError(
            "Wan VAE SHA256 mismatch: "
            f"{checkpoint_hash}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Wan VAE decoding"
        )

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )

    evaluation_artifact = torch.load(
        evaluation_artifact_path,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "target_latent",
        "correct_fused_latent",
        "shuffled_fused_latent",
        "transport_mask",
        "latent_frame_indices",
    }

    missing = sorted(
        required - set(artifact)
    )

    if missing:
        raise ValueError(
            f"artifact missing keys: {missing}"
        )

    if "transport_mask" not in evaluation_artifact:
        raise ValueError(
            "evaluation mask artifact is missing transport_mask"
        )

    target_latent = artifact[
        "target_latent"
    ].float().contiguous()

    correct_latent = artifact[
        "correct_fused_latent"
    ].float().contiguous()

    shuffled_latent = artifact[
        "shuffled_fused_latent"
    ].float().contiguous()

    native_mask = artifact[
        "transport_mask"
    ].bool().contiguous()

    evaluation_mask = evaluation_artifact[
        "transport_mask"
    ].bool().contiguous()

    expected_shape = (
        1,
        21,
        16,
        60,
        104,
    )

    for name, latent in (
        ("target_latent", target_latent),
        ("correct_fused_latent", correct_latent),
        ("shuffled_fused_latent", shuffled_latent),
    ):
        if tuple(latent.shape) != expected_shape:
            raise ValueError(
                f"{name} has unexpected shape: "
                f"{tuple(latent.shape)}"
            )

        if not bool(
            torch.isfinite(latent).all()
        ):
            raise ValueError(
                f"{name} contains non-finite values"
            )

    expected_mask_shape = (
        21,
        1,
        60,
        104,
    )

    if tuple(native_mask.shape) != expected_mask_shape:
        raise ValueError(
            "native transport mask has unexpected shape"
        )

    if tuple(evaluation_mask.shape) != expected_mask_shape:
        raise ValueError(
            "evaluation transport mask has unexpected shape"
        )

    indices = artifact[
        "latent_frame_indices"
    ].to(
        dtype=torch.long,
        device="cpu",
    )

    expected_indices = torch.arange(
        0,
        81,
        4,
        dtype=torch.long,
    )

    if not torch.equal(
        indices,
        expected_indices,
    ):
        raise ValueError(
            "latent frame indices are not [0,4,...,80]"
        )

    torch.cuda.reset_peak_memory_stats()

    gpu_before = gpu_used_memory_mib()
    overall_started = time.perf_counter()

    codec, model_load_seconds = timed_cuda(
        lambda: RealWonderWanVAECodec(
            checkpoint_path,
            device="cuda",
            dtype=torch.bfloat16,
        )
    )

    def decode_cpu(
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        decoded, seconds = timed_cuda(
            lambda: codec.decode_latents(
                latent
            )
        )

        decoded_cpu = (
            decoded.detach()
            .cpu()
            .contiguous()
        )

        del decoded

        codec.clear_cache()
        torch.cuda.empty_cache()

        return decoded_cpu, seconds

    target_decoded, target_decode_seconds = (
        decode_cpu(target_latent)
    )

    correct_decoded, correct_decode_seconds = (
        decode_cpu(correct_latent)
    )

    shuffled_decoded, shuffled_decode_seconds = (
        decode_cpu(shuffled_latent)
    )

    target_01 = pixels_01(
        target_decoded
    )

    correct_01 = pixels_01(
        correct_decoded
    )

    shuffled_01 = pixels_01(
        shuffled_decoded
    )

    expected_decoded_shape = (
        1,
        81,
        3,
        480,
        832,
    )

    for name, decoded in (
        ("target", target_decoded),
        ("correct", correct_decoded),
        ("shuffled", shuffled_decoded),
    ):
        if tuple(decoded.shape) != expected_decoded_shape:
            raise RuntimeError(
                f"{name} decoded shape is unexpected: "
                f"{tuple(decoded.shape)}"
            )

    native_pixel_mask = pixel_mask(
        native_mask,
        pixel_frames=81,
        height=480,
        width=832,
    )

    evaluation_pixel_mask = pixel_mask(
        evaluation_mask,
        pixel_frames=81,
        height=480,
        width=832,
    )

    full_pixel_mask = torch.ones(
        (81, 1, 480, 832),
        dtype=torch.bool,
    )

    target_uint8 = uint8_frames(
        target_01
    )

    correct_uint8 = uint8_frames(
        correct_01
    )

    shuffled_uint8 = uint8_frames(
        shuffled_01
    )

    videos = {
        "target_reconstruction": target_uint8,
        "correct_fused": correct_uint8,
        "shuffled_fused": shuffled_uint8,
    }

    video_paths = {}

    for name, frames in videos.items():
        path = output_dir / f"{name}.mp4"

        write_video(
            path,
            frames,
            args.fps,
        )

        video_paths[name] = str(path)

    comparison_frames = labeled_three_panel(
        [
            (
                "Target VAE reconstruction",
                target_uint8,
            ),
            (
                f"{args.label} Correct",
                correct_uint8,
            ),
            (
                f"{args.label} Shuffled",
                shuffled_uint8,
            ),
        ]
    )

    comparison_video_path = (
        output_dir
        / "target_correct_shuffled.mp4"
    )

    write_video(
        comparison_video_path,
        comparison_frames,
        args.fps,
    )

    selected_images = {}

    for name, frame_index in (
        ("first", 0),
        ("quarter", 20),
        ("mid", 40),
        ("three_quarter", 60),
        ("final", 80),
    ):
        path = (
            output_dir
            / f"comparison_{name}.png"
        )

        Image.fromarray(
            comparison_frames[
                frame_index
            ]
        ).save(path)

        selected_images[name] = str(path)

    metrics = {
        "full_frame": {
            "correct_vs_target": (
                masked_frame_error_metrics(
                    correct_01,
                    target_01,
                    full_pixel_mask,
                )
            ),
            "shuffled_vs_target": (
                masked_frame_error_metrics(
                    shuffled_01,
                    target_01,
                    full_pixel_mask,
                )
            ),
        },
        "native_transport_region": {
            "correct_vs_target": (
                masked_frame_error_metrics(
                    correct_01,
                    target_01,
                    native_pixel_mask,
                )
            ),
            "shuffled_vs_target": (
                masked_frame_error_metrics(
                    shuffled_01,
                    target_01,
                    native_pixel_mask,
                )
            ),
        },
        "common_evaluation_region": {
            "correct_vs_target": (
                masked_frame_error_metrics(
                    correct_01,
                    target_01,
                    evaluation_pixel_mask,
                )
            ),
            "shuffled_vs_target": (
                masked_frame_error_metrics(
                    shuffled_01,
                    target_01,
                    evaluation_pixel_mask,
                )
            ),
        },
        "correct_vs_shuffled": {
            "full_frame": (
                masked_frame_error_metrics(
                    correct_01,
                    shuffled_01,
                    full_pixel_mask,
                )
            ),
            "common_evaluation_region": (
                masked_frame_error_metrics(
                    correct_01,
                    shuffled_01,
                    evaluation_pixel_mask,
                )
            ),
        },
    }

    sharpness = {
        "target_common_region": (
            sharpness_metrics(
                target_01,
                evaluation_pixel_mask,
            )
        ),
        "correct_common_region": (
            sharpness_metrics(
                correct_01,
                evaluation_pixel_mask,
            )
        ),
        "shuffled_common_region": (
            sharpness_metrics(
                shuffled_01,
                evaluation_pixel_mask,
            )
        ),
        "target_full_frame": (
            sharpness_metrics(
                target_01,
                full_pixel_mask,
            )
        ),
        "correct_full_frame": (
            sharpness_metrics(
                correct_01,
                full_pixel_mask,
            )
        ),
        "shuffled_full_frame": (
            sharpness_metrics(
                shuffled_01,
                full_pixel_mask,
            )
        ),
    }

    temporal = {
        "target": temporal_metrics(
            target_01
        ),
        "correct": temporal_metrics(
            correct_01
        ),
        "shuffled": temporal_metrics(
            shuffled_01
        ),
    }

    target_uint8_tensor = torch.from_numpy(
        target_uint8
    )

    checks = {
        "latent_indices_are_0_to_80_stride4": True,
        "target_decode_shape_is_1x81x3x480x832": (
            tuple(target_decoded.shape)
            == expected_decoded_shape
        ),
        "correct_decode_shape_matches_target": (
            correct_decoded.shape
            == target_decoded.shape
        ),
        "shuffled_decode_shape_matches_target": (
            shuffled_decoded.shape
            == target_decoded.shape
        ),
        "target_finite": bool(
            torch.isfinite(
                target_decoded
            ).all()
        ),
        "correct_finite": bool(
            torch.isfinite(
                correct_decoded
            ).all()
        ),
        "shuffled_finite": bool(
            torch.isfinite(
                shuffled_decoded
            ).all()
        ),
        "correct_not_collapsed": (
            float(correct_01.std()) > 0.01
        ),
        "shuffled_not_collapsed": (
            float(shuffled_01.std()) > 0.01
        ),
        "native_mask_has_support": bool(
            native_pixel_mask.any()
        ),
        "evaluation_mask_has_support": bool(
            evaluation_pixel_mask.any()
        ),
    }

    report = {
        "stage": (
            "precomputed_wan_transport_artifact_decode"
        ),
        "label": args.label,
        "artifact": {
            "path": str(
                artifact_path
            ),
            "sha256": sha256(
                artifact_path
            ),
            "transport_validity_mode": (
                artifact.get(
                    "transport_validity_mode"
                )
            ),
        },
        "evaluation_mask_artifact": {
            "path": str(
                evaluation_artifact_path
            ),
            "sha256": sha256(
                evaluation_artifact_path
            ),
        },
        "checkpoint": {
            "path": str(
                checkpoint_path
            ),
            "sha256": checkpoint_hash,
        },
        "layouts": {
            "latent": "[B,Tz,16,H,W]",
            "decoded": "[B,T,C,H,W]",
        },
        "shapes": {
            "target_latent": list(
                target_latent.shape
            ),
            "target_decoded": list(
                target_decoded.shape
            ),
            "native_latent_mask": list(
                native_mask.shape
            ),
            "evaluation_latent_mask": list(
                evaluation_mask.shape
            ),
        },
        "statistics": {
            "target_decoded": tensor_stats(
                target_01
            ),
            "correct_decoded": tensor_stats(
                correct_01
            ),
            "shuffled_decoded": tensor_stats(
                shuffled_01
            ),
        },
        "hashes": {
            "target_uint8_frames_sha256": (
                tensor_sha256(
                    target_uint8_tensor
                )
            ),
        },
        "metrics": metrics,
        "sharpness": sharpness,
        "temporal": temporal,
        "checks": checks,
        "all_checks_pass": all(
            checks.values()
        ),
        "runtime_seconds": {
            "model_load": (
                model_load_seconds
            ),
            "target_decode": (
                target_decode_seconds
            ),
            "correct_decode": (
                correct_decode_seconds
            ),
            "shuffled_decode": (
                shuffled_decode_seconds
            ),
            "total": (
                time.perf_counter()
                - overall_started
            ),
        },
        "gpu_memory_mib": {
            "whole_device_before": (
                gpu_before
            ),
            "whole_device_after": (
                gpu_used_memory_mib()
            ),
            "torch_peak_allocated": float(
                torch.cuda.max_memory_allocated()
                / (1024**2)
            ),
            "torch_peak_reserved": float(
                torch.cuda.max_memory_reserved()
                / (1024**2)
            ),
        },
        "artifacts": {
            "videos": video_paths,
            "comparison_video": str(
                comparison_video_path
            ),
            "selected_images": (
                selected_images
            ),
        },
        "interpretation_boundary": (
            "These are Wan VAE-only proxy results. "
            "They do not establish final RealWonder "
            "diffusion-generation quality."
        ),
    }

    report_path = (
        output_dir / "decode_report.json"
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
                "label": args.label,
                "checks": checks,
                "all_checks_pass": (
                    report[
                        "all_checks_pass"
                    ]
                ),
                "target_uint8_sha256": (
                    report[
                        "hashes"
                    ][
                        "target_uint8_frames_sha256"
                    ]
                ),
                "full_frame": {
                    "correct": (
                        metrics[
                            "full_frame"
                        ][
                            "correct_vs_target"
                        ][
                            "aggregate"
                        ]
                    ),
                    "shuffled": (
                        metrics[
                            "full_frame"
                        ][
                            "shuffled_vs_target"
                        ][
                            "aggregate"
                        ]
                    ),
                },
                "common_region": {
                    "correct": (
                        metrics[
                            "common_evaluation_region"
                        ][
                            "correct_vs_target"
                        ][
                            "aggregate"
                        ]
                    ),
                    "shuffled": (
                        metrics[
                            "common_evaluation_region"
                        ][
                            "shuffled_vs_target"
                        ][
                            "aggregate"
                        ]
                    ),
                },
                "sharpness_common": {
                    "target": (
                        sharpness[
                            "target_common_region"
                        ][
                            "aggregate"
                        ]
                    ),
                    "correct": (
                        sharpness[
                            "correct_common_region"
                        ][
                            "aggregate"
                        ]
                    ),
                    "shuffled": (
                        sharpness[
                            "shuffled_common_region"
                        ][
                            "aggregate"
                        ]
                    ),
                },
                "runtime_seconds": (
                    report[
                        "runtime_seconds"
                    ]
                ),
            },
            indent=2,
        )
    )

    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
