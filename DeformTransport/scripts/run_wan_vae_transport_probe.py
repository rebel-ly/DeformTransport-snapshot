"""Run the gated Wan VAE-only baseline and hard-latent transport probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.hard_transport import hard_point_transport  # noqa: E402
from deform_transport.transport_metrics import (  # noqa: E402
    aggregate_frame_metrics,
    compare_correct_and_shuffled,
    masked_frame_metrics,
)
from deform_transport.transport_payloads import (  # noqa: E402
    load_realwonder_rgb_crop,
    point_support_mask,
)
from deform_transport.transport_ready import validate_transport_ready  # noqa: E402
from deform_transport.wan_vae_codec import (  # noqa: E402
    RealWonderWanVAECodec,
    causal_latent_frame_end_indices,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "artifacts"
    / "transport_validation"
    / "santa_cloth_21f"
    / "transport_ready.pt"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "wan_models"
    / "Wan2.1-Fun-V1.1-1.3B-InP"
    / "Wan2.1_VAE.pth"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "wan_vae"
EXPECTED_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_used_memory_mib() -> int | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices == "-1":
        return None

    gpu_id = visible_devices.split(",", maxsplit=1)[0].strip() or "0"

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


def _timed_cuda(function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    return output, time.perf_counter() - started


def _pixel_video_from_paths(paths) -> torch.Tensor:
    frames = [load_realwonder_rgb_crop(path) for path in paths]
    return torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).mul(2.0).sub(1.0)


def _pixels_01(decoded: torch.Tensor) -> torch.Tensor:
    return decoded[0].mul(0.5).add(0.5).clamp(0.0, 1.0)


def _uint8_frames(frames: torch.Tensor) -> np.ndarray:
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


def _write_video(path: Path, frames: np.ndarray, fps: int = 8) -> None:
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _source_comparison(source: torch.Tensor, reconstruction: torch.Tensor) -> Image.Image:
    source_image = Image.fromarray(_uint8_frames(source)[0])
    reconstruction_image = Image.fromarray(_uint8_frames(reconstruction)[0])
    canvas = Image.new("RGB", (1664, 512), "black")
    for panel_index, (label, image) in enumerate(
        (("Input source RGB", source_image), ("Wan VAE reconstruction", reconstruction_image))
    ):
        resized = image.resize((832, 480), resample=Image.Resampling.BILINEAR)
        left = panel_index * 832
        canvas.paste(resized, (left, 32))
        ImageDraw.Draw(canvas).text((left + 10, 9), label, fill="white")
    return canvas


def _labeled_four_panel(frames_by_label: list[tuple[str, np.ndarray]]) -> np.ndarray:
    outputs = []
    frame_count = len(frames_by_label[0][1])
    for frame_index in range(frame_count):
        panels = []
        for label, frames in frames_by_label:
            image = Image.fromarray(frames[frame_index]).resize(
                (416, 240), resample=Image.Resampling.BILINEAR
            )
            panel = Image.new("RGB", (416, 272), "black")
            panel.paste(image, (0, 32))
            ImageDraw.Draw(panel).text((8, 9), label, fill="white")
            panels.append(np.asarray(panel))
        outputs.append(np.concatenate(panels, axis=1))
    return np.stack(outputs)


def _expand_latent_mask_to_pixel_frames(mask: torch.Tensor, pixel_frames: int) -> torch.Tensor:
    expanded = [mask[:1]]
    for latent_index in range(1, mask.shape[0]):
        expanded.append(mask[latent_index : latent_index + 1].repeat(4, 1, 1, 1))
    expanded_mask = torch.cat(expanded, dim=0)
    return expanded_mask[:pixel_frames]


def _latent_frame_metrics(prediction, target, mask):
    metrics = []
    for frame_index in range(prediction.shape[0]):
        expanded = mask[frame_index].expand(prediction.shape[1], -1, -1)
        selected = (prediction[frame_index] - target[frame_index])[expanded]
        metrics.append(
            {
                "frame": frame_index,
                "support_cells": int(mask[frame_index].sum()),
                "masked_l1": float(selected.abs().mean()),
                "masked_mse": float(selected.square().mean()),
                "masked_psnr_db": None,
            }
        )
    return metrics


def _tensor_statistics(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "l2_norm": float(torch.linalg.vector_norm(tensor)),
        "finite": bool(torch.isfinite(tensor).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-ready", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--visual-inspection", choices=("pending", "pass", "fail"), default="pending"
    )
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != EXPECTED_SHA256:
        raise ValueError(f"Wan VAE SHA256 mismatch: {checkpoint_hash}")
    state = torch.load(
        args.transport_ready.resolve(), map_location="cpu", weights_only=False
    )
    validate_transport_ready(state)
    if not torch.cuda.is_available():
        raise RuntimeError("the VAE-only probe requires the verified CUDA runtime")

    torch.cuda.reset_peak_memory_stats()
    gpu_before = _gpu_used_memory_mib()
    overall_started = time.perf_counter()
    codec, load_seconds = _timed_cuda(
        lambda: RealWonderWanVAECodec(
            checkpoint, device="cuda", dtype=torch.bfloat16
        )
    )
    model_parameters = sum(parameter.numel() for parameter in codec.model.parameters())

    source_crop = load_realwonder_rgb_crop(state["paths"]["initial_rgb"])
    source_pixels = source_crop.mul(2.0).sub(1.0).unsqueeze(0).unsqueeze(2)
    source_latent, source_encode_seconds = _timed_cuda(
        lambda: codec.encode_pixels(source_pixels)
    )
    source_decoded, source_decode_seconds = _timed_cuda(
        lambda: codec.decode_latents(source_latent)
    )
    source_decoded = source_decoded.cpu()
    source_frames_01 = source_pixels[0].permute(1, 0, 2, 3).mul(0.5).add(0.5)
    source_reconstruction_01 = _pixels_01(source_decoded)
    source_reconstruction_mae = float(
        (source_frames_01 - source_reconstruction_01).abs().mean()
    )
    source_comparison_path = output_dir / "source_original_reconstruction.png"
    _source_comparison(source_frames_01, source_reconstruction_01).save(
        source_comparison_path
    )
    baseline = {
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
            "source": "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth",
        },
        "model_parameters": model_parameters,
        "compute_dtype": "torch.bfloat16",
        "input": {
            "shape": list(source_pixels.shape),
            "range": [float(source_pixels.min()), float(source_pixels.max())],
            "layout": "[B,C,T,H,W]",
        },
        "latent": _tensor_statistics(source_latent),
        "decoded": _tensor_statistics(source_decoded),
        "decoded_layout": "[B,T,C,H,W]",
        "source_reconstruction_mae_0_1": source_reconstruction_mae,
        "finite": bool(
            torch.isfinite(source_latent).all()
            and torch.isfinite(source_decoded).all()
        ),
        "runtime_seconds": {
            "model_load": load_seconds,
            "source_encode": source_encode_seconds,
            "source_decode": source_decode_seconds,
        },
        "source_comparison": str(source_comparison_path),
    }
    baseline_path = output_dir / "baseline_report.json"
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    if args.baseline_only:
        baseline["gpu_memory_mib"] = {
            "whole_device_before": gpu_before,
            "whole_device_after": _gpu_used_memory_mib(),
            "torch_peak_allocated": float(
                torch.cuda.max_memory_allocated() / (1024**2)
            ),
            "torch_peak_reserved": float(
                torch.cuda.max_memory_reserved() / (1024**2)
            ),
        }
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(json.dumps(baseline, indent=2))
        return

    future_pixels = _pixel_video_from_paths(state["paths"]["coarse_rgb_frames"])
    target_latent, future_encode_seconds = _timed_cuda(
        lambda: codec.encode_pixels(future_pixels)
    )
    latent_frame_indices = causal_latent_frame_end_indices(future_pixels.shape[2])
    if target_latent.shape[1] != latent_frame_indices.numel():
        raise RuntimeError("causal frame mapping does not match encoded temporal slots")

    def run_transport(mode):
        return hard_point_transport(
            source_latent[0, 0],
            state["source_points_2d_latent"].cuda(),
            state["points_2d_latent"][latent_frame_indices].cuda(),
            state["source_visible"].cuda(),
            state["source_valid"].cuda(),
            state["projection_valid"][latent_frame_indices].cuda(),
            state["point_id"].cuda(),
            object_id=state["object_id"].cuda(),
            mode=mode,
            seed=args.seed,
        )

    correct_result, correct_transport_seconds = _timed_cuda(
        lambda: run_transport("correct")
    )
    shuffled_result, shuffled_transport_seconds = _timed_cuda(
        lambda: run_transport("shuffled")
    )
    raw_correct_latent = correct_result["transported_grid"].unsqueeze(0)
    raw_shuffled_latent = shuffled_result["transported_grid"].unsqueeze(0)
    latent_mask = correct_result["transport_mask"].unsqueeze(0)
    fused_correct_latent = torch.where(
        latent_mask, raw_correct_latent, target_latent
    )
    fused_shuffled_latent = torch.where(
        latent_mask, raw_shuffled_latent, target_latent
    )

    def decode_cpu(latent):
        decoded, seconds = _timed_cuda(lambda: codec.decode_latents(latent))
        decoded_cpu = decoded.cpu()
        del decoded
        codec.clear_cache()
        torch.cuda.empty_cache()
        return decoded_cpu, seconds

    original_decoded, original_decode_seconds = decode_cpu(target_latent)
    raw_correct_decoded, raw_correct_decode_seconds = decode_cpu(raw_correct_latent)
    raw_shuffled_decoded, raw_shuffled_decode_seconds = decode_cpu(
        raw_shuffled_latent
    )
    fused_correct_decoded, fused_correct_decode_seconds = decode_cpu(
        fused_correct_latent
    )
    fused_shuffled_decoded, fused_shuffled_decode_seconds = decode_cpu(
        fused_shuffled_latent
    )

    target_pixels_01 = future_pixels[0].permute(1, 0, 2, 3).mul(0.5).add(0.5)
    original_01 = _pixels_01(original_decoded)
    raw_correct_01 = _pixels_01(raw_correct_decoded)
    raw_shuffled_01 = _pixels_01(raw_shuffled_decoded)
    fused_correct_01 = _pixels_01(fused_correct_decoded)
    fused_shuffled_01 = _pixels_01(fused_shuffled_decoded)

    target_support = point_support_mask(
        state["points_2d_latent"],
        state["projection_valid"],
        height=state["latent_height"],
        width=state["latent_width"],
    )
    pixel_transport_mask = _expand_latent_mask_to_pixel_frames(
        correct_result["transport_mask"].cpu(), target_pixels_01.shape[0]
    )
    pixel_shared_mask = pixel_transport_mask & target_support
    pixel_shared_mask = F.interpolate(
        pixel_shared_mask.float(), size=(480, 832), mode="nearest"
    ).bool()
    fused_correct_metrics = masked_frame_metrics(
        fused_correct_01, target_pixels_01, pixel_shared_mask
    )
    fused_shuffled_metrics = masked_frame_metrics(
        fused_shuffled_01, target_pixels_01, pixel_shared_mask
    )
    decoded_comparison = compare_correct_and_shuffled(
        fused_correct_metrics, fused_shuffled_metrics
    )
    raw_correct_metrics = masked_frame_metrics(
        raw_correct_01, target_pixels_01, pixel_shared_mask
    )
    raw_shuffled_metrics = masked_frame_metrics(
        raw_shuffled_01, target_pixels_01, pixel_shared_mask
    )
    raw_decoded_comparison = compare_correct_and_shuffled(
        raw_correct_metrics, raw_shuffled_metrics
    )
    full_mask = torch.ones(
        (target_pixels_01.shape[0], 1, 480, 832), dtype=torch.bool
    )
    original_reconstruction_metrics = masked_frame_metrics(
        original_01, target_pixels_01, full_mask
    )

    latent_target_support = target_support[latent_frame_indices].cuda()
    latent_shared_mask = correct_result["transport_mask"] & latent_target_support
    correct_latent_metrics = _latent_frame_metrics(
        raw_correct_latent[0], target_latent[0], latent_shared_mask
    )
    shuffled_latent_metrics = _latent_frame_metrics(
        raw_shuffled_latent[0], target_latent[0], latent_shared_mask
    )
    latent_comparison = compare_correct_and_shuffled(
        correct_latent_metrics, shuffled_latent_metrics
    )

    fairness = {
        "transport_masks_equal": bool(
            torch.equal(
                correct_result["transport_mask"],
                shuffled_result["transport_mask"],
            )
        ),
        "contribution_counts_equal": bool(
            torch.equal(
                correct_result["contribution_count"],
                shuffled_result["contribution_count"],
            )
        ),
        "valid_point_masks_equal": bool(
            torch.equal(
                correct_result["valid_point_mask"],
                shuffled_result["valid_point_mask"],
            )
        ),
        "permutations_differ": not bool(
            torch.equal(
                correct_result["permutation"], shuffled_result["permutation"]
            )
        ),
    }
    latent_outputs_path = output_dir / "vae_latent_outputs.pt"
    torch.save(
        {
            "format_version": 1,
            "latent_frame_indices": latent_frame_indices,
            "source_latent": source_latent.detach().cpu(),
            "target_latent": target_latent.detach().cpu(),
            "correct_transported_latent": raw_correct_latent.detach().cpu(),
            "shuffled_transported_latent": raw_shuffled_latent.detach().cpu(),
            "correct_fused_latent": fused_correct_latent.detach().cpu(),
            "shuffled_fused_latent": fused_shuffled_latent.detach().cpu(),
            "transport_mask": correct_result["transport_mask"].detach().cpu(),
            "contribution_count": correct_result["contribution_count"].detach().cpu(),
            "shuffled_permutation": shuffled_result["permutation"].detach().cpu(),
        },
        latent_outputs_path,
    )

    videos = {
        "target_input": _uint8_frames(target_pixels_01),
        "original_reconstruction": _uint8_frames(original_01),
        "raw_correct": _uint8_frames(raw_correct_01),
        "raw_shuffled": _uint8_frames(raw_shuffled_01),
        "fused_correct": _uint8_frames(fused_correct_01),
        "fused_shuffled": _uint8_frames(fused_shuffled_01),
    }
    video_paths = {}
    for name, frames in videos.items():
        path = output_dir / f"{name}.mp4"
        _write_video(path, frames)
        video_paths[name] = str(path)
    comparison_frames = _labeled_four_panel(
        [
            ("Original VAE recon", videos["original_reconstruction"]),
            ("Correct masked replace", videos["fused_correct"]),
            ("Shuffled masked replace", videos["fused_shuffled"]),
            ("Coarse RGB input", videos["target_input"]),
        ]
    )
    comparison_video_path = output_dir / "original_correct_shuffled_target.mp4"
    _write_video(comparison_video_path, comparison_frames)
    for name, frame_index in (
        ("first", 0),
        ("mid", len(comparison_frames) // 2),
        ("final", len(comparison_frames) - 1),
    ):
        Image.fromarray(comparison_frames[frame_index]).save(
            output_dir / f"comparison_{name}.png"
        )

    latent_different = not torch.equal(raw_correct_latent, raw_shuffled_latent)
    pixel_frame_count = int(future_pixels.shape[2])
    expected_latent_slots = int(latent_frame_indices.numel())
    expected_future_latent_shape = [
        1,
        expected_latent_slots,
        *list(source_latent.shape[2:]),
    ]

    engineering_checks = {
        "source_encode_decode_finite": baseline["finite"],
        "source_latent_shape_is_1x1x16x60x104": list(source_latent.shape)
        == [1, 1, 16, 60, 104],
        "future_latent_shape_matches_temporal_mapping": list(target_latent.shape)
        == expected_future_latent_shape,
        "future_decode_returns_all_pixel_frames": int(original_decoded.shape[1])
        == pixel_frame_count,
        "correct_and_shuffled_latents_differ": latent_different,
        "fused_correct_is_finite": bool(torch.isfinite(fused_correct_decoded).all()),
        "fused_shuffled_is_finite": bool(torch.isfinite(fused_shuffled_decoded).all()),
        "fused_correct_is_not_collapsed": float(fused_correct_01.std()) > 0.01,
        "fused_shuffled_is_not_collapsed": float(fused_shuffled_01.std()) > 0.01,
        **fairness,
    }
    positive_signal_checks = {
        "correct_latent_l1_below_shuffled": latent_comparison[
            "overall_correct_better"
        ],
        "correct_decoded_l1_below_shuffled": decoded_comparison[
            "overall_correct_better"
        ],
        "correct_decoded_not_worse_in_majority": decoded_comparison[
            "correct_not_worse_fraction"
        ]
        >= 0.5,
    }
    automated_passed = all(engineering_checks.values()) and all(
        positive_signal_checks.values()
    )
    visual_passed = args.visual_inspection == "pass"
    gpu_after = _gpu_used_memory_mib()
    report = {
        "stage": "wan_vae_only_latent_transport",
        "case": state["case_name"],
        "is_robot_action": False,
        "baseline": baseline,
        "actual_temporal_mapping": {
            "pixel_frames": int(future_pixels.shape[2]),
            "latent_slots": int(target_latent.shape[1]),
            "causal_chunk_end_frame_indices": latent_frame_indices.tolist(),
            "note": "frame zero is encoded alone; later slots encode four-frame causal chunks",
        },
        "latent_statistics": {
            "target_original": _tensor_statistics(target_latent),
            "correct_transported": _tensor_statistics(raw_correct_latent),
            "shuffled_transported": _tensor_statistics(raw_shuffled_latent),
            "correct_masked_replace": _tensor_statistics(fused_correct_latent),
            "shuffled_masked_replace": _tensor_statistics(fused_shuffled_latent),
        },
        "composition": {
            "method": "masked_replace",
            "inside_transport_mask": "transported source latent",
            "outside_transport_mask": "original encoded coarse-RGB latent",
            "blend_alpha_inside": 1.0,
        },
        "latent_proxy_metrics": {
            "correct": correct_latent_metrics,
            "shuffled": shuffled_latent_metrics,
            "comparison": latent_comparison,
        },
        "decoded_proxy_metrics": {
            "correct_masked_replace": {
                "frames": fused_correct_metrics,
                "aggregate": aggregate_frame_metrics(fused_correct_metrics),
            },
            "shuffled_masked_replace": {
                "frames": fused_shuffled_metrics,
                "aggregate": aggregate_frame_metrics(fused_shuffled_metrics),
            },
            "comparison": decoded_comparison,
            "raw_transport_comparison": raw_decoded_comparison,
            "original_vae_reconstruction": aggregate_frame_metrics(
                original_reconstruction_metrics
            ),
        },
        "target_note": (
            "All target comparisons use the saved RealWonder coarse RGB input or "
            "its VAE latent. This is a geometry-aligned proxy, not real future-video ground truth."
        ),
        "fairness": fairness,
        "engineering_checks": engineering_checks,
        "positive_signal_checks": positive_signal_checks,
        "automated_acceptance_passed": automated_passed,
        "visual_inspection": {
            "status": args.visual_inspection,
            "note": args.visual_note,
        },
        "task3a_passed": automated_passed and visual_passed,
        "runtime_seconds": {
            "model_load": load_seconds,
            "source_encode": source_encode_seconds,
            "source_decode": source_decode_seconds,
            "future_video_encode": future_encode_seconds,
            "correct_transport": correct_transport_seconds,
            "shuffled_transport": shuffled_transport_seconds,
            "original_decode": original_decode_seconds,
            "raw_correct_decode": raw_correct_decode_seconds,
            "raw_shuffled_decode": raw_shuffled_decode_seconds,
            "fused_correct_decode": fused_correct_decode_seconds,
            "fused_shuffled_decode": fused_shuffled_decode_seconds,
            "total": time.perf_counter() - overall_started,
        },
        "gpu_memory_mib": {
            "whole_device_before": gpu_before,
            "whole_device_after": gpu_after,
            "torch_peak_allocated": float(
                torch.cuda.max_memory_allocated() / (1024**2)
            ),
            "torch_peak_reserved": float(
                torch.cuda.max_memory_reserved() / (1024**2)
            ),
        },
        "artifacts": {
            "baseline_report": str(baseline_path),
            "source_original_reconstruction": str(source_comparison_path),
            "latent_outputs": str(latent_outputs_path),
            "videos": video_paths,
            "comparison_video": str(comparison_video_path),
            "comparison_first": str(output_dir / "comparison_first.png"),
            "comparison_mid": str(output_dir / "comparison_mid.png"),
            "comparison_final": str(output_dir / "comparison_final.png"),
        },
        "not_downloaded": [
            "Wan diffusion transformer",
            "RealWonder distilled video checkpoint",
            "T5",
            "CLIP",
            "SAM",
            "FLUX",
            "QWM",
        ],
        "not_validated": ["future video generation", "robot action conditioning"],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "shapes": {
                    "source_latent": list(source_latent.shape),
                    "future_latent": list(target_latent.shape),
                    "original_decoded": list(original_decoded.shape),
                },
                "latent_comparison": latent_comparison,
                "decoded_comparison": decoded_comparison,
                "engineering_checks": engineering_checks,
                "positive_signal_checks": positive_signal_checks,
                "automated_acceptance_passed": automated_passed,
                "visual_inspection": args.visual_inspection,
                "task3a_passed": report["task3a_passed"],
                "runtime_seconds": report["runtime_seconds"],
                "gpu_memory_mib": report["gpu_memory_mib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
