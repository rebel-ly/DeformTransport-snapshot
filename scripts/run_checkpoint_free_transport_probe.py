"""Run Correct/Shuffled hard transport on the saved 21-frame Santa cloth run."""

from __future__ import annotations

import argparse
import json
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
    transport_coverage_metrics,
)
from deform_transport.transport_payloads import (  # noqa: E402
    coordinate_identity_payload,
    load_realwonder_rgb_crop,
    load_realwonder_rgb_grid,
    load_realwonder_rgb_grid_sequence,
    point_support_mask,
)
from deform_transport.transport_ready import validate_transport_ready  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "artifacts"
    / "transport_validation"
    / "santa_cloth_21f"
    / "transport_ready.pt"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "checkpoint_free"


def _gpu_used_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


def _run_transport(payload, state, *, mode, seed, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = hard_point_transport(
        payload.to(device),
        state["source_points_2d_latent"].to(device),
        state["points_2d_latent"].to(device),
        state["source_visible"].to(device),
        state["source_valid"].to(device),
        state["projection_valid"].to(device),
        state["point_id"].to(device),
        object_id=state["object_id"].to(device),
        mode=mode,
        seed=seed,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return result, elapsed


def _cpu_result(result):
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in result.items()
    }


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    frames = torch.as_tensor(frames, dtype=torch.float32).clamp(0.0, 1.0)
    return (
        frames.permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8).numpy()
    )


def _upsample_transport(result: dict, *, height=480, width=832) -> torch.Tensor:
    grids = result["transported_grid"].detach().cpu()
    masks = result["transport_mask"].detach().cpu().float()
    grids = F.interpolate(grids, size=(height, width), mode="bilinear", align_corners=False)
    masks = F.interpolate(masks, size=(height, width), mode="nearest")
    return grids * masks


def _upsample_mask(mask: torch.Tensor, *, height=480, width=832) -> torch.Tensor:
    mask = F.interpolate(
        mask.detach().cpu().float(), size=(height, width), mode="nearest"
    )
    return mask.expand(-1, 3, -1, -1)


def _count_heatmap(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.detach().cpu().float()
    maximum = float(counts.max())
    normalized = torch.log1p(counts) / np.log1p(max(maximum, 1.0))
    normalized = F.interpolate(normalized, size=(480, 832), mode="nearest")
    red = normalized
    green = normalized.sqrt()
    blue = (1.0 - normalized) * (normalized > 0)
    return torch.cat([red, green, blue], dim=1)


def _write_video(path: Path, frames: np.ndarray, *, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _save_frame_sequence(directory: Path, frames: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(frames):
        Image.fromarray(frame).save(directory / f"frame_{frame_index:04d}.png")


def _labeled_comparison(
    correct: np.ndarray, shuffled: np.ndarray, target: np.ndarray
) -> np.ndarray:
    outputs = []
    labels = ("Correct identity", "Shuffled identity", "Coarse RGB proxy")
    for frame_index in range(len(correct)):
        panels = []
        for label, frame in zip(
            labels, (correct[frame_index], shuffled[frame_index], target[frame_index])
        ):
            panel = Image.new("RGB", (frame.shape[1], frame.shape[0] + 32), "black")
            panel.paste(Image.fromarray(frame), (0, 32))
            ImageDraw.Draw(panel).text((10, 9), label, fill="white")
            panels.append(np.asarray(panel))
        outputs.append(np.concatenate(panels, axis=1))
    return np.stack(outputs)


def _source_visible_overlay(state: dict, output_path: Path) -> None:
    image = np.asarray(Image.open(state["paths"]["initial_rgb"]).convert("RGB")).copy()
    uv = state["source_points_2d_render"][state["source_visible"]].round().long()
    x = uv[:, 0].clamp(0, image.shape[1] - 1).numpy()
    y = uv[:, 1].clamp(0, image.shape[0] - 1).numpy()
    for offset_y, offset_x in ((0, 0), (0, 1), (1, 0)):
        yy = np.clip(y + offset_y, 0, image.shape[0] - 1)
        xx = np.clip(x + offset_x, 0, image.shape[1] - 1)
        image[yy, xx] = np.array([0, 255, 255], dtype=np.uint8)
    Image.fromarray(image).save(output_path)


def _source_cell_occupancy(state: dict, output_path: Path) -> None:
    cells = state["source_points_2d_latent"][state["source_visible"]]
    occupancy = torch.zeros(
        state["latent_height"], state["latent_width"], dtype=torch.bool
    )
    occupancy[cells[:, 1], cells[:, 0]] = True
    image = torch.zeros(3, state["latent_height"], state["latent_width"])
    image[0] = occupancy.float() * 0.1
    image[1] = occupancy.float()
    image[2] = occupancy.float() * 0.4
    image = F.interpolate(image.unsqueeze(0), size=(480, 832), mode="nearest")[0]
    Image.fromarray(_to_uint8(image.unsqueeze(0))[0]).save(output_path)


def _save_local_zoom(
    comparison_frames: np.ndarray, mask: torch.Tensor, output_path: Path
) -> None:
    final_mask = F.interpolate(
        mask[-1:].detach().cpu().float(), size=(480, 832), mode="nearest"
    )[0, 0]
    coordinates = torch.nonzero(final_mask, as_tuple=False)
    if not coordinates.numel():
        return
    y0, x0 = coordinates.min(dim=0).values.tolist()
    y1, x1 = coordinates.max(dim=0).values.tolist()
    padding = 24
    y0 = max(32, y0 + 32 - padding)
    y1 = min(comparison_frames.shape[1], y1 + 32 + padding + 1)
    panel_width = 832
    crops = []
    for panel_index in range(3):
        left = panel_index * panel_width
        crops.append(
            comparison_frames[-1, y0:y1, left + max(0, x0 - padding) : left + min(panel_width, x1 + padding + 1)]
        )
    minimum_height = min(crop.shape[0] for crop in crops)
    minimum_width = min(crop.shape[1] for crop in crops)
    zoom = np.concatenate(
        [crop[:minimum_height, :minimum_width] for crop in crops], axis=1
    )
    Image.fromarray(zoom).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-ready", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--visual-inspection",
        choices=("pending", "pass", "fail"),
        default="pending",
    )
    parser.add_argument("--visual-note", default="")
    args = parser.parse_args()

    overall_started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(
        args.transport_ready.resolve(), map_location="cpu", weights_only=False
    )
    validate_transport_ready(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gpu_before = _gpu_used_memory_mib()

    coordinate_payload = coordinate_identity_payload(
        state["latent_height"], state["latent_width"]
    )
    rgb_payload = load_realwonder_rgb_grid(
        state["paths"]["initial_rgb"],
        height=state["latent_height"],
        width=state["latent_width"],
    )
    target_rgb = load_realwonder_rgb_grid_sequence(
        state["paths"]["coarse_rgb_frames"],
        height=state["latent_height"],
        width=state["latent_width"],
    )
    target_support = point_support_mask(
        state["points_2d_latent"],
        state["projection_valid"],
        height=state["latent_height"],
        width=state["latent_width"],
    ).to(device)

    coordinate_correct, time_coordinate_correct = _run_transport(
        coordinate_payload, state, mode="correct", seed=args.seed, device=device
    )
    coordinate_shuffled, time_coordinate_shuffled = _run_transport(
        coordinate_payload, state, mode="shuffled", seed=args.seed, device=device
    )
    rgb_correct, time_rgb_correct = _run_transport(
        rgb_payload, state, mode="correct", seed=args.seed, device=device
    )
    rgb_shuffled, time_rgb_shuffled = _run_transport(
        rgb_payload, state, mode="shuffled", seed=args.seed, device=device
    )

    shared_support = (
        rgb_correct["transport_mask"]
        & rgb_shuffled["transport_mask"]
        & target_support
    )
    correct_frame_metrics = masked_frame_metrics(
        rgb_correct["transported_grid"], target_rgb.to(device), shared_support
    )
    shuffled_frame_metrics = masked_frame_metrics(
        rgb_shuffled["transported_grid"], target_rgb.to(device), shared_support
    )
    comparison = compare_correct_and_shuffled(
        correct_frame_metrics, shuffled_frame_metrics
    )
    correct_coverage = transport_coverage_metrics(rgb_correct)
    shuffled_coverage = transport_coverage_metrics(rgb_shuffled)

    eligible = rgb_correct["source_point_mask"]
    correct_features = rgb_correct["point_features"][eligible]
    shuffled_features = rgb_shuffled["point_features"][eligible]
    same_feature_multiset = torch.equal(
        torch.sort(correct_features, dim=0).values,
        torch.sort(shuffled_features, dim=0).values,
    )
    fairness = {
        "transport_masks_equal": bool(
            torch.equal(rgb_correct["transport_mask"], rgb_shuffled["transport_mask"])
        ),
        "contribution_counts_equal": bool(
            torch.equal(
                rgb_correct["contribution_count"],
                rgb_shuffled["contribution_count"],
            )
        ),
        "valid_point_masks_equal": bool(
            torch.equal(
                rgb_correct["valid_point_mask"], rgb_shuffled["valid_point_mask"]
            )
        ),
        "source_point_masks_equal": bool(
            torch.equal(
                rgb_correct["source_point_mask"], rgb_shuffled["source_point_mask"]
            )
        ),
        "source_feature_multisets_equal": bool(same_feature_multiset),
        "coordinate_and_rgb_masks_equal": bool(
            torch.equal(
                coordinate_correct["transport_mask"], rgb_correct["transport_mask"]
            )
        ),
        "correct_and_shuffled_permutations_differ": not bool(
            torch.equal(rgb_correct["permutation"], rgb_shuffled["permutation"])
        ),
    }
    coordinate_difference = (
        coordinate_correct["transported_grid"]
        - coordinate_shuffled["transported_grid"]
    ).abs()
    coordinate_difference = float(
        coordinate_difference[
            shared_support.expand(-1, coordinate_difference.shape[1], -1, -1)
        ].mean()
    )

    saved_results = {
        "format_version": 1,
        "case_name": state["case_name"],
        "seed": args.seed,
        "coordinate": {
            "correct": _cpu_result(coordinate_correct),
            "shuffled": _cpu_result(coordinate_shuffled),
        },
        "rgb": {
            "correct": _cpu_result(rgb_correct),
            "shuffled": _cpu_result(rgb_shuffled),
            "target_grid": target_rgb,
            "target_support": target_support.cpu(),
            "shared_support": shared_support.cpu(),
        },
    }
    results_path = output_dir / "transport_outputs.pt"
    torch.save(saved_results, results_path)

    correct_visual = _to_uint8(_upsample_transport(rgb_correct))
    shuffled_visual = _to_uint8(_upsample_transport(rgb_shuffled))
    coordinate_correct_visual = _to_uint8(_upsample_transport(coordinate_correct))
    coordinate_shuffled_visual = _to_uint8(_upsample_transport(coordinate_shuffled))
    target_crops = torch.stack(
        [load_realwonder_rgb_crop(path) for path in state["paths"]["coarse_rgb_frames"]]
    )
    target_visual = _to_uint8(target_crops)
    shared_mask_visual = _to_uint8(_upsample_mask(shared_support))
    target_shared_visual = (
        target_visual.astype(np.float32)
        * (shared_mask_visual.astype(np.float32) / 255.0)
    ).round().astype(np.uint8)
    mask_visual = _to_uint8(_upsample_mask(rgb_correct["transport_mask"]))
    count_visual = _to_uint8(_count_heatmap(rgb_correct["contribution_count"]))
    comparison_visual = _labeled_comparison(
        correct_visual, shuffled_visual, target_shared_visual
    )

    video_paths = {
        "correct_rgb": output_dir / "correct_rgb_transport.mp4",
        "shuffled_rgb": output_dir / "shuffled_rgb_transport.mp4",
        "coarse_rgb_target": output_dir / "coarse_rgb_target.mp4",
        "comparison": output_dir / "correct_shuffled_target.mp4",
        "coordinate_correct": output_dir / "coordinate_correct.mp4",
        "coordinate_shuffled": output_dir / "coordinate_shuffled.mp4",
        "transport_mask": output_dir / "transport_mask.mp4",
        "contribution_count": output_dir / "contribution_count.mp4",
    }
    _write_video(video_paths["correct_rgb"], correct_visual)
    _write_video(video_paths["shuffled_rgb"], shuffled_visual)
    _write_video(video_paths["coarse_rgb_target"], target_visual)
    _write_video(video_paths["comparison"], comparison_visual)
    _write_video(video_paths["coordinate_correct"], coordinate_correct_visual)
    _write_video(video_paths["coordinate_shuffled"], coordinate_shuffled_visual)
    _write_video(video_paths["transport_mask"], mask_visual)
    _write_video(video_paths["contribution_count"], count_visual)
    _save_frame_sequence(output_dir / "transport_mask_frames", mask_visual)
    _save_frame_sequence(output_dir / "contribution_count_frames", count_visual)
    _source_visible_overlay(state, output_dir / "source_visible_overlay.png")
    _source_cell_occupancy(state, output_dir / "source_latent_cells.png")
    _save_local_zoom(
        comparison_visual,
        shared_support,
        output_dir / "local_zoom_final.png",
    )
    Image.fromarray(comparison_visual[0]).save(output_dir / "comparison_first.png")
    Image.fromarray(comparison_visual[len(comparison_visual) // 2]).save(
        output_dir / "comparison_mid.png"
    )
    Image.fromarray(comparison_visual[-1]).save(output_dir / "comparison_final.png")

    engineering_checks = {
        **fairness,
        "correct_has_no_nan_or_inf": not correct_coverage["contains_nan_or_inf"],
        "shuffled_has_no_nan_or_inf": not shuffled_coverage["contains_nan_or_inf"],
        "mask_matches_positive_count": bool(
            torch.equal(
                rgb_correct["transport_mask"],
                rgb_correct["contribution_count"] > 0,
            )
        ),
        "counts_are_nonnegative_integers": (
            rgb_correct["contribution_count"].dtype == torch.long
            and bool((rgb_correct["contribution_count"] >= 0).all())
        ),
    }
    positive_signal_checks = {
        "overall_correct_l1_below_shuffled": comparison["overall_correct_better"],
        "correct_not_worse_in_majority_of_frames": (
            comparison["correct_not_worse_fraction"] >= 0.5
        ),
        "coordinate_payload_changes_under_shuffle": coordinate_difference > 1e-4,
    }
    automated_passed = all(engineering_checks.values()) and all(
        positive_signal_checks.values()
    )
    visual_passed = args.visual_inspection == "pass"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_after = _gpu_used_memory_mib()
    report = {
        "stage": "checkpoint_free_hard_point_transport",
        "case": state["case_name"],
        "driver": "wind",
        "is_robot_action": False,
        "device": str(device),
        "seed": args.seed,
        "frames": int(state["frame_ids"].numel()),
        "points": int(state["point_id"].numel()),
        "source_visible_points": int(state["source_visible"].sum()),
        "source_visible_ratio": float(state["source_visible"].float().mean()),
        "payloads": {
            "coordinate_identity": {
                "shape": list(coordinate_payload.shape),
                "correct_vs_shuffled_shared_mask_mae": coordinate_difference,
            },
            "initial_rgb_proxy": {
                "shape": list(rgb_payload.shape),
                "range": [float(rgb_payload.min()), float(rgb_payload.max())],
                "preprocessing": "512 RGB -> PIL bilinear 832 -> vertical crop 176:656 -> area pool 60x104",
            },
        },
        "target_note": (
            "The future RealWonder coarse RGB render is a geometry-aligned visual "
            "proxy, not ground-truth future video. Errors are computed only on the "
            "intersection of transport coverage and target point support."
        ),
        "correct": {
            "coverage": correct_coverage,
            "frame_metrics": correct_frame_metrics,
            "aggregate_metrics": aggregate_frame_metrics(correct_frame_metrics),
        },
        "shuffled": {
            "coverage": shuffled_coverage,
            "frame_metrics": shuffled_frame_metrics,
            "aggregate_metrics": aggregate_frame_metrics(shuffled_frame_metrics),
        },
        "comparison": comparison,
        "fairness": fairness,
        "engineering_checks": engineering_checks,
        "positive_signal_checks": positive_signal_checks,
        "automated_acceptance_passed": automated_passed,
        "visual_inspection": {
            "status": args.visual_inspection,
            "note": args.visual_note,
        },
        "task2_passed": automated_passed and visual_passed,
        "runtime_seconds": {
            "coordinate_correct": time_coordinate_correct,
            "coordinate_shuffled": time_coordinate_shuffled,
            "rgb_correct": time_rgb_correct,
            "rgb_shuffled": time_rgb_shuffled,
            "mean_transport_per_frame": (
                time_coordinate_correct
                + time_coordinate_shuffled
                + time_rgb_correct
                + time_rgb_shuffled
            )
            / (4 * state["frame_ids"].numel()),
            "total_including_metrics_and_visualization": time.perf_counter()
            - overall_started,
        },
        "gpu_memory_mib": {
            "whole_device_before": gpu_before,
            "whole_device_after": gpu_after,
            "torch_peak_allocated": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
            "torch_peak_reserved": (
                float(torch.cuda.max_memory_reserved(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
        },
        "artifacts": {
            "transport_ready": str(args.transport_ready.resolve()),
            "transport_outputs": str(results_path),
            "source_visible_overlay": str(output_dir / "source_visible_overlay.png"),
            "source_latent_cells": str(output_dir / "source_latent_cells.png"),
            "comparison_first": str(output_dir / "comparison_first.png"),
            "comparison_mid": str(output_dir / "comparison_mid.png"),
            "comparison_final": str(output_dir / "comparison_final.png"),
            "local_zoom_final": str(output_dir / "local_zoom_final.png"),
            "videos": {name: str(path) for name, path in video_paths.items()},
        },
        "commands": {
            "export_transport_ready": "python scripts/export_transport_ready.py",
            "run_probe": "python scripts/run_checkpoint_free_transport_probe.py --seed 0",
            "tests": "python -m unittest discover -s tests -v",
        },
        "not_validated": [
            "Wan VAE latent transport",
            "future video generation",
            "robot action conditioning",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "automated_acceptance_passed": automated_passed,
                "visual_inspection": args.visual_inspection,
                "task2_passed": report["task2_passed"],
                "comparison": comparison,
                "fairness": fairness,
                "runtime_seconds": report["runtime_seconds"],
                "gpu_memory_mib": report["gpu_memory_mib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
