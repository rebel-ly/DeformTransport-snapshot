"""Inspect RealWonder's bundled reconstruction in the exact latent coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.trajectory import (  # noqa: E402
    map_image_uv_to_latent,
    project_points_realwonder,
)


def _tensor_pair(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def inspect_case(demo_data: Path, output_dir: Path, point_stride: int) -> dict:
    camera = torch.load(demo_data / "camera.pt", map_location="cpu")
    crop_start = 176
    config_path = demo_data / "config.yaml"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("crop_start:"):
                crop_start = int(line.split(":", 1)[1].strip())
                break

    object_reports = []
    projected_for_overlay = []
    for point_path in sorted((demo_data / "fg_pcs").glob("pc_*.pt")):
        point_data = torch.load(point_path, map_location="cpu")
        points = point_data["points"].to(torch.float32)
        uv, depth, render_valid = project_points_realwonder(
            points, camera["K"], camera["R"], camera["T"], image_size=512
        )
        latent_xy, video_uv, crop_valid = map_image_uv_to_latent(
            uv, crop_start=crop_start
        )
        valid = render_valid & crop_valid
        valid_cells = latent_xy[valid]
        linear_cells = valid_cells[:, 1] * 104 + valid_cells[:, 0]
        counts = torch.bincount(linear_cells, minlength=60 * 104)
        occupied = counts > 0
        point_count = int(points.shape[0])
        valid_count = int(valid.sum())
        occupied_count = int(occupied.sum())

        object_reports.append(
            {
                "file": str(point_path.relative_to(REPO_ROOT)),
                "point_count": point_count,
                "render_valid_count": int(render_valid.sum()),
                "render_valid_ratio": float(render_valid.float().mean()),
                "crop_valid_count": valid_count,
                "crop_valid_ratio": valid_count / point_count,
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
                "uv_min": _tensor_pair(uv.amin(dim=0)),
                "uv_max": _tensor_pair(uv.amax(dim=0)),
                "video_uv_min": _tensor_pair(video_uv.amin(dim=0)),
                "video_uv_max": _tensor_pair(video_uv.amax(dim=0)),
                "latent_grid": [60, 104],
                "occupied_latent_cells": occupied_count,
                "latent_occupancy_ratio": occupied_count / (60 * 104),
                "mean_points_per_occupied_cell": valid_count / max(occupied_count, 1),
                "max_points_in_one_cell": int(counts.max()),
            }
        )
        projected_for_overlay.append(video_uv[valid][::point_stride])

    output_dir.mkdir(parents=True, exist_ok=True)
    first_frame = Image.open(demo_data / "first_frame.png").convert("RGB")
    overlay = first_frame.copy()
    draw = ImageDraw.Draw(overlay)
    colors = ["#00ff88", "#ffcc00", "#00aaff", "#ff6688"]
    for object_id, uv in enumerate(projected_for_overlay):
        color = colors[object_id % len(colors)]
        for x, y in uv.tolist():
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    overlay_path = output_dir / f"{demo_data.name}_projection.png"
    overlay.save(overlay_path)

    report = {
        "case": demo_data.name,
        "data_path": str(demo_data),
        "uses_bundled_realwonder_data": True,
        "source_render_size": [512, 512],
        "video_size": [480, 832],
        "crop_start": crop_start,
        "objects": object_reports,
        "projection_overlay": str(overlay_path),
    }
    report_path = output_dir / f"{demo_data.name}_projection.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-data",
        type=Path,
        default=REPO_ROOT / "demo_web" / "demo_data" / "santa_cloth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "stage1",
    )
    parser.add_argument("--point-stride", type=int, default=80)
    args = parser.parse_args()
    report = inspect_case(args.demo_data.resolve(), args.output_dir.resolve(), args.point_stride)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
