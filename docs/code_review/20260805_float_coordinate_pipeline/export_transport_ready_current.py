"""Convert a saved RealWonder trajectory into the transport-ready contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deform_transport.transport_ready import (  # noqa: E402
    build_transport_ready,
    save_transport_ready,
)


DEFAULT_SOURCE = (
    REPO_ROOT
    / "artifacts"
    / "exhaustive_validation"
    / "santa_action_suite_20260802"
    / "right_s1_21f"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts"
    / "transport_validation"
    / "santa_cloth_21f"
    / "transport_ready.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-name", default="santa_cloth")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    trajectory_path = source_dir / "point_trajectories.pt"
    flow_path = source_dir / "flows.npy"
    source_raster_path = source_dir / "flow_source_point_indices.npy"
    initial_rgb_path = source_dir / "frame_initial.png"
    coarse_rgb_paths = sorted(source_dir.glob("frame_[0-9][0-9][0-9][0-9].png"))
    required = [
        trajectory_path,
        flow_path,
        source_raster_path,
        initial_rgb_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required saved artifacts: {missing}")

    trajectory = torch.load(
        trajectory_path, map_location="cpu", weights_only=False
    )
    source_raster = np.load(source_raster_path)
    state = build_transport_ready(
        trajectory,
        torch.from_numpy(source_raster),
        case_name=args.case_name,
        source_trajectory_path=trajectory_path,
        initial_rgb_path=initial_rgb_path,
        coarse_rgb_paths=coarse_rgb_paths,
        flow_path=flow_path,
        source_raster_path=source_raster_path,
    )
    output_path = save_transport_ready(state, args.output.resolve())
    source_cells = state["source_points_2d_latent"][state["source_visible"]]
    occupied_source_cells = torch.unique(source_cells, dim=0).shape[0]
    summary = {
        "case": state["case_name"],
        "frames": int(state["frame_ids"].numel()),
        "points": int(state["point_id"].numel()),
        "source_visible_points": int(state["source_visible"].sum()),
        "source_visible_ratio": float(state["source_visible"].float().mean()),
        "occupied_source_latent_cells": int(occupied_source_cells),
        "min_future_valid_ratio": float(
            state["projection_valid"].float().mean(dim=1).min()
        ),
        "binding_shape": (
            list(state["point_particle_binding"].shape)
            if state["point_particle_binding"] is not None
            else None
        ),
        "output": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

