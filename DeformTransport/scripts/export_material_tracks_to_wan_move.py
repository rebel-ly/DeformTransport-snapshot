#!/usr/bin/env python3
"""Export architecture-defined persistent material tracks to Wan-Move."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch


OUT_H = 480
OUT_W = 832
VAE_STRIDE = 8
SOURCE_SIZE = 512
RESIZED_SIZE = 832
CROP_TOP = 176


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fill_nonfinite_tracks(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    result = tracks.copy()
    for point_index in range(result.shape[1]):
        good = visibility[:, point_index] & np.isfinite(result[:, point_index]).all(axis=1)
        valid_frames = np.flatnonzero(good)
        if len(valid_frames) == 0:
            raise RuntimeError(f"selected point {point_index} has no valid frame")
        first = int(valid_frames[0])
        result[:first, point_index] = result[first, point_index]
        last = first
        for frame in range(first, result.shape[0]):
            if good[frame]:
                last = frame
            else:
                result[frame, point_index] = result[last, point_index]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--transport-ready", type=Path, required=True)
    parser.add_argument("--visibility-contract", type=Path, required=True)
    parser.add_argument("--trajectory-source", type=Path, required=True)
    parser.add_argument("--aligned-input-dir", type=Path, required=True)
    parser.add_argument("--realwonder-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    transport_path = args.transport_ready.resolve()
    visibility_path = args.visibility_contract.resolve()
    trajectory_path = args.trajectory_source.resolve()
    aligned_dir = args.aligned_input_dir.resolve()
    baseline_path = args.realwonder_baseline.resolve()
    image_path = aligned_dir / "resized_input_image.png"
    prompt_path = aligned_dir / "prompt.txt"
    frames_dir = aligned_dir / "frames"
    required = [transport_path, visibility_path, trajectory_path, image_path, prompt_path, baseline_path]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    transport = torch.load(transport_path, map_location="cpu", weights_only=False)
    visibility_contract = torch.load(visibility_path, map_location="cpu", weights_only=False)
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    assert args.case == "tree"
    assert transport["case_name"] == "tree_official_precomputed_geometry_official_dynamics_aligned_step0_to_160"
    assert transport["coordinate_system"]["points_2d_render"] == "realwonder_512_uv_xy"
    assert transport["coordinate_system"]["points_2d_video"] == "realwonder_resized_832_crop_480_uv_xy"
    assert int(transport["video_height"]) == OUT_H and int(transport["video_width"]) == OUT_W
    assert torch.equal(transport["frame_ids"], torch.arange(81, dtype=torch.long))
    assert torch.equal(transport["simulation_steps"], torch.arange(0, 161, 2, dtype=torch.long))
    assert transport["alignment_mapping"]["aligned_contract"] == "frames=S0,S2,S4,...,S160"
    assert transport["material_type"] == ["mpm_elastic"]

    points = transport["points_2d_video"].float().numpy()
    point_ids = transport["point_id"].long().numpy()
    bindings = transport["point_particle_binding"].long().numpy()
    projection_valid = transport["projection_valid"].bool().numpy()
    assert points.shape == (81, 15774, 2)
    assert point_ids.shape == (15774,) and np.array_equal(point_ids, np.arange(15774))
    assert bindings.shape == (15774, 5)
    assert np.isfinite(points).all()

    assert visibility_contract["artifact_kind"] == "aligned_raster_visibility_contract"
    assert visibility_contract["visibility_definition"]["not_projection_only"] is True
    assert torch.equal(visibility_contract["point_id"], transport["point_id"])
    assert torch.equal(visibility_contract["frame_ids"], transport["frame_ids"])
    authoritative_visibility = visibility_contract["aligned_visible"].bool().numpy()
    aligned_projection_valid = visibility_contract["aligned_projection_valid"].bool().numpy()
    assert authoritative_visibility.shape == (81, 15774)
    assert np.all(~authoritative_visibility | aligned_projection_valid)
    assert np.array_equal(aligned_projection_valid, projection_valid)

    raw_object = trajectory["objects"][0]
    assert trajectory["coordinate_system"] == "pytorch3d_world_and_realwonder_512_uv"
    assert int(trajectory["image_size"]) == SOURCE_SIZE
    assert raw_object["material_type"] == "mpm_elastic"
    assert tuple(raw_object["initial_points_uv"].shape) == (15774, 2)
    assert tuple(raw_object["points_uv"].shape) == (80, 15774, 2)
    assert tuple(raw_object["binding_particle_indices"].shape) == (15774, 5)
    assert torch.equal(raw_object["binding_particle_indices"].cpu(), transport["point_particle_binding"])
    raw_uv = torch.cat([raw_object["initial_points_uv"].unsqueeze(0), raw_object["points_uv"]], dim=0).float().numpy()
    mapped = raw_uv * (RESIZED_SIZE / SOURCE_SIZE)
    mapped[..., 1] -= CROP_TOP
    mapping_max_abs_diff = float(np.abs(mapped - points).max())
    assert mapping_max_abs_diff <= 1e-4, mapping_max_abs_diff

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None and image.shape == (OUT_H, OUT_W, 3)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    assert len(frame_paths) == 81
    assert sha256(image_path) == sha256(frame_paths[0])
    assert prompt_path.read_text(encoding="utf-8").strip()

    finite = np.isfinite(points).all(axis=-1)
    in_bounds = (
        (points[..., 0] >= 0.0) & (points[..., 0] < OUT_W)
        & (points[..., 1] >= 0.0) & (points[..., 1] < OUT_H)
    )
    visibility = authoritative_visibility & projection_valid & finite & in_bounds
    source_ids = np.flatnonzero(visibility[0])
    assert len(source_ids) > 0
    xy0 = points[0, source_ids]
    cell_x = np.floor(xy0[:, 0] / VAE_STRIDE).astype(np.int64)
    cell_y = np.floor(xy0[:, 1] / VAE_STRIDE).astype(np.int64)
    cell_id = cell_y * (OUT_W // VAE_STRIDE) + cell_x
    center_x = (cell_x + 0.5) * VAE_STRIDE
    center_y = (cell_y + 0.5) * VAE_STRIDE
    dist2 = (xy0[:, 0] - center_x) ** 2 + (xy0[:, 1] - center_y) ** 2
    order = np.lexsort((source_ids, dist2, cell_id))
    sorted_cells = cell_id[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_cells[1:] != sorted_cells[:-1]
    selected_ids = source_ids[order[first]].astype(np.int64)
    track_count = len(selected_ids)
    assert track_count == len(np.unique(cell_id))

    selected_tracks = points[:, selected_ids].astype(np.float32)
    selected_visibility = visibility[:, selected_ids].astype(np.bool_)
    selected_tracks = fill_nonfinite_tracks(selected_tracks, selected_visibility)
    tracks_out = selected_tracks[None]
    visibility_out = selected_visibility[None]
    assert tracks_out.shape == (1, 81, track_count, 2) and tracks_out.dtype == np.float32
    assert visibility_out.shape == (1, 81, track_count) and visibility_out.dtype == np.bool_
    assert np.isfinite(tracks_out).all() and visibility_out[0, 0].all()
    selected_cells = (
        np.floor(selected_tracks[0, :, 1] / 8).astype(np.int64) * 104
        + np.floor(selected_tracks[0, :, 0] / 8).astype(np.int64)
    )
    assert len(np.unique(selected_cells)) == track_count

    track_path = output / f"{args.case}_material_tracks_correct.npy"
    selected_visibility_path = output / f"{args.case}_material_visibility_correct.npy"
    ids_path = output / f"{args.case}_material_point_ids.npy"
    np.save(track_path, tracks_out)
    np.save(selected_visibility_path, visibility_out)
    np.save(ids_path, selected_ids)

    overlay = image.copy()
    for x, y in selected_tracks[0]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    overlay_path = output / f"{args.case}_tracks_frame0_overlay.png"
    assert cv2.imwrite(str(overlay_path), overlay)

    dynamic_path = output / f"{args.case}_material_tracks_overlay.mp4"
    writer = cv2.VideoWriter(str(dynamic_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (OUT_W, OUT_H))
    assert writer.isOpened()
    for frame_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        assert frame is not None and frame.shape == image.shape
        for point_index in np.flatnonzero(selected_visibility[frame_index]):
            x, y = selected_tracks[frame_index, point_index]
            cv2.circle(frame, (int(round(x)), int(round(y))), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    step_magnitude = np.linalg.norm(selected_tracks[1:] - selected_tracks[:-1], axis=-1)
    valid_pairs = selected_visibility[1:] & selected_visibility[:-1]
    final_displacement = np.linalg.norm(selected_tracks[-1] - selected_tracks[0], axis=-1)
    per_track_visibility = selected_visibility.mean(axis=0)
    report = {
        "status": "TREE_BRIDGE_GO",
        "case": args.case,
        "readiness": {
            "A_authoritative_persistent_trajectories": True,
            "B_persistent_material_id": True,
            "C_native_81_state_contract": True,
            "C_contract": "S0,S2,S4,...,S160; no 165-to-81 window selection",
            "D_authoritative_image_prompt_lineage": True,
            "E_deterministic_coordinate_mapping": True,
            "F_overlay_generated": True,
            "G_no_reconstruction_or_physics_rerun": True,
        },
        "lineage": {
            "trajectory_source": str(trajectory_path),
            "transport_ready": str(transport_path),
            "visibility_contract": str(visibility_path),
            "aligned_input_image": str(image_path),
            "prompt": str(prompt_path),
            "realwonder_baseline_record_only": str(baseline_path),
        },
        "coordinate_mapping": {
            "source": "RealWonder 512x512 UV",
            "output": "Wan-Move input 832x480 pixel XY",
            "formula": "x=uv_x*832/512; y=uv_y*832/512-176",
            "mapping_max_abs_diff_vs_aligned_transport": mapping_max_abs_diff,
        },
        "sampling": {
            "rule": "one persistent point per occupied frame-0 Wan VAE 8x8 cell; nearest to cell center; tie minimum material ID",
            "manual_track_count": False,
            "source_material_point_count": int(len(point_ids)),
            "source_visible_point_count": int(visibility[0].sum()),
            "selected_track_count": int(track_count),
        },
        "tracks": {
            "shape": list(tracks_out.shape),
            "dtype": str(tracks_out.dtype),
            "x_range": [float(selected_tracks[..., 0].min()), float(selected_tracks[..., 0].max())],
            "y_range": [float(selected_tracks[..., 1].min()), float(selected_tracks[..., 1].max())],
            "in_bounds_fraction": float(in_bounds[:, selected_ids].mean()),
        },
        "visibility": {
            "definition": visibility_contract["visibility_definition"],
            "shape": list(visibility_out.shape),
            "dtype": str(visibility_out.dtype),
            "global_fraction": float(selected_visibility.mean()),
            "per_track_mean": float(per_track_visibility.mean()),
            "per_track_min": float(per_track_visibility.min()),
        },
        "motion": {
            "valid_step_mean_px": float(step_magnitude[valid_pairs].mean()),
            "valid_step_p95_px": float(np.quantile(step_magnitude[valid_pairs], 0.95)),
            "start_to_end_mean_px": float(final_displacement.mean()),
            "start_to_end_p95_px": float(np.quantile(final_displacement, 0.95)),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    inputs = required + [report_path, track_path, selected_visibility_path, ids_path]
    (output / "input_sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in inputs), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
