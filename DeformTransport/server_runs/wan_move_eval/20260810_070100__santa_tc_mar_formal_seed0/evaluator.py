#!/usr/bin/env python3
"""Frozen CPU-only TC-MAR evaluator for the formal Santa comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


ANCHORS = np.arange(4, 81, 4, dtype=np.int64)
PATCH_OFFSETS = np.arange(8, dtype=np.float32) - 3.5
BOOTSTRAP_COUNT = 10_000
BOOTSTRAP_SEED = 0
EXPECTED_CORRECT_SHA256 = "1c20d32c82049e0cb5fd8f19b08957b4e5b0b70020267c357293cc568d43cf14"
EXPECTED_SHUFFLED_SHA256 = "489e1c7570827e748609cf25914aa4fea8060f84047e7f7342c015325f07d638"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return result.stdout


def read_video(path: Path) -> tuple[dict[int, np.ndarray], tuple[int, int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AssertionError(f"cannot open video: {path}")
    selected: dict[int, np.ndarray] = {}
    count = 0
    shape = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if shape is None:
            shape = bgr.shape
        assert bgr.shape == shape, f"inconsistent frame shape in {path}"
        if count in ANCHORS:
            selected[count] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        count += 1
    cap.release()
    assert shape is not None
    assert set(selected) == set(ANCHORS.tolist()), f"missing anchors in {path}"
    return selected, (count, shape[0], shape[1])


def complete_patch_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    low_x = points[..., 0] + PATCH_OFFSETS[0]
    high_x = points[..., 0] + PATCH_OFFSETS[-1]
    low_y = points[..., 1] + PATCH_OFFSETS[0]
    high_y = points[..., 1] + PATCH_OFFSETS[-1]
    return (low_x >= 0.0) & (high_x <= width - 1) & (low_y >= 0.0) & (high_y <= height - 1)


def bilinear_patches(rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample centered 8x8 RGB patches with exact float32 bilinear interpolation."""
    h, w = rgb.shape[:2]
    x = points[:, 0, None, None] + PATCH_OFFSETS[None, None, :]
    y = points[:, 1, None, None] + PATCH_OFFSETS[None, :, None]
    x = np.broadcast_to(x, (len(points), 8, 8))
    y = np.broadcast_to(y, (len(points), 8, 8))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    image = rgb.astype(np.float32) / 255.0
    p00 = image[y0, x0]
    p01 = image[y0, x1]
    p10 = image[y1, x0]
    p11 = image[y1, x1]
    return (
        p00 * (1.0 - dx) * (1.0 - dy)
        + p01 * dx * (1.0 - dy)
        + p10 * (1.0 - dx) * dy
        + p11 * dx * dy
    ).astype(np.float32)


def mean_lab(patches_rgb01: np.ndarray) -> np.ndarray:
    n = patches_rgb01.shape[0]
    stacked = np.ascontiguousarray(patches_rgb01.reshape(n * 8, 8, 3), dtype=np.float32)
    lab = cv2.cvtColor(stacked, cv2.COLOR_RGB2LAB).reshape(n, 8, 8, 3)
    return lab.mean(axis=(1, 2), dtype=np.float64)


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def paired_summary(correct: np.ndarray, shuffled: np.ndarray) -> dict[str, float | list[float]]:
    difference = shuffled - correct
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_means = np.empty(BOOTSTRAP_COUNT, dtype=np.float64)
    chunk = 256
    for start in range(0, BOOTSTRAP_COUNT, chunk):
        stop = min(start + chunk, BOOTSTRAP_COUNT)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        bootstrap_means[start:stop] = difference[indices].mean(axis=1)
    ci = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "mean_difference_shuffled_minus_correct": float(np.mean(difference)),
        "median_difference_shuffled_minus_correct": float(np.median(difference)),
        "fraction_shuffled_gt_correct": float(np.mean(difference > 0.0)),
        "relative_mean_improvement": float(np.mean(difference) / np.mean(shuffled)),
        "bootstrap_95_ci_mean_difference": [float(ci[0]), float(ci[1])],
        "bootstrap_resamples": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    dt_root = out_dir.parents[2]
    wan_root = dt_root.parent / "Wan-Move"
    batch = dt_root / "server_runs/wan_move_formal/20260809_195255__santa_correct_vs_identity_shuffled_seed0"
    correct_path = batch / "correct/santa_formal_correct_seed0.mp4"
    shuffled_path = batch / "shuffled/santa_formal_identity_shuffled_seed0.mp4"
    source_path = dt_root / "server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"
    tracks_path = dt_root / "server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_tracks_correct.npy"
    visibility_path = dt_root / "server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_visibility_correct.npy"

    assert sha256(correct_path) == EXPECTED_CORRECT_SHA256
    assert sha256(shuffled_path) == EXPECTED_SHUFFLED_SHA256
    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    assert source_bgr is not None
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    tracks = np.load(tracks_path, allow_pickle=False)
    visibility = np.load(visibility_path, allow_pickle=False)
    assert source.shape == (480, 832, 3), source.shape
    assert tracks.shape == (1, 81, 1277, 2), tracks.shape
    assert tracks.dtype == np.float32, tracks.dtype
    assert visibility.shape == (1, 81, 1277), visibility.shape
    assert visibility.dtype == np.bool_, visibility.dtype
    assert np.isfinite(tracks[0, np.r_[0, ANCHORS]]).all()
    correct_frames, correct_info = read_video(correct_path)
    shuffled_frames, shuffled_info = read_video(shuffled_path)
    assert correct_info == (81, 464, 832), correct_info
    assert shuffled_info == correct_info, (correct_info, shuffled_info)
    if args.check_only:
        print("All frozen input and shape assertions passed.")
        return

    true_tracks = tracks[0]
    true_visibility = visibility[0]
    source_points = true_tracks[0]
    future_points = true_tracks[ANCHORS].copy()
    future_points[..., 1] *= 464.0 / 480.0
    source_patch_valid = complete_patch_mask(source_points, 832, 480)
    future_patch_valid = complete_patch_mask(future_points, 832, 464)
    visibility_valid = true_visibility[0] & true_visibility[ANCHORS].all(axis=0)
    geometry_valid = source_patch_valid & future_patch_valid.all(axis=0)
    valid = visibility_valid & geometry_valid
    valid_ids = np.flatnonzero(valid)
    assert len(valid_ids) > 0

    source_patches = bilinear_patches(source, source_points[valid])
    source_lab = mean_lab(source_patches)
    lab_correct = np.empty((len(valid_ids), len(ANCHORS)), dtype=np.float64)
    lab_shuffled = np.empty_like(lab_correct)
    rgb_correct = np.empty_like(lab_correct)
    rgb_shuffled = np.empty_like(lab_correct)

    for anchor_index, anchor in enumerate(ANCHORS):
        points = future_points[anchor_index, valid]
        correct_patches = bilinear_patches(correct_frames[int(anchor)], points)
        shuffled_patches = bilinear_patches(shuffled_frames[int(anchor)], points)
        lab_correct[:, anchor_index] = np.linalg.norm(mean_lab(correct_patches) - source_lab, axis=1)
        lab_shuffled[:, anchor_index] = np.linalg.norm(mean_lab(shuffled_patches) - source_lab, axis=1)
        rgb_correct[:, anchor_index] = np.abs(correct_patches - source_patches).mean(axis=(1, 2, 3), dtype=np.float64)
        rgb_shuffled[:, anchor_index] = np.abs(shuffled_patches - source_patches).mean(axis=(1, 2, 3), dtype=np.float64)

    per_track_lab_correct = lab_correct.mean(axis=1)
    per_track_lab_shuffled = lab_shuffled.mean(axis=1)
    per_track_rgb_correct = rgb_correct.mean(axis=1)
    per_track_rgb_shuffled = rgb_shuffled.mean(axis=1)
    lab_paired = paired_summary(per_track_lab_correct, per_track_lab_shuffled)
    rgb_paired = paired_summary(per_track_rgb_correct, per_track_rgb_shuffled)
    decision = "GO" if lab_paired["bootstrap_95_ci_mean_difference"][0] > 0.0 else "STOP"

    report = {
        "protocol": "TC-MAR (Track-Conditioned Material Appearance Retention)",
        "implementation": {
            "coordinate_mapping": "x_out=x_input; y_out=y_input*464/480",
            "patch_sampling": "exact float32 bilinear sampling on an 8x8 grid centered at each point, offsets -3.5..+3.5; no clamping",
            "lab_conversion": "bilinearly sampled RGB float32 [0,1] -> cv2.cvtColor(..., COLOR_RGB2LAB); OpenCV float Lab has L in [0,100] and approximately a,b in [-127,127]; mean Lab vector per patch",
            "rgb_l1": "mean absolute elementwise difference between corresponding source and future 8x8 RGB [0,1] patches",
            "anchors": ANCHORS.tolist(),
            "complete_case_rule": "track retained only when source and all 20 future patches are complete and visibility is true at source/all anchors",
        },
        "counts": {
            "total_tracks": int(len(valid)),
            "valid_tracks": int(valid.sum()),
            "excluded_tracks": int((~valid).sum()),
            "excluded_for_visibility": int((~visibility_valid).sum()),
            "excluded_for_patch_geometry": int((~geometry_valid).sum()),
            "excluded_observations_patch_geometry": int((~future_patch_valid).sum() + (~source_patch_valid).sum() * len(ANCHORS)),
            "valid_observations_per_method": int(valid.sum() * len(ANCHORS)),
        },
        "primary_lab_tc_mar": {
            "correct": summary(per_track_lab_correct),
            "shuffled": summary(per_track_lab_shuffled),
            "paired": lab_paired,
        },
        "secondary_rgb_l1": {
            "correct": summary(per_track_rgb_correct),
            "shuffled": summary(per_track_rgb_shuffled),
            "paired": rgb_paired,
        },
        "pre_registered_decision": decision,
        "decision_rule": "GO iff lower bound of 95% track-bootstrap CI for mean(Shuffled Lab - Correct Lab) > 0",
    }

    track_rows: list[dict[str, object]] = []
    valid_lookup = {int(track_id): row for row, track_id in enumerate(valid_ids)}
    for track_id in range(len(valid)):
        row_id = valid_lookup.get(track_id)
        track_rows.append({
            "track_id": track_id,
            "valid": int(valid[track_id]),
            "source_patch_valid": int(source_patch_valid[track_id]),
            "all_anchor_patches_valid": int(future_patch_valid[:, track_id].all()),
            "all_required_visibility": int(visibility_valid[track_id]),
            "lab_correct": "" if row_id is None else per_track_lab_correct[row_id],
            "lab_shuffled": "" if row_id is None else per_track_lab_shuffled[row_id],
            "lab_difference_shuffled_minus_correct": "" if row_id is None else per_track_lab_shuffled[row_id] - per_track_lab_correct[row_id],
            "rgb_l1_correct": "" if row_id is None else per_track_rgb_correct[row_id],
            "rgb_l1_shuffled": "" if row_id is None else per_track_rgb_shuffled[row_id],
            "rgb_l1_difference_shuffled_minus_correct": "" if row_id is None else per_track_rgb_shuffled[row_id] - per_track_rgb_correct[row_id],
        })
    write_csv(out_dir / "per_track_metrics.csv", list(track_rows[0]), track_rows)

    anchor_rows: list[dict[str, object]] = []
    for anchor_index, anchor in enumerate(ANCHORS):
        anchor_rows.append({
            "anchor": int(anchor),
            "valid_tracks": len(valid_ids),
            "lab_correct_mean": float(lab_correct[:, anchor_index].mean()),
            "lab_shuffled_mean": float(lab_shuffled[:, anchor_index].mean()),
            "lab_difference_mean_shuffled_minus_correct": float((lab_shuffled[:, anchor_index] - lab_correct[:, anchor_index]).mean()),
            "rgb_l1_correct_mean": float(rgb_correct[:, anchor_index].mean()),
            "rgb_l1_shuffled_mean": float(rgb_shuffled[:, anchor_index].mean()),
            "rgb_l1_difference_mean_shuffled_minus_correct": float((rgb_shuffled[:, anchor_index] - rgb_correct[:, anchor_index]).mean()),
        })
    write_csv(out_dir / "per_anchor_metrics.csv", list(anchor_rows[0]), anchor_rows)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    inputs = [source_path, tracks_path, visibility_path, correct_path, shuffled_path]
    (out_dir / "input_sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path}\n" for path in inputs), encoding="utf-8"
    )
    (out_dir / "deformtransport_git_head.txt").write_text(git_output(dt_root, "rev-parse", "HEAD"), encoding="utf-8")
    (out_dir / "deformtransport_git_status.txt").write_text(git_output(dt_root, "status", "--short", "--branch"), encoding="utf-8")
    (out_dir / "wanmove_git_head.txt").write_text(git_output(wan_root, "rev-parse", "HEAD"), encoding="utf-8")
    (out_dir / "wanmove_git_status.txt").write_text(git_output(wan_root, "status", "--short", "--branch"), encoding="utf-8")
    (out_dir / "wanmove_rng_patch.diff").write_text(git_output(wan_root, "diff", "--", "wan/wan_move.py"), encoding="utf-8")
    audit_lines = [
        "TC-MAR frozen formal evaluation audit",
        f"coordinate mapping: {report['implementation']['coordinate_mapping']}",
        f"Lab conversion: {report['implementation']['lab_conversion']}",
        f"patch sampling: {report['implementation']['patch_sampling']}",
        f"anchors: {ANCHORS.tolist()}",
        f"valid tracks: {report['counts']['valid_tracks']} / {report['counts']['total_tracks']}",
        f"excluded tracks: {report['counts']['excluded_tracks']}",
        f"excluded patch-geometry observations: {report['counts']['excluded_observations_patch_geometry']}",
        f"bootstrap: unit=material track, resamples={BOOTSTRAP_COUNT}, seed={BOOTSTRAP_SEED}",
        f"decision: {decision}",
    ]
    (out_dir / "audit.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
