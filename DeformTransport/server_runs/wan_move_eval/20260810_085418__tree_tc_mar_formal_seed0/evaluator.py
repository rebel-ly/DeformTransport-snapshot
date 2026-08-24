#!/usr/bin/env python3
"""Frozen visibility-aware Tree TC-MAR evaluator (CPU only)."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


ANCHORS = np.arange(4, 81, 4, dtype=np.int64)
OFFSETS = np.arange(8, dtype=np.float32) - 3.5
BOOTSTRAP_COUNT = 10000
BOOTSTRAP_SEED = 0
EXPECTED_CORRECT_SHA = "b7c6bccdb29b589552bcf65eeea193c0861460edf2fe796fbb3738edcf1a1d13"
EXPECTED_SHUFFLED_SHA = "d0a2a8240b83f48ef25213d14a67d892ee906db72ba1142c7886b9b73e04f5be"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    frames = []
    shape = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        shape = bgr.shape if shape is None else shape
        assert bgr.shape == shape
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(frames) == 81 and shape == (464, 832, 3), (len(frames), shape)
    return frames


def complete_patch(points, width, height):
    return (
        (points[..., 0] + OFFSETS[0] >= 0.0)
        & (points[..., 0] + OFFSETS[-1] <= width - 1)
        & (points[..., 1] + OFFSETS[0] >= 0.0)
        & (points[..., 1] + OFFSETS[-1] <= height - 1)
    )


def bilinear_patches(rgb, points):
    h, w = rgb.shape[:2]
    x = points[:, 0, None, None] + OFFSETS[None, None, :]
    y = points[:, 1, None, None] + OFFSETS[None, :, None]
    x = np.broadcast_to(x, (len(points), 8, 8))
    y = np.broadcast_to(y, (len(points), 8, 8))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    image = rgb.astype(np.float32) / 255.0
    return (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    ).astype(np.float32)


def mean_lab(patches):
    count = patches.shape[0]
    stacked = np.ascontiguousarray(patches.reshape(count * 8, 8, 3), dtype=np.float32)
    return cv2.cvtColor(stacked, cv2.COLOR_RGB2LAB).reshape(count, 8, 8, 3).mean(axis=(1, 2), dtype=np.float64)


def summary(values):
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def paired(correct, shuffled):
    difference = shuffled - correct
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_COUNT, dtype=np.float64)
    for start in range(0, BOOTSTRAP_COUNT, 256):
        stop = min(start + 256, BOOTSTRAP_COUNT)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        means[start:stop] = difference[indices].mean(axis=1)
    ci = np.percentile(means, [2.5, 97.5])
    return {
        "mean_difference_shuffled_minus_correct": float(difference.mean()),
        "median_difference_shuffled_minus_correct": float(np.median(difference)),
        "fraction_shuffled_gt_correct": float(np.mean(difference > 0)),
        "relative_mean_improvement": float(difference.mean() / shuffled.mean()),
        "bootstrap_95_ci_mean_difference": [float(ci[0]), float(ci[1])],
        "bootstrap_resamples": BOOTSTRAP_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_comparisons(correct, shuffled, out_dir):
    writer = cv2.VideoWriter(str(out_dir / "tree_correct_vs_shuffled.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (1664, 464))
    assert writer.isOpened()
    for c_rgb, s_rgb in zip(correct, shuffled):
        c = cv2.cvtColor(c_rgb, cv2.COLOR_RGB2BGR)
        s = cv2.cvtColor(s_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(c, "Correct", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(s, "Identity-Shuffled", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(np.concatenate([c, s], axis=1))
    writer.release()
    sample_ids = [0, 20, 40, 60, 80]
    rows = []
    for label, videos in (("Correct", correct), ("Identity-Shuffled", shuffled)):
        panels = []
        for frame_id in sample_ids:
            panel = cv2.cvtColor(videos[frame_id], cv2.COLOR_RGB2BGR)
            cv2.putText(panel, f"{label}  t={frame_id}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    assert cv2.imwrite(str(out_dir / "tree_contact_sheet.png"), np.concatenate(rows, axis=0))


def main():
    out_dir = Path(__file__).resolve().parent
    dt = out_dir.parents[2]
    wan = dt.parent / "Wan-Move"
    batch = dt / "server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0"
    bridge = dt / "server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks"
    source_path = dt / "server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/resized_input_image.png"
    tracks_path = bridge / "tree_material_tracks_correct.npy"
    visibility_path = bridge / "tree_material_visibility_correct.npy"
    correct_path = batch / "correct/tree_formal_correct_seed0.mp4"
    shuffled_path = batch / "shuffled/tree_formal_identity_shuffled_seed0.mp4"
    assert sha256(correct_path) == EXPECTED_CORRECT_SHA
    assert sha256(shuffled_path) == EXPECTED_SHUFFLED_SHA
    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    assert source_bgr is not None and source_bgr.shape == (480, 832, 3)
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    tracks = np.load(tracks_path, allow_pickle=False)
    visibility = np.load(visibility_path, allow_pickle=False)
    assert tracks.shape == (1, 81, 713, 2) and tracks.dtype == np.float32
    assert visibility.shape == (1, 81, 713) and visibility.dtype == np.bool_
    assert np.isfinite(tracks).all()
    correct_frames = read_video(correct_path)
    shuffled_frames = read_video(shuffled_path)
    make_comparisons(correct_frames, shuffled_frames, out_dir)

    true_tracks = tracks[0]
    true_visibility = visibility[0]
    source_points = true_tracks[0]
    future_points = true_tracks[ANCHORS].copy()
    future_points[..., 1] *= 464.0 / 480.0
    source_valid = complete_patch(source_points, 832, 480)
    future_boundary_valid = complete_patch(future_points, 832, 464)
    visible = true_visibility[ANCHORS]
    geometry_valid = future_boundary_valid & source_valid[None, :]
    observation_valid = visible & geometry_valid
    valid_track_mask = observation_valid.any(axis=0)
    valid_track_ids = np.flatnonzero(valid_track_mask)
    assert len(valid_track_ids) > 0
    assert np.array_equal(observation_valid, visible & geometry_valid)

    source_patches = bilinear_patches(source, source_points)
    source_lab = mean_lab(source_patches)
    lab_correct = np.full((713, 20), np.nan, dtype=np.float64)
    lab_shuffled = np.full_like(lab_correct, np.nan)
    rgb_correct = np.full_like(lab_correct, np.nan)
    rgb_shuffled = np.full_like(lab_correct, np.nan)
    anchor_rows = []
    for anchor_index, anchor in enumerate(ANCHORS):
        ids = np.flatnonzero(observation_valid[anchor_index])
        points = future_points[anchor_index, ids]
        cp = bilinear_patches(correct_frames[int(anchor)], points)
        sp = bilinear_patches(shuffled_frames[int(anchor)], points)
        lab_correct[ids, anchor_index] = np.linalg.norm(mean_lab(cp) - source_lab[ids], axis=1)
        lab_shuffled[ids, anchor_index] = np.linalg.norm(mean_lab(sp) - source_lab[ids], axis=1)
        rgb_correct[ids, anchor_index] = np.abs(cp - source_patches[ids]).mean(axis=(1, 2, 3), dtype=np.float64)
        rgb_shuffled[ids, anchor_index] = np.abs(sp - source_patches[ids]).mean(axis=(1, 2, 3), dtype=np.float64)
        anchor_rows.append({
            "anchor": int(anchor),
            "visible_tracks": int(visible[anchor_index].sum()),
            "excluded_boundary": int((visible[anchor_index] & ~geometry_valid[anchor_index]).sum()),
            "valid_tracks": int(len(ids)),
            "lab_correct_mean": float(np.nanmean(lab_correct[:, anchor_index])),
            "lab_shuffled_mean": float(np.nanmean(lab_shuffled[:, anchor_index])),
            "lab_difference_mean_shuffled_minus_correct": float(np.nanmean(lab_shuffled[:, anchor_index] - lab_correct[:, anchor_index])),
            "rgb_l1_correct_mean": float(np.nanmean(rgb_correct[:, anchor_index])),
            "rgb_l1_shuffled_mean": float(np.nanmean(rgb_shuffled[:, anchor_index])),
            "rgb_l1_difference_mean_shuffled_minus_correct": float(np.nanmean(rgb_shuffled[:, anchor_index] - rgb_correct[:, anchor_index])),
        })

    def track_mean(values):
        return np.array([np.nanmean(values[track_id]) for track_id in valid_track_ids], dtype=np.float64)

    plc, pls = track_mean(lab_correct), track_mean(lab_shuffled)
    prc, prs = track_mean(rgb_correct), track_mean(rgb_shuffled)
    lab_pair = paired(plc, pls)
    rgb_pair = paired(prc, prs)
    decision = "GO" if lab_pair["bootstrap_95_ci_mean_difference"][0] > 0 else "STOP"
    counts = {
        "selected_tracks": 713,
        "valid_tracks_at_least_one_visible_anchor": int(len(valid_track_ids)),
        "total_anchor_slots": int(713 * 20),
        "visible_anchor_observations": int(visible.sum()),
        "excluded_invisible_observations": int((~visible).sum()),
        "excluded_boundary_observations": int((visible & ~geometry_valid).sum()),
        "valid_observations_per_method": int(observation_valid.sum()),
    }
    report = {
        "protocol": "Tree visibility-aware TC-MAR",
        "implementation": {
            "coordinate_mapping": "x_out=x_input; y_out=y_input*464/480",
            "patch_sampling": "exact float32 bilinear 8x8 centered grid, offsets -3.5..+3.5, no clamping",
            "lab_conversion": "RGB float32 [0,1] -> OpenCV COLOR_RGB2LAB; mean Lab per patch; L2 source-to-future",
            "rgb_l1": "mean absolute elementwise source-to-future 8x8 RGB [0,1] difference",
            "anchors": ANCHORS.tolist(),
            "visibility_rule": "use TRUE visibility at each future anchor; per-track mean over valid visible anchors; require >=1 valid anchor; identical observation set for both arms",
        },
        "counts": counts,
        "primary_lab_tc_mar": {"correct": summary(plc), "shuffled": summary(pls), "paired": lab_pair},
        "secondary_rgb_l1": {"correct": summary(prc), "shuffled": summary(prs), "paired": rgb_pair},
        "pre_registered_tree_decision": decision,
        "decision_rule": "GO iff lower bound of 95% valid-track bootstrap CI for mean(Shuffled Lab - Correct Lab) > 0",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "per_anchor_metrics.csv", list(anchor_rows[0]), anchor_rows)
    track_rows = []
    lookup = {int(track_id): row for row, track_id in enumerate(valid_track_ids)}
    for track_id in range(713):
        row = lookup.get(track_id)
        track_rows.append({
            "track_id": track_id,
            "valid_anchor_count": int(observation_valid[:, track_id].sum()),
            "valid_for_paired_analysis": int(row is not None),
            "lab_correct": "" if row is None else plc[row],
            "lab_shuffled": "" if row is None else pls[row],
            "lab_difference_shuffled_minus_correct": "" if row is None else pls[row] - plc[row],
            "rgb_l1_correct": "" if row is None else prc[row],
            "rgb_l1_shuffled": "" if row is None else prs[row],
            "rgb_l1_difference_shuffled_minus_correct": "" if row is None else prs[row] - prc[row],
        })
    write_csv(out_dir / "per_track_metrics.csv", list(track_rows[0]), track_rows)
    inputs = [source_path, tracks_path, visibility_path, correct_path, shuffled_path]
    (out_dir / "input_sha256.txt").write_text("".join(f"{sha256(path)}  {path}\n" for path in inputs), encoding="utf-8")
    (out_dir / "deformtransport_git_head.txt").write_text(git_output(dt, "rev-parse", "HEAD"))
    (out_dir / "deformtransport_git_status.txt").write_text(git_output(dt, "status", "--short", "--branch"))
    (out_dir / "wanmove_git_head.txt").write_text(git_output(wan, "rev-parse", "HEAD"))
    (out_dir / "wanmove_git_status.txt").write_text(git_output(wan, "status", "--short", "--branch"))
    (out_dir / "wanmove_rng_patch.diff").write_text(git_output(wan, "diff", "--", "wan/wan_move.py"))
    (out_dir / "audit.txt").write_text(
        "Tree visibility-aware TC-MAR formal audit\n"
        + f"counts={json.dumps(counts, sort_keys=True)}\n"
        + "observation_set_correct_equals_shuffled=true\n"
        + "anchors=4,8,...,80\npatch=8x8 exact bilinear\nbootstrap=10000 tracks seed=0\n"
        + f"decision={decision}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
