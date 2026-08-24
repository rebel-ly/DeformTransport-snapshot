#!/usr/bin/env python3
"""Build the frozen source-identity-shuffled control for Wan-Move tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--correct-tracks", type=Path, required=True)
    parser.add_argument("--correct-visibility", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    correct = np.load(args.correct_tracks.resolve(), allow_pickle=False)
    visibility = np.load(args.correct_visibility.resolve(), allow_pickle=False)
    assert correct.ndim == 4 and correct.shape[0] == 1 and correct.shape[1] == 81 and correct.shape[-1] == 2
    assert correct.dtype == np.float32
    assert visibility.shape == correct.shape[:-1] and visibility.dtype == np.bool_
    count = correct.shape[2]
    rng = np.random.default_rng(0)
    attempts = 0
    while True:
        attempts += 1
        permutation = rng.permutation(count)
        if not np.any(permutation == np.arange(count)):
            break
    shuffled = correct.copy()
    shuffled[0, 0] = correct[0, 0, permutation]
    fixed_points = int(np.sum(permutation == np.arange(count)))
    future_diff = float(np.max(np.abs(shuffled[:, 1:] - correct[:, 1:])))
    visibility_copy = visibility.copy()
    assert fixed_points == 0
    assert future_diff == 0.0
    assert np.array_equal(visibility_copy, visibility)
    track_path = output / f"{args.case}_material_tracks_identity_shuffled.npy"
    permutation_path = output / "source_identity_permutation.npy"
    visibility_path = output / f"{args.case}_material_visibility_identity_shuffled.npy"
    np.save(track_path, shuffled)
    np.save(permutation_path, permutation.astype(np.int64))
    np.save(visibility_path, visibility_copy)
    displacement = np.linalg.norm(shuffled[0, 0] - correct[0, 0], axis=-1)
    report = {
        "seed": 0,
        "derangement_attempts": attempts,
        "num_tracks": int(count),
        "fixed_points": fixed_points,
        "future_trajectory_max_abs_diff": future_diff,
        "visibility_bit_identical": bool(np.array_equal(visibility_copy, visibility)),
        "source_binding_displacement_mean_px": float(displacement.mean()),
        "source_binding_displacement_p50_px": float(np.median(displacement)),
        "source_binding_displacement_p95_px": float(np.quantile(displacement, 0.95)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
