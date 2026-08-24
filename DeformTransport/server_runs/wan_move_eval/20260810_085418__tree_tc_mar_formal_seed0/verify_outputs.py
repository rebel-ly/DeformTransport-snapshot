#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_ci(values):
    rng = np.random.default_rng(0)
    means = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 256):
        stop = min(start + 256, 10000)
        ids = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[ids].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


out = Path(__file__).resolve().parent
dt = out.parents[2]
batch = dt / "server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0"
bridge = dt / "server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks"
report = json.loads((out / "report.json").read_text())
with (out / "per_track_metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
valid = [row for row in rows if row["valid_for_paired_analysis"] == "1"]
assert len(rows) == 713 and len(valid) == 709
visibility = np.load(bridge / "tree_material_visibility_correct.npy", allow_pickle=False)[0]
anchors = np.arange(4, 81, 4)
visible = visibility[anchors]
assert int(visible.sum()) == 8743
assert int((~visible).sum()) == 5517
assert int(visible.any(axis=0).sum()) == 709

checks = {}
for prefix, section in (("lab", "primary_lab_tc_mar"), ("rgb_l1", "secondary_rgb_l1")):
    correct = np.array([float(row[prefix + "_correct"]) for row in valid])
    shuffled = np.array([float(row[prefix + "_shuffled"]) for row in valid])
    difference = shuffled - correct
    paired = report[section]["paired"]
    ci = bootstrap_ci(difference)
    checks[section] = {
        "correct_mean": bool(np.isclose(correct.mean(), report[section]["correct"]["mean"], atol=1e-12, rtol=0)),
        "shuffled_mean": bool(np.isclose(shuffled.mean(), report[section]["shuffled"]["mean"], atol=1e-12, rtol=0)),
        "paired_mean": bool(np.isclose(difference.mean(), paired["mean_difference_shuffled_minus_correct"], atol=1e-12, rtol=0)),
        "paired_median": bool(np.isclose(np.median(difference), paired["median_difference_shuffled_minus_correct"], atol=1e-12, rtol=0)),
        "fraction": bool(np.isclose(np.mean(difference > 0), paired["fraction_shuffled_gt_correct"], atol=1e-12, rtol=0)),
        "bootstrap_ci": bool(np.allclose(ci, paired["bootstrap_95_ci_mean_difference"], atol=1e-12, rtol=0)),
        "label_swap_sign_exact": bool(np.array_equal(correct - shuffled, -difference)),
    }
    assert all(checks[section].values())

correct_video = batch / "correct/tree_formal_correct_seed0.mp4"
shuffled_video = batch / "shuffled/tree_formal_identity_shuffled_seed0.mp4"
for arm, path in (("correct", correct_video), ("shuffled", shuffled_video)):
    assert (batch / arm / "exit_code.txt").read_text().strip() == "0"
    expected = (batch / arm / "output_sha256.txt").read_text().split()[0]
    assert sha256(path) == expected
cap = cv2.VideoCapture(str(out / "tree_correct_vs_shuffled.mp4"))
count = 0
shape = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    shape = frame.shape
    count += 1
cap.release()
assert count == 81 and shape == (464, 1664, 3)
sheet = cv2.imread(str(out / "tree_contact_sheet.png"))
assert sheet is not None and sheet.shape == (928, 4160, 3)
result = {
    "status": "TREE_TC_MAR_OUTPUT_AUDIT_OK",
    "metric_recomputation_checks": checks,
    "visibility_counts_recomputed": {"selected_tracks": 713, "valid_tracks": 709, "visible_observations": 8743, "excluded_invisible": 5517},
    "formal_video_sha_matches": True,
    "comparison_video": {"frames": count, "shape": list(shape)},
    "contact_sheet_shape": list(sheet.shape),
    "decision": report["pre_registered_tree_decision"],
}
(out / "verification_report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
