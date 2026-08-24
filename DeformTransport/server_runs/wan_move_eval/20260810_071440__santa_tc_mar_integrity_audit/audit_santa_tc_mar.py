#!/usr/bin/env python3
"""Independent, read-only integrity audit of the frozen Santa TC-MAR result."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


EXPECTED_ANCHORS = list(range(4, 81, 4))
EXPECTED_TRACKS = 1277
EXPECTED_BOOTSTRAP_SEED = 0
EXPECTED_BOOTSTRAP_COUNT = 10000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(EXPECTED_BOOTSTRAP_SEED)
    means = np.empty(EXPECTED_BOOTSTRAP_COUNT, dtype=np.float64)
    for start in range(0, EXPECTED_BOOTSTRAP_COUNT, 256):
        stop = min(start + 256, EXPECTED_BOOTSTRAP_COUNT)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=0.0, atol=1e-12))


def metric_audit(rows: list[dict[str, str]], report: dict, prefix: str, section: str) -> dict:
    correct = np.array([float(row[f"{prefix}_correct"]) for row in rows], dtype=np.float64)
    shuffled = np.array([float(row[f"{prefix}_shuffled"]) for row in rows], dtype=np.float64)
    stored_difference = np.array(
        [float(row[f"{prefix}_difference_shuffled_minus_correct"]) for row in rows], dtype=np.float64
    )
    difference = shuffled - correct
    swapped_difference = correct - shuffled
    paired = report[section]["paired"]
    calculated = {
        "correct": {
            "mean": float(correct.mean()),
            "median": float(np.median(correct)),
            "p95": float(np.percentile(correct, 95)),
        },
        "shuffled": {
            "mean": float(shuffled.mean()),
            "median": float(np.median(shuffled)),
            "p95": float(np.percentile(shuffled, 95)),
        },
        "paired": {
            "mean_difference_shuffled_minus_correct": float(difference.mean()),
            "median_difference_shuffled_minus_correct": float(np.median(difference)),
            "fraction_shuffled_gt_correct": float(np.mean(difference > 0.0)),
            "relative_mean_improvement": float(difference.mean() / shuffled.mean()),
            "bootstrap_95_ci_mean_difference": bootstrap_ci(difference),
        },
    }
    checks = {
        "stored_per_track_difference_exact": bool(np.array_equal(stored_difference, difference)),
        "label_swap_sign_exact": bool(np.array_equal(swapped_difference, -difference)),
    }
    for arm in ("correct", "shuffled"):
        for statistic in ("mean", "median", "p95"):
            checks[f"{arm}_{statistic}_matches"] = close(
                calculated[arm][statistic], report[section][arm][statistic]
            )
    for statistic in (
        "mean_difference_shuffled_minus_correct",
        "median_difference_shuffled_minus_correct",
        "fraction_shuffled_gt_correct",
        "relative_mean_improvement",
    ):
        checks[f"paired_{statistic}_matches"] = close(calculated["paired"][statistic], paired[statistic])
    checks["bootstrap_ci_matches"] = all(
        close(a, b)
        for a, b in zip(calculated["paired"]["bootstrap_95_ci_mean_difference"], paired["bootstrap_95_ci_mean_difference"])
    )
    return {"calculated": calculated, "checks": checks, "ok": all(checks.values())}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    dt_root = out_dir.parents[2]
    frozen = dt_root / "server_runs/wan_move_eval/20260810_070100__santa_tc_mar_formal_seed0"
    report = json.loads((frozen / "report.json").read_text(encoding="utf-8"))
    with (frozen / "per_track_metrics.csv").open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    valid_rows = [row for row in all_rows if row["valid"] == "1"]
    with (frozen / "per_anchor_metrics.csv").open(newline="", encoding="utf-8") as handle:
        anchor_rows = list(csv.DictReader(handle))

    evaluator_source = (frozen / "evaluator.py").read_text(encoding="utf-8")
    reference_checks = {
        "true_source_points": "source_points = true_tracks[0]" in evaluator_source,
        "true_future_points": "future_points = true_tracks[ANCHORS].copy()" in evaluator_source,
        "single_true_source_patch_reference": "source_patches = bilinear_patches(source, source_points[valid])" in evaluator_source,
        "shared_true_future_points": "points = future_points[anchor_index, valid]" in evaluator_source,
        "correct_uses_shared_points": "bilinear_patches(correct_frames[int(anchor)], points)" in evaluator_source,
        "shuffled_uses_shared_points": "bilinear_patches(shuffled_frames[int(anchor)], points)" in evaluator_source,
        "no_permuted_evaluation_reference": "perm(" not in evaluator_source and "permutation" not in evaluator_source,
    }

    sha_checks = []
    for line in (frozen / "input_sha256.txt").read_text(encoding="utf-8").splitlines():
        expected, raw_path = line.split("  ", 1)
        path = Path(raw_path)
        sha_checks.append({"path": str(path), "expected": expected, "actual": file_sha256(path), "ok": file_sha256(path) == expected})

    dt_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dt_root, text=True).strip()
    wan_root = dt_root.parent / "Wan-Move"
    wan_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=wan_root, text=True).strip()
    provenance_checks = {
        "deformtransport_head_matches": (frozen / "deformtransport_git_head.txt").read_text().strip() == dt_head,
        "wanmove_head_matches": (frozen / "wanmove_git_head.txt").read_text().strip() == wan_head,
        "deformtransport_status_recorded": (frozen / "deformtransport_git_status.txt").stat().st_size > 0,
        "wanmove_status_recorded": (frozen / "wanmove_git_status.txt").stat().st_size > 0,
        "wanmove_rng_patch_recorded": (frozen / "wanmove_rng_patch.diff").stat().st_size > 0,
    }

    fixed_checks = {
        "track_count_report": report["counts"]["total_tracks"] == EXPECTED_TRACKS,
        "track_count_csv": len(all_rows) == EXPECTED_TRACKS,
        "valid_track_count": len(valid_rows) == EXPECTED_TRACKS,
        "anchors_report": report["implementation"]["anchors"] == EXPECTED_ANCHORS,
        "anchors_csv": [int(row["anchor"]) for row in anchor_rows] == EXPECTED_ANCHORS,
        "bootstrap_seed_lab": report["primary_lab_tc_mar"]["paired"]["bootstrap_seed"] == EXPECTED_BOOTSTRAP_SEED,
        "bootstrap_count_lab": report["primary_lab_tc_mar"]["paired"]["bootstrap_resamples"] == EXPECTED_BOOTSTRAP_COUNT,
        "bootstrap_seed_rgb": report["secondary_rgb_l1"]["paired"]["bootstrap_seed"] == EXPECTED_BOOTSTRAP_SEED,
        "bootstrap_count_rgb": report["secondary_rgb_l1"]["paired"]["bootstrap_resamples"] == EXPECTED_BOOTSTRAP_COUNT,
    }
    lab = metric_audit(valid_rows, report, "lab", "primary_lab_tc_mar")
    rgb = metric_audit(valid_rows, report, "rgb_l1", "secondary_rgb_l1")
    ok = (
        all(fixed_checks.values())
        and all(reference_checks.values())
        and all(item["ok"] for item in sha_checks)
        and all(provenance_checks.values())
        and lab["ok"]
        and rgb["ok"]
    )
    result = {
        "status": "SANTA_TC_MAR_AUDIT_OK" if ok else "SANTA_TC_MAR_AUDIT_FAILED",
        "frozen_artifact": str(frozen),
        "frozen_evaluator_sha256": file_sha256(frozen / "evaluator.py"),
        "fixed_contract_checks": fixed_checks,
        "evaluation_reference_checks": reference_checks,
        "input_sha256_checks": sha_checks,
        "git_provenance_checks": provenance_checks,
        "lab_recomputation": lab,
        "rgb_recomputation": rgb,
    }
    (out_dir / "audit_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_dir / result["status"]).write_text(result["status"] + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
