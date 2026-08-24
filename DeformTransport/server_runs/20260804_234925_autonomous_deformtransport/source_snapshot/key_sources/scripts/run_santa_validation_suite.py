"""Run checkpoint-free Santa cloth action, repeatability, and horizon probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "run_realwonder_trajectory_probe.py"
DEMO_DATA = REPO_ROOT / "demo_web" / "demo_data" / "santa_cloth"

RUNS = (
    {"name": "none_s1_4f", "frames": 4, "direction": "none", "strength": 1.0},
    {"name": "right_s0_4f", "frames": 4, "direction": "right", "strength": 0.0},
    {"name": "left_s1_4f", "frames": 4, "direction": "left", "strength": 1.0},
    {"name": "right_s0p5_4f", "frames": 4, "direction": "right", "strength": 0.5},
    {"name": "right_s1_4f_a", "frames": 4, "direction": "right", "strength": 1.0},
    {"name": "right_s2_4f", "frames": 4, "direction": "right", "strength": 2.0},
    {"name": "right_s1_4f_b", "frames": 4, "direction": "right", "strength": 1.0},
    {"name": "right_s1_21f", "frames": 21, "direction": "right", "strength": 1.0},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_probe(spec: dict, output_root: Path) -> tuple[Path, dict]:
    output_dir = output_root / spec["name"]
    command = [
        sys.executable,
        str(PROBE),
        "--demo-data",
        str(DEMO_DATA),
        "--frames",
        str(spec["frames"]),
        "--direction",
        spec["direction"],
        "--strength",
        str(spec["strength"]),
        "--object-id",
        "0",
        "--seed",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    print(
        f"[suite] running {spec['name']}: direction={spec['direction']} "
        f"strength={spec['strength']} frames={spec['frames']}",
        flush=True,
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"probe {spec['name']} failed with code {result.returncode}")
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    print(
        f"[suite] completed {spec['name']} in "
        f"{report['runtime_seconds']['total_before_report']:.2f}s",
        flush=True,
    )
    return output_dir, report


def _trajectory_summary(output_dir: Path, report: dict) -> dict:
    export = torch.load(output_dir / "point_trajectories.pt", map_location="cpu")
    object_state = export["objects"][0]
    final_delta_3d = object_state["points_3d"][-1] - object_state["initial_points_3d"]
    final_delta_uv = object_state["points_uv"][-1] - object_state["initial_points_uv"]
    flow_frames = [
        item
        for item in report["objects"][0]["flow_comparison"]
        if item.get("compared_points", 0) > 0
    ]
    return {
        "frames": report["frames"],
        "direction": report["force"][0]["direction"],
        "strength": report["force"][0]["strength"],
        "point_count": int(object_state["points_3d"].shape[1]),
        "all_positions_finite": bool(torch.isfinite(object_state["points_3d"]).all()),
        "binding_matches_simulator": report["all_bindings_match_simulator"],
        "min_projection_valid_ratio": min(
            report["objects"][0]["projection_valid_ratio_per_frame"]
        ),
        "final_mean_delta_3d": [float(v) for v in final_delta_3d.mean(0).tolist()],
        "final_mean_delta_uv": [float(v) for v in final_delta_uv.mean(0).tolist()],
        "final_mean_displacement_3d": float(
            torch.linalg.vector_norm(final_delta_3d, dim=1).mean()
        ),
        "final_mean_displacement_px": float(
            torch.linalg.vector_norm(final_delta_uv, dim=1).mean()
        ),
        "min_flow_compared_ratio": (
            min(item["compared_ratio"] for item in flow_frames) if flow_frames else 0.0
        ),
        "max_median_endpoint_error_px": (
            max(item["median_endpoint_error_px"] for item in flow_frames)
            if flow_frames
            else None
        ),
        "min_mean_cosine_similarity": (
            min(item["mean_cosine_similarity"] for item in flow_frames)
            if flow_frames
            else None
        ),
        "runtime_seconds": report["runtime_seconds"],
        "gpu_memory_mib": report["gpu_memory_mib"],
        "output_size_bytes": report["output_size_bytes"],
        "report_path": str(output_dir / "report.json"),
    }


def _repeatability(output_a: Path, output_b: Path) -> dict:
    export_a = torch.load(output_a / "point_trajectories.pt", map_location="cpu")
    export_b = torch.load(output_b / "point_trajectories.pt", map_location="cpu")
    points_a = export_a["objects"][0]["points_3d"]
    points_b = export_b["objects"][0]["points_3d"]
    uv_a = export_a["objects"][0]["points_uv"]
    uv_b = export_b["objects"][0]["points_uv"]
    flows_a = np.load(output_a / "flows.npy")
    flows_b = np.load(output_b / "flows.npy")
    point_error = torch.linalg.vector_norm(points_a - points_b, dim=-1).numpy()
    uv_error = torch.linalg.vector_norm(uv_a - uv_b, dim=-1).numpy()
    flow_error = np.linalg.norm(flows_a - flows_b, axis=1)
    support_a = np.linalg.norm(flows_a, axis=1) > 1e-4
    support_b = np.linalg.norm(flows_b, axis=1) > 1e-4
    shared_support = support_a & support_b
    union_support = support_a | support_b
    images_a = sorted(output_a.glob("frame_*.png"))
    images_b = sorted(output_b.glob("frame_*.png"))
    image_hashes_a = [_sha256(path) for path in images_a]
    image_hashes_b = [_sha256(path) for path in images_b]
    image_error = np.stack(
        [
            np.abs(
                np.asarray(Image.open(path_a), dtype=np.int16)
                - np.asarray(Image.open(path_b), dtype=np.int16)
            )
            for path_a, path_b in zip(images_a, images_b)
        ]
    )
    report_a = json.loads((output_a / "report.json").read_text(encoding="utf-8"))
    report_b = json.loads((output_b / "report.json").read_text(encoding="utf-8"))
    flow_metrics_a = report_a["objects"][0]["flow_comparison"]
    flow_metrics_b = report_b["objects"][0]["flow_comparison"]
    median_epe_delta = [
        abs(a["median_endpoint_error_px"] - b["median_endpoint_error_px"])
        for a, b in zip(flow_metrics_a, flow_metrics_b)
    ]
    cosine_delta = [
        abs(a["mean_cosine_similarity"] - b["mean_cosine_similarity"])
        for a, b in zip(flow_metrics_a, flow_metrics_b)
    ]
    return {
        "points_exact": bool(torch.equal(points_a, points_b)),
        "uv_exact": bool(torch.equal(uv_a, uv_b)),
        "flows_exact": bool(np.array_equal(flows_a, flows_b)),
        "images_exact": image_hashes_a == image_hashes_b,
        "max_abs_point_difference": float((points_a - points_b).abs().max()),
        "max_abs_uv_difference": float((uv_a - uv_b).abs().max()),
        "max_abs_flow_difference": float(np.max(np.abs(flows_a - flows_b))),
        "point_error_m": {
            "mean": float(point_error.mean()),
            "p99": float(np.quantile(point_error, 0.99)),
            "max": float(point_error.max()),
        },
        "uv_error_px": {
            "mean": float(uv_error.mean()),
            "p99": float(np.quantile(uv_error, 0.99)),
            "max": float(uv_error.max()),
        },
        "dense_flow_error_px": {
            "all_mean": float(flow_error.mean()),
            "all_p99": float(np.quantile(flow_error, 0.99)),
            "shared_support_mean": float(flow_error[shared_support].mean()),
            "shared_support_p99": float(
                np.quantile(flow_error[shared_support], 0.99)
            ),
            "max": float(flow_error.max()),
            "support_disagreement_ratio": float(
                np.logical_xor(support_a, support_b).sum()
                / max(int(union_support.sum()), 1)
            ),
        },
        "render_error_8bit": {
            "mae": float(image_error.mean()),
            "p99": float(np.quantile(image_error, 0.99)),
            "max": int(image_error.max()),
            "changed_value_ratio": float((image_error > 0).mean()),
        },
        "max_median_epe_metric_difference_px": max(median_epe_delta),
        "max_cosine_metric_difference": max(cosine_delta),
    }


def _practical_repeatability_checks(metrics: dict) -> dict:
    return {
        "point_error_p99_below_0p05mm": metrics["point_error_m"]["p99"] < 5e-5,
        "uv_error_p99_below_0p01px": metrics["uv_error_px"]["p99"] < 0.01,
        "uv_error_max_below_0p05px": metrics["uv_error_px"]["max"] < 0.05,
        "shared_flow_error_p99_below_0p01px": (
            metrics["dense_flow_error_px"]["shared_support_p99"] < 0.01
        ),
        "flow_support_disagreement_below_0p01pct": (
            metrics["dense_flow_error_px"]["support_disagreement_ratio"] < 1e-4
        ),
        "render_mae_below_0p01_level": metrics["render_error_8bit"]["mae"] < 0.01,
        "median_epe_metric_difference_below_0p001px": (
            metrics["max_median_epe_metric_difference_px"] < 0.001
        ),
        "cosine_metric_difference_below_0p001": (
            metrics["max_cosine_metric_difference"] < 0.001
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "exhaustive_validation" / "santa_action_suite",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Reuse existing run directories and regenerate only suite_report.json.",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    outputs = {}
    reports = {}
    summaries = {}
    for spec in RUNS:
        if args.analyze_only:
            output_dir = output_root / spec["name"]
            report_path = output_dir / "report.json"
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"analysis-only run is missing {report_path}"
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            output_dir, report = _run_probe(spec, output_root)
        outputs[spec["name"]] = output_dir
        reports[spec["name"]] = report
        summaries[spec["name"]] = _trajectory_summary(output_dir, report)

    repeatability = _repeatability(
        outputs["right_s1_4f_a"], outputs["right_s1_4f_b"]
    )
    zero_force_equivalence = _repeatability(
        outputs["none_s1_4f"], outputs["right_s0_4f"]
    )
    none_x = abs(summaries["none_s1_4f"]["final_mean_delta_uv"][0])
    left_x = summaries["left_s1_4f"]["final_mean_delta_uv"][0]
    right_half_x = abs(summaries["right_s0p5_4f"]["final_mean_delta_uv"][0])
    right_one_x = abs(summaries["right_s1_4f_a"]["final_mean_delta_uv"][0])
    right_two_x = abs(summaries["right_s2_4f"]["final_mean_delta_uv"][0])
    right_signed_x = summaries["right_s1_4f_a"]["final_mean_delta_uv"][0]
    long_summary = summaries["right_s1_21f"]

    causal_and_horizon_checks = {
        "left_right_have_opposite_horizontal_response": left_x * right_signed_x < 0,
        "none_has_less_horizontal_response_than_right_s1": none_x < right_one_x,
        "right_strength_response_is_monotonic": (
            right_half_x < right_one_x < right_two_x
        ),
        "long_positions_finite": long_summary["all_positions_finite"],
        "long_binding_matches": long_summary["binding_matches_simulator"],
        "long_projection_valid_at_least_99pct": (
            long_summary["min_projection_valid_ratio"] >= 0.99
        ),
        "long_flow_median_epe_below_1px": (
            long_summary["max_median_endpoint_error_px"] is not None
            and long_summary["max_median_endpoint_error_px"] < 1.0
        ),
        "long_flow_cosine_above_0p9": (
            long_summary["min_mean_cosine_similarity"] is not None
            and long_summary["min_mean_cosine_similarity"] > 0.9
        ),
    }
    strict_repeatability_checks = {
        "repeat_points_exact": repeatability["points_exact"],
        "repeat_uv_exact": repeatability["uv_exact"],
        "repeat_flows_exact": repeatability["flows_exact"],
        "repeat_images_exact": repeatability["images_exact"],
    }
    practical_repeatability_checks = _practical_repeatability_checks(repeatability)
    zero_force_equivalence_checks = _practical_repeatability_checks(
        zero_force_equivalence
    )
    scientific_checks = {
        **causal_and_horizon_checks,
        **{
            f"repeat_{name}": passed
            for name, passed in practical_repeatability_checks.items()
        },
        **{
            f"zero_force_{name}": passed
            for name, passed in zero_force_equivalence_checks.items()
        },
    }
    suite_report = {
        "case": "santa_cloth",
        "uses_bundled_realwonder_data": True,
        "runs": summaries,
        "repeatability": repeatability,
        "zero_force_equivalence": zero_force_equivalence,
        "causal_and_horizon_checks": causal_and_horizon_checks,
        "strict_repeatability_checks": strict_repeatability_checks,
        "strict_repeatability_passed": all(strict_repeatability_checks.values()),
        "practical_repeatability_checks": practical_repeatability_checks,
        "practical_repeatability_passed": all(
            practical_repeatability_checks.values()
        ),
        "zero_force_equivalence_checks": zero_force_equivalence_checks,
        "zero_force_equivalence_passed": all(
            zero_force_equivalence_checks.values()
        ),
        "scientific_checks": scientific_checks,
        "all_scientific_checks_passed": all(scientific_checks.values()),
    }
    report_path = output_root / "suite_report.json"
    report_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")
    print(json.dumps(suite_report, indent=2))
    print(f"[suite] report: {report_path}")


if __name__ == "__main__":
    main()
