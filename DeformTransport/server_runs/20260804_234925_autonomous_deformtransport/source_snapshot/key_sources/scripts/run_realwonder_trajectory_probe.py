"""Run a short, checkpoint-free RealWonder simulation and export trajectories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_WEB = REPO_ROOT / "demo_web"
for import_root in (REPO_ROOT, DEMO_WEB):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from case_handlers.base import get_demo_case_handler  # noqa: E402
import case_handlers  # noqa: E402,F401  Register official handlers.
from simulation_engine import InteractiveSimulator  # noqa: E402


def _gpu_used_memory_mib() -> int | None:
    """Return whole-device memory use; this includes non-probe processes."""

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


def _motion_statistics(object_state: dict) -> list[dict]:
    points_3d = object_state["points_3d"]
    points_uv = object_state["points_uv"]
    previous_3d = torch.cat(
        [object_state["initial_points_3d"].unsqueeze(0), points_3d[:-1]], dim=0
    )
    previous_uv = torch.cat(
        [object_state["initial_points_uv"].unsqueeze(0), points_uv[:-1]], dim=0
    )
    displacement_3d = torch.linalg.vector_norm(points_3d - previous_3d, dim=-1)
    displacement_uv = torch.linalg.vector_norm(points_uv - previous_uv, dim=-1)
    signed_displacement_3d = points_3d - previous_3d
    signed_displacement_uv = points_uv - previous_uv
    return [
        {
            "frame": frame_index,
            "mean_delta_3d": [
                float(value)
                for value in signed_displacement_3d[frame_index].mean(dim=0).tolist()
            ],
            "mean_delta_uv": [
                float(value)
                for value in signed_displacement_uv[frame_index].mean(dim=0).tolist()
            ],
            "mean_displacement_3d": float(displacement_3d[frame_index].mean()),
            "max_displacement_3d": float(displacement_3d[frame_index].max()),
            "mean_displacement_px": float(displacement_uv[frame_index].mean()),
            "max_displacement_px": float(displacement_uv[frame_index].max()),
        }
        for frame_index in range(points_3d.shape[0])
    ]


def _directory_size_bytes(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _compare_point_displacements(
    object_state: dict,
    flows: np.ndarray,
    *,
    flow_source_point_indices: np.ndarray | None = None,
    object_point_offset: int = 0,
) -> list[dict]:
    current_uv = object_state["points_uv"]
    current_valid = object_state["projection_valid"]
    initial_uv = object_state["initial_points_uv"]
    initial_valid = object_state["initial_projection_valid"]
    frame_metrics = []

    if len(flows) != current_uv.shape[0]:
        raise ValueError(
            f"flow count {len(flows)} does not match trajectory frames {current_uv.shape[0]}"
        )

    for frame_index in range(current_uv.shape[0]):
        if frame_index == 0:
            previous_uv = initial_uv
            previous_valid = initial_valid
        else:
            previous_uv = current_uv[frame_index - 1]
            previous_valid = current_valid[frame_index - 1]

        expected = current_uv[frame_index] - previous_uv
        x = previous_uv[:, 0].round().to(torch.long).clamp(0, 511)
        y = previous_uv[:, 1].round().to(torch.long).clamp(0, 511)
        flow_tensor = torch.from_numpy(flows[frame_index])
        sampled = flow_tensor[:, y, x].T
        sampled_nonzero = torch.linalg.vector_norm(sampled, dim=1) > 1e-4
        valid = previous_valid & current_valid[frame_index] & sampled_nonzero
        visibility_filter_applied = flow_source_point_indices is not None
        if visibility_filter_applied:
            source_indices = torch.from_numpy(flow_source_point_indices[frame_index])
            sampled_source = source_indices[y, x]
            object_point_end = object_point_offset + previous_uv.shape[0]
            valid = valid & (sampled_source >= object_point_offset) & (
                sampled_source < object_point_end
            )

        if valid.any():
            error = torch.linalg.vector_norm(sampled[valid] - expected[valid], dim=1)
            expected_magnitude = torch.linalg.vector_norm(expected[valid], dim=1)
            sampled_magnitude = torch.linalg.vector_norm(sampled[valid], dim=1)
            cosine = torch.nn.functional.cosine_similarity(
                sampled[valid], expected[valid], dim=1, eps=1e-6
            )
            frame_metrics.append(
                {
                    "frame": frame_index,
                    "visibility_filter_applied": visibility_filter_applied,
                    "compared_points": int(valid.sum()),
                    "compared_ratio": float(valid.float().mean()),
                    "median_endpoint_error_px": float(error.median()),
                    "mean_endpoint_error_px": float(error.mean()),
                    "median_expected_motion_px": float(expected_magnitude.median()),
                    "median_sampled_flow_px": float(sampled_magnitude.median()),
                    "mean_cosine_similarity": float(cosine.mean()),
                }
            )
        else:
            frame_metrics.append(
                {
                    "frame": frame_index,
                    "visibility_filter_applied": visibility_filter_applied,
                    "compared_points": 0,
                    "compared_ratio": 0.0,
                }
            )
    return frame_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-data",
        type=Path,
        default=REPO_ROOT / "demo_web" / "demo_data" / "santa_cloth",
    )
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--direction", choices=("left", "none", "right"), default="right")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--object-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "stage1_dynamic" / "santa_cloth",
    )
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.strength < 0:
        parser.error("--strength must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started_at = time.perf_counter()
    gpu_memory_before_mib = _gpu_used_memory_mib()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    demo_data = args.demo_data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_started_at = time.perf_counter()
    simulator = InteractiveSimulator(
        str(demo_data),
        device="cuda",
        config_overrides={
            "debug": False,
            "seed": args.seed,
            "skip_force_fields": True,
            "export_point_trajectories": True,
        },
    )
    setup_seconds = time.perf_counter() - setup_started_at
    if not 0 <= args.object_id < len(simulator.material_type):
        raise ValueError(
            f"object id {args.object_id} is outside [0, {len(simulator.material_type) - 1}]"
        )
    handler = get_demo_case_handler(simulator.config["example_name"], simulator.config)
    forces = handler.get_force_config_from_ui(
        [{"obj_idx": args.object_id, "direction": args.direction, "strength": args.strength}]
    )
    handler.set_forces(forces)
    handler.configure_simulation(simulator)
    simulator.set_demo_case_handler(handler)

    # Establish exactly the same initial point render used as the first flow source.
    initial_frame = simulator.render_preview()
    initial_frame.save(output_dir / "frame_initial.png")
    flows = []
    flow_source_point_indices = []
    simulation_started_at = time.perf_counter()
    for frame_id in range(args.frames):
        updated_points = None
        for step_in_frame in range(simulator.frame_steps):
            updated_points = simulator.step(
                extract_points=(step_in_frame == simulator.frame_steps - 1)
            )
        previous_fragments = simulator.svr._prev_fg_frags_idx
        if previous_fragments is None:
            flow_source_point_indices.append(None)
        else:
            flow_source_point_indices.append(
                previous_fragments[0, :, :, 0]
                .detach()
                .to(device="cpu", dtype=torch.long)
                .numpy()
            )
        frame, flow, _, _ = simulator.render_and_flow(updated_points, frame_id=frame_id)
        frame.save(output_dir / f"frame_{frame_id:04d}.png")
        flows.append(flow)
    simulation_seconds = time.perf_counter() - simulation_started_at

    flows_array = np.stack(flows).astype(np.float32)
    np.save(output_dir / "flows.npy", flows_array)
    if all(indices is not None for indices in flow_source_point_indices):
        flow_source_point_indices_array = np.stack(flow_source_point_indices).astype(
            np.int32
        )
        flow_source_indices_path = output_dir / "flow_source_point_indices.npy"
        np.save(flow_source_indices_path, flow_source_point_indices_array)
    else:
        flow_source_point_indices_array = None
        flow_source_indices_path = None
    trajectory_path = simulator.save_point_trajectories(
        output_dir / "point_trajectories.pt"
    )
    export = torch.load(trajectory_path, map_location="cpu")
    object_reports = []
    object_point_offset = 0
    for object_id, object_state in enumerate(export["objects"]):
        exported_binding = object_state["binding_particle_indices"]
        simulator_binding = simulator.closest_indices.get(object_id)
        binding_expected = simulator_binding is not None
        if binding_expected:
            simulator_binding = simulator_binding.detach().to(device="cpu", dtype=torch.long)
            if type(simulator_binding) is not torch.Tensor:
                simulator_binding = simulator_binding.as_subclass(torch.Tensor)
            binding_matches_simulator = bool(
                exported_binding is not None
                and torch.equal(exported_binding, simulator_binding)
            )
        else:
            binding_matches_simulator = exported_binding is None

        simulator_initial = simulator.fg_pcs_pt3d[object_id]["points"].detach().to(
            device="cpu", dtype=torch.float32
        )
        if type(simulator_initial) is not torch.Tensor:
            simulator_initial = simulator_initial.as_subclass(torch.Tensor)
        initial_points_match_simulator = torch.equal(
            object_state["initial_points_3d"], simulator_initial
        )
        object_reports.append(
            {
                "object_id": object_id,
                "material_type": object_state["material_type"],
                "point_count": int(object_state["points_3d"].shape[1]),
                "binding_expected": binding_expected,
                "binding_shape": (
                    list(exported_binding.shape) if exported_binding is not None else None
                ),
                "binding_matches_simulator": binding_matches_simulator,
                "initial_points_match_simulator": bool(initial_points_match_simulator),
                "all_positions_finite": bool(
                    torch.isfinite(object_state["points_3d"]).all()
                ),
                "projection_valid_ratio_per_frame": [
                    float(frame.float().mean())
                    for frame in object_state["projection_valid"]
                ],
                "motion_per_frame": _motion_statistics(object_state),
                "flow_comparison": _compare_point_displacements(
                    object_state,
                    flows_array,
                    flow_source_point_indices=flow_source_point_indices_array,
                    object_point_offset=object_point_offset,
                ),
            }
        )
        object_point_offset += int(object_state["points_3d"].shape[1])

    first_object = object_reports[0]
    final_frame_mae = float(
        np.abs(np.asarray(frame, dtype=np.float32) - np.asarray(initial_frame, dtype=np.float32)).mean()
    )
    gpu_memory_after_mib = _gpu_used_memory_mib()
    report = {
        "case": demo_data.name,
        "seed": args.seed,
        "frames": args.frames,
        "physics_steps_per_frame": simulator.frame_steps,
        "target_object_id": args.object_id,
        "force": forces,
        "object_count": len(export["objects"]),
        "all_bindings_match_simulator": all(
            item["binding_matches_simulator"] for item in object_reports
        ),
        "all_initial_points_match_simulator": all(
            item["initial_points_match_simulator"] for item in object_reports
        ),
        "all_positions_finite": all(
            item["all_positions_finite"] for item in object_reports
        ),
        "frame_ids": export["frame_ids"].tolist(),
        "simulation_steps": export["simulation_steps"].tolist(),
        "objects": object_reports,
        # First-object aliases retain compatibility with the original Santa report.
        "point_count": first_object["point_count"],
        "binding_shape": first_object["binding_shape"],
        "binding_matches_simulator": first_object["binding_matches_simulator"],
        "initial_points_match_simulator": first_object[
            "initial_points_match_simulator"
        ],
        "projection_valid_ratio_per_frame": first_object[
            "projection_valid_ratio_per_frame"
        ],
        "motion_per_frame": first_object["motion_per_frame"],
        "final_frame_mae_from_initial": final_frame_mae,
        "flow_comparison": first_object["flow_comparison"],
        "runtime_seconds": {
            "setup": setup_seconds,
            "simulation_and_render": simulation_seconds,
            "total_before_report": time.perf_counter() - started_at,
        },
        "gpu_memory_mib": {
            "whole_device_before": gpu_memory_before_mib,
            "whole_device_after": gpu_memory_after_mib,
            "torch_peak_allocated": (
                float(torch.cuda.max_memory_allocated() / (1024**2))
                if torch.cuda.is_available()
                else None
            ),
            "torch_peak_reserved": (
                float(torch.cuda.max_memory_reserved() / (1024**2))
                if torch.cuda.is_available()
                else None
            ),
            "note": "whole-device values include other processes; Torch peaks exclude Genesis/GsTaichi allocations",
        },
        "trajectory_path": str(trajectory_path),
        "flows_path": str(output_dir / "flows.npy"),
        "flow_source_point_indices_path": (
            str(flow_source_indices_path) if flow_source_indices_path else None
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_size_bytes"] = _directory_size_bytes(output_dir)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
