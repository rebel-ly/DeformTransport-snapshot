import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

repo = Path("/workspace/DeformTransport")
demo_web = repo / "demo_web"
demo_data = demo_web / "demo_data" / "tree"

official_config_path = Path(sys.argv[1])
output = Path(sys.argv[2])

# 正式源码 simulation.* 优先；
# demo_web 仅提供 InteractiveSimulator 与预处理资产。
for p in (str(repo), str(demo_web)):
    while p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, str(repo))
sys.path.insert(1, str(demo_web))

from simulation_engine import InteractiveSimulator

official = OmegaConf.to_container(
    OmegaConf.load(official_config_path),
    resolve=True,
)

overrides = dict(official)
overrides["output_folder"] = str(output / "sim_tmp")
overrides["debug"] = False
overrides["export_point_trajectories"] = True
overrides["disable_genesis_visualizer"] = True
overrides.pop("skip_force_fields", None)

(output / "sim_tmp").mkdir(parents=True, exist_ok=True)

print("========== BUILD TREE ==========")

simulator = InteractiveSimulator(
    str(demo_data),
    device="cuda",
    config_overrides=overrides,
)

# -------- runtime contract --------
assert type(simulator.case_handler).__name__ == "Tree"
assert (
    type(simulator.case_handler).__module__
    == "simulation.case_simulation.tree"
)
assert simulator.demo_case_handler is None

assert simulator.material_type == ["mpm_elastic"]
assert simulator.dt == 0.01
assert simulator.substeps == 20
assert simulator.frame_steps == 2
assert simulator.config["MPM_grid_density"] == 32
assert simulator.config["particle_size"] == 0.02
assert simulator.config["gravity"] == -1

source_point_count = int(
    simulator.fg_pcs_pt3d[0]["points"].shape[0]
)

assert source_point_count == 15774

print("official_case_handler =", type(simulator.case_handler).__module__)
print("demo_case_handler =", simulator.demo_case_handler)
print("source_point_count =", source_point_count)
print("frame_steps =", simulator.frame_steps)
print("substeps =", simulator.substeps)
print(
    "MPM_grid_density =",
    simulator.config["MPM_grid_density"],
)

# -------- source state S0 --------
print()
print("========== SOURCE S0 ==========")

initial_frame = simulator.render_preview()
initial_frame.save(output / "frame_initial.png")

flows = []
flow_source_point_indices = []

# -------- 80 future states: S2 ... S160 --------
print()
print("========== SIMULATE 80 FUTURE FRAMES ==========")

for future_index in range(80):
    updated_points = None

    for step_in_frame in range(simulator.frame_steps):
        updated_points = simulator.step(
            extract_points=(
                step_in_frame
                == simulator.frame_steps - 1
            )
        )

    assert updated_points is not None

    # 当前 renderer 中保存的是这次 flow 的 source raster。
    previous_fragments = simulator.svr._prev_fg_frags_idx

    if previous_fragments is None:
        flow_source_point_indices.append(None)
    else:
        flow_source_point_indices.append(
            previous_fragments[
                0, :, :, 0
            ].detach().cpu().long().numpy()
        )

    frame, flow, _, _ = simulator.render_and_flow(
        updated_points,
        frame_id=future_index,
    )

    frame.save(
        output / f"frame_future_{future_index:04d}.png"
    )

    flows.append(
        np.asarray(flow, dtype=np.float32)
    )

    if (
        future_index == 0
        or (future_index + 1) % 10 == 0
        or future_index == 79
    ):
        print(
            f"future={future_index:02d} "
            f"simulation_step={simulator.step_count}"
        )

# -------- save geometry products --------
trajectory_path = simulator.save_point_trajectories(
    output / "point_trajectories.pt"
)

flows_np = np.stack(flows).astype(np.float32)

np.save(
    output / "flows.npy",
    flows_np,
)

if all(
    x is not None
    for x in flow_source_point_indices
):
    np.save(
        output / "flow_source_point_indices.npy",
        np.stack(
            flow_source_point_indices
        ).astype(np.int32),
    )

# -------- trajectory validation --------
export = torch.load(
    trajectory_path,
    map_location="cpu",
)

assert len(export["objects"]) == 1

obj = export["objects"][0]

points = obj["points_3d"].float()
uv = obj["points_uv"].float()
binding = obj["binding_particle_indices"]

frame_ids = export["frame_ids"].long()
simulation_steps = export["simulation_steps"].long()

assert tuple(points.shape) == (80, 15774, 3)
assert tuple(uv.shape) == (80, 15774, 2)
assert tuple(binding.shape) == (15774, 5)

assert torch.isfinite(points).all()
assert torch.isfinite(uv).all()

expected_frame_ids = torch.arange(
    80, dtype=torch.long
)

expected_steps = torch.arange(
    2, 161, 2, dtype=torch.long
)

assert torch.equal(
    frame_ids,
    expected_frame_ids,
)

assert torch.equal(
    simulation_steps,
    expected_steps,
)

# -------- motion --------
initial_points = obj["initial_points_3d"].float()
initial_uv = obj["initial_points_uv"].float()

final_delta_3d = points[-1] - initial_points
final_delta_uv = uv[-1] - initial_uv

mean_motion_3d = float(
    torch.linalg.vector_norm(
        final_delta_3d,
        dim=-1,
    ).mean()
)

max_motion_3d = float(
    torch.linalg.vector_norm(
        final_delta_3d,
        dim=-1,
    ).max()
)

mean_motion_uv = float(
    torch.linalg.vector_norm(
        final_delta_uv,
        dim=-1,
    ).mean()
)

max_motion_uv = float(
    torch.linalg.vector_norm(
        final_delta_uv,
        dim=-1,
    ).max()
)

assert mean_motion_3d > 1e-8
assert mean_motion_uv > 1e-6

report = {
    "case": "tree",
    "experiment":
        "official_precomputed_geometry_official_tree_dynamics",

    "geometry_source":
        str(demo_data),

    "runtime_config":
        str(official_config_path),

    "case_handler":
        (
            type(simulator.case_handler).__module__
            + "."
            + type(simulator.case_handler).__name__
        ),

    "demo_case_handler_attached": False,
    "extra_ui_force": False,

    "time_contract": {
        "source_state": "S0",
        "future_states":
            "S2,S4,...,S160",
        "aligned_pixel_states": 81,
        "future_render_frames": 80,
        "latent_pixel_indices":
            list(range(0, 81, 4)),
        "latent_physical_states":
            [0] + list(range(8, 161, 8)),
    },

    "physics": {
        "dt": simulator.dt,
        "substeps": simulator.substeps,
        "frame_steps": simulator.frame_steps,
        "MPM_grid_density":
            simulator.config["MPM_grid_density"],
        "particle_size":
            simulator.config["particle_size"],
        "gravity":
            simulator.config["gravity"],
    },

    "trajectory": {
        "points_shape":
            list(points.shape),
        "uv_shape":
            list(uv.shape),
        "binding_shape":
            list(binding.shape),
        "finite": True,
        "frame_ids":
            frame_ids.tolist(),
        "simulation_steps":
            simulation_steps.tolist(),
        "final_mean_motion_3d":
            mean_motion_3d,
        "final_max_motion_3d":
            max_motion_3d,
        "final_mean_motion_uv":
            mean_motion_uv,
        "final_max_motion_uv":
            max_motion_uv,
    },

    "flow_shape":
        list(flows_np.shape),

    "flow_source_raster_saved":
        (
            output
            / "flow_source_point_indices.npy"
        ).exists(),
}

(output / "report.json").write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("========== FINAL REPORT ==========")
print(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
)

print()
print("TREE_80_FUTURE_81_ALIGNED_OK")
