from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/workspace/DeformTransport")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from deform_transport.transport_ready import validate_transport_ready


raw_path = Path(sys.argv[1]).resolve()
source_dir = Path(sys.argv[2]).resolve()
raster_path = Path(sys.argv[3]).resolve()
output_dir = Path(sys.argv[4]).resolve()

aligned_path = output_dir / "aligned_transport_ready.pt"
visibility_path = output_dir / "aligned_visibility_contract.pt"
report_path = output_dir / "report.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(
        torch.equal(
            a.detach().cpu(),
            b.detach().cpu(),
        )
    )


def visibility_from_raster(
    raster_frame: np.ndarray,
    *,
    point_count: int,
    valid: torch.Tensor,
) -> torch.Tensor:
    values = np.asarray(raster_frame).reshape(-1)
    selected = values[values >= 0]

    visible = torch.zeros(
        point_count,
        dtype=torch.bool,
    )

    if selected.size:
        ids = np.unique(
            selected.astype(
                np.int64,
                copy=False,
            )
        )

        if ids[0] < 0 or ids[-1] >= point_count:
            raise ValueError(
                "raster contains out-of-range point ID"
            )

        visible[torch.from_numpy(ids)] = True

    return (
        visible
        & valid.detach().cpu().to(torch.bool)
    )


print("========== LOAD RAW CONTRACT ==========")

raw = torch.load(
    raw_path,
    map_location="cpu",
    weights_only=False,
)

validate_transport_ready(raw)

raw_frame_count = int(
    raw["frame_ids"].numel()
)
point_count = int(
    raw["point_id"].numel()
)

assert raw_frame_count == 80
assert point_count == 15774

expected_raw_ids = torch.arange(
    80,
    dtype=torch.long,
)

expected_raw_steps = torch.arange(
    2,
    161,
    2,
    dtype=torch.long,
)

assert tensor_equal(
    raw["frame_ids"],
    expected_raw_ids,
)

assert tensor_equal(
    raw["simulation_steps"],
    expected_raw_steps,
)

print("raw_frames =", raw_frame_count)
print("points =", point_count)


print()
print("========== LOAD 81-STATE RASTER ==========")

raster = np.load(
    raster_path,
    mmap_mode="r",
    allow_pickle=False,
)

if tuple(raster.shape) != (
    81,
    512,
    512,
):
    raise ValueError(
        f"unexpected raster shape: {raster.shape}"
    )

if not np.issubdtype(
    raster.dtype,
    np.integer,
):
    raise TypeError(
        f"raster must be integer, got {raster.dtype}"
    )

print("raster_shape =", raster.shape)
print("raster_dtype =", raster.dtype)


print()
print("========== BUILD ALIGNED CONTRACT ==========")

aligned = copy.deepcopy(raw)

aligned["case_name"] = (
    str(raw["case_name"])
    + "_aligned_step0_to_160"
)

aligned["frame_ids"] = torch.arange(
    81,
    dtype=torch.long,
)

aligned["simulation_steps"] = torch.arange(
    0,
    161,
    2,
    dtype=torch.long,
)

temporal_fields = {
    "points_3d":
        "source_points_3d",
    "points_2d_render":
        "source_points_2d_render",
    "points_2d_video":
        "source_points_2d_video",
    "points_2d_latent":
        "source_points_2d_latent",
    "depth":
        "source_depth",
    "render_projection_valid":
        "source_render_projection_valid",
    "projection_valid":
        "source_valid",
}

if (
    "points_2d_latent_continuous" in raw
    and
    "source_points_2d_latent_continuous" in raw
):
    temporal_fields[
        "points_2d_latent_continuous"
    ] = (
        "source_points_2d_latent_continuous"
    )

for future_key, source_key in temporal_fields.items():
    future = raw[future_key]
    source = raw[source_key]

    if future.shape[0] != 80:
        raise ValueError(
            f"{future_key}: expected 80 future states, "
            f"got {future.shape[0]}"
        )

    if tuple(source.shape) != tuple(
        future.shape[1:]
    ):
        raise ValueError(
            f"{future_key}: source/future shape mismatch"
        )

    aligned[future_key] = torch.cat(
        [
            source.unsqueeze(0),
            future,
        ],
        dim=0,
    ).contiguous()


print()
print("========== ALIGN CAMERA ==========")

aligned_camera = {}
camera_static_max_abs = {}

for key in ("K", "R", "T"):
    value = raw["camera"][key].detach().cpu()

    if value.shape[0] != 80:
        raise ValueError(
            f"camera {key} expected 80 frames, "
            f"got {tuple(value.shape)}"
        )

    reference = value[0:1]

    difference = float(
        (
            value.to(torch.float64)
            - reference.to(torch.float64)
        )
        .abs()
        .max()
    )

    camera_static_max_abs[key] = difference

    if difference > 1e-6:
        raise RuntimeError(
            f"camera {key} is not static: "
            f"max difference={difference}"
        )

    aligned_camera[key] = torch.cat(
        [
            reference,
            value,
        ],
        dim=0,
    ).contiguous()

aligned["camera"] = aligned_camera


print()
print("========== ALIGN RGB PATHS ==========")

paths = copy.deepcopy(
    raw.get("paths", {})
)

old_coarse_paths = list(
    paths.get(
        "coarse_rgb_frames",
        [],
    )
)

if len(old_coarse_paths) != 80:
    raise ValueError(
        "raw coarse RGB count must be 80, "
        f"got {len(old_coarse_paths)}"
    )

initial_rgb_path = (
    source_dir / "frame_initial.png"
).resolve()

if not initial_rgb_path.is_file():
    raise FileNotFoundError(
        initial_rgb_path
    )

aligned_coarse_paths = [
    str(initial_rgb_path)
] + [
    str(Path(path).resolve())
    for path in old_coarse_paths
]

if len(aligned_coarse_paths) != 81:
    raise RuntimeError(
        "aligned coarse RGB count is not 81"
    )

if not all(
    Path(path).is_file()
    for path in aligned_coarse_paths
):
    raise FileNotFoundError(
        "one or more aligned RGB paths do not exist"
    )

paths["initial_rgb"] = str(
    initial_rgb_path
)

paths["coarse_rgb_frames"] = (
    aligned_coarse_paths
)

# This remains the physically correct 80-transition flow:
# S0->S2 ... S158->S160.
aligned["paths"] = paths


aligned[
    "alignment_extension_version"
] = 1

aligned["alignment_mapping"] = {
    "pixel_frame_count": 81,

    "raw_contract":
        "source=S0; future=S2,S4,...,S160",

    "aligned_contract":
        "frames=S0,S2,S4,...,S160",

    "frame_0":
        "source state S0 and frame_initial.png",

    "frames_1_to_80":
        "raw future states 0..79 = S2..S160",

    "old_future_index_by_aligned_frame":
        [-1] + list(range(80)),

    "raster_index_by_aligned_frame":
        list(range(81)),

    "flow_transition_count": 80,

    "aligned_flow_mapping":
        "S0->S2, S2->S4, ..., S158->S160",

    "source_transport_ready":
        str(raw_path),

    "source_transport_ready_sha256":
        sha256(raw_path),

    "source_raster":
        str(raster_path),

    "source_raster_sha256":
        sha256(raster_path),
}


print()
print("========== VALIDATE ALIGNED TRANSPORT ==========")

validate_transport_ready(aligned)


print()
print("========== BUILD VISIBILITY ==========")

visible_frames = []

for frame_index in range(81):
    visible_frames.append(
        visibility_from_raster(
            raster[frame_index],
            point_count=point_count,
            valid=aligned[
                "projection_valid"
            ][frame_index],
        )
    )

aligned_visible = torch.stack(
    visible_frames,
    dim=0,
)

source_and_aligned_visible = (
    aligned_visible
    & aligned["source_visible"].unsqueeze(0)
)

visible_counts = (
    aligned_visible.sum(dim=1)
)

source_and_visible_counts = (
    source_and_aligned_visible.sum(dim=1)
)


visibility_contract = {
    "format_version": 1,

    "artifact_kind":
        "aligned_raster_visibility_contract",

    "case_name":
        aligned["case_name"],

    "frame_ids":
        aligned["frame_ids"].clone(),

    "simulation_steps":
        aligned["simulation_steps"].clone(),

    "point_id":
        aligned["point_id"].clone(),

    "source_visible":
        aligned["source_visible"].clone(),

    "aligned_projection_valid":
        aligned[
            "projection_valid"
        ].clone(),

    "aligned_visible":
        aligned_visible,

    "source_and_aligned_visible":
        source_and_aligned_visible,

    "visible_point_counts":
        visible_counts,

    "source_and_visible_point_counts":
        source_and_visible_counts,

    "raster_index_by_frame":
        torch.arange(
            81,
            dtype=torch.long,
        ),

    "old_future_index_by_frame":
        torch.tensor(
            [-1] + list(range(80)),
            dtype=torch.long,
        ),

    "visibility_definition": {
        "raster":
            "unique non-negative frontmost point IDs "
            "from exact PyTorch3D point rasterization",

        "timeline":
            "S0,S2,S4,...,S160",

        "validity":
            "intersection with aligned projection/crop validity",

        "not_projection_only":
            True,

        "not_depth_threshold":
            True,
    },

    "source_files": {
        "transport_ready":
            str(raw_path),

        "transport_ready_sha256":
            sha256(raw_path),

        "raster_npy":
            str(raster_path),

        "raster_npy_sha256":
            sha256(raster_path),
    },
}


print()
print("========== STRONG CHECKS ==========")

checks = {}

checks[
    "aligned_frame_ids_0_to_80"
] = tensor_equal(
    aligned["frame_ids"],
    torch.arange(
        81,
        dtype=torch.long,
    ),
)

checks[
    "aligned_steps_0_to_160_stride2"
] = tensor_equal(
    aligned["simulation_steps"],
    torch.arange(
        0,
        161,
        2,
        dtype=torch.long,
    ),
)

for future_key, source_key in temporal_fields.items():

    checks[
        f"{future_key}_frame0_equals_source"
    ] = tensor_equal(
        aligned[future_key][0],
        raw[source_key],
    )

    checks[
        f"{future_key}_frames1_80_equal_raw0_79"
    ] = tensor_equal(
        aligned[future_key][1:],
        raw[future_key],
    )

checks[
    "visibility_frame0_equals_source_visible"
] = tensor_equal(
    aligned_visible[0],
    aligned["source_visible"],
)

checks[
    "visibility_subset_projection_valid"
] = not bool(
    (
        aligned_visible
        & ~aligned["projection_valid"]
    ).any()
)

checks[
    "source_and_visibility_subset_source_visible"
] = not bool(
    (
        source_and_aligned_visible
        & ~aligned[
            "source_visible"
        ].unsqueeze(0)
    ).any()
)

checks[
    "aligned_coarse_rgb_count_81"
] = (
    len(aligned_coarse_paths) == 81
)

checks[
    "aligned_coarse_frame0_is_initial"
] = (
    Path(
        aligned_coarse_paths[0]
    ).resolve()
    == initial_rgb_path
)

checks[
    "all_aligned_rgb_paths_exist"
] = all(
    Path(path).is_file()
    for path in aligned_coarse_paths
)

checks[
    "camera_static"
] = all(
    value <= 1e-6
    for value in (
        camera_static_max_abs.values()
    )
)

checks[
    "binding_preserved"
] = tensor_equal(
    aligned["point_particle_binding"],
    raw["point_particle_binding"],
)

checks[
    "source_visible_count_preserved"
] = (
    int(
        aligned["source_visible"].sum()
    )
    == int(
        raw["source_visible"].sum()
    )
)

if (
    "points_2d_latent_continuous"
    in aligned
):
    checks[
        "continuous_floor_recovers_discrete"
    ] = tensor_equal(
        torch.floor(
            aligned[
                "points_2d_latent_continuous"
            ]
        ).to(torch.long),
        aligned[
            "points_2d_latent"
        ],
    )

all_checks_pass = all(
    checks.values()
)

for key, value in checks.items():
    print(key, "=", value)

print(
    "all_checks_pass =",
    all_checks_pass,
)

if not all_checks_pass:
    raise RuntimeError(
        "Tree aligned contract validation failed"
    )


print()
print("========== SAVE ==========")

torch.save(
    aligned,
    aligned_path,
)

torch.save(
    visibility_contract,
    visibility_path,
)


report = {
    "case":
        aligned["case_name"],

    "stage":
        "tree_aligned_81state_transport_and_visibility",

    "timeline": {
        "aligned_states":
            "S0,S2,S4,...,S160",

        "pixel_frame_count":
            81,

        "transition_flow_count":
            80,

        "simulation_steps":
            aligned[
                "simulation_steps"
            ].tolist(),

        "latent_pixel_indices":
            list(range(0, 81, 4)),

        "latent_physical_states":
            list(range(0, 161, 8)),
    },

    "transport": {
        "frame_count":
            int(
                aligned[
                    "frame_ids"
                ].numel()
            ),

        "point_count":
            int(
                aligned[
                    "point_id"
                ].numel()
            ),

        "points_3d_shape":
            list(
                aligned[
                    "points_3d"
                ].shape
            ),

        "binding_shape":
            list(
                aligned[
                    "point_particle_binding"
                ].shape
            ),

        "source_visible_count":
            int(
                aligned[
                    "source_visible"
                ].sum()
            ),

        "coarse_rgb_count":
            len(
                aligned_coarse_paths
            ),
    },

    "visibility": {
        "shape":
            list(
                aligned_visible.shape
            ),

        "minimum":
            int(
                visible_counts.min()
            ),

        "mean":
            float(
                visible_counts
                .to(torch.float32)
                .mean()
            ),

        "maximum":
            int(
                visible_counts.max()
            ),

        "first":
            int(
                visible_counts[0]
            ),

        "middle":
            int(
                visible_counts[40]
            ),

        "final":
            int(
                visible_counts[-1]
            ),

        "source_and_visible_first":
            int(
                source_and_visible_counts[0]
            ),

        "source_and_visible_final":
            int(
                source_and_visible_counts[-1]
            ),
    },

    "camera_static_max_abs":
        camera_static_max_abs,

    "checks":
        checks,

    "all_checks_pass":
        all_checks_pass,
}


report["outputs"] = {
    "aligned_transport_ready": {
        "path":
            str(aligned_path),

        "sha256":
            sha256(aligned_path),

        "bytes":
            aligned_path.stat().st_size,
    },

    "aligned_visibility_contract": {
        "path":
            str(visibility_path),

        "sha256":
            sha256(visibility_path),

        "bytes":
            visibility_path.stat().st_size,
    },
}

report_path.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print(json.dumps(
    report,
    ensure_ascii=False,
    indent=2,
))

print()
print("TREE_ALIGNED_81STATE_CONTRACT_OK")
