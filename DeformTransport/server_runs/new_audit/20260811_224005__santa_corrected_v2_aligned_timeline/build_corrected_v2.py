import os
import ast
import json
import hashlib
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Frozen inputs
# ============================================================

ALDIR = Path(
    "/workspace/DeformTransport/server_runs/"
    "20260804_234925_autonomous_deformtransport/"
    "prepared_inputs/"
    "official_santa_81f_aligned_contract_20260806_192643"
)

ALIGNED_PATH = (
    ALDIR / "outputs/aligned_transport_ready.pt"
)

VIS_PATH = (
    ALDIR / "outputs/aligned_visibility_contract.pt"
)

BUILDER_REPORT_PATH = (
    ALDIR / "outputs/report.json"
)

LEGACY_EXPORTER = Path(
    "/workspace/DeformTransport/scripts/"
    "export_santa_material_tracks_to_wan_move_visibility_corrected.py"
)

FAILED_V1_BRIDGE = Path(
    "/workspace/DeformTransport/server_runs/"
    "wan_move_bridge/"
    "20260811_024330__santa_corrected_physical_visibility"
)

OUT = Path(os.environ["OUT"]).resolve()

OUT_W = 832
OUT_H = 480
VAE_STRIDE = 8
GRID_W = 104
GRID_H = 60


# ============================================================
# Utilities
# ============================================================

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def extract_function(path, function_name):
    """
    Extract exactly one function definition from the legacy
    exporter without importing/executing that module.
    """
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    node = None

    for item in tree.body:
        if (
            isinstance(
                item,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and item.name == function_name
        ):
            node = item
            break

    if node is None:
        raise RuntimeError(
            f"function not found: {function_name}"
        )

    module = ast.Module(
        body=[node],
        type_ignores=[],
    )

    ast.fix_missing_locations(module)

    ns = {
        "np": np,
    }

    exec(
        compile(
            module,
            filename=str(path),
            mode="exec",
        ),
        ns,
        ns,
    )

    return ns[function_name]


# ============================================================
# Load authoritative artifacts
# ============================================================

A = torch.load(
    str(ALIGNED_PATH),
    map_location="cpu",
)

V = torch.load(
    str(VIS_PATH),
    map_location="cpu",
)

builder_report = json.load(
    open(
        BUILDER_REPORT_PATH,
        "r",
        encoding="utf-8",
    )
)


# ============================================================
# Hard provenance gates
# ============================================================

if not builder_report.get(
    "all_checks_pass",
    False,
):
    raise RuntimeError(
        "authoritative aligned builder did not pass"
    )

if not torch.equal(
    A["frame_ids"],
    torch.arange(
        81,
        dtype=torch.long,
    ),
):
    raise RuntimeError(
        "frame IDs are not 0..80"
    )

if not torch.equal(
    A["simulation_steps"],
    torch.arange(
        0,
        801,
        10,
        dtype=torch.long,
    ),
):
    raise RuntimeError(
        "simulation steps are not 0..800"
    )

if not torch.equal(
    A["point_id"],
    V["point_id"],
):
    raise RuntimeError(
        "point ID mismatch"
    )

if not torch.equal(
    A["frame_ids"],
    V["frame_ids"],
):
    raise RuntimeError(
        "frame ID mismatch"
    )

if not torch.equal(
    A["simulation_steps"],
    V["simulation_steps"],
):
    raise RuntimeError(
        "simulation-step mismatch"
    )

if not torch.equal(
    A["projection_valid"],
    V["aligned_projection_valid"],
):
    raise RuntimeError(
        "projection-valid mismatch"
    )

if not torch.equal(
    A["source_visible"],
    V["source_visible"],
):
    raise RuntimeError(
        "source-visible mismatch"
    )

if not torch.equal(
    A["points_2d_video"][0],
    A["source_points_2d_video"],
):
    raise RuntimeError(
        "aligned frame0 != source video coordinates"
    )

if not torch.equal(
    A["points_2d_render"][0],
    A["source_points_2d_render"],
):
    raise RuntimeError(
        "aligned frame0 != source render coordinates"
    )

point_id = (
    A["point_id"]
    .cpu()
    .numpy()
    .astype(np.int64)
)

if not np.array_equal(
    point_id,
    np.arange(
        len(point_id),
        dtype=np.int64,
    ),
):
    raise RuntimeError(
        "point_id is not identical to tensor index"
    )


# ============================================================
# Authoritative aligned geometry
#
# IMPORTANT:
# DO NOT transform raw old trajectory again.
#
# This tensor is already:
#   frame0  = source / step0
#   frame1  = step10
#   ...
#   frame80 = step800
#
# Coordinate system:
#   832 x 480 Wan/RealWonder video pixels.
# ============================================================

tracks_all = (
    A["points_2d_video"]
    .cpu()
    .numpy()
    .astype(np.float32)
)

projection_valid = (
    A["projection_valid"]
    .cpu()
    .numpy()
    .astype(bool)
)

source_visible = (
    A["source_visible"]
    .cpu()
    .numpy()
    .astype(bool)
)

joint_visible = (
    V["source_and_aligned_visible"]
    .cpu()
    .numpy()
    .astype(bool)
)

if tracks_all.shape != (
    81,
    28264,
    2,
):
    raise RuntimeError(
        f"unexpected track shape {tracks_all.shape}"
    )

if joint_visible.shape != (
    81,
    28264,
):
    raise RuntimeError(
        f"unexpected visibility shape {joint_visible.shape}"
    )


# ============================================================
# Defensive finite / image-domain gates
# ============================================================

finite = np.isfinite(
    tracks_all
).all(axis=-1)

in_frame = (
    (tracks_all[..., 0] >= 0.0)
    & (tracks_all[..., 0] < OUT_W)
    & (tracks_all[..., 1] >= 0.0)
    & (tracks_all[..., 1] < OUT_H)
)

visibility_all = (
    joint_visible
    & finite
    & in_frame
)


# ============================================================
# TRUE SOURCE carrier selection
#
# Frozen rule:
#   Candidate material points must be physically source-visible.
#
#   One persistent material point per occupied
#   Wan-VAE 8x8 TRUE SOURCE cell.
#
# Tie-break:
#   nearest to cell centre;
#   then lowest material point index.
# ============================================================

source_valid = (
    source_visible
    & projection_valid[0]
    & finite[0]
    & in_frame[0]
)

source_ids = np.flatnonzero(
    source_valid
)

if len(source_ids) == 0:
    raise RuntimeError(
        "no true-source candidates"
    )

xy0 = tracks_all[
    0,
    source_ids,
]

cell_x = np.floor(
    xy0[:, 0] / VAE_STRIDE
).astype(np.int64)

cell_y = np.floor(
    xy0[:, 1] / VAE_STRIDE
).astype(np.int64)

if (
    cell_x.min() < 0
    or cell_x.max() >= GRID_W
    or cell_y.min() < 0
    or cell_y.max() >= GRID_H
):
    raise RuntimeError(
        "source cell outside latent grid"
    )

cell_id = (
    cell_y * GRID_W
    + cell_x
)

center_x = (
    cell_x + 0.5
) * VAE_STRIDE

center_y = (
    cell_y + 0.5
) * VAE_STRIDE

dist2 = (
    (xy0[:, 0] - center_x) ** 2
    +
    (xy0[:, 1] - center_y) ** 2
)

order = np.lexsort(
    (
        source_ids,
        dist2,
        cell_id,
    )
)

sorted_cells = cell_id[
    order
]

first_of_cell = np.ones(
    len(order),
    dtype=bool,
)

first_of_cell[1:] = (
    sorted_cells[1:]
    != sorted_cells[:-1]
)

selected_ids = source_ids[
    order[first_of_cell]
].astype(np.int64)

N = len(selected_ids)

expected_cell_count = len(
    np.unique(cell_id)
)

if N != expected_cell_count:
    raise RuntimeError(
        "selected count != source occupied-cell count"
    )


# ============================================================
# Raw aligned tracks + authoritative visibility
# ============================================================

tracks_raw = tracks_all[
    :,
    selected_ids,
].astype(np.float32)

visibility = visibility_all[
    :,
    selected_ids,
].astype(bool)

selected_xy0 = tracks_raw[0]

selected_cx = np.floor(
    selected_xy0[:, 0] / VAE_STRIDE
).astype(np.int64)

selected_cy = np.floor(
    selected_xy0[:, 1] / VAE_STRIDE
).astype(np.int64)

selected_cell_id = (
    selected_cy * GRID_W
    + selected_cx
).astype(np.int64)

if len(
    np.unique(
        selected_cell_id
    )
) != N:
    raise RuntimeError(
        "duplicate true-source cells"
    )

if not visibility[0].all():
    raise RuntimeError(
        "selected tracks are not all visible at true source"
    )

if not source_visible[
    selected_ids
].all():
    raise RuntimeError(
        "selected source point is not source-visible"
    )


# ============================================================
# Reuse EXACT legacy invisible-coordinate filling function.
#
# This keeps the intervention limited to fixing the
# temporal/source-cell contract.
# ============================================================

fill_nonfinite_tracks = extract_function(
    LEGACY_EXPORTER,
    "fill_nonfinite_tracks",
)

tracks_filled = fill_nonfinite_tracks(
    tracks_raw,
    visibility,
).astype(np.float32)

if not np.isfinite(
    tracks_filled
).all():
    raise RuntimeError(
        "filled tracks contain nonfinite values"
    )

# Filling MUST NOT alter visible observations.
if not np.array_equal(
    tracks_filled[visibility],
    tracks_raw[visibility],
):
    raise RuntimeError(
        "legacy filling changed visible coordinates"
    )

# Visible raw coordinates must be EXACT authoritative geometry.
authoritative_selected = tracks_all[
    :,
    selected_ids,
]

if not np.array_equal(
    tracks_raw[visibility],
    authoritative_selected[visibility],
):
    raise RuntimeError(
        "visible raw tracks differ from authoritative aligned geometry"
    )


# ============================================================
# Output tensors
# ============================================================

tracks_out = tracks_filled[None]
raw_out = tracks_raw[None]
visibility_out = visibility[None]

if tracks_out.shape != (
    1,
    81,
    N,
    2,
):
    raise RuntimeError(
        tracks_out.shape
    )

if visibility_out.shape != (
    1,
    81,
    N,
):
    raise RuntimeError(
        visibility_out.shape
    )

if tracks_out.dtype != np.float32:
    raise RuntimeError(
        tracks_out.dtype
    )

if visibility_out.dtype != np.bool_:
    raise RuntimeError(
        visibility_out.dtype
    )


# ============================================================
# Save formal artifacts
# ============================================================

RAW_PATH = (
    OUT
    / "santa_material_tracks_correct_raw_aligned.npy"
)

TRACK_PATH = (
    OUT
    / "santa_material_tracks_correct.npy"
)

VIS_OUT_PATH = (
    OUT
    / "santa_material_visibility_correct.npy"
)

IDS_PATH = (
    OUT
    / "santa_material_point_ids.npy"
)

CELL_PATH = (
    OUT
    / "santa_material_source_vae_cell_ids.npy"
)

np.save(
    RAW_PATH,
    raw_out,
)

np.save(
    TRACK_PATH,
    tracks_out,
)

np.save(
    VIS_OUT_PATH,
    visibility_out,
)

np.save(
    IDS_PATH,
    selected_ids,
)

np.save(
    CELL_PATH,
    selected_cell_id,
)


# ============================================================
# Formal contract text
# ============================================================

contract_text = """DeformTransport Evidence V4.1
Santa Corrected-V2 Aligned Bridge Contract

Timeline:
frame 0  = true source state / simulation step 0
frame 1  = simulation step 10
...
frame 80 = simulation step 800

Geometry source:
aligned_transport_ready.pt / points_2d_video

Visibility source:
aligned_visibility_contract.pt /
source_and_aligned_visible

Carrier selection:
Candidate material points must be physically source-visible.

Exactly one persistent material point per occupied
Wan-VAE 8x8 TRUE SOURCE cell.

Tie-break:
nearest point to VAE-cell center, then lowest material point index.

Invisible-coordinate filling:
Reuse the exact fill_nonfinite_tracks() implementation
from the legacy corrected exporter.

Visible coordinates are never modified by filling.

This artifact supersedes the failed audited bridge:
20260811_024330__santa_corrected_physical_visibility

The superseded artifact used old-future geometry steps10..810
with aligned visibility steps0..800 and therefore violated
the temporal/source-cell contract.
"""

(
    OUT
    / "FROZEN_CORRECTED_V2_CONTRACT.txt"
).write_text(
    contract_text,
    encoding="utf-8",
)


# ============================================================
# Structured report
# ============================================================

visible_count = int(
    visibility.sum()
)

total_count = int(
    visibility.size
)

raw_vs_filled = np.linalg.norm(
    tracks_raw
    - tracks_filled,
    axis=-1,
)

report = {
    "artifact_kind":
        "santa_corrected_v2_aligned_material_track_bridge",

    "status":
        "PASS",

    "formal_use":
        "Phase0A semantic visibility audit candidate",

    "supersedes_failed_artifact":
        str(FAILED_V1_BRIDGE),

    "timeline": {
        "frame_ids":
            list(range(81)),

        "simulation_steps":
            list(range(0, 801, 10)),

        "frame0":
            "true source / step0",

        "frame80":
            "step800",
    },

    "authoritative_inputs": {
        "aligned_transport_ready": {
            "path":
                str(ALIGNED_PATH),

            "sha256":
                sha256(ALIGNED_PATH),
        },

        "aligned_visibility_contract": {
            "path":
                str(VIS_PATH),

            "sha256":
                sha256(VIS_PATH),
        },

        "aligned_builder_report": {
            "path":
                str(BUILDER_REPORT_PATH),

            "sha256":
                sha256(
                    BUILDER_REPORT_PATH
                ),

            "all_checks_pass":
                bool(
                    builder_report[
                        "all_checks_pass"
                    ]
                ),

            "check_count":
                len(
                    builder_report[
                        "checks"
                    ]
                ),
        },

        "legacy_fill_operator": {
            "path":
                str(LEGACY_EXPORTER),

            "sha256":
                sha256(
                    LEGACY_EXPORTER
                ),

            "function":
                "fill_nonfinite_tracks",
        },
    },

    "coordinate_system": {
        "tracks":
            "realwonder_resized_832_crop_480_uv_xy",

        "input_resolution":
            [480, 832],

        "vae_stride":
            8,

        "latent_grid":
            [60, 104],
    },

    "sampling": {
        "candidate_points":
            int(len(source_ids)),

        "occupied_true_source_cells":
            int(
                expected_cell_count
            ),

        "selected_tracks":
            int(N),

        "one_per_true_source_cell":
            bool(
                len(
                    np.unique(
                        selected_cell_id
                    )
                )
                == N
                == expected_cell_count
            ),

        "all_selected_source_visible":
            bool(
                source_visible[
                    selected_ids
                ].all()
            ),

        "all_selected_frame0_visible":
            bool(
                visibility[
                    0
                ].all()
            ),
    },

    "tracks": {
        "raw_shape":
            list(raw_out.shape),

        "filled_shape":
            list(tracks_out.shape),

        "dtype":
            str(
                tracks_out.dtype
            ),

        "frame0_exact_source":
            bool(
                np.array_equal(
                    tracks_raw[0],
                    A[
                        "source_points_2d_video"
                    ]
                    .cpu()
                    .numpy()[
                        selected_ids
                    ]
                    .astype(np.float32)
                )
            ),

        "visible_raw_exact_authoritative":
            bool(
                np.array_equal(
                    tracks_raw[
                        visibility
                    ],
                    authoritative_selected[
                        visibility
                    ],
                )
            ),

        "visible_filling_unchanged":
            bool(
                np.array_equal(
                    tracks_filled[
                        visibility
                    ],
                    tracks_raw[
                        visibility
                    ],
                )
            ),

        "invisible_slots_changed_by_fill":
            int(
                (
                    raw_vs_filled[
                        ~visibility
                    ]
                    > 0
                ).sum()
            ),
    },

    "visibility": {
        "shape":
            list(
                visibility_out.shape
            ),

        "dtype":
            str(
                visibility_out.dtype
            ),

        "true_count":
            visible_count,

        "total_count":
            total_count,

        "global_fraction":
            float(
                visibility.mean()
            ),

        "frame0_visible":
            int(
                visibility[0].sum()
            ),

        "frame40_visible":
            int(
                visibility[40].sum()
            ),

        "frame80_visible":
            int(
                visibility[80].sum()
            ),
    },
}

REPORT_PATH = (
    OUT
    / "report.json"
)

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# Hash manifest
# ============================================================

hash_targets = [
    RAW_PATH,
    TRACK_PATH,
    VIS_OUT_PATH,
    IDS_PATH,
    CELL_PATH,
    OUT
    / "FROZEN_CORRECTED_V2_CONTRACT.txt",
    REPORT_PATH,
    OUT
    / "build_corrected_v2.py",
]

lines = []

for path in hash_targets:
    lines.append(
        f"{sha256(path)}  {path.name}"
    )

(
    OUT
    / "SHA256SUMS.txt"
).write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


# ============================================================
# Concise terminal summary
# ============================================================

print("===== CORRECTED-V2 BUILD PASS =====")
print("OUT =", OUT)

print(
    "N =",
    N
)

print(
    "SOURCE_CANDIDATES =",
    len(source_ids)
)

print(
    "TRUE_SOURCE_CELLS =",
    expected_cell_count
)

print(
    "ONE_PER_TRUE_SOURCE_CELL =",
    report[
        "sampling"
    ][
        "one_per_true_source_cell"
    ]
)

print(
    "FRAME0_EXACT_SOURCE =",
    report[
        "tracks"
    ][
        "frame0_exact_source"
    ]
)

print(
    "VISIBLE_RAW_EXACT_AUTHORITATIVE =",
    report[
        "tracks"
    ][
        "visible_raw_exact_authoritative"
    ]
)

print(
    "VISIBLE_FILLING_UNCHANGED =",
    report[
        "tracks"
    ][
        "visible_filling_unchanged"
    ]
)

print(
    "VISIBILITY_FRACTION =",
    report[
        "visibility"
    ][
        "global_fraction"
    ]
)

print(
    "FRAME0_VISIBLE =",
    report[
        "visibility"
    ][
        "frame0_visible"
    ]
)

print(
    "FRAME40_VISIBLE =",
    report[
        "visibility"
    ][
        "frame40_visible"
    ]
)

print(
    "FRAME80_VISIBLE =",
    report[
        "visibility"
    ][
        "frame80_visible"
    ]
)

print("STATUS = PASS")
