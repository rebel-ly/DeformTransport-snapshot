from pathlib import Path
import hashlib
import json
import os

import cv2
import numpy as np
import torch


# ============================================================
# Frozen geometry contract
# ============================================================

SRC_A = Path(
    "/workspace/DeformTransport/server_runs/"
    "20260804_234925_autonomous_deformtransport/"
    "04_smoke/"
    "OFFICIAL_SANTA_81F_ASSEMBLY_MANUAL_20260805_003555/"
    "final_sim/point_trajectories.pt"
)

SRC_B = Path(
    "/workspace/DeformTransport/server_runs/"
    "20260804_234925_autonomous_deformtransport/"
    "04_smoke/"
    "OFFICIAL_SANTA_81F_CHAIN_20260805_050719/"
    "final_sim/point_trajectories.pt"
)

ALIGNED = Path(
    "/workspace/DeformTransport/server_runs/"
    "20260804_234925_autonomous_deformtransport/"
    "prepared_inputs/"
    "official_santa_81f_aligned_final_sim_20260806_234410"
)

VIS_CONTRACT = Path(
    "/workspace/DeformTransport/server_runs/"
    "20260804_234925_autonomous_deformtransport/"
    "prepared_inputs/"
    "official_santa_81f_aligned_contract_20260806_192643/"
    "outputs/aligned_visibility_contract.pt"
)

OUT = Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=True)


# RealWonder trajectory coordinate system:
# 512x512 UV.
#
# Established aligned-video transform:
#
# 512x512
# -> resize 832x832
# -> center crop y=[176,656)
# -> 480x832.
#
SOURCE_SIZE = 512
RESIZED_SIZE = 832

OUT_H = 480
OUT_W = 832

CROP_TOP = (RESIZED_SIZE - OUT_H) // 2

SCALE = RESIZED_SIZE / SOURCE_SIZE

# Wan VAE spatial stride used by trajectory conditioning.
VAE_STRIDE = 8


# ============================================================
# Helpers
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


def load_object(path):
    x = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if x.get("coordinate_system") != (
        "pytorch3d_world_and_realwonder_512_uv"
    ):
        raise RuntimeError(
            f"unexpected coordinate system: "
            f"{x.get('coordinate_system')}"
        )

    if x.get("image_size") != 512:
        raise RuntimeError(
            f"unexpected image_size: "
            f"{x.get('image_size')}"
        )

    if len(x["objects"]) != 1:
        raise RuntimeError(
            "expected exactly one Santa cloth object"
        )

    obj = x["objects"][0]

    if obj.get("material_type") != "pbd_cloth":
        raise RuntimeError(
            f"unexpected material: "
            f"{obj.get('material_type')}"
        )

    return x, obj


def transform_uv(uv):
    """
    RealWonder 512x512 UV -> aligned 480x832 pixel XY.
    """

    xy = uv.astype(np.float32).copy()

    xy[..., 0] = (
        xy[..., 0] * SCALE
    )

    xy[..., 1] = (
        xy[..., 1] * SCALE
        - CROP_TOP
    )

    return xy


def fill_nonfinite_tracks(
    tracks,
    visibility,
):
    """
    Keep exported coordinates finite even where visibility=False.

    This is important because Wan-Move may temporally average
    coordinates before applying visibility.

    Invisible coordinates are filled from nearest valid
    positions, while visibility itself remains unchanged.
    """

    result = tracks.copy()

    T, N, _ = result.shape

    for n in range(N):
        good = (
            visibility[:, n]
            & np.isfinite(
                result[:, n]
            ).all(axis=1)
        )

        ids = np.flatnonzero(good)

        if len(ids) == 0:
            # Should not occur because all selected points
            # are valid at frame 0.
            result[:, n] = 0.0
            continue

        # Forward fill.
        last = ids[0]

        for t in range(T):
            if (
                good[t]
                and np.isfinite(
                    result[t, n]
                ).all()
            ):
                last = t
            else:
                result[t, n] = result[
                    last, n
                ]

        # Backward fill in case invalid values occurred
        # before first good observation.
        first = ids[0]

        for t in range(first - 1, -1, -1):
            result[t, n] = result[
                first, n
            ]

    return result


# ============================================================
# Load both historical 81f candidates and compare geometry.
# ============================================================

meta_a, obj_a = load_object(SRC_A)
meta_b, obj_b = load_object(SRC_B)

uv_a = (
    obj_a["points_uv"]
    .float()
    .numpy()
)

uv_b = (
    obj_b["points_uv"]
    .float()
    .numpy()
)

valid_a = (
    obj_a["projection_valid"]
    .bool()
    .numpy()
)

valid_b = (
    obj_b["projection_valid"]
    .bool()
    .numpy()
)

depth_a = (
    obj_a["depth"]
    .float()
    .numpy()
)

depth_b = (
    obj_b["depth"]
    .float()
    .numpy()
)

if uv_a.shape != (
    81,
    28264,
    2,
):
    raise RuntimeError(
        f"unexpected uv shape: {uv_a.shape}"
    )

if valid_a.shape != (
    81,
    28264,
):
    raise RuntimeError(
        f"unexpected validity shape: "
        f"{valid_a.shape}"
    )


finite_both = (
    np.isfinite(uv_a)
    & np.isfinite(uv_b)
).all(axis=-1)

if finite_both.any():
    candidate_uv_maxdiff = float(
        np.abs(
            uv_a[finite_both]
            - uv_b[finite_both]
        ).max()
    )
else:
    candidate_uv_maxdiff = float("nan")

candidate_valid_equal = bool(
    np.array_equal(
        valid_a,
        valid_b,
    )
)

finite_depth = (
    np.isfinite(depth_a)
    & np.isfinite(depth_b)
)

if finite_depth.any():
    candidate_depth_maxdiff = float(
        np.abs(
            depth_a[finite_depth]
            - depth_b[finite_depth]
        ).max()
    )
else:
    candidate_depth_maxdiff = float("nan")


print(
    "candidate_uv_maxdiff =",
    candidate_uv_maxdiff,
)

print(
    "candidate_valid_equal =",
    candidate_valid_equal,
)

print(
    "candidate_depth_maxdiff =",
    candidate_depth_maxdiff,
)


# Use the later CHAIN artifact as canonical.
# The comparison diagnostics above preserve provenance.
source_path = SRC_B

uv = uv_b
projection_valid = valid_b
depth = depth_b


# ============================================================
# Transform to Wan-Move input-image pixel coordinates.
# ============================================================

tracks_all = transform_uv(uv)

finite = np.isfinite(
    tracks_all
).all(axis=-1)

in_frame = (
    (tracks_all[..., 0] >= 0.0)
    & (tracks_all[..., 0] < OUT_W)
    & (tracks_all[..., 1] >= 0.0)
    & (tracks_all[..., 1] < OUT_H)
)

# ============================================================
# Corrected physical visibility contract.
#
# Old bridge incorrectly used projection_valid as visibility.
# The corrected bridge requires:
#   - source point physically visible at t=0 for carrier selection
#   - source_and_aligned_visible at future frames
#   - finite and in-frame coordinates as defensive validity gates
# ============================================================

visibility_contract = torch.load(
    VIS_CONTRACT,
    map_location="cpu",
    weights_only=False,
)

source_visible = (
    visibility_contract["source_visible"]
    .cpu()
    .numpy()
    .astype(bool)
)

aligned_projection_valid = (
    visibility_contract["aligned_projection_valid"]
    .cpu()
    .numpy()
    .astype(bool)
)

source_and_aligned_visible = (
    visibility_contract["source_and_aligned_visible"]
    .cpu()
    .numpy()
    .astype(bool)
)

if source_visible.shape != (28264,):
    raise RuntimeError(
        f"unexpected source_visible shape: {source_visible.shape}"
    )

if aligned_projection_valid.shape != (81, 28264):
    raise RuntimeError(
        f"unexpected aligned projection shape: "
        f"{aligned_projection_valid.shape}"
    )

if source_and_aligned_visible.shape != (81, 28264):
    raise RuntimeError(
        f"unexpected physical visibility shape: "
        f"{source_and_aligned_visible.shape}"
    )

# Provenance gate: geometry used by bridge must match
# geometry used by the authoritative aligned contract.
if not np.array_equal(
    projection_valid,
    aligned_projection_valid,
):
    raise RuntimeError(
        "projection-valid mismatch between trajectory source "
        "and authoritative visibility contract"
    )

visibility_all = (
    source_and_aligned_visible
    & finite
    & in_frame
)


# ============================================================
# Architecture-defined material-point sampling:
#
# select exactly one persistent material point from each
# occupied 8x8 Wan-Move VAE cell in frame 0.
#
# Tie-breaker:
# nearest point to cell center, then lowest point index.
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
        "no valid initial material points"
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

grid_w = OUT_W // VAE_STRIDE
grid_h = OUT_H // VAE_STRIDE

if grid_w != 104 or grid_h != 60:
    raise RuntimeError(
        "unexpected Wan-Move latent grid"
    )

cell_id = (
    cell_y * grid_w
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
    + (xy0[:, 1] - center_y) ** 2
)


# Sort:
# cell ID
# distance to cell center
# original material point index
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

selected_local = order[
    first_of_cell
]

selected_ids = source_ids[
    selected_local
]

selected_ids = selected_ids.astype(
    np.int64
)

N = len(selected_ids)

if N <= 0:
    raise RuntimeError(
        "no selected material trajectories"
    )


tracks = tracks_all[
    :,
    selected_ids,
].astype(np.float32)

visibility = visibility_all[
    :,
    selected_ids,
].astype(bool)


# Keep coordinates finite even when invisible.
tracks = fill_nonfinite_tracks(
    tracks,
    visibility,
)


# ============================================================
# Exact Wan-Move contract.
# ============================================================

tracks_out = tracks[None]

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

if not np.isfinite(
    tracks_out
).all():
    raise RuntimeError(
        "tracks contain nonfinite coordinates"
    )

if not visibility_out[
    0, 0
].all():
    raise RuntimeError(
        "selected tracks must all be visible at t=0"
    )


# Verify one track per occupied VAE cell at t0.
t0 = tracks_out[0, 0]

t0_cx = np.floor(
    t0[:, 0] / 8
).astype(np.int64)

t0_cy = np.floor(
    t0[:, 1] / 8
).astype(np.int64)

t0_cell = (
    t0_cy * 104
    + t0_cx
)

if len(np.unique(t0_cell)) != N:
    raise RuntimeError(
        "duplicate t0 VAE cells"
    )


# ============================================================
# Save.
# ============================================================

TRACK_PATH = (
    OUT / "santa_material_tracks_correct.npy"
)

VIS_PATH = (
    OUT / "santa_material_visibility_correct.npy"
)

IDS_PATH = (
    OUT / "santa_material_point_ids.npy"
)

np.save(
    TRACK_PATH,
    tracks_out,
)

np.save(
    VIS_PATH,
    visibility_out,
)

np.save(
    IDS_PATH,
    selected_ids,
)


# ============================================================
# Diagnostics.
# ============================================================

disp = tracks[1:] - tracks[:-1]

step_mag = np.linalg.norm(
    disp,
    axis=-1,
)

valid_pairs = (
    visibility[1:]
    & visibility[:-1]
)

if valid_pairs.any():
    step_mean = float(
        step_mag[
            valid_pairs
        ].mean()
    )

    step_p95 = float(
        np.quantile(
            step_mag[
                valid_pairs
            ],
            0.95,
        )
    )
else:
    step_mean = float("nan")
    step_p95 = float("nan")


start = tracks[0]

end = tracks[-1]

total_disp = np.linalg.norm(
    end - start,
    axis=-1,
)

visibility_fraction = (
    visibility.mean(
        axis=0
    )
)

report = {
    "method":
        "DeformTransport material trajectories "
        "to Wan-Move",

    "source":
        str(source_path),

    "source_coordinate_system":
        "pytorch3d_world_and_realwonder_512_uv",

    "source_image_size":
        512,

    "alignment":
        {
            "resize":
                "512x512 -> 832x832",

            "scale":
                SCALE,

            "center_crop_top":
                CROP_TOP,

            "output_size":
                [OUT_H, OUT_W],

            "formula":
                "x=1.625*u; y=1.625*v-176",
        },

    "sampling":
        {
            "type":
                "one persistent point per occupied "
                "Wan-Move VAE 8x8 source cell",

            "manual_track_count":
                False,

            "vae_stride":
                8,

            "latent_grid":
                [60, 104],

            "selected_tracks":
                int(N),

            "initial_valid_points":
                int(source_valid.sum()),
        },

    "tracks":
        {
            "shape":
                list(tracks_out.shape),

            "dtype":
                str(tracks_out.dtype),

            "x_min":
                float(tracks[:, :, 0].min()),

            "x_max":
                float(tracks[:, :, 0].max()),

            "y_min":
                float(tracks[:, :, 1].min()),

            "y_max":
                float(tracks[:, :, 1].max()),
        },

    "visibility":
        {
            "shape":
                list(visibility_out.shape),

            "dtype":
                str(visibility_out.dtype),

            "global_fraction":
                float(
                    visibility.mean()
                ),

            "track_fraction_mean":
                float(
                    visibility_fraction.mean()
                ),

            "track_fraction_min":
                float(
                    visibility_fraction.min()
                ),
        },

    "motion":
        {
            "valid_step_mean_px":
                step_mean,

            "valid_step_p95_px":
                step_p95,

            "start_to_end_mean_px":
                float(
                    total_disp.mean()
                ),

            "start_to_end_p95_px":
                float(
                    np.quantile(
                        total_disp,
                        0.95,
                    )
                ),
        },

    "candidate_comparison":
        {
            "uv_max_abs_diff":
                candidate_uv_maxdiff,

            "projection_valid_equal":
                candidate_valid_equal,

            "depth_max_abs_diff":
                candidate_depth_maxdiff,
        },

    "wan_move_contract":
        {
            "tracks":
                "[1,81,N,2] float32 pixel XY",

            "visibility":
                "[1,81,N] bool",

            "input_resolution":
                [480, 832],

            "wan_move_additional_scale":
                "1.0 when used with aligned "
                "480x832 input image",
        },
}


# ============================================================
# Static first-frame visualization.
# ============================================================

IMAGE_PATH = (
    ALIGNED
    / "resized_input_image.png"
)

image = cv2.imread(
    str(IMAGE_PATH),
    cv2.IMREAD_COLOR,
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )

if image.shape[:2] != (
    OUT_H,
    OUT_W,
):
    raise RuntimeError(
        f"unexpected image shape "
        f"{image.shape}"
    )

overlay = image.copy()

for x, y in tracks[0]:
    cv2.circle(
        overlay,
        (
            int(round(x)),
            int(round(y)),
        ),
        1,
        (0, 255, 255),
        -1,
        lineType=cv2.LINE_AA,
    )

OVERLAY_PATH = (
    OUT / "santa_tracks_frame0_overlay.png"
)

cv2.imwrite(
    str(OVERLAY_PATH),
    overlay,
)


# ============================================================
# Dynamic visualization.
#
# Diagnostic only; does not affect model input.
# ============================================================

frame_paths = sorted(
    (ALIGNED / "frames").glob("*.png")
)

if len(frame_paths) != 81:
    frame_paths = sorted(
        (ALIGNED / "frames").glob("*")
    )

if len(frame_paths) != 81:
    raise RuntimeError(
        f"expected 81 aligned frames, "
        f"got {len(frame_paths)}"
    )

VIDEO_PATH = (
    OUT / "santa_material_tracks_overlay.mp4"
)

writer = cv2.VideoWriter(
    str(VIDEO_PATH),
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    12.0,
    (OUT_W, OUT_H),
)

if not writer.isOpened():
    raise RuntimeError(
        "failed to open visualization writer"
    )

for t, p in enumerate(frame_paths):
    frame = cv2.imread(
        str(p),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise RuntimeError(
            f"cannot read {p}"
        )

    for n in range(N):
        if not visibility[t, n]:
            continue

        x, y = tracks[t, n]

        cv2.circle(
            frame,
            (
                int(round(x)),
                int(round(y)),
            ),
            1,
            (0, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

    writer.write(frame)

writer.release()


REPORT_PATH = (
    OUT / "report.json"
)

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


SHA_PATH = OUT / "sha256.txt"

sha_files = [
    TRACK_PATH,
    VIS_PATH,
    IDS_PATH,
    REPORT_PATH,
    source_path,
    IMAGE_PATH,
]

SHA_PATH.write_text(
    "\n".join(
        f"{sha256(p)}  {p}"
        for p in sha_files
    ) + "\n",
    encoding="utf-8",
)


print()
print("===== WAN-MOVE MATERIAL TRACK EXPORT =====")

print(
    json.dumps(
        report,
        indent=2,
    )
)

print()
print("TRACKS =", TRACK_PATH)
print("VISIBILITY =", VIS_PATH)
print("POINT_IDS =", IDS_PATH)
print("FRAME0_OVERLAY =", OVERLAY_PATH)
print("VIDEO_OVERLAY =", VIDEO_PATH)

print()
print("===== SHA256 =====")
print(SHA_PATH.read_text())

print(
    "SANTA_WAN_MOVE_CORRECT_TRACK_EXPORT_OK"
)
