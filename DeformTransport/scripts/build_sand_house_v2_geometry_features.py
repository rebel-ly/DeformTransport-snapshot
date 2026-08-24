from pathlib import Path
import json
import os

import torch


RUN = Path(os.environ["RUN"])

GEOM_PATH = (
    RUN
    / "transport_42f"
    / "sand_house_transport_geometry_42f.pt"
)

TRAJ_PATH = (
    RUN
    / "raw_sim"
    / "point_trajectories_42f.pt"
)

OUT_DIR = RUN / "v2_prep"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda")

H = 60
W = 104
HW = H * W

EPS = 1e-8


def scatter_stats(index, value):
    """
    index: [M] flattened latent-cell index
    value: [M] float
    returns count, mean, std: [HW]
    """
    count = torch.bincount(
        index,
        minlength=HW,
    ).float()

    total = torch.zeros(
        HW,
        device=DEVICE,
        dtype=torch.float32,
    )

    total_sq = torch.zeros_like(total)

    total.scatter_add_(0, index, value)
    total_sq.scatter_add_(0, index, value * value)

    denom = count.clamp_min(1.0)

    mean = total / denom

    var = (
        total_sq / denom
        - mean * mean
    ).clamp_min(0.0)

    std = torch.sqrt(var)

    return count, mean, std


def reshape_map(x):
    return x.reshape(H, W).cpu()


geom = torch.load(
    GEOM_PATH,
    map_location="cpu",
    weights_only=False,
)

traj = torch.load(
    TRAJ_PATH,
    map_location="cpu",
    weights_only=False,
)

obj = traj["objects"][0]

source_xy = (
    geom[
        "source_points_2d_latent_continuous"
    ]
    .float()
)

target_xy = (
    geom[
        "points_2d_latent_continuous"
    ]
    .float()
)

source_valid = (
    geom["source_valid"]
    .bool()
)

target_valid = (
    geom["target_valid_self"]
    .bool()
)

visible = (
    geom["raster_visible_self"]
    .bool()
)

source_depth = (
    obj["initial_depth"]
    .float()
)

target_depth = (
    obj["depth"]
    .float()
)

points_3d = (
    obj["points_3d"]
    .float()
)

frame_ids = (
    geom["frame_ids"]
    .long()
)

T, N, _ = target_xy.shape

assert T == 42
assert N == source_xy.shape[0]
assert H == geom["latent_height"]
assert W == geom["latent_width"]


print("===== V2 GEOMETRY FEATURE BUILD =====")
print("T =", T)
print("N =", N)
print("latent =", H, W)


# ------------------------------------------------------------
# Persistent per-point visibility stability
# ------------------------------------------------------------

visibility_persistence = (
    visible.float().mean(dim=0)
)


# ------------------------------------------------------------
# Source latent-cell occupancy
# ------------------------------------------------------------

src = source_xy.to(DEVICE)

src_x = torch.floor(src[:, 0]).long()
src_y = torch.floor(src[:, 1]).long()

src_in_bounds = (
    (src_x >= 0)
    & (src_x < W)
    & (src_y >= 0)
    & (src_y < H)
)

src_ok = (
    source_valid.to(DEVICE)
    & src_in_bounds
)

src_index = (
    src_y * W + src_x
)

source_cell_count = torch.bincount(
    src_index[src_ok],
    minlength=HW,
).float()


feature_names = [
    "target_support_count",
    "eligible_count",
    "eligible_fraction",
    "source_occupancy_mean",
    "motion_latent_mean",
    "motion_latent_std",
    "depth_change_abs_mean",
    "depth_change_std",
    "motion_3d_mean",
    "speed_3d_mean",
    "visibility_persistence_mean",
    "crop_margin_mean",
    "confidence_prior_v0",
]

all_features = []

summary_rows = []


for t in range(T):

    tgt = target_xy[t].to(DEVICE)

    tx = torch.floor(tgt[:, 0]).long()
    ty = torch.floor(tgt[:, 1]).long()

    tgt_in_bounds = (
        (tx >= 0)
        & (tx < W)
        & (ty >= 0)
        & (ty < H)
    )

    tv = (
        target_valid[t].to(DEVICE)
        & tgt_in_bounds
    )

    eligible = (
        tv
        & source_valid.to(DEVICE)
        & src_in_bounds
    )

    tgt_index = ty * W + tx


    # --------------------------------------------------------
    # All visible target support
    # --------------------------------------------------------

    target_count = torch.bincount(
        tgt_index[tv],
        minlength=HW,
    ).float()


    # --------------------------------------------------------
    # Transport-eligible support
    # --------------------------------------------------------

    idx = tgt_index[eligible]

    eligible_count = torch.bincount(
        idx,
        minlength=HW,
    ).float()

    eligible_fraction = (
        eligible_count
        / target_count.clamp_min(1.0)
    )


    # --------------------------------------------------------
    # source-cell ambiguity
    # --------------------------------------------------------

    src_occ_per_point = (
        source_cell_count[
            src_index[eligible]
        ]
    )

    _, source_occ_mean, _ = scatter_stats(
        idx,
        src_occ_per_point,
    )


    # --------------------------------------------------------
    # 2D transport displacement
    # --------------------------------------------------------

    delta_xy = (
        tgt[eligible]
        - src[eligible]
    )

    dx = delta_xy[:, 0]
    dy = delta_xy[:, 1]

    motion_mag = torch.linalg.norm(
        delta_xy,
        dim=-1,
    )

    _, motion_mean, _ = scatter_stats(
        idx,
        motion_mag,
    )

    _, _, dx_std = scatter_stats(
        idx,
        dx,
    )

    _, _, dy_std = scatter_stats(
        idx,
        dy,
    )

    motion_std = torch.sqrt(
        dx_std * dx_std
        + dy_std * dy_std
    )


    # --------------------------------------------------------
    # Depth change / depth conflict
    # --------------------------------------------------------

    dz = (
        target_depth[t].to(DEVICE)[eligible]
        - source_depth.to(DEVICE)[eligible]
    )

    _, depth_abs_mean, _ = scatter_stats(
        idx,
        torch.abs(dz),
    )

    _, _, depth_std = scatter_stats(
        idx,
        dz,
    )


    # --------------------------------------------------------
    # 3D displacement from source
    # --------------------------------------------------------

    p0 = points_3d[0].to(DEVICE)
    pt = points_3d[t].to(DEVICE)

    motion3 = torch.linalg.norm(
        pt[eligible] - p0[eligible],
        dim=-1,
    )

    _, motion3_mean, _ = scatter_stats(
        idx,
        motion3,
    )


    # --------------------------------------------------------
    # Temporal speed between Wan slots
    # --------------------------------------------------------

    if t == 0:
        speed3 = torch.zeros(
            int(eligible.sum()),
            device=DEVICE,
        )
    else:
        prev = points_3d[t - 1].to(DEVICE)

        speed3 = torch.linalg.norm(
            pt[eligible]
            - prev[eligible],
            dim=-1,
        )

    _, speed3_mean, _ = scatter_stats(
        idx,
        speed3,
    )


    # --------------------------------------------------------
    # Visibility persistence
    # --------------------------------------------------------

    vp = (
        visibility_persistence
        .to(DEVICE)[eligible]
    )

    _, persistence_mean, _ = scatter_stats(
        idx,
        vp,
    )


    # --------------------------------------------------------
    # Crop / latent-boundary margin
    # --------------------------------------------------------

    valid_xy = tgt[eligible]

    margin = torch.minimum(
        torch.minimum(
            valid_xy[:, 0],
            (W - 1) - valid_xy[:, 0],
        ),
        torch.minimum(
            valid_xy[:, 1],
            (H - 1) - valid_xy[:, 1],
        ),
    ).clamp_min(0.0)

    _, crop_margin_mean, _ = scatter_stats(
        idx,
        margin,
    )


    # --------------------------------------------------------
    # Conservative analytic confidence prior.
    #
    # IMPORTANT:
    # Diagnostic initialization only.
    # Not treated as the final v2 gate.
    # --------------------------------------------------------

    support_cells = eligible_count > 0

    if support_cells.any():

        disp_scale = torch.quantile(
            motion_std[support_cells],
            0.75,
        ).clamp_min(1e-4)

        depth_scale = torch.quantile(
            depth_std[support_cells],
            0.75,
        ).clamp_min(1e-4)

        coherence = torch.exp(
            -motion_std / disp_scale
        )

        depth_coherence = torch.exp(
            -depth_std / depth_scale
        )

        persistence_gate = torch.sqrt(
            persistence_mean.clamp(0, 1)
        )

        confidence = (
            eligible_fraction
            * coherence
            * depth_coherence
            * persistence_gate
        )

        confidence[~support_cells] = 0.0

    else:
        confidence = torch.zeros(
            HW,
            device=DEVICE,
        )


    feature_stack = torch.stack(
        [
            target_count,
            eligible_count,
            eligible_fraction,
            source_occ_mean,
            motion_mean,
            motion_std,
            depth_abs_mean,
            depth_std,
            motion3_mean,
            speed3_mean,
            persistence_mean,
            crop_margin_mean,
            confidence,
        ],
        dim=0,
    )

    all_features.append(
        feature_stack.reshape(
            len(feature_names),
            H,
            W,
        ).cpu()
    )


    active = eligible_count > 0

    row = {
        "slot": int(t),
        "pixel_frame": int(frame_ids[t]),
        "target_points": int(tv.sum()),
        "eligible_points": int(eligible.sum()),
        "active_latent_cells": int(active.sum()),
        "mean_eligible_fraction":
            float(
                eligible_fraction[active].mean()
            ) if active.any() else 0.0,
        "mean_motion_std":
            float(
                motion_std[active].mean()
            ) if active.any() else 0.0,
        "mean_confidence":
            float(
                confidence[active].mean()
            ) if active.any() else 0.0,
    }

    summary_rows.append(row)

    if t in {0, 5, 7, 10, 20, 30, 40, 41}:
        print(
            f"t={t:02d}",
            f"frame={int(frame_ids[t]):03d}",
            f"target={row['target_points']}",
            f"eligible={row['eligible_points']}",
            f"cells={row['active_latent_cells']}",
            f"support={row['mean_eligible_fraction']:.4f}",
            f"motion_std={row['mean_motion_std']:.4f}",
            f"conf={row['mean_confidence']:.4f}",
        )


features = torch.stack(
    all_features,
    dim=0,
)

assert features.shape == (
    42,
    len(feature_names),
    60,
    104,
)


state = {
    "format_version": 1,
    "case": "sand_house",

    "frame_ids": frame_ids,

    "feature_names": feature_names,

    "features": features,

    "latent_height": H,
    "latent_width": W,

    "source_point_count": N,

    "robot_occlusion_available": False,

    "visibility_contract":
        "projection + crop + front-surface raster",

    "confidence_prior_v0_note":
        "diagnostic initialization only; "
        "not final adaptive gate",
}

output_path = (
    OUT_DIR
    / "sand_house_v2_geometry_features.pt"
)

torch.save(
    state,
    output_path,
)


summary = {
    "feature_names": feature_names,
    "shape": list(features.shape),

    "robot_occlusion_available": False,

    "frames": summary_rows,
}

summary_path = (
    OUT_DIR
    / "sand_house_v2_geometry_features.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print("\n===== RESULT =====")
print("shape =", tuple(features.shape))
print("features =", feature_names)

for i, name in enumerate(feature_names):
    x = features[:, i]

    print(
        name,
        "min=",
        float(x.min()),
        "mean=",
        float(x.mean()),
        "max=",
        float(x.max()),
    )

print("saved =", output_path)
print(
    "\nSANDHOUSE_V2_GEOMETRY_FEATURES_OK"
)
