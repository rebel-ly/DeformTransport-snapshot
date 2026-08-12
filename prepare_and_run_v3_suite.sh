#!/usr/bin/env bash
set -euo pipefail

DT=/workspace/DeformTransport
WAN=/workspace/Wan-Move
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
WANENV=/workspace/tools/miniforge3/envs/wan-move

# Make Wan-Move importable even when Python scripts
# are executed from timestamped DeformTransport run dirs.
export PYTHONPATH="$WAN:${PYTHONPATH:-}"

cd "$DT"

STAMP=$(date +%Y%m%d_%H%M%S)

SUITE="$DT/server_runs/wan_move_method_suite/${STAMP}__v3s_v3b_v3c_v3d_v3e_correct_seed0"
ART="$SUITE/artifacts"

mkdir -p \
  "$ART" \
  "$SUITE/santa" \
  "$SUITE/tree"

echo "$SUITE" > \
"$DT/server_runs/wan_move_method_dev/current_v3_suite.txt"

echo
echo "============================================"
echo "DeformTransport V3 one-shot suite"
echo "SUITE=$SUITE"
echo "============================================"
echo


# ============================================================
# 0. Freeze protocol BEFORE any result
# ============================================================

cat > "$SUITE/PROTOCOL_FROZEN.txt" <<'EOF'
DeformTransport V3 one-shot development suite.

No result-dependent tuning is allowed between variants.

REFERENCE:
Old Correct formal Wan-Move seed0.

CANDIDATES:

V3S
Actual-Wan-grid source de-duplication.
Existing Correct conditioning track set is de-duplicated at t=0 on the
actual Wan conditioning domain 832x464 and VAE grid 104x58.
Nearest material point to latent-cell center wins; tie by material ID.
Future overlap handling remains original Wan-Move behavior.

V3B
2D deterministic hard arbitration.
For multiple material trajectories mapped to one future latent cell,
choose the point whose continuous projected position is nearest to
that latent-cell center.
Tie by persistent material ID.
No feature averaging.

V3C
3D depth-aware hard arbitration.
For multiple material trajectories mapped to one future latent cell,
choose the visible material point with minimum positive camera depth.
Tie by persistent material ID.
No feature averaging.

V3D
V3C target arbitration plus continuous source-feature sampling.
The source t=0 condition feature is bilinearly sampled from the
continuous material coordinate rather than floor-quantized source cell.

V3E
V3C depth-aware arbitration plus fixed 3x3 local source-condition
feature transport.
Each target latent cell keeps exactly one depth-nearest material
contributor.
No feature averaging.
Patch radius is frozen to one latent cell; no patch-size sweep.

FROZEN GENERATION:
Correct only
seed=0
frame_num=81
sample_steps=40
sample_shift=3.0
dtype=bf16
t5_cpu=True
offload_model=True
same image
same prompt
same checkpoint

Existing DeformTransport trajectory-internal RNG freeze must remain active.

POST-HOC EVALUATION ONLY AFTER ALL 10 GENERATIONS FINISH:

Appearance:
Frozen TC-MAR Lab using the same authoritative material evaluation tracks.

Motion:
Frozen TC-ME using the same authoritative material trajectories.

Motion safety:
Reject a candidate for a case iff lower 95% bootstrap CI of
(candidate TC-ME - OldCorrect TC-ME) > 0.

Appearance selection among motion-safe candidates:
1. both cases significant TC-MAR improvement;
2. one case significant improvement and no significant regression
   in the other;
3. no significant changes;
4. any significant appearance regression rejects candidate.

If candidates remain tied:
rank by mean relative TC-MAR improvement over Santa + Tree.

No metric, threshold, candidate definition, patch size, alpha,
guidance scale, track count, or arbitration rule may be changed
after generated results are inspected.
EOF


# ============================================================
# 1. Provenance before touching live Wan-Move code
# ============================================================

cd "$WAN"

git rev-parse HEAD \
> "$SUITE/wanmove_git_head.txt"

git status --short \
> "$SUITE/wanmove_git_status_before.txt"

git diff -- \
  wan/wan_move.py \
  wan/modules/trajectory.py \
> "$SUITE/wanmove_pre_suite.diff"

cp wan/wan_move.py \
"$SUITE/wan_move.py.pre_suite"

cp wan/modules/trajectory.py \
"$SUITE/trajectory.py.pre_suite"


# ============================================================
# 2. Require formal RNG fairness patch
# ============================================================

grep -q \
"DeformTransport formal protocol: freeze trajectory-internal RNG" \
wan/wan_move.py || {
    echo
    echo "ERROR:"
    echo "Formal RNG freeze patch is missing from wan/wan_move.py"
    echo "Refusing to launch suite."
    exit 31
}


# ============================================================
# 3. Restore pre-V3A trajectory.py
#
# Do NOT use git reset.
# Use the exact preserved snapshot made before V3A.
# ============================================================

V3A_PTR="$DT/server_runs/wan_move_method_dev/current_v3a.txt"

if [ ! -f "$V3A_PTR" ]; then
    echo "ERROR: missing $V3A_PTR"
    echo "Refusing to guess baseline trajectory.py"
    exit 32
fi

V3A_DEV=$(cat "$V3A_PTR")

BASE_TRAJ="$V3A_DEV/trajectory.py.before"

if [ ! -f "$BASE_TRAJ" ]; then
    echo "ERROR: missing preserved pre-V3A snapshot:"
    echo "$BASE_TRAJ"
    exit 33
fi

cp "$BASE_TRAJ" \
wan/modules/trajectory.py

cp "$BASE_TRAJ" \
"$SUITE/trajectory.py.pre_v3a_baseline"

echo "PRE_V3A_BASELINE_RESTORED"


# ============================================================
# 4. Build:
#
# - persistent material IDs
# - camera-depth sidecars
# - V3S actual-Wan-grid de-duplicated tracks
#
# Tree depth is STRICTLY checked.
# If depth contract cannot be established, STOP before GPU.
# ============================================================

cd "$DT"

cat > "$SUITE/build_suite_artifacts.py" <<'PY'
from pathlib import Path
import json

import numpy as np
import torch


DT = Path("/workspace/DeformTransport")

SUITE = Path(
    (
        DT
        / "server_runs/wan_move_method_dev/current_v3_suite.txt"
    ).read_text().strip()
)

ART = SUITE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


CASES = {
    "santa": {
        "bridge": (
            DT
            / "server_runs/wan_move_bridge/"
              "20260809_010015__santa_correct_tracks"
        ),
        "tracks": "santa_material_tracks_correct.npy",
        "vis": "santa_material_visibility_correct.npy",
        "ids_candidates": [
            "santa_material_point_ids.npy",
            "material_point_ids.npy",
        ],
        "raw": (
            DT
            / "server_runs/"
              "20260804_234925_autonomous_deformtransport/"
              "04_smoke/"
              "OFFICIAL_SANTA_81F_CHAIN_20260805_050719/"
              "final_sim/point_trajectories.pt"
        ),
        "expected_full_points": 28264,
    },

    "tree": {
        "bridge": (
            DT
            / "server_runs/wan_move_bridge/"
              "20260810_072215__tree_correct_tracks"
        ),
        "tracks": "tree_material_tracks_correct.npy",
        "vis": "tree_material_visibility_correct.npy",
        "ids_candidates": [
            "tree_material_point_ids.npy",
            "material_point_ids.npy",
        ],
        "raw": None,
        "expected_full_points": 15774,
    },
}


def find_ids(bridge, candidates):

    for name in candidates:
        p = bridge / name

        if p.is_file():
            return p

    hits = sorted(
        bridge.glob("*point_ids*.npy")
    )

    if len(hits) == 1:
        return hits[0]

    raise RuntimeError(
        f"cannot uniquely locate material point IDs "
        f"in {bridge}: {hits}"
    )


def tree_raw_from_report(bridge):

    rp = bridge / "report.json"

    if not rp.is_file():
        raise RuntimeError(
            f"missing tree bridge report: {rp}"
        )

    report = json.loads(
        rp.read_text()
    )

    source = (
        report
        .get("lineage", {})
        .get("trajectory_source")
    )

    if not source:
        raise RuntimeError(
            "tree bridge report has no "
            "lineage.trajectory_source"
        )

    p = Path(source)

    # First accept a path that is already valid
    # inside the current runtime.
    if p.is_file():
        return p

    # Historical reports may contain the host-side path.
    #
    # Host:
    # /mnt/sdbd/home/liuyu_qyh/...
    #
    # Container bind mount:
    # /workspace/...
    host_prefix = Path(
        "/mnt/sdbd/home/liuyu_qyh"
    )

    try:
        rel = p.relative_to(
            host_prefix
        )

        mapped = (
            Path("/workspace")
            / rel
        )

    except ValueError:
        mapped = None

    if (
        mapped is not None
        and mapped.is_file()
    ):
        print(
            "TREE_PATH_REMAP:",
            p,
            "->",
            mapped,
            flush=True,
        )

        return mapped

    raise RuntimeError(
        "tree trajectory source cannot be resolved\n"
        f"reported_path={p}\n"
        f"container_mapped_path={mapped}"
    )


def extract_full_depth(
    raw_path,
    expected_n,
):

    obj = torch.load(
        raw_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "objects" not in obj
        or len(obj["objects"]) != 1
    ):
        raise RuntimeError(
            f"unexpected trajectory object structure: "
            f"{raw_path}"
        )

    ro = obj["objects"][0]

    if "depth" not in ro:
        raise RuntimeError(
            "NO_POINT_DEPTH_KEY\n"
            f"path={raw_path}\n"
            f"object_keys={sorted(ro.keys())}\n"
            "V3C/V3D/V3E require explicit point-wise "
            "camera depth; refusing to invent it."
        )

    depth = (
        ro["depth"]
        .float()
        .numpy()
    )

    if depth.shape == (
        81,
        expected_n,
    ):
        full = depth

    elif depth.shape == (
        80,
        expected_n,
    ):

        initial = None

        for key in (
            "initial_depth",
            "depth_initial",
            "initial_points_depth",
        ):

            if key not in ro:
                continue

            candidate = (
                ro[key]
                .float()
                .numpy()
            )

            if candidate.shape == (
                expected_n,
            ):
                initial = candidate
                break

        if initial is None:
            raise RuntimeError(
                f"depth has shape {depth.shape}, "
                "but no accepted initial-depth vector "
                f"exists in {raw_path}; "
                f"keys={sorted(ro.keys())}"
            )

        full = np.concatenate(
            [
                initial[None],
                depth,
            ],
            axis=0,
        )

    else:
        raise RuntimeError(
            f"unsupported depth shape "
            f"{depth.shape} in {raw_path}"
        )

    return (
        full.astype(
            np.float32
        ),
        obj,
    )


report = {
    "actual_wan_domain": [
        464,
        832,
    ],
    "vae_stride": 8,
    "latent_grid": [
        58,
        104,
    ],
    "cases": {},
}


for case, cfg in CASES.items():

    print(
        f"\n===== BUILD {case.upper()} =====",
        flush=True,
    )

    bridge = cfg["bridge"]

    track_path = (
        bridge
        / cfg["tracks"]
    )

    vis_path = (
        bridge
        / cfg["vis"]
    )

    ids_path = find_ids(
        bridge,
        cfg["ids_candidates"],
    )

    tracks = np.load(
        track_path
    ).astype(
        np.float32
    )

    visibility = np.load(
        vis_path
    ).astype(
        bool
    )

    material_ids = np.load(
        ids_path
    ).astype(
        np.int64
    )

    assert tracks.ndim == 4
    assert tracks.shape[0] == 1
    assert tracks.shape[1] == 81
    assert tracks.shape[-1] == 2

    assert (
        visibility.shape
        == tracks.shape[:-1]
    )

    assert (
        material_ids.shape
        == (tracks.shape[2],)
    )

    selected_n = tracks.shape[2]

    expected_n = int(
        cfg["expected_full_points"]
    )

    if case == "santa":
        raw_path = cfg["raw"]

    else:
        raw_path = tree_raw_from_report(
            bridge
        )

    full_depth, raw_obj = (
        extract_full_depth(
            raw_path,
            expected_n,
        )
    )

    if material_ids.min() < 0:
        raise RuntimeError(
            f"{case}: negative material ID"
        )

    if material_ids.max() >= expected_n:
        raise RuntimeError(
            f"{case}: material IDs out of range"
        )

    depth = full_depth[
        :,
        material_ids,
    ]

    # Wan-Move temporal conditioning slots:
    # 0,4,8,...,80.
    temporal_vis = (
        visibility[0, ::4]
    )

    temporal_depth = (
        depth[::4]
    )

    checked_depth = (
        temporal_depth[
            temporal_vis
        ]
    )

    if checked_depth.size == 0:
        raise RuntimeError(
            f"{case}: no visible depth samples"
        )

    if not np.isfinite(
        checked_depth
    ).all():
        raise RuntimeError(
            f"{case}: visible depth "
            "contains non-finite values"
        )

    positive_fraction = float(
        (
            checked_depth
            > 0
        ).mean()
    )

    if positive_fraction < 0.999:
        raise RuntimeError(
            f"{case}: depth sign/contract "
            "cannot be safely inferred; "
            f"positive_fraction="
            f"{positive_fraction}. "
            "Refusing implicit sign flip."
        )

    case_out = (
        ART
        / case
    )

    case_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        case_out
        / f"{case}_old_correct_depth.npy",
        depth[None],
    )

    np.save(
        case_out
        / f"{case}_old_correct_ids.npy",
        material_ids,
    )

    # --------------------------------------------------------
    # V3S
    #
    # Existing Correct track set,
    # but source-cell de-dup is performed on ACTUAL
    # Wan 832x464 conditioning domain.
    # --------------------------------------------------------

    xy0 = (
        tracks[0, 0]
        .astype(
            np.float64
        )
        .copy()
    )

    # Aligned input track domain is 832x480.
    # Wan generated/conditioning image domain is 832x464.
    xy0[:, 1] *= (
        464.0
        / 480.0
    )

    valid0 = (
        visibility[0, 0]
        & np.isfinite(
            xy0
        ).all(axis=1)
        & (xy0[:, 0] >= 0)
        & (xy0[:, 0] < 832)
        & (xy0[:, 1] >= 0)
        & (xy0[:, 1] < 464)
    )

    candidate_ids = np.where(
        valid0
    )[0]

    cell_x = np.floor(
        xy0[
            candidate_ids,
            0,
        ]
        / 8
    ).astype(
        np.int64
    )

    cell_y = np.floor(
        xy0[
            candidate_ids,
            1,
        ]
        / 8
    ).astype(
        np.int64
    )

    cell_id = (
        cell_y * 104
        + cell_x
    )

    center_x = (
        cell_x
        + 0.5
    ) * 8.0

    center_y = (
        cell_y
        + 0.5
    ) * 8.0

    dist2 = (
        (
            xy0[
                candidate_ids,
                0,
            ]
            - center_x
        ) ** 2
        +
        (
            xy0[
                candidate_ids,
                1,
            ]
            - center_y
        ) ** 2
    )

    # Primary:
    # cell id
    #
    # Within cell:
    # nearest point to cell center
    #
    # Exact tie:
    # smaller persistent material ID.
    order = np.lexsort(
        (
            material_ids[
                candidate_ids
            ],
            dist2,
            cell_id,
        )
    )

    sorted_cells = (
        cell_id[
            order
        ]
    )

    first = np.ones(
        len(order),
        dtype=bool,
    )

    if len(order) > 1:
        first[1:] = (
            sorted_cells[1:]
            != sorted_cells[:-1]
        )

    keep = (
        candidate_ids[
            order[first]
        ]
    )

    v3s_tracks = tracks[
        :,
        :,
        keep,
        :,
    ]

    v3s_visibility = visibility[
        :,
        :,
        keep,
    ]

    v3s_ids = material_ids[
        keep
    ]

    v3s_depth = depth[
        :,
        keep,
    ]

    # Verify source-grid uniqueness after actual Wan scaling.
    check_xy = (
        v3s_tracks[0, 0]
        .astype(
            np.float64
        )
        .copy()
    )

    check_xy[:, 1] *= (
        464.0
        / 480.0
    )

    check_x = np.floor(
        check_xy[:, 0]
        / 8
    ).astype(int)

    check_y = np.floor(
        check_xy[:, 1]
        / 8
    ).astype(int)

    check_cells = (
        check_y * 104
        + check_x
    )

    assert (
        len(
            np.unique(
                check_cells
            )
        )
        == len(keep)
    )

    np.save(
        case_out
        / f"{case}_v3s_tracks.npy",
        v3s_tracks,
    )

    np.save(
        case_out
        / f"{case}_v3s_visibility.npy",
        v3s_visibility,
    )

    np.save(
        case_out
        / f"{case}_v3s_ids.npy",
        v3s_ids,
    )

    np.save(
        case_out
        / f"{case}_v3s_depth.npy",
        v3s_depth[None],
    )

    report["cases"][case] = {
        "old_track_count":
            int(
                selected_n
            ),

        "v3s_track_count":
            int(
                len(keep)
            ),

        "removed_source_collisions":
            int(
                selected_n
                - len(keep)
            ),

        "depth_source":
            str(
                raw_path
            ),

        "visible_depth_positive_fraction":
            positive_fraction,

        "ids_source":
            str(
                ids_path
            ),
    }


out = (
    ART
    / "artifact_report.json"
)

out.write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n"
)

print(
    json.dumps(
        report,
        indent=2,
    )
)

print(
    "\nARTIFACT_PREFLIGHT_OK"
)
PY


"$PY" \
"$SUITE/build_suite_artifacts.py" \
| tee "$SUITE/artifact_build.log"


# ============================================================
# 5. Install one frozen multi-mode trajectory implementation
#
# DT_TRANSPORT_VARIANT selects:
#
# baseline / v3s
# v3b
# v3c
# v3d
# v3e
# ============================================================

cd "$WAN"

cat > "$SUITE/patch_trajectory_suite.py" <<'PY'
from pathlib import Path


p = Path(
    "/workspace/Wan-Move/"
    "wan/modules/trajectory.py"
)

text = p.read_text()

start = text.index(
    "def create_pos_feature_map("
)

end = text.index(
    "\ndef get_video_track_video(",
    start,
)


new = r'''
_DT_CONTEXT = None


def _dt_mode():
    import os

    return (
        os.environ
        .get(
            "DT_TRANSPORT_VARIANT",
            "baseline",
        )
        .strip()
        .lower()
    )


def _dt_load_sidecars(
    n,
    device,
):
    import os
    import numpy as np

    depth_path = os.environ.get(
        "DT_TRACK_DEPTH_PATH",
        "",
    )

    ids_path = os.environ.get(
        "DT_TRACK_IDS_PATH",
        "",
    )

    if (
        not depth_path
        or not ids_path
    ):
        raise RuntimeError(
            "DT custom transport requires "
            "DT_TRACK_DEPTH_PATH and "
            "DT_TRACK_IDS_PATH"
        )

    depth_np = np.load(
        depth_path
    )

    ids_np = np.load(
        ids_path
    )

    if (
        depth_np.ndim == 3
        and depth_np.shape[0] == 1
    ):
        depth_np = depth_np[0]

    if depth_np.shape != (
        81,
        n,
    ):
        raise RuntimeError(
            f"depth sidecar shape "
            f"{depth_np.shape}, "
            f"expected (81,{n})"
        )

    if ids_np.shape != (
        n,
    ):
        raise RuntimeError(
            f"ID sidecar shape "
            f"{ids_np.shape}, "
            f"expected ({n},)"
        )

    depth = (
        torch
        .from_numpy(
            depth_np
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    ids = (
        torch
        .from_numpy(
            ids_np.astype(
                np.int64
            )
        )
        .to(
            device=device,
        )
    )

    return (
        depth,
        ids,
    )


def create_pos_feature_map(
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    downsample_ratios: list[int],
    height: int,
    width: int,
    pos_emb_dim: int,
    track_num: int = -1,
    t_down_strategy: str = "sample",
    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
    dtype: torch.dtype = torch.float32,
):

    global _DT_CONTEXT

    assert (
        t_down_strategy
        in [
            "sample",
            "average",
        ]
    ), "Invalid strategy"

    t, n, _ = pred_tracks.shape

    (
        t_down,
        h_down,
        w_down,
    ) = downsample_ratios

    feature_map = torch.zeros(
        (
            (t - 1)
            // t_down
            + 1,

            height
            // h_down,

            width
            // w_down,

            pos_emb_dim,
        ),
        device=device,
        dtype=dtype,
    )

    track_pos = (
        -torch.ones(
            n,
            (t - 1)
            // t_down
            + 1,
            2,
            dtype=torch.long,
        )
    )

    if track_num == -1:
        track_num = n

    # Preserve original Wan-Move RNG consumption.
    tracks_idx = (
        torch.randperm(n)[
            :track_num
        ]
    )

    tracks = pred_tracks[
        :,
        tracks_idx,
    ]

    visibility = pred_visibility[
        :,
        tracks_idx,
    ]

    # Preserve second original randperm.
    tracks_embs = get_pos_emb(
        torch.randperm(n)[
            :track_num
        ],
        pos_emb_dim,
        device=device,
        dtype=dtype,
    )

    for t_idx in range(
        0,
        t,
        t_down,
    ):

        if (
            t_down_strategy
            == "sample"
            or t_idx == 0
        ):
            cur_tracks = tracks[
                t_idx
            ]

            cur_visibility = visibility[
                t_idx
            ]

        else:
            cur_tracks = (
                tracks[
                    t_idx:
                    t_idx
                    + t_down
                ]
                .mean(
                    dim=0
                )
            )

            cur_visibility = (
                torch.any(
                    visibility[
                        t_idx:
                        t_idx
                        + t_down
                    ],
                    dim=0,
                )
            )

        for i in range(
            track_num
        ):

            if (
                not cur_visibility[i]
                or cur_tracks[i][0] < 0
                or cur_tracks[i][1] < 0
                or cur_tracks[i][0] >= width
                or cur_tracks[i][1] >= height
            ):
                continue

            x, y = cur_tracks[i]

            x = int(
                x
                // w_down
            )

            y = int(
                y
                // h_down
            )

            feature_map[
                t_idx
                // t_down,
                y,
                x,
            ] += tracks_embs[i]

            track_pos[
                i,
                t_idx
                // t_down,
                0,
            ] = y

            track_pos[
                i,
                t_idx
                // t_down,
                1,
            ] = x


    mode = _dt_mode()

    if mode in {
        "v3b",
        "v3c",
        "v3d",
        "v3e",
    }:

        if (
            t_down_strategy
            != "sample"
        ):
            raise RuntimeError(
                "DT suite is frozen "
                "to Wan-Move sample "
                "temporal downsampling"
            )

        depth, ids = (
            _dt_load_sidecars(
                n,
                pred_tracks.device,
            )
        )

        idx_device = (
            tracks_idx
            .to(
                pred_tracks.device
            )
        )

        _DT_CONTEXT = {
            "tracks":
                tracks,

            "visibility":
                visibility,

            "depth":
                depth[
                    :,
                    idx_device,
                ],

            "ids":
                ids[
                    idx_device
                ],

            "t_down":
                int(
                    t_down
                ),

            "h_down":
                int(
                    h_down
                ),

            "w_down":
                int(
                    w_down
                ),

            "height":
                int(
                    height
                ),

            "width":
                int(
                    width
                ),
        }

    else:
        _DT_CONTEXT = None

    return (
        feature_map,
        track_pos,
    )


def _dt_original_replace(
    vae_feature,
    track_pos,
):

    b, _, t, h, w = (
        vae_feature.shape
    )

    assert (
        b
        == track_pos.shape[0]
    ), "Batch size mismatch."

    n = track_pos.shape[1]

    # Preserve original third randperm.
    track_pos = track_pos[
        :,
        torch.randperm(n),
        :,
        :,
    ]

    current_pos = (
        track_pos[
            :,
            :,
            1:,
            :,
        ]
    )

    mask = (
        (current_pos[..., 0] >= 0)
        &
        (current_pos[..., 1] >= 0)
    )

    valid_indices = (
        mask
        .nonzero(
            as_tuple=False
        )
    )

    if (
        valid_indices.shape[0]
        == 0
    ):
        return vae_feature

    batch_idx = valid_indices[
        :,
        0,
    ]

    track_idx = valid_indices[
        :,
        1,
    ]

    t_rel = valid_indices[
        :,
        2,
    ]

    t_target = (
        t_rel
        + 1
    )

    h_target = (
        current_pos[
            batch_idx,
            track_idx,
            t_rel,
            0,
        ]
        .long()
    )

    w_target = (
        current_pos[
            batch_idx,
            track_idx,
            t_rel,
            1,
        ]
        .long()
    )

    h_source = (
        track_pos[
            batch_idx,
            track_idx,
            0,
            0,
        ]
        .long()
    )

    w_source = (
        track_pos[
            batch_idx,
            track_idx,
            0,
            1,
        ]
        .long()
    )

    src_features = (
        vae_feature[
            batch_idx,
            :,
            0,
            h_source,
            w_source,
        ]
    )

    vae_feature[
        batch_idx,
        :,
        t_target,
        h_target,
        w_target,
    ] = src_features

    return vae_feature


def _dt_bilinear_source_features(
    vae_feature,
    source_xy,
    stride_y,
    stride_x,
):

    # Treat the discrete VAE condition latent as a
    # cell-centered continuous feature field.
    #
    # latent index:
    # pixel / stride - 0.5
    #
    # This is a fixed geometric convention.
    # No tunable alpha is introduced.

    source_xy = (
        source_xy
        .to(
            device=vae_feature.device,
            dtype=torch.float32,
        )
    )

    _, c, _, h, w = (
        vae_feature.shape
    )

    lx = (
        source_xy[:, 0]
        / float(
            stride_x
        )
        - 0.5
    )

    ly = (
        source_xy[:, 1]
        / float(
            stride_y
        )
        - 0.5
    )

    if w > 1:
        gx = (
            lx
            * (
                2.0
                / (
                    w - 1
                )
            )
            - 1.0
        )
    else:
        gx = torch.zeros_like(
            lx
        )

    if h > 1:
        gy = (
            ly
            * (
                2.0
                / (
                    h - 1
                )
            )
            - 1.0
        )
    else:
        gy = torch.zeros_like(
            ly
        )

    grid = torch.stack(
        [
            gx,
            gy,
        ],
        dim=-1,
    ).view(
        1,
        1,
        -1,
        2,
    )

    src = (
        vae_feature[
            0,
            :,
            0,
        ]
        .unsqueeze(0)
    )

    sampled = (
        torch.nn.functional.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    )

    # [1,C,1,N]
    # -> [N,C]
    return (
        sampled[
            0,
            :,
            0,
            :,
        ]
        .transpose(
            0,
            1
        )
        .contiguous()
    )


def replace_feature(
    vae_feature: torch.Tensor,
    track_pos: torch.Tensor,
) -> torch.Tensor:

    import math

    global _DT_CONTEXT

    mode = _dt_mode()

    if mode in {
        "baseline",
        "v3s",
        "",
    }:
        return _dt_original_replace(
            vae_feature,
            track_pos,
        )

    if mode not in {
        "v3b",
        "v3c",
        "v3d",
        "v3e",
    }:
        raise RuntimeError(
            f"unknown "
            f"DT_TRANSPORT_VARIANT="
            f"{mode}"
        )

    if _DT_CONTEXT is None:
        raise RuntimeError(
            "DT transport context missing; "
            "create_pos_feature_map "
            "must run first"
        )

    if (
        vae_feature.shape[0]
        != 1
        or track_pos.shape[0]
        != 1
    ):
        raise RuntimeError(
            "DT suite freezes "
            "batch size 1"
        )

    n = track_pos.shape[1]

    # Preserve original Wan-Move third randperm
    # RNG consumption.
    #
    # Reorder every associated material field
    # identically.
    perm = torch.randperm(
        n
    )

    track_pos = track_pos[
        :,
        perm,
        :,
        :,
    ]

    perm_device = perm.to(
        _DT_CONTEXT[
            "tracks"
        ].device
    )

    tracks = (
        _DT_CONTEXT[
            "tracks"
        ][
            :,
            perm_device,
        ]
    )

    visibility = (
        _DT_CONTEXT[
            "visibility"
        ][
            :,
            perm_device,
        ]
    )

    depth = (
        _DT_CONTEXT[
            "depth"
        ][
            :,
            perm_device,
        ]
    )

    ids = (
        _DT_CONTEXT[
            "ids"
        ][
            perm_device
        ]
    )

    t_down = (
        _DT_CONTEXT[
            "t_down"
        ]
    )

    stride_y = (
        _DT_CONTEXT[
            "h_down"
        ]
    )

    stride_x = (
        _DT_CONTEXT[
            "w_down"
        ]
    )

    tracks_sampled = (
        tracks[
            ::t_down
        ]
    )

    vis_sampled = (
        visibility[
            ::t_down
        ]
    )

    depth_sampled = (
        depth[
            ::t_down
        ]
    )

    latent_t = (
        track_pos.shape[2]
    )

    assert (
        tracks_sampled.shape[0]
        == latent_t
    )

    assert (
        depth_sampled.shape[0]
        == latent_t
    )

    src_h = (
        track_pos[
            0,
            :,
            0,
            0,
        ]
        .to(
            tracks.device
        )
    )

    src_w = (
        track_pos[
            0,
            :,
            0,
            1,
        ]
        .to(
            tracks.device
        )
    )

    src_valid = (
        (src_h >= 0)
        &
        (src_w >= 0)
    )

    if mode == "v3d":

        source_features = (
            _dt_bilinear_source_features(
                vae_feature,
                tracks_sampled[0],
                stride_y,
                stride_x,
            )
        )

    else:
        source_features = None

    H = vae_feature.shape[-2]
    W = vae_feature.shape[-1]

    for tau in range(
        1,
        latent_t,
    ):

        target_h = (
            track_pos[
                0,
                :,
                tau,
                0,
            ]
            .to(
                tracks.device
            )
        )

        target_w = (
            track_pos[
                0,
                :,
                tau,
                1,
            ]
            .to(
                tracks.device
            )
        )

        valid = (
            src_valid
            &
            (target_h >= 0)
            &
            (target_w >= 0)
            &
            vis_sampled[tau]
        )

        indices = torch.where(
            valid
        )[0]

        if (
            indices.numel()
            == 0
        ):
            continue


        # ====================================================
        # V3B / V3C / V3D:
        # exactly one source material identity per target cell.
        # ====================================================

        if mode in {
            "v3b",
            "v3c",
            "v3d",
        }:

            groups = {}

            for ii in indices.tolist():

                hh = int(
                    target_h[
                        ii
                    ].item()
                )

                ww = int(
                    target_w[
                        ii
                    ].item()
                )

                groups.setdefault(
                    (
                        hh,
                        ww,
                    ),
                    [],
                ).append(
                    ii
                )

            for (
                hh,
                ww,
            ), members in groups.items():

                best = None
                best_key = None

                for ii in members:

                    material_id = int(
                        ids[
                            ii
                        ].item()
                    )

                    if mode == "v3b":

                        x = float(
                            tracks_sampled[
                                tau,
                                ii,
                                0,
                            ].item()
                        )

                        y = float(
                            tracks_sampled[
                                tau,
                                ii,
                                1,
                            ].item()
                        )

                        # Squared quantization error
                        # to target latent-cell center.
                        score = (
                            (
                                x
                                - (
                                    ww
                                    + 0.5
                                )
                                * stride_x
                            ) ** 2
                            +
                            (
                                y
                                - (
                                    hh
                                    + 0.5
                                )
                                * stride_y
                            ) ** 2
                        )

                    else:

                        score = float(
                            depth_sampled[
                                tau,
                                ii,
                            ].item()
                        )

                        if (
                            not math.isfinite(
                                score
                            )
                            or score <= 0
                        ):
                            continue

                    key = (
                        score,
                        material_id,
                    )

                    if (
                        best_key is None
                        or key
                        < best_key
                    ):
                        best_key = key
                        best = ii

                if best is None:
                    continue

                if mode == "v3d":

                    feat = (
                        source_features[
                            best
                        ]
                    )

                else:

                    source_h = int(
                        src_h[
                            best
                        ].item()
                    )

                    source_w = int(
                        src_w[
                            best
                        ].item()
                    )

                    feat = (
                        vae_feature[
                            0,
                            :,
                            0,
                            source_h,
                            source_w,
                        ]
                    )

                vae_feature[
                    0,
                    :,
                    tau,
                    hh,
                    ww,
                ] = feat


        # ====================================================
        # V3E:
        #
        # depth-aware 3x3 local source-condition patch.
        #
        # Every target latent cell still has exactly one
        # winning material contributor.
        # ====================================================

        else:

            groups = {}

            for ii in indices.tolist():

                source_h0 = int(
                    src_h[
                        ii
                    ].item()
                )

                source_w0 = int(
                    src_w[
                        ii
                    ].item()
                )

                target_h0 = int(
                    target_h[
                        ii
                    ].item()
                )

                target_w0 = int(
                    target_w[
                        ii
                    ].item()
                )

                z = float(
                    depth_sampled[
                        tau,
                        ii,
                    ].item()
                )

                if (
                    not math.isfinite(
                        z
                    )
                    or z <= 0
                ):
                    continue

                material_id = int(
                    ids[
                        ii
                    ].item()
                )

                for dy in (
                    -1,
                    0,
                    1,
                ):
                    for dx in (
                        -1,
                        0,
                        1,
                    ):

                        source_h = (
                            source_h0
                            + dy
                        )

                        source_w = (
                            source_w0
                            + dx
                        )

                        hh = (
                            target_h0
                            + dy
                        )

                        ww = (
                            target_w0
                            + dx
                        )

                        if not (
                            0 <= source_h < H
                            and 0 <= source_w < W
                            and 0 <= hh < H
                            and 0 <= ww < W
                        ):
                            continue

                        groups.setdefault(
                            (
                                hh,
                                ww,
                            ),
                            [],
                        ).append(
                            (
                                z,
                                material_id,
                                source_h,
                                source_w,
                            )
                        )

            for (
                hh,
                ww,
            ), candidates in groups.items():

                (
                    z,
                    material_id,
                    source_h,
                    source_w,
                ) = min(
                    candidates,
                    key=lambda item: (
                        item[0],
                        item[1],
                    ),
                )

                vae_feature[
                    0,
                    :,
                    tau,
                    hh,
                    ww,
                ] = (
                    vae_feature[
                        0,
                        :,
                        0,
                        source_h,
                        source_w,
                    ]
                )

    return vae_feature
'''


patched = (
    text[:start]
    + new
    + text[end:]
)

p.write_text(
    patched
)

print(
    "TRAJECTORY_SUITE_PATCHED"
)
PY


"$PY" \
"$SUITE/patch_trajectory_suite.py"

"$PY" -m py_compile \
  wan/modules/trajectory.py \
  wan/wan_move.py

echo "PATCH_COMPILE_OK"


# ============================================================
# 6. Synthetic unit tests
# ============================================================

cat > "$SUITE/unit_test_suite.py" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

# Force the Wan-Move repository ahead of DeformTransport,\n# because both repositories contain a top-level package named wan.\nWAN_ROOT = "/workspace/Wan-Move"\nsys.path.insert(0, WAN_ROOT)\n
import numpy as np
import torch

from wan.modules.trajectory import (
    create_pos_feature_map,
    replace_feature,
)


def run(
    mode,
    tracks,
    visibility,
    depth,
    ids,
    feature,
):

    with tempfile.TemporaryDirectory() as td:

        td = Path(td)

        depth_path = td / "depth.npy"
        ids_path = td / "ids.npy"

        np.save(
            depth_path,
            depth[
                None
            ].astype(
                np.float32
            ),
        )

        np.save(
            ids_path,
            ids.astype(
                np.int64
            ),
        )

        os.environ[
            "DT_TRANSPORT_VARIANT"
        ] = mode

        os.environ[
            "DT_TRACK_DEPTH_PATH"
        ] = str(
            depth_path
        )

        os.environ[
            "DT_TRACK_IDS_PATH"
        ] = str(
            ids_path
        )

        torch.manual_seed(
            0
        )

        _, pos = (
            create_pos_feature_map(
                torch.from_numpy(
                    tracks
                ),
                torch.from_numpy(
                    visibility
                ),
                [
                    4,
                    8,
                    8,
                ],
                16,
                16,
                feature.shape[1],
                track_num=tracks.shape[1],
                device=feature.device,
            )
        )

        return replace_feature(
            feature.clone(),
            pos.unsqueeze(0),
        )


T = 81
N = 2

tracks = np.zeros(
    (
        T,
        N,
        2,
    ),
    np.float32,
)

visibility = np.ones(
    (
        T,
        N,
    ),
    bool,
)

depth = np.ones(
    (
        T,
        N,
    ),
    np.float32,
)

ids = np.array(
    [
        10,
        20,
    ],
    dtype=np.int64,
)

# Distinct source cells:
#
# track 0 -> source latent cell (0,0)
# track 1 -> source latent cell (0,1)
tracks[:, 0] = [
    4,
    4,
]

tracks[:, 1] = [
    12,
    4,
]

# At t>=4 both collide at latent cell (1,1).
tracks[
    4:,
    0,
] = [
    9,
    9,
]

tracks[
    4:,
    1,
] = [
    15,
    15,
]


feature = torch.zeros(
    1,
    2,
    21,
    2,
    2,
    dtype=torch.float32,
)

feature[
    0,
    :,
    0,
    0,
    0,
] = torch.tensor(
    [
        1.0,
        10.0,
    ]
)

feature[
    0,
    :,
    0,
    0,
    1,
] = torch.tensor(
    [
        3.0,
        30.0,
    ]
)


# ------------------------------------------------------------
# V3B
#
# target latent center = (12,12).
# track0 position = (9,9)
# track1 position = (15,15)
#
# Equal squared distance.
# Material ID 10 must win.
# ------------------------------------------------------------

out = run(
    "v3b",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

got = float(
    out[
        0,
        0,
        1,
        1,
        1,
    ]
)

assert abs(
    got
    - 1.0
) < 1e-6

print(
    "V3B_UNIT_TEST_OK"
)


# ------------------------------------------------------------
# V3C
#
# Track 1 is closer in camera depth.
# Must transport feature value 3.
# ------------------------------------------------------------

depth[:, 0] = 2.0
depth[:, 1] = 1.0

out = run(
    "v3c",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

got = float(
    out[
        0,
        0,
        1,
        1,
        1,
    ]
)

assert abs(
    got
    - 3.0
) < 1e-6

print(
    "V3C_UNIT_TEST_OK"
)


# ------------------------------------------------------------
# V3D
#
# Continuous source sampling smoke test.
# ------------------------------------------------------------

out = run(
    "v3d",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

assert torch.isfinite(
    out
).all()

print(
    "V3D_UNIT_TEST_OK"
)


# ------------------------------------------------------------
# V3E
#
# Local 3x3 transport smoke test.
# ------------------------------------------------------------

out = run(
    "v3e",
    tracks,
    visibility,
    depth,
    ids,
    feature,
)

assert torch.isfinite(
    out
).all()

print(
    "V3E_UNIT_TEST_OK"
)

print(
    "V3_SUITE_UNIT_TEST_OK"
)
PY


cd "$WAN"

"$PY" \
"$SUITE/unit_test_suite.py" \
| tee "$SUITE/unit_test.log"


# ============================================================
# 7. Freeze installed code provenance
# ============================================================

sha256sum \
  wan/wan_move.py \
  wan/modules/trajectory.py \
> "$SUITE/installed_source_sha256.txt"

git diff -- \
  wan/wan_move.py \
  wan/modules/trajectory.py \
> "$SUITE/wanmove_suite.diff"


# ============================================================
# 8. Shared runner for all ten generations
# ============================================================

cd "$DT"

cat > "$SUITE/run_one.sh" <<'SH2'
#!/usr/bin/env bash
set -euo pipefail

CASE="${1:?usage: run_one.sh santa|tree v3s|v3b|v3c|v3d|v3e gpu}"
VAR="${2:?variant}"
GPU="${3:?gpu}"

DT=/workspace/DeformTransport
WAN=/workspace/Wan-Move
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python
WANENV=/workspace/tools/miniforge3/envs/wan-move

SUITE=$(cat \
"$DT/server_runs/wan_move_method_dev/current_v3_suite.txt")

ART="$SUITE/artifacts/$CASE"
RUN="$SUITE/$CASE/$VAR"

mkdir -p \
"$RUN"


if [ "$CASE" = "santa" ]; then

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/prompt.txt"

    OLD_TRACK="$DT/server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_tracks_correct.npy"

    OLD_VIS="$DT/server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_visibility_correct.npy"

elif [ "$CASE" = "tree" ]; then

    IMAGE="$DT/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/resized_input_image.png"

    PROMPT_FILE="$DT/server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0/prompt.txt"

    OLD_TRACK="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_tracks_correct.npy"

    OLD_VIS="$DT/server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_visibility_correct.npy"

else

    echo "Unknown case: $CASE"
    exit 42
fi


if [ "$VAR" = "v3s" ]; then

    TRACK="$ART/${CASE}_v3s_tracks.npy"
    VIS="$ART/${CASE}_v3s_visibility.npy"
    IDS="$ART/${CASE}_v3s_ids.npy"
    DEPTH="$ART/${CASE}_v3s_depth.npy"

    MODE="v3s"

elif [[ "$VAR" =~ ^v3[bcde]$ ]]; then

    TRACK="$OLD_TRACK"
    VIS="$OLD_VIS"

    IDS="$ART/${CASE}_old_correct_ids.npy"
    DEPTH="$ART/${CASE}_old_correct_depth.npy"

    MODE="$VAR"

else

    echo "Unknown variant: $VAR"
    exit 43
fi


for path in \
"$IMAGE" \
"$PROMPT_FILE" \
"$TRACK" \
"$VIS" \
"$IDS" \
"$DEPTH"
do
    test -f "$path" || {
        echo "MISSING: $path"
        exit 41
    }
done


export CUDA_VISIBLE_DEVICES="$GPU"

export CUDA_HOME="$WANENV"
export CUDA_PATH="$WANENV"

export PATH="$WANENV/bin:$PATH"

export LD_LIBRARY_PATH="$WANENV/targets/x86_64-linux/lib:$WANENV/lib:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1

export DT_TRANSPORT_VARIANT="$MODE"
export DT_TRACK_IDS_PATH="$IDS"
export DT_TRACK_DEPTH_PATH="$DEPTH"


printf '%s\n' \
"$BASHPID" \
> "$RUN/pid.txt"

date -Iseconds \
> "$RUN/start_time.txt"


cat > "$RUN/contract.txt" <<EOF
case=$CASE
variant=$VAR
mode=$MODE
gpu=$GPU

track=$TRACK
visibility=$VIS
ids=$IDS
depth=$DEPTH

image=$IMAGE
prompt=$PROMPT_FILE

seed=0
frame_num=81
steps=40
shift=3.0
dtype=bf16
t5_cpu=True
offload_model=True
EOF


PROMPT=$(cat \
"$PROMPT_FILE")


cd "$WAN"


set +e

"$PY" generate.py \
  --task wan-move-i2v \
  --size '480*832' \
  --frame_num 81 \
  --ckpt_dir "$WAN/Wan-Move-14B-480P" \
  --image "$IMAGE" \
  --track "$TRACK" \
  --track_visibility "$VIS" \
  --prompt "$PROMPT" \
  --base_seed 0 \
  --sample_steps 40 \
  --sample_shift 3.0 \
  --t5_cpu \
  --offload_model True \
  --dtype bf16 \
  --save_file "$RUN/${CASE}_${VAR}_correct_seed0.mp4" \
  > "$RUN/stdout.log" \
  2> "$RUN/stderr.log"

EC=$?

set -e


printf '%s\n' \
"$EC" \
> "$RUN/exit_code.txt"

date -Iseconds \
> "$RUN/end_time.txt"


if \
    [ "$EC" -eq 0 ] \
    && \
    [ -f "$RUN/${CASE}_${VAR}_correct_seed0.mp4" ]
then

    sha256sum \
    "$RUN/${CASE}_${VAR}_correct_seed0.mp4" \
    > "$RUN/output_sha256.txt"

else

    echo \
    "FAILED case=$CASE variant=$VAR ec=$EC" \
    >&2

    exit "$EC"
fi
SH2


chmod +x \
"$SUITE/run_one.sh"


# ============================================================
# 9. Per-case sequential queues
# ============================================================

cat > "$SUITE/queue_case.sh" <<'SH2'
#!/usr/bin/env bash
set -euo pipefail

CASE="${1:?case}"
GPU="${2:?gpu}"

DT=/workspace/DeformTransport

SUITE=$(cat \
"$DT/server_runs/wan_move_method_dev/current_v3_suite.txt")


for VAR in \
v3s \
v3b \
v3c \
v3d \
v3e
do

    echo
    echo "============================================"
    echo "[$(date -Iseconds)]"
    echo "START case=$CASE variant=$VAR GPU=$GPU"
    echo "============================================"

    bash \
    "$SUITE/run_one.sh" \
    "$CASE" \
    "$VAR" \
    "$GPU"

    echo
    echo "[$(date -Iseconds)]"
    echo "DONE case=$CASE variant=$VAR GPU=$GPU"

done


date -Iseconds \
> "$SUITE/$CASE/QUEUE_DONE.txt"

echo
echo "============================================"
echo "$CASE QUEUE COMPLETE"
echo "============================================"
SH2


chmod +x \
"$SUITE/queue_case.sh"


# ============================================================
# 10. Freeze DeformTransport-side provenance
# ============================================================

cd "$DT"

git rev-parse HEAD \
> "$SUITE/deformtransport_git_head.txt"

git status --short \
> "$SUITE/deformtransport_git_status.txt"

sha256sum \
"$SUITE/run_one.sh" \
"$SUITE/queue_case.sh" \
"$SUITE/build_suite_artifacts.py" \
"$SUITE/patch_trajectory_suite.py" \
"$SUITE/unit_test_suite.py" \
> "$SUITE/suite_script_sha256.txt"


# ============================================================
# 11. GPU1/GPU2 safety
#
# GPU0/GPU3 are never touched.
# ============================================================

check_gpu_pair () {

    local snapshot

    snapshot=$(
        nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )

    for idx in 1 2
    do

        local line
        local mem
        local util

        line=$(
            printf '%s\n' \
            "$snapshot" \
            | awk -F',' -v i="$idx" \
              '$1+0==i {print $0}'
        )

        mem=$(
            echo "$line" \
            | awk -F',' \
              '{gsub(/ /,"",$2);print $2}'
        )

        util=$(
            echo "$line" \
            | awk -F',' \
              '{gsub(/ /,"",$3);print $3}'
        )

        if \
            [ -z "$mem" ] \
            || \
            [ -z "$util" ] \
            || \
            [ "$mem" -ge 1000 ] \
            || \
            [ "$util" -gt 5 ]
        then
            return 1
        fi

    done

    return 0
}


echo
echo "========== GPU CHECK #1 =========="

nvidia-smi \
--query-gpu=index,memory.used,memory.total,utilization.gpu \
--format=csv,noheader


if ! check_gpu_pair
then
    echo
    echo "GPU1/2 are not safely free."
    echo
    echo "SETUP IS COMPLETE."
    echo "NO GENERATION WAS LAUNCHED."
    echo
    echo "SUITE=$SUITE"
    echo
    echo "When GPU1/2 become free, run:"
    echo
    echo "nohup bash \"$SUITE/queue_case.sh\" santa 1 > \"$SUITE/santa/queue.log\" 2>&1 < /dev/null &"
    echo "nohup bash \"$SUITE/queue_case.sh\" tree 2 > \"$SUITE/tree/queue.log\" 2>&1 < /dev/null &"
    exit 50
fi


sleep 15


echo
echo "========== GPU CHECK #2 =========="

nvidia-smi \
--query-gpu=index,memory.used,memory.total,utilization.gpu \
--format=csv,noheader


if ! check_gpu_pair
then
    echo
    echo "GPU1/2 became busy during second check."
    echo
    echo "SETUP IS COMPLETE."
    echo "NO GENERATION WAS LAUNCHED."
    echo
    echo "SUITE=$SUITE"
    exit 51
fi


# ============================================================
# 12. Launch two independent five-run queues
# ============================================================

nohup bash \
"$SUITE/queue_case.sh" \
santa \
1 \
> "$SUITE/santa/queue.log" \
2>&1 \
< /dev/null &

SANTA_PID=$!

echo "$SANTA_PID" \
> "$SUITE/santa/queue_pid.txt"


nohup bash \
"$SUITE/queue_case.sh" \
tree \
2 \
> "$SUITE/tree/queue.log" \
2>&1 \
< /dev/null &

TREE_PID=$!

echo "$TREE_PID" \
> "$SUITE/tree/queue_pid.txt"


echo
echo "============================================"
echo "V3_SUITE_LAUNCHED"
echo "============================================"

echo "SUITE=$SUITE"

echo \
"Santa queue PID=$SANTA_PID GPU1"

echo \
"Tree queue PID=$TREE_PID GPU2"

echo
echo "Santa order:"
echo "V3S -> V3B -> V3C -> V3D -> V3E"

echo
echo "Tree order:"
echo "V3S -> V3B -> V3C -> V3D -> V3E"

echo
echo "Do not modify:"
echo "$WAN/wan/wan_move.py"
echo "$WAN/wan/modules/trajectory.py"
echo
