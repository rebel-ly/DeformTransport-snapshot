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

    if not p.is_file():
        raise RuntimeError(
            f"tree trajectory source does not exist: {p}"
        )

    return p


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
