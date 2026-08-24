from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
import torch


DT = Path("/workspace/DeformTransport")

RUN = Path(
    (
        DT
        / "server_runs/wan_move_method_dev/"
          "current_v3d_formal_validation.txt"
    ).read_text().strip()
)

ART = RUN / "artifacts"

V3SUITE = (
    DT
    / "server_runs/wan_move_method_suite/"
      "20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0"
)

SANDROOT = (
    DT
    / "server_runs/sand_house_cached_sim_20260808_174356"
)

SAND_FRAME_IDS = np.arange(
    0,
    161,
    2,
    dtype=np.int64,
)

assert SAND_FRAME_IDS.shape == (81,)
assert SAND_FRAME_IDS[0] == 0
assert SAND_FRAME_IDS[-1] == 160


def sha256(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(4 << 20),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        return x

    return None


def make_derangement(n, seed=0):
    rng = np.random.default_rng(seed)
    identity = np.arange(n)

    for attempt in range(10000):
        perm = rng.permutation(n)

        if np.all(
            perm != identity
        ):
            return perm.astype(np.int64), attempt + 1

    raise RuntimeError(
        f"failed to obtain derangement for n={n}"
    )


def build_shuffled(
    correct_tracks,
):
    assert (
        correct_tracks.ndim == 4
        and correct_tracks.shape[0] == 1
        and correct_tracks.shape[1] == 81
        and correct_tracks.shape[-1] == 2
    )

    n = correct_tracks.shape[2]

    perm, attempts = make_derangement(
        n,
        seed=0,
    )

    shuffled = correct_tracks.copy()

    shuffled[
        0,
        0,
        :,
        :,
    ] = correct_tracks[
        0,
        0,
        perm,
        :,
    ]

    assert np.array_equal(
        shuffled[
            0,
            1:,
        ],
        correct_tracks[
            0,
            1:,
        ],
    )

    assert not np.array_equal(
        shuffled[
            0,
            0,
        ],
        correct_tracks[
            0,
            0,
        ],
    )

    fixed = int(
        np.sum(
            perm
            == np.arange(n)
        )
    )

    assert fixed == 0

    return shuffled, perm, attempts


# ============================================================
# A. Santa + Tree V3D Identity-Shuffled
# ============================================================

DEV_CASES = {
    "santa": {
        "tracks":
            DT
            / "server_runs/wan_move_bridge/"
              "20260809_010015__santa_correct_tracks/"
              "santa_material_tracks_correct.npy",

        "vis":
            DT
            / "server_runs/wan_move_bridge/"
              "20260809_010015__santa_correct_tracks/"
              "santa_material_visibility_correct.npy",

        "ids":
            V3SUITE
            / "artifacts/santa/"
              "santa_old_correct_ids.npy",

        "depth":
            V3SUITE
            / "artifacts/santa/"
              "santa_old_correct_depth.npy",
    },

    "tree": {
        "tracks":
            DT
            / "server_runs/wan_move_bridge/"
              "20260810_072215__tree_correct_tracks/"
              "tree_material_tracks_correct.npy",

        "vis":
            DT
            / "server_runs/wan_move_bridge/"
              "20260810_072215__tree_correct_tracks/"
              "tree_material_visibility_correct.npy",

        "ids":
            V3SUITE
            / "artifacts/tree/"
              "tree_old_correct_ids.npy",

        "depth":
            V3SUITE
            / "artifacts/tree/"
              "tree_old_correct_depth.npy",
    },
}


report = {
    "development_shuffled": {},
    "sandhouse": {},
}


for case, cfg in DEV_CASES.items():

    for p in cfg.values():
        if not p.is_file():
            raise FileNotFoundError(p)

    tracks = np.load(
        cfg["tracks"]
    ).astype(np.float32)

    vis = np.load(
        cfg["vis"]
    ).astype(bool)

    ids = np.load(
        cfg["ids"]
    ).astype(np.int64)

    depth = np.load(
        cfg["depth"]
    ).astype(np.float32)

    n = tracks.shape[2]

    assert tracks.shape == (
        1,
        81,
        n,
        2,
    )

    assert vis.shape == (
        1,
        81,
        n,
    )

    assert ids.shape == (
        n,
    )

    assert depth.shape == (
        1,
        81,
        n,
    )

    shuffled, perm, attempts = (
        build_shuffled(
            tracks
        )
    )

    out = ART / case
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    tr_out = (
        out
        / f"{case}_v3d_identity_shuffled_tracks.npy"
    )

    vi_out = (
        out
        / f"{case}_v3d_visibility.npy"
    )

    id_out = (
        out
        / f"{case}_v3d_ids.npy"
    )

    de_out = (
        out
        / f"{case}_v3d_depth.npy"
    )

    pe_out = (
        out
        / f"{case}_identity_shuffle_perm.npy"
    )

    np.save(
        tr_out,
        shuffled,
    )

    np.save(
        vi_out,
        vis,
    )

    np.save(
        id_out,
        ids,
    )

    np.save(
        de_out,
        depth,
    )

    np.save(
        pe_out,
        perm,
    )

    report[
        "development_shuffled"
    ][
        case
    ] = {
        "n_tracks":
            int(n),

        "derangement_seed":
            0,

        "derangement_attempts":
            int(attempts),

        "fixed_points":
            int(
                np.sum(
                    perm
                    == np.arange(n)
                )
            ),

        "future_tracks_bit_identical":
            bool(
                np.array_equal(
                    shuffled[
                        0,
                        1:
                    ],
                    tracks[
                        0,
                        1:
                    ],
                )
            ),

        "visibility_bit_identical":
            True,

        "depth_bit_identical":
            True,

        "ids_bit_identical":
            True,
    }


# ============================================================
# B. SandHouse strict asset discovery
# ============================================================

if not SANDROOT.is_dir():
    raise RuntimeError(
        f"SandHouse root missing: {SANDROOT}"
    )


def unique_same_content(
    paths,
    kind,
):
    paths = sorted(
        {
            p.resolve()
            for p in paths
            if p.is_file()
        }
    )

    if not paths:
        raise RuntimeError(
            f"no {kind} candidate found"
        )

    groups = {}

    for p in paths:
        h = sha256(p)
        groups.setdefault(
            h,
            [],
        ).append(p)

    if len(groups) != 1:
        lines = "\n".join(
            str(p)
            for p in paths
        )

        raise RuntimeError(
            f"multiple distinct {kind} candidates:\n"
            f"{lines}"
        )

    # Same bytes -> deterministic shortest path.
    return sorted(
        paths,
        key=lambda p: (
            len(str(p)),
            str(p),
        ),
    )[0]


# ---------- image ----------

image_candidates = []

for name in [
    "resized_input_image.png",
    "input_image.png",
    "input.png",
]:

    image_candidates.extend(
        SANDROOT.rglob(name)
    )

image_path = unique_same_content(
    image_candidates,
    "SandHouse input image",
)

img = cv2.imread(
    str(image_path),
    cv2.IMREAD_COLOR,
)

if img is None:
    raise RuntimeError(
        f"cannot decode SandHouse image: {image_path}"
    )

if img.shape != (
    480,
    832,
    3,
):
    raise RuntimeError(
        f"SandHouse input image shape "
        f"{img.shape}, expected (480,832,3)"
    )


# ---------- prompt ----------

prompt_candidates = list(
    SANDROOT.rglob(
        "prompt.txt"
    )
)

prompt_path = unique_same_content(
    prompt_candidates,
    "SandHouse prompt",
)

prompt_text = (
    prompt_path
    .read_text(
        encoding="utf-8"
    )
    .strip()
)

if not prompt_text:
    raise RuntimeError(
        f"empty prompt: {prompt_path}"
    )


# ---------- raw trajectory ----------

preferred_raw = (
    SANDROOT
    / "raw_sim/point_trajectories.pt"
)

if preferred_raw.is_file():
    raw_path = preferred_raw

else:
    raw_hits = sorted(
        SANDROOT.rglob(
            "point_trajectories.pt"
        )
    )

    if len(raw_hits) != 1:
        raise RuntimeError(
            "cannot uniquely resolve SandHouse "
            "point_trajectories.pt:\n"
            + "\n".join(
                str(p)
                for p in raw_hits
            )
        )

    raw_path = raw_hits[0]


print(
    "SANDHOUSE_RAW_TRAJECTORY =",
    raw_path,
    flush=True,
)

raw = torch.load(
    raw_path,
    map_location="cpu",
    weights_only=False,
)

if not isinstance(raw, dict):
    raise RuntimeError(
        "SandHouse point trajectory is not dict"
    )

objects = raw.get(
    "objects"
)

if not isinstance(
    objects,
    (list, tuple),
):
    raise RuntimeError(
        "SandHouse trajectory has no objects list"
    )


sand_candidates = []

for idx, obj in enumerate(objects):

    if not isinstance(
        obj,
        dict,
    ):
        continue

    material = str(
        obj.get(
            "material_type",
            ""
        )
    ).lower()

    if "sand" in material:
        sand_candidates.append(
            (
                idx,
                obj,
            )
        )


if len(
    sand_candidates
) != 1:

    materials = [
        str(
            x.get(
                "material_type",
                ""
            )
        )
        if isinstance(
            x,
            dict,
        )
        else type(x).__name__
        for x in objects
    ]

    raise RuntimeError(
        "expected exactly one SandHouse sand "
        f"material object, found "
        f"{len(sand_candidates)}; "
        f"materials={materials}"
    )


sand_obj_index, sand = (
    sand_candidates[0]
)

material_type = str(
    sand.get(
        "material_type",
        ""
    )
)

print(
    "SANDHOUSE_MATERIAL =",
    material_type,
    flush=True,
)


# ============================================================
# Extract full 165-state raw 512 UV sequence
# ============================================================

def full_sequence(
    obj,
    future_key,
    initial_keys,
    last_dims,
    name,
):

    if future_key not in obj:
        raise RuntimeError(
            f"missing {name} key {future_key}; "
            f"available={sorted(obj.keys())}"
        )

    a = to_numpy(
        obj[
            future_key
        ]
    )

    if a is None:
        raise RuntimeError(
            f"{future_key} is not tensor/ndarray"
        )

    # Only validate explicit trailing dimensions.
    #
    # For scalar per-material fields such as depth and
    # projection_valid, last_dims == (), and Python's
    # a.shape[-0:] incorrectly means the ENTIRE shape.
    if (
        len(last_dims) > 0
        and tuple(
            a.shape[
                -len(last_dims):
            ]
        ) != tuple(
            last_dims
        )
    ):
        raise RuntimeError(
            f"{name} unexpected shape {a.shape}"
        )

    if a.shape[0] >= 165:
        return a[:165]

    if a.shape[0] == 164:

        init = None

        for key in initial_keys:

            if key not in obj:
                continue

            b = to_numpy(
                obj[key]
            )

            if b is not None and tuple(
                b.shape
            ) == tuple(
                a.shape[1:]
            ):
                init = b
                break

        if init is None:
            raise RuntimeError(
                f"{name} has 164 future states "
                "but no accepted initial state; "
                f"keys={sorted(obj.keys())}"
            )

        return np.concatenate(
            [
                init[
                    None
                ],
                a,
            ],
            axis=0,
        )

    raise RuntimeError(
        f"{name} has unsupported temporal "
        f"shape {a.shape}"
    )


uv_full = full_sequence(
    sand,
    "points_uv",
    [
        "initial_points_uv",
        "points_uv_initial",
    ],
    (
        2,
    ),
    "points_uv",
).astype(np.float32)


N = uv_full.shape[1]

if N < 100:
    raise RuntimeError(
        f"unexpected SandHouse point count N={N}"
    )


# ---------- depth ----------

depth_full = full_sequence(
    sand,
    "depth",
    [
        "initial_depth",
        "depth_initial",
        "initial_points_depth",
    ],
    tuple(),
    "depth",
).astype(np.float32)

if depth_full.shape != (
    165,
    N,
):
    raise RuntimeError(
        f"SandHouse depth shape "
        f"{depth_full.shape}, expected (165,{N})"
    )


# ---------- projection validity ----------

projection_full = full_sequence(
    sand,
    "projection_valid",
    [
        "initial_projection_valid",
        "projection_valid_initial",
    ],
    tuple(),
    "projection_valid",
).astype(bool)

if projection_full.shape != (
    165,
    N,
):
    raise RuntimeError(
        f"SandHouse projection_valid "
        f"shape {projection_full.shape}"
    )


# ============================================================
# Coordinate-system contract
# ============================================================

coordinate_system = str(
    raw.get(
        "coordinate_system",
        ""
    )
)

image_size = raw.get(
    "image_size",
    None,
)

if (
    "realwonder_512" not in coordinate_system
    and image_size != 512
):
    raise RuntimeError(
        "cannot establish raw 512x512 UV contract: "
        f"coordinate_system={coordinate_system!r}, "
        f"image_size={image_size!r}"
    )


xy_full = uv_full.copy()

xy_full[
    ...,
    0,
] *= (
    832.0
    / 512.0
)

xy_full[
    ...,
    1,
] = (
    xy_full[
        ...,
        1,
    ]
    * (
        832.0
        / 512.0
    )
    - 176.0
)


# ============================================================
# Authoritative visibility
#
# Projection-only visibility is NOT accepted.
# First inspect the raw sand object.
# If unavailable, use transport_ready_165f only if its ordering
# can be proven against the raw trajectories.
# ============================================================

visibility_full = None
visibility_source = None

# --------------------------------------------------------
# SandHouse held-out visibility contract.
#
# The cached transport-ready artifact does not contain an
# explicit 165xN point-visibility tensor.  Use the frozen,
# independently constructed native-camera z-buffer contract:
#
# front-most persistent material point per 512x512 camera
# raster pixel, minimum positive camera depth, exact depth
# ties by minimum persistent material ID.
# --------------------------------------------------------

zbuffer_visibility_path = (
    SANDROOT
    / "sandhouse_raw512_zbuffer_visibility_165.npy"
)

if zbuffer_visibility_path.is_file():

    visibility_full = np.load(
        zbuffer_visibility_path
    ).astype(bool)

    if visibility_full.shape != (
        165,
        N,
    ):
        raise RuntimeError(
            "SandHouse z-buffer visibility shape "
            f"{visibility_full.shape}, expected "
            f"(165,{N})"
        )

    visibility_source = str(
        zbuffer_visibility_path
    )

    ordering_proof = {
        "kind":
            "same_raw_persistent_point_order",

        "definition":
            "native 512x512 camera z-buffer from "
            "the same raw points_uv/depth/projection_valid",

        "tie_break":
            "minimum persistent raw material ID",
    }

    print(
        "USING_SANDHOUSE_RAW512_ZBUFFER_VISIBILITY =",
        zbuffer_visibility_path,
        flush=True,
    )


for key in [
    "aligned_visible",
    "raster_visible",
    "visibility",
    "visible",
]:

    if key not in sand:
        continue

    a = to_numpy(
        sand[key]
    )

    if a is None:
        continue

    if a.shape == (
        165,
        N,
    ):
        visibility_full = (
            a.astype(bool)
        )

        visibility_source = (
            f"{raw_path}::objects[{sand_obj_index}]::{key}"
        )

        break


def recursive_candidates(
    obj,
    names,
    path="root",
):

    found = []

    if isinstance(
        obj,
        dict,
    ):
        for k, v in obj.items():

            p = f"{path}.{k}"

            if k in names:
                a = to_numpy(v)

                if a is not None:
                    found.append(
                        (
                            p,
                            a,
                        )
                    )

            if isinstance(
                v,
                dict,
            ):
                found.extend(
                    recursive_candidates(
                        v,
                        names,
                        p,
                    )
                )

            elif isinstance(
                v,
                (list, tuple),
            ):
                for i, x in enumerate(v):
                    if isinstance(
                        x,
                        dict,
                    ):
                        found.extend(
                            recursive_candidates(
                                x,
                                names,
                                f"{p}[{i}]",
                            )
                        )

    return found


transport_path = None


if visibility_full is None:

    preferred_transport = (
        SANDROOT
        / "transport_ready_165f.pt"
    )

    if preferred_transport.is_file():
        transport_path = preferred_transport

    else:
        hits = sorted(
            SANDROOT.rglob(
                "transport_ready_165f.pt"
            )
        )

        if len(hits) != 1:
            raise RuntimeError(
                "SandHouse has no raw raster visibility "
                "and transport_ready_165f.pt is not uniquely "
                "resolvable:\n"
                + "\n".join(
                    str(p)
                    for p in hits
                )
            )

        transport_path = hits[0]


    print(
        "LOADING_SANDHOUSE_TRANSPORT_READY =",
        transport_path,
        flush=True,
    )

    tr = torch.load(
        transport_path,
        map_location="cpu",
        weights_only=False,
    )


    vis_candidates = recursive_candidates(
        tr,
        {
            "aligned_visible",
            "raster_visible",
            "visibility",
            "visible",
        },
    )


    matched_vis = [
        (
            p,
            a,
        )
        for p, a in vis_candidates
        if a.shape == (
            165,
            N,
        )
    ]


    if len(
        matched_vis
    ) != 1:

        raise RuntimeError(
            "cannot uniquely find authoritative "
            "165xN SandHouse visibility in "
            f"{transport_path}; "
            f"matches="
            f"{[(p,a.shape) for p,a in matched_vis]}"
        )


    vis_name, vis_array = (
        matched_vis[0]
    )


    # --------------------------------------------------------
    # Prove transport-ready point ordering.
    # Prefer 2D coordinate agreement.
    # --------------------------------------------------------

    point_candidates = recursive_candidates(
        tr,
        {
            "points_2d_video",
            "points_video",
            "tracks",
            "points_uv",
        },
    )


    ordering_proved = False
    ordering_proof = None


    rng = np.random.default_rng(
        20260810
    )

    sample_n = min(
        2000,
        N,
    )

    sample_ids = np.sort(
        rng.choice(
            N,
            size=sample_n,
            replace=False,
        )
    )

    sample_frames = [
        0,
        80,
        160,
    ]


    for name, a in point_candidates:

        if a.shape != (
            165,
            N,
            2,
        ):
            continue

        aa = a.astype(
            np.float32,
            copy=False,
        )

        err_aligned = float(
            np.nanmax(
                np.abs(
                    aa[
                        sample_frames
                    ][
                        :,
                        sample_ids
                    ]
                    -
                    xy_full[
                        sample_frames
                    ][
                        :,
                        sample_ids
                    ]
                )
            )
        )

        err_raw = float(
            np.nanmax(
                np.abs(
                    aa[
                        sample_frames
                    ][
                        :,
                        sample_ids
                    ]
                    -
                    uv_full[
                        sample_frames
                    ][
                        :,
                        sample_ids
                    ]
                )
            )
        )

        if min(
            err_aligned,
            err_raw,
        ) <= 1e-3:

            ordering_proved = True

            ordering_proof = {
                "kind":
                    "2d_coordinate_ordering",

                "field":
                    name,

                "max_abs_err_aligned":
                    err_aligned,

                "max_abs_err_raw":
                    err_raw,
            }

            break


    # If 2D points are not stored, accept exact binding agreement.
    if not ordering_proved:

        raw_binding = None

        for key in [
            "binding_particle_indices",
            "point_particle_binding",
        ]:

            if key in sand:
                b = to_numpy(
                    sand[key]
                )

                if (
                    b is not None
                    and b.shape[0] == N
                ):
                    raw_binding = b
                    break


        binding_candidates = recursive_candidates(
            tr,
            {
                "binding_particle_indices",
                "point_particle_binding",
            },
        )


        if raw_binding is not None:

            for name, b in binding_candidates:

                if b.shape != raw_binding.shape:
                    continue

                if np.array_equal(
                    b[
                        sample_ids
                    ],
                    raw_binding[
                        sample_ids
                    ],
                ):

                    ordering_proved = True

                    ordering_proof = {
                        "kind":
                            "binding_ordering",

                        "field":
                            name,

                        "sample_count":
                            int(sample_n),
                    }

                    break


    if not ordering_proved:

        raise RuntimeError(
            "SandHouse authoritative visibility was found, "
            "but its point ordering cannot be proven against "
            "raw persistent trajectories. "
            "Refusing to guess alignment."
        )


    visibility_full = (
        vis_array.astype(bool)
    )

    visibility_source = (
        f"{transport_path}::{vis_name}"
    )


else:

    ordering_proof = {
        "kind":
            "same_raw_object",
    }


# ============================================================
# Final physical visibility
# ============================================================

finite = np.isfinite(
    xy_full
).all(
    axis=-1
)

in_frame = (
    (
        xy_full[
            ...,
            0
        ]
        >= 0
    )
    &
    (
        xy_full[
            ...,
            0
        ]
        < 832
    )
    &
    (
        xy_full[
            ...,
            1
        ]
        >= 0
    )
    &
    (
        xy_full[
            ...,
            1
        ]
        < 480
    )
)

visibility_full = (
    visibility_full
    &
    projection_full
    &
    finite
    &
    in_frame
)


# ============================================================
# Sample frozen 81 SandHouse states
# ============================================================

tracks_81_all = (
    xy_full[
        SAND_FRAME_IDS
    ]
    .astype(
        np.float32
    )
)

vis_81_all = (
    visibility_full[
        SAND_FRAME_IDS
    ]
    .astype(bool)
)

depth_81_all = (
    depth_full[
        SAND_FRAME_IDS
    ]
    .astype(
        np.float32
    )
)


# ============================================================
# Source point selection:
# one persistent point / occupied 8x8 source cell
# in aligned 832x480 domain.
# ============================================================

source_ids = np.flatnonzero(
    vis_81_all[
        0
    ]
)

if len(
    source_ids
) == 0:

    raise RuntimeError(
        "SandHouse frame0 has zero authoritative "
        "source-visible material points"
    )


xy0 = tracks_81_all[
    0,
    source_ids,
]

cell_x = np.floor(
    xy0[
        :,
        0
    ]
    / 8
).astype(
    np.int64
)

cell_y = np.floor(
    xy0[
        :,
        1
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
            :,
            0
        ]
        - center_x
    ) ** 2
    +
    (
        xy0[
            :,
            1
        ]
        - center_y
    ) ** 2
)


# persistent material ID = raw point index
order = np.lexsort(
    (
        source_ids,
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
    first[
        1:
    ] = (
        sorted_cells[
            1:
        ]
        !=
        sorted_cells[
            :-1
        ]
    )


selected_ids = (
    source_ids[
        order[
            first
        ]
    ]
    .astype(
        np.int64
    )
)


tracks = (
    tracks_81_all[
        :,
        selected_ids,
    ]
    .copy()
)

visibility = (
    vis_81_all[
        :,
        selected_ids,
    ]
    .copy()
)

depth = (
    depth_81_all[
        :,
        selected_ids,
    ]
    .copy()
)


# ============================================================
# Fill coordinates only where invisible.
# Visibility remains authoritative.
# ============================================================

for n in range(
    tracks.shape[1]
):

    good = (
        visibility[
            :,
            n
        ]
        &
        np.isfinite(
            tracks[
                :,
                n
            ]
        ).all(
            axis=1
        )
    )

    ids_good = np.flatnonzero(
        good
    )

    if len(
        ids_good
    ) == 0:
        raise RuntimeError(
            f"selected SandHouse point "
            f"{selected_ids[n]} has no valid state"
        )

    first_good = int(
        ids_good[
            0
        ]
    )

    tracks[
        :first_good,
        n,
    ] = tracks[
        first_good,
        n,
    ]

    last = first_good

    for t in range(
        first_good,
        81,
    ):

        if good[
            t
        ]:
            last = t

        else:
            tracks[
                t,
                n,
            ] = tracks[
                last,
                n,
            ]


if not np.isfinite(
    tracks
).all():
    raise RuntimeError(
        "SandHouse exported tracks still contain nonfinite values"
    )


if not visibility[
    0
].all():
    raise RuntimeError(
        "SandHouse selected source tracks must all be visible at t0"
    )


visible_depth = depth[
    visibility
]

if (
    visible_depth.size == 0
    or not np.isfinite(
        visible_depth
    ).all()
):

    raise RuntimeError(
        "SandHouse visible depth invalid"
    )


positive_fraction = float(
    (
        visible_depth
        > 0
    ).mean()
)

if positive_fraction < 0.999:

    raise RuntimeError(
        "SandHouse camera depth is not consistently "
        f"positive: {positive_fraction}"
    )


tracks_out = tracks[
    None
].astype(
    np.float32
)

vis_out = visibility[
    None
].astype(bool)

depth_out = depth[
    None
].astype(
    np.float32
)

ids_out = selected_ids.astype(
    np.int64
)


# Correct + Identity-Shuffled
shuffled_out, sand_perm, sand_attempts = (
    build_shuffled(
        tracks_out
    )
)


sand_out = (
    ART
    / "sandhouse"
)

sand_out.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    sand_out
    / "sandhouse_v3d_correct_tracks.npy",
    tracks_out,
)

np.save(
    sand_out
    / "sandhouse_v3d_identity_shuffled_tracks.npy",
    shuffled_out,
)

np.save(
    sand_out
    / "sandhouse_v3d_visibility.npy",
    vis_out,
)

np.save(
    sand_out
    / "sandhouse_v3d_depth.npy",
    depth_out,
)

np.save(
    sand_out
    / "sandhouse_v3d_ids.npy",
    ids_out,
)

np.save(
    sand_out
    / "sandhouse_identity_shuffle_perm.npy",
    sand_perm,
)

np.save(
    sand_out
    / "sandhouse_frame_ids_165_to_81.npy",
    SAND_FRAME_IDS,
)


(
    sand_out
    / "input_image_path.txt"
).write_text(
    str(
        image_path
    )
    + "\n"
)

(
    sand_out
    / "prompt_path.txt"
).write_text(
    str(
        prompt_path
    )
    + "\n"
)


report[
    "sandhouse"
] = {
    "status":
        "SANDHOUSE_HELDOUT_BRIDGE_GO",

    "sand_root":
        str(
            SANDROOT
        ),

    "raw_trajectory":
        str(
            raw_path
        ),

    "material_type":
        material_type,

    "raw_point_count":
        int(
            N
        ),

    "frame_ids":
        SAND_FRAME_IDS.tolist(),

    "temporal_rule":
        "uniform raw state stride 2: 0,2,...,160",

    "visibility_source":
        visibility_source,

    "visibility_ordering_proof":
        ordering_proof,

    "input_image":
        str(
            image_path
        ),

    "prompt":
        str(
            prompt_path
        ),

    "source_visible_points":
        int(
            vis_81_all[
                0
            ].sum()
        ),

    "selected_tracks":
        int(
            len(
                selected_ids
            )
        ),

    "source_sampling":
        "one point per occupied aligned-832x480 8x8 cell; "
        "nearest cell center; tie persistent raw point ID",

    "visible_depth_positive_fraction":
        positive_fraction,

    "derangement_seed":
        0,

    "derangement_attempts":
        int(
            sand_attempts
        ),

    "fixed_points":
        int(
            np.sum(
                sand_perm
                == np.arange(
                    len(
                        sand_perm
                    )
                )
            )
        ),

    "future_tracks_bit_identical":
        bool(
            np.array_equal(
                shuffled_out[
                    0,
                    1:
                ],
                tracks_out[
                    0,
                    1:
                ],
            )
        ),
}


report_path = (
    ART
    / "validation_artifact_report.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n"
)


# Input checksums.
inputs = [
    image_path,
    prompt_path,
    raw_path,
]

if transport_path is not None:
    inputs.append(
        transport_path
    )


with (
    ART
    / "input_sha256.txt"
).open(
    "w"
) as f:

    for p in inputs:

        f.write(
            f"{sha256(p)}  {p}\n"
        )


print(
    json.dumps(
        report,
        indent=2,
    )
)

print(
    "V3D_VALIDATION_ARTIFACT_PREFLIGHT_OK"
)
