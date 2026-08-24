#!/usr/bin/env bash
set -euo pipefail

DT=/workspace/DeformTransport
PY=/workspace/tools/miniforge3/envs/wan-move/bin/python

SUITE=/workspace/DeformTransport/server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0

cd "$DT" || exit 1

test -x "$PY" || {
    echo "ERROR: missing $PY"
    exit 2
}

test -d "$SUITE" || {
    echo "ERROR: missing suite $SUITE"
    exit 2
}

STAMP=$(date +%Y%m%d_%H%M%S)

RUN="$DT/server_runs/wan_move_method_eval/${STAMP}__v3s_v3b_v3c_v3d_v3e_joint_eval"

mkdir -p "$RUN"

echo "$RUN" > \
"$DT/server_runs/wan_move_method_eval/current_v3_joint_eval.txt"


# ============================================================
# 0. Freeze evaluation/selection protocol
# ============================================================

cat > "$RUN/PROTOCOL_FROZEN.txt" <<'EOF'
Development cases:
Santa + Tree only.

SandHouse remains held-out and is NOT used for V3 method selection.

Candidate set:
V3S
V3B
V3C
V3D
V3E

Reference for V3 development:
Old Correct Wan-Move seed0.

RealWonder:
reported as a system-level baseline only.
It is NOT used to select the V3 winner.


APPEARANCE = frozen TC-MAR

- common future-video domain: 832x464
- RealWonder 832x480:
  torch bicubic -> 464x832
  align_corners=False
  antialias=False
- source appearance:
  authoritative aligned input image in 832x480
- patch:
  8x8
  offsets -3.5 ... +3.5
  exact bilinear sampling
- anchors:
  4,8,...,80
- evaluation support:
  authoritative FULL Correct material tracks + visibility
  identical support for every candidate
- aggregation:
  observation -> material-track mean
- paired bootstrap:
  unit = material track
  n = 10000
  seed = 0
- lower TC-MAR is better


MOTION = frozen TC-ME

- generated video motion:
  torchvision RAFT-Large C_T_SKHT_V2
- common video:
  832x464
- RAFT input:
  torch area -> 240x416
  then official RAFT transforms
- physics reference:
  DIRECT persistent material displacement
  NOT simulation RAFT

  q_i,t = (x_i,t / 2, y_i,t / 2)

  d_i,t =
  (P_i,t+1 - P_i,t) / 2

- validity:
  visibility[t] & visibility[t+1]
- evaluation support:
  identical across methods
- transition metric:
  mean EPE
- paired bootstrap:
  unit = transition
  n = 10000
  seed = 0
- lower TC-ME is better


FROZEN METHOD SELECTION

Step 1: Motion safety.

For each case compute:

candidate TC-ME - OldCorrect TC-ME

If lower bound of 95% CI > 0:
candidate has significant motion regression
and is rejected.


Step 2: Appearance.

For every motion-safe candidate compute:

OldCorrect TC-MAR Lab - candidate TC-MAR Lab

IMPROVE:
95% CI lower > 0

REGRESS:
95% CI upper < 0

otherwise:
TIE


Step 3: Priority.

Tier 3:
Santa IMPROVE
Tree IMPROVE

Tier 2:
one IMPROVE
one TIE

Tier 1:
Santa TIE
Tree TIE

Any significant appearance REGRESS:
reject.


Step 4:
within the same tier only,
tie-break using unweighted mean relative
TC-MAR Lab improvement over Santa + Tree.


Step 5:
if no candidate survives,
report NO_WINNER.

Do NOT introduce another metric after observing results.
EOF


# ============================================================
# 1. Unified evaluator
# ============================================================

cat > "$RUN/eval_v3.py" <<'PYV3'
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


BOOT_SEED = 0
BOOT_N = 10000

ANCHORS = list(
    range(
        4,
        81,
        4,
    )
)

OFF = (
    np.arange(
        8,
        dtype=np.float32,
    )
    - 3.5
)

CANDIDATES = [
    "v3s",
    "v3b",
    "v3c",
    "v3d",
    "v3e",
]


CASES = {
    "santa": {
        "source":
            "server_runs/"
            "20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/"
            "official_santa_81f_aligned_final_sim_20260806_234410/"
            "resized_input_image.png",

        "rw":
            "server_runs/"
            "20260804_234925_autonomous_deformtransport/"
            "12_soft_transport_dev/"
            "20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/"
            "baseline/"
            "aligned_santa_baseline_seed0.mp4",

        "old_correct":
            "server_runs/"
            "wan_move_formal/"
            "20260809_195255__santa_correct_vs_identity_shuffled_seed0/"
            "correct/"
            "santa_formal_correct_seed0.mp4",

        "tracks":
            "server_runs/"
            "wan_move_bridge/"
            "20260809_010015__santa_correct_tracks/"
            "santa_material_tracks_correct.npy",

        "vis":
            "server_runs/"
            "wan_move_bridge/"
            "20260809_010015__santa_correct_tracks/"
            "santa_material_visibility_correct.npy",

        "expect": {
            "n": 1277,
            "valid_tracks": 1277,
            "obs": 25540,
            "lab": 21.271405488892093,
            "rgb": 0.14561786886907022,
            "tcme": 0.6146068110130727,
        },
    },

    "tree": {
        "source":
            "server_runs/"
            "20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/"
            "tree_official_precomputed_aligned_final_sim_20260807_185055/"
            "resized_input_image.png",

        "rw":
            "server_runs/"
            "20260804_234925_autonomous_deformtransport/"
            "12_soft_transport_dev/"
            "20260807_203228__tree__realwonder_baseline_seed0/"
            "tree_realwonder_baseline_seed0.mp4",

        "old_correct":
            "server_runs/"
            "wan_move_formal/"
            "20260810_073902__tree_correct_vs_identity_shuffled_seed0/"
            "correct/"
            "tree_formal_correct_seed0.mp4",

        "tracks":
            "server_runs/"
            "wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_tracks_correct.npy",

        "vis":
            "server_runs/"
            "wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_visibility_correct.npy",

        "expect": {
            "n": 713,
            "valid_tracks": 709,
            "obs": 8743,
            "lab": 18.96363288940394,
            "rgb": 0.14480022402703113,
            "tcme": 0.3678485654760152,
        },
    },
}


def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for block in iter(
            lambda: f.read(
                1 << 20
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def candidate_path(
    suite,
    case,
    var,
):

    return (
        suite
        / case
        / var
        / f"{case}_{var}_correct_seed0.mp4"
    )


def method_paths(
    root,
    suite,
    case,
):

    cfg = CASES[
        case
    ]

    paths = {
        "rw":
            root
            / cfg["rw"],

        "old_correct":
            root
            / cfg["old_correct"],
    }

    for var in CANDIDATES:

        paths[var] = (
            candidate_path(
                suite,
                case,
                var,
            )
        )

    return paths


def read_rgb_image(
    path,
):

    bgr = cv2.imread(
        str(
            path
        ),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise RuntimeError(
            f"cannot read image {path}"
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return (
        rgb.astype(
            np.float32
        )
        / 255.0
    )


def to_common(
    rgb_u8,
):

    h, w = rgb_u8.shape[
        :2
    ]

    x = (
        torch
        .from_numpy(
            rgb_u8
        )
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(
            0
        )
        .float()
        / 255.0
    )

    if (
        h,
        w,
    ) == (
        480,
        832,
    ):

        x = F.interpolate(
            x,
            size=(
                464,
                832,
            ),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        ).clamp_(
            0,
            1,
        )

    elif (
        h,
        w,
    ) != (
        464,
        832,
    ):

        raise AssertionError(
            (
                h,
                w,
            )
        )

    return (
        x[
            0
        ]
        .permute(
            1,
            2,
            0,
        )
        .numpy()
        .astype(
            np.float32
        )
    )


def read_video_common(
    path,
):

    cap = cv2.VideoCapture(
        str(
            path
        )
    )

    frames = []

    while True:

        ok, bgr = cap.read()

        if not ok:
            break

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(
            to_common(
                rgb
            )
        )

    cap.release()

    if len(
        frames
    ) != 81:

        raise AssertionError(
            f"{path}: "
            f"frames={len(frames)}"
        )

    return np.stack(
        frames,
        axis=0,
    )


def sample_patches(
    img,
    centers,
):

    H, W = img.shape[
        :2
    ]

    centers = np.asarray(
        centers,
        np.float32,
    )

    xs = (
        centers[
            :,
            0,
            None,
            None,
        ]
        + OFF[
            None,
            None,
            :,
        ]
    )

    ys = (
        centers[
            :,
            1,
            None,
            None,
        ]
        + OFF[
            None,
            :,
            None,
        ]
    )

    xs = np.broadcast_to(
        xs,
        (
            len(
                centers
            ),
            8,
            8,
        ),
    )

    ys = np.broadcast_to(
        ys,
        (
            len(
                centers
            ),
            8,
            8,
        ),
    )

    valid = (
        (
            xs.min(
                (
                    1,
                    2,
                )
            )
            >= 0
        )
        &
        (
            ys.min(
                (
                    1,
                    2,
                )
            )
            >= 0
        )
        &
        (
            xs.max(
                (
                    1,
                    2,
                )
            )
            <= W - 1
        )
        &
        (
            ys.max(
                (
                    1,
                    2,
                )
            )
            <= H - 1
        )
    )

    if not np.all(
        valid
    ):
        raise RuntimeError(
            "sample_patches invalid centers"
        )

    x0 = np.floor(
        xs
    ).astype(
        np.int64
    )

    y0 = np.floor(
        ys
    ).astype(
        np.int64
    )

    x1 = np.minimum(
        x0 + 1,
        W - 1,
    )

    y1 = np.minimum(
        y0 + 1,
        H - 1,
    )

    wx = (
        xs
        - x0
    )[
        ...,
        None,
    ]

    wy = (
        ys
        - y0
    )[
        ...,
        None,
    ]

    Ia = img[
        y0,
        x0,
    ]

    Ib = img[
        y0,
        x1,
    ]

    Ic = img[
        y1,
        x0,
    ]

    Id = img[
        y1,
        x1,
    ]

    return (
        (
            1 - wx
        )
        *
        (
            1 - wy
        )
        *
        Ia

        +

        wx
        *
        (
            1 - wy
        )
        *
        Ib

        +

        (
            1 - wx
        )
        *
        wy
        *
        Ic

        +

        wx
        *
        wy
        *
        Id
    ).astype(
        np.float32
    )


def patch_mean_lab(
    patches,
):

    n = patches.shape[
        0
    ]

    flat = (
        patches
        .reshape(
            n * 8,
            8,
            3,
        )
        .astype(
            np.float32
        )
    )

    lab = cv2.cvtColor(
        flat,
        cv2.COLOR_RGB2LAB,
    )

    return (
        lab
        .reshape(
            n,
            8,
            8,
            3,
        )
        .mean(
            axis=(
                1,
                2,
            )
        )
    )


def stats(
    x,
):

    x = np.asarray(
        x,
        float,
    )

    return {
        "n":
            int(
                len(
                    x
                )
            ),

        "mean":
            float(
                x.mean()
            ),

        "median":
            float(
                np.median(
                    x
                )
            ),

        "p95":
            float(
                np.percentile(
                    x,
                    95,
                )
            ),
    }


def bootstrap_mean_ci(
    diff,
):

    d = np.asarray(
        diff,
        np.float64,
    )

    n = len(
        d
    )

    rng = np.random.default_rng(
        BOOT_SEED
    )

    values = []

    for start in range(
        0,
        BOOT_N,
        500,
    ):

        k = min(
            500,
            BOOT_N - start,
        )

        idx = rng.integers(
            0,
            n,
            size=(
                k,
                n,
            ),
        )

        values.append(
            d[
                idx
            ].mean(
                axis=1
            )
        )

    values = np.concatenate(
        values
    )

    return [
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    ]


def aggregate(
    rows,
    n,
):

    sums = np.zeros(
        n,
        np.float64,
    )

    count = np.zeros(
        n,
        np.int64,
    )

    for ids, vals in rows:

        np.add.at(
            sums,
            ids,
            vals,
        )

        np.add.at(
            count,
            ids,
            1,
        )

    result = np.full(
        n,
        np.nan,
        np.float64,
    )

    valid = (
        count
        > 0
    )

    result[
        valid
    ] = (
        sums[
            valid
        ]
        /
        count[
            valid
        ]
    )

    return (
        result,
        count,
    )


# ============================================================
# TC-MAR
# ============================================================

def appearance_case(
    root,
    suite,
    case,
):

    cfg = CASES[
        case
    ]

    tracks = np.load(
        root
        / cfg["tracks"]
    )[
        0
    ].astype(
        np.float32
    )

    visibility = np.load(
        root
        / cfg["vis"]
    )[
        0
    ].astype(
        bool
    )

    n = tracks.shape[
        1
    ]

    assert (
        tracks.shape
        == (
            81,
            n,
            2,
        )
    )

    assert (
        visibility.shape
        == (
            81,
            n,
        )
    )

    assert (
        n
        == cfg["expect"]["n"]
    )

    source = read_rgb_image(
        root
        / cfg["source"]
    )

    src_centers = tracks[
        0
    ]

    src_valid = (
        (
            src_centers[
                :,
                0,
            ]
            - 3.5
            >= 0
        )
        &
        (
            src_centers[
                :,
                0,
            ]
            + 3.5
            <= 831
        )
        &
        (
            src_centers[
                :,
                1,
            ]
            - 3.5
            >= 0
        )
        &
        (
            src_centers[
                :,
                1,
            ]
            + 3.5
            <= 479
        )
    )

    src_patch = np.full(
        (
            n,
            8,
            8,
            3,
        ),
        np.nan,
        np.float32,
    )

    good = np.where(
        src_valid
    )[
        0
    ]

    src_patch[
        good
    ] = sample_patches(
        source,
        src_centers[
            good
        ],
    )

    src_lab = np.full(
        (
            n,
            3,
        ),
        np.nan,
        np.float32,
    )

    src_lab[
        good
    ] = patch_mean_lab(
        src_patch[
            good
        ]
    )

    paths = method_paths(
        root,
        suite,
        case,
    )

    per_method = {}

    counts_ref = None
    total_obs_ref = None

    for method, path in paths.items():

        print(
            "APPEARANCE LOAD",
            case,
            method,
            path,
            flush=True,
        )

        video = read_video_common(
            path
        )

        lab_rows = []
        rgb_rows = []

        total_obs = 0

        for t in ANCHORS:

            centers = tracks[
                t
            ].copy()

            centers[
                :,
                1,
            ] *= (
                464.0
                / 480.0
            )

            future_valid = (
                (
                    centers[
                        :,
                        0,
                    ]
                    - 3.5
                    >= 0
                )
                &
                (
                    centers[
                        :,
                        0,
                    ]
                    + 3.5
                    <= 831
                )
                &
                (
                    centers[
                        :,
                        1,
                    ]
                    - 3.5
                    >= 0
                )
                &
                (
                    centers[
                        :,
                        1,
                    ]
                    + 3.5
                    <= 463
                )
            )

            valid = (
                visibility[
                    t
                ]
                &
                src_valid
                &
                future_valid
                &
                np.isfinite(
                    centers
                ).all(
                    axis=1
                )
            )

            ids = np.where(
                valid
            )[
                0
            ]

            total_obs += len(
                ids
            )

            patch = sample_patches(
                video[
                    t
                ],
                centers[
                    ids
                ],
            )

            lab = np.linalg.norm(
                patch_mean_lab(
                    patch
                )
                - src_lab[
                    ids
                ],
                axis=1,
            )

            rgb = np.abs(
                patch
                - src_patch[
                    ids
                ]
            ).mean(
                axis=(
                    1,
                    2,
                    3,
                )
            )

            lab_rows.append(
                (
                    ids,
                    lab,
                )
            )

            rgb_rows.append(
                (
                    ids,
                    rgb,
                )
            )

        lab, count = aggregate(
            lab_rows,
            n,
        )

        rgb, count2 = aggregate(
            rgb_rows,
            n,
        )

        assert np.array_equal(
            count,
            count2,
        )

        if counts_ref is None:

            counts_ref = count
            total_obs_ref = total_obs

        else:

            assert np.array_equal(
                counts_ref,
                count,
            )

            assert (
                total_obs
                == total_obs_ref
            )

        per_method[
            method
        ] = {
            "lab":
                lab,

            "rgb":
                rgb,

            "count":
                count,
        }

        del video

    valid_tracks = (
        counts_ref
        > 0
    )

    exp = cfg[
        "expect"
    ]

    if (
        int(
            valid_tracks.sum()
        )
        != exp[
            "valid_tracks"
        ]
        or
        int(
            total_obs_ref
        )
        != exp[
            "obs"
        ]
    ):

        raise AssertionError(
            (
                case,
                valid_tracks.sum(),
                total_obs_ref,
                exp,
            )
        )

    old_lab = float(
        per_method[
            "old_correct"
        ][
            "lab"
        ][
            valid_tracks
        ].mean()
    )

    old_rgb = float(
        per_method[
            "old_correct"
        ][
            "rgb"
        ][
            valid_tracks
        ].mean()
    )

    if (
        abs(
            old_lab
            - exp["lab"]
        )
        > 0.002
        or
        abs(
            old_rgb
            - exp["rgb"]
        )
        > 0.002
    ):

        raise AssertionError(
            f"{case}: "
            "frozen appearance reproduction mismatch: "
            f"lab={old_lab} "
            f"rgb={old_rgb}"
        )

    report = {
        "case":
            case,

        "metric":
            "Track-Conditioned Material Appearance Retention (TC-MAR)",

        "anchors":
            ANCHORS,

        "valid_tracks":
            int(
                valid_tracks.sum()
            ),

        "total_valid_anchor_observations":
            int(
                total_obs_ref
            ),

        "methods":
            {},

        "vs_old_correct":
            {},

        "vs_realwonder":
            {},
    }

    for method in paths:

        report[
            "methods"
        ][
            method
        ] = {
            "tc_mar_lab":
                stats(
                    per_method[
                        method
                    ][
                        "lab"
                    ][
                        valid_tracks
                    ]
                ),

            "tc_mar_rgb_l1":
                stats(
                    per_method[
                        method
                    ][
                        "rgb"
                    ][
                        valid_tracks
                    ]
                ),

            "sha256":
                sha256(
                    paths[
                        method
                    ]
                ),
        }

    old = per_method[
        "old_correct"
    ][
        "lab"
    ][
        valid_tracks
    ]

    rw = per_method[
        "rw"
    ][
        "lab"
    ][
        valid_tracks
    ]

    for var in CANDIDATES:

        candidate = per_method[
            var
        ][
            "lab"
        ][
            valid_tracks
        ]

        diff = (
            old
            - candidate
        )

        ci = bootstrap_mean_ci(
            diff
        )

        if ci[
            0
        ] > 0:

            status = "IMPROVE"

        elif ci[
            1
        ] < 0:

            status = "REGRESS"

        else:

            status = "TIE"

        report[
            "vs_old_correct"
        ][
            var
        ] = {
            "definition":
                "OldCorrect TC-MAR Lab - candidate; "
                "positive means candidate better",

            "paired_mean_difference":
                float(
                    diff.mean()
                ),

            "paired_median_difference":
                float(
                    np.median(
                        diff
                    )
                ),

            "fraction_candidate_better":
                float(
                    np.mean(
                        diff
                        > 0
                    )
                ),

            "bootstrap_95_ci":
                ci,

            "status":
                status,

            "relative_mean_improvement":
                float(
                    (
                        old.mean()
                        - candidate.mean()
                    )
                    /
                    old.mean()
                ),
        }

        rw_diff = (
            rw
            - candidate
        )

        rw_ci = bootstrap_mean_ci(
            rw_diff
        )

        report[
            "vs_realwonder"
        ][
            var
        ] = {
            "definition":
                "RealWonder TC-MAR Lab - candidate; "
                "positive means candidate better",

            "paired_mean_difference":
                float(
                    rw_diff.mean()
                ),

            "bootstrap_95_ci":
                rw_ci,

            "decision":
                (
                    "WIN"
                    if rw_ci[0] > 0
                    else
                    (
                        "LOSS"
                        if rw_ci[1] < 0
                        else
                        "TIE"
                    )
                ),
        }

    return report


# ============================================================
# TC-ME
# ============================================================

def bilinear_flow(
    flow,
    centers,
):

    H, W = flow.shape[
        1:
    ]

    centers = np.asarray(
        centers,
        np.float32,
    )

    x = centers[
        :,
        0
    ]

    y = centers[
        :,
        1
    ]

    if (
        np.any(
            x < 0
        )
        or
        np.any(
            x > W - 1
        )
        or
        np.any(
            y < 0
        )
        or
        np.any(
            y > H - 1
        )
    ):
        raise RuntimeError(
            "flow sample out of bounds"
        )

    x0 = np.floor(
        x
    ).astype(
        np.int64
    )

    y0 = np.floor(
        y
    ).astype(
        np.int64
    )

    x1 = np.minimum(
        x0 + 1,
        W - 1,
    )

    y1 = np.minimum(
        y0 + 1,
        H - 1,
    )

    wx = (
        x
        - x0
    )

    wy = (
        y
        - y0
    )

    f = flow.transpose(
        1,
        2,
        0,
    )

    Ia = f[
        y0,
        x0,
    ]

    Ib = f[
        y0,
        x1,
    ]

    Ic = f[
        y1,
        x0,
    ]

    Id = f[
        y1,
        x1,
    ]

    return (
        (
            1 - wx
        )[
            :,
            None
        ]
        *
        (
            1 - wy
        )[
            :,
            None
        ]
        *
        Ia

        +

        wx[
            :,
            None
        ]
        *
        (
            1 - wy
        )[
            :,
            None
        ]
        *
        Ib

        +

        (
            1 - wx
        )[
            :,
            None
        ]
        *
        wy[
            :,
            None
        ]
        *
        Ic

        +

        wx[
            :,
            None
        ]
        *
        wy[
            :,
            None
        ]
        *
        Id
    )


def load_raft_cached(
    device,
):

    from torchvision.models.optical_flow import (
        raft_large,
        Raft_Large_Weights,
    )

    weights = (
        Raft_Large_Weights
        .C_T_SKHT_V2
    )

    cache = (
        Path(
            torch.hub.get_dir()
        )
        / "checkpoints"
        / os.path.basename(
            weights.url
        )
    )

    print(
        "RAFT_EXPECTED_CACHE",
        cache,
        flush=True,
    )

    if not cache.exists():

        raise RuntimeError(
            f"RAFT weight not cached: "
            f"{cache}"
        )

    model = (
        raft_large(
            weights=weights,
            progress=False,
        )
        .eval()
        .to(
            device
        )
    )

    return (
        model,
        weights.transforms(),
    )


def motion_case(
    root,
    suite,
    case,
    batch,
):

    cfg = CASES[
        case
    ]

    tracks = np.load(
        root
        / cfg["tracks"]
    )[
        0
    ].astype(
        np.float32
    )

    visibility = np.load(
        root
        / cfg["vis"]
    )[
        0
    ].astype(
        bool
    )

    references = []

    for t in range(
        80
    ):

        valid = (
            visibility[
                t
            ]
            &
            visibility[
                t + 1
            ]
            &
            np.isfinite(
                tracks[
                    t
                ]
            ).all(
                axis=1
            )
            &
            np.isfinite(
                tracks[
                    t + 1
                ]
            ).all(
                axis=1
            )
        )

        ids = np.where(
            valid
        )[
            0
        ]

        centers = (
            tracks[
                t,
                ids,
            ]
            / 2.0
        )

        ref = (
            (
                tracks[
                    t + 1,
                    ids,
                ]
                -
                tracks[
                    t,
                    ids,
                ]
            )
            / 2.0
        )

        inbound = (
            (
                centers[
                    :,
                    0
                ]
                >= 0
            )
            &
            (
                centers[
                    :,
                    0
                ]
                <= 415
            )
            &
            (
                centers[
                    :,
                    1
                ]
                >= 0
            )
            &
            (
                centers[
                    :,
                    1
                ]
                <= 239
            )
        )

        centers = centers[
            inbound
        ]

        ref = ref[
            inbound
        ]

        if len(
            centers
        ) == 0:

            raise RuntimeError(
                f"{case}: transition "
                f"{t} zero support"
            )

        references.append(
            (
                centers,
                ref,
            )
        )

    device = torch.device(
        "cuda:0"
    )

    model, transforms = (
        load_raft_cached(
            device
        )
    )

    paths = method_paths(
        root,
        suite,
        case,
    )

    per_method = {}

    for method, path in paths.items():

        print(
            "MOTION LOAD",
            case,
            method,
            path,
            flush=True,
        )

        video = read_video_common(
            path
        )

        values = []

        with torch.inference_mode():

            for start in range(
                0,
                80,
                batch,
            ):

                end = min(
                    80,
                    start
                    + batch,
                )

                a = (
                    torch
                    .from_numpy(
                        video[
                            start:end
                        ]
                    )
                    .permute(
                        0,
                        3,
                        1,
                        2,
                    )
                    .to(
                        device
                    )
                )

                b = (
                    torch
                    .from_numpy(
                        video[
                            start + 1:
                            end + 1
                        ]
                    )
                    .permute(
                        0,
                        3,
                        1,
                        2,
                    )
                    .to(
                        device
                    )
                )

                a = F.interpolate(
                    a,
                    size=(
                        240,
                        416,
                    ),
                    mode="area",
                )

                b = F.interpolate(
                    b,
                    size=(
                        240,
                        416,
                    ),
                    mode="area",
                )

                a, b = transforms(
                    a,
                    b,
                )

                prediction = (
                    model(
                        a,
                        b,
                    )[
                        -1
                    ]
                    .float()
                    .cpu()
                    .numpy()
                )

                for j, t in enumerate(
                    range(
                        start,
                        end,
                    )
                ):

                    centers, ref = (
                        references[
                            t
                        ]
                    )

                    pred_flow = (
                        bilinear_flow(
                            prediction[
                                j
                            ],
                            centers,
                        )
                    )

                    epe = np.linalg.norm(
                        pred_flow
                        - ref,
                        axis=1,
                    )

                    values.append(
                        float(
                            epe.mean()
                        )
                    )

                print(
                    "RAFT",
                    case,
                    method,
                    end,
                    "/80",
                    flush=True,
                )

        per_method[
            method
        ] = np.asarray(
            values,
            np.float64,
        )

        del video

        torch.cuda.empty_cache()

    del model

    torch.cuda.empty_cache()

    observed_old = float(
        per_method[
            "old_correct"
        ].mean()
    )

    expected_old = cfg[
        "expect"
    ][
        "tcme"
    ]

    if abs(
        observed_old
        - expected_old
    ) > 0.005:

        raise AssertionError(
            f"{case}: frozen TC-ME "
            f"reproduction mismatch "
            f"{observed_old} "
            f"expected {expected_old}"
        )

    report = {
        "case":
            case,

        "metric":
            "Track-Conditioned Motion Error (TC-ME)",

        "methods":
            {},

        "vs_old_correct":
            {},

        "vs_realwonder":
            {},
    }

    for method, values in (
        per_method.items()
    ):

        report[
            "methods"
        ][
            method
        ] = {
            "transition_mean_epe_mean":
                float(
                    values.mean()
                ),

            "transition_mean_epe_median":
                float(
                    np.median(
                        values
                    )
                ),

            "transition_mean_epe_p95":
                float(
                    np.percentile(
                        values,
                        95,
                    )
                ),

            "sha256":
                sha256(
                    paths[
                        method
                    ]
                ),
        }

    old = per_method[
        "old_correct"
    ]

    rw = per_method[
        "rw"
    ]

    for var in CANDIDATES:

        candidate = per_method[
            var
        ]

        diff = (
            candidate
            - old
        )

        ci = bootstrap_mean_ci(
            diff
        )

        report[
            "vs_old_correct"
        ][
            var
        ] = {
            "definition":
                "candidate TC-ME - OldCorrect TC-ME; "
                "positive means candidate worse",

            "paired_mean_difference":
                float(
                    diff.mean()
                ),

            "paired_median_difference":
                float(
                    np.median(
                        diff
                    )
                ),

            "fraction_candidate_worse":
                float(
                    np.mean(
                        diff
                        > 0
                    )
                ),

            "bootstrap_95_ci":
                ci,

            "motion_safety":
                (
                    "REGRESSION_STOP"
                    if ci[0] > 0
                    else
                    "PASS"
                ),
        }

        rw_diff = (
            rw
            - candidate
        )

        rw_ci = bootstrap_mean_ci(
            rw_diff
        )

        report[
            "vs_realwonder"
        ][
            var
        ] = {
            "definition":
                "RealWonder TC-ME - candidate; "
                "positive means candidate better",

            "paired_mean_difference":
                float(
                    rw_diff.mean()
                ),

            "bootstrap_95_ci":
                rw_ci,

            "decision":
                (
                    "WIN"
                    if rw_ci[0] > 0
                    else
                    (
                        "LOSS"
                        if rw_ci[1] < 0
                        else
                        "TIE"
                    )
                ),
        }

    return report


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--suite",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "appearance",
            "motion",
        ],
        required=True,
    )

    parser.add_argument(
        "--case",
        choices=[
            "santa",
            "tree",
        ],
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    )

    suite = Path(
        args.suite
    )

    out = Path(
        args.out
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "appearance":

        report = {
            "protocol":
                "frozen TC-MAR "
                "development evaluation",

            "cases":
                {},
        }

        for case in [
            "santa",
            "tree",
        ]:

            print(
                "=== APPEARANCE",
                case,
                "===",
                flush=True,
            )

            report[
                "cases"
            ][
                case
            ] = appearance_case(
                root,
                suite,
                case,
            )

        (
            out
            / "appearance_report.json"
        ).write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n"
        )

        print(
            "APPEARANCE_EVAL_DONE",
            flush=True,
        )

    else:

        if args.case is None:

            raise RuntimeError(
                "--case required "
                "for motion"
            )

        print(
            "=== MOTION",
            args.case,
            "===",
            flush=True,
        )

        report = motion_case(
            root,
            suite,
            args.case,
            args.batch,
        )

        (
            out
            / f"{args.case}_motion_report.json"
        ).write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n"
        )

        print(
            "MOTION_EVAL_DONE",
            args.case,
            flush=True,
        )


if __name__ == "__main__":
    main()
PYV3


# ============================================================
# 2. Frozen selector
# ============================================================

cat > "$RUN/select_v3.py" <<'PYSEL'
import json
import sys
from pathlib import Path


CANDIDATES = [
    "v3s",
    "v3b",
    "v3c",
    "v3d",
    "v3e",
]

CASES = [
    "santa",
    "tree",
]


out = Path(
    sys.argv[
        1
    ]
)

appearance = json.loads(
    (
        out
        / "appearance_report.json"
    ).read_text()
)

motion = {
    case:
        json.loads(
            (
                out
                / f"{case}_motion_report.json"
            ).read_text()
        )
    for case in CASES
}


rows = []


for candidate in CANDIDATES:

    motion_status = {
        case:
            motion[
                case
            ][
                "vs_old_correct"
            ][
                candidate
            ][
                "motion_safety"
            ]
        for case in CASES
    }

    appearance_status = {
        case:
            appearance[
                "cases"
            ][
                case
            ][
                "vs_old_correct"
            ][
                candidate
            ][
                "status"
            ]
        for case in CASES
    }

    motion_safe = all(
        status == "PASS"
        for status in (
            motion_status.values()
        )
    )

    appearance_regression = any(
        status == "REGRESS"
        for status in (
            appearance_status.values()
        )
    )

    improve_count = sum(
        status == "IMPROVE"
        for status in (
            appearance_status.values()
        )
    )

    tie_count = sum(
        status == "TIE"
        for status in (
            appearance_status.values()
        )
    )

    if (
        not motion_safe
        or
        appearance_regression
    ):

        tier = -1
        eligible = False

    elif improve_count == 2:

        tier = 3
        eligible = True

    elif (
        improve_count == 1
        and
        tie_count == 1
    ):

        tier = 2
        eligible = True

    elif tie_count == 2:

        tier = 1
        eligible = True

    else:

        tier = -1
        eligible = False

    relative = [
        appearance[
            "cases"
        ][
            case
        ][
            "vs_old_correct"
        ][
            candidate
        ][
            "relative_mean_improvement"
        ]
        for case in CASES
    ]

    row = {
        "candidate":
            candidate,

        "motion_safety":
            motion_status,

        "appearance_status":
            appearance_status,

        "eligible":
            eligible,

        "tier":
            tier,

        "mean_relative_tc_mar_lab_improvement":
            sum(
                relative
            )
            / 2.0,

        "appearance": {
            case: {
                "old_correct":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "methods"
                    ][
                        "old_correct"
                    ][
                        "tc_mar_lab"
                    ][
                        "mean"
                    ],

                "candidate":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "methods"
                    ][
                        candidate
                    ][
                        "tc_mar_lab"
                    ][
                        "mean"
                    ],

                "old_minus_candidate_ci":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "vs_old_correct"
                    ][
                        candidate
                    ][
                        "bootstrap_95_ci"
                    ],

                "vs_rw":
                    appearance[
                        "cases"
                    ][
                        case
                    ][
                        "vs_realwonder"
                    ][
                        candidate
                    ][
                        "decision"
                    ],
            }

            for case in CASES
        },

        "motion": {
            case: {
                "old_correct":
                    motion[
                        case
                    ][
                        "methods"
                    ][
                        "old_correct"
                    ][
                        "transition_mean_epe_mean"
                    ],

                "candidate":
                    motion[
                        case
                    ][
                        "methods"
                    ][
                        candidate
                    ][
                        "transition_mean_epe_mean"
                    ],

                "candidate_minus_old_ci":
                    motion[
                        case
                    ][
                        "vs_old_correct"
                    ][
                        candidate
                    ][
                        "bootstrap_95_ci"
                    ],

                "vs_rw":
                    motion[
                        case
                    ][
                        "vs_realwonder"
                    ][
                        candidate
                    ][
                        "decision"
                    ],
            }

            for case in CASES
        },
    }

    rows.append(
        row
    )


eligible = [
    row
    for row in rows
    if row[
        "eligible"
    ]
]


if eligible:

    winner = max(
        eligible,
        key=lambda row: (
            row[
                "tier"
            ],
            row[
                "mean_relative_tc_mar_lab_improvement"
            ],
        ),
    )[
        "candidate"
    ]

    decision = (
        "WINNER_SELECTED"
    )

else:

    winner = None
    decision = (
        "NO_WINNER"
    )


report = {
    "decision":
        decision,

    "winner":
        winner,

    "selection_basis":
        "Santa+Tree development only; "
        "motion safety -> "
        "TC-MAR Lab significance tier -> "
        "mean relative TC-MAR Lab "
        "improvement tie-break",

    "sandhouse_used_for_selection":
        False,

    "candidates":
        rows,
}


(
    out
    / "selection_report.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n"
)


with (
    out
    / "selection_summary.tsv"
).open(
    "w"
) as f:

    f.write(
        "candidate\t"
        "motion_santa\t"
        "motion_tree\t"
        "app_santa\t"
        "app_tree\t"
        "eligible\t"
        "tier\t"
        "mean_rel_tc_mar\n"
    )

    for row in rows:

        f.write(
            f"{row['candidate']}\t"
            f"{row['motion_safety']['santa']}\t"
            f"{row['motion_safety']['tree']}\t"
            f"{row['appearance_status']['santa']}\t"
            f"{row['appearance_status']['tree']}\t"
            f"{row['eligible']}\t"
            f"{row['tier']}\t"
            f"{row['mean_relative_tc_mar_lab_improvement']:.8f}\n"
        )


print(
    "=== FROZEN V3 SELECTION ==="
)

for row in rows:

    print(
        row[
            "candidate"
        ],

        "motion",
        row[
            "motion_safety"
        ],

        "appearance",
        row[
            "appearance_status"
        ],

        "eligible",
        row[
            "eligible"
        ],

        "tier",
        row[
            "tier"
        ],

        "mean_rel",
        f"{row['mean_relative_tc_mar_lab_improvement']:.6f}",
    )


print(
    "DECISION=",
    decision,
)

print(
    "WINNER=",
    winner,
)
PYSEL


# ============================================================
# 3. Syntax validation
# ============================================================

"$PY" -m py_compile \
"$RUN/eval_v3.py" \
"$RUN/select_v3.py"

echo "EVALUATOR_COMPILE_OK"


# ============================================================
# 4. Verify all ten V3 generations again
# ============================================================

for CASE in santa tree
do

    for VAR in \
    v3s \
    v3b \
    v3c \
    v3d \
    v3e
    do

        VIDEO="$SUITE/$CASE/$VAR/${CASE}_${VAR}_correct_seed0.mp4"

        test -s "$VIDEO" || {
            echo "ERROR missing video:"
            echo "$VIDEO"
            exit 10
        }

        EC=$(
            cat \
            "$SUITE/$CASE/$VAR/exit_code.txt"
        )

        test "$EC" = "0" || {
            echo "ERROR nonzero generation exit:"
            echo "$CASE $VAR"
            exit 11
        }

    done
done

echo "ALL_10_INPUT_GENERATIONS_OK"


# ============================================================
# 5. RAFT cache check
# ============================================================

"$PY" - <<'PYRAFTCHECK'
import os
import torch

from pathlib import Path
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
)


weights = (
    Raft_Large_Weights
    .C_T_SKHT_V2
)

path = (
    Path(
        torch.hub.get_dir()
    )
    / "checkpoints"
    / os.path.basename(
        weights.url
    )
)

print(
    "RAFT_CACHE=",
    path
)

if not path.exists():

    raise SystemExit(
        "RAFT_WEIGHT_NOT_CACHED"
    )

print(
    "RAFT_CACHE_OK"
)
PYRAFTCHECK


# ============================================================
# 6. Provenance
# ============================================================

(
    cd "$DT"

    git rev-parse HEAD \
    > "$RUN/deformtransport_git_head.txt"

    git status --short \
    > "$RUN/deformtransport_git_status.txt"
) || true


(
    cd /workspace/Wan-Move

    git rev-parse HEAD \
    > "$RUN/wanmove_git_head.txt"

    git status --short \
    > "$RUN/wanmove_git_status.txt"
) || true


sha256sum \
"$RUN/eval_v3.py" \
"$RUN/select_v3.py" \
"$RUN/PROTOCOL_FROZEN.txt" \
> "$RUN/evaluator_sha256.txt"


# ============================================================
# 7. GPU1 / GPU2 check
# ============================================================

echo
echo "========== GPU PREFLIGHT =========="

nvidia-smi \
--query-gpu=index,memory.used,memory.total,utilization.gpu \
--format=csv,noheader


mapfile -t G < <(
    nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
)


okgpu () {

    local idx="$1"
    local min_free="$2"

    local line
    local total
    local used
    local free

    line=$(
        nvidia-smi \
        --query-gpu=index,memory.total,memory.used \
        --format=csv,noheader,nounits \
        | awk -F',' -v i="$idx" \
          '$1+0==i {print $0}'
    )

    total=$(
        echo "$line" \
        | awk -F',' \
          '{gsub(/ /,"",$2); print $2}'
    )

    used=$(
        echo "$line" \
        | awk -F',' \
          '{gsub(/ /,"",$3); print $3}'
    )

    if [ -z "$total" ] || [ -z "$used" ]; then
        return 1
    fi

    free=$((total - used))

    echo \
    "GPU${idx}: total=${total}MiB used=${used}MiB free=${free}MiB"

    [ "$free" -ge "$min_free" ]
}


# RAFT evaluation policy:
# GPU1 may already host a light workload, but must retain
# >=20 GiB free memory.
# GPU2 must also retain >=20 GiB free memory.
#
# Compute contention can affect runtime, not metric definition.
if ! okgpu 1 20480 || ! okgpu 2 20480
then

    echo
    echo "ERROR:"
    echo "GPU1/2 are not both free."
    echo "Evaluation NOT launched."
    echo
    echo "RUN=$RUN"

    exit 20
fi


echo "GPU1_GPU2_FREE_OK"


# ============================================================
# 8. Supervisor
#
# CPU:
#   appearance Santa+Tree
#
# GPU1:
#   Santa TC-ME
#
# GPU2:
#   Tree TC-ME
#
# Selection automatically starts only after all 3 succeed.
# ============================================================

cat > "$RUN/supervisor.sh" <<'SUP'
#!/usr/bin/env bash

set -u

DT=/workspace/DeformTransport

PY=/workspace/tools/miniforge3/envs/wan-move/bin/python

RUN="$(cd "$(dirname "$0")" && pwd)"

SUITE=/workspace/DeformTransport/server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0


echo "[$(date -Iseconds)] START joint evaluation"


# ------------------------------------------------------------
# CPU appearance
# ------------------------------------------------------------

"$PY" \
"$RUN/eval_v3.py" \
--root "$DT" \
--suite "$SUITE" \
--out "$RUN" \
--mode appearance \
> "$RUN/appearance_stdout.log" \
2> "$RUN/appearance_stderr.log" &

P1=$!


# ------------------------------------------------------------
# GPU1 Santa motion
# ------------------------------------------------------------

CUDA_VISIBLE_DEVICES=1 \
"$PY" \
"$RUN/eval_v3.py" \
--root "$DT" \
--suite "$SUITE" \
--out "$RUN" \
--mode motion \
--case santa \
--batch 8 \
> "$RUN/santa_motion_stdout.log" \
2> "$RUN/santa_motion_stderr.log" &

P2=$!


# ------------------------------------------------------------
# GPU2 Tree motion
# ------------------------------------------------------------

CUDA_VISIBLE_DEVICES=2 \
"$PY" \
"$RUN/eval_v3.py" \
--root "$DT" \
--suite "$SUITE" \
--out "$RUN" \
--mode motion \
--case tree \
--batch 8 \
> "$RUN/tree_motion_stdout.log" \
2> "$RUN/tree_motion_stderr.log" &

P3=$!


printf '%s\n' "$P1" \
> "$RUN/appearance_pid.txt"

printf '%s\n' "$P2" \
> "$RUN/santa_motion_pid.txt"

printf '%s\n' "$P3" \
> "$RUN/tree_motion_pid.txt"


# ------------------------------------------------------------
# Wait for all three.
# ------------------------------------------------------------

wait "$P1"
E1=$?

wait "$P2"
E2=$?

wait "$P3"
E3=$?


echo "$E1" \
> "$RUN/appearance_exit_code.txt"

echo "$E2" \
> "$RUN/santa_motion_exit_code.txt"

echo "$E3" \
> "$RUN/tree_motion_exit_code.txt"


# ------------------------------------------------------------
# Frozen selector
# ------------------------------------------------------------

if \
    [ "$E1" -eq 0 ] \
    && \
    [ "$E2" -eq 0 ] \
    && \
    [ "$E3" -eq 0 ]
then

    "$PY" \
    "$RUN/select_v3.py" \
    "$RUN" \
    > "$RUN/selection_stdout.log" \
    2> "$RUN/selection_stderr.log"

    ES=$?

    echo "$ES" \
    > "$RUN/selection_exit_code.txt"

    if [ "$ES" -eq 0 ]
    then

        date -Iseconds \
        > "$RUN/EVAL_DONE.txt"

        echo \
        "[$(date -Iseconds)] EVAL_DONE"

        cat \
        "$RUN/selection_stdout.log"

        exit 0
    fi
fi


date -Iseconds \
> "$RUN/EVAL_FAILED.txt"

echo \
"[$(date -Iseconds)] EVAL_FAILED appearance=$E1 santa_motion=$E2 tree_motion=$E3"

exit 1
SUP


chmod +x \
"$RUN/supervisor.sh"


# ============================================================
# 9. Launch
# ============================================================

nohup bash \
"$RUN/supervisor.sh" \
> "$RUN/supervisor.log" \
2>&1 \
< /dev/null &

PID=$!

echo "$PID" \
> "$RUN/supervisor_pid.txt"


echo
echo "============================================"
echo "V3_JOINT_EVAL_LAUNCHED"
echo "============================================"

echo "RUN=$RUN"
echo "SUPERVISOR_PID=$PID"

echo
echo "CPU:"
echo "Santa + Tree TC-MAR"

echo
echo "GPU1:"
echo "Santa RAFT TC-ME"

echo
echo "GPU2:"
echo "Tree RAFT TC-ME"

echo
echo "Frozen selector will run automatically"
echo "after all three evaluations finish."

echo "============================================"

echo
echo "Monitor:"
echo
echo "tail -f \"$RUN/supervisor.log\""

echo
echo "Santa motion:"
echo
echo "tail -f \"$RUN/santa_motion_stdout.log\""

echo
echo "Tree motion:"
echo
echo "tail -f \"$RUN/tree_motion_stdout.log\""

echo
echo "Final result:"
echo
echo "cat \"$RUN/selection_stdout.log\""
echo
echo "cat \"$RUN/selection_summary.tsv\""
