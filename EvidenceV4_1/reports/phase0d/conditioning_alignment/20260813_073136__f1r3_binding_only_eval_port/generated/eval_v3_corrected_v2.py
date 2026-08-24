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

CANDIDATES = ['dt_full']


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
            '/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy',

        "vis":
            '/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy',

        "expect": {
            "n": 1257,
            "valid_tracks": 1110,
            "obs": 9766,
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

        for case in ['santa']:

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
