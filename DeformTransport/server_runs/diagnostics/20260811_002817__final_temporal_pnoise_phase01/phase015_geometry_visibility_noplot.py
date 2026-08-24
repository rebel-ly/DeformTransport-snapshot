import json
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path("/workspace/DeformTransport")
RUN = Path.cwd()

EVAL = ROOT / "server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/eval_v3.py"

spec = importlib.util.spec_from_file_location("ev", EVAL)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


# ============================================================
# PART A — TREE 36–56 DYNAMIC EVENT AUDIT
# ============================================================

cfg = ev.CASES["tree"]

tracks = np.load(
    ROOT / cfg["tracks"]
)[0].astype(np.float64)

vis = np.load(
    ROOT / cfg["vis"]
)[0].astype(bool)

# 416 x 240 evaluation coordinates
xy = tracks / 2.0

T, N, _ = xy.shape


# ------------------------------------------------------------
# Fixed source kNN graph
# ------------------------------------------------------------

p0 = xy[0]

D = np.sum(
    (p0[:, None, :] - p0[None, :, :]) ** 2,
    axis=-1,
)

np.fill_diagonal(D, np.inf)

knn = np.argsort(D, axis=1)[:, :6]

edges = set()

for i in range(N):
    for j in knn[i]:
        edges.add(
            tuple(
                sorted(
                    (i, int(j))
                )
            )
        )

edges = np.asarray(
    sorted(edges),
    dtype=np.int64,
)

l0 = np.linalg.norm(
    p0[edges[:, 0]]
    - p0[edges[:, 1]],
    axis=1,
)

l0 = np.maximum(
    l0,
    1e-4,
)


rows = []

for t in range(1, T):

    v = (
        xy[t]
        - xy[t - 1]
    )

    speed = np.linalg.norm(
        v,
        axis=1,
    )

    accel_mean = None
    reversal_fraction = None

    if t >= 2:

        vp = (
            xy[t - 1]
            - xy[t - 2]
        )

        prev_speed = np.linalg.norm(
            vp,
            axis=1,
        )

        accel = np.linalg.norm(
            v - vp,
            axis=1,
        )

        accel_mean = float(
            accel.mean()
        )

        good = (
            (speed > 0.05)
            &
            (prev_speed > 0.05)
        )

        if good.any():

            cos = (
                np.sum(
                    v[good]
                    * vp[good],
                    axis=1,
                )
                /
                (
                    speed[good]
                    * prev_speed[good]
                    + 1e-12
                )
            )

            reversal_fraction = float(
                np.mean(
                    cos < 0
                )
            )

        else:
            reversal_fraction = 0.0


    lt = np.linalg.norm(
        xy[t, edges[:, 0]]
        - xy[t, edges[:, 1]],
        axis=1,
    )

    deformation = float(
        np.median(
            np.abs(
                np.log(
                    (lt + 1e-4)
                    / l0
                )
            )
        )
    )

    churn = float(
        np.mean(
            vis[t]
            != vis[t - 1]
        )
    )

    rows.append({
        "frame":
            int(t),

        "mean_step_px":
            float(
                speed.mean()
            ),

        "median_step_px":
            float(
                np.median(speed)
            ),

        "mean_accel_px":
            accel_mean,

        "reversal_fraction":
            reversal_fraction,

        "knn_log_distortion":
            deformation,

        "visible_fraction":
            float(
                vis[t].mean()
            ),

        "visibility_churn":
            churn,
    })


def finite_mean(items, key):
    vals = [
        x[key]
        for x in items
        if x[key] is not None
        and np.isfinite(x[key])
    ]

    return (
        float(np.mean(vals))
        if vals else None
    )


def top_frames(key, k=10):
    a = [
        x
        for x in rows
        if x[key] is not None
        and np.isfinite(x[key])
    ]

    a = sorted(
        a,
        key=lambda z: z[key],
        reverse=True,
    )

    return [
        {
            "frame":
                x["frame"],

            key:
                x[key],
        }
        for x in a[:k]
    ]


event = [
    x
    for x in rows
    if 36 <= x["frame"] <= 56
]

outside = [
    x
    for x in rows
    if not (
        36 <= x["frame"] <= 56
    )
]

metrics = [
    "mean_step_px",
    "mean_accel_px",
    "reversal_fraction",
    "knn_log_distortion",
    "visibility_churn",
]


tree_summary = {
    "event_window":
        [36, 56],

    "n_tracks":
        int(N),

    "source_knn_edges":
        int(len(edges)),

    "event_means":
        {},

    "outside_means":
        {},

    "event_over_outside_ratio":
        {},

    "top_frames":
        {},
}


for key in metrics:

    a = finite_mean(
        event,
        key,
    )

    b = finite_mean(
        outside,
        key,
    )

    tree_summary[
        "event_means"
    ][key] = a

    tree_summary[
        "outside_means"
    ][key] = b

    tree_summary[
        "event_over_outside_ratio"
    ][key] = (
        float(a / b)
        if a is not None
        and b is not None
        and abs(b) > 1e-12
        else None
    )

    tree_summary[
        "top_frames"
    ][key] = top_frames(key)


tree_summary["selected_frames"] = {
    str(t):
        rows[t - 1]
    for t in [
        32,
        36,
        40,
        44,
        48,
        52,
        56,
        60,
    ]
}


# ============================================================
# PART B — SANTA VISIBILITY PROVENANCE
# ============================================================

cfg = ev.CASES["santa"]

santa_tracks = np.load(
    ROOT / cfg["tracks"]
)[0].astype(np.float32)

bridge_vis = np.load(
    ROOT / cfg["vis"]
)[0].astype(bool)


visibility_summary = {
    "bridge_shape":
        list(
            bridge_vis.shape
        ),

    "bridge_true_fraction":
        float(
            bridge_vis.mean()
        ),

    "bridge_all_true":
        bool(
            bridge_vis.all()
        ),

    "per_frame_true_fraction":
        [
            float(x)
            for x
            in bridge_vis.mean(
                axis=1
            )
        ],

    "candidate_assets":
        [],
}


candidates = [
    ROOT
    / "server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_contract_20260806_192643/outputs/aligned_visibility_contract.pt"
]


for p in ROOT.glob(
    "server_runs/**/artifacts/santa/*visibility*.npy"
):
    candidates.append(p)


def collect_arrays(
    obj,
    name="root",
):
    out = []

    if torch.is_tensor(obj):
        out.append(
            (
                name,
                obj.detach()
                .cpu()
                .numpy(),
            )
        )

    elif isinstance(
        obj,
        np.ndarray,
    ):
        out.append(
            (
                name,
                obj,
            )
        )

    elif isinstance(
        obj,
        dict,
    ):
        for k, v in obj.items():
            out.extend(
                collect_arrays(
                    v,
                    name
                    + "."
                    + str(k),
                )
            )

    elif isinstance(
        obj,
        (list, tuple),
    ):
        for i, v in enumerate(obj):
            out.extend(
                collect_arrays(
                    v,
                    name
                    + f"[{i}]",
                )
            )

    return out


seen = set()

for p in candidates:

    p = Path(p)

    if (
        not p.exists()
        or str(p) in seen
    ):
        continue

    seen.add(str(p))

    rec = {
        "path":
            str(p),

        "arrays":
            [],
    }

    try:

        if p.suffix == ".npy":
            obj = np.load(
                p,
                allow_pickle=False,
            )

        else:
            obj = torch.load(
                p,
                map_location="cpu",
            )

        arrays = collect_arrays(obj)

        for name, a in arrays:

            a = np.asarray(a)

            rr = {
                "name":
                    name,

                "shape":
                    list(
                        a.shape
                    ),

                "dtype":
                    str(
                        a.dtype
                    ),
            }

            if a.size:

                try:
                    rr["min"] = float(
                        np.nanmin(a)
                    )

                    rr["max"] = float(
                        np.nanmax(a)
                    )

                except Exception:
                    pass


                unique = np.unique(a)

                if (
                    a.dtype == bool
                    or
                    (
                        len(unique) <= 2
                        and
                        np.all(
                            np.isin(
                                unique,
                                [0, 1],
                            )
                        )
                    )
                ):

                    b = a.astype(bool)

                    rr[
                        "true_fraction"
                    ] = float(
                        b.mean()
                    )


                if (
                    a.shape
                    == bridge_vis.shape
                ):

                    b = a.astype(bool)

                    rr[
                        "same_shape_as_bridge"
                    ] = True

                    rr[
                        "equal_to_bridge"
                    ] = bool(
                        np.array_equal(
                            b,
                            bridge_vis,
                        )
                    )

                    rr[
                        "difference_fraction"
                    ] = float(
                        np.mean(
                            b
                            != bridge_vis
                        )
                    )

            rec[
                "arrays"
            ].append(rr)

    except Exception as e:

        rec["error"] = repr(e)

    visibility_summary[
        "candidate_assets"
    ].append(rec)


# ============================================================
# PART C — SAVE SANTA VISIBILITY OVERLAY WITHOUT MATPLOTLIB
# ============================================================

video = ev.read_video_common(
    ROOT / cfg["rw"]
)

frames = [
    0,
    16,
    32,
    48,
    64,
    80,
]

panels = []

for t in frames:

    im = np.asarray(
        video[t]
    ).copy()

    if im.dtype != np.uint8:

        if im.max() <= 1.5:
            im = np.clip(
                im * 255,
                0,
                255,
            ).astype(np.uint8)

        else:
            im = np.clip(
                im,
                0,
                255,
            ).astype(np.uint8)


    # RGB -> BGR for OpenCV
    im = cv2.cvtColor(
        im,
        cv2.COLOR_RGB2BGR,
    )


    pts = santa_tracks[t].copy()

    pts[:, 1] *= (
        464.0
        / 480.0
    )


    # Sample every 8th track
    for i in range(
        0,
        len(pts),
        8,
    ):

        x, y = pts[i]

        if (
            np.isfinite(x)
            and np.isfinite(y)
            and
            0 <= x < 832
            and
            0 <= y < 464
        ):

            color = (
                (0, 255, 0)
                if bridge_vis[t, i]
                else
                (0, 0, 255)
            )

            cv2.circle(
                im,
                (
                    int(round(x)),
                    int(round(y)),
                ),
                2,
                color,
                -1,
            )


    text = (
        f"t={t}  "
        f"visible={bridge_vis[t].mean():.3f}"
    )

    cv2.putText(
        im,
        text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    panels.append(im)


top = np.hstack(
    panels[:3]
)

bottom = np.hstack(
    panels[3:]
)

montage = np.vstack(
    [top, bottom]
)

overlay_path = (
    RUN
    / "santa_visibility_overlay.png"
)

cv2.imwrite(
    str(overlay_path),
    montage,
)


# ============================================================
# SAVE
# ============================================================

report = {
    "tree_dynamic_event":
        tree_summary,

    "santa_visibility":
        visibility_summary,

    "santa_overlay":
        str(overlay_path),
}


out = (
    RUN
    / "phase015_geometry_visibility.json"
)

out.write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n"
)


print(
    "===== TREE EVENT 36-56 ====="
)

print(
    json.dumps(
        tree_summary,
        indent=2,
    )
)

print()
print(
    "===== SANTA VISIBILITY ====="
)

print(
    "bridge_all_true =",
    visibility_summary[
        "bridge_all_true"
    ]
)

print(
    "bridge_true_fraction =",
    visibility_summary[
        "bridge_true_fraction"
    ]
)

for rec in visibility_summary[
    "candidate_assets"
]:

    print()
    print(
        rec["path"]
    )

    if "error" in rec:
        print(
            " ERROR:",
            rec["error"],
        )
        continue

    for a in rec[
        "arrays"
    ]:

        if (
            "true_fraction" in a
            or
            a.get(
                "same_shape_as_bridge",
                False,
            )
        ):
            print(
                " ",
                a,
            )


print()
print(
    "OVERLAY:",
    overlay_path,
)

print(
    "SAVED:",
    out,
)
