import sys
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/DeformTransport")
LAUNCH = ROOT / "server_runs/system_eval_launcher_20260810"
OLD = ROOT / "server_runs/system_eval/20260810_013925__santa_tree_rw_vs_wanmove_manual"

sys.path.insert(0, str(LAUNCH))

from system_cpu import (
    CASES,
    ANCHORS,
    read_rgb_image,
    read_video_common,
    sample_patches,
    patch_mean_lab,
    bootstrap_mean_ci,
    stats,
)

BATCH = Path(
    open(
        ROOT / "server_runs/wan_move_method_dev/current_v3a_batch.txt"
    ).read().strip()
)

OUT = Path(
    open(
        ROOT / "server_runs/wan_move_method_dev/current_v3a_eval.txt"
    ).read().strip()
)

V3A = {
    "santa": BATCH / "santa/santa_v3a_collision_mean_correct_seed0.mp4",
    "tree": BATCH / "tree/tree_v3a_collision_mean_correct_seed0.mp4",
}


def load_old_per_track(case):
    p = OLD / f"{case}_per_track.csv"

    rows = {}

    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            i = int(r["track_id"])
            rows[i] = {
                "old_correct_lab": float(r["correct_lab"]),
                "rw_lab": float(r["rw_lab"]),
                "old_correct_rgb": float(r["correct_rgb_l1"]),
                "rw_rgb": float(r["rw_rgb_l1"]),
                "valid_anchor_count": int(r["valid_anchor_count"]),
            }

    return rows


def eval_case(case, cfg):

    source = read_rgb_image(ROOT / cfg["source"])

    tracks = np.load(
        ROOT / cfg["tracks"]
    )[0].astype(np.float32)

    vis = np.load(
        ROOT / cfg["vis"]
    )[0].astype(bool)

    video = read_video_common(
        V3A[case]
    )

    n = tracks.shape[1]

    p0 = tracks[0].copy()

    source_valid = (
        np.isfinite(p0).all(axis=1)
        & (p0[:,0] - 3.5 >= 0)
        & (p0[:,0] + 3.5 <= 831)
        & (p0[:,1] - 3.5 >= 0)
        & (p0[:,1] + 3.5 <= 479)
    )

    ids0 = np.where(source_valid)[0]

    src_patch = np.full(
        (n,8,8,3),
        np.nan,
        dtype=np.float32,
    )

    src_patch[ids0] = sample_patches(
        source,
        p0[ids0],
    )

    src_lab = np.full(
        (n,3),
        np.nan,
        dtype=np.float32,
    )

    src_lab[ids0] = patch_mean_lab(
        src_patch[ids0]
    )

    lab_sum = np.zeros(n, dtype=np.float64)
    rgb_sum = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)

    for t in ANCHORS:

        centers = tracks[t].copy()
        centers[:,1] *= 464.0 / 480.0

        future_valid = (
            np.isfinite(centers).all(axis=1)
            & (centers[:,0] - 3.5 >= 0)
            & (centers[:,0] + 3.5 <= 831)
            & (centers[:,1] - 3.5 >= 0)
            & (centers[:,1] + 3.5 <= 463)
        )

        valid = (
            source_valid
            & future_valid
            & vis[t]
        )

        ids = np.where(valid)[0]

        patch = sample_patches(
            video[t],
            centers[ids],
        )

        lab = patch_mean_lab(patch)

        lab_err = np.linalg.norm(
            lab - src_lab[ids],
            axis=1,
        )

        rgb_err = np.abs(
            patch - src_patch[ids]
        ).mean(axis=(1,2,3))

        np.add.at(lab_sum, ids, lab_err)
        np.add.at(rgb_sum, ids, rgb_err)
        np.add.at(count, ids, 1)

    valid = count > 0

    v3a_lab = np.full(n, np.nan)
    v3a_rgb = np.full(n, np.nan)

    v3a_lab[valid] = lab_sum[valid] / count[valid]
    v3a_rgb[valid] = rgb_sum[valid] / count[valid]

    old = load_old_per_track(case)

    ids = np.array(
        sorted(old.keys()),
        dtype=int,
    )

    assert np.all(valid[ids])

    old_lab = np.array([
        old[i]["old_correct_lab"]
        for i in ids
    ])

    rw_lab = np.array([
        old[i]["rw_lab"]
        for i in ids
    ])

    old_rgb = np.array([
        old[i]["old_correct_rgb"]
        for i in ids
    ])

    rw_rgb = np.array([
        old[i]["rw_rgb"]
        for i in ids
    ])

    new_lab = v3a_lab[ids]
    new_rgb = v3a_rgb[ids]

    d_app_lab = old_lab - new_lab
    d_app_rgb = old_rgb - new_rgb

    d_rw_lab = rw_lab - new_lab
    d_rw_rgb = rw_rgb - new_rgb

    ci_app_lab = bootstrap_mean_ci(d_app_lab)
    ci_app_rgb = bootstrap_mean_ci(d_app_rgb)

    appearance_decision = (
        "GO"
        if ci_app_lab[0] > 0
        else "STOP"
    )

    result = {
        "case": case,
        "valid_tracks": int(len(ids)),

        "tc_mar_lab": {
            "realwonder": stats(rw_lab),
            "old_correct": stats(old_lab),
            "v3a_correct": stats(new_lab),

            "old_correct_minus_v3a": {
                "paired_mean_difference":
                    float(d_app_lab.mean()),
                "paired_median_difference":
                    float(np.median(d_app_lab)),
                "fraction_v3a_better":
                    float(np.mean(d_app_lab > 0)),
                "bootstrap_95_ci":
                    ci_app_lab,
            },

            "realwonder_minus_v3a": {
                "paired_mean_difference":
                    float(d_rw_lab.mean()),
                "bootstrap_95_ci":
                    bootstrap_mean_ci(d_rw_lab),
            },
        },

        "tc_mar_rgb": {
            "realwonder": stats(rw_rgb),
            "old_correct": stats(old_rgb),
            "v3a_correct": stats(new_rgb),

            "old_correct_minus_v3a": {
                "paired_mean_difference":
                    float(d_app_rgb.mean()),
                "bootstrap_95_ci":
                    ci_app_rgb,
            },

            "realwonder_minus_v3a": {
                "paired_mean_difference":
                    float(d_rw_rgb.mean()),
                "bootstrap_95_ci":
                    bootstrap_mean_ci(d_rw_rgb),
            },
        },

        "appearance_development_decision":
            appearance_decision,
    }

    csv_path = OUT / f"{case}_v3a_per_track.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "track_id",
            "old_correct_lab",
            "v3a_lab",
            "rw_lab",
            "old_minus_v3a_lab",
            "rw_minus_v3a_lab",
            "old_correct_rgb",
            "v3a_rgb",
            "rw_rgb",
        ])

        for k,i in enumerate(ids):
            w.writerow([
                i,
                old_lab[k],
                new_lab[k],
                rw_lab[k],
                d_app_lab[k],
                d_rw_lab[k],
                old_rgb[k],
                new_rgb[k],
                rw_rgb[k],
            ])

    return result


report = {
    "method": "V3A Collision-Aware Material Aggregation",
    "hard_development_gate": (
        "appearance GO iff 95% CI lower bound of "
        "(OldCorrect TC-MAR Lab - V3A TC-MAR Lab) > 0"
    ),
    "cases": {},
}

for case in ["santa", "tree"]:
    print(f"\n===== {case.upper()} =====", flush=True)

    result = eval_case(
        case,
        CASES[case],
    )

    report["cases"][case] = result

    print(
        json.dumps(
            result,
            indent=2,
        ),
        flush=True,
    )

out = OUT / "v3a_appearance_report.json"

with open(out, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nREPORT={out}")
