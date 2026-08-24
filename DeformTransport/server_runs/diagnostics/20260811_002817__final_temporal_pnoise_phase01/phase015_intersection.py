import json
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/DeformTransport")
RUN = Path.cwd()

EVALPY = ROOT / "server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/eval_v3.py"

SUITE = ROOT / "server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0"

ANCHORS = list(range(4, 81, 4))
EARLY = list(range(4, 41, 4))
LATE = list(range(44, 81, 4))


spec = importlib.util.spec_from_file_location(
    "ev",
    EVALPY,
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


cfg = ev.CASES["tree"]

tracks = np.load(
    ROOT / cfg["tracks"]
)[0].astype(np.float32)

vis = np.load(
    ROOT / cfg["vis"]
)[0].astype(bool)


# ============================================================
# 1. Reconstruct exact Tree motion-balanced set
# ============================================================

appearance_ids = np.load(
    RUN / "tree_balanced_track_ids.npy"
).astype(np.int64)

keep = np.zeros(
    tracks.shape[1],
    dtype=bool,
)

keep[appearance_ids] = True


for t in ANCHORS:
    prev = t - 1

    centers = tracks[prev] / 2.0

    valid = (
        vis[prev]
        & vis[t]
        & np.isfinite(tracks[prev]).all(axis=1)
        & np.isfinite(tracks[t]).all(axis=1)
        & (centers[:, 0] >= 0)
        & (centers[:, 0] <= 415)
        & (centers[:, 1] >= 0)
        & (centers[:, 1] <= 239)
    )

    keep &= valid


ids = np.where(keep)[0]

print("appearance_balanced =", len(appearance_ids))
print("motion_intersection  =", len(ids))

np.save(
    RUN / "tree_motion_appearance_intersection_ids.npy",
    ids,
)


# ============================================================
# 2. Source appearance
# ============================================================

source = ev.read_rgb_image(
    ROOT / cfg["source"]
)

src_patch = ev.sample_patches(
    source,
    tracks[0, ids],
)

src_lab = ev.patch_mean_lab(
    src_patch
)


# ============================================================
# 3. Recompute TC-MAR on EXACT SAME 133 tracks
# ============================================================

paths = {
    "rw":
        ROOT / cfg["rw"],

    "v3d":
        SUITE
        / "tree"
        / "v3d"
        / "tree_v3d_correct_seed0.mp4",
}

errors = {
    "rw": {},
    "v3d": {},
}


for method, path in paths.items():

    print("LOAD", method, path)

    video = ev.read_video_common(path)

    for t in ANCHORS:

        centers = tracks[t, ids].copy()

        centers[:, 1] *= 464.0 / 480.0

        patches = ev.sample_patches(
            video[t],
            centers,
        )

        lab = ev.patch_mean_lab(
            patches
        )

        errors[method][t] = np.linalg.norm(
            lab - src_lab,
            axis=1,
        ).astype(np.float64)


# ============================================================
# 4. Physics reference displacement + relative EPE
# ============================================================

motion_old = json.loads(
    (RUN / "temporal_motion.json").read_text()
)

report = {
    "case": "tree",
    "appearance_balanced_tracks": int(len(appearance_ids)),
    "intersection_tracks": int(len(ids)),
    "anchors": {},
}


for t in ANCHORS:

    rw = errors["rw"][t]
    v3d = errors["v3d"][t]

    diff = rw - v3d

    ci = ev.bootstrap_mean_ci(diff)

    ref = np.linalg.norm(
        (
            tracks[t, ids]
            - tracks[t - 1, ids]
        ) / 2.0,
        axis=1,
    )

    old = (
        motion_old["cases"]["tree"]
        ["anchors"][str(t)]
    )

    ref_mean = float(ref.mean())
    ref_median = float(np.median(ref))

    report["anchors"][str(t)] = {
        "tcmar_rw":
            float(rw.mean()),

        "tcmar_v3d":
            float(v3d.mean()),

        "tcmar_rw_minus_v3d":
            float(diff.mean()),

        "tcmar_ci":
            ci,

        "physics_step_mean_px":
            ref_mean,

        "physics_step_median_px":
            ref_median,

        "tcme_rw":
            float(old["rw"]),

        "tcme_v3d":
            float(old["v3d"]),

        "relative_epe_rw":
            float(old["rw"] / ref_mean)
            if ref_mean > 1e-8
            else None,

        "relative_epe_v3d":
            float(old["v3d"] / ref_mean)
            if ref_mean > 1e-8
            else None,
    }


# ============================================================
# 5. Early / Late TC-MAR on same 133 tracks
# ============================================================

for name, ts in {
    "early_4_40": EARLY,
    "late_44_80": LATE,
}.items():

    rw = np.stack(
        [errors["rw"][t] for t in ts],
        axis=0,
    ).mean(axis=0)

    v3d = np.stack(
        [errors["v3d"][t] for t in ts],
        axis=0,
    ).mean(axis=0)

    diff = rw - v3d

    ci = ev.bootstrap_mean_ci(diff)

    report[name] = {
        "rw":
            float(rw.mean()),

        "v3d":
            float(v3d.mean()),

        "rw_minus_v3d":
            float(diff.mean()),

        "ci":
            ci,

        "decision":
            (
                "WIN"
                if ci[0] > 0
                else
                "LOSS"
                if ci[1] < 0
                else
                "TIE"
            ),
    }


OUT = RUN / "phase015_intersection_refmotion.json"

OUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
    + "\n"
)


print()
print("===== SAME-TRACK TC-MAR =====")
print("EARLY:", report["early_4_40"])
print("LATE :", report["late_44_80"])

print()
print("===== REFERENCE MOTION =====")

for t in ANCHORS:
    r = report["anchors"][str(t)]

    print(
        f"t={t:02d}",
        f"ref={r['physics_step_mean_px']:.4f}px",
        f"RW_EPE={r['tcme_rw']:.4f}",
        f"V3D_EPE={r['tcme_v3d']:.4f}",
        f"RW/ref={r['relative_epe_rw']:.3f}",
        f"V3D/ref={r['relative_epe_v3d']:.3f}",
    )

print()
print("SAVED:", OUT)
