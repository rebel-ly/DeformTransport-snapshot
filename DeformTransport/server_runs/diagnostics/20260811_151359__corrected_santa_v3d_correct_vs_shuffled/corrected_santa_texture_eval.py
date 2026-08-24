import json
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/workspace/DeformTransport")
RUN = Path.cwd()

EVAL = ROOT / (
    "server_runs/wan_move_method_eval/"
    "20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/"
    "eval_v3.py"
)

BRIDGE = ROOT / (
    "server_runs/wan_move_bridge/"
    "20260811_024330__santa_corrected_physical_visibility"
)

RERUN = Path(
    (
        ROOT /
        "server_runs/wan_move_formal/"
        "current_santa_corrected_v3d_rerun.txt"
    ).read_text().strip()
)

TRACK = BRIDGE / "santa_material_tracks_correct.npy"
VIS = BRIDGE / "santa_material_visibility_correct.npy"

CORRECT = (
    RERUN / "correct" /
    "santa_v3d_corrected_visibility_correct_seed0.mp4"
)

SHUFFLED = (
    RERUN / "shuffled" /
    "santa_v3d_corrected_visibility_shuffled_seed0.mp4"
)

EXPECTED_SHA = {
    "correct":
        "df5e50ad3446f64666d48497110ef8622738d145dd95ef2241676451099c643d",
    "shuffled":
        "4017af9ee9174cb649e454577074f9c037ba1c10893e28440c25bd38a973295a",
}

spec = importlib.util.spec_from_file_location("ev", EVAL)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

ANCH = list(range(4, 81, 4))
EARLY = list(range(4, 41, 4))
LATE = list(range(44, 81, 4))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1024 * 1024), b""):
            h.update(x)
    return h.hexdigest()


def boot(x):
    return ev.bootstrap_mean_ci(
        np.asarray(x, np.float64)
    )


def lab_patch(x):
    x = np.asarray(x, np.float32)

    if x.max() > 1.5:
        x = x / 255.0

    n, h, w, c = x.shape

    y = cv2.cvtColor(
        x.reshape(n * h, w, 3),
        cv2.COLOR_RGB2LAB,
    )

    return y.reshape(n, h, w, 3)


def tex_errors(src, cur):
    a = lab_patch(src)
    b = lab_patch(cur)

    patch = np.sqrt(
        np.mean(
            np.sum((b - a) ** 2, axis=-1),
            axis=(1, 2),
        )
    )

    La = a[..., 0]
    Lb = b[..., 0]

    gya, gxa = np.gradient(
        La,
        axis=(1, 2),
    )

    gyb, gxb = np.gradient(
        Lb,
        axis=(1, 2),
    )

    grad = np.mean(
        np.sqrt(
            (gxb - gxa) ** 2 +
            (gyb - gya) ** 2
        ),
        axis=(1, 2),
    )

    return (
        patch.astype(np.float64),
        grad.astype(np.float64),
    )


# EXACT frozen Phase-01.5 support rule.
def valid_all(tr, vis):
    xy = tr / 2.0

    good = (
        vis
        & np.isfinite(tr).all(axis=2)
        & (xy[:, :, 0] >= 0)
        & (xy[:, :, 0] <= 415)
        & (xy[:, :, 1] >= 0)
        & (xy[:, :, 1] <= 239)
    )

    return np.where(
        good.all(axis=0)
    )[0]


assert sha256(CORRECT) == EXPECTED_SHA["correct"]
assert sha256(SHUFFLED) == EXPECTED_SHA["shuffled"]

cfg = ev.CASES["santa"]

tr = np.load(TRACK)[0].astype(np.float32)
vis = np.load(VIS)[0].astype(bool)

ids = valid_all(tr, vis)

print("STRICT_ALL_FRAME_VISIBLE_TRACKS =", len(ids))

# Corrected contract previously established 63 all-frame-visible tracks.
assert len(ids) == 63, (
    "unexpected corrected Santa strict support",
    len(ids),
)

src_img = ev.read_rgb_image(
    ROOT / cfg["source"]
)

src_patch = ev.sample_patches(
    src_img,
    tr[0, ids],
)

src_mean = ev.patch_mean_lab(src_patch)

paths = {
    "correct": CORRECT,
    "shuffled": SHUFFLED,
}

errs = {
    m: {
        k: {}
        for k in [
            "tcmar",
            "patch",
            "grad",
        ]
    }
    for m in paths
}

for method, path in paths.items():

    print(
        "LOAD",
        method,
        path,
        flush=True,
    )

    vid = ev.read_video_common(path)

    assert len(vid) >= 81

    for t in ANCH:

        xy = tr[t, ids].copy()

        # Exact frozen appearance-domain contract:
        # source 480x832 -> common video 464x832
        xy[:, 1] *= 464.0 / 480.0

        p = ev.sample_patches(
            vid[t],
            xy,
        )

        mean = ev.patch_mean_lab(p)

        errs[method]["tcmar"][t] = (
            np.linalg.norm(
                mean - src_mean,
                axis=1,
            ).astype(np.float64)
        )

        patch, grad = tex_errors(
            src_patch,
            p,
        )

        errs[method]["patch"][t] = patch
        errs[method]["grad"][t] = grad


report = {
    "protocol":
        "Corrected Santa V3D Correct vs Identity-Shuffled; "
        "exact frozen Phase-01.5 appearance/texture protocol",
    "lower_is_better": True,
    "difference_definition":
        "shuffled_minus_correct; positive CI means Correct WIN",
    "n_tracks": int(len(ids)),
    "track_ids": ids.tolist(),
    "input_sha256": {
        "correct": sha256(CORRECT),
        "shuffled": sha256(SHUFFLED),
    },
    "anchors": {},
}


def summarize(ts, key):

    c = np.stack([
        errs["correct"][key][t]
        for t in ts
    ]).mean(0)

    s = np.stack([
        errs["shuffled"][key][t]
        for t in ts
    ]).mean(0)

    # Lower is better, so positive shuffled-correct favors Correct.
    d = s - c
    ci = boot(d)

    return {
        "correct": float(c.mean()),
        "shuffled": float(s.mean()),
        "shuffled_minus_correct":
            float(d.mean()),
        "ci": ci,
        "decision":
            "CORRECT_WIN"
            if ci[0] > 0
            else (
                "CORRECT_LOSS"
                if ci[1] < 0
                else "TIE"
            ),
    }


for t in ANCH:
    report["anchors"][str(t)] = {}

    for key, label in [
        ("tcmar", "TC-MAR"),
        ("patch", "TC-Patch-Lab"),
        ("grad", "TC-Grad"),
    ]:
        c = errs["correct"][key][t]
        s = errs["shuffled"][key][t]

        d = s - c
        ci = boot(d)

        report["anchors"][str(t)][label] = {
            "correct": float(c.mean()),
            "shuffled": float(s.mean()),
            "shuffled_minus_correct":
                float(d.mean()),
            "ci": ci,
            "decision":
                "CORRECT_WIN"
                if ci[0] > 0
                else (
                    "CORRECT_LOSS"
                    if ci[1] < 0
                    else "TIE"
                ),
        }


report["early"] = {}
report["late"] = {}

for key, label in [
    ("tcmar", "TC-MAR"),
    ("patch", "TC-Patch-Lab"),
    ("grad", "TC-Grad"),
]:
    report["early"][label] = summarize(
        EARLY,
        key,
    )

    report["late"][label] = summarize(
        LATE,
        key,
    )


(RUN / "corrected_santa_texture_eval.json").write_text(
    json.dumps(
        report,
        indent=2,
    ) + "\n"
)


print()
print("===== CORRECTED SANTA SUMMARY =====")
print("n_tracks =", report["n_tracks"])

print()
print("EARLY")
for k, v in report["early"].items():
    print(k, v)

print()
print("LATE")
for k, v in report["late"].items():
    print(k, v)

print()
print("CORRECTED_SANTA_TEXTURE_EVAL_DONE")
