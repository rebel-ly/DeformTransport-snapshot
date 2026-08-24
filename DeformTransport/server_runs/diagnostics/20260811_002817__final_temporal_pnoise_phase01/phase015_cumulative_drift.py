import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/workspace/DeformTransport")
RUN = Path.cwd()

EVALPY = ROOT / "server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/eval_v3.py"

SUITE = ROOT / "server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0"

spec = importlib.util.spec_from_file_location("ev", EVALPY)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

cfg = ev.CASES["tree"]

tracks = np.load(ROOT / cfg["tracks"])[0].astype(np.float32)
vis = np.load(ROOT / cfg["vis"])[0].astype(bool)

# ------------------------------------------------------------
# Exact all-frame support in 416x240 RAFT coordinates
# ------------------------------------------------------------

xy = tracks / 2.0

valid = (
    vis
    & np.isfinite(tracks).all(axis=2)
    & (xy[:, :, 0] >= 0)
    & (xy[:, :, 0] <= 415)
    & (xy[:, :, 1] >= 0)
    & (xy[:, :, 1] <= 239)
)

keep = valid.all(axis=0)
ids = np.where(keep)[0].astype(np.int64)

print("strict_all_frame_tracks =", len(ids), flush=True)

np.save(
    RUN / "tree_cumulative_drift_track_ids.npy",
    ids,
)

paths = {
    "rw": ROOT / cfg["rw"],
    "v3d":
        SUITE / "tree" / "v3d"
        / "tree_v3d_correct_seed0.mp4",
}

device = torch.device("cuda:0")

model, transforms = ev.load_raft_cached(device)

all_errors = {}

for method, path in paths.items():

    print("LOAD", method, path, flush=True)

    video = ev.read_video_common(path)

    # Generated-material position starts at physical source position.
    pred_xy = xy[0, ids].copy()

    errors = {}
    pred_history = {
        0: pred_xy.copy()
    }

    # Process consecutive frame pairs in small batches.
    batch = 4

    for start in range(1, 81, batch):

        ts = list(
            range(
                start,
                min(start + batch, 81),
            )
        )

        prevs = [t - 1 for t in ts]

        a = (
            torch.from_numpy(video[prevs])
            .permute(0, 3, 1, 2)
            .to(device)
        )

        b = (
            torch.from_numpy(video[ts])
            .permute(0, 3, 1, 2)
            .to(device)
        )

        a = F.interpolate(
            a,
            size=(240, 416),
            mode="area",
        )

        b = F.interpolate(
            b,
            size=(240, 416),
            mode="area",
        )

        a, b = transforms(a, b)

        with torch.inference_mode():
            flows = (
                model(a, b)[-1]
                .float()
                .cpu()
                .numpy()
            )

        # Important:
        # recursively sample flow at generated/advection position,
        # not at physical position.
        for j, t in enumerate(ts):

            f = ev.bilinear_flow(
                flows[j],
                pred_xy,
            )

            pred_xy = pred_xy + f

            pred_history[t] = pred_xy.copy()

            phys_xy = xy[t, ids]

            errors[t] = np.linalg.norm(
                pred_xy - phys_xy,
                axis=1,
            ).astype(np.float64)

        print(
            method,
            "processed",
            ts[-1],
            "/80",
            flush=True,
        )

    all_errors[method] = errors

    del video
    torch.cuda.empty_cache()


def boot_ci(x, nboot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)

    vals = []

    for _ in range(nboot // 250):
        idx = rng.integers(
            0,
            n,
            size=(250, n),
        )
        vals.append(
            x[idx].mean(axis=1)
        )

    vals = np.concatenate(vals)

    return [
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    ]


anchors = list(range(4, 81, 4))

report = {
    "case": "tree",
    "metric": "TC-CDE",
    "strict_all_frame_tracks": int(len(ids)),
    "anchors": {},
}

for t in anchors:

    rw = all_errors["rw"][t]
    v3d = all_errors["v3d"][t]

    diff = rw - v3d

    ci = boot_ci(diff)

    physics_net = np.linalg.norm(
        xy[t, ids] - xy[0, ids],
        axis=1,
    )

    net_mean = float(physics_net.mean())

    report["anchors"][str(t)] = {
        "rw_cde_px": float(rw.mean()),
        "v3d_cde_px": float(v3d.mean()),
        "rw_minus_v3d": float(diff.mean()),
        "ci": ci,
        "decision":
            "WIN" if ci[0] > 0
            else "LOSS" if ci[1] < 0
            else "TIE",

        "physics_net_displacement_px":
            net_mean,

        "rw_cde_over_net_motion":
            float(rw.mean() / net_mean)
            if net_mean > 1e-8 else None,

        "v3d_cde_over_net_motion":
            float(v3d.mean() / net_mean)
            if net_mean > 1e-8 else None,
    }


for name, ts in {
    "early_4_40": list(range(4, 41, 4)),
    "late_44_80": list(range(44, 81, 4)),
}.items():

    rw = np.stack(
        [all_errors["rw"][t] for t in ts]
    ).mean(axis=0)

    v3d = np.stack(
        [all_errors["v3d"][t] for t in ts]
    ).mean(axis=0)

    d = rw - v3d
    ci = boot_ci(d)

    report[name] = {
        "rw": float(rw.mean()),
        "v3d": float(v3d.mean()),
        "rw_minus_v3d": float(d.mean()),
        "ci": ci,
        "decision":
            "WIN" if ci[0] > 0
            else "LOSS" if ci[1] < 0
            else "TIE",
    }


OUT = RUN / "phase015_cumulative_drift_tree.json"

OUT.write_text(
    json.dumps(report, indent=2) + "\n"
)

print()
print("===== TC-CDE SUMMARY =====")
print("EARLY:", report["early_4_40"])
print("LATE :", report["late_44_80"])

print()
print("===== SELECTED ANCHORS =====")

for t in [4, 20, 40, 48, 60, 80]:
    print(
        "t=",
        t,
        report["anchors"][str(t)],
    )

print()
print("SAVED:", OUT)
