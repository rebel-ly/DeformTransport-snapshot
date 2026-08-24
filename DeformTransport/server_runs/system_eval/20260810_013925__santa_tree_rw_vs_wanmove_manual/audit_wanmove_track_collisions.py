import json
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/DeformTransport")
RUN = ROOT / "server_runs/system_eval/20260810_013925__santa_tree_rw_vs_wanmove_manual"

CASES = {
    "santa": {
        "tracks": ROOT / "server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_tracks_correct.npy",
        "vis": ROOT / "server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_visibility_correct.npy",
    },
    "tree": {
        "tracks": ROOT / "server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_tracks_correct.npy",
        "vis": ROOT / "server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_visibility_correct.npy",
    },
}

# Formal Wan-Move output:
# input-aligned coordinate domain = 832x480
# generated domain = 832x464
#
# VAE spatial stride = 8.
#
# create_pos_feature_map() uses t = 0,4,8,...,80
# and floor(x/8), floor(y/8).

TIME = list(range(0, 81, 4))

OUT_H = 464
OUT_W = 832
STRIDE = 8

LAT_H = OUT_H // STRIDE  # 58
LAT_W = OUT_W // STRIDE  # 104


def analyze_case(name, cfg):

    tracks = np.load(cfg["tracks"])[0].astype(np.float64)
    vis = np.load(cfg["vis"])[0].astype(bool)

    assert tracks.shape[:2] == vis.shape
    assert tracks.shape[0] == 81
    assert tracks.shape[2] == 2

    # Wan-Move scales input track y 480 -> output y 464.
    scaled = tracks.copy()
    scaled[..., 1] *= OUT_H / 480.0

    rows = []

    all_valid = 0
    all_unique = 0

    for t in TIME:

        xy = scaled[t]

        valid = (
            vis[t]
            & np.isfinite(xy).all(axis=1)
            & (xy[:,0] >= 0)
            & (xy[:,0] < OUT_W)
            & (xy[:,1] >= 0)
            & (xy[:,1] < OUT_H)
        )

        ids = np.where(valid)[0]

        xcell = np.floor(
            xy[ids,0] / STRIDE
        ).astype(np.int64)

        ycell = np.floor(
            xy[ids,1] / STRIDE
        ).astype(np.int64)

        assert np.all(
            (xcell >= 0) & (xcell < LAT_W)
        )

        assert np.all(
            (ycell >= 0) & (ycell < LAT_H)
        )

        flat = ycell * LAT_W + xcell

        unique, counts = np.unique(
            flat,
            return_counts=True
        )

        n_valid = len(ids)
        n_unique = len(unique)

        collisions = n_valid - n_unique

        collision_tracks = int(
            counts[counts > 1].sum()
        )

        collision_cells = int(
            np.sum(counts > 1)
        )

        max_mult = (
            int(counts.max())
            if len(counts)
            else 0
        )

        rows.append({
            "frame": t,
            "latent_t": t // 4,
            "valid_tracks": int(n_valid),
            "unique_cells": int(n_unique),
            "lost_assignments_if_one_feature_per_cell":
                int(collisions),
            "assignment_collision_rate":
                float(
                    collisions / n_valid
                ) if n_valid else 0.0,
            "tracks_in_collision_cells":
                collision_tracks,
            "collision_cells":
                collision_cells,
            "max_cell_multiplicity":
                max_mult,
            "latent_grid_coverage":
                float(
                    n_unique / (LAT_H * LAT_W)
                ),
        })

        all_valid += n_valid
        all_unique += n_unique

    rates = np.array([
        r["assignment_collision_rate"]
        for r in rows
    ])

    valid_counts = np.array([
        r["valid_tracks"]
        for r in rows
    ])

    unique_counts = np.array([
        r["unique_cells"]
        for r in rows
    ])

    result = {
        "case": name,
        "input_tracks": int(
            tracks.shape[1]
        ),
        "wanmove_training_track_range":
            "1-200 according to Wan-Move paper",
        "latent_grid": [
            LAT_H,
            LAT_W
        ],
        "latent_cells":
            LAT_H * LAT_W,
        "temporal_slots":
            TIME,
        "summary": {
            "mean_valid_tracks":
                float(valid_counts.mean()),
            "mean_unique_cells":
                float(unique_counts.mean()),
            "mean_collision_rate":
                float(rates.mean()),
            "median_collision_rate":
                float(np.median(rates)),
            "max_collision_rate":
                float(rates.max()),
            "total_valid_assignments":
                int(all_valid),
            "total_unique_cell_assignments":
                int(all_unique),
            "total_assignment_loss":
                int(all_valid-all_unique),
            "overall_assignment_loss_rate":
                float(
                    (all_valid-all_unique)
                    / all_valid
                ),
        },
        "per_latent_time": rows,
    }

    return result


report = {
    "purpose":
        "Audit current Wan-Move material-track density "
        "and latent-cell collision rate. "
        "Diagnostic only; no metric selection or tuning.",
    "cases": {}
}

for name,cfg in CASES.items():
    report["cases"][name] = analyze_case(
        name,cfg
    )

out = RUN / "wanmove_track_collision_audit.json"

with open(out,"w") as f:
    json.dump(
        report,
        f,
        indent=2
    )

for name in ["santa","tree"]:
    x = report["cases"][name]
    s = x["summary"]

    print("\n==============================")
    print(name.upper())
    print("==============================")

    print("input tracks:",
          x["input_tracks"])

    print("mean valid tracks:",
          round(s["mean_valid_tracks"],2))

    print("mean unique cells:",
          round(s["mean_unique_cells"],2))

    print("mean collision rate:",
          round(
              100*s["mean_collision_rate"],
              2
          ), "%")

    print("max collision rate:",
          round(
              100*s["max_collision_rate"],
              2
          ), "%")

    print("overall assignment loss:",
          s["total_assignment_loss"])

    print("overall loss rate:",
          round(
              100*s[
                  "overall_assignment_loss_rate"
              ],
              2
          ), "%")

    print("\nframe  valid unique collision% max_mult")

    for r in x["per_latent_time"]:
        print(
            f'{r["frame"]:>5} '
            f'{r["valid_tracks"]:>6} '
            f'{r["unique_cells"]:>6} '
            f'{100*r["assignment_collision_rate"]:>9.2f} '
            f'{r["max_cell_multiplicity"]:>8}'
        )

print("\nREPORT =",out)
