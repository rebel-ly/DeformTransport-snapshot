import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path("/workspace/DeformTransport")
RUN = ROOT / "server_runs/system_eval/20260810_013925__santa_tree_rw_vs_wanmove_manual"

ANCHORS = np.arange(4, 81, 4)

CASES = {
    "santa": {
        "csv": RUN / "santa_per_track.csv",
        "tracks": ROOT / "server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_tracks_correct.npy",
        "vis": ROOT / "server_runs/wan_move_bridge/20260809_010015__santa_correct_tracks/santa_material_visibility_correct.npy",
    },
    "tree": {
        "csv": RUN / "tree_per_track.csv",
        "tracks": ROOT / "server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_tracks_correct.npy",
        "vis": ROOT / "server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks/tree_material_visibility_correct.npy",
    },
}


def read_csv(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tid = int(r["track_id"])
            rows[tid] = {
                k: float(v)
                for k, v in r.items()
                if k not in ("track_id", "valid_anchor_count")
            }
            rows[tid]["valid_anchor_count"] = int(r["valid_anchor_count"])
    return rows


def group_summary(ids, motion, rows):
    rw = np.array([rows[i]["rw_lab"] for i in ids])
    correct = np.array([rows[i]["correct_lab"] for i in ids])
    shuffled = np.array([rows[i]["shuffled_lab"] for i in ids])

    rw_proxy = np.array([rows[i]["rw_proxy_l1"] for i in ids])
    correct_proxy = np.array([rows[i]["correct_proxy_l1"] for i in ids])

    rw_minus_correct = rw - correct
    shuffled_minus_correct = shuffled - correct
    proxy_rw_minus_correct = rw_proxy - correct_proxy

    return {
        "n": int(len(ids)),
        "motion_px": {
            "mean": float(np.mean(motion[ids])),
            "median": float(np.median(motion[ids])),
            "min": float(np.min(motion[ids])),
            "max": float(np.max(motion[ids])),
        },
        "tc_mar_lab": {
            "realwonder_mean": float(np.mean(rw)),
            "correct_mean": float(np.mean(correct)),
            "shuffled_mean": float(np.mean(shuffled)),
            "rw_minus_correct_mean": float(np.mean(rw_minus_correct)),
            "rw_minus_correct_median": float(np.median(rw_minus_correct)),
            "fraction_rw_worse_than_correct": float(np.mean(rw_minus_correct > 0)),
            "shuffled_minus_correct_mean": float(np.mean(shuffled_minus_correct)),
        },
        "proxy_track_l1": {
            "realwonder_mean": float(np.mean(rw_proxy)),
            "correct_mean": float(np.mean(correct_proxy)),
            "rw_minus_correct_mean": float(np.mean(proxy_rw_minus_correct)),
        },
    }


def main():
    report = {
        "purpose": (
            "POST-HOC DIAGNOSTIC ONLY. "
            "Not a new primary metric and not a replacement for frozen system gates."
        ),
        "motion_definition": (
            "Per-track mean Euclidean displacement from P_i(0) "
            "to visible TC-MAR anchors 4,8,...,80 in the aligned 832x480 domain."
        ),
        "cases": {},
    }

    for case, cfg in CASES.items():
        rows = read_csv(cfg["csv"])
        tracks = np.load(cfg["tracks"])[0].astype(np.float64)
        vis = np.load(cfg["vis"])[0].astype(bool)

        n = tracks.shape[1]
        motion = np.full(n, np.nan, dtype=np.float64)

        valid_ids = []

        for i in range(n):
            if i not in rows:
                continue

            vals = []
            p0 = tracks[0, i]

            for t in ANCHORS:
                if not vis[t, i]:
                    continue

                pt = tracks[t, i]

                if np.isfinite(p0).all() and np.isfinite(pt).all():
                    vals.append(np.linalg.norm(pt - p0))

            if vals:
                motion[i] = float(np.mean(vals))
                valid_ids.append(i)

        valid_ids = np.asarray(valid_ids, dtype=int)

        # Equal-count quartiles by motion rank.
        ordered = valid_ids[np.argsort(motion[valid_ids])]
        quartiles = np.array_split(ordered, 4)

        case_report = {
            "valid_tracks": int(len(valid_ids)),
            "overall_motion_px": {
                "mean": float(np.mean(motion[valid_ids])),
                "median": float(np.median(motion[valid_ids])),
                "p25": float(np.percentile(motion[valid_ids], 25)),
                "p75": float(np.percentile(motion[valid_ids], 75)),
                "p95": float(np.percentile(motion[valid_ids], 95)),
            },
            "quartiles": {},
        }

        for q, ids in enumerate(quartiles, start=1):
            case_report["quartiles"][f"Q{q}"] = group_summary(
                ids, motion, rows
            )

        # Pearson diagnostic correlations.
        d_rw = np.array([
            rows[i]["rw_lab"] - rows[i]["correct_lab"]
            for i in valid_ids
        ])

        d_shuf = np.array([
            rows[i]["shuffled_lab"] - rows[i]["correct_lab"]
            for i in valid_ids
        ])

        case_report["correlations"] = {
            "motion_vs_rw_minus_correct_lab": float(
                np.corrcoef(motion[valid_ids], d_rw)[0, 1]
            ),
            "motion_vs_shuffled_minus_correct_lab": float(
                np.corrcoef(motion[valid_ids], d_shuf)[0, 1]
            ),
        }

        report["cases"][case] = case_report

    out = RUN / "motion_stratified_appearance_diagnostic.json"

    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nREPORT={out}")


if __name__ == "__main__":
    main()
