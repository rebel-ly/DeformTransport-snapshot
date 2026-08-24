import sys
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/DeformTransport")
RUN = ROOT / "server_runs/system_eval/20260810_013925__santa_tree_rw_vs_wanmove_manual"
LAUNCH = ROOT / "server_runs/system_eval_launcher_20260810"

sys.path.insert(0, str(LAUNCH))

from system_cpu import (
    CASES,
    read_rgb_image,
    read_video_common,
    sample_patches,
    patch_mean_lab,
)

KEY_FRAMES = [0,1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80]


def main():

    report = {
        "purpose": (
            "POST-HOC DIAGNOSTIC ONLY: determine whether Wan-Move "
            "appearance deficit already exists at frame 0 or develops "
            "during temporal generation."
        ),
        "cases": {}
    }

    all_csv = []

    for case, cfg in CASES.items():

        P = lambda rel: ROOT / rel

        print(f"\n===== {case.upper()} =====", flush=True)

        source = read_rgb_image(P(cfg["source"]))

        tracks = np.load(P(cfg["tracks"]))[0].astype(np.float32)
        vis = np.load(P(cfg["vis"]))[0].astype(bool)

        videos = {
            m: read_video_common(P(cfg[m]))
            for m in ["rw", "correct", "shuffled"]
        }

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

        case_rows = []

        for t in range(81):

            centers480 = tracks[t].copy()

            finite = np.isfinite(
                centers480
            ).all(axis=1)

            centers464 = centers480.copy()
            centers464[:,1] *= 464.0 / 480.0

            bound = (
                (centers464[:,0] - 3.5 >= 0)
                & (centers464[:,0] + 3.5 <= 831)
                & (centers464[:,1] - 3.5 >= 0)
                & (centers464[:,1] + 3.5 <= 463)
            )

            valid = (
                source_valid
                & finite
                & bound
                & vis[t]
            )

            ids = np.where(valid)[0]

            if len(ids) == 0:
                continue

            motion = np.linalg.norm(
                centers480[ids]
                - p0[ids],
                axis=1,
            )

            method_values = {}

            for method in [
                "rw",
                "correct",
                "shuffled",
            ]:

                patch = sample_patches(
                    videos[method][t],
                    centers464[ids],
                )

                lab = patch_mean_lab(
                    patch
                )

                lab_err = np.linalg.norm(
                    lab - src_lab[ids],
                    axis=1,
                )

                rgb_err = np.abs(
                    patch - src_patch[ids]
                ).mean(axis=(1,2,3))

                method_values[method] = {
                    "lab": float(
                        np.mean(lab_err)
                    ),
                    "rgb": float(
                        np.mean(rgb_err)
                    ),
                }

            row = {
                "case": case,
                "frame": t,
                "valid_tracks": int(len(ids)),
                "mean_motion_px": float(
                    np.mean(motion)
                ),

                "rw_lab":
                    method_values["rw"]["lab"],

                "correct_lab":
                    method_values["correct"]["lab"],

                "shuffled_lab":
                    method_values["shuffled"]["lab"],

                "rw_minus_correct_lab":
                    method_values["rw"]["lab"]
                    - method_values["correct"]["lab"],

                "shuffled_minus_correct_lab":
                    method_values["shuffled"]["lab"]
                    - method_values["correct"]["lab"],

                "rw_rgb":
                    method_values["rw"]["rgb"],

                "correct_rgb":
                    method_values["correct"]["rgb"],

                "shuffled_rgb":
                    method_values["shuffled"]["rgb"],
            }

            case_rows.append(row)
            all_csv.append(row)

        report["cases"][case] = {
            "frames": case_rows
        }

        print(
            "frame | motion | RW Lab | Correct Lab | "
            "Shuffled Lab | RW-Correct | Shuf-Correct"
        )

        for r in case_rows:
            if r["frame"] not in KEY_FRAMES:
                continue

            print(
                f'{r["frame"]:>5d} | '
                f'{r["mean_motion_px"]:>7.2f} | '
                f'{r["rw_lab"]:>6.2f} | '
                f'{r["correct_lab"]:>11.2f} | '
                f'{r["shuffled_lab"]:>12.2f} | '
                f'{r["rw_minus_correct_lab"]:>10.2f} | '
                f'{r["shuffled_minus_correct_lab"]:>12.2f}'
            )

        # Simple diagnostic summaries.
        by_t = {
            r["frame"]: r
            for r in case_rows
        }

        report["cases"][case][
            "diagnostic_summary"
        ] = {
            "frame0_rw_minus_correct_lab":
                by_t[0]["rw_minus_correct_lab"],

            "frame1_rw_minus_correct_lab":
                by_t[1]["rw_minus_correct_lab"],

            "frame4_rw_minus_correct_lab":
                by_t[4]["rw_minus_correct_lab"],

            "frame40_rw_minus_correct_lab":
                by_t[40]["rw_minus_correct_lab"],

            "frame80_rw_minus_correct_lab":
                by_t[80]["rw_minus_correct_lab"],

            "frame0_correct_lab":
                by_t[0]["correct_lab"],

            "frame0_rw_lab":
                by_t[0]["rw_lab"],

            "frame0_shuffled_lab":
                by_t[0]["shuffled_lab"],
        }

    csv_path = RUN / "temporal_appearance_drift.csv"

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        cols = list(
            all_csv[0].keys()
        )

        w = csv.DictWriter(
            f,
            fieldnames=cols,
        )

        w.writeheader()
        w.writerows(all_csv)

    out = (
        RUN
        / "temporal_appearance_drift.json"
    )

    with open(
        out,
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(f"\nREPORT={out}")
    print(f"CSV={csv_path}")


if __name__ == "__main__":
    main()
