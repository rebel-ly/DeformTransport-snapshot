import sys
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/DeformTransport")
LAUNCH = ROOT / "server_runs/system_eval_launcher_20260810"

sys.path.insert(0, str(LAUNCH))

from system_raft_tcme import (
    CASES,
    check_weights_cached,
    read_video_raft_grid,
    infer_flows,
    evaluate_method,
    bootstrap_mean_ci,
)

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights,
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

OLD = ROOT / "server_runs/system_eval/20260810_013925__santa_tree_rw_vs_wanmove_manual"

V3A = {
    "santa":
        BATCH / "santa/santa_v3a_collision_mean_correct_seed0.mp4",

    "tree":
        BATCH / "tree/tree_v3a_collision_mean_correct_seed0.mp4",
}


def load_old(case):

    p = OLD / f"{case}_raft_tcme_per_transition.csv"

    rows = []

    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "old_correct":
                    float(r["correct_mean_epe"]),

                "rw":
                    float(r["rw_mean_epe"]),
            })

    assert len(rows) == 80

    return rows


def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        required=True,
        choices=["santa","tree"],
    )

    args = parser.parse_args()

    case = args.case
    cfg = CASES[case]

    device = torch.device("cuda:0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    print(
        "DEVICE=",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    tracks = np.load(
        ROOT / cfg["tracks"]
    )[0].astype(np.float32)

    vis = np.load(
        ROOT / cfg["vis"]
    )[0].astype(bool)

    weights = Raft_Large_Weights.C_T_SKHT_V2

    check_weights_cached(weights)

    transforms = weights.transforms()

    model = raft_large(
        weights=weights,
        progress=False,
    ).to(device)

    model.eval()

    print(
        "RAFT_MODEL_READY",
        flush=True,
    )

    frames = read_video_raft_grid(
        V3A[case]
    )

    flows = infer_flows(
        model,
        transforms,
        frames,
        device,
    )

    evaluated = evaluate_method(
        flows,
        tracks,
        vis,
    )

    v3a = evaluated[
        "transition_mean_epe"
    ]

    old_rows = load_old(case)

    old = np.array([
        r["old_correct"]
        for r in old_rows
    ])

    rw = np.array([
        r["rw"]
        for r in old_rows
    ])

    # Positive means V3A is worse.
    regression = v3a - old

    # Positive means V3A better than RW.
    rw_minus_v3a = rw - v3a

    ci_reg = bootstrap_mean_ci(
        regression
    )

    ci_rw = bootstrap_mean_ci(
        rw_minus_v3a
    )

    motion_safety = (
        "REGRESSION_STOP"
        if ci_reg[0] > 0
        else "PASS"
    )

    rw_decision = (
        "WIN"
        if ci_rw[0] > 0
        else (
            "LOSS"
            if ci_rw[1] < 0
            else "TIE"
        )
    )

    result = {
        "case": case,

        "old_correct_tcme_mean":
            float(old.mean()),

        "v3a_tcme_mean":
            float(v3a.mean()),

        "realwonder_tcme_mean":
            float(rw.mean()),

        "v3a_minus_old_correct": {
            "definition":
                "positive means V3A motion is worse",

            "paired_mean_difference":
                float(regression.mean()),

            "paired_median_difference":
                float(np.median(regression)),

            "fraction_v3a_worse":
                float(np.mean(regression > 0)),

            "bootstrap_95_ci":
                ci_reg,
        },

        "realwonder_minus_v3a": {
            "definition":
                "positive means V3A better than RealWonder",

            "paired_mean_difference":
                float(rw_minus_v3a.mean()),

            "bootstrap_95_ci":
                ci_rw,

            "decision":
                rw_decision,
        },

        "motion_safety":
            motion_safety,
    }

    out = OUT / f"{case}_v3a_motion_report.json"

    with open(out, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    csv_out = OUT / f"{case}_v3a_motion_per_transition.csv"

    with open(
        csv_out,
        "w",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "transition",
            "rw",
            "old_correct",
            "v3a",
            "v3a_minus_old",
            "rw_minus_v3a",
        ])

        for t in range(80):
            w.writerow([
                t,
                rw[t],
                old[t],
                v3a[t],
                regression[t],
                rw_minus_v3a[t],
            ])

    print(
        json.dumps(
            result,
            indent=2,
        ),
        flush=True,
    )

    print(f"\nREPORT={out}")


if __name__ == "__main__":
    main()
