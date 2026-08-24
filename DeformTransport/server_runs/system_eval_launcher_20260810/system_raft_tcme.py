import argparse
import csv
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights,
)


BOOT_N = 10000
BOOT_SEED = 0
RAFT_H = 240
RAFT_W = 416
BATCH_SIZE = 4


CASES = {
    "santa": {
        "rw":
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "12_soft_transport_dev/"
            "20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/"
            "baseline/aligned_santa_baseline_seed0.mp4",

        "correct":
            "server_runs/wan_move_formal/"
            "20260809_195255__santa_correct_vs_identity_shuffled_seed0/"
            "correct/santa_formal_correct_seed0.mp4",

        "shuffled":
            "server_runs/wan_move_formal/"
            "20260809_195255__santa_correct_vs_identity_shuffled_seed0/"
            "shuffled/santa_formal_identity_shuffled_seed0.mp4",

        "tracks":
            "server_runs/wan_move_bridge/"
            "20260809_010015__santa_correct_tracks/"
            "santa_material_tracks_correct.npy",

        "vis":
            "server_runs/wan_move_bridge/"
            "20260809_010015__santa_correct_tracks/"
            "santa_material_visibility_correct.npy",

        "expected_tracks": 1277,
    },

    "tree": {
        "rw":
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "12_soft_transport_dev/"
            "20260807_203228__tree__realwonder_baseline_seed0/"
            "tree_realwonder_baseline_seed0.mp4",

        "correct":
            "server_runs/wan_move_formal/"
            "20260810_073902__tree_correct_vs_identity_shuffled_seed0/"
            "correct/tree_formal_correct_seed0.mp4",

        "shuffled":
            "server_runs/wan_move_formal/"
            "20260810_073902__tree_correct_vs_identity_shuffled_seed0/"
            "shuffled/tree_formal_identity_shuffled_seed0.mp4",

        "tracks":
            "server_runs/wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_tracks_correct.npy",

        "vis":
            "server_runs/wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_visibility_correct.npy",

        "expected_tracks": 713,
    },
}


def stats(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }


def bootstrap_mean_ci(diff):
    d = np.asarray(diff, dtype=np.float64)
    n = len(d)

    if n == 0:
        raise RuntimeError("bootstrap received empty input")

    rng = np.random.default_rng(BOOT_SEED)

    vals = []
    chunk = 500

    for start in range(0, BOOT_N, chunk):
        k = min(chunk, BOOT_N - start)

        idx = rng.integers(
            0,
            n,
            size=(k, n),
        )

        vals.append(
            d[idx].mean(axis=1)
        )

    vals = np.concatenate(vals)

    return [
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    ]


def check_weights_cached(weights):
    filename = Path(
        urlparse(weights.url).path
    ).name

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / filename
    )

    if not checkpoint.exists():
        raise RuntimeError(
            "RAFT_WEIGHT_NOT_CACHED: "
            f"{checkpoint}"
        )

    print(
        f"RAFT_WEIGHT_CACHE_OK={checkpoint}",
        flush=True,
    )


def frame_to_raft_grid(rgb_u8):
    """
    Frozen spatial protocol:

    RealWonder native:
      832x480
      -> bicubic 832x464
      -> area 416x240

    Wan-Move native:
      832x464
      -> area 416x240

    All output float RGB [0,1].
    """

    h, w = rgb_u8.shape[:2]

    if w != 832:
        raise AssertionError(
            f"Unexpected width {w}"
        )

    x = (
        torch.from_numpy(rgb_u8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )

    if h == 480:
        x = F.interpolate(
            x,
            size=(464, 832),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        ).clamp_(0, 1)

    elif h != 464:
        raise AssertionError(
            f"Unexpected height {h}"
        )

    x = F.interpolate(
        x,
        size=(RAFT_H, RAFT_W),
        mode="area",
    )

    return x[0].contiguous()


def read_video_raft_grid(path):
    cap = cv2.VideoCapture(str(path))

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
            frame_to_raft_grid(rgb)
        )

    cap.release()

    if len(frames) != 81:
        raise AssertionError(
            f"{path}: expected 81 frames, "
            f"got {len(frames)}"
        )

    out = torch.stack(
        frames,
        dim=0,
    )

    if out.shape != (
        81,
        3,
        RAFT_H,
        RAFT_W,
    ):
        raise AssertionError(
            f"bad RAFT tensor shape: "
            f"{out.shape}"
        )

    return out


def bilinear_sample_flow(flow, xy):
    """
    flow: [2,H,W] NumPy
    xy:   [N,2] float coordinates
          in RAFT 416x240 pixel domain
    """

    H, W = flow.shape[1:]

    x = xy[:, 0]
    y = xy[:, 1]

    if np.any(~np.isfinite(xy)):
        raise RuntimeError(
            "non-finite flow sample coordinates"
        )

    if (
        np.any(x < 0)
        or np.any(x > W - 1)
        or np.any(y < 0)
        or np.any(y > H - 1)
    ):
        bad = np.where(
            (x < 0)
            | (x > W - 1)
            | (y < 0)
            | (y > H - 1)
        )[0]

        raise RuntimeError(
            "visible persistent track lies outside "
            f"RAFT grid; bad count={len(bad)}"
        )

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)

    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = x - x0
    wy = y - y0

    Ia = flow[:, y0, x0].T
    Ib = flow[:, y0, x1].T
    Ic = flow[:, y1, x0].T
    Id = flow[:, y1, x1].T

    pred = (
        (1 - wx)[:, None]
        * (1 - wy)[:, None]
        * Ia

        + wx[:, None]
        * (1 - wy)[:, None]
        * Ib

        + (1 - wx)[:, None]
        * wy[:, None]
        * Ic

        + wx[:, None]
        * wy[:, None]
        * Id
    )

    return pred.astype(np.float32)


@torch.inference_mode()
def infer_flows(
    model,
    transforms,
    frames,
    device,
):
    """
    Returns [80,2,240,416].
    """

    results = []

    for start in range(
        0,
        80,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            80,
        )

        img1 = frames[start:end]
        img2 = frames[start + 1:end + 1]

        img1, img2 = transforms(
            img1,
            img2,
        )

        img1 = img1.to(
            device,
            non_blocking=True,
        )

        img2 = img2.to(
            device,
            non_blocking=True,
        )

        pred = model(
            img1,
            img2,
        )[-1]

        results.append(
            pred.detach().cpu()
        )

        print(
            f"RAFT transitions "
            f"{start:02d}-{end-1:02d}/79",
            flush=True,
        )

    out = torch.cat(
        results,
        dim=0,
    )

    assert out.shape == (
        80,
        2,
        RAFT_H,
        RAFT_W,
    )

    return out.numpy().astype(
        np.float32
    )


def evaluate_method(
    flows,
    tracks,
    vis,
):
    """
    Reference is persistent material-point
    image displacement.

    Original track domain:
      832x480

    RAFT domain:
      416x240

    Therefore:
      q = P / 2
      d = (P[t+1]-P[t]) / 2
    """

    transition_rows = []

    pooled_epe = []
    pooled_mag = []

    for t in range(80):

        valid = (
            vis[t]
            & vis[t + 1]
            & np.isfinite(
                tracks[t]
            ).all(axis=1)
            & np.isfinite(
                tracks[t + 1]
            ).all(axis=1)
        )

        ids = np.where(valid)[0]

        if len(ids) == 0:
            raise RuntimeError(
                f"transition {t}: "
                "no visible persistent tracks"
            )

        q = (
            tracks[t, ids]
            * 0.5
        ).astype(np.float32)

        ref = (
            (
                tracks[t + 1, ids]
                - tracks[t, ids]
            )
            * 0.5
        ).astype(np.float32)

        pred = bilinear_sample_flow(
            flows[t],
            q,
        )

        err_vec = pred - ref

        epe = np.linalg.norm(
            err_vec,
            axis=1,
        )

        pred_mag = np.linalg.norm(
            pred,
            axis=1,
        )

        ref_mag = np.linalg.norm(
            ref,
            axis=1,
        )

        mag_err = np.abs(
            pred_mag - ref_mag
        )

        pooled_epe.append(epe)
        pooled_mag.append(mag_err)

        transition_rows.append({
            "transition": t,
            "frame0": t,
            "frame1": t + 1,
            "visible_tracks": int(len(ids)),
            "mean_epe": float(
                np.mean(epe)
            ),
            "median_epe": float(
                np.median(epe)
            ),
            "p95_epe": float(
                np.percentile(epe, 95)
            ),
            "mean_magnitude_error": float(
                np.mean(mag_err)
            ),
        })

    pooled_epe = np.concatenate(
        pooled_epe
    )

    pooled_mag = np.concatenate(
        pooled_mag
    )

    transition_mean_epe = np.array(
        [
            r["mean_epe"]
            for r in transition_rows
        ],
        dtype=np.float64,
    )

    transition_mean_mag = np.array(
        [
            r["mean_magnitude_error"]
            for r in transition_rows
        ],
        dtype=np.float64,
    )

    return {
        "transition_rows":
            transition_rows,

        "transition_mean_epe":
            transition_mean_epe,

        "transition_mean_mag":
            transition_mean_mag,

        "summary": {
            "transition_level_epe":
                stats(
                    transition_mean_epe
                ),

            "transition_level_magnitude_error":
                stats(
                    transition_mean_mag
                ),

            "pooled_track_transition_epe":
                stats(
                    pooled_epe
                ),

            "pooled_track_transition_magnitude_error":
                stats(
                    pooled_mag
                ),
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        choices=["santa", "tree"],
        required=True,
    )

    parser.add_argument(
        "--root",
        default="/workspace/DeformTransport",
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    case = args.case
    cfg = CASES[case]

    root = Path(args.root)
    outdir = Path(args.out)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cuda:0")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    print(
        f"CASE={case}",
        flush=True,
    )

    print(
        f"CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    print(
        f"DEVICE={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    tracks = (
        np.load(
            root / cfg["tracks"]
        )[0]
        .astype(np.float32)
    )

    vis = (
        np.load(
            root / cfg["vis"]
        )[0]
        .astype(bool)
    )

    n = cfg["expected_tracks"]

    assert tracks.shape == (
        81,
        n,
        2,
    )

    assert vis.shape == (
        81,
        n,
    )

    # ----------------------------------------
    # Pre-register actual visible observation
    # counts before seeing method predictions.
    # ----------------------------------------

    visible_per_transition = []

    for t in range(80):
        valid = (
            vis[t]
            & vis[t + 1]
        )

        visible_per_transition.append(
            int(valid.sum())
        )

    if min(
        visible_per_transition
    ) <= 0:
        raise RuntimeError(
            "At least one transition has "
            "zero visible material tracks"
        )

    print(
        "VISIBLE_TRACKS_PER_TRANSITION "
        f"min={min(visible_per_transition)} "
        f"max={max(visible_per_transition)} "
        f"mean={np.mean(visible_per_transition):.3f}",
        flush=True,
    )

    weights = (
        Raft_Large_Weights
        .C_T_SKHT_V2
    )

    check_weights_cached(
        weights
    )

    transforms = weights.transforms()

    print(
        "Loading RAFT-Large...",
        flush=True,
    )

    model = raft_large(
        weights=weights,
        progress=False,
    ).to(device)

    model.eval()

    print(
        "RAFT model loaded.",
        flush=True,
    )

    method_results = {}

    for method in [
        "rw",
        "correct",
        "shuffled",
    ]:

        path = root / cfg[method]

        print(
            "\n==========================",
            flush=True,
        )

        print(
            f"{case}: {method}",
            flush=True,
        )

        print(
            f"VIDEO={path}",
            flush=True,
        )

        print(
            "==========================",
            flush=True,
        )

        frames = read_video_raft_grid(
            path
        )

        print(
            "Video decoded and mapped "
            "to 416x240.",
            flush=True,
        )

        flows = infer_flows(
            model,
            transforms,
            frames,
            device,
        )

        del frames

        evaluated = evaluate_method(
            flows,
            tracks,
            vis,
        )

        del flows

        torch.cuda.empty_cache()

        method_results[method] = (
            evaluated
        )

        print(
            json.dumps(
                evaluated["summary"],
                indent=2,
            ),
            flush=True,
        )

    rw = (
        method_results["rw"]
        ["transition_mean_epe"]
    )

    correct = (
        method_results["correct"]
        ["transition_mean_epe"]
    )

    shuffled = (
        method_results["shuffled"]
        ["transition_mean_epe"]
    )

    assert len(rw) == 80
    assert len(correct) == 80
    assert len(shuffled) == 80

    diff_rw = rw - correct
    diff_shuffled = (
        shuffled - correct
    )

    ci_rw = bootstrap_mean_ci(
        diff_rw
    )

    ci_shuffled = bootstrap_mean_ci(
        diff_shuffled
    )

    if ci_rw[0] > 0:
        system_decision = "WIN"
    elif ci_rw[1] < 0:
        system_decision = "LOSS"
    else:
        system_decision = "TIE"

    report = {
        "case": case,

        "metric_name":
            "TC-ME: Track-Conditioned "
            "Motion Error",

        "reference":
            "persistent material trajectory "
            "projected displacement; "
            "NOT simulation-RAFT flow",

        "protocol": {
            "frame_alignment":
                "frame-index",

            "generated_domain":
                "832x464",

            "raft_input":
                "416x240",

            "realwonder_mapping":
                "832x480 -> bicubic 832x464 "
                "-> area 416x240",

            "wanmove_mapping":
                "832x464 -> area 416x240",

            "track_mapping":
                "x_raft=x_480*0.5; "
                "y_raft=y_480*0.5",

            "reference_displacement":
                "(P[t+1]-P[t])*0.5",

            "sampling":
                "bilinear RAFT-flow sampling "
                "at true source material point",

            "validity":
                "visibility[t] AND "
                "visibility[t+1]",

            "bootstrap": {
                "unit":
                    "transition",

                "n_transitions":
                    80,

                "resamples":
                    BOOT_N,

                "seed":
                    BOOT_SEED,
            },
        },

        "selected_tracks":
            int(n),

        "visible_tracks_per_transition": {
            "min":
                int(min(
                    visible_per_transition
                )),

            "max":
                int(max(
                    visible_per_transition
                )),

            "mean":
                float(
                    np.mean(
                        visible_per_transition
                    )
                ),
        },

        "methods": {
            method:
                method_results[method]
                ["summary"]

            for method in [
                "rw",
                "correct",
                "shuffled",
            ]
        },

        "system_test": {
            "definition":
                "RealWonder mean TC-ME "
                "- Correct mean TC-ME "
                "per transition; "
                "positive means Correct better",

            "paired_mean_difference":
                float(
                    np.mean(diff_rw)
                ),

            "paired_median_difference":
                float(
                    np.median(diff_rw)
                ),

            "fraction_rw_worse_than_correct":
                float(
                    np.mean(
                        diff_rw > 0
                    )
                ),

            "bootstrap_95_ci":
                ci_rw,

            "decision":
                system_decision,
        },

        "correct_vs_shuffled": {
            "definition":
                "Shuffled - Correct "
                "per-transition mean TC-ME",

            "paired_mean_difference":
                float(
                    np.mean(
                        diff_shuffled
                    )
                ),

            "paired_median_difference":
                float(
                    np.median(
                        diff_shuffled
                    )
                ),

            "fraction_shuffled_worse":
                float(
                    np.mean(
                        diff_shuffled > 0
                    )
                ),

            "bootstrap_95_ci":
                ci_shuffled,
        },
    }

    # ----------------------------------------
    # Per-transition CSV
    # ----------------------------------------

    csv_path = (
        outdir
        / f"{case}_raft_tcme_per_transition.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "transition",
            "visible_tracks",

            "rw_mean_epe",
            "correct_mean_epe",
            "shuffled_mean_epe",

            "rw_magnitude_error",
            "correct_magnitude_error",
            "shuffled_magnitude_error",

            "rw_minus_correct_epe",
            "shuffled_minus_correct_epe",
        ])

        for t in range(80):

            writer.writerow([
                t,

                method_results["rw"]
                ["transition_rows"][t]
                ["visible_tracks"],

                rw[t],
                correct[t],
                shuffled[t],

                method_results["rw"]
                ["transition_mean_mag"][t],

                method_results["correct"]
                ["transition_mean_mag"][t],

                method_results["shuffled"]
                ["transition_mean_mag"][t],

                diff_rw[t],
                diff_shuffled[t],
            ])

    report_path = (
        outdir
        / f"{case}_raft_tcme_report.json"
    )

    with open(
        report_path,
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        "\n==========================",
        flush=True,
    )

    print(
        f"{case.upper()} FINAL TC-ME",
        flush=True,
    )

    print(
        "==========================",
        flush=True,
    )

    print(
        json.dumps(
            report,
            indent=2,
        ),
        flush=True,
    )

    print(
        f"\nREPORT={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
