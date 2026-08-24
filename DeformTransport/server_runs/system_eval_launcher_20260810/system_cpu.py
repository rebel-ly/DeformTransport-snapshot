import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


OFF = np.arange(8, dtype=np.float32) - 3.5
ANCHORS = list(range(4, 81, 4))

BOOT_SEED = 0
BOOT_N = 10000


CASES = {
    "santa": {
        "source":
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/"
            "resized_input_image.png",

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

        "proxy":
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/"
            "simulation.mp4",

        "tracks":
            "server_runs/wan_move_bridge/"
            "20260809_010015__santa_correct_tracks/"
            "santa_material_tracks_correct.npy",

        "vis":
            "server_runs/wan_move_bridge/"
            "20260809_010015__santa_correct_tracks/"
            "santa_material_visibility_correct.npy",

        "expect": {
            "n": 1277,
            "valid_tracks": 1277,
            "obs": 25540,
            "lab_correct": 21.271405,
            "lab_shuffled": 31.804367,
            "rgb_correct": 0.145618,
            "rgb_shuffled": 0.218374,
        },
    },

    "tree": {
        "source":
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/"
            "resized_input_image.png",

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

        "proxy":
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/"
            "simulation.mp4",

        "tracks":
            "server_runs/wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_tracks_correct.npy",

        "vis":
            "server_runs/wan_move_bridge/"
            "20260810_072215__tree_correct_tracks/"
            "tree_material_visibility_correct.npy",

        "expect": {
            "n": 713,
            "valid_tracks": 709,
            "obs": 8743,
            "lab_correct": 18.963633,
            "lab_shuffled": 19.862847,
            "rgb_correct": 0.144800,
            "rgb_shuffled": 0.155809,
        },
    },
}


def read_rgb_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")

    return (
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        .astype(np.float32)
        / 255.0
    )


def to_common(rgb_u8):
    """
    Common evaluation domain:
      native Wan-Move : 832x464
      RealWonder/proxy: 832x480 -> bicubic -> 832x464
    """
    h, w = rgb_u8.shape[:2]

    x = (
        torch.from_numpy(rgb_u8)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )

    if (h, w) == (480, 832):
        x = F.interpolate(
            x,
            size=(464, 832),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        ).clamp_(0, 1)

    elif (h, w) != (464, 832):
        raise AssertionError(
            f"Unexpected video resolution {(h,w)}"
        )

    return (
        x[0]
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
    )


def read_video_common(path):
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(to_common(rgb))

    cap.release()

    if len(frames) != 81:
        raise AssertionError(
            f"{path}: expected 81 frames, got {len(frames)}"
        )

    arr = np.stack(frames, axis=0).astype(np.float32)

    if arr.shape != (81, 464, 832, 3):
        raise AssertionError(
            f"{path}: unexpected common shape {arr.shape}"
        )

    return arr


def sample_patches(img, centers):
    """
    8x8 exact bilinear patch centered at XY.
    Offsets are frozen: -3.5 ... +3.5
    """

    H, W = img.shape[:2]
    centers = np.asarray(centers, dtype=np.float32)

    xs = centers[:, 0, None, None] + OFF[None, None, :]
    ys = centers[:, 1, None, None] + OFF[None, :, None]

    xs = np.broadcast_to(xs, (len(centers), 8, 8))
    ys = np.broadcast_to(ys, (len(centers), 8, 8))

    valid = (
        (xs.min(axis=(1, 2)) >= 0)
        & (ys.min(axis=(1, 2)) >= 0)
        & (xs.max(axis=(1, 2)) <= W - 1)
        & (ys.max(axis=(1, 2)) <= H - 1)
    )

    if not np.all(valid):
        raise RuntimeError(
            "sample_patches received out-of-bound centers"
        )

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)

    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = (xs - x0)[..., None]
    wy = (ys - y0)[..., None]

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    out = (
        (1 - wx) * (1 - wy) * Ia
        + wx * (1 - wy) * Ib
        + (1 - wx) * wy * Ic
        + wx * wy * Id
    )

    return out.astype(np.float32)


def patch_mean_lab(patches):
    """
    OpenCV float RGB [0,1] -> CIE Lab:
      L approximately [0,100]
      a/b approximately [-127,127]
    """

    n = patches.shape[0]

    flat = (
        patches
        .reshape(n * 8, 8, 3)
        .astype(np.float32)
    )

    lab = cv2.cvtColor(
        flat,
        cv2.COLOR_RGB2LAB
    ).reshape(n, 8, 8, 3)

    return lab.mean(axis=(1, 2))


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

    rng = np.random.default_rng(BOOT_SEED)
    values = []

    chunk = 500

    for start in range(0, BOOT_N, chunk):
        k = min(chunk, BOOT_N - start)

        idx = rng.integers(
            0,
            n,
            size=(k, n),
        )

        values.append(
            d[idx].mean(axis=1)
        )

    values = np.concatenate(values)

    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def aggregate_obs(rows, n_tracks, key):
    sums = np.zeros(n_tracks, dtype=np.float64)
    counts = np.zeros(n_tracks, dtype=np.int64)

    for row in rows:
        ids = row["ids"]
        vals = row[key]

        np.add.at(sums, ids, vals)
        np.add.at(counts, ids, 1)

    out = np.full(
        n_tracks,
        np.nan,
        dtype=np.float64,
    )

    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]

    return out, counts


def eval_case(root, name, cfg, outdir):

    P = lambda rel: root / rel

    print(f"[{name}] loading source...", flush=True)

    source = read_rgb_image(P(cfg["source"]))

    if source.shape[:2] != (480, 832):
        raise AssertionError(source.shape)

    tracks = (
        np.load(P(cfg["tracks"]))[0]
        .astype(np.float32)
    )

    vis = (
        np.load(P(cfg["vis"]))[0]
        .astype(bool)
    )

    n = tracks.shape[1]

    assert tracks.shape == (81, n, 2)
    assert vis.shape == (81, n)
    assert n == cfg["expect"]["n"]

    print(f"[{name}] loading 4 videos...", flush=True)

    videos = {
        k: read_video_common(P(cfg[k]))
        for k in [
            "rw",
            "correct",
            "shuffled",
            "proxy",
        ]
    }

    source_centers = tracks[0]

    source_valid = (
        (source_centers[:, 0] - 3.5 >= 0)
        & (source_centers[:, 0] + 3.5 <= 831)
        & (source_centers[:, 1] - 3.5 >= 0)
        & (source_centers[:, 1] + 3.5 <= 479)
    )

    src_patch = np.full(
        (n, 8, 8, 3),
        np.nan,
        dtype=np.float32,
    )

    good = np.where(source_valid)[0]

    src_patch[good] = sample_patches(
        source,
        source_centers[good],
    )

    src_lab = np.full(
        (n, 3),
        np.nan,
        dtype=np.float32,
    )

    src_lab[good] = patch_mean_lab(
        src_patch[good]
    )

    rows = {
        m: []
        for m in [
            "rw",
            "correct",
            "shuffled",
        ]
    }

    proxy_rows = {
        m: []
        for m in [
            "rw",
            "correct",
            "shuffled",
        ]
    }

    total_obs = 0

    for t in ANCHORS:

        centers = tracks[t].copy()

        # 832x480 track domain -> 832x464 output domain
        centers[:, 1] *= 464.0 / 480.0

        future_valid = (
            (centers[:, 0] - 3.5 >= 0)
            & (centers[:, 0] + 3.5 <= 831)
            & (centers[:, 1] - 3.5 >= 0)
            & (centers[:, 1] + 3.5 <= 463)
        )

        valid = (
            vis[t]
            & source_valid
            & future_valid
            & np.isfinite(centers).all(axis=1)
        )

        ids = np.where(valid)[0]

        total_obs += len(ids)

        proxy_patch = sample_patches(
            videos["proxy"][t],
            centers[ids],
        )

        for method in [
            "rw",
            "correct",
            "shuffled",
        ]:

            patch = sample_patches(
                videos[method][t],
                centers[ids],
            )

            lab = patch_mean_lab(patch)

            lab_error = np.linalg.norm(
                lab - src_lab[ids],
                axis=1,
            )

            rgb_l1 = np.abs(
                patch - src_patch[ids]
            ).mean(axis=(1, 2, 3))

            proxy_l1 = np.abs(
                patch - proxy_patch
            ).mean(axis=(1, 2, 3))

            rows[method].append({
                "t": t,
                "ids": ids,
                "lab": lab_error,
                "rgb": rgb_l1,
            })

            proxy_rows[method].append({
                "t": t,
                "ids": ids,
                "pl1": proxy_l1,
            })

    result = {
        "case": name,
        "anchors": ANCHORS,
        "selected_tracks": n,
        "total_valid_anchor_observations":
            int(total_obs),
        "methods": {},
        "proxy": {},
    }

    pertrack = {}
    counts_ref = None

    for method in [
        "rw",
        "correct",
        "shuffled",
    ]:

        lab, count = aggregate_obs(
            rows[method],
            n,
            "lab",
        )

        rgb, count2 = aggregate_obs(
            rows[method],
            n,
            "rgb",
        )

        pl1, count3 = aggregate_obs(
            proxy_rows[method],
            n,
            "pl1",
        )

        assert np.array_equal(count, count2)
        assert np.array_equal(count, count3)

        if counts_ref is None:
            counts_ref = count
        else:
            assert np.array_equal(
                counts_ref,
                count,
            )

        valid = count > 0

        pertrack[method] = {
            "lab": lab,
            "rgb": rgb,
            "pl1": pl1,
            "cnt": count,
        }

        result["methods"][method] = {
            "tc_mar_lab":
                stats(lab[valid]),

            "tc_mar_rgb_l1":
                stats(rgb[valid]),
        }

        result["proxy"][method] = {
            "track_support_l1":
                stats(pl1[valid])
        }

        result["proxy"][method][
            "global_future_mae"
        ] = float(
            np.mean(
                np.abs(
                    videos[method][1:]
                    - videos["proxy"][1:]
                )
            )
        )

    valid = counts_ref > 0

    result["valid_tracks"] = int(
        valid.sum()
    )

    result["excluded_tracks"] = int(
        (~valid).sum()
    )

    exp = cfg["expect"]

    if (
        int(valid.sum())
        != exp["valid_tracks"]
        or total_obs != exp["obs"]
    ):
        raise AssertionError(
            f"{name}: observation contract mismatch: "
            f"valid={valid.sum()}, "
            f"obs={total_obs}; "
            f"expected="
            f"{exp['valid_tracks']},"
            f"{exp['obs']}"
        )

    # --------------------------------------------------
    # Frozen Correct/Shuffled reproduction sanity check
    # --------------------------------------------------

    reproduction = {
        "lab_correct_abs_err":
            abs(
                result["methods"]["correct"]
                ["tc_mar_lab"]["mean"]
                - exp["lab_correct"]
            ),

        "lab_shuffled_abs_err":
            abs(
                result["methods"]["shuffled"]
                ["tc_mar_lab"]["mean"]
                - exp["lab_shuffled"]
            ),

        "rgb_correct_abs_err":
            abs(
                result["methods"]["correct"]
                ["tc_mar_rgb_l1"]["mean"]
                - exp["rgb_correct"]
            ),

        "rgb_shuffled_abs_err":
            abs(
                result["methods"]["shuffled"]
                ["tc_mar_rgb_l1"]["mean"]
                - exp["rgb_shuffled"]
            ),
    }

    result["frozen_reproduction"] = (
        reproduction
    )

    if max(
        reproduction.values()
    ) > 0.002:

        raise AssertionError(
            f"{name}: frozen TC-MAR reproduction "
            f"mismatch: {reproduction}"
        )

    # --------------------------------------------------
    # System test: RealWonder vs Correct
    # --------------------------------------------------

    for metric, key in [
        ("tc_mar_lab", "lab"),
        ("tc_mar_rgb_l1", "rgb"),
    ]:

        rw = (
            pertrack["rw"][key][valid]
        )

        correct = (
            pertrack["correct"][key][valid]
        )

        shuffled = (
            pertrack["shuffled"][key][valid]
        )

        diff_rw = rw - correct
        diff_shuffled = (
            shuffled - correct
        )

        ci = bootstrap_mean_ci(diff_rw)

        decision = (
            "WIN"
            if ci[0] > 0
            else (
                "LOSS"
                if ci[1] < 0
                else "TIE"
            )
        )

        result[
            f"{metric}_system_test"
        ] = {
            "definition":
                "RealWonder - Correct; "
                "positive means Correct lower error",

            "paired_mean_difference":
                float(diff_rw.mean()),

            "paired_median_difference":
                float(
                    np.median(diff_rw)
                ),

            "fraction_rw_worse_than_correct":
                float(
                    np.mean(diff_rw > 0)
                ),

            "bootstrap_95_ci":
                ci,

            "decision":
                decision,

            "correct_vs_shuffled_reproduction":
                {
                    "paired_mean_difference":
                        float(
                            diff_shuffled.mean()
                        ),

                    "bootstrap_95_ci":
                        bootstrap_mean_ci(
                            diff_shuffled
                        ),
                },
        }

    # --------------------------------------------------
    # Supporting proxy comparison
    # --------------------------------------------------

    rw_proxy = (
        pertrack["rw"]["pl1"][valid]
    )

    correct_proxy = (
        pertrack["correct"]["pl1"][valid]
    )

    diff_proxy = (
        rw_proxy - correct_proxy
    )

    result["proxy"][
        "rw_vs_correct_track_support"
    ] = {
        "definition":
            "RealWonder proxy-L1 - Correct proxy-L1; "
            "supporting diagnostic only",

        "paired_mean_difference":
            float(diff_proxy.mean()),

        "paired_median_difference":
            float(
                np.median(diff_proxy)
            ),

        "bootstrap_95_ci":
            bootstrap_mean_ci(
                diff_proxy
            ),
    }

    # --------------------------------------------------
    # Per-track CSV
    # --------------------------------------------------

    csv_path = (
        outdir
        / f"{name}_per_track.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "track_id",
            "valid_anchor_count",

            "rw_lab",
            "correct_lab",
            "shuffled_lab",

            "rw_rgb_l1",
            "correct_rgb_l1",
            "shuffled_rgb_l1",

            "rw_proxy_l1",
            "correct_proxy_l1",
            "shuffled_proxy_l1",
        ])

        for i in range(n):

            if not valid[i]:
                continue

            writer.writerow([
                i,
                int(counts_ref[i]),

                pertrack["rw"]["lab"][i],
                pertrack["correct"]["lab"][i],
                pertrack["shuffled"]["lab"][i],

                pertrack["rw"]["rgb"][i],
                pertrack["correct"]["rgb"][i],
                pertrack["shuffled"]["rgb"][i],

                pertrack["rw"]["pl1"][i],
                pertrack["correct"]["pl1"][i],
                pertrack["shuffled"]["pl1"][i],
            ])

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/workspace/DeformTransport",
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "protocol": {
            "common_domain":
                "832x464",

            "rw_proxy_transform":
                "float RGB [0,1] -> "
                "torch bicubic 464x832, "
                "align_corners=False, "
                "antialias=False",

            "frame_alignment":
                "frame-index; "
                "fps metadata ignored",

            "tc_mar_patch":
                "8x8, offsets -3.5..+3.5, "
                "exact bilinear",

            "anchors":
                ANCHORS,

            "bootstrap": {
                "unit":
                    "material track",
                "n":
                    BOOT_N,
                "seed":
                    BOOT_SEED,
            },

            "proxy_wording":
                "geometry-aligned simulation proxy; "
                "NOT real future RGB ground truth",
        },

        "cases": {},
    }

    for name, cfg in CASES.items():

        print(
            "\n==============================",
            flush=True,
        )

        print(
            f"CPU SYSTEM EVAL: {name}",
            flush=True,
        )

        print(
            "==============================",
            flush=True,
        )

        report["cases"][name] = (
            eval_case(
                root,
                name,
                cfg,
                outdir,
            )
        )

        print(
            json.dumps(
                report["cases"][name],
                indent=2,
            ),
            flush=True,
        )

    report_path = (
        outdir
        / "cpu_report.json"
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
        "\nCPU_REPORT =",
        report_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
