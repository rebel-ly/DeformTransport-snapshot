from pathlib import Path
import json
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


RUN = Path(
    "/workspace/DeformTransport/server_runs/"
    "sand_house_cached_sim_20260808_174356"
)

CTD = (
    RUN
    / "ctd_dev"
    / "20260808_235818__counterfactual_transport_debiasing"
)

OUT = CTD / "final_ctd_evaluation_seed0"
OUT.mkdir(parents=True, exist_ok=True)


TARGET_DIR = (
    RUN
    / "v1_frozen"
    / "20260808_201448__sand_house_final_sim_seed0"
    / "frames"
)

MASK_ART = (
    RUN
    / "v1_frozen"
    / "20260808_203011__frozen_quality075_ramp4"
    / "sand_house_quality075_ramp4_full_generation.pt"
)


VIDEOS = {
    "REALWONDER": (
        RUN
        / "v1_frozen"
        / "20260808_202951__sand_house_realwonder_baseline_seed0"
        / "sand_house_realwonder_seed0.mp4"
    ),

    "V1_CORRECT": (
        RUN
        / "v1_frozen"
        / "20260808_203833__sand_house_frozen_correct_condition075_seed0"
        / "sand_house_frozen_correct_condition075_seed0.mp4"
    ),

    "V1_SHUFFLED": (
        RUN
        / "v1_frozen"
        / "20260808_204447__sand_house_frozen_shuffled_energy_matched_seed0"
        / "sand_house_frozen_shuffled_energy_matched_seed0.mp4"
    ),

    "CTD_DIFF": (
        CTD
        / "20260809_000033__sand_house_ctd_diff_seed0"
        / "sand_house_ctd_diff_seed0.mp4"
    ),

    "CTD_ORTH": (
        CTD
        / "20260809_000809__sand_house_ctd_orth_seed0"
        / "sand_house_ctd_orth_seed0.mp4"
    ),
}


# Historical values already frozen and reproduced by the
# previous unified evaluator.
FROZEN = {
    "REALWONDER": {
        "global_MAE": 12.088175773620605,
        "local_L1": 34.30699157714844,
    },

    "V1_CORRECT": {
        "global_MAE": 12.121809959411621,
        "local_L1": 34.35251235961914,
    },

    "V1_SHUFFLED": {
        "global_MAE": 12.279459953308105,
        "local_L1": 35.113548278808594,
    },
}


def load_video(path):
    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)

    cap.release()

    if len(frames) != 165:
        raise RuntimeError(
            f"{path}: expected 165 frames, got {len(frames)}"
        )

    arr = np.stack(frames)

    if arr.shape != (165, 480, 832, 3):
        raise RuntimeError(
            f"{path}: unexpected shape {arr.shape}"
        )

    return arr


def load_target():
    paths = sorted(
        TARGET_DIR.glob("*.png")
    )

    if len(paths) != 165:
        raise RuntimeError(
            f"expected 165 target frames, got {len(paths)}"
        )

    frames = [
        cv2.imread(str(p), cv2.IMREAD_COLOR)
        for p in paths
    ]

    arr = np.stack(frames)

    if arr.shape != (165, 480, 832, 3):
        raise RuntimeError(
            f"target shape {arr.shape}"
        )

    return arr


target = load_target()


art = torch.load(
    MASK_ART,
    map_location="cpu",
    weights_only=True,
)

mask = art["transport_mask"].bool()
indices = art["latent_frame_indices"].long()

if tuple(mask.shape) != (42, 1, 60, 104):
    raise RuntimeError(
        f"unexpected mask shape {tuple(mask.shape)}"
    )

expected_indices = torch.arange(
    0,
    165,
    4,
)

if not torch.equal(
    indices,
    expected_indices,
):
    raise RuntimeError(
        "latent index mismatch"
    )


future_ids = indices[1:].tolist()

future_masks = F.interpolate(
    mask[1:].float(),
    size=(480, 832),
    mode="nearest",
)[:, 0].bool().numpy()


def evaluate(video):
    x = video.astype(np.float32)
    y = target.astype(np.float32)

    diff = x - y

    mae = float(
        np.abs(diff).mean()
    )

    mse = float(
        (diff * diff).mean()
    )

    psnr = float(
        10.0
        * math.log10(
            255.0 ** 2
            / max(mse, 1e-12)
        )
    )


    local_abs_sum = 0.0
    local_sq_sum = 0.0
    local_count = 0

    bg_abs_sum = 0.0
    bg_count = 0

    per_anchor = []


    for k, frame_id in enumerate(
        future_ids
    ):
        m = future_masks[k]

        d = diff[frame_id]

        local = d[m]
        bg = d[~m]

        local_abs_sum += float(
            np.abs(local).sum(
                dtype=np.float64
            )
        )

        local_sq_sum += float(
            (local * local).sum(
                dtype=np.float64
            )
        )

        local_count += int(
            local.size
        )

        bg_abs_sum += float(
            np.abs(bg).sum(
                dtype=np.float64
            )
        )

        bg_count += int(
            bg.size
        )


        anchor_l1 = float(
            np.abs(local).mean()
        )

        anchor_mse = float(
            (local * local).mean()
        )

        per_anchor.append({
            "frame":
                int(frame_id),

            "local_L1":
                anchor_l1,

            "local_MSE":
                anchor_mse,
        })


    local_l1 = (
        local_abs_sum
        / local_count
    )

    local_mse = (
        local_sq_sum
        / local_count
    )

    local_psnr = float(
        10.0
        * math.log10(
            255.0 ** 2
            / max(local_mse, 1e-12)
        )
    )

    background_l1 = (
        bg_abs_sum
        / bg_count
    )


    return {
        "global_MAE":
            mae,

        "global_MSE":
            mse,

        "global_PSNR":
            psnr,

        "local_L1":
            float(local_l1),

        "local_MSE":
            float(local_mse),

        "local_PSNR":
            local_psnr,

        "background_L1":
            float(background_l1),

        "per_anchor":
            per_anchor,
    }


results = {}

for name, path in VIDEOS.items():

    print()
    print(
        f"===== {name} ====="
    )

    result = evaluate(
        load_video(path)
    )

    results[name] = result

    for key in (
        "global_MAE",
        "global_MSE",
        "global_PSNR",
        "local_L1",
        "local_MSE",
        "local_PSNR",
        "background_L1",
    ):
        print(
            key,
            "=",
            result[key],
        )


# ------------------------------------------------------------
# Reproduction guard.
# ------------------------------------------------------------

print()
print(
    "===== HISTORICAL REPRODUCTION ====="
)

for name in (
    "REALWONDER",
    "V1_CORRECT",
    "V1_SHUFFLED",
):

    dg = abs(
        results[name]["global_MAE"]
        - FROZEN[name]["global_MAE"]
    )

    dl = abs(
        results[name]["local_L1"]
        - FROZEN[name]["local_L1"]
    )

    ok = (
        dg <= 0.01
        and dl <= 0.01
    )

    print(
        name,
        "global_diff=",
        dg,
        "local_diff=",
        dl,
        "pass=",
        ok,
    )

    if not ok:
        raise RuntimeError(
            "historical appearance reproduction failed"
        )


def improvement(new, old):
    return float(
        100.0
        * (old - new)
        / old
    )


comparisons = {}

for method in (
    "CTD_DIFF",
    "CTD_ORTH",
):

    comparisons[
        f"{method}_vs_RW_local_L1_improvement_percent"
    ] = improvement(
        results[method]["local_L1"],
        results["REALWONDER"]["local_L1"],
    )

    comparisons[
        f"{method}_vs_V1_correct_local_L1_improvement_percent"
    ] = improvement(
        results[method]["local_L1"],
        results["V1_CORRECT"]["local_L1"],
    )

    comparisons[
        f"{method}_vs_RW_global_MAE_improvement_percent"
    ] = improvement(
        results[method]["global_MAE"],
        results["REALWONDER"]["global_MAE"],
    )

    comparisons[
        f"{method}_vs_V1_correct_global_MAE_improvement_percent"
    ] = improvement(
        results[method]["global_MAE"],
        results["V1_CORRECT"]["global_MAE"],
    )


def anchor_wins(a, b):
    aa = results[a]["per_anchor"]
    bb = results[b]["per_anchor"]

    wins = []

    for x, y in zip(aa, bb):
        if (
            x["local_L1"]
            < y["local_L1"]
        ):
            wins.append(
                x["frame"]
            )

    return wins


anchor_wins_result = {}

for method in (
    "CTD_DIFF",
    "CTD_ORTH",
):
    anchor_wins_result[
        f"{method}_vs_RW"
    ] = anchor_wins(
        method,
        "REALWONDER",
    )

    anchor_wins_result[
        f"{method}_vs_V1_correct"
    ] = anchor_wins(
        method,
        "V1_CORRECT",
    )


anchor_counts = {
    k: len(v)
    for k, v
    in anchor_wins_result.items()
}


global_ranking = sorted(
    results,
    key=lambda x:
        results[x]["global_MAE"],
)

local_ranking = sorted(
    results,
    key=lambda x:
        results[x]["local_L1"],
)


report = {
    "experiment":
        "SandHouse CTD appearance evaluation",

    "warning":
        "simulation RGB is a geometry-aligned engineering proxy, not real future-video ground truth",

    "results":
        results,

    "comparisons":
        comparisons,

    "anchor_win_counts":
        anchor_counts,

    "anchor_win_frames":
        anchor_wins_result,

    "global_MAE_ranking":
        global_ranking,

    "local_L1_ranking":
        local_ranking,
}


REPORT = OUT / "appearance_report.json"

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "===== COMPARISONS ====="
)

print(
    json.dumps(
        comparisons,
        indent=2,
    )
)

print()
print(
    "===== ANCHOR WINS ====="
)

print(
    json.dumps(
        anchor_counts,
        indent=2,
    )
)

print()
print(
    "GLOBAL RANKING =",
    global_ranking,
)

print(
    "LOCAL RANKING =",
    local_ranking,
)

print()
print(
    "saved =",
    REPORT,
)

print(
    "CTD_APPEARANCE_EVALUATION_OK"
)
