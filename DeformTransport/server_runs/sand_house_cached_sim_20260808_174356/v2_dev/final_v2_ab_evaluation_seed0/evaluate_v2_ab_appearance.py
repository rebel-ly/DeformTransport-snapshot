from pathlib import Path
import json
import math
import cv2
import numpy as np
import torch


RUN = Path(
    "/workspace/DeformTransport/server_runs/"
    "sand_house_cached_sim_20260808_174356"
)

OUT = RUN / "v2_dev/final_v2_ab_evaluation_seed0"
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

    "V2A_CORRECT": (
        RUN
        / "v2_dev"
        / "20260808_231107__sand_house_v2A_correct_seed0"
        / "sand_house_v2A_correct_seed0.mp4"
    ),

    "V2A_SHUFFLED": (
        RUN
        / "v2_dev"
        / "20260808_231835__sand_house_v2A_shuffled_seed0"
        / "sand_house_v2A_shuffled_seed0.mp4"
    ),

    "V2B_CORRECT": (
        RUN
        / "v2_dev"
        / "20260808_225223__sand_house_v2B_correct_seed0"
        / "sand_house_v2B_correct_seed0.mp4"
    ),

    "V2B_SHUFFLED": (
        RUN
        / "v2_dev"
        / "20260808_230335__sand_house_v2B_shuffled_seed0"
        / "sand_house_v2B_shuffled_seed0.mp4"
    ),
}


# ============================================================
# Frozen historical metrics.
# Used ONLY as implementation-reproduction checks,
# never for changing V2 formulas.
# ============================================================

FROZEN_GLOBAL = {
    "REALWONDER": {
        "MAE": 12.088175773620605,
        "MSE": 815.610107421875,
    },
    "V1_CORRECT": {
        "MAE": 12.121809959411621,
        "MSE": 815.603271484375,
    },
    "V1_SHUFFLED": {
        "MAE": 12.279459953308105,
        "MSE": 836.7311401367188,
    },
}

FROZEN_LOCAL = {
    "REALWONDER": {
        "local_L1": 34.30699157714844,
        "local_MSE": 2413.3642578125,
    },
    "V1_CORRECT": {
        "local_L1": 34.35251235961914,
        "local_MSE": 2406.968017578125,
    },
    "V1_SHUFFLED": {
        "local_L1": 35.113548278808594,
        "local_MSE": 2507.9248046875,
    },
}


def read_video(path):
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
            f"{path}: expected 165 frames, "
            f"got {len(frames)}"
        )

    arr = np.stack(frames, axis=0)

    if arr.shape != (165, 480, 832, 3):
        raise RuntimeError(
            f"{path}: bad shape {arr.shape}"
        )

    return arr


def read_targets():
    paths = sorted(
        TARGET_DIR.glob("*.png")
    )

    if len(paths) != 165:
        raise RuntimeError(
            f"expected 165 target frames, "
            f"got {len(paths)}"
        )

    arr = np.stack(
        [
            cv2.imread(
                str(path),
                cv2.IMREAD_COLOR,
            )
            for path in paths
        ],
        axis=0,
    )

    if arr.shape != (165, 480, 832, 3):
        raise RuntimeError(
            f"target shape={arr.shape}"
        )

    return arr


print("===== LOAD TARGET =====")

target = read_targets()

print("target =", target.shape)


print()
print("===== LOAD QUALITY MASK =====")

art = torch.load(
    MASK_ART,
    map_location="cpu",
    weights_only=True,
)

transport_mask = (
    art["transport_mask"]
    .bool()
)

latent_indices = (
    art["latent_frame_indices"]
    .long()
)

assert tuple(
    transport_mask.shape
) == (42, 1, 60, 104)

expected_indices = torch.arange(
    0,
    165,
    4,
)

if not torch.equal(
    latent_indices,
    expected_indices,
):
    raise RuntimeError(
        "latent frame indices mismatch"
    )

# Future only: slots 1..41.
future_indices = (
    latent_indices[1:]
    .tolist()
)

future_masks = (
    torch.nn.functional.interpolate(
        transport_mask[1:].float(),
        size=(480, 832),
        mode="nearest",
    )[:, 0]
    .bool()
    .numpy()
)

assert future_masks.shape == (
    41,
    480,
    832,
)


def evaluate(video):
    x = video.astype(np.float32)
    y = target.astype(np.float32)

    diff = x - y

    global_mae = float(
        np.mean(np.abs(diff))
    )

    global_mse = float(
        np.mean(diff * diff)
    )

    global_psnr = float(
        10.0
        * math.log10(
            (255.0 * 255.0)
            / max(global_mse, 1e-12)
        )
    )

    local_abs_sum = 0.0
    local_sq_sum = 0.0
    local_value_count = 0

    bg_abs_sum = 0.0
    bg_value_count = 0

    per_anchor = []

    for k, frame_id in enumerate(
        future_indices
    ):
        mask = future_masks[k]

        frame_diff = diff[frame_id]

        local = frame_diff[mask]

        background = frame_diff[~mask]

        if local.size == 0:
            raise RuntimeError(
                f"empty local mask slot={k+1}"
            )

        local_abs = np.abs(local)
        local_sq = local * local

        local_abs_sum += float(
            local_abs.sum(dtype=np.float64)
        )

        local_sq_sum += float(
            local_sq.sum(dtype=np.float64)
        )

        local_value_count += int(
            local.size
        )

        bg_abs_sum += float(
            np.abs(background).sum(
                dtype=np.float64
            )
        )

        bg_value_count += int(
            background.size
        )

        anchor_l1 = float(
            np.mean(local_abs)
        )

        anchor_mse = float(
            np.mean(local_sq)
        )

        anchor_psnr = float(
            10.0
            * math.log10(
                (255.0 * 255.0)
                / max(
                    anchor_mse,
                    1e-12,
                )
            )
        )

        per_anchor.append({
            "pixel_frame":
                int(frame_id),

            "local_L1":
                anchor_l1,

            "local_MSE":
                anchor_mse,

            "local_PSNR":
                anchor_psnr,
        })

    local_l1 = (
        local_abs_sum
        / local_value_count
    )

    local_mse = (
        local_sq_sum
        / local_value_count
    )

    local_psnr = float(
        10.0
        * math.log10(
            (255.0 * 255.0)
            / max(local_mse, 1e-12)
        )
    )

    background_l1 = (
        bg_abs_sum
        / bg_value_count
    )

    return {
        "global_MAE":
            float(global_mae),

        "global_MSE":
            float(global_mse),

        "global_PSNR":
            float(global_psnr),

        "local_L1":
            float(local_l1),

        "local_MSE":
            float(local_mse),

        "local_PSNR":
            float(local_psnr),

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

    video = read_video(path)

    result = evaluate(video)

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


# ============================================================
# Historical reproduction check.
# ============================================================

print()
print(
    "===== FROZEN V1 REPRODUCTION CHECK ====="
)

reproduction = {}

# Decoder/float reductions may differ at tiny levels.
# These tolerances are deliberately much smaller than
# any scientific V2 comparison we care about.

GLOBAL_MAE_TOL = 0.01
GLOBAL_MSE_TOL = 0.5
LOCAL_L1_TOL = 0.01
LOCAL_MSE_TOL = 0.5

for name in (
    "REALWONDER",
    "V1_CORRECT",
    "V1_SHUFFLED",
):
    r = results[name]

    dg_mae = abs(
        r["global_MAE"]
        - FROZEN_GLOBAL[name]["MAE"]
    )

    dg_mse = abs(
        r["global_MSE"]
        - FROZEN_GLOBAL[name]["MSE"]
    )

    dl_l1 = abs(
        r["local_L1"]
        - FROZEN_LOCAL[name]["local_L1"]
    )

    dl_mse = abs(
        r["local_MSE"]
        - FROZEN_LOCAL[name]["local_MSE"]
    )

    ok = (
        dg_mae <= GLOBAL_MAE_TOL
        and dg_mse <= GLOBAL_MSE_TOL
        and dl_l1 <= LOCAL_L1_TOL
        and dl_mse <= LOCAL_MSE_TOL
    )

    reproduction[name] = {
        "global_MAE_abs_diff":
            dg_mae,

        "global_MSE_abs_diff":
            dg_mse,

        "local_L1_abs_diff":
            dl_l1,

        "local_MSE_abs_diff":
            dl_mse,

        "pass":
            ok,
    }

    print(
        name,
        reproduction[name],
    )


if not all(
    x["pass"]
    for x in reproduction.values()
):
    raise RuntimeError(
        "Frozen V1 appearance metrics "
        "were not reproduced. "
        "Do not interpret V2 results."
    )


# ============================================================
# Comparisons.
# Lower is better for L1/MSE.
# ============================================================

def improvement(lower_metric_new, old):
    return float(
        100.0
        * (old - lower_metric_new)
        / old
    )


comparisons = {
    "V2A_correct_vs_V1_correct_local_L1_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["local_L1"],
            results["V1_CORRECT"]["local_L1"],
        ),

    "V2B_correct_vs_V1_correct_local_L1_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["local_L1"],
            results["V1_CORRECT"]["local_L1"],
        ),

    "V2A_correct_vs_RW_local_L1_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["local_L1"],
            results["REALWONDER"]["local_L1"],
        ),

    "V2B_correct_vs_RW_local_L1_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["local_L1"],
            results["REALWONDER"]["local_L1"],
        ),

    "V2A_correct_vs_V2A_shuffled_local_L1_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["local_L1"],
            results["V2A_SHUFFLED"]["local_L1"],
        ),

    "V2B_correct_vs_V2B_shuffled_local_L1_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["local_L1"],
            results["V2B_SHUFFLED"]["local_L1"],
        ),

    "V2A_correct_vs_V1_correct_global_MAE_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["global_MAE"],
            results["V1_CORRECT"]["global_MAE"],
        ),

    "V2B_correct_vs_V1_correct_global_MAE_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["global_MAE"],
            results["V1_CORRECT"]["global_MAE"],
        ),
}


# Per-anchor Correct-vs-control wins.

def anchor_wins(a, b):
    av = results[a]["per_anchor"]
    bv = results[b]["per_anchor"]

    assert len(av) == len(bv) == 41

    wins = []

    for x, y in zip(av, bv):
        assert (
            x["pixel_frame"]
            == y["pixel_frame"]
        )

        if (
            x["local_L1"]
            < y["local_L1"]
        ):
            wins.append(
                x["pixel_frame"]
            )

    return wins


anchor_summary = {
    "V2A_correct_wins_vs_V1_correct":
        anchor_wins(
            "V2A_CORRECT",
            "V1_CORRECT",
        ),

    "V2B_correct_wins_vs_V1_correct":
        anchor_wins(
            "V2B_CORRECT",
            "V1_CORRECT",
        ),

    "V2A_correct_wins_vs_V2A_shuffled":
        anchor_wins(
            "V2A_CORRECT",
            "V2A_SHUFFLED",
        ),

    "V2B_correct_wins_vs_V2B_shuffled":
        anchor_wins(
            "V2B_CORRECT",
            "V2B_SHUFFLED",
        ),

    "V2A_correct_wins_vs_RW":
        anchor_wins(
            "V2A_CORRECT",
            "REALWONDER",
        ),

    "V2B_correct_wins_vs_RW":
        anchor_wins(
            "V2B_CORRECT",
            "REALWONDER",
        ),
}

anchor_counts = {
    key:
        len(value)
    for key, value
    in anchor_summary.items()
}


ranking_global = sorted(
    results,
    key=lambda n:
        results[n]["global_MAE"],
)

ranking_local = sorted(
    results,
    key=lambda n:
        results[n]["local_L1"],
)


report = {
    "experiment":
        "SandHouse unified V1/V2 appearance evaluation",

    "warning":
        "simulation RGB is an engineering geometry-aligned proxy, not real future-video ground truth",

    "mask_definition":
        "frozen Quality transport_mask nearest-upsampled 60x104 -> 480x832",

    "future_anchor_count":
        41,

    "results":
        results,

    "frozen_v1_reproduction":
        reproduction,

    "comparisons":
        comparisons,

    "anchor_win_counts":
        anchor_counts,

    "anchor_win_frames":
        anchor_summary,

    "global_MAE_ranking":
        ranking_global,

    "local_L1_ranking":
        ranking_local,
}


report_path = (
    OUT
    / "appearance_report.json"
)

report_path.write_text(
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
    ranking_global,
)

print(
    "LOCAL RANKING =",
    ranking_local,
)

print(
    "saved =",
    report_path,
)

print(
    "UNIFIED_V2_APPEARANCE_EVALUATION_OK"
)
