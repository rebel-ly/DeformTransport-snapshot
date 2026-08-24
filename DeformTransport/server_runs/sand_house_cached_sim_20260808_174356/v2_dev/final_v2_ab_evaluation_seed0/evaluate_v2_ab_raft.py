from pathlib import Path
import json
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights,
)


RUN = Path(
    "/workspace/DeformTransport/server_runs/"
    "sand_house_cached_sim_20260808_174356"
)

OUT = RUN / "v2_dev/final_v2_ab_evaluation_seed0"
OUT.mkdir(parents=True, exist_ok=True)

FINAL = (
    RUN
    / "v1_frozen"
    / "20260808_201448__sand_house_final_sim_seed0"
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


# Frozen SandHouse V1 RAFT values.
FROZEN = {
    "REALWONDER": {
        "EPE_global":
            0.2949581444263458,

        "EPE_transport_region":
            1.0056567192077637,
    },

    "V1_CORRECT": {
        "EPE_global":
            0.29138579964637756,

        "EPE_transport_region":
            0.9911830425262451,
    },

    "V1_SHUFFLED": {
        "EPE_global":
            0.2991297245025635,

        "EPE_transport_region":
            1.0215282440185547,
    },
}


device = torch.device("cuda")

weights = (
    Raft_Large_Weights.C_T_SKHT_V2
)

model = raft_large(
    weights=weights,
    progress=False,
).to(device).eval()

transforms = weights.transforms()


reference = torch.from_numpy(
    np.load(
        FINAL / "flows.npy"
    )
).float()

if tuple(reference.shape) != (
    164, 2, 240, 416
):
    raise RuntimeError(
        f"reference flow shape="
        f"{tuple(reference.shape)}"
    )


art = torch.load(
    MASK_ART,
    map_location="cpu",
    weights_only=True,
)

latent_mask = (
    art["transport_mask"]
    .bool()
)

if tuple(latent_mask.shape) != (
    42, 1, 60, 104
):
    raise RuntimeError(
        "mask shape mismatch"
    )


# Causal destination-frame mapping:
#
# flow j corresponds to destination pixel frame j+1.
# Frames 1..4 use latent slot 1,
# frames 5..8 slot 2,
# ...
# frames 161..164 slot 41.

flow_masks = []

for destination in range(
    1,
    165,
):
    slot = (
        (destination + 3)
        // 4
    )

    if slot < 1:
        slot = 1

    if slot > 41:
        slot = 41

    flow_masks.append(
        latent_mask[slot]
    )

flow_masks = torch.stack(
    flow_masks,
    dim=0,
).float()

if tuple(flow_masks.shape) != (
    164, 1, 60, 104
):
    raise RuntimeError(
        f"flow mask shape="
        f"{tuple(flow_masks.shape)}"
    )

flow_masks = F.interpolate(
    flow_masks,
    size=(240, 416),
    mode="nearest",
).bool()


def read_video(path):
    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(
        str(path)
    )

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        # cv2 BGR -> RGB.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = cv2.resize(
            frame,
            (416, 240),
            interpolation=cv2.INTER_AREA,
        )

        frames.append(frame)

    cap.release()

    if len(frames) != 165:
        raise RuntimeError(
            f"{path}: got {len(frames)} frames"
        )

    return np.stack(
        frames,
        axis=0,
    )


@torch.inference_mode()
def predict_flow(frames):
    flows = []

    batch_size = 8

    for start in range(
        0,
        164,
        batch_size,
    ):
        end = min(
            164,
            start + batch_size,
        )

        im1 = torch.from_numpy(
            frames[start:end]
        ).permute(
            0, 3, 1, 2
        ).float().to(device)

        im2 = torch.from_numpy(
            frames[start+1:end+1]
        ).permute(
            0, 3, 1, 2
        ).float().to(device)

        im1 = im1 / 255.0
        im2 = im2 / 255.0

        im1, im2 = transforms(
            im1,
            im2,
        )

        pred = model(
            im1,
            im2,
        )[-1]

        flows.append(
            pred.cpu()
        )

        print(
            f"RAFT {start:03d}:{end:03d}",
            flush=True,
        )

    return torch.cat(
        flows,
        dim=0,
    )


def evaluate(pred):
    ref = reference

    diff = pred - ref

    epe = torch.linalg.norm(
        diff,
        dim=1,
    )

    pred_mag = torch.linalg.norm(
        pred,
        dim=1,
    )

    ref_mag = torch.linalg.norm(
        ref,
        dim=1,
    )

    mag_error = (
        pred_mag - ref_mag
    ).abs()

    dot = (
        pred[:, 0] * ref[:, 0]
        + pred[:, 1] * ref[:, 1]
    )

    denom = (
        pred_mag * ref_mag
    ).clamp_min(1e-8)

    cosine = (
        dot / denom
    ).clamp(
        -1.0,
        1.0,
    )

    angle = (
        torch.acos(cosine)
        * 180.0
        / math.pi
    )

    mask = flow_masks[:, 0]

    return {
        "EPE_global":
            float(
                epe.mean()
            ),

        "magnitude_error_global":
            float(
                mag_error.mean()
            ),

        "angle_error_deg_global":
            float(
                angle.mean()
            ),

        "pred_flow_magnitude_mean":
            float(
                pred_mag.mean()
            ),

        "reference_flow_magnitude_mean":
            float(
                ref_mag.mean()
            ),

        "EPE_transport_region":
            float(
                epe[mask].mean()
            ),

        "magnitude_error_transport_region":
            float(
                mag_error[mask].mean()
            ),

        "angle_error_deg_transport_region":
            float(
                angle[mask].mean()
            ),

        "transport_pixels":
            int(
                mask.sum()
            ),
    }


results = {}

for name, path in VIDEOS.items():

    print()
    print(
        "========================================"
    )
    print(
        "RAFT:",
        name,
        path,
    )
    print(
        "========================================"
    )

    frames = read_video(path)

    pred = predict_flow(frames)

    metrics = evaluate(pred)

    results[name] = metrics

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


# Frozen V1 reproducibility check.

print()
print(
    "===== FROZEN V1 RAFT REPRODUCTION ====="
)

reproduction = {}

EPE_TOL = 5e-4

for name in (
    "REALWONDER",
    "V1_CORRECT",
    "V1_SHUFFLED",
):
    dg = abs(
        results[name]["EPE_global"]
        - FROZEN[name]["EPE_global"]
    )

    dt = abs(
        results[name]["EPE_transport_region"]
        - FROZEN[name][
            "EPE_transport_region"
        ]
    )

    ok = (
        dg <= EPE_TOL
        and dt <= EPE_TOL
    )

    reproduction[name] = {
        "global_EPE_abs_diff":
            dg,

        "transport_EPE_abs_diff":
            dt,

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
        "Frozen SandHouse RAFT metrics "
        "not reproduced. "
        "Do not interpret V2."
    )


def improvement(new, old):
    return float(
        100.0
        * (old - new)
        / old
    )


comparisons = {
    "V2A_correct_vs_V1_correct_global_EPE_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["EPE_global"],
            results["V1_CORRECT"]["EPE_global"],
        ),

    "V2B_correct_vs_V1_correct_global_EPE_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["EPE_global"],
            results["V1_CORRECT"]["EPE_global"],
        ),

    "V2A_correct_vs_V1_correct_transport_EPE_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["EPE_transport_region"],
            results["V1_CORRECT"]["EPE_transport_region"],
        ),

    "V2B_correct_vs_V1_correct_transport_EPE_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["EPE_transport_region"],
            results["V1_CORRECT"]["EPE_transport_region"],
        ),

    "V2A_correct_vs_RW_transport_EPE_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["EPE_transport_region"],
            results["REALWONDER"]["EPE_transport_region"],
        ),

    "V2B_correct_vs_RW_transport_EPE_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["EPE_transport_region"],
            results["REALWONDER"]["EPE_transport_region"],
        ),

    "V2A_correct_vs_V2A_shuffled_transport_EPE_improvement_percent":
        improvement(
            results["V2A_CORRECT"]["EPE_transport_region"],
            results["V2A_SHUFFLED"]["EPE_transport_region"],
        ),

    "V2B_correct_vs_V2B_shuffled_transport_EPE_improvement_percent":
        improvement(
            results["V2B_CORRECT"]["EPE_transport_region"],
            results["V2B_SHUFFLED"]["EPE_transport_region"],
        ),
}


global_ranking = sorted(
    results,
    key=lambda n:
        results[n]["EPE_global"],
)

transport_ranking = sorted(
    results,
    key=lambda n:
        results[n]["EPE_transport_region"],
)


report = {
    "experiment":
        "SandHouse unified V1/V2 RAFT motion evaluation",

    "reference_flow":
        str(FINAL / "flows.npy"),

    "reference_shape":
        list(reference.shape),

    "raft_weights":
        "Raft_Large_Weights.C_T_SKHT_V2",

    "video_preprocessing":
        "480x832 RGB -> area resize 240x416 -> torchvision RAFT transforms",

    "transport_region":
        "frozen Quality transport mask causally mapped to destination frames 1..164",

    "results":
        results,

    "frozen_v1_reproduction":
        reproduction,

    "comparisons":
        comparisons,

    "global_EPE_ranking":
        global_ranking,

    "transport_EPE_ranking":
        transport_ranking,
}


report_path = (
    OUT
    / "raft_report.json"
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
    "GLOBAL EPE RANKING =",
    global_ranking,
)

print(
    "TRANSPORT EPE RANKING =",
    transport_ranking,
)

print(
    "saved =",
    report_path,
)

print(
    "UNIFIED_V2_RAFT_EVALUATION_OK"
)
