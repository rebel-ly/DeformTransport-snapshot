from pathlib import Path
import json
import math
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    raft_large,
)


base_path = Path(os.environ["BASE"])
correct_path = Path(os.environ["CORRECT"])
shuffled_path = Path(os.environ["SHUFFLED"])
artifact_path = Path(os.environ["ART"])
final = Path(os.environ["FINAL"])
out = Path(os.environ["OUT"])

device = torch.device("cuda:0")

weights = (
    Raft_Large_Weights
    .C_T_SKHT_V2
)

model = raft_large(
    weights=weights,
    progress=False,
).to(device).eval()

transforms = weights.transforms()


def read_video(path):
    reader = imageio.get_reader(path)

    frames = []

    try:
        for x in reader:
            frames.append(
                np.asarray(
                    x,
                    dtype=np.uint8,
                )
            )
    finally:
        reader.close()

    value = np.stack(frames)

    if value.shape != (
        165, 480, 832, 3
    ):
        raise RuntimeError(
            f"{path}: bad video shape {value.shape}"
        )

    return value


@torch.inference_mode()
def video_flow(path, batch_size=8):
    video = read_video(path)

    # Exact historical preprocessing:
    # 480x832 -> area resize 240x416.
    tensor = torch.from_numpy(
        video.copy()
    ).permute(
        0, 3, 1, 2
    ).float()

    tensor = F.interpolate(
        tensor,
        size=(240, 416),
        mode="area",
    )

    tensor = (
        tensor
        .clamp(0, 255)
        / 255.0
    )

    outputs = []

    for start in range(
        0,
        tensor.shape[0] - 1,
        batch_size,
    ):
        end = min(
            start + batch_size,
            tensor.shape[0] - 1,
        )

        first = tensor[
            start:end
        ].to(device)

        second = tensor[
            start + 1:end + 1
        ].to(device)

        first, second = transforms(
            first,
            second,
        )

        prediction = model(
            first,
            second,
        )[-1]

        outputs.append(
            prediction
            .float()
            .cpu()
        )

        del first, second, prediction

    return torch.cat(
        outputs,
        dim=0,
    )


reference_np = np.load(
    final / "flows.npy",
    allow_pickle=False,
)

if reference_np.shape != (
    164, 2, 240, 416
):
    raise RuntimeError(
        f"reference flow shape = {reference_np.shape}"
    )

reference = torch.from_numpy(
    reference_np.astype(
        np.float32
    )
)


state = torch.load(
    artifact_path,
    map_location="cpu",
    weights_only=True,
)

mask = state[
    "transport_mask"
].bool()

if tuple(mask.shape) != (
    42, 1, 60, 104
):
    raise RuntimeError(
        f"mask shape {tuple(mask.shape)}"
    )

indices = (
    state[
        "latent_frame_indices"
    ]
    .long()
    .tolist()
)

if indices != list(
    range(0, 165, 4)
):
    raise RuntimeError(
        "unexpected latent frame indices"
    )


# --------------------------------------------------
# Historical Tree/Santa causal mask semantics.
#
# flow j corresponds:
#   frame j -> frame j+1
#
# destination frames:
#   1..4   -> latent slot 1
#   5..8   -> latent slot 2
#   ...
#   161..164 -> latent slot 41
# --------------------------------------------------

destination_masks = torch.cat(
    [
        mask[i : i + 1].repeat(
            4, 1, 1, 1
        )
        for i in range(
            1,
            mask.shape[0],
        )
    ],
    dim=0,
)

if tuple(
    destination_masks.shape
) != (
    164, 1, 60, 104
):
    raise RuntimeError(
        f"causal mask shape = "
        f"{tuple(destination_masks.shape)}"
    )

transport_mask = F.interpolate(
    destination_masks.float(),
    size=(240, 416),
    mode="nearest",
).bool()[:, 0]


def evaluate(pred):
    if pred.shape != reference.shape:
        raise RuntimeError(
            f"prediction flow shape {pred.shape}"
        )

    delta = pred - reference

    epe = torch.linalg.vector_norm(
        delta,
        dim=1,
    )

    pred_mag = torch.linalg.vector_norm(
        pred,
        dim=1,
    )

    ref_mag = torch.linalg.vector_norm(
        reference,
        dim=1,
    )

    mag_error = torch.abs(
        pred_mag - ref_mag
    )

    dot = (
        pred * reference
    ).sum(dim=1)

    denom = (
        pred_mag
        * ref_mag
    )

    # Same robust convention for near-zero flow:
    # angle only where both vectors are nonzero.
    angle_valid = denom > 1e-8

    cosine = torch.zeros_like(dot)

    cosine[angle_valid] = (
        dot[angle_valid]
        / denom[angle_valid]
    ).clamp(-1.0, 1.0)

    angle = torch.zeros_like(dot)

    angle[angle_valid] = (
        torch.acos(
            cosine[angle_valid]
        )
        * 180.0
        / math.pi
    )

    region = transport_mask

    region_angle_valid = (
        region & angle_valid
    )

    return {
        "EPE_global":
            float(epe.mean()),

        "magnitude_error_global":
            float(
                mag_error.mean()
            ),

        "angle_error_deg_global":
            float(
                angle[
                    angle_valid
                ].mean()
            )
            if bool(angle_valid.any())
            else 0.0,

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
                epe[region].mean()
            ),

        "magnitude_error_transport_region":
            float(
                mag_error[
                    region
                ].mean()
            ),

        "angle_error_deg_transport_region":
            float(
                angle[
                    region_angle_valid
                ].mean()
            )
            if bool(
                region_angle_valid.any()
            )
            else 0.0,

        "transport_pixels":
            int(region.sum()),
    }


videos = {
    "BASELINE": base_path,
    "CORRECT": correct_path,
    "SHUFFLED_MATCH": shuffled_path,
}

metrics = {}

for name, path in videos.items():
    print(
        "RAFT:",
        name,
        path,
        flush=True,
    )

    flow = video_flow(path)

    metrics[name] = evaluate(
        flow
    )

    print(
        json.dumps(
            metrics[name],
            indent=2,
        ),
        flush=True,
    )

    del flow

    torch.cuda.empty_cache()


def improve(baseline, candidate):
    return (
        baseline - candidate
    ) / baseline * 100.0


report = {
    "experiment":
        "SandHouse three-way RAFT motion diagnostic",

    "reference_flow":
        str(final / "flows.npy"),

    "reference_shape":
        list(reference.shape),

    "video_preprocessing":
        "480x832 RGB -> area resize 240x416 -> torchvision RAFT transforms",

    "raft_weights":
        "Raft_Large_Weights.C_T_SKHT_V2",

    "transport_region":
        "Quality transport mask mapped causally to destination frames 1..164",

    "metrics":
        metrics,

    "comparisons": {
        "correct_vs_baseline_global_EPE_improvement_percent":
            improve(
                metrics["BASELINE"][
                    "EPE_global"
                ],
                metrics["CORRECT"][
                    "EPE_global"
                ],
            ),

        "correct_vs_matched_global_EPE_improvement_percent":
            improve(
                metrics[
                    "SHUFFLED_MATCH"
                ][
                    "EPE_global"
                ],
                metrics["CORRECT"][
                    "EPE_global"
                ],
            ),

        "correct_vs_baseline_transport_EPE_improvement_percent":
            improve(
                metrics["BASELINE"][
                    "EPE_transport_region"
                ],
                metrics["CORRECT"][
                    "EPE_transport_region"
                ],
            ),

        "correct_vs_matched_transport_EPE_improvement_percent":
            improve(
                metrics[
                    "SHUFFLED_MATCH"
                ][
                    "EPE_transport_region"
                ],
                metrics["CORRECT"][
                    "EPE_transport_region"
                ],
            ),
    },

    "global_EPE_ranking":
        sorted(
            metrics,
            key=lambda x:
                metrics[x][
                    "EPE_global"
                ],
        ),

    "transport_EPE_ranking":
        sorted(
            metrics,
            key=lambda x:
                metrics[x][
                    "EPE_transport_region"
                ],
        ),
}

report_path = out / "report.json"

report_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "\n===== FINAL REPORT ====="
)

print(
    json.dumps(
        report,
        indent=2,
    )
)

print(
    "\nSANDHOUSE_RAFT_MOTION_OK"
)
