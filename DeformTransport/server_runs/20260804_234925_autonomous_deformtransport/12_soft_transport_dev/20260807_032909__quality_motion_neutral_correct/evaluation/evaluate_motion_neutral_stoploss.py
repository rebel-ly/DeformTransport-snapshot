from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    raft_large,
)


sim_data = Path(os.environ["SIM_DATA"])
motion_eval_dir = Path(os.environ["MOTION_EVAL_DIR"])
condition_eval_dir = Path(os.environ["COND_EVAL_DIR"])
quality_artifact = Path(os.environ["QUALITY_ARTIFACT"])
mn_video_path = Path(os.environ["MN_VIDEO"])
output_dir = Path(os.environ["MN_EVAL_DIR"])

batch_size = int(os.environ.get("RAFT_BATCH_SIZE", "4"))

output_dir.mkdir(parents=True, exist_ok=True)


def read_video(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = imageio.get_reader(path)

    try:
        frames = np.stack([frame for frame in reader])
    finally:
        reader.close()

    expected = (81, 480, 832, 3)

    if tuple(frames.shape) != expected:
        raise RuntimeError(
            f"{path}: expected {expected}, got {frames.shape}"
        )

    return frames


def extract_mask(state: dict) -> torch.Tensor:
    mask = torch.as_tensor(
        state["transport_mask"]
    ).detach().cpu()

    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]

    if tuple(mask.shape) != (21, 60, 104):
        raise RuntimeError(
            f"unexpected mask shape: {tuple(mask.shape)}"
        )

    return mask > 0


def expand_latent_mask(
    latent_mask: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    expanded = [latent_mask[0]]

    for index in range(1, 21):
        expanded.extend([latent_mask[index]] * 4)

    mask = torch.stack(expanded, dim=0)

    if mask.shape[0] != 81:
        raise RuntimeError(
            f"expanded frame count is {mask.shape[0]}"
        )

    return (
        F.interpolate(
            mask[:, None].float(),
            size=(height, width),
            mode="nearest",
        )[:, 0]
        > 0.5
    )


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return (
        F.max_pool2d(
            mask[:, None].float(),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[:, 0]
        > 0.5
    )


def masked_mean(
    values: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = values[mask]

    if selected.size == 0:
        raise RuntimeError("empty evaluation mask")

    return float(selected.mean())


def gradient_error(
    generated: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    generated = generated.astype(np.float32)
    target = target.astype(np.float32)

    generated_dx = generated[:, :, 1:] - generated[:, :, :-1]
    target_dx = target[:, :, 1:] - target[:, :, :-1]

    generated_dy = generated[:, 1:] - generated[:, :-1]
    target_dy = target[:, 1:] - target[:, :-1]

    dx_error = np.abs(
        generated_dx - target_dx
    ).mean(axis=-1)

    dy_error = np.abs(
        generated_dy - target_dy
    ).mean(axis=-1)

    dx_mask = mask[:, :, 1:] | mask[:, :, :-1]
    dy_mask = mask[:, 1:] | mask[:, :-1]

    numerator = (
        float(dx_error[dx_mask].sum())
        + float(dy_error[dy_mask].sum())
    )

    denominator = int(dx_mask.sum()) + int(dy_mask.sum())

    if denominator == 0:
        raise RuntimeError("empty gradient mask")

    return numerator / denominator


def evaluate_rgb(
    generated: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    generated_f = generated.astype(np.float32)
    target_f = target.astype(np.float32)

    difference = generated_f - target_f

    l1_map = np.abs(difference).mean(axis=-1)
    mse_map = np.square(difference).mean(axis=-1)

    l1 = masked_mean(l1_map, mask)
    mse = masked_mean(mse_map, mask)

    generated_delta = generated_f[1:] - generated_f[:-1]
    target_delta = target_f[1:] - target_f[:-1]

    temporal_map = np.abs(
        generated_delta - target_delta
    ).mean(axis=-1)

    temporal_mask = mask[1:] | mask[:-1]

    return {
        "l1": l1,
        "mse": mse,
        "psnr": float(
            10.0 * math.log10((255.0 ** 2) / mse)
        ),
        "temporal_change_error": masked_mean(
            temporal_map,
            temporal_mask,
        ),
        "gradient_error": gradient_error(
            generated,
            target,
            mask,
        ),
        "selected_frame_l1": {
            str(index): masked_mean(
                l1_map[index],
                mask[index],
            )
            for index in (0, 20, 40, 60, 80)
        },
    }


def infer_flow(
    frames: np.ndarray,
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    outputs = []

    start_time = time.time()

    for start in range(0, 80, batch_size):
        end = min(80, start + batch_size)

        image1 = torch.from_numpy(
            frames[start:end]
        ).permute(0, 3, 1, 2).float().div_(255.0)

        image2 = torch.from_numpy(
            frames[start + 1:end + 1]
        ).permute(0, 3, 1, 2).float().div_(255.0)

        image1 = F.interpolate(
            image1.to(device),
            size=(240, 416),
            mode="area",
        )

        image2 = F.interpolate(
            image2.to(device),
            size=(240, 416),
            mode="area",
        )

        image1, image2 = preprocess(image1, image2)

        with torch.inference_mode():
            prediction = model(image1, image2)[-1]

        outputs.append(
            prediction.detach().cpu().float().numpy()
        )

        print(
            f"motion_neutral_raft: {end}/80",
            flush=True,
        )

    torch.cuda.synchronize()

    result = np.concatenate(outputs, axis=0)

    if tuple(result.shape) != (80, 2, 240, 416):
        raise RuntimeError(
            f"unexpected flow shape: {result.shape}"
        )

    print(
        "motion_neutral_raft_seconds =",
        round(time.time() - start_time, 4),
    )

    return result.astype(np.float32)


def evaluate_flow(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    difference = predicted - target

    epe = np.sqrt(
        np.square(difference).sum(axis=1)
    )

    predicted_magnitude = np.sqrt(
        np.square(predicted).sum(axis=1)
    )

    target_magnitude = np.sqrt(
        np.square(target).sum(axis=1)
    )

    magnitude_error = np.abs(
        predicted_magnitude - target_magnitude
    )

    dot = (predicted * target).sum(axis=1)

    cosine = np.clip(
        dot
        / (
            predicted_magnitude
            * target_magnitude
            + 1e-6
        ),
        -1.0,
        1.0,
    )

    angle = np.degrees(np.arccos(cosine))

    direction_mask = (
        mask
        & (target_magnitude > 0.25)
        & (predicted_magnitude > 0.05)
    )

    return {
        "pixel_count": int(mask.sum()),
        "epe_mean": float(epe[mask].mean()),
        "epe_median": float(np.median(epe[mask])),
        "epe_p90": float(np.percentile(epe[mask], 90)),
        "flow_magnitude_mae": float(
            magnitude_error[mask].mean()
        ),
        "target_magnitude_mean": float(
            target_magnitude[mask].mean()
        ),
        "predicted_magnitude_mean": float(
            predicted_magnitude[mask].mean()
        ),
        "direction_cosine_mean": float(
            cosine[direction_mask].mean()
        ),
        "angle_error_degrees_mean": float(
            angle[direction_mask].mean()
        ),
    }


def improvement_percent(
    candidate: float,
    reference: float,
) -> float:
    return float(
        (reference - candidate)
        / reference
        * 100.0
    )


target_video = read_video(
    sim_data / "simulation.mp4"
)

motion_neutral_video = read_video(
    mn_video_path
)

artifact = torch.load(
    quality_artifact,
    map_location="cpu",
    weights_only=False,
)

latent_mask = extract_mask(artifact)

rgb_core_t = expand_latent_mask(
    latent_mask,
    480,
    832,
)

rgb_dilated_t = dilate(
    rgb_core_t,
    radius=8,
)

rgb_regions = {
    "transport_core": rgb_core_t.numpy(),
    "transport_dilated_8px": rgb_dilated_t.numpy(),
    "background_outside_dilated": (
        ~rgb_dilated_t.numpy()
    ),
}

motion_neutral_local = {
    region: evaluate_rgb(
        motion_neutral_video,
        target_video,
        mask,
    )
    for region, mask in rgb_regions.items()
}

existing_local_path = (
    condition_eval_dir
    / "condition_space_local_metrics.json"
)

existing_local = json.loads(
    existing_local_path.read_text(
        encoding="utf-8"
    )
)

local_comparisons = {}

for region in (
    "transport_core",
    "transport_dilated_8px",
):
    candidate = motion_neutral_local[region]

    references = existing_local[
        "metrics_against_simulation_target"
    ][region]

    local_comparisons[region] = {}

    for reference_name in (
        "realwonder_baseline",
        "quality_inter_step_correct",
    ):
        reference = references[reference_name]

        local_comparisons[region][
            f"motion_neutral_vs_{reference_name}"
        ] = {
            key: improvement_percent(
                candidate[key],
                reference[key],
            )
            for key in (
                "l1",
                "mse",
                "temporal_change_error",
                "gradient_error",
            )
        }


cached_flow_paths = {
    "simulation_target": (
        motion_eval_dir
        / "simulation_target_raft_flow.npy"
    ),
    "realwonder_baseline": (
        motion_eval_dir
        / "realwonder_baseline_raft_flow.npy"
    ),
    "quality_inter_step_correct": (
        motion_eval_dir
        / "quality_correct_raft_flow.npy"
    ),
}

cached_flows = {}

for name, path in cached_flow_paths.items():
    flow = np.load(
        path,
        allow_pickle=False,
    ).astype(np.float32)

    if tuple(flow.shape) != (80, 2, 240, 416):
        raise RuntimeError(
            f"{path}: unexpected shape {flow.shape}"
        )

    cached_flows[name] = flow


device = torch.device("cuda:0")

weights = Raft_Large_Weights.C_T_SKHT_V2
preprocess = weights.transforms()

print("Loading cached RAFT-Large...", flush=True)

model = raft_large(
    weights=weights,
    progress=False,
).to(device).eval()

torch.cuda.reset_peak_memory_stats(device)

motion_neutral_flow = infer_flow(
    motion_neutral_video,
    model,
    preprocess,
    device,
)

flow_output_path = (
    output_dir
    / "motion_neutral_correct_raft_flow.npy"
)

np.save(
    flow_output_path,
    motion_neutral_flow.astype(np.float16),
    allow_pickle=False,
)

flow_rgb_mask = expand_latent_mask(
    latent_mask,
    240,
    416,
)

flow_core_transition = (
    flow_rgb_mask[:-1]
    | flow_rgb_mask[1:]
)

flow_dilated_rgb = dilate(
    flow_rgb_mask,
    radius=4,
)

flow_dilated_transition = (
    flow_dilated_rgb[:-1]
    | flow_dilated_rgb[1:]
)

target_flow = cached_flows["simulation_target"]

target_magnitude = np.sqrt(
    np.square(target_flow).sum(axis=1)
)

flow_regions = {
    "transport_core_moving_gt_0p25": (
        flow_core_transition.numpy()
        & (target_magnitude > 0.25)
    ),
    "transport_dilated_moving_gt_0p25": (
        flow_dilated_transition.numpy()
        & (target_magnitude > 0.25)
    ),
}

motion_metrics = {}

for region, mask in flow_regions.items():
    motion_metrics[region] = {
        "realwonder_baseline": evaluate_flow(
            cached_flows["realwonder_baseline"],
            target_flow,
            mask,
        ),
        "quality_inter_step_correct": evaluate_flow(
            cached_flows["quality_inter_step_correct"],
            target_flow,
            mask,
        ),
        "motion_neutral_correct": evaluate_flow(
            motion_neutral_flow,
            target_flow,
            mask,
        ),
    }

motion_comparisons = {}

for region, values in motion_metrics.items():
    candidate = values["motion_neutral_correct"]

    motion_comparisons[region] = {}

    for reference_name in (
        "realwonder_baseline",
        "quality_inter_step_correct",
    ):
        reference = values[reference_name]

        motion_comparisons[region][
            f"motion_neutral_vs_{reference_name}"
        ] = {
            "epe_improvement_percent": (
                improvement_percent(
                    candidate["epe_mean"],
                    reference["epe_mean"],
                )
            ),
            "magnitude_mae_improvement_percent": (
                improvement_percent(
                    candidate["flow_magnitude_mae"],
                    reference["flow_magnitude_mae"],
                )
            ),
            "angle_improvement_percent": (
                improvement_percent(
                    candidate[
                        "angle_error_degrees_mean"
                    ],
                    reference[
                        "angle_error_degrees_mean"
                    ],
                )
            ),
        }


core_local = motion_neutral_local["transport_core"]
core_references = existing_local[
    "metrics_against_simulation_target"
]["transport_core"]

core_motion = motion_metrics[
    "transport_core_moving_gt_0p25"
]

decision_flags = {
    "local_l1_better_than_baseline": (
        core_local["l1"]
        <
        core_references[
            "realwonder_baseline"
        ]["l1"]
    ),
    "local_l1_better_than_quality_inter_step": (
        core_local["l1"]
        <
        core_references[
            "quality_inter_step_correct"
        ]["l1"]
    ),
    "motion_epe_better_than_baseline": (
        core_motion[
            "motion_neutral_correct"
        ]["epe_mean"]
        <
        core_motion[
            "realwonder_baseline"
        ]["epe_mean"]
    ),
    "motion_epe_better_than_quality_inter_step": (
        core_motion[
            "motion_neutral_correct"
        ]["epe_mean"]
        <
        core_motion[
            "quality_inter_step_correct"
        ]["epe_mean"]
    ),
}

decision_flags[
    "advance_to_motion_neutral_shuffled"
] = (
    decision_flags[
        "local_l1_better_than_baseline"
    ]
    and decision_flags[
        "motion_epe_better_than_baseline"
    ]
)

checks = {
    "target_video_shape_valid": (
        tuple(target_video.shape)
        == (81, 480, 832, 3)
    ),
    "motion_neutral_video_shape_valid": (
        tuple(motion_neutral_video.shape)
        == (81, 480, 832, 3)
    ),
    "motion_neutral_flow_shape_valid": (
        tuple(motion_neutral_flow.shape)
        == (80, 2, 240, 416)
    ),
    "all_values_finite": bool(
        np.isfinite(motion_neutral_video).all()
        and np.isfinite(motion_neutral_flow).all()
    ),
    "core_masks_nonempty": bool(
        rgb_core_t.any()
        and flow_regions[
            "transport_core_moving_gt_0p25"
        ].any()
    ),
}

report = {
    "stage": "motion_neutral_stoploss_evaluation",
    "motion_neutral_video": str(mn_video_path),
    "local_metrics_against_simulation_target": (
        motion_neutral_local
    ),
    "local_comparisons_improvement_percent": (
        local_comparisons
    ),
    "motion_metrics_against_simulation_flow": (
        motion_metrics
    ),
    "motion_comparisons_improvement_percent": (
        motion_comparisons
    ),
    "decision_flags": decision_flags,
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "gpu_peak_allocated_mib": float(
        torch.cuda.max_memory_allocated(device)
        / 1024
        / 1024
    ),
    "motion_neutral_flow": str(flow_output_path),
}

report_path = (
    output_dir
    / "motion_neutral_stoploss_metrics.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print()
print("========== MOTION-NEUTRAL RESULTS ==========")

for region in (
    "transport_core",
    "transport_dilated_8px",
):
    values = motion_neutral_local[region]

    print()
    print("RGB REGION:", region)
    print(
        "motion_neutral_correct",
        "L1=", round(values["l1"], 6),
        "PSNR=", round(values["psnr"], 6),
        "Temporal=",
        round(values["temporal_change_error"], 6),
        "Gradient=",
        round(values["gradient_error"], 6),
    )
    print(
        "comparisons =",
        local_comparisons[region],
    )

for region in (
    "transport_core_moving_gt_0p25",
    "transport_dilated_moving_gt_0p25",
):
    print()
    print("FLOW REGION:", region)

    for method, values in motion_metrics[region].items():
        print(
            method,
            "EPE=", round(values["epe_mean"], 6),
            "MagnitudeMAE=",
            round(values["flow_magnitude_mae"], 6),
            "Angle=",
            round(
                values["angle_error_degrees_mean"],
                6,
            ),
        )

    print(
        "comparisons =",
        motion_comparisons[region],
    )

print()
print("decision_flags =", decision_flags)
print("all_checks_pass =", report["all_checks_pass"])

if not report["all_checks_pass"]:
    raise SystemExit(1)
