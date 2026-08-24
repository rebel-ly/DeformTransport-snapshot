from __future__ import annotations

import json
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


output_dir = Path(os.environ["COND_EVAL_DIR"])
cond_root = Path(os.environ["COND_ROOT"])
motion_eval_dir = Path(os.environ["MOTION_EVAL_DIR"])
quality_artifact = Path(os.environ["QUALITY_ARTIFACT"])

batch_size = int(
    os.environ.get("RAFT_BATCH_SIZE", "4")
)

flow_height = 240
flow_width = 416

video_paths = {
    "condition_correct": (
        cond_root
        / "quality_condition_correct"
        / "aligned_santa_quality_condition_correct_seed0.mp4"
    ),
    "condition_shuffled": (
        cond_root
        / "quality_condition_shuffled"
        / "aligned_santa_quality_condition_shuffled_seed0.mp4"
    ),
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


def infer_flow(
    name: str,
    frames: np.ndarray,
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    pair_count = 80
    start_time = time.time()

    for start in range(0, pair_count, batch_size):
        end = min(pair_count, start + batch_size)

        image1 = torch.from_numpy(
            frames[start:end]
        ).permute(
            0, 3, 1, 2
        ).float().div_(255.0)

        image2 = torch.from_numpy(
            frames[start + 1:end + 1]
        ).permute(
            0, 3, 1, 2
        ).float().div_(255.0)

        image1 = F.interpolate(
            image1.to(device),
            size=(flow_height, flow_width),
            mode="area",
        )

        image2 = F.interpolate(
            image2.to(device),
            size=(flow_height, flow_width),
            mode="area",
        )

        image1, image2 = preprocess(
            image1,
            image2,
        )

        with torch.inference_mode():
            flow = model(image1, image2)[-1]

        outputs.append(
            flow.detach().cpu().float().numpy()
        )

        print(
            f"{name}: {end}/{pair_count}",
            flush=True,
        )

    torch.cuda.synchronize()

    result = np.concatenate(
        outputs,
        axis=0,
    ).astype(np.float32)

    expected = (
        80,
        2,
        flow_height,
        flow_width,
    )

    if tuple(result.shape) != expected:
        raise RuntimeError(
            f"{name}: expected {expected}, got {result.shape}"
        )

    output_path = (
        output_dir
        / f"{name}_raft_flow.npy"
    )

    np.save(
        output_path,
        result.astype(np.float16),
        allow_pickle=False,
    )

    print(
        f"{name}: completed in "
        f"{time.time() - start_time:.2f}s",
        flush=True,
    )

    return result


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
            f"unexpected mask: {tuple(mask.shape)}"
        )

    return mask > 0


def expand_mask(latent_mask: torch.Tensor) -> torch.Tensor:
    expanded = [latent_mask[0]]

    for index in range(1, 21):
        expanded.extend([latent_mask[index]] * 4)

    mask = torch.stack(expanded, dim=0)

    mask = F.interpolate(
        mask[:, None].float(),
        size=(flow_height, flow_width),
        mode="nearest",
    )[:, 0]

    return mask > 0.5


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
        predicted_magnitude
        - target_magnitude
    )

    dot = (predicted * target).sum(axis=1)

    cosine = dot / (
        predicted_magnitude
        * target_magnitude
        + 1e-6
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    direction_mask = (
        mask
        & (target_magnitude > 0.25)
        & (predicted_magnitude > 0.05)
    )

    return {
        "pixel_count": int(mask.sum()),
        "epe_mean": float(epe[mask].mean()),
        "epe_median": float(
            np.median(epe[mask])
        ),
        "epe_p90": float(
            np.percentile(epe[mask], 90)
        ),
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


device = torch.device("cuda:0")

weights = Raft_Large_Weights.C_T_SKHT_V2
preprocess = weights.transforms()

print("Loading cached RAFT-Large...", flush=True)

model = raft_large(
    weights=weights,
    progress=False,
).to(device).eval()

torch.cuda.reset_peak_memory_stats(device)

cached_flows = {}

for name, path in cached_flow_paths.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    flow = np.load(
        path,
        allow_pickle=False,
    ).astype(np.float32)

    if tuple(flow.shape) != (
        80,
        2,
        flow_height,
        flow_width,
    ):
        raise RuntimeError(
            f"{path}: bad shape {flow.shape}"
        )

    cached_flows[name] = flow

new_flows = {}

for name, path in video_paths.items():
    frames = read_video(path)

    new_flows[name] = infer_flow(
        name=name,
        frames=frames,
        model=model,
        preprocess=preprocess,
        device=device,
    )

target_flow = cached_flows["simulation_target"]

artifact = torch.load(
    quality_artifact,
    map_location="cpu",
    weights_only=False,
)

rgb_mask = expand_mask(
    extract_mask(artifact)
)

core_transition = (
    rgb_mask[:-1] | rgb_mask[1:]
)

dilated_rgb = dilate(
    rgb_mask,
    radius=4,
)

dilated_transition = (
    dilated_rgb[:-1] | dilated_rgb[1:]
)

target_magnitude = np.sqrt(
    np.square(target_flow).sum(axis=1)
)

regions = {
    "transport_core_all": (
        core_transition.numpy()
    ),
    "transport_core_moving_gt_0p25": (
        core_transition.numpy()
        & (target_magnitude > 0.25)
    ),
    "transport_dilated_all": (
        dilated_transition.numpy()
    ),
    "transport_dilated_moving_gt_0p25": (
        dilated_transition.numpy()
        & (target_magnitude > 0.25)
    ),
}

all_flows = {
    **cached_flows,
    **new_flows,
}

metrics = {}

for region_name, region_mask in regions.items():
    metrics[region_name] = {
        method: evaluate_flow(
            predicted=flow,
            target=target_flow,
            mask=region_mask,
        )
        for method, flow in all_flows.items()
        if method != "simulation_target"
    }

comparisons = {}

for region_name in (
    "transport_core_moving_gt_0p25",
    "transport_dilated_moving_gt_0p25",
):
    values = metrics[region_name]

    correct = values["condition_correct"]
    shuffled = values["condition_shuffled"]
    baseline = values["realwonder_baseline"]
    inter_step = values[
        "quality_inter_step_correct"
    ]

    comparisons[region_name] = {
        "condition_correct_vs_shuffled": {
            "epe_improvement_percent": (
                improvement_percent(
                    correct["epe_mean"],
                    shuffled["epe_mean"],
                )
            ),
            "magnitude_mae_improvement_percent": (
                improvement_percent(
                    correct["flow_magnitude_mae"],
                    shuffled["flow_magnitude_mae"],
                )
            ),
        },
        "condition_correct_vs_baseline": {
            "epe_improvement_percent": (
                improvement_percent(
                    correct["epe_mean"],
                    baseline["epe_mean"],
                )
            ),
            "magnitude_mae_improvement_percent": (
                improvement_percent(
                    correct["flow_magnitude_mae"],
                    baseline["flow_magnitude_mae"],
                )
            ),
        },
        "condition_correct_vs_inter_step": {
            "epe_improvement_percent": (
                improvement_percent(
                    correct["epe_mean"],
                    inter_step["epe_mean"],
                )
            ),
            "magnitude_mae_improvement_percent": (
                improvement_percent(
                    correct["flow_magnitude_mae"],
                    inter_step["flow_magnitude_mae"],
                )
            ),
        },
    }

checks = {
    "cached_flows_valid": all(
        tuple(flow.shape)
        == (80, 2, 240, 416)
        for flow in cached_flows.values()
    ),
    "new_flows_valid": all(
        tuple(flow.shape)
        == (80, 2, 240, 416)
        for flow in new_flows.values()
    ),
    "all_flows_finite": all(
        np.isfinite(flow).all()
        for flow in all_flows.values()
    ),
    "moving_core_nonempty": bool(
        regions[
            "transport_core_moving_gt_0p25"
        ].any()
    ),
}

decision_flags = {
    "condition_correct_epe_better_than_shuffled": (
        metrics[
            "transport_core_moving_gt_0p25"
        ]["condition_correct"]["epe_mean"]
        <
        metrics[
            "transport_core_moving_gt_0p25"
        ]["condition_shuffled"]["epe_mean"]
    ),
    "condition_correct_epe_better_than_baseline": (
        metrics[
            "transport_core_moving_gt_0p25"
        ]["condition_correct"]["epe_mean"]
        <
        metrics[
            "transport_core_moving_gt_0p25"
        ]["realwonder_baseline"]["epe_mean"]
    ),
    "condition_correct_epe_better_than_inter_step": (
        metrics[
            "transport_core_moving_gt_0p25"
        ]["condition_correct"]["epe_mean"]
        <
        metrics[
            "transport_core_moving_gt_0p25"
        ]["quality_inter_step_correct"]["epe_mean"]
    ),
}

report = {
    "stage": "condition_space_incremental_raft",
    "evaluator": {
        "model": "torchvision RAFT-Large",
        "weights": "C_T_SKHT_V2",
        "resolution": [240, 416],
        "batch_size": batch_size,
        "gpu_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device)
            / 1024
            / 1024
        ),
    },
    "metrics_against_simulation_flow": metrics,
    "comparisons": comparisons,
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "decision_flags": decision_flags,
}

report_path = (
    output_dir
    / "condition_space_raft_metrics.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print()
print("========== RAFT RESULTS ==========")

for region_name in (
    "transport_core_moving_gt_0p25",
    "transport_dilated_moving_gt_0p25",
):
    print()
    print("REGION:", region_name)

    for method in (
        "realwonder_baseline",
        "quality_inter_step_correct",
        "condition_correct",
        "condition_shuffled",
    ):
        values = metrics[region_name][method]

        print(
            method,
            "EPE=",
            round(values["epe_mean"], 6),
            "MagnitudeMAE=",
            round(
                values["flow_magnitude_mae"],
                6,
            ),
            "Angle=",
            round(
                values[
                    "angle_error_degrees_mean"
                ],
                6,
            ),
            "Cosine=",
            round(
                values[
                    "direction_cosine_mean"
                ],
                6,
            ),
        )

    print(
        "comparisons =",
        comparisons[region_name],
    )

print()
print("decision_flags =", decision_flags)
print("all_checks_pass =", report["all_checks_pass"])

if not report["all_checks_pass"]:
    raise SystemExit(1)
