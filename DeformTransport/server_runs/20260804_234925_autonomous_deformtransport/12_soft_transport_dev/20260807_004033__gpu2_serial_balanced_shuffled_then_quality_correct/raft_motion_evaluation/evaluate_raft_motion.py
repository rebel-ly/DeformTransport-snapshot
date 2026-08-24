from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    raft_large,
)


OUTPUT_DIR = Path(
    os.environ["MOTION_EVAL_DIR"]
)

ALIGNED_FINAL_SIM = Path(
    os.environ["ALIGNED_FINAL_SIM"]
)

FULL_GEN_ROOT = Path(
    os.environ["FULL_GEN_ROOT"]
)

SECOND_WAVE_ROOT = Path(
    os.environ["SECOND_WAVE_ROOT"]
)

BALANCED_ARTIFACT = Path(
    os.environ["BALANCED_ARTIFACT"]
)

BATCH_SIZE = int(
    os.environ.get(
        "RAFT_BATCH_SIZE",
        "4",
    )
)

FLOW_HEIGHT = 240
FLOW_WIDTH = 416

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VIDEO_PATHS = {
    "realwonder_baseline": (
        FULL_GEN_ROOT
        / "baseline"
        / "aligned_santa_baseline_seed0.mp4"
    ),
    "balanced_correct": (
        FULL_GEN_ROOT
        / "balanced_correct_retry_gpu2_20260807_001201"
        / "aligned_santa_balanced_ramp4_correct_seed0.mp4"
    ),
    "balanced_shuffled": (
        SECOND_WAVE_ROOT
        / "balanced_shuffled"
        / "aligned_santa_balanced_ramp4_shuffled_seed0.mp4"
    ),
    "quality_correct": (
        SECOND_WAVE_ROOT
        / "quality_correct"
        / "aligned_santa_quality_ramp4_correct_seed0.mp4"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_target_frames(
    frame_dir: Path,
) -> np.ndarray:
    paths = sorted(
        frame_dir.glob(
            "frame_*.png"
        )
    )

    if len(paths) != 81:
        raise RuntimeError(
            "Expected 81 target PNG frames, "
            f"found {len(paths)}"
        )

    frames = np.stack(
        [
            np.asarray(
                Image.open(path).convert(
                    "RGB"
                )
            )
            for path in paths
        ]
    )

    return frames


def read_video(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    reader = imageio.get_reader(path)

    try:
        frames = np.stack(
            [frame for frame in reader]
        )
    finally:
        reader.close()

    expected = (
        81,
        480,
        832,
        3,
    )

    if tuple(frames.shape) != expected:
        raise RuntimeError(
            f"{path}: expected {expected}, "
            f"got {frames.shape}"
        )

    return frames


def extract_latent_mask(
    state: dict,
) -> torch.Tensor:
    if "transport_mask" not in state:
        raise KeyError(
            "artifact missing transport_mask"
        )

    mask = torch.as_tensor(
        state["transport_mask"]
    ).detach().cpu()

    while (
        mask.ndim > 3
        and mask.shape[0] == 1
    ):
        mask = mask.squeeze(0)

    if (
        mask.ndim == 4
        and mask.shape[1] == 1
    ):
        mask = mask[:, 0]

    if tuple(mask.shape) != (
        21,
        60,
        104,
    ):
        raise RuntimeError(
            "Unexpected transport mask shape: "
            f"{tuple(mask.shape)}"
        )

    return mask > 0


def expand_mask_to_rgb_frames(
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    expanded = [
        latent_mask[0]
    ]

    for index in range(1, 21):
        expanded.extend(
            [latent_mask[index]] * 4
        )

    result = torch.stack(
        expanded,
        dim=0,
    )

    if result.shape[0] != 81:
        raise RuntimeError(
            "Expanded mask does not have "
            "81 frames"
        )

    result = F.interpolate(
        result[:, None].float(),
        size=(
            FLOW_HEIGHT,
            FLOW_WIDTH,
        ),
        mode="nearest",
    )[:, 0]

    return result > 0.5


def dilate(
    mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    return (
        F.max_pool2d(
            mask[:, None].float(),
            kernel_size=(
                radius * 2 + 1
            ),
            stride=1,
            padding=radius,
        )[:, 0]
        > 0.5
    )


def infer_flows(
    name: str,
    frames: np.ndarray,
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    flows = []

    pair_count = (
        frames.shape[0] - 1
    )

    start_time = time.time()

    for start in range(
        0,
        pair_count,
        BATCH_SIZE,
    ):
        end = min(
            pair_count,
            start + BATCH_SIZE,
        )

        image1 = torch.from_numpy(
            frames[start:end]
        ).permute(
            0,
            3,
            1,
            2,
        ).float().div_(255.0)

        image2 = torch.from_numpy(
            frames[start + 1:end + 1]
        ).permute(
            0,
            3,
            1,
            2,
        ).float().div_(255.0)

        image1 = image1.to(
            device,
            non_blocking=True,
        )

        image2 = image2.to(
            device,
            non_blocking=True,
        )

        image1 = F.interpolate(
            image1,
            size=(
                FLOW_HEIGHT,
                FLOW_WIDTH,
            ),
            mode="area",
        )

        image2 = F.interpolate(
            image2,
            size=(
                FLOW_HEIGHT,
                FLOW_WIDTH,
            ),
            mode="area",
        )

        image1, image2 = preprocess(
            image1,
            image2,
        )

        with torch.inference_mode():
            prediction = model(
                image1,
                image2,
            )[-1]

        flows.append(
            prediction.detach()
            .cpu()
            .float()
            .numpy()
        )

        completed = end

        print(
            f"{name}: "
            f"{completed}/{pair_count} pairs",
            flush=True,
        )

    torch.cuda.synchronize()

    result = np.concatenate(
        flows,
        axis=0,
    ).astype(np.float32)

    elapsed = (
        time.time() - start_time
    )

    if tuple(result.shape) != (
        80,
        2,
        FLOW_HEIGHT,
        FLOW_WIDTH,
    ):
        raise RuntimeError(
            f"{name}: unexpected flow "
            f"shape {result.shape}"
        )

    output_path = (
        OUTPUT_DIR
        / f"{name}_raft_flow.npy"
    )

    np.save(
        output_path,
        result.astype(np.float16),
        allow_pickle=False,
    )

    print(
        f"{name}: done in "
        f"{elapsed:.2f}s, saved "
        f"{output_path}",
        flush=True,
    )

    return result


def evaluate_flow(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    difference = (
        predicted - target
    )

    endpoint_error = np.sqrt(
        np.square(difference).sum(
            axis=1
        )
    )

    predicted_magnitude = np.sqrt(
        np.square(predicted).sum(
            axis=1
        )
    )

    target_magnitude = np.sqrt(
        np.square(target).sum(
            axis=1
        )
    )

    magnitude_error = np.abs(
        predicted_magnitude
        - target_magnitude
    )

    selected_epe = (
        endpoint_error[mask]
    )

    selected_magnitude_error = (
        magnitude_error[mask]
    )

    selected_target_magnitude = (
        target_magnitude[mask]
    )

    selected_predicted_magnitude = (
        predicted_magnitude[mask]
    )

    if selected_epe.size == 0:
        raise RuntimeError(
            "Evaluation mask is empty"
        )

    dot = (
        predicted * target
    ).sum(axis=1)

    cosine = dot / (
        predicted_magnitude
        * target_magnitude
        + 1e-6
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    angle = np.degrees(
        np.arccos(cosine)
    )

    direction_mask = (
        mask
        & (target_magnitude > 0.25)
        & (predicted_magnitude > 0.05)
    )

    if direction_mask.any():
        angle_mean = float(
            angle[direction_mask].mean()
        )

        cosine_mean = float(
            cosine[
                direction_mask
            ].mean()
        )
    else:
        angle_mean = float("nan")
        cosine_mean = float("nan")

    selected_transitions = {}

    for transition in (
        0,
        20,
        40,
        60,
        79,
    ):
        transition_mask = mask[
            transition
        ]

        if transition_mask.any():
            selected_transitions[
                str(transition)
            ] = float(
                endpoint_error[
                    transition
                ][transition_mask].mean()
            )

    return {
        "pixel_count": int(
            selected_epe.size
        ),
        "epe_mean": float(
            selected_epe.mean()
        ),
        "epe_median": float(
            np.median(
                selected_epe
            )
        ),
        "epe_p90": float(
            np.percentile(
                selected_epe,
                90,
            )
        ),
        "flow_magnitude_mae": float(
            selected_magnitude_error.mean()
        ),
        "target_magnitude_mean": float(
            selected_target_magnitude.mean()
        ),
        "predicted_magnitude_mean": float(
            selected_predicted_magnitude.mean()
        ),
        "direction_cosine_mean": (
            cosine_mean
        ),
        "angle_error_degrees_mean": (
            angle_mean
        ),
        "selected_transition_epe": (
            selected_transitions
        ),
    }


def improvement_percent(
    candidate: float,
    reference: float,
) -> float:
    if reference == 0:
        return float("nan")

    return float(
        (
            reference
            - candidate
        )
        / reference
        * 100.0
    )


torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda:0"
)

weights = (
    Raft_Large_Weights.C_T_SKHT_V2
)

preprocess = weights.transforms()

print(
    "Loading RAFT-Large "
    "C_T_SKHT_V2...",
    flush=True,
)

model = raft_large(
    weights=weights,
    progress=True,
).to(device).eval()

torch.cuda.reset_peak_memory_stats(
    device
)

target_frames = read_target_frames(
    ALIGNED_FINAL_SIM / "frames"
)

generated_frames = {
    name: read_video(path)
    for name, path in VIDEO_PATHS.items()
}

all_frames = {
    "simulation_target": (
        target_frames
    ),
    **generated_frames,
}

all_flows = {}

for name, frames in all_frames.items():
    all_flows[name] = infer_flows(
        name=name,
        frames=frames,
        model=model,
        preprocess=preprocess,
        device=device,
    )

target_flow = all_flows[
    "simulation_target"
]

stored_flow_path = (
    ALIGNED_FINAL_SIM
    / "flows.npy"
)

stored_flow = np.load(
    stored_flow_path,
    allow_pickle=False,
).astype(np.float32)

stored_flow_epe = np.sqrt(
    np.square(
        target_flow - stored_flow
    ).sum(axis=1)
)

artifact = torch.load(
    BALANCED_ARTIFACT,
    map_location="cpu",
    weights_only=False,
)

latent_mask = extract_latent_mask(
    artifact
)

rgb_mask = expand_mask_to_rgb_frames(
    latent_mask
)

core_transition_mask = (
    rgb_mask[:-1]
    | rgb_mask[1:]
)

dilated_rgb_mask = dilate(
    rgb_mask,
    radius=4,
)

dilated_transition_mask = (
    dilated_rgb_mask[:-1]
    | dilated_rgb_mask[1:]
)

target_magnitude = np.sqrt(
    np.square(
        target_flow
    ).sum(axis=1)
)

core_mask = (
    core_transition_mask.numpy()
)

dilated_mask = (
    dilated_transition_mask.numpy()
)

regions = {
    "transport_core_all": (
        core_mask
    ),
    "transport_core_moving_gt_0p25": (
        core_mask
        & (target_magnitude > 0.25)
    ),
    "transport_dilated_all": (
        dilated_mask
    ),
    "transport_dilated_moving_gt_0p25": (
        dilated_mask
        & (target_magnitude > 0.25)
    ),
    "background_outside_dilated": (
        ~dilated_mask
    ),
}

metrics_by_region = {}

for region_name, region_mask in (
    regions.items()
):
    metrics_by_region[
        region_name
    ] = {
        method: evaluate_flow(
            predicted=all_flows[
                method
            ],
            target=target_flow,
            mask=region_mask,
        )
        for method in VIDEO_PATHS
    }

comparisons = {}

for region_name in (
    "transport_core_all",
    "transport_core_moving_gt_0p25",
    "transport_dilated_all",
    "transport_dilated_moving_gt_0p25",
):
    values = metrics_by_region[
        region_name
    ]

    comparisons[
        region_name
    ] = {
        "balanced_correct_vs_shuffled_epe_improvement_percent": (
            improvement_percent(
                values[
                    "balanced_correct"
                ]["epe_mean"],
                values[
                    "balanced_shuffled"
                ]["epe_mean"],
            )
        ),
        "quality_correct_vs_baseline_epe_improvement_percent": (
            improvement_percent(
                values[
                    "quality_correct"
                ]["epe_mean"],
                values[
                    "realwonder_baseline"
                ]["epe_mean"],
            )
        ),
        "balanced_correct_vs_shuffled_magnitude_mae_improvement_percent": (
            improvement_percent(
                values[
                    "balanced_correct"
                ][
                    "flow_magnitude_mae"
                ],
                values[
                    "balanced_shuffled"
                ][
                    "flow_magnitude_mae"
                ],
            )
        ),
        "quality_correct_vs_baseline_magnitude_mae_improvement_percent": (
            improvement_percent(
                values[
                    "quality_correct"
                ][
                    "flow_magnitude_mae"
                ],
                values[
                    "realwonder_baseline"
                ][
                    "flow_magnitude_mae"
                ],
            )
        ),
    }

flow_files = {
    name: {
        "path": str(
            OUTPUT_DIR
            / f"{name}_raft_flow.npy"
        ),
        "sha256": sha256(
            OUTPUT_DIR
            / f"{name}_raft_flow.npy"
        ),
    }
    for name in all_flows
}

checks = {
    "target_frame_count_81": (
        target_frames.shape[0] == 81
    ),
    "all_generated_frame_counts_81": all(
        frames.shape[0] == 81
        for frames in generated_frames.values()
    ),
    "all_flow_shapes_valid": all(
        tuple(flow.shape)
        == (80, 2, 240, 416)
        for flow in all_flows.values()
    ),
    "all_flows_finite": all(
        np.isfinite(flow).all()
        for flow in all_flows.values()
    ),
    "core_mask_nonempty": bool(
        core_mask.any()
    ),
    "moving_core_mask_nonempty": bool(
        regions[
            "transport_core_moving_gt_0p25"
        ].any()
    ),
}

report = {
    "stage": (
        "raft_motion_adherence_evaluation"
    ),
    "evaluator": {
        "model": (
            "torchvision RAFT-Large"
        ),
        "weights": (
            "C_T_SKHT_V2"
        ),
        "input_resolution": [
            FLOW_HEIGHT,
            FLOW_WIDTH,
        ],
        "batch_size": BATCH_SIZE,
        "gpu_peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(
                device
            )
            / 1024
            / 1024
        ),
    },
    "target_flow_reproduction": {
        "stored_flow_path": str(
            stored_flow_path
        ),
        "stored_flow_shape": list(
            stored_flow.shape
        ),
        "regenerated_vs_stored_epe_mean": float(
            stored_flow_epe.mean()
        ),
        "regenerated_vs_stored_epe_p90": float(
            np.percentile(
                stored_flow_epe,
                90,
            )
        ),
    },
    "region_coverage": {
        name: float(mask.mean())
        for name, mask in regions.items()
    },
    "metrics_against_simulation_flow": (
        metrics_by_region
    ),
    "comparisons": comparisons,
    "flow_files": flow_files,
    "checks": checks,
    "all_checks_pass": all(
        checks.values()
    ),
    "decision_rules": {
        "correspondence_mechanism": (
            "Balanced Correct should have "
            "lower EPE than Balanced Shuffled."
        ),
        "baseline_improvement": (
            "Quality Correct should have "
            "lower EPE than RealWonder Baseline."
        ),
    },
}

report_path = (
    OUTPUT_DIR
    / "raft_motion_metrics.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("========== KEY RESULTS ==========")

for region_name in (
    "transport_core_moving_gt_0p25",
    "transport_dilated_moving_gt_0p25",
):
    values = metrics_by_region[
        region_name
    ]

    comparison = comparisons[
        region_name
    ]

    print()
    print("REGION:", region_name)

    for method in (
        "realwonder_baseline",
        "balanced_correct",
        "balanced_shuffled",
        "quality_correct",
    ):
        print(
            method,
            "EPE=",
            round(
                values[method][
                    "epe_mean"
                ],
                6,
            ),
            "MagnitudeMAE=",
            round(
                values[method][
                    "flow_magnitude_mae"
                ],
                6,
            ),
            "Angle=",
            round(
                values[method][
                    "angle_error_degrees_mean"
                ],
                6,
            ),
        )

    print(
        "Correct_vs_Shuffled_EPE_improvement_percent=",
        round(
            comparison[
                "balanced_correct_vs_shuffled_epe_improvement_percent"
            ],
            4,
        ),
    )

    print(
        "Quality_vs_Baseline_EPE_improvement_percent=",
        round(
            comparison[
                "quality_correct_vs_baseline_epe_improvement_percent"
            ],
            4,
        ),
    )

print()
print(
    "target_regenerated_vs_stored_EPE=",
    report[
        "target_flow_reproduction"
    ][
        "regenerated_vs_stored_epe_mean"
    ],
)

print(
    "all_checks_pass=",
    report["all_checks_pass"],
)

if not report["all_checks_pass"]:
    raise SystemExit(1)
