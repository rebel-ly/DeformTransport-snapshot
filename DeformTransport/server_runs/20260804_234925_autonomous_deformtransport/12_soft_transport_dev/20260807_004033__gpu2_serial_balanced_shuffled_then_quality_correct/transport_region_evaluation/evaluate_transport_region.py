from __future__ import annotations

import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


aligned_final_sim = Path(
    os.environ["ALIGNED_FINAL_SIM"]
)

full_gen_root = Path(
    os.environ["FULL_GEN_ROOT"]
)

second_wave_root = Path(
    os.environ["SECOND_WAVE_ROOT"]
)

balanced_artifact = Path(
    os.environ["BALANCED_ARTIFACT"]
)

quality_artifact = Path(
    os.environ["QUALITY_ARTIFACT"]
)

output_dir = Path(
    os.environ["LOCAL_EVAL_DIR"]
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


video_paths = {
    "simulation_target": (
        aligned_final_sim / "simulation.mp4"
    ),
    "realwonder_baseline": (
        full_gen_root
        / "baseline"
        / "aligned_santa_baseline_seed0.mp4"
    ),
    "balanced_correct": (
        full_gen_root
        / "balanced_correct_retry_gpu2_20260807_001201"
        / "aligned_santa_balanced_ramp4_correct_seed0.mp4"
    ),
    "balanced_shuffled": (
        second_wave_root
        / "balanced_shuffled"
        / "aligned_santa_balanced_ramp4_shuffled_seed0.mp4"
    ),
    "quality_correct": (
        second_wave_root
        / "quality_correct"
        / "aligned_santa_quality_ramp4_correct_seed0.mp4"
    ),
}

display_names = {
    "simulation_target": "Simulation Target",
    "realwonder_baseline": "RealWonder Baseline",
    "balanced_correct": "Balanced Correct",
    "balanced_shuffled": "Balanced Shuffled",
    "quality_correct": "Quality Correct",
}


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

    expected_shape = (
        81,
        480,
        832,
        3,
    )

    if tuple(frames.shape) != expected_shape:
        raise RuntimeError(
            f"{path}: expected {expected_shape}, "
            f"got {frames.shape}"
        )

    return frames


def extract_mask(state: dict) -> torch.Tensor:
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
            "unexpected transport mask shape: "
            f"{tuple(mask.shape)}"
        )

    return mask > 0


def expand_mask(
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    # Wan VAE temporal contract:
    # latent slot 0 -> frame 0
    # each later latent slot -> four RGB frames
    expanded = [latent_mask[0]]

    for index in range(1, 21):
        expanded.extend(
            [latent_mask[index]] * 4
        )

    video_mask = torch.stack(
        expanded,
        dim=0,
    )

    if video_mask.shape[0] != 81:
        raise RuntimeError(
            f"expanded frame count "
            f"{video_mask.shape[0]} != 81"
        )

    video_mask = F.interpolate(
        video_mask[:, None].float(),
        size=(480, 832),
        mode="nearest",
    )[:, 0]

    return video_mask > 0.5


def dilate(
    mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
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
        raise RuntimeError(
            "evaluation mask is empty"
        )

    return float(selected.mean())


def gradient_error(
    generated: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    generated = generated.astype(
        np.float32
    )

    target = target.astype(
        np.float32
    )

    generated_dx = (
        generated[:, :, 1:, :]
        - generated[:, :, :-1, :]
    )

    target_dx = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    generated_dy = (
        generated[:, 1:, :, :]
        - generated[:, :-1, :, :]
    )

    target_dy = (
        target[:, 1:, :, :]
        - target[:, :-1, :, :]
    )

    dx_error = np.abs(
        generated_dx - target_dx
    ).mean(axis=-1)

    dy_error = np.abs(
        generated_dy - target_dy
    ).mean(axis=-1)

    dx_mask = (
        mask[:, :, 1:]
        | mask[:, :, :-1]
    )

    dy_mask = (
        mask[:, 1:, :]
        | mask[:, :-1, :]
    )

    numerator = (
        float(dx_error[dx_mask].sum())
        + float(dy_error[dy_mask].sum())
    )

    denominator = (
        int(dx_mask.sum())
        + int(dy_mask.sum())
    )

    if denominator == 0:
        raise RuntimeError(
            "gradient mask is empty"
        )

    return numerator / denominator


def evaluate_against_target(
    generated: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    generated_f = generated.astype(
        np.float32
    )

    target_f = target.astype(
        np.float32
    )

    difference = (
        generated_f - target_f
    )

    l1_map = np.abs(
        difference
    ).mean(axis=-1)

    mse_map = np.square(
        difference
    ).mean(axis=-1)

    l1 = masked_mean(
        l1_map,
        mask,
    )

    mse = masked_mean(
        mse_map,
        mask,
    )

    psnr = float(
        10.0
        * math.log10(
            (255.0 ** 2) / mse
        )
    )

    generated_delta = (
        generated_f[1:]
        - generated_f[:-1]
    )

    target_delta = (
        target_f[1:]
        - target_f[:-1]
    )

    temporal_error_map = np.abs(
        generated_delta
        - target_delta
    ).mean(axis=-1)

    temporal_mask = (
        mask[1:]
        | mask[:-1]
    )

    selected_frame_l1 = {}

    for frame_index in (
        0,
        20,
        40,
        60,
        80,
    ):
        selected_frame_l1[
            str(frame_index)
        ] = masked_mean(
            l1_map[frame_index],
            mask[frame_index],
        )

    return {
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "temporal_change_error": (
            masked_mean(
                temporal_error_map,
                temporal_mask,
            )
        ),
        "gradient_error": (
            gradient_error(
                generated,
                target,
                mask,
            )
        ),
        "selected_frame_l1": (
            selected_frame_l1
        ),
    }


def pairwise_l1(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> float:
    difference = np.abs(
        first.astype(np.float32)
        - second.astype(np.float32)
    ).mean(axis=-1)

    return masked_mean(
        difference,
        mask,
    )


def improvement_percent(
    better: float,
    worse: float,
) -> float:
    return float(
        (worse - better)
        / worse
        * 100.0
    )


videos = {
    name: read_video(path)
    for name, path in video_paths.items()
}

balanced_state = torch.load(
    balanced_artifact,
    map_location="cpu",
    weights_only=False,
)

quality_state = torch.load(
    quality_artifact,
    map_location="cpu",
    weights_only=False,
)

balanced_latent_mask = extract_mask(
    balanced_state
)

quality_latent_mask = extract_mask(
    quality_state
)

if not torch.equal(
    balanced_latent_mask,
    quality_latent_mask,
):
    raise RuntimeError(
        "Balanced and Quality masks differ"
    )

core_mask_torch = expand_mask(
    balanced_latent_mask
)

dilated_mask_torch = dilate(
    core_mask_torch,
    radius=8,
)

core_mask = core_mask_torch.numpy()

dilated_mask = (
    dilated_mask_torch.numpy()
)

background_mask = (
    ~dilated_mask
)

regions = {
    "transport_core": core_mask,
    "transport_dilated_8px": (
        dilated_mask
    ),
    "background_outside_dilated": (
        background_mask
    ),
}

target = videos[
    "simulation_target"
]

method_names = [
    "realwonder_baseline",
    "balanced_correct",
    "balanced_shuffled",
    "quality_correct",
]

metrics_by_region = {}

for region_name, region_mask in (
    regions.items()
):
    metrics_by_region[
        region_name
    ] = {
        method: evaluate_against_target(
            videos[method],
            target,
            region_mask,
        )
        for method in method_names
    }

pairwise_by_region = {}

for region_name, region_mask in (
    regions.items()
):
    pairwise_by_region[
        region_name
    ] = {
        "balanced_correct_vs_shuffled_l1": (
            pairwise_l1(
                videos[
                    "balanced_correct"
                ],
                videos[
                    "balanced_shuffled"
                ],
                region_mask,
            )
        ),
        "baseline_vs_balanced_correct_l1": (
            pairwise_l1(
                videos[
                    "realwonder_baseline"
                ],
                videos[
                    "balanced_correct"
                ],
                region_mask,
            )
        ),
        "baseline_vs_quality_correct_l1": (
            pairwise_l1(
                videos[
                    "realwonder_baseline"
                ],
                videos[
                    "quality_correct"
                ],
                region_mask,
            )
        ),
    }

correct_vs_shuffled = {}

for region_name in (
    "transport_core",
    "transport_dilated_8px",
):
    correct_metrics = (
        metrics_by_region[
            region_name
        ]["balanced_correct"]
    )

    shuffled_metrics = (
        metrics_by_region[
            region_name
        ]["balanced_shuffled"]
    )

    correct_vs_shuffled[
        region_name
    ] = {
        "l1_improvement_percent": (
            improvement_percent(
                correct_metrics["l1"],
                shuffled_metrics["l1"],
            )
        ),
        "mse_improvement_percent": (
            improvement_percent(
                correct_metrics["mse"],
                shuffled_metrics["mse"],
            )
        ),
        "temporal_error_improvement_percent": (
            improvement_percent(
                correct_metrics[
                    "temporal_change_error"
                ],
                shuffled_metrics[
                    "temporal_change_error"
                ],
            )
        ),
        "gradient_error_improvement_percent": (
            improvement_percent(
                correct_metrics[
                    "gradient_error"
                ],
                shuffled_metrics[
                    "gradient_error"
                ],
            )
        ),
    }

coverage = {
    "transport_core_global_fraction": (
        float(core_mask.mean())
    ),
    "transport_dilated_global_fraction": (
        float(dilated_mask.mean())
    ),
    "transport_core_selected_frames": {
        str(index): float(
            core_mask[index].mean()
        )
        for index in (
            0,
            20,
            40,
            60,
            80,
        )
    },
}

selected_indices = [
    0,
    20,
    40,
    60,
    80,
]

# Transport mask overlay
overlay_canvas = Image.new(
    "RGB",
    (
        416 * len(selected_indices),
        270,
    ),
    "white",
)

overlay_draw = ImageDraw.Draw(
    overlay_canvas
)

for column, frame_index in enumerate(
    selected_indices
):
    frame = target[
        frame_index
    ].astype(np.float32)

    core = core_mask[
        frame_index
    ]

    dilated = dilated_mask[
        frame_index
    ]

    overlay = frame.copy()

    overlay[dilated] = (
        0.72 * overlay[dilated]
        + 0.28
        * np.array(
            [255.0, 220.0, 0.0],
            dtype=np.float32,
        )
    )

    overlay[core] = (
        0.58 * overlay[core]
        + 0.42
        * np.array(
            [0.0, 255.0, 80.0],
            dtype=np.float32,
        )
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    image = Image.fromarray(
        overlay
    ).resize(
        (416, 240),
        Image.Resampling.BILINEAR,
    )

    left = column * 416

    overlay_draw.text(
        (left + 8, 8),
        f"Transport support | frame {frame_index}",
        fill="black",
    )

    overlay_canvas.paste(
        image,
        (left, 30),
    )

overlay_path = (
    output_dir
    / "transport_mask_overlay.png"
)

overlay_canvas.save(
    overlay_path
)

# Five-way transport-region crops
row_order = [
    "simulation_target",
    "realwonder_baseline",
    "balanced_correct",
    "balanced_shuffled",
    "quality_correct",
]

crop_canvas = Image.new(
    "RGB",
    (
        416 * len(selected_indices),
        270 * len(row_order),
    ),
    "white",
)

crop_draw = ImageDraw.Draw(
    crop_canvas
)

for column, frame_index in enumerate(
    selected_indices
):
    mask = dilated_mask[
        frame_index
    ]

    ys, xs = np.where(mask)

    if len(xs) == 0:
        x0, y0, x1, y1 = (
            0,
            0,
            832,
            480,
        )
    else:
        padding = 24

        x0 = max(
            0,
            int(xs.min()) - padding,
        )

        x1 = min(
            832,
            int(xs.max()) + padding + 1,
        )

        y0 = max(
            0,
            int(ys.min()) - padding,
        )

        y1 = min(
            480,
            int(ys.max()) + padding + 1,
        )

    for row, name in enumerate(
        row_order
    ):
        frame = videos[name][
            frame_index
        ]

        crop = Image.fromarray(
            frame[
                y0:y1,
                x0:x1,
            ]
        )

        crop.thumbnail(
            (416, 240),
            Image.Resampling.BILINEAR,
        )

        fitted = Image.new(
            "RGB",
            (416, 240),
            "black",
        )

        fitted.paste(
            crop,
            (
                (416 - crop.width) // 2,
                (240 - crop.height) // 2,
            ),
        )

        left = column * 416
        top = row * 270

        crop_draw.text(
            (left + 8, top + 8),
            (
                f"{display_names[name]} "
                f"| frame {frame_index}"
            ),
            fill="black",
        )

        crop_canvas.paste(
            fitted,
            (left, top + 30),
        )

crop_path = (
    output_dir
    / "five_way_transport_region_crops.png"
)

crop_canvas.save(
    crop_path
)

checks = {
    "balanced_quality_masks_equal": True,
    "latent_mask_shape": (
        tuple(
            balanced_latent_mask.shape
        )
        == (21, 60, 104)
    ),
    "video_mask_shape": (
        tuple(core_mask.shape)
        == (81, 480, 832)
    ),
    "core_mask_nonempty": bool(
        core_mask.any()
    ),
    "dilated_mask_nonempty": bool(
        dilated_mask.any()
    ),
    "correct_better_than_shuffled_core_l1": (
        metrics_by_region[
            "transport_core"
        ]["balanced_correct"]["l1"]
        <
        metrics_by_region[
            "transport_core"
        ]["balanced_shuffled"]["l1"]
    ),
}

report = {
    "stage": (
        "transport_region_local_evaluation"
    ),
    "coverage": coverage,
    "metrics_against_simulation_target": (
        metrics_by_region
    ),
    "pairwise_metrics": (
        pairwise_by_region
    ),
    "correct_vs_shuffled_improvement": (
        correct_vs_shuffled
    ),
    "checks": checks,
    "all_checks_pass": all(
        checks.values()
    ),
    "transport_mask_overlay": str(
        overlay_path
    ),
    "transport_region_contact_sheet": str(
        crop_path
    ),
    "interpretation_boundary": (
        "The simulation video is a coarse mechanism "
        "reference, not real-world RGB ground truth."
    ),
}

metrics_path = (
    output_dir
    / "transport_region_metrics.json"
)

metrics_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        report,
        indent=2,
    )
)

if not report["all_checks_pass"]:
    raise SystemExit(1)
