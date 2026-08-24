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


output_dir = Path(os.environ["COND_EVAL_DIR"])
aligned_final_sim = Path(os.environ["ALIGNED_FINAL_SIM"])
full_gen_root = Path(os.environ["FULL_GEN_ROOT"])
second_wave_root = Path(os.environ["SECOND_WAVE_ROOT"])
cond_root = Path(os.environ["COND_ROOT"])
quality_artifact = Path(os.environ["QUALITY_ARTIFACT"])

output_dir.mkdir(parents=True, exist_ok=True)

video_paths = {
    "simulation_target": (
        aligned_final_sim / "simulation.mp4"
    ),
    "realwonder_baseline": (
        full_gen_root
        / "baseline"
        / "aligned_santa_baseline_seed0.mp4"
    ),
    "quality_inter_step_correct": (
        second_wave_root
        / "quality_correct"
        / "aligned_santa_quality_ramp4_correct_seed0.mp4"
    ),
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

display_names = {
    "simulation_target": "Simulation Target",
    "realwonder_baseline": "RealWonder Baseline",
    "quality_inter_step_correct": "Quality Inter-step Correct",
    "condition_correct": "Condition Correct",
    "condition_shuffled": "Condition Shuffled",
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


def extract_mask(state: dict) -> torch.Tensor:
    if "transport_mask" not in state:
        raise KeyError("artifact missing transport_mask")

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


def expand_mask(latent_mask: torch.Tensor) -> torch.Tensor:
    expanded = [latent_mask[0]]

    for index in range(1, 21):
        expanded.extend([latent_mask[index]] * 4)

    mask = torch.stack(expanded, dim=0)

    if tuple(mask.shape) != (81, 60, 104):
        raise RuntimeError(
            f"unexpected expanded mask: {tuple(mask.shape)}"
        )

    mask = F.interpolate(
        mask[:, None].float(),
        size=(480, 832),
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

    dx_mask = mask[:, :, 1:] | mask[:, :, :-1]
    dy_mask = mask[:, 1:, :] | mask[:, :-1, :]

    numerator = (
        float(dx_error[dx_mask].sum())
        + float(dy_error[dy_mask].sum())
    )

    denominator = int(dx_mask.sum()) + int(dy_mask.sum())

    if denominator == 0:
        raise RuntimeError("empty gradient mask")

    return numerator / denominator


def evaluate(
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

    psnr = float(
        10.0 * math.log10((255.0 ** 2) / mse)
    )

    generated_delta = (
        generated_f[1:] - generated_f[:-1]
    )
    target_delta = (
        target_f[1:] - target_f[:-1]
    )

    temporal_error = np.abs(
        generated_delta - target_delta
    ).mean(axis=-1)

    temporal_mask = mask[1:] | mask[:-1]

    selected_frame_l1 = {}

    for frame_index in (0, 20, 40, 60, 80):
        selected_frame_l1[str(frame_index)] = (
            masked_mean(
                l1_map[frame_index],
                mask[frame_index],
            )
        )

    return {
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "temporal_change_error": masked_mean(
            temporal_error,
            temporal_mask,
        ),
        "gradient_error": gradient_error(
            generated,
            target,
            mask,
        ),
        "selected_frame_l1": selected_frame_l1,
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


videos = {
    name: read_video(path)
    for name, path in video_paths.items()
}

artifact = torch.load(
    quality_artifact,
    map_location="cpu",
    weights_only=False,
)

latent_mask = extract_mask(artifact)
core_mask_t = expand_mask(latent_mask)
dilated_mask_t = dilate(core_mask_t, radius=8)

core_mask = core_mask_t.numpy()
dilated_mask = dilated_mask_t.numpy()
background_mask = ~dilated_mask

regions = {
    "transport_core": core_mask,
    "transport_dilated_8px": dilated_mask,
    "background_outside_dilated": background_mask,
}

target = videos["simulation_target"]

method_names = [
    "realwonder_baseline",
    "quality_inter_step_correct",
    "condition_correct",
    "condition_shuffled",
]

metrics = {}

for region_name, region_mask in regions.items():
    metrics[region_name] = {
        method: evaluate(
            videos[method],
            target,
            region_mask,
        )
        for method in method_names
    }

comparisons = {}

for region_name in (
    "transport_core",
    "transport_dilated_8px",
):
    values = metrics[region_name]

    condition_correct = values["condition_correct"]
    condition_shuffled = values["condition_shuffled"]
    baseline = values["realwonder_baseline"]
    inter_step = values["quality_inter_step_correct"]

    comparisons[region_name] = {
        "condition_correct_vs_shuffled": {
            key: improvement_percent(
                condition_correct[key],
                condition_shuffled[key],
            )
            for key in (
                "l1",
                "mse",
                "temporal_change_error",
                "gradient_error",
            )
        },
        "condition_correct_vs_baseline": {
            key: improvement_percent(
                condition_correct[key],
                baseline[key],
            )
            for key in (
                "l1",
                "mse",
                "temporal_change_error",
                "gradient_error",
            )
        },
        "condition_correct_vs_inter_step": {
            key: improvement_percent(
                condition_correct[key],
                inter_step[key],
            )
            for key in (
                "l1",
                "mse",
                "temporal_change_error",
                "gradient_error",
            )
        },
    }

checks = {
    "all_video_shapes_valid": all(
        tuple(video.shape) == (81, 480, 832, 3)
        for video in videos.values()
    ),
    "all_videos_finite": all(
        np.isfinite(video).all()
        for video in videos.values()
    ),
    "latent_mask_shape_valid": (
        tuple(latent_mask.shape)
        == (21, 60, 104)
    ),
    "core_mask_shape_valid": (
        tuple(core_mask.shape)
        == (81, 480, 832)
    ),
    "core_mask_nonempty": bool(core_mask.any()),
    "dilated_mask_nonempty": bool(dilated_mask.any()),
}

decision_flags = {
    "condition_correct_l1_better_than_shuffled_core": (
        metrics["transport_core"]["condition_correct"]["l1"]
        <
        metrics["transport_core"]["condition_shuffled"]["l1"]
    ),
    "condition_correct_l1_better_than_baseline_core": (
        metrics["transport_core"]["condition_correct"]["l1"]
        <
        metrics["transport_core"]["realwonder_baseline"]["l1"]
    ),
    "condition_correct_temporal_better_than_baseline_core": (
        metrics["transport_core"]["condition_correct"][
            "temporal_change_error"
        ]
        <
        metrics["transport_core"]["realwonder_baseline"][
            "temporal_change_error"
        ]
    ),
}

selected_frames = [0, 20, 40, 60, 80]
row_order = [
    "simulation_target",
    "realwonder_baseline",
    "quality_inter_step_correct",
    "condition_correct",
    "condition_shuffled",
]

cell_width = 416
cell_height = 270

canvas = Image.new(
    "RGB",
    (
        cell_width * len(selected_frames),
        cell_height * len(row_order),
    ),
    "white",
)

draw = ImageDraw.Draw(canvas)

for column, frame_index in enumerate(selected_frames):
    mask = dilated_mask[frame_index]

    ys, xs = np.where(mask)

    if len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, 832, 480
    else:
        padding = 24

        x0 = max(0, int(xs.min()) - padding)
        x1 = min(832, int(xs.max()) + padding + 1)
        y0 = max(0, int(ys.min()) - padding)
        y1 = min(480, int(ys.max()) + padding + 1)

    for row, name in enumerate(row_order):
        frame = videos[name][frame_index]

        crop = Image.fromarray(
            frame[y0:y1, x0:x1]
        )

        crop.thumbnail(
            (cell_width, 240),
            Image.Resampling.BILINEAR,
        )

        fitted = Image.new(
            "RGB",
            (cell_width, 240),
            "black",
        )

        fitted.paste(
            crop,
            (
                (cell_width - crop.width) // 2,
                (240 - crop.height) // 2,
            ),
        )

        left = column * cell_width
        top = row * cell_height

        draw.text(
            (left + 8, top + 8),
            (
                f"{display_names[name]} "
                f"| frame {frame_index}"
            ),
            fill="black",
        )

        canvas.paste(
            fitted,
            (left, top + 30),
        )

contact_sheet_path = (
    output_dir
    / "condition_space_contact_sheet.png"
)

canvas.save(contact_sheet_path)

report = {
    "stage": "condition_space_local_evaluation",
    "coverage": {
        "transport_core_fraction": float(
            core_mask.mean()
        ),
        "transport_dilated_fraction": float(
            dilated_mask.mean()
        ),
    },
    "metrics_against_simulation_target": metrics,
    "comparisons_improvement_percent": comparisons,
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "decision_flags": decision_flags,
    "contact_sheet": str(contact_sheet_path),
    "interpretation_boundary": (
        "Simulation Target is a coarse mechanism "
        "reference, not real RGB ground truth."
    ),
}

metrics_path = (
    output_dir
    / "condition_space_local_metrics.json"
)

metrics_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("========== LOCAL RESULTS ==========")

for region_name in (
    "transport_core",
    "transport_dilated_8px",
):
    print()
    print("REGION:", region_name)

    for method in method_names:
        values = metrics[region_name][method]

        print(
            method,
            "L1=",
            round(values["l1"], 6),
            "PSNR=",
            round(values["psnr"], 6),
            "Temporal=",
            round(
                values["temporal_change_error"],
                6,
            ),
            "Gradient=",
            round(values["gradient_error"], 6),
        )

    print(
        "ConditionCorrect_vs_Shuffled =",
        comparisons[region_name][
            "condition_correct_vs_shuffled"
        ],
    )

    print(
        "ConditionCorrect_vs_Baseline =",
        comparisons[region_name][
            "condition_correct_vs_baseline"
        ],
    )

    print(
        "ConditionCorrect_vs_InterStep =",
        comparisons[region_name][
            "condition_correct_vs_inter_step"
        ],
    )

print()
print("decision_flags =", decision_flags)
print("all_checks_pass =", report["all_checks_pass"])

if not report["all_checks_pass"]:
    raise SystemExit(1)
