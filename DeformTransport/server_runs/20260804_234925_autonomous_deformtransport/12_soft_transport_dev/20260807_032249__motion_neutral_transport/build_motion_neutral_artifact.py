from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F


source_path = Path(os.environ["QUALITY_ARTIFACT"])
output_path = Path(os.environ["MOTION_NEUTRAL_ARTIFACT"])
output_dir = output_path.parent

state = torch.load(
    source_path,
    map_location="cpu",
    weights_only=False,
)

required_keys = {
    "correct_transport_residual",
    "shuffled_transport_residual",
    "transport_mask",
}

missing = required_keys.difference(state)

if missing:
    raise KeyError(f"missing keys: {sorted(missing)}")


def normalize_residual(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu()

    if tuple(value.shape) != (1, 21, 16, 60, 104):
        raise RuntimeError(
            f"unexpected residual shape: {tuple(value.shape)}"
        )

    return value.to(torch.float32)


def normalize_mask(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu()

    while value.ndim > 4 and value.shape[0] == 1:
        value = value.squeeze(0)

    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]

    if tuple(value.shape) != (21, 60, 104):
        raise RuntimeError(
            f"unexpected mask shape: {tuple(value.shape)}"
        )

    return value[None, :, None].to(torch.float32)


correct = normalize_residual(
    state["correct_transport_residual"]
)

shuffled = normalize_residual(
    state["shuffled_transport_residual"]
)

mask = normalize_mask(state["transport_mask"])


def masked_local_highpass(
    residual: torch.Tensor,
    spatial_mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    batch, frames, channels, height, width = (
        residual.shape
    )

    flat_residual = residual.reshape(
        batch * frames,
        channels,
        height,
        width,
    )

    flat_mask = spatial_mask.expand(
        batch,
        frames,
        1,
        height,
        width,
    ).reshape(
        batch * frames,
        1,
        height,
        width,
    )

    padding = kernel_size // 2

    numerator = F.avg_pool2d(
        flat_residual * flat_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    denominator = F.avg_pool2d(
        flat_mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    local_mean = numerator / denominator.clamp_min(1e-6)

    highpass = (
        flat_residual - local_mean
    ) * flat_mask

    return highpass.reshape_as(residual)


def mean_abs(value: torch.Tensor) -> float:
    return float(value.abs().mean())


original_correct_energy = mean_abs(correct)
original_shuffled_energy = mean_abs(shuffled)

candidates = {}

for kernel in (3, 5, 7, 9):
    correct_hp = masked_local_highpass(
        correct,
        mask,
        kernel,
    )

    shuffled_hp = masked_local_highpass(
        shuffled,
        mask,
        kernel,
    )

    correct_ratio = (
        mean_abs(correct_hp)
        / original_correct_energy
    )

    shuffled_ratio = (
        mean_abs(shuffled_hp)
        / original_shuffled_energy
    )

    candidates[kernel] = {
        "correct": correct_hp,
        "shuffled": shuffled_hp,
        "correct_retained_ratio": correct_ratio,
        "shuffled_retained_ratio": shuffled_ratio,
        "correct_mean_abs": mean_abs(correct_hp),
        "shuffled_mean_abs": mean_abs(shuffled_hp),
    }

selected_kernel = min(
    candidates,
    key=lambda kernel: abs(
        candidates[kernel][
            "correct_retained_ratio"
        ]
        - 0.50
    ),
)

selected = candidates[selected_kernel]

correct_hp = selected["correct"]
shuffled_hp = selected["shuffled"]

checks = {
    "correct_shape_valid": (
        tuple(correct_hp.shape)
        == tuple(correct.shape)
    ),
    "shuffled_shape_valid": (
        tuple(shuffled_hp.shape)
        == tuple(shuffled.shape)
    ),
    "correct_finite": bool(
        torch.isfinite(correct_hp).all()
    ),
    "shuffled_finite": bool(
        torch.isfinite(shuffled_hp).all()
    ),
    "correct_slot0_zero": bool(
        torch.count_nonzero(
            correct_hp[:, 0]
        ) == 0
    ),
    "shuffled_slot0_zero": bool(
        torch.count_nonzero(
            shuffled_hp[:, 0]
        ) == 0
    ),
    "correct_outside_mask_zero": bool(
        torch.count_nonzero(
            correct_hp * (1.0 - mask)
        ) == 0
    ),
    "shuffled_outside_mask_zero": bool(
        torch.count_nonzero(
            shuffled_hp * (1.0 - mask)
        ) == 0
    ),
    "correct_nonzero": bool(
        torch.count_nonzero(correct_hp) > 0
    ),
    "shuffled_nonzero": bool(
        torch.count_nonzero(shuffled_hp) > 0
    ),
}

if not all(checks.values()):
    raise RuntimeError(f"contract failed: {checks}")

output_state = deepcopy(state)

output_state[
    "correct_transport_residual"
] = correct_hp.to(
    state["correct_transport_residual"].dtype
)

output_state[
    "shuffled_transport_residual"
] = shuffled_hp.to(
    state["shuffled_transport_residual"].dtype
)

output_state["motion_neutral_metadata"] = {
    "source_artifact": str(source_path),
    "method": (
        "masked local-mean subtraction "
        "from transported latent residual"
    ),
    "candidate_kernels": [3, 5, 7, 9],
    "selection_target_retained_ratio": 0.50,
    "selected_kernel": selected_kernel,
    "correct_retained_ratio": selected[
        "correct_retained_ratio"
    ],
    "shuffled_retained_ratio": selected[
        "shuffled_retained_ratio"
    ],
}

torch.save(output_state, output_path)

report = {
    "source_artifact": str(source_path),
    "output_artifact": str(output_path),
    "original_correct_mean_abs": (
        original_correct_energy
    ),
    "original_shuffled_mean_abs": (
        original_shuffled_energy
    ),
    "candidate_summary": {
        str(kernel): {
            key: value
            for key, value in values.items()
            if not torch.is_tensor(value)
        }
        for kernel, values in candidates.items()
    },
    "selected_kernel": selected_kernel,
    "selected_correct_retained_ratio": (
        selected["correct_retained_ratio"]
    ),
    "selected_shuffled_retained_ratio": (
        selected["shuffled_retained_ratio"]
    ),
    "selected_correct_mean_abs": (
        selected["correct_mean_abs"]
    ),
    "selected_shuffled_mean_abs": (
        selected["shuffled_mean_abs"]
    ),
    "checks": checks,
    "all_checks_pass": all(checks.values()),
}

report_path = (
    output_dir
    / "motion_neutral_artifact_report.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
