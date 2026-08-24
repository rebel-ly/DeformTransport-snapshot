from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


base_path = Path(sys.argv[1]).resolve()
candidate_path = Path(sys.argv[2]).resolve()
full_path = Path(sys.argv[3]).resolve()
report_path = Path(sys.argv[4]).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


for p in (base_path,):
    if not p.is_file():
        raise FileNotFoundError(p)

for p in (
    candidate_path,
    full_path,
    report_path,
):
    if p.exists():
        raise FileExistsError(
            f"refusing to overwrite: {p}"
        )


base = torch.load(
    base_path,
    map_location="cpu",
    weights_only=False,
)


required = {
    "latent_frame_indices",
    "target_latent",
    "correct_transported_latent",
    "shuffled_transported_latent",
    "transport_mask",
    "contribution_count",
    "transport_weight",
    "hard_reference_mask",
    "soft_only_mask",
    "transport_validity_mode",
}

missing = sorted(required - set(base))

if missing:
    raise KeyError(
        f"missing base keys: {missing}"
    )


indices = base[
    "latent_frame_indices"
].to(torch.long)

target = base[
    "target_latent"
].to(torch.float32)

raw_correct = base[
    "correct_transported_latent"
].to(torch.float32)

raw_shuffled = base[
    "shuffled_transported_latent"
].to(torch.float32)

hard = base[
    "hard_reference_mask"
].to(torch.bool)

soft_only = base[
    "soft_only_mask"
].to(torch.bool)

weight = base[
    "transport_weight"
].to(torch.float32)

base_mask = base[
    "transport_mask"
].to(torch.bool)

base_count = base[
    "contribution_count"
].to(torch.long)


expected_indices = torch.arange(
    0,
    81,
    4,
    dtype=torch.long,
)

assert torch.equal(
    indices,
    expected_indices,
)

assert tuple(target.shape) == (
    1,
    21,
    16,
    60,
    104,
)

assert tuple(hard.shape) == (
    21,
    1,
    60,
    104,
)

assert (
    base["transport_validity_mode"]
    == "source_and_future_visible"
)


# ------------------------------------------------------------
# Frozen Santa Quality recipe.
# ------------------------------------------------------------
tau = 0.75
alpha = 0.25

schedule = torch.tensor(
    [
        0.0,
        0.0833333358168602,
        0.1666666716337204,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    ],
    dtype=torch.float32,
)


candidate_mask = (
    hard
    |
    (
        soft_only
        & (weight >= tau)
    )
)


# Quality mask must remain a subset of the
# already accepted Gate-0.25 Soft support.
if bool(
    (
        candidate_mask
        & ~base_mask
    ).any()
):
    raise RuntimeError(
        "Quality mask exceeds Gate-0.25 base support"
    )


# Preserve actual contribution counts on retained cells.
candidate_count = torch.where(
    candidate_mask,
    base_count,
    torch.zeros_like(base_count),
)

if not torch.equal(
    candidate_mask,
    candidate_count > 0,
):
    raise RuntimeError(
        "candidate mask/count contract failed"
    )


mask5 = candidate_mask.unsqueeze(0)

alpha5 = schedule.reshape(
    1,
    21,
    1,
    1,
    1,
)


correct_fused = (
    target
    + alpha5
    * mask5.to(torch.float32)
    * (
        raw_correct
        - target
    )
)

shuffled_fused = (
    target
    + alpha5
    * mask5.to(torch.float32)
    * (
        raw_shuffled
        - target
    )
)


correct_residual = (
    correct_fused
    - target
)

shuffled_residual = (
    shuffled_fused
    - target
)


# ------------------------------------------------------------
# Critical contracts.
# ------------------------------------------------------------
for name, tensor in {
    "target": target,
    "correct_fused": correct_fused,
    "shuffled_fused": shuffled_fused,
    "correct_residual": correct_residual,
    "shuffled_residual": shuffled_residual,
}.items():

    if not bool(
        torch.isfinite(tensor).all()
    ):
        raise RuntimeError(
            f"{name} contains NaN/Inf"
        )


assert torch.equal(
    correct_fused[:, 0],
    target[:, 0],
)

assert torch.equal(
    shuffled_fused[:, 0],
    target[:, 0],
)

assert torch.equal(
    correct_residual[:, 0],
    torch.zeros_like(
        correct_residual[:, 0]
    ),
)

assert torch.equal(
    shuffled_residual[:, 0],
    torch.zeros_like(
        shuffled_residual[:, 0]
    ),
)


candidate = {
    "format_version": 1,

    "artifact_kind":
        "wan_temporal_alpha_schedule_candidate",

    "candidate_id":
        "tree_quality_a0p250_t0p750_ramp4",

    "group":
        "quality",

    "base_artifact":
        str(base_path),

    "alpha":
        alpha,

    "threshold":
        tau,

    "schedule_name":
        "ramp4",

    "alpha_schedule":
        schedule,

    "transport_validity_mode":
        base["transport_validity_mode"],

    "latent_frame_indices":
        indices,

    "target_latent":
        target,

    "correct_fused_latent":
        correct_fused,

    "shuffled_fused_latent":
        shuffled_fused,

    "transport_mask":
        candidate_mask,
}


torch.save(
    candidate,
    candidate_path,
)


# ------------------------------------------------------------
# Full-generation-compatible form.
# ------------------------------------------------------------
full = {
    "format_version": 1,

    "artifact_kind":
        "wan_full_generation_transport",

    "candidate_id":
        candidate[
            "candidate_id"
        ],

    "group":
        "quality",

    "base_artifact":
        str(base_path),

    "alpha":
        alpha,

    "threshold":
        tau,

    "schedule_name":
        "ramp4",

    "alpha_schedule":
        schedule,

    "transport_validity_mode":
        base["transport_validity_mode"],

    "latent_frame_indices":
        indices,

    "target_latent":
        target,

    # Preserve the raw geometric transport for audit.
    "correct_transported_latent":
        raw_correct,

    "shuffled_transported_latent":
        raw_shuffled,

    # Production replace loader reads these.
    "correct_fused_latent":
        correct_fused,

    "shuffled_fused_latent":
        shuffled_fused,

    # Production residual loader reads these first.
    "correct_transport_residual":
        correct_residual,

    "shuffled_transport_residual":
        shuffled_residual,

    "transport_mask":
        candidate_mask,

    "contribution_count":
        candidate_count,

    "compatibility_contract": {
        "residual_definition":
            "fused_latent - artifact_target_latent",

        "correct_slot0_residual_zero":
            True,

        "shuffled_slot0_residual_zero":
            True,

        "mask_equals_count_positive":
            True,

        "intended_injection_mode":
            "inter_step_residual",

        "intended_injection_scale":
            1.0,

        "intended_injection_step":
            0,
    },
}


torch.save(
    full,
    full_path,
)


# ------------------------------------------------------------
# Reload validation.
# ------------------------------------------------------------
candidate_reload = torch.load(
    candidate_path,
    map_location="cpu",
    weights_only=False,
)

full_reload = torch.load(
    full_path,
    map_location="cpu",
    weights_only=False,
)


checks = {
    "indices_exact":
        torch.equal(
            indices,
            expected_indices,
        ),

    "quality_mask_subset_gate025":
        not bool(
            (
                candidate_mask
                & ~base_mask
            ).any()
        ),

    "mask_count_contract":
        torch.equal(
            full_reload[
                "transport_mask"
            ],
            full_reload[
                "contribution_count"
            ] > 0,
        ),

    "correct_slot0_fused_exact":
        torch.equal(
            correct_fused[:, 0],
            target[:, 0],
        ),

    "shuffled_slot0_fused_exact":
        torch.equal(
            shuffled_fused[:, 0],
            target[:, 0],
        ),

    "correct_slot0_residual_zero":
        torch.equal(
            full_reload[
                "correct_transport_residual"
            ][:, 0],
            torch.zeros_like(
                target[:, 0]
            ),
        ),

    "shuffled_slot0_residual_zero":
        torch.equal(
            full_reload[
                "shuffled_transport_residual"
            ][:, 0],
            torch.zeros_like(
                target[:, 0]
            ),
        ),

    "correct_residual_exact":
        torch.equal(
            full_reload[
                "correct_transport_residual"
            ],
            full_reload[
                "correct_fused_latent"
            ]
            -
            full_reload[
                "target_latent"
            ],
        ),

    "shuffled_residual_exact":
        torch.equal(
            full_reload[
                "shuffled_transport_residual"
            ],
            full_reload[
                "shuffled_fused_latent"
            ]
            -
            full_reload[
                "target_latent"
            ],
        ),

    "candidate_saved_exact":
        torch.equal(
            candidate_reload[
                "correct_fused_latent"
            ],
            correct_fused,
        ),
}


correct_residual_mean_abs = float(
    correct_residual.abs().mean()
)

shuffled_residual_mean_abs = float(
    shuffled_residual.abs().mean()
)

correct_vs_shuffled_mean_abs = float(
    (
        correct_fused
        - shuffled_fused
    ).abs().mean()
)


report = {
    "status":
        "TREE_QUALITY_RAMP4_FULL_GENERATION_READY",

    "base_artifact":
        str(base_path),

    "base_sha256":
        sha256(base_path),

    "candidate":
        str(candidate_path),

    "candidate_sha256":
        sha256(candidate_path),

    "full_generation":
        str(full_path),

    "full_generation_sha256":
        sha256(full_path),

    "frozen_recipe": {
        "soft_gate":
            0.25,

        "quality_tau":
            tau,

        "alpha":
            alpha,

        "schedule":
            schedule.tolist(),

        "mask_formula":
            (
                "hard_reference_mask OR "
                "(soft_only_mask AND "
                "transport_weight >= 0.75)"
            ),

        "fused_formula":
            (
                "target + alpha_schedule * mask * "
                "(raw_transport - target)"
            ),
    },

    "support": {
        "gate025_cells":
            int(base_mask.sum()),

        "hard_cells":
            int(hard.sum()),

        "quality_cells":
            int(candidate_mask.sum()),

        "removed_from_gate025":
            int(
                (
                    base_mask
                    & ~candidate_mask
                ).sum()
            ),
    },

    "residual": {
        "correct_mean_abs":
            correct_residual_mean_abs,

        "shuffled_mean_abs":
            shuffled_residual_mean_abs,

        "correct_vs_shuffled_mean_abs":
            correct_vs_shuffled_mean_abs,
    },

    "checks":
        checks,

    "all_checks_pass":
        all(checks.values()),
}


report_path.write_text(
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


if not report[
    "all_checks_pass"
]:
    raise SystemExit(1)


print(
    "TREE_QUALITY_RAMP4_BUILD_OK"
)
