from pathlib import Path
import hashlib
import json
import os

import torch


soft_path = Path(os.environ["SOFT"])
out_dir = Path(os.environ["OUT"])

artifact_path = (
    out_dir
    / "sand_house_quality075_ramp4_full_generation.pt"
)

report_path = (
    out_dir
    / "build_report.json"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


state = torch.load(
    soft_path,
    map_location="cpu",
    weights_only=True,
)

required = {
    "target_latent",
    "correct_transported_latent",
    "shuffled_transported_latent",
    "hard_reference_mask",
    "soft_only_mask",
    "raw_soft_contribution_count",
    "transport_weight",
    "transport_mask",
    "latent_frame_indices",
}

missing = sorted(required - set(state))

if missing:
    raise ValueError(
        f"missing soft artifact keys: {missing}"
    )


target = state[
    "target_latent"
].float().contiguous()

raw_correct = state[
    "correct_transported_latent"
].float().contiguous()

raw_shuffled = state[
    "shuffled_transported_latent"
].float().contiguous()

hard_mask = state[
    "hard_reference_mask"
].bool().contiguous()

soft_only = state[
    "soft_only_mask"
].bool().contiguous()

raw_soft_count = state[
    "raw_soft_contribution_count"
].long().contiguous()

transport_weight = state[
    "transport_weight"
].float().contiguous()

gate025_mask = state[
    "transport_mask"
].bool().contiguous()

latent_indices = state[
    "latent_frame_indices"
].long().contiguous()


assert tuple(target.shape) == (
    1, 42, 16, 60, 104
)

assert raw_correct.shape == target.shape
assert raw_shuffled.shape == target.shape

assert tuple(hard_mask.shape) == (
    42, 1, 60, 104
)

assert tuple(soft_only.shape) == (
    42, 1, 60, 104
)

assert tuple(transport_weight.shape) == (
    42, 1, 60, 104
)

assert latent_indices.tolist() == list(
    range(0, 165, 4)
)


# --------------------------------------------------
# Frozen Tree->Santa->SandHouse recipe.
# --------------------------------------------------

quality_tau = 0.75
alpha = 0.25

quality_mask = (
    hard_mask
    | (
        soft_only
        & (
            transport_weight
            >= quality_tau
        )
    )
).contiguous()


# tau=.75 must remain a subset of Gate=.25.
assert not bool(
    (
        quality_mask
        & ~gate025_mask
    ).any()
)


quality_count = torch.where(
    quality_mask,
    raw_soft_count,
    torch.zeros_like(
        raw_soft_count
    ),
).contiguous()

assert torch.equal(
    quality_mask,
    quality_count > 0,
)


# --------------------------------------------------
# Dynamic Ramp4.
#
# slot0 = 0
# slot1 = alpha/3
# slot2 = 2alpha/3
# slot3 onward = alpha
# --------------------------------------------------

slots = target.shape[1]

schedule = torch.full(
    (slots,),
    alpha,
    dtype=torch.float32,
)

schedule[0] = 0.0

if slots > 1:
    schedule[1] = alpha / 3.0

if slots > 2:
    schedule[2] = (
        2.0 * alpha / 3.0
    )

schedule_5d = schedule.view(
    1,
    slots,
    1,
    1,
    1,
)


mask_5d = quality_mask.unsqueeze(0)


correct_fused = torch.where(
    mask_5d,
    target
    + schedule_5d
    * (
        raw_correct
        - target
    ),
    target,
).contiguous()


shuffled_fused = torch.where(
    mask_5d,
    target
    + schedule_5d
    * (
        raw_shuffled
        - target
    ),
    target,
).contiguous()


correct_residual = (
    correct_fused
    - target
).contiguous()

shuffled_residual = (
    shuffled_fused
    - target
).contiguous()


# Explicitly zero outside quality mask.
correct_residual = torch.where(
    mask_5d,
    correct_residual,
    torch.zeros_like(
        correct_residual
    ),
).contiguous()

shuffled_residual = torch.where(
    mask_5d,
    shuffled_residual,
    torch.zeros_like(
        shuffled_residual
    ),
).contiguous()


assert float(
    correct_residual[:, 0]
    .abs()
    .max()
) == 0.0

assert float(
    shuffled_residual[:, 0]
    .abs()
    .max()
) == 0.0


correct_mean_abs = float(
    correct_residual.abs().mean()
)

shuffled_mean_abs = float(
    shuffled_residual.abs().mean()
)

if shuffled_mean_abs <= 0:
    raise RuntimeError(
        "shuffled residual energy is zero"
    )


# Frozen cross-case Correct global scale.
correct_scale = 0.75

# Energy matching uses artifact residual energy only.
shuffled_scale = (
    correct_scale
    * correct_mean_abs
    / shuffled_mean_abs
)


correct_scaled_energy = (
    correct_scale
    * correct_mean_abs
)

shuffled_scaled_energy = (
    shuffled_scale
    * shuffled_mean_abs
)


assert abs(
    correct_scaled_energy
    - shuffled_scaled_energy
) < 1e-10


output = {
    "format_version": 4,

    "artifact_kind":
        "deformtransport_frozen_quality_ramp4",

    "case":
        "sand_house",

    "frozen_recipe": {
        "soft_gate": 0.25,
        "quality_tau": 0.75,
        "alpha": 0.25,
        "schedule": "ramp4",
        "correct_condition_scale": (
            correct_scale
        ),
        "shuffled_energy_matched_scale": (
            shuffled_scale
        ),
    },

    "latent_frame_indices":
        latent_indices,

    "target_latent":
        target,

    # Loader compatibility.
    "correct_transported_latent":
        correct_fused,

    "shuffled_transported_latent":
        shuffled_fused,

    "correct_fused_latent":
        correct_fused,

    "shuffled_fused_latent":
        shuffled_fused,

    # Preferred explicit residual path.
    "correct_transport_residual":
        correct_residual,

    "shuffled_transport_residual":
        shuffled_residual,

    "transport_mask":
        quality_mask,

    "contribution_count":
        quality_count,

    "alpha_schedule":
        schedule,

    "transport_weight":
        transport_weight,

    "source_soft_artifact":
        str(soft_path),
}


for name, tensor in (
    ("correct_fused", correct_fused),
    ("shuffled_fused", shuffled_fused),
    ("correct_residual", correct_residual),
    ("shuffled_residual", shuffled_residual),
):
    if not bool(
        torch.isfinite(
            tensor
        ).all()
    ):
        raise RuntimeError(
            f"{name} contains NaN/Inf"
        )


torch.save(
    output,
    artifact_path,
)


loaded = torch.load(
    artifact_path,
    map_location="cpu",
    weights_only=True,
)

assert tuple(
    loaded[
        "correct_transport_residual"
    ].shape
) == (
    1, 42, 16, 60, 104
)

assert torch.equal(
    loaded["transport_mask"],
    loaded[
        "contribution_count"
    ] > 0,
)


report = {
    "status":
        "SANDHOUSE_FROZEN_QUALITY_RAMP4_READY",

    "source_soft_artifact":
        str(soft_path),

    "source_soft_sha256":
        sha256(soft_path),

    "artifact":
        str(artifact_path),

    "artifact_sha256":
        sha256(artifact_path),

    "shape":
        list(target.shape),

    "latent_frame_indices":
        latent_indices.tolist(),

    "recipe": {
        "soft_gate": 0.25,
        "quality_tau": 0.75,
        "alpha": 0.25,
        "schedule":
            schedule.tolist(),
    },

    "support": {
        "gate025_cells":
            int(
                gate025_mask.sum()
            ),

        "hard_cells":
            int(
                hard_mask.sum()
            ),

        "quality_cells":
            int(
                quality_mask.sum()
            ),

        "removed_from_gate025":
            int(
                (
                    gate025_mask
                    & ~quality_mask
                ).sum()
            ),
    },

    "residual": {
        "correct_mean_abs":
            correct_mean_abs,

        "shuffled_mean_abs":
            shuffled_mean_abs,

        "correct_vs_shuffled_mean_abs":
            float(
                (
                    correct_residual
                    - shuffled_residual
                )
                .abs()
                .mean()
            ),
    },

    "full_generation_scales": {
        "correct":
            correct_scale,

        "shuffled_energy_matched":
            shuffled_scale,

        "correct_scaled_mean_abs":
            correct_scaled_energy,

        "shuffled_scaled_mean_abs":
            shuffled_scaled_energy,
    },

    "checks": {
        "quality_subset_gate025":
            not bool(
                (
                    quality_mask
                    & ~gate025_mask
                ).any()
            ),

        "mask_count_contract":
            bool(
                torch.equal(
                    quality_mask,
                    quality_count > 0,
                )
            ),

        "correct_slot0_zero":
            float(
                correct_residual[:, 0]
                .abs()
                .max()
            ) == 0.0,

        "shuffled_slot0_zero":
            float(
                shuffled_residual[:, 0]
                .abs()
                .max()
            ) == 0.0,

        "energy_match_exact":
            abs(
                correct_scaled_energy
                - shuffled_scaled_energy
            ) < 1e-10,
    },
}

report[
    "all_checks_pass"
] = all(
    report[
        "checks"
    ].values()
)


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

print(
    "\nSANDHOUSE_FROZEN_QUALITY_RAMP4_OK"
)
