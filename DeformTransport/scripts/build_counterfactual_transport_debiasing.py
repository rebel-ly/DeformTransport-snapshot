from pathlib import Path
import hashlib
import json
import os

import torch


V1 = Path(os.environ["V1"])
OUT = Path(os.environ["OUT"])
OUT.mkdir(parents=True, exist_ok=True)


# Frozen historical actual V1 injection scales.
CORRECT_SCALE = 0.75
SHUFFLED_SCALE = 0.710907468820995

EXPECTED_MASK_CELLS = 48109


def mean_abs(x):
    return float(
        x.double().abs().mean()
    )


def l2(x):
    return float(
        torch.linalg.vector_norm(
            x.double()
        )
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


print("===== LOAD FROZEN V1 =====")

art = torch.load(
    V1,
    map_location="cpu",
    weights_only=True,
)

target = art["target_latent"].float()

correct_raw = (
    art["correct_transport_residual"]
    .float()
)

shuffled_raw = (
    art["shuffled_transport_residual"]
    .float()
)

mask = (
    art["transport_mask"]
    .bool()
)

count = art["contribution_count"].clone()

indices = (
    art["latent_frame_indices"]
    .long()
)


if int(mask.sum()) != EXPECTED_MASK_CELLS:
    raise RuntimeError(
        f"mask cells changed: {int(mask.sum())}"
    )


correct = (
    correct_raw
    * CORRECT_SCALE
)

shuffled = (
    shuffled_raw
    * SHUFFLED_SCALE
)


mask5 = mask.unsqueeze(0)

outside = ~mask5.expand(
    -1, -1, 16, -1, -1
)


if float(
    correct[outside].abs().max()
) != 0.0:
    raise RuntimeError(
        "Correct nonzero outside mask"
    )

if float(
    shuffled[outside].abs().max()
) != 0.0:
    raise RuntimeError(
        "Shuffled nonzero outside mask"
    )


print(
    "Correct actual mean_abs =",
    mean_abs(correct),
)

print(
    "Shuffled actual mean_abs =",
    mean_abs(shuffled),
)


# ============================================================
# CTD-DIFF
#
# Negative-control subtraction.
# No manually chosen coefficient.
# ============================================================

delta_diff = (
    correct - shuffled
)


# ============================================================
# CTD-ORTH
#
# Remove the global shuffled nuisance direction from
# Correct using a closed-form least-squares projection.
#
# beta = <C,S> / <S,S>
# ============================================================

c64 = correct.double().reshape(-1)
s64 = shuffled.double().reshape(-1)

denom = torch.dot(
    s64,
    s64,
)

if float(denom) <= 0:
    raise RuntimeError(
        "zero shuffled residual energy"
    )

beta = float(
    torch.dot(
        c64,
        s64,
    )
    / denom
)

delta_orth = (
    correct
    - beta * shuffled
)


# Verify orthogonality numerically.

orth_dot = float(
    torch.dot(
        delta_orth.double().reshape(-1),
        s64,
    )
)

orth_norm_product = (
    torch.linalg.vector_norm(
        delta_orth.double().reshape(-1)
    )
    *
    torch.linalg.vector_norm(s64)
)

relative_orthogonality = float(
    abs(orth_dot)
    /
    max(
        float(orth_norm_product),
        1e-30,
    )
)


for name, x in {
    "DIFF": delta_diff,
    "ORTH": delta_orth,
}.items():

    if not torch.isfinite(x).all():
        raise RuntimeError(
            f"{name}: nonfinite"
        )

    if float(
        x[outside].abs().max()
    ) != 0.0:
        raise RuntimeError(
            f"{name}: nonzero outside mask"
        )

    if float(
        x[:, 0].abs().max()
    ) != 0.0:
        raise RuntimeError(
            f"{name}: slot0 not zero"
        )


print()
print("===== COUNTERFACTUAL DIAGNOSTICS =====")

print(
    "beta =",
    beta,
)

print(
    "DIFF mean_abs =",
    mean_abs(delta_diff),
)

print(
    "ORTH mean_abs =",
    mean_abs(delta_orth),
)

print(
    "relative orthogonality =",
    relative_orthogonality,
)


# Correlation / cosine similarity.

cosine = float(
    torch.dot(c64, s64)
    /
    (
        torch.linalg.vector_norm(c64)
        * torch.linalg.vector_norm(s64)
    ).clamp_min(1e-30)
)

print(
    "Correct/Shuffled cosine =",
    cosine,
)


def make_artifact(
    name,
    residual,
    formula,
):
    state = dict(art)

    residual = (
        residual
        .float()
        .contiguous()
    )

    transported = (
        target + residual
    )

    # We deliberately expose the same residual for both modes.
    # This artifact represents one final debiased method, not
    # another Correct-vs-Shuffled comparison.

    state[
        "correct_transport_residual"
    ] = residual

    state[
        "shuffled_transport_residual"
    ] = residual

    state[
        "correct_transported_latent"
    ] = transported

    state[
        "shuffled_transported_latent"
    ] = transported

    state[
        "correct_fused_latent"
    ] = transported

    state[
        "shuffled_fused_latent"
    ] = transported

    state[
        "transported_latent"
    ] = transported

    state[
        "target_latent"
    ] = target

    state[
        "transport_mask"
    ] = mask

    state[
        "contribution_count"
    ] = count

    state[
        "latent_frame_indices"
    ] = indices

    state[
        "ctd_method"
    ] = name

    state[
        "ctd_formula"
    ] = formula

    state[
        "ctd_parameter_free"
    ] = True

    state[
        "ctd_outcome_blind_formula"
    ] = True

    state[
        "ctd_inference_scale"
    ] = 1.0

    state[
        "ctd_correct_source_scale"
    ] = CORRECT_SCALE

    state[
        "ctd_shuffled_source_scale"
    ] = SHUFFLED_SCALE

    state[
        "ctd_projection_beta"
    ] = beta

    state[
        "ctd_residual_mean_abs"
    ] = mean_abs(residual)

    return state


diff_art = make_artifact(
    "CTD-DIFF",
    delta_diff,
    "Delta = Delta_correct_actual - Delta_shuffled_actual",
)

orth_art = make_artifact(
    "CTD-ORTH",
    delta_orth,
    "Delta = Delta_correct_actual - "
    "beta * Delta_shuffled_actual; "
    "beta=<C,S>/<S,S>",
)


PATH_DIFF = (
    OUT
    / "sand_house_ctd_diff.pt"
)

PATH_ORTH = (
    OUT
    / "sand_house_ctd_orth.pt"
)

torch.save(
    diff_art,
    PATH_DIFF,
)

torch.save(
    orth_art,
    PATH_ORTH,
)


report = {
    "method":
        "Counterfactual Transport Debiasing",

    "source":
        str(V1),

    "formula_frozen_before_generation":
        True,

    "manual_hyperparameters":
        0,

    "correct_actual_mean_abs":
        mean_abs(correct),

    "shuffled_actual_mean_abs":
        mean_abs(shuffled),

    "correct_shuffled_cosine":
        cosine,

    "projection_beta":
        beta,

    "diff_mean_abs":
        mean_abs(delta_diff),

    "orth_mean_abs":
        mean_abs(delta_orth),

    "orth_relative_dot":
        relative_orthogonality,

    "mask_cells":
        int(mask.sum()),

    "slot0_zero":
        True,

    "inference_scale":
        1.0,

    "artifacts": {
        "CTD_DIFF":
            str(PATH_DIFF),

        "CTD_ORTH":
            str(PATH_ORTH),
    },
}


REPORT = (
    OUT / "report.json"
)

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


SHA = OUT / "sha256.txt"

SHA.write_text(
    "\n".join([
        f"{sha256(V1)}  {V1}",
        f"{sha256(PATH_DIFF)}  {PATH_DIFF}",
        f"{sha256(PATH_ORTH)}  {PATH_ORTH}",
    ])
    + "\n",
    encoding="utf-8",
)


print()
print("===== REPORT =====")

print(
    json.dumps(
        report,
        indent=2,
    )
)

print()
print("===== SHA256 =====")

print(
    SHA.read_text()
)

print(
    "COUNTERFACTUAL_TRANSPORT_DEBIASING_BUILD_OK"
)
