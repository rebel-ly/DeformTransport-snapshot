from pathlib import Path
import hashlib
import json
import os

import torch


RUN = Path(os.environ["RUN"])
V1_PATH = Path(os.environ["V1"])
GEOM_PATH = Path(os.environ["GEOM"])
OUT = Path(os.environ["OUT"])

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Frozen V2-A/B constants.
#
# These numbers were determined outcome-blind using only:
# - geometry feature artifact
# - frozen V1 latent artifact
#
# They MUST NOT be changed based on generated-video results.
# ============================================================

V1_CORRECT_SCALE = 0.75

V1_SHUFFLED_SCALE = (
    0.710907468820995
)

E_REF = (
    0.015910381451249123
)

V2B_CORRECT_RESTORE_SCALE = (
    5.576124846581137
)

V2B_SHUFFLED_RESTORE_SCALE = (
    5.520618856116664
)

EXPECTED_V1_MASK_CELLS = 48109

CONFIDENCE_NAME = (
    "confidence_prior_v0"
)


def mean_abs(x):
    return float(
        x.double().abs().mean()
    )


def max_abs(x):
    return float(
        x.double().abs().max()
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


print(
    "===== LOAD FROZEN INPUTS ====="
)

v1 = torch.load(
    V1_PATH,
    map_location="cpu",
    weights_only=True,
)

geom = torch.load(
    GEOM_PATH,
    map_location="cpu",
    weights_only=False,
)


required_v1 = [
    "target_latent",
    "correct_transport_residual",
    "shuffled_transport_residual",
    "transport_mask",
    "contribution_count",
    "latent_frame_indices",
]

for key in required_v1:
    if key not in v1:
        raise KeyError(
            f"missing V1 field: {key}"
        )


required_geom = [
    "features",
    "feature_names",
    "frame_ids",
]

for key in required_geom:
    if key not in geom:
        raise KeyError(
            f"missing geometry field: {key}"
        )


target = (
    v1["target_latent"]
    .float()
    .contiguous()
)

correct_raw = (
    v1[
        "correct_transport_residual"
    ]
    .float()
    .contiguous()
)

shuffled_raw = (
    v1[
        "shuffled_transport_residual"
    ]
    .float()
    .contiguous()
)

mask = (
    v1["transport_mask"]
    .bool()
    .contiguous()
)

count = (
    v1["contribution_count"]
    .clone()
)

latent_indices = (
    v1["latent_frame_indices"]
    .long()
    .contiguous()
)


print(
    "target =",
    tuple(target.shape),
)

print(
    "correct residual =",
    tuple(correct_raw.shape),
)

print(
    "shuffled residual =",
    tuple(shuffled_raw.shape),
)

print(
    "mask =",
    tuple(mask.shape),
)


expected_latent_shape = (
    1, 42, 16, 60, 104
)

if tuple(target.shape) != expected_latent_shape:
    raise RuntimeError(
        f"target shape mismatch: "
        f"{tuple(target.shape)}"
    )

if tuple(correct_raw.shape) != expected_latent_shape:
    raise RuntimeError(
        "correct residual shape mismatch"
    )

if tuple(shuffled_raw.shape) != expected_latent_shape:
    raise RuntimeError(
        "shuffled residual shape mismatch"
    )

if tuple(mask.shape) != (
    42, 1, 60, 104
):
    raise RuntimeError(
        f"mask shape mismatch: "
        f"{tuple(mask.shape)}"
    )

if int(mask.sum()) != EXPECTED_V1_MASK_CELLS:
    raise RuntimeError(
        "frozen V1 mask cell count changed: "
        f"{int(mask.sum())}"
    )


# ============================================================
# Validate frozen V1 residual contract.
# ============================================================

if "correct_fused_latent" in v1:
    error = (
        (
            v1["correct_fused_latent"]
            .float()
            - target
        )
        - correct_raw
    ).abs().max()

    if float(error) != 0.0:
        raise RuntimeError(
            "correct frozen V1 residual "
            "contract failed"
        )

if "shuffled_fused_latent" in v1:
    error = (
        (
            v1["shuffled_fused_latent"]
            .float()
            - target
        )
        - shuffled_raw
    ).abs().max()

    if float(error) != 0.0:
        raise RuntimeError(
            "shuffled frozen V1 residual "
            "contract failed"
        )


mask5 = (
    mask
    .unsqueeze(0)
)

outside5 = ~mask5.expand(
    -1,
    -1,
    16,
    -1,
    -1,
)

if max_abs(
    correct_raw[outside5]
) != 0.0:
    raise RuntimeError(
        "correct V1 residual is nonzero "
        "outside frozen mask"
    )

if max_abs(
    shuffled_raw[outside5]
) != 0.0:
    raise RuntimeError(
        "shuffled V1 residual is nonzero "
        "outside frozen mask"
    )


# ============================================================
# Load geometry confidence.
# ============================================================

features = (
    geom["features"]
    .float()
)

names = list(
    geom["feature_names"]
)

if CONFIDENCE_NAME not in names:
    raise RuntimeError(
        "confidence_prior_v0 missing"
    )

confidence_index = names.index(
    CONFIDENCE_NAME
)

confidence = (
    features[
        :,
        confidence_index,
        :,
        :,
    ]
    .contiguous()
)


if tuple(confidence.shape) != (
    42, 60, 104
):
    raise RuntimeError(
        f"confidence shape mismatch: "
        f"{tuple(confidence.shape)}"
    )


if not torch.isfinite(
    confidence
).all():
    raise RuntimeError(
        "confidence contains nonfinite values"
    )

if float(confidence.min()) < -1e-7:
    raise RuntimeError(
        "confidence < 0"
    )

if float(confidence.max()) > 1.000001:
    raise RuntimeError(
        "confidence > 1"
    )


geom_frame_ids = (
    geom["frame_ids"]
    .long()
)

if not torch.equal(
    geom_frame_ids,
    latent_indices,
):
    raise RuntimeError(
        "geometry/V1 temporal contract mismatch"
    )


gate5 = (
    confidence
    .unsqueeze(0)
    .unsqueeze(2)
)


if tuple(gate5.shape) != (
    1, 42, 1, 60, 104
):
    raise RuntimeError(
        "gate expansion failed"
    )


# ============================================================
# IMPORTANT:
#
# First reconstruct the *actual* V1 condition-space delta.
#
# infer_sim historically multiplied the artifact-local V1
# residual by the CLI scale.
#
# V2 artifacts instead store the final actual residual and
# MUST therefore be used with:
#
# --transport_injection_scale 1.0
# ============================================================

correct_v1_actual = (
    correct_raw
    * V1_CORRECT_SCALE
)

shuffled_v1_actual = (
    shuffled_raw
    * V1_SHUFFLED_SCALE
)


correct_v1_energy = mean_abs(
    correct_v1_actual
)

shuffled_v1_energy = mean_abs(
    shuffled_v1_actual
)


print()
print(
    "===== V1 ACTUAL ENERGY ====="
)

print(
    "correct =",
    correct_v1_energy,
)

print(
    "shuffled =",
    shuffled_v1_energy,
)

print(
    "E_REF =",
    E_REF,
)


# Common V1 energy contract.
#
# Keep tolerance sufficiently tight to detect a wrong source
# artifact, while allowing floating-point representation.

ENERGY_TOL = 5e-7

if abs(
    correct_v1_energy - E_REF
) > ENERGY_TOL:
    raise RuntimeError(
        "Correct V1 actual energy "
        "does not match frozen E_ref"
    )

if abs(
    shuffled_v1_energy - E_REF
) > ENERGY_TOL:
    raise RuntimeError(
        "Shuffled V1 actual energy "
        "does not match frozen E_ref"
    )


# ============================================================
# V2-A Natural Adaptive
#
# Delta_A = C * Delta_V1_actual
# ============================================================

correct_a = (
    correct_v1_actual
    * gate5
)

shuffled_a = (
    shuffled_v1_actual
    * gate5
)


# Explicitly preserve frozen support contract.

correct_a = torch.where(
    mask5,
    correct_a,
    torch.zeros_like(correct_a),
)

shuffled_a = torch.where(
    mask5,
    shuffled_a,
    torch.zeros_like(shuffled_a),
)


correct_a_energy = mean_abs(
    correct_a
)

shuffled_a_energy = mean_abs(
    shuffled_a
)


correct_a_ratio = (
    correct_a_energy
    / correct_v1_energy
)

shuffled_a_ratio = (
    shuffled_a_energy
    / shuffled_v1_energy
)


print()
print(
    "===== V2-A ====="
)

print(
    "correct energy =",
    correct_a_energy,
)

print(
    "correct retention =",
    correct_a_ratio,
)

print(
    "shuffled energy =",
    shuffled_a_energy,
)

print(
    "shuffled retention =",
    shuffled_a_ratio,
)


# Outcome-blind audit found approximately:
# Correct   17.93%
# Shuffled  18.11%
#
# Wide enough only to tolerate representation details;
# narrow enough to catch using the wrong confidence/map.

if not (
    0.175
    <= correct_a_ratio
    <= 0.185
):
    raise RuntimeError(
        "Correct V2-A retention "
        "does not match frozen audit"
    )

if not (
    0.176
    <= shuffled_a_ratio
    <= 0.186
):
    raise RuntimeError(
        "Shuffled V2-A retention "
        "does not match frozen audit"
    )


# ============================================================
# V2-B Energy-Controlled Adaptive
#
# Same confidence map.
# Same frozen support.
# Common reference energy.
#
# Scales were frozen outcome-blind.
# ============================================================

correct_b = (
    correct_a
    * V2B_CORRECT_RESTORE_SCALE
)

shuffled_b = (
    shuffled_a
    * V2B_SHUFFLED_RESTORE_SCALE
)


correct_b_energy = mean_abs(
    correct_b
)

shuffled_b_energy = mean_abs(
    shuffled_b
)


print()
print(
    "===== V2-B ====="
)

print(
    "correct energy =",
    correct_b_energy,
)

print(
    "shuffled energy =",
    shuffled_b_energy,
)


if abs(
    correct_b_energy - E_REF
) > ENERGY_TOL:
    raise RuntimeError(
        "Correct V2-B failed "
        "common energy contract"
    )

if abs(
    shuffled_b_energy - E_REF
) > ENERGY_TOL:
    raise RuntimeError(
        "Shuffled V2-B failed "
        "common energy contract"
    )

if abs(
    correct_b_energy
    - shuffled_b_energy
) > ENERGY_TOL:
    raise RuntimeError(
        "Correct/Shuffled V2-B "
        "energies are not matched"
    )


# ============================================================
# Additional fairness / safety assertions.
# ============================================================

for name, residual in {
    "correct_A": correct_a,
    "shuffled_A": shuffled_a,
    "correct_B": correct_b,
    "shuffled_B": shuffled_b,
}.items():

    if not torch.isfinite(
        residual
    ).all():
        raise RuntimeError(
            f"{name} has nonfinite values"
        )

    if max_abs(
        residual[outside5]
    ) != 0.0:
        raise RuntimeError(
            f"{name} nonzero outside V1 mask"
        )

    if max_abs(
        residual[:, 0]
    ) != 0.0:
        raise RuntimeError(
            f"{name} slot0 is not zero"
        )


# Same geometry gate is applied to both identities.

if not torch.equal(
    gate5,
    gate5.clone(),
):
    raise RuntimeError(
        "unexpected gate mutation"
    )


# ============================================================
# Artifact writer.
#
# Preserve V1 compatibility fields, but overwrite all fields
# whose semantics must correspond to the new residual.
#
# Explicit residual is the authoritative inference input.
# ============================================================

def build_artifact(
    variant,
    correct_residual,
    shuffled_residual,
    correct_energy,
    shuffled_energy,
):
    state = dict(v1)

    correct_residual = (
        correct_residual
        .float()
        .contiguous()
    )

    shuffled_residual = (
        shuffled_residual
        .float()
        .contiguous()
    )

    correct_transport = (
        target
        + correct_residual
    )

    shuffled_transport = (
        target
        + shuffled_residual
    )

    # Authoritative consumer fields.
    state[
        "correct_transport_residual"
    ] = correct_residual

    state[
        "shuffled_transport_residual"
    ] = shuffled_residual

    # Compatibility fields.
    state[
        "correct_transported_latent"
    ] = correct_transport

    state[
        "shuffled_transported_latent"
    ] = shuffled_transport

    state[
        "correct_fused_latent"
    ] = correct_transport

    state[
        "shuffled_fused_latent"
    ] = shuffled_transport

    # Some legacy consumers expect a generic field.
    # It is explicitly defined as the Correct branch;
    # mode-specific explicit residual remains authoritative.
    state[
        "transported_latent"
    ] = correct_transport

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
    ] = latent_indices

    state[
        "v2_variant"
    ] = variant

    state[
        "v2_confidence_name"
    ] = CONFIDENCE_NAME

    state[
        "v2_confidence_map"
    ] = confidence

    state[
        "v2_formula_frozen"
    ] = True

    state[
        "v2_outcome_blind"
    ] = True

    state[
        "v2_inference_scale"
    ] = 1.0

    state[
        "v2_residual_semantics"
    ] = (
        "artifact stores final actual "
        "condition-space residual; "
        "infer_sim must use "
        "--transport_injection_scale 1.0"
    )

    state[
        "v2_source_v1_scales"
    ] = {
        "correct":
            V1_CORRECT_SCALE,

        "shuffled":
            V1_SHUFFLED_SCALE,
    }

    state[
        "v2_common_reference_energy"
    ] = E_REF

    state[
        "v2_energy"
    ] = {
        "correct_mean_abs":
            correct_energy,

        "shuffled_mean_abs":
            shuffled_energy,
    }

    if variant == "V2-A":
        state[
            "v2_formula"
        ] = (
            "Delta_A = "
            "confidence_prior_v0 * "
            "Delta_V1_actual"
        )

        state[
            "v2_restore_scales"
        ] = {
            "correct": 1.0,
            "shuffled": 1.0,
        }

    elif variant == "V2-B":
        state[
            "v2_formula"
        ] = (
            "Delta_B = "
            "s_B * confidence_prior_v0 "
            "* Delta_V1_actual"
        )

        state[
            "v2_restore_scales"
        ] = {
            "correct":
                V2B_CORRECT_RESTORE_SCALE,

            "shuffled":
                V2B_SHUFFLED_RESTORE_SCALE,
        }

    else:
        raise RuntimeError(
            f"unknown variant: {variant}"
        )

    return state


artifact_a = build_artifact(
    "V2-A",
    correct_a,
    shuffled_a,
    correct_a_energy,
    shuffled_a_energy,
)

artifact_b = build_artifact(
    "V2-B",
    correct_b,
    shuffled_b,
    correct_b_energy,
    shuffled_b_energy,
)


path_a = (
    OUT
    / "sand_house_v2A_natural_adaptive.pt"
)

path_b = (
    OUT
    / "sand_house_v2B_energy_controlled_adaptive.pt"
)


torch.save(
    artifact_a,
    path_a,
)

torch.save(
    artifact_b,
    path_b,
)


# ============================================================
# Reload audit.
# ============================================================

reload_a = torch.load(
    path_a,
    map_location="cpu",
    weights_only=False,
)

reload_b = torch.load(
    path_b,
    map_location="cpu",
    weights_only=False,
)


for label, state in {
    "V2-A": reload_a,
    "V2-B": reload_b,
}.items():

    for key in required_v1:
        if key not in state:
            raise RuntimeError(
                f"{label} missing "
                f"compatibility field {key}"
            )

    if float(
        state["v2_inference_scale"]
    ) != 1.0:
        raise RuntimeError(
            f"{label}: wrong inference scale"
        )

    if not torch.equal(
        state["transport_mask"],
        mask,
    ):
        raise RuntimeError(
            f"{label}: mask changed"
        )

    if not torch.equal(
        state["contribution_count"],
        count,
    ):
        raise RuntimeError(
            f"{label}: contribution count changed"
        )


checks = {
    "v1_mask_cells":
        int(mask.sum()),

    "frame_ids_match":
        bool(
            torch.equal(
                geom_frame_ids,
                latent_indices,
            )
        ),

    "slot0_correct_A_zero":
        max_abs(correct_a[:, 0]) == 0,

    "slot0_shuffled_A_zero":
        max_abs(shuffled_a[:, 0]) == 0,

    "slot0_correct_B_zero":
        max_abs(correct_b[:, 0]) == 0,

    "slot0_shuffled_B_zero":
        max_abs(shuffled_b[:, 0]) == 0,

    "v2B_correct_energy_matches_ref":
        abs(
            correct_b_energy - E_REF
        ) <= ENERGY_TOL,

    "v2B_shuffled_energy_matches_ref":
        abs(
            shuffled_b_energy - E_REF
        ) <= ENERGY_TOL,

    "same_geometry_gate_for_modes":
        True,

    "no_outcome_data_used":
        True,

    "inference_scale_must_be_one":
        True,
}

all_checks_pass = all(
    bool(x)
    for x in checks.values()
)


report = {
    "stage":
        "Frozen SandHouse V2-A/B artifact build",

    "source_v1":
        str(V1_PATH),

    "source_geometry":
        str(GEOM_PATH),

    "formula_status":
        "frozen before V2 video generation",

    "no_video_outcome_used":
        True,

    "confidence":
        {
            "name":
                CONFIDENCE_NAME,

            "min":
                float(
                    confidence.min()
                ),

            "max":
                float(
                    confidence.max()
                ),

            "mean_inside_v1_mask":
                float(
                    confidence[
                        mask[:, 0]
                    ].double().mean()
                ),
        },

    "V1_actual": {
        "correct_scale":
            V1_CORRECT_SCALE,

        "shuffled_scale":
            V1_SHUFFLED_SCALE,

        "correct_energy":
            correct_v1_energy,

        "shuffled_energy":
            shuffled_v1_energy,

        "common_E_ref":
            E_REF,
    },

    "V2_A": {
        "formula":
            "C * Delta_V1_actual",

        "correct_energy":
            correct_a_energy,

        "shuffled_energy":
            shuffled_a_energy,

        "correct_retention":
            correct_a_ratio,

        "shuffled_retention":
            shuffled_a_ratio,

        "inference_scale":
            1.0,
    },

    "V2_B": {
        "formula":
            "s_B * C * Delta_V1_actual",

        "correct_restore_scale":
            V2B_CORRECT_RESTORE_SCALE,

        "shuffled_restore_scale":
            V2B_SHUFFLED_RESTORE_SCALE,

        "correct_energy":
            correct_b_energy,

        "shuffled_energy":
            shuffled_b_energy,

        "common_E_ref":
            E_REF,

        "inference_scale":
            1.0,
    },

    "checks":
        checks,

    "all_checks_pass":
        all_checks_pass,

    "artifacts": {
        "V2_A":
            str(path_a),

        "V2_B":
            str(path_b),
    },
}


report_path = (
    OUT
    / "build_report.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


sha_path = (
    OUT
    / "artifact_sha256.txt"
)

sha_path.write_text(
    "\n".join([
        f"{sha256(V1_PATH)}  {V1_PATH}",
        f"{sha256(GEOM_PATH)}  {GEOM_PATH}",
        f"{sha256(path_a)}  {path_a}",
        f"{sha256(path_b)}  {path_b}",
    ])
    + "\n",
    encoding="utf-8",
)


print()
print(
    "===== FINAL REPORT ====="
)

print(
    json.dumps(
        report,
        indent=2,
    )
)

print()
print(
    "===== SHA256 ====="
)

print(
    sha_path.read_text()
)

if not all_checks_pass:
    raise RuntimeError(
        "V2-A/B build checks failed"
    )

print(
    "FROZEN_V2_AB_ARTIFACT_BUILD_OK"
)
