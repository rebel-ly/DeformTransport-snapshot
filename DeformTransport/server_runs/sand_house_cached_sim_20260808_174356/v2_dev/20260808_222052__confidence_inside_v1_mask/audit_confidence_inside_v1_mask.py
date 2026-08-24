import json
import math
import sys
from pathlib import Path

import torch


geo_path = Path(sys.argv[1])
v1_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])

report_path = out_dir / "report.json"


def qstats(x):
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q01": None,
            "q05": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "max": None,
        }

    qs = torch.tensor(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        dtype=torch.float32,
    )
    qv = torch.quantile(x, qs)

    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "q01": float(qv[0]),
        "q05": float(qv[1]),
        "q10": float(qv[2]),
        "q25": float(qv[3]),
        "q50": float(qv[4]),
        "q75": float(qv[5]),
        "q90": float(qv[6]),
        "q95": float(qv[7]),
        "q99": float(qv[8]),
        "max": float(x.max()),
    }


def meanabs(x):
    return float(x.detach().float().abs().mean())


def weighted_confidence(delta, gate):
    """
    Sum |delta| * C / Sum |delta|.
    This is the residual-energy-weighted confidence.
    """
    a = delta.detach().float().abs()
    g = gate.detach().float()
    denom = a.sum()
    if float(denom) == 0.0:
        return None
    return float((a * g).sum() / denom)


def safe_ratio(a, b):
    if b == 0:
        return None
    return float(a / b)


print("LOAD_GEOMETRY:", geo_path)
geo = torch.load(geo_path, map_location="cpu", weights_only=False)

print("LOAD_V1:", v1_path)
v1 = torch.load(v1_path, map_location="cpu", weights_only=False)

# ---------------------------------------------------------------------
# 1. Schema / contract validation
# ---------------------------------------------------------------------

assert geo["case"] == "sand_house"
assert v1["case"] == "sand_house"

feature_names = list(geo["feature_names"])
assert "confidence_prior_v0" in feature_names, feature_names
confidence_index = feature_names.index("confidence_prior_v0")

features = geo["features"].float()
confidence = features[:, confidence_index]

assert tuple(features.shape) == (42, 13, 60, 104)
assert tuple(confidence.shape) == (42, 60, 104)

mask4 = v1["transport_mask"].bool()
assert tuple(mask4.shape) == (42, 1, 60, 104)

mask = mask4[:, 0]
assert tuple(mask.shape) == (42, 60, 104)

mask_count = int(mask.sum())
assert mask_count == 48109, (
    f"Expected frozen V1 mask count 48109, got {mask_count}"
)

geo_frame_ids = geo["frame_ids"].long()
v1_frame_ids = v1["latent_frame_indices"].long()

assert torch.equal(
    geo_frame_ids, v1_frame_ids
), (
    f"Geometry/V1 frame IDs differ:\n"
    f"geo={geo_frame_ids.tolist()}\n"
    f"v1={v1_frame_ids.tolist()}"
)

assert torch.isfinite(confidence).all()

cmin = float(confidence.min())
cmax = float(confidence.max())

assert cmin >= -1e-6, cmin
assert cmax <= 1.0 + 1e-6, cmax

# ---------------------------------------------------------------------
# 2. Confirm formal Delta_V1 semantics
# ---------------------------------------------------------------------

target = v1["target_latent"].float()

corr_fused = v1["correct_fused_latent"].float()
shuf_fused = v1["shuffled_fused_latent"].float()

corr_raw = v1["correct_transport_residual"].float()
shuf_raw = v1["shuffled_transport_residual"].float()

expected_shape = (1, 42, 16, 60, 104)

assert tuple(target.shape) == expected_shape
assert tuple(corr_raw.shape) == expected_shape
assert tuple(shuf_raw.shape) == expected_shape

corr_fused_relation_max_abs = float(
    ((corr_fused - target) - corr_raw).abs().max()
)
shuf_fused_relation_max_abs = float(
    ((shuf_fused - target) - shuf_raw).abs().max()
)

# Formal artifact contract should be numerically exact/up to fp tolerance.
assert corr_fuse
assert shuf

# V1 resid
mask_broad
outside_broadcast = 1.0 

corr_outside


shuf_outside_mask_max_abs = 
    (shuf_raw.abs() * outsi
)

assert corr_outside_mask


recipe = dict(v1["frozen_recipe"])


s

# These are t
delta_corr_v1 = corr_raw * correct_con
delta_shuf_v1 = shuf_r

# -------------
# 3. Confidence restric
# -----------------------

confidence_in_mask = co

confidence_stats = qsta

bins = [
    (0.00, 0.0
    (0.05, 0.10),
    (0
    (0.20, 0.40),
    (0
    (0.60, 0.80),
    (0
]

confidence_bins = []

for low, high in bins:
 
        sel = (confidenc
        label = f"[{low:
    else:
        sel = 
        label = f"[{low:

    n = 

    confidence_bins.ap
        {
            "bin": label,
            "count": n,
   
            "mean_confidence
     
                if n > 0
     

        }
  

assert sum(x["count"] for x in 

# -----------------------------
# 4. V2-A Natural Adaptive
#
# Delta_A = C *
#
# Here Delta_V1 is explicitl
# i.e. raw artifact residual
# --------------------------

gate = confidence[None, :, 

delta_corr_a = delta_corr_v
delta_shuf_a = delta_shuf_v1

raw_energy_corr = meanabs(c
raw_energy_shuf = meanabs(sh

v1_energy_corr = meanabs(de
v1_energy_shuf = meanabs(del

natural_energy_corr = meanabs
natur


retention_shuf 

energy_weighted_conf_corr = weighted_confidence(


energy_weighted_conf_shuf = weighted_
    del
)

# These should mathematically m
if retention_corr is not None:
    assert abs(retent

if ret
    assert abs(retention_shuf - ener

# --------------------------
# 5. V2-B strict to
#
# Common E_ref = frozen V
#
# Correct:
#   De
#
# Shuffled:
#   Delta_shuf_B = s_shuf


# -------------------

E_ref = v1_en

assert natural_ene
assert natural_energy_s


s_shuf_b = E_ref / natural_energy

delta_corr_b = delta_corr_a * s_corr_b
delta_shuf_b = delta_shuf_

energy_corr_b = meanabs(de
energy_shuf_b = meanabs(delta_shuf_b)

# composed scale if futur

# Delta_B =
# raw_residual
# * frozen_condition_scale
# * C
# * restora
#
composed_corr_scale = correct_c
composed_shuf_scale = shuffled_condition_scale * s_shuf_b

energy_match


assert energy_match_corr_error < 
assert energy_match_shuf_error < 

# ----------------------------------------
# 6. Per-slot audit
# --------------------------------------

per_slot = []

for t in range(42):
    mt = mask[t]
    ct =

    corr_v1_t = delta_corr_v1[:, t
    shuf_v1_t = delta_shuf_v1[:, t : t + 1

    corr_a_t = delta_corr_a[:, t : t + 1]
    shu

    corr_v1_e = meanabs(corr_v1_t)
    shuf_v1

    corr_a_e = meanabs(corr_a_t)
  

    per_slot.append(
        {
            "

            "v1_ma
            "confidence": qstats(ct),
   
v2a_natural_mean_abs": corr_a_e,
            "correct_energy_retention": safe_ratio(
                corr_a_e, corr_v1_e
            ),
            "shuffled_v1_actual_mean_abs": shuf_v1_e,
            "shuffled_v2a_natural_mean_abs": shuf_a_e,
            "shuffled_energy_retention": safe_ratio(
                shuf_a_e, shuf_v1_e
            ),
        }
    )

# ---------------------------------------------------------------------
# 7. Additional fairness / sanity evidence
# ---------------------------------------------------------------------

alpha_schedule = v1["alpha_schedule"].float()

slot0_corr_max_abs = float(delta_corr_v1[:, 0].abs().max())
slot0_shuf_max_abs = float(delta_shuf_v1[:, 0].abs().max())

assert float(alpha_schedule[0]) == 0.0
assert slot0_corr_max_abs < 1e-7
assert slot0_shuf_max_abs < 1e-7

report = {
    "audit_name": "sand_house_confidence_inside_frozen_v1_mask",
    "case": "sand_house",
    "outcome_blind": True,
    "forbidden_inputs_confirmed_unused": [
        "baseline mp4",
        "correct mp4",
        "shuffled mp4",
        "RGB evaluation metrics",
        "RAFT evaluation metrics",
    ],
    "inputs": {
        "geometry_feature_artifact": str(geo_path),
        "v1_frozen_artifact": str(v1_path),
    },
    "contracts": {
        "geometry_shape": list(features.shape),
        "confidence_feature_index": confidence_index,
        "confidence_feature_name": "confidence_prior_v0",
        "v1_transport_mask_shape": list(mask4.shape),
        "v1_transport_mask_cells": mask_count,
        "latent_frame_indices": v1_frame_ids.tolist(),
        "correct_fused_minus_target_equals_residual_max_abs_error":
            corr_fused_relation_max_abs,
        "shuffled_fused_minus_target_equals_residual_max_abs_error":
            shuf_fused_relation_max_abs,
        "correct_residual_outside_mask_max_abs":
            corr_outside_mask_max_abs,
        "shuffled_residual_outside_mask_max_abs":
            shuf_outside_mask_max_abs,
        "alpha_schedule": alpha_schedule.tolist(),
        "slot0_correct_delta_max_abs": slot0_corr_max_abs,
        "slot0_shuffled_delta_max_abs": slot0_shuf_max_abs,
    },
    "frozen_v1_recipe": recipe,
    "confidence_inside_v1_mask": confidence_stats,
    "confidence_bins": confidence_bins,
    "per_slot": per_slot,
    "energy": {
        "definition": (
            "mean absolute value over full "
            "(1,T,C,H,W) transport residual tensor"
        ),
        "correct_raw_artifact_residual_mean_abs":
            raw_energy_corr,
        "shuffled_raw_artifact_residual_mean_abs":
            raw_energy_shuf,
        "correct_frozen_condition_scale":
            correct_condition_scale,
        "shuffled_frozen_energy_matched_scale":
            shuffled_condition_scale,
        "correct_v1_actual_mean_abs":
            v1_energy_corr,
        "shuffled_v1_actual_mean_abs":
            v1_energy_shuf,
        "common_v2b_E_ref":
            E_ref,
    },
    "v2a_natural_adaptive": {
        "formula": "Delta_A = C * Delta_V1_actual",
        "correct_natural_mean_abs":
            natural_energy_corr,
        "shuffled_natural_mean_abs":
            natural_energy_shuf,
        "correct_energy_retention_ratio":
            retention_corr,
        "shuffled_energy_retention_ratio":
            retention_shuf,
        "correct_residual_energy_weighted_confidence":
            energy_weighted_conf_corr,
        "shuffled_residual_energy_weighted_confidence":
            energy_weighted_conf_shuf,
    },
    "v2b_energy_controlled_adaptive": {
        "formula_correct": (
            "Delta_corr_B = s_corr_B * C * Delta_corr_V1_actual"
        ),
        "formula_shuffled": (
            "Delta_shuf_B = s_shuf_B * C * Delta_shuf_V1_actual"
        ),
        "common_reference_energy":
            E_ref,
        "correct_restore_scale_on_gated_actual_delta":
            s_corr_b,
        "shuffled_restore_scale_on_gated_actual_delta":
            s_shuf_b,
        "correct_composed_scale_if_starting_from_raw_residual":
            composed_corr_scale,
        "shuffled_composed_scale_if_starting_from_raw_residual":
            composed_shuf_scale,
        "correct_final_mean_abs":
            energy_corr_b,
        "shuffled_final_mean_abs":
            energy_shuf_b,
        "correct_energy_match_abs_error":
            energy_match_corr_error,
        "shuffled_energy_match_abs_error":
            energy_match_shuf_error,
    },
    "fairness_definition": {
        "same_geometry_confidence_C_for_correct_and_shuffled": True,
        "same_v1_transport_mask": True,
        "same_common_v2b_reference_energy": True,
        "v2b_correct_and_shuffled_differ_only_in_material_source_identity":
            True,
        "no_video_outcome_used_to_choose_gate_or_scale": True,
    },
}

report_path.write_text(
    json.dumps(report, indent=2, sort_keys=False),
    encoding="utf-8",
)

print("=" * 88)
print("AUDIT_OK")
print("REPORT:", report_path)
print()

print("V1_MASK_CELLS:", mask_count)

print()
print("CONFIDENCE_INSIDE_V1_MASK")
for k, v in confidence_stats.items():
    print(f"  {k}: {v}")

print()
print("CONFIDENCE_BINS")
for row in confidence_bins:
    print(
        f"  {row['bin']:>12} "
        f"count={row['count']:6d} "
        f"fraction={row['fraction_of_v1_cells']:.6f} "
        f"mean={row['mean_confidence_in_bin']}"
    )

print()
print("V1_ENERGY")
print("  correct_raw:", raw_energy_corr)
print("  shuffled_raw:", raw_energy_shuf)
print("  correct_scale:", correct_condition_scale)
print("  shuffled_scale:", shuffled_condition_scale)
print("  correct_actual:", v1_energy_corr)
print("  shuffled_actual:", v1_energy_shuf)
print("  E_ref:", E_ref)

print()
print("V2A_NATURAL")
print("  correct_energy:", natural_energy_corr)
print("  shuffled_energy:", natural_energy_shuf)
print("  correct_retention:", retention_corr)
print("  shuffled_retention:", retention_shuf)
print(
    "  correct_residual_energy_weighted_confidence:",
    energy_weighted_conf_corr,
)
print(
    "  shuffled_residual_energy_weighted_confidence:",
    energy_weighted_conf_shuf,
)

print()
print("V2B_ENERGY_CONTROL")
print("  correct_restore_scale:", s_corr_b)
print("  shuffled_restore_scale:", s_shuf_b)
print("  correct_composed_raw_scale:", composed_corr_scale)
print("  shuffled_composed_raw_scale:", composed_shuf_scale)
print("  correct_final_energy:", energy_corr_b)
print("  shuffled_final_energy:", energy_shuf_b)

print()
print("CONTRACT_CHECKS")
print(
    "  corr_fused-target=residual maxerr:",
    corr_fused_relation_max_abs,
)
print(
    "  shuf_fused-target=residual maxerr:",
    shuf_fused_relation_max_abs,
)
print(
    "  corr_outside_mask_max:",
    corr_outside_mask_max_abs,
)
print(
    "  shuf_outside_mask_max:",
    shuf_outside_mask_max_abs,
)
print("  slot0_correct_max:", slot0_corr_max_abs)
print("  slot0_shuffled_max:", slot0_shuf_max_abs)

print()
print("PER_SLOT_COMPACT")
print(
    "slot frame cells conf_mean conf_q50 conf_q75 conf_q90 "
    "corr_ret shuf_ret"
)

for r in per_slot:
    c = r["confidence"]

    def fmt(x):
        if x is None:
            return "NA"
        return f"{x:.6f}"

    print(
        f"{r['slot']:02d} "
        f"{r['pixel_frame_id']:03d} "
        f"{r['v1_mask_cells']:5d} "
        f"{fmt(c['mean'])} "
        f"{fmt(c['q50'])} "
        f"{fmt(c['q75'])} "
        f"{fmt(c['q90'])} "
        f"{fmt(r['correct_energy_retention'])} "
        f"{fmt(r['shuffled_energy_retention'])}"
    )

print()
print("OUTCOME_BLIND_AUDIT_COMPLETE")
