import json
import sys
from pathlib import Path
import torch

geo_path = Path(sys.argv[1])
v1_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])

def meanabs(x):
    return float(x.float().abs().mean())

def stats(x):
    x = x.float().reshape(-1)
    if x.numel() == 0:
        return {"count": 0}
    qs = torch.tensor(
        [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
    )
    q = torch.quantile(x, qs)
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q10": float(q[2]),
        "q25": float(q[3]),
        "q50": float(q[4]),
        "q75": float(q[5]),
        "q90": float(q[6]),
        "q95": float(q[7]),
        "q99": float(q[8]),
        "max": float(x.max()),
    }

geo = torch.load(geo_path, map_location="cpu", weights_only=False)
v1 = torch.load(v1_path, map_location="cpu", weights_only=False)

assert geo["case"] == "sand_house"
assert v1["case"] == "sand_house"

names = list(geo["feature_names"])
ci = names.index("confidence_prior_v0")

features = geo["features"].float()
C = features[:, ci]

mask = v1["transport_mask"][:,0].bool()

assert tuple(C.shape) == (42,60,104)
assert tuple(mask.shape) == (42,60,104)
assert int(mask.sum()) == 48109
assert torch.equal(
    geo["frame_ids"].long(),
    v1["latent_frame_indices"].long()
)
assert torch.isfinite(C).all()
assert float(C.min()) >= -1e-6
assert float(C.max()) <= 1.000001

target = v1["target_latent"].float()
corr_raw = v1["correct_transport_residual"].float()
shuf_raw = v1["shuffled_transport_residual"].float()

corr_fused = v1["correct_fused_latent"].float()
shuf_fused = v1["shuffled_fused_latent"].float()

shape = (1,42,16,60,104)
assert tuple(target.shape) == shape
assert tuple(corr_raw.shape) == shape
assert tuple(shuf_raw.shape) == shape

corr_relation_err = float(
    ((corr_fused - target) - corr_raw).abs().max()
)
shuf_relation_err = float(
    ((shuf_fused - target) - shuf_raw).abs().max()
)

assert corr_relation_err < 1e-5
assert shuf_relation_err < 1e-5

M = mask[None,:,None,:,:].float()

corr_outside = float(
    (corr_raw.abs() * (1.0-M)).max()
)
shuf_outside = float(
    (shuf_raw.abs() * (1.0-M)).max()
)

assert corr_outside < 1e-6
assert shuf_outside < 1e-6

recipe = v1["frozen_recipe"]
corr_scale = float(recipe["correct_condition_scale"])
shuf_scale = float(recipe["shuffled_energy_matched_scale"])

corr_v1 = corr_raw * corr_scale
shuf_v1 = shuf_raw * shuf_scale

E_corr_v1 = meanabs(corr_v1)
E_shuf_v1 = meanabs(shuf_v1)

gate = C[None,:,None,:,:] * M

corr_A = corr_v1 * gate
shuf_A = shuf_v1 * gate

E_corr_A = meanabs(corr_A)
E_shuf_A = meanabs(shuf_A)

ret_corr = E_corr_A / E_corr_v1
ret_shuf = E_shuf_A / E_shuf_v1

wc_corr = float(
    (corr_v1.abs() * gate).sum() / corr_v1.abs().sum()
)
wc_shuf = float(
    (shuf_v1.abs() * gate).sum() / shuf_v1.abs().sum()
)

assert abs(ret_corr - wc_corr) < 1e-6
assert abs(ret_shuf - wc_shuf) < 1e-6

E_ref = E_corr_v1

s_corr_B = E_ref / E_corr_A
s_shuf_B = E_ref / E_shuf_A

corr_B = corr_A * s_corr_B
shuf_B = shuf_A * s_shuf_B

E_corr_B = meanabs(corr_B)
E_shuf_B = meanabs(shuf_B)

assert abs(E_corr_B - E_ref) < 1e-6
assert abs(E_shuf_B - E_ref) < 1e-6

cin = C[mask]

edges = [
    (0.00,0.05),
    (0.05,0.10),
    (0.10,0.20),
    (0.20,0.40),
    (0.40,0.60),
    (0.60,0.80),
    (0.80,1.01),
]

bins = []
for lo, hi in edges:
    sel = (cin >= lo) & (cin < hi)
    n = int(sel.sum())
    bins.append({
        "low": lo,
        "high": 1.0 if hi > 1 else hi,
        "count": n,
        "fraction": n / int(mask.sum()),
        "mean": float(cin[sel].mean()) if n else None,
    })

assert sum(x["count"] for x in bins) == 48109

per_slot = []
frame_ids = v1["latent_frame_indices"].long()

for t in range(42):
    mt = mask[t]
    ct = C[t][mt]

    ev1c = meanabs(corr_v1[:,t:t+1])
    ev1s = meanabs(shuf_v1[:,t:t+1])
    eac = meanabs(corr_A[:,t:t+1])
    eas = meanabs(shuf_A[:,t:t+1])

    per_slot.append({
        "slot": t,
        "pixel_frame": int(frame_ids[t]),
        "cells": int(mt.sum()),
        "confidence": stats(ct),
        "correct_retention": eac / ev1c if ev1c else None,
        "shuffled_retention": eas / ev1s if ev1s else None,
    })

alpha = v1["alpha_schedule"].float()
assert float(alpha[0]) == 0.0
assert float(corr_v1[:,0].abs().max()) < 1e-7
assert float(shuf_v1[:,0].abs().max()) < 1e-7

report = {
    "case": "sand_house",
    "outcome_blind": True,
    "v1_mask_cells": 48109,
    "confidence_inside_v1_mask": stats(cin),
    "confidence_bins": bins,
    "per_slot": per_slot,
    "v1_contract": {
        "correct_fused_minus_target_residual_maxerr":
            corr_relation_err,
        "shuffled_fused_minus_target_residual_maxerr":
            shuf_relation_err,
        "correct_outside_mask_maxabs": corr_outside,
        "shuffled_outside_mask_maxabs": shuf_outside,
    },
    "energy": {
        "correct_raw_meanabs": meanabs(corr_raw),
        "shuffled_raw_meanabs": meanabs(shuf_raw),
        "correct_v1_scale": corr_scale,
        "shuffled_v1_scale": shuf_scale,
        "correct_v1_actual_meanabs": E_corr_v1,
        "shuffled_v1_actual_meanabs": E_shuf_v1,
        "E_ref": E_ref,
    },
    "v2a": {
        "formula": "Delta_A = C * Delta_V1_actual",
        "correct_meanabs": E_corr_A,
        "shuffled_meanabs": E_shuf_A,
        "correct_retention": ret_corr,
        "shuffled_retention": ret_shuf,
        "correct_energy_weighted_confidence": wc_corr,
        "shuffled_energy_weighted_confidence": wc_shuf,
    },
    "v2b": {
        "formula": "Delta_B = s_B * C * Delta_V1_actual",
        "correct_restore_scale": s_corr_B,
        "shuffled_restore_scale": s_shuf_B,
        "correct_composed_raw_scale":
            corr_scale * s_corr_B,
        "shuffled_composed_raw_scale":
            shuf_scale * s_shuf_B,
        "correct_final_meanabs": E_corr_B,
        "shuffled_final_meanabs": E_shuf_B,
    },
    "fairness": {
        "same_C": True,
        "same_transport_mask": True,
        "same_E_ref": True,
        "no_video_metrics_used": True,
        "identity_is_only_correct_vs_shuffled_difference": True,
    },
}

(out_dir / "report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("AUDIT_OK")
print("V1_MASK_CELLS", int(mask.sum()))

print("\nCONFIDENCE")
for k,v in report["confidence_inside_v1_mask"].items():
    print(k, v)

print("\nBINS")
for b in bins:
    print(b)

print("\nENERGY")
for k,v in report["energy"].items():
    print(k, v)

print("\nV2A")
for k,v in report["v2a"].items():
    print(k, v)

print("\nV2B")
for k,v in report["v2b"].items():
    print(k, v)

print("\nCONTRACT")
for k,v in report["v1_contract"].items():
    print(k, v)

print("\nPER_SLOT")
print("slot frame cells conf_mean q50 q75 q90 corr_ret shuf_ret")
for r in per_slot:
    s = r["confidence"]
    def f(x):
        return "NA" if x is None else f"{x:.6f}"
    print(
        f"{r['slot']:02d} "
        f"{r['pixel_frame']:03d} "
        f"{r['cells']:5d} "
        f"{f(s.get('mean'))} "
        f"{f(s.get('q50'))} "
        f"{f(s.get('q75'))} "
        f"{f(s.get('q90'))} "
        f"{f(r['correct_retention'])} "
        f"{f(r['shuffled_retention'])}"
    )

print("\nOUTCOME_BLIND_AUDIT_COMPLETE")
