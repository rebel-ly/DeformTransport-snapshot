import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/DeformTransport")
import infer_sim

DT = Path("/workspace/DeformTransport")
ROOT = Path.cwd()

CASES = {
    "santa": {
        "A0": DT / (
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/"
            "official_santa_81f_aligned_final_sim_20260806_234410/"
            "noises.npy"
        ),
    },
    "tree": {
        "A0": DT / (
            "server_runs/20260804_234925_autonomous_deformtransport/"
            "prepared_inputs/"
            "tree_official_precomputed_aligned_final_sim_20260807_185055/"
            "noises.npy"
        ),
    },
}

for case in CASES:
    CASES[case]["A1"] = (
        ROOT / "variants" / case /
        "A1_full_persistent_noises.npy"
    )
    CASES[case]["A2"] = (
        ROOT / "variants" / case /
        "A2_block3_reanchored_noises.npy"
    )

ANCHORS = np.arange(0, 81, 4)

TOLS = {
    "mean": 0.02,
    "variance": 0.05,
    "skew": 0.10,
    "excess_kurtosis": 0.20,
    "spatial_lag1": 0.02,
    "channel_offdiag_abs": 0.02,
}

RHO_P95_TOL = 0.10
RHO_MAX_TOL = 0.15


def load_runtime(path):
    # Important:
    # reset seed before EVERY A0/A1/A2 call so the degradation
    # Gaussian is exactly shared across variants.
    torch.manual_seed(0)

    out = infer_sim.load_noise(
        noise_path=str(path),
        target_frames=21,
        channel_dim=16,
        downsample_mode="nearest",
        eval_degradation=0.5,
    )

    q1 = out["structured_noise"].float()
    q2 = out["structured_noise_sde"].float()

    q = torch.cat([q1, q2], dim=1)

    assert tuple(q.shape) == (21, 32, 60, 104)

    return q.cpu().numpy().astype(np.float64)


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    a = a - a.mean()
    b = b - b.mean()

    den = np.sqrt(
        np.dot(a, a) *
        np.dot(b, b)
    )

    if den <= 1e-20:
        return float("nan")

    return float(np.dot(a, b) / den)


def stats_one_frame(x):
    # x: [C,H,W]
    flat = x.reshape(-1)

    mean = float(flat.mean())
    var = float(flat.var())

    std = max(np.sqrt(var), 1e-12)
    z = (flat - mean) / std

    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)

    # Horizontal + vertical spatial lag-1 correlation.
    h1 = x[:, :, :-1].reshape(-1)
    h2 = x[:, :, 1:].reshape(-1)

    v1 = x[:, :-1, :].reshape(-1)
    v2 = x[:, 1:, :].reshape(-1)

    spatial = corr(
        np.concatenate([h1, v1]),
        np.concatenate([h2, v2]),
    )

    # Channel correlation.
    c = x.reshape(x.shape[0], -1)
    cc = np.corrcoef(c)

    off = cc[
        ~np.eye(cc.shape[0], dtype=bool)
    ]

    channel_offdiag_abs = float(
        np.mean(np.abs(off))
    )

    return {
        "mean": mean,
        "variance": var,
        "skew": skew,
        "excess_kurtosis": kurt,
        "spatial_lag1": spatial,
        "channel_offdiag_abs": channel_offdiag_abs,
    }


def stats_all(q):
    return [
        stats_one_frame(q[t])
        for t in range(q.shape[0])
    ]


def summarize_deltas(base, var):
    out = {}

    for key, tol in TOLS.items():
        d = np.array([
            abs(var[t][key] - base[t][key])
            for t in range(len(base))
        ])

        p95 = float(np.percentile(d, 95))
        mx = float(d.max())

        out[key] = {
            "p95_abs_delta": p95,
            "max_abs_delta": mx,
            "tolerance": tol,
            "max_allowed": 2.0 * tol,
            "pass": bool(
                p95 <= tol
                and mx <= 2.0 * tol
            ),
        }

    out["all_pass"] = bool(
        all(x["pass"] for x in out.values()
            if isinstance(x, dict))
    )

    return out


def intervention_mask(a0_path, v_path):
    a0 = np.load(a0_path, mmap_mode="r")
    v = np.load(v_path, mmap_mode="r")

    # Only the 21 frames actually read by RealWonder.
    a0 = np.asarray(a0[ANCHORS])
    v = np.asarray(v[ANCHORS])

    return np.any(v != a0, axis=-1)


def temporal_corr(q, mask=None):
    rho0 = []
    rho1 = []

    for t in range(q.shape[0]):
        if mask is None:
            rho0.append(
                corr(q[t], q[0])
            )
        else:
            m = mask[t]

            if not m.any():
                rho0.append(None)
            else:
                rho0.append(
                    corr(
                        q[t][:, m],
                        q[0][:, m],
                    )
                )

        if t == 0:
            rho1.append(None)
            continue

        if mask is None:
            rho1.append(
                corr(q[t], q[t-1])
            )
        else:
            # Use intervention support from either neighboring frame.
            m = mask[t] | mask[t-1]

            if not m.any():
                rho1.append(None)
            else:
                rho1.append(
                    corr(
                        q[t][:, m],
                        q[t-1][:, m],
                    )
                )

    return {
        "rho0": rho0,
        "rho1": rho1,
    }


def rho_delta_gate(base_q, var_q, mask):
    # Same support mask for both A0 and variant.
    b = temporal_corr(base_q, mask)
    v = temporal_corr(var_q, mask)

    ds = []

    rows = []

    for t in range(1, 21):
        rb = b["rho1"][t]
        rv = v["rho1"][t]

        if rb is None or rv is None:
            continue

        d = abs(rv - rb)
        ds.append(d)

        rows.append({
            "latent_index": t,
            "pixel_frame": int(ANCHORS[t]),
            "A0_rho1": rb,
            "variant_rho1": rv,
            "abs_delta": d,
        })

    if not ds:
        return {
            "pass": False,
            "reason": "no valid support transitions",
            "rows": rows,
        }

    ds = np.asarray(ds)

    p95 = float(np.percentile(ds, 95))
    mx = float(ds.max())

    return {
        "p95_abs_delta": p95,
        "max_abs_delta": mx,
        "p95_tolerance": RHO_P95_TOL,
        "max_tolerance": RHO_MAX_TOL,
        "pass": bool(
            p95 <= RHO_P95_TOL
            and mx <= RHO_MAX_TOL
        ),
        "rows": rows,
    }


report = {
    "protocol": {
        "seed": 0,
        "eval_degradation": 0.5,
        "target_frames": 21,
        "channels": 32,
        "same_degradation_rng_for_all_variants": True,
        "legality_gate": {
            "p95_delta_le_tolerance": True,
            "max_delta_le_2x_tolerance": True,
        },
        "rho1_support_gate": {
            "p95_abs_A_variant_minus_A0": RHO_P95_TOL,
            "max_abs_A_variant_minus_A0": RHO_MAX_TOL,
        },
    },
    "cases": {},
}

for case, paths in CASES.items():

    print("\n" + "=" * 78)
    print(case.upper())
    print("=" * 78)

    q = {
        name: load_runtime(path)
        for name, path in paths.items()
    }

    # Hard RNG/identity gates.
    frame0_a1 = np.array_equal(
        q["A0"][0],
        q["A1"][0],
    )

    frame0_a2 = np.array_equal(
        q["A0"][0],
        q["A2"][0],
    )

    block_anchors = [
        0, 3, 6, 9, 12, 15, 18
    ]

    a2_anchor_exact = all(
        np.array_equal(
            q["A0"][t],
            q["A2"][t],
        )
        for t in block_anchors
    )

    print("A1 runtime frame0 exact A0 =", frame0_a1)
    print("A2 runtime frame0 exact A0 =", frame0_a2)
    print("A2 native block anchors exact A0 =", a2_anchor_exact)

    s = {
        name: stats_all(x)
        for name, x in q.items()
    }

    legality = {
        "A1": summarize_deltas(
            s["A0"], s["A1"]
        ),
        "A2": summarize_deltas(
            s["A0"], s["A2"]
        ),
    }

    masks = {
        "A1": intervention_mask(
            paths["A0"],
            paths["A1"],
        ),
        "A2": intervention_mask(
            paths["A0"],
            paths["A2"],
        ),
    }

    rho = {}

    for v in ["A1", "A2"]:
        rho[v] = {
            "global_A0":
                temporal_corr(q["A0"]),
            "global_variant":
                temporal_corr(q[v]),
            "support_A0":
                temporal_corr(
                    q["A0"],
                    masks[v],
                ),
            "support_variant":
                temporal_corr(
                    q[v],
                    masks[v],
                ),
            "compatibility_gate":
                rho_delta_gate(
                    q["A0"],
                    q[v],
                    masks[v],
                ),
        }

    routes = {}

    for v in ["A1", "A2"]:
        hard = (
            frame0_a1
            if v == "A1"
            else (
                frame0_a2
                and a2_anchor_exact
            )
        )

        routes[v] = bool(
            hard
            and legality[v]["all_pass"]
            and rho[v]["compatibility_gate"]["pass"]
        )

    if routes["A1"] and routes["A2"]:
        recommendation = "KEEP_A1_AND_A2"
    elif routes["A1"]:
        recommendation = "KEEP_A1_ONLY"
    elif routes["A2"]:
        recommendation = "ROUTE_TO_A2_ONLY"
    else:
        recommendation = "STOP_BEFORE_GPU_GENERATION"

    rec = {
        "runtime_hard_gates": {
            "A1_frame0_exact_A0": frame0_a1,
            "A2_frame0_exact_A0": frame0_a2,
            "A2_block_anchors_exact_A0": a2_anchor_exact,
        },
        "legality": legality,
        "rho": rho,
        "route_pass": routes,
        "recommendation": recommendation,
    }

    report["cases"][case] = rec

    for v in ["A1", "A2"]:
        print()
        print(v)
        print(
            " legality =",
            legality[v]["all_pass"],
        )
        print(
            " rho1 support P95 =",
            round(
                rho[v]["compatibility_gate"]
                ["p95_abs_delta"], 6
            ),
        )
        print(
            " rho1 support MAX  =",
            round(
                rho[v]["compatibility_gate"]
                ["max_abs_delta"], 6
            ),
        )
        print(
            " route pass =",
            routes[v],
        )

    print()
    print(
        "RECOMMENDATION =",
        recommendation,
    )


Path(
    "persistent_noise_runtime_audit.json"
).write_text(
    json.dumps(report, indent=2) + "\n"
)

print("\nPERSISTENT_NOISE_RUNTIME_AUDIT_DONE")
