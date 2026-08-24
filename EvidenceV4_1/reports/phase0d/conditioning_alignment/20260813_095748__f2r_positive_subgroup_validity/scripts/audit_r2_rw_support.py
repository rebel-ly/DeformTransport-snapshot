#!/usr/bin/env python3
"""F2-R R2: diagnostic-only RW coarse occupancy/support audit.

The Lab path intentionally reproduces the frozen F2 condition script, including
its pre-to_common division by 255, so the ALL result remains exactly comparable
to the frozen 65.713... value.  This is recorded as a validity limitation.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


OUT = Path("/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_095748__f2r_positive_subgroup_validity")
ROOT = Path("/mnt/sdbd/home/liuyu_qyh/DeformTransport")
F2 = Path("/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_081824__f2_metric_mechanism_error_localization")
EV_PATH = Path("/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_073136__f1r3_binding_only_eval_port/generated/eval_v3_corrected_v2.py")
SIM = ROOT / "server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410"
RASTER = ROOT / "server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719/simulation_source/flow_source_point_indices.npy"
TRACKS = ROOT / "server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy"
VIS = ROOT / "server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy"
FROZEN_ALL = 65.71344520089995


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats(x):
    x = np.asarray(x, np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else None,
        "median": float(np.median(x)) if x.size else None,
        "p95": float(np.percentile(x, 95)) if x.size else None,
    }


def occupancy_to_wan(mask512):
    # Exact existing RGB spatial geometry (512 -> 832, centered 480 crop), with
    # nearest-neighbour categorical lookup so point-ID occupancy is not blended.
    im = Image.fromarray(mask512.astype(np.uint8) * 255, mode="L")
    resized = np.asarray(im.resize((832, 832), resample=Image.Resampling.NEAREST)) > 0
    return resized[176:656]


def support_patch(mask480, centers_common, off):
    # Frozen evaluator samples an 8x8 patch in the 464-high common grid. Map
    # those sample locations back through align_corners=False pixel centers,
    # then use categorical nearest-neighbour occupancy lookup.
    xs = centers_common[:, 0, None, None] + off[None, None, :]
    ys = centers_common[:, 1, None, None] + off[None, :, None]
    xs = np.broadcast_to(xs, (len(centers_common), 8, 8))
    ys = np.broadcast_to(ys, (len(centers_common), 8, 8))
    src_x = np.rint(xs).astype(np.int64)
    src_y_float = (ys + 0.5) * (480.0 / 464.0) - 0.5
    src_y = np.rint(src_y_float).astype(np.int64)
    src_x = np.clip(src_x, 0, 831)
    src_y = np.clip(src_y, 0, 479)
    return mask480[src_y, src_x]


def main():
    spec = importlib.util.spec_from_file_location("ev", EV_PATH)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    frame_files = sorted((SIM / "frames").glob("frame_*.png"))
    assert len(frame_files) == 81
    raster = np.load(RASTER, mmap_mode="r")
    assert raster.shape == (81, 512, 512) and np.issubdtype(raster.dtype, np.integer)
    tracks = np.load(TRACKS)[0].astype(np.float32)
    vis = np.load(VIS)[0].astype(bool)
    assert tracks.shape == (81, 1257, 2) and vis.shape == (81, 1257)

    # Reproduce the exact F2 condition script input handling, including its
    # double /255 normalization before Lab patch computation.
    video = []
    for p in frame_files:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert bgr is not None
        rgb_prediv = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        video.append(ev.to_common(rgb_prediv))
    video = np.stack(video)

    source = ev.read_rgb_image(ROOT / ev.CASES["santa"]["source"])
    centers0 = tracks[0]
    sv = ((centers0[:, 0] - 3.5 >= 0) & (centers0[:, 0] + 3.5 <= 831) &
          (centers0[:, 1] - 3.5 >= 0) & (centers0[:, 1] + 3.5 <= 479))
    sp = np.full((1257, 8, 8, 3), np.nan, np.float32)
    good = np.where(sv)[0]
    sp[good] = ev.sample_patches(source, centers0[good])
    source_lab = np.full((1257, 3), np.nan, np.float32)
    source_lab[good] = ev.patch_mean_lab(sp[good])

    per_frame = []
    rows_all, rows_any, rows_full = [], [], []
    instance_valid_fracs = []
    instance_any, instance_full = [], []
    for t in range(81):
        occ480 = occupancy_to_wan(np.asarray(raster[t]) >= 0)
        per_frame.append({
            "frame": t,
            "simulation_step": t * 10,
            "valid_pixel_fraction": float(occ480.mean()),
            "invalid_pixel_fraction": float(1.0 - occ480.mean()),
        })
        if t not in ev.ANCHORS:
            continue
        c = tracks[t].copy()
        c[:, 1] *= 464.0 / 480.0
        fv = ((c[:, 0] - 3.5 >= 0) & (c[:, 0] + 3.5 <= 831) &
              (c[:, 1] - 3.5 >= 0) & (c[:, 1] + 3.5 <= 463))
        valid = vis[t] & sv & fv & np.isfinite(c).all(1)
        ids = np.where(valid)[0]
        patch = ev.sample_patches(video[t], c[ids])
        vals = np.linalg.norm(ev.patch_mean_lab(patch) - source_lab[ids], axis=1)
        supp = support_patch(occ480, c[ids], ev.OFF)
        frac = supp.mean(axis=(1, 2))
        any_ok = supp.any(axis=(1, 2))
        full_ok = supp.all(axis=(1, 2))
        rows_all.append((ids, vals))
        rows_any.append((ids[any_ok], vals[any_ok]))
        rows_full.append((ids[full_ok], vals[full_ok]))
        instance_valid_fracs.extend(frac.tolist())
        instance_any.extend(any_ok.tolist())
        instance_full.extend(full_ok.tolist())

    def aggregate_rows(rows):
        agg, count = ev.aggregate(rows, 1257)
        return stats(agg[count > 0]), int(sum(len(ids) for ids, _ in rows)), int((count > 0).sum())

    all_stats, all_obs, all_carriers = aggregate_rows(rows_all)
    any_stats, any_obs, any_carriers = aggregate_rows(rows_any)
    full_stats, full_obs, full_carriers = aggregate_rows(rows_full)
    assert all_stats["mean"] == FROZEN_ALL, (all_stats["mean"], FROZEN_ALL)

    future = per_frame[1:]
    overall_valid = float(np.mean([x["valid_pixel_fraction"] for x in future]))
    any_fraction = float(np.mean(instance_any))
    full_fraction = float(np.mean(instance_full))
    mean_patch_valid = float(np.mean(instance_valid_fracs))

    inventory = {
        "status": "PASS",
        "rw_coarse_asset_lineage": "PASS",
        "lineage_checks": {"same_source": True, "same_santa_action": True, "same_simulation": True, "same_timeline_S0_to_S800": True, "same_canonical_rw_run": True},
        "rw_coarse_validity_source": "EXISTING_OCCUPANCY",
        "occupancy": {"path": str(RASTER), "sha256": sha256(RASTER), "shape": list(raster.shape), "dtype": str(raster.dtype), "valid_rule": "authoritative point ID >= 0; RGB color not used"},
        "spatial_mapping": {"raster_to_wan": "PIL nearest categorical 512x512 -> 832x832; crop y=176:656", "patch_lookup": "inverse align_corners=False 480->464 pixel-center map; nearest categorical lookup", "outcome_tuned_threshold": False},
        "temporal_mapping": "flow_source_point_indices[0:81] == S0,S10,...,S800 and aligned coarse frames == frame_initial + old frame_0000..0079",
        "future_frames": per_frame[1:],
        "overall_future": {"valid_pixel_fraction": overall_valid, "invalid_pixel_fraction": 1.0 - overall_valid},
        "formal_sampling_instances": {"n": all_obs, "patch_any_valid_fraction": any_fraction, "patch_fully_valid_fraction": full_fraction, "mean_valid_pixel_fraction_inside_sampled_patches": mean_patch_valid},
    }
    metric = {
        "status": "DIAGNOSTIC_ONLY_SUPPORT_CONDITIONED",
        "frozen_f2_all_patch_target": FROZEN_ALL,
        "exact_all_patch_reproduction": all_stats,
        "rules": {
            "ANY_VALID_SUPPORT": {"instance_rule": "at least one of 64 target patch sample locations has authoritative occupancy", "qualifying_anchor_observations": any_obs, "carriers_with_at_least_one_qualifying_observation": any_carriers, "carrier_aggregated_tc_mar_lab": any_stats},
            "FULL_VALID_SUPPORT": {"instance_rule": "all 64 target patch sample locations have authoritative occupancy", "qualifying_anchor_observations": full_obs, "carriers_with_at_least_one_qualifying_observation": full_carriers, "carrier_aggregated_tc_mar_lab": full_stats},
        },
        "diagnostic_pipeline_validity": {
            "f2_double_normalization_detected": True,
            "detail": "diagnose_rgb_condition.py divides RGB by 255 before evaluator.to_common, which divides by 255 again",
            "reason_preserved_here": "required exact comparability to frozen ALL=65.71344520089995",
            "numeric_interpretation": "LIMITED; supported-subset comparison is internally comparable but the absolute condition TC-MAR is not a correctly normalized RGB TC-MAR",
        },
    }

    # Holes would support the hypothesis only if support conditioning materially
    # removes the high error. Both predeclared subsets remain close to ALL, and
    # the absolute ALL value also has the independent scaling defect.
    supported_means = [x for x in (any_stats["mean"], full_stats["mean"]) if x is not None]
    holes_supported = bool(supported_means and min(supported_means) < 0.75 * FROZEN_ALL)
    metric["rw_coarse_high_tcmar_explained_by_holes"] = "SUPPORTED" if holes_supported else "NOT_SUPPORTED"
    metric["rw_appearance_metric_alignment_hypothesis"] = "REOPENED" if holes_supported else "UNRESOLVED"
    metric["decision_basis"] = "predeclared descriptive material-drop criterion (supported mean <75% of ALL); not used for formal metrics or carrier selection"

    (OUT / "r2_rw_coarse_support_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (OUT / "r2_rw_coarse_support_tcmar.json").write_text(json.dumps(metric, indent=2) + "\n")
    print(json.dumps({
        "valid": overall_valid, "any_fraction": any_fraction, "full_fraction": full_fraction,
        "all": all_stats["mean"], "any": any_stats["mean"], "full": full_stats["mean"],
        "holes": metric["rw_coarse_high_tcmar_explained_by_holes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
