#!/usr/bin/env python3
"""DIAGNOSTIC_ONLY: reproduce frozen TC-MAR/TC-ME and persist underlying arrays."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


FROZEN = {
    "mar_lab": {
        "rw": {"mean": 13.639900159270573, "median": 9.546106088056899, "p95": 40.176803651452055},
        "dt_full": {"mean": 17.144317299874714, "median": 14.77844642547139, "p95": 37.01063968507866},
    },
    "mar_rgb": {
        "rw": {"mean": 0.0971740217773049, "median": 0.04986683472192713, "p95": 0.3180196253644923},
        "dt_full": {"mean": 0.11649965856279348, "median": 0.07873046770691872, "p95": 0.3197196369059383},
    },
    "me": {
        "rw": {"mean": 0.5869890665947547, "median": 0.5070961964161795, "p95": 1.153139444856402},
        "dt_full": {"mean": 0.7265499674289193, "median": 0.6906036221851227, "p95": 1.3080249503965733},
    },
}


def load_module(path):
    spec = importlib.util.spec_from_file_location("frozen_f1r4_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def absdiffs(actual, expected):
    return {k: abs(float(actual[k]) - float(expected[k])) for k in ("mean", "median", "p95")}


def mar(ev, root, suite, ids, tracks, visibility, out):
    n = tracks.shape[1]
    source = ev.read_rgb_image(root / ev.CASES["santa"]["source"])
    src_centers = tracks[0]
    src_valid = ((src_centers[:, 0] - 3.5 >= 0) & (src_centers[:, 0] + 3.5 <= 831) &
                 (src_centers[:, 1] - 3.5 >= 0) & (src_centers[:, 1] + 3.5 <= 479))
    src_patch = np.full((n, 8, 8, 3), np.nan, np.float32)
    good = np.where(src_valid)[0]
    src_patch[good] = ev.sample_patches(source, src_centers[good])
    src_lab = np.full((n, 3), np.nan, np.float32)
    src_lab[good] = ev.patch_mean_lab(src_patch[good])
    paths = ev.method_paths(root, suite, "santa")
    result = {}
    counts_ref = None
    observations_ref = None
    for method in ("rw", "dt_full"):
        video = ev.read_video_common(paths[method])
        lab_rows, rgb_rows, observations = [], [], 0
        for frame in ev.ANCHORS:
            centers = tracks[frame].copy()
            centers[:, 1] *= 464.0 / 480.0
            future = ((centers[:, 0] - 3.5 >= 0) & (centers[:, 0] + 3.5 <= 831) &
                      (centers[:, 1] - 3.5 >= 0) & (centers[:, 1] + 3.5 <= 463))
            valid = visibility[frame] & src_valid & future & np.isfinite(centers).all(axis=1)
            carrier = np.where(valid)[0]
            observations += len(carrier)
            patch = ev.sample_patches(video[frame], centers[carrier])
            lab = np.linalg.norm(ev.patch_mean_lab(patch) - src_lab[carrier], axis=1)
            rgb = np.abs(patch - src_patch[carrier]).mean(axis=(1, 2, 3))
            lab_rows.append((carrier, lab)); rgb_rows.append((carrier, rgb))
        lab, counts = ev.aggregate(lab_rows, n)
        rgb, counts2 = ev.aggregate(rgb_rows, n)
        assert np.array_equal(counts, counts2)
        if counts_ref is None:
            counts_ref, observations_ref = counts, observations
        else:
            assert np.array_equal(counts_ref, counts) and observations_ref == observations
        result[method] = {"lab": lab, "rgb": rgb}
    valid = counts_ref > 0
    np.savez_compressed(
        out / "per_carrier_mar_diagnostic.npz",
        material_id=ids, valid=valid, anchor_count=counts_ref,
        rw_lab=result["rw"]["lab"], dt_lab=result["dt_full"]["lab"],
        delta_lab=result["dt_full"]["lab"] - result["rw"]["lab"],
        rw_rgb=result["rw"]["rgb"], dt_rgb=result["dt_full"]["rgb"],
        delta_rgb=result["dt_full"]["rgb"] - result["rw"]["rgb"],
    )
    summaries, diffs = {}, {}
    for metric, key in (("mar_lab", "lab"), ("mar_rgb", "rgb")):
        summaries[metric] = {}
        diffs[metric] = {}
        for method in ("rw", "dt_full"):
            actual = ev.stats(result[method][key][valid])
            summaries[metric][method] = actual
            diffs[metric][method] = absdiffs(actual, FROZEN[metric][method])
    report = {
        "status": "DIAGNOSTIC_ONLY", "per_sample_unit": "per carrier: mean across valid frozen anchors",
        "N": n, "valid_tracks": int(valid.sum()), "valid_anchor_observations": int(observations_ref),
        "summaries": summaries, "formal_abs_diff": diffs,
        "delta_lab": {"mean": float(np.mean((result['dt_full']['lab'] - result['rw']['lab'])[valid])),
                      "median": float(np.median((result['dt_full']['lab'] - result['rw']['lab'])[valid])),
                      "p05": float(np.percentile((result['dt_full']['lab'] - result['rw']['lab'])[valid], 5)),
                      "p95": float(np.percentile((result['dt_full']['lab'] - result['rw']['lab'])[valid], 95)),
                      "fraction_dt_better": float(np.mean((result['dt_full']['lab'] - result['rw']['lab'])[valid] < 0))},
        "delta_rgb": {"mean": float(np.mean((result['dt_full']['rgb'] - result['rw']['rgb'])[valid])),
                      "median": float(np.median((result['dt_full']['rgb'] - result['rw']['rgb'])[valid])),
                      "p05": float(np.percentile((result['dt_full']['rgb'] - result['rw']['rgb'])[valid], 5)),
                      "p95": float(np.percentile((result['dt_full']['rgb'] - result['rw']['rgb'])[valid], 95)),
                      "fraction_dt_better": float(np.mean((result['dt_full']['rgb'] - result['rw']['rgb'])[valid] < 0))},
    }
    (out / "per_carrier_mar_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def motion(ev, root, suite, ids, tracks, visibility, out, batch):
    n = tracks.shape[1]
    support, reference, ref_magnitude = [], [], np.full((80, n), np.nan, np.float64)
    for frame in range(80):
        valid = (visibility[frame] & visibility[frame + 1] &
                 np.isfinite(tracks[frame]).all(axis=1) & np.isfinite(tracks[frame + 1]).all(axis=1))
        carrier = np.where(valid)[0]
        centers = tracks[frame, carrier] / 2.0
        ref = (tracks[frame + 1, carrier] - tracks[frame, carrier]) / 2.0
        inbound = ((centers[:, 0] >= 0) & (centers[:, 0] <= 415) &
                   (centers[:, 1] >= 0) & (centers[:, 1] <= 239))
        carrier, centers, ref = carrier[inbound], centers[inbound], ref[inbound]
        support.append((carrier, centers)); reference.append(ref)
        ref_magnitude[frame, carrier] = np.linalg.norm(ref, axis=1)
    device = torch.device("cuda:0")
    model, transforms = ev.load_raft_cached(device)
    paths = ev.method_paths(root, suite, "santa")
    epe = {method: np.full((80, n), np.nan, np.float64) for method in ("rw", "dt_full")}
    transition_means = {}
    for method in ("rw", "dt_full"):
        video = ev.read_video_common(paths[method])
        with torch.inference_mode():
            for start in range(0, 80, batch):
                end = min(80, start + batch)
                a = torch.from_numpy(video[start:end]).permute(0, 3, 1, 2).to(device)
                b = torch.from_numpy(video[start + 1:end + 1]).permute(0, 3, 1, 2).to(device)
                a = F.interpolate(a, size=(240, 416), mode="area")
                b = F.interpolate(b, size=(240, 416), mode="area")
                a, b = transforms(a, b)
                prediction = model(a, b)[-1].float().cpu().numpy()
                for j, frame in enumerate(range(start, end)):
                    carrier, centers = support[frame]
                    pred = ev.bilinear_flow(prediction[j], centers)
                    values = np.linalg.norm(pred - reference[frame], axis=1)
                    epe[method][frame, carrier] = values
        transition_means[method] = np.asarray([epe[method][frame, support[frame][0]].mean() for frame in range(80)], np.float64)
        del video
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    delta = epe["dt_full"] - epe["rw"]
    np.savez_compressed(
        out / "per_transition_me_diagnostic.npz",
        material_id=ids, rw_epe=epe["rw"], dt_epe=epe["dt_full"], delta_epe=delta,
        reference_motion_magnitude=ref_magnitude,
        rw_transition_mean=transition_means["rw"], dt_transition_mean=transition_means["dt_full"],
        delta_transition_mean=transition_means["dt_full"] - transition_means["rw"],
    )
    summaries, diffs = {}, {}
    for method in ("rw", "dt_full"):
        values = transition_means[method]
        actual = {"mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
        summaries[method] = actual
        diffs[method] = absdiffs(actual, FROZEN["me"][method])
    finite_delta = delta[np.isfinite(delta)]
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "formal_per_sample_unit": "per transition mean of per-carrier EPE; 80 transitions",
        "diagnostic_expansion": "per-transition/per-carrier EPE on the identical formal support",
        "summaries": summaries, "formal_abs_diff": diffs,
        "delta_epe": {"N_instances": int(finite_delta.size), "mean": float(finite_delta.mean()),
                      "median": float(np.median(finite_delta)), "p05": float(np.percentile(finite_delta, 5)),
                      "p95": float(np.percentile(finite_delta, 95)), "fraction_dt_better": float(np.mean(finite_delta < 0))},
        "temporal_halves_fixed_not_tuned": {
            "early_0_39": float(np.nanmean(delta[:40])),
            "late_40_79": float(np.nanmean(delta[40:])),
        },
    }
    (out / "per_transition_me_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluator", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--suite", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--tracks", required=True); ap.add_argument("--visibility", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ev = load_module(args.evaluator); root, suite = Path(args.root), Path(args.suite)
    ids = np.load(args.ids).astype(np.int64)
    tracks = np.load(args.tracks)[0].astype(np.float32)
    visibility = np.load(args.visibility)[0].astype(bool)
    mar_report = mar(ev, root, suite, ids, tracks, visibility, out)
    motion_report = motion(ev, root, suite, ids, tracks, visibility, out, args.batch)
    all_diffs = []
    for report in (mar_report, motion_report):
        for metric in report["formal_abs_diff"].values():
            for values in metric.values() if "rw" in metric else [metric]:
                all_diffs.extend(values.values())
    max_diff = max(all_diffs)
    summary = {"PER_SAMPLE_DIAGNOSTIC_REPRODUCES_FORMAL": "PASS" if max_diff == 0 else "FAIL",
               "PER_SAMPLE_FORMAL_MAX_ABS_DIFF": max_diff}
    (out / "per_sample_reproduction_gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if max_diff != 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
