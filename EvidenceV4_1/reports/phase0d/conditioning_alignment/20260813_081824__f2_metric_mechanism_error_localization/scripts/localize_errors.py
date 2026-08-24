#!/usr/bin/env python3
"""DIAGNOSTIC_ONLY joins, stratification, correlations, and motion diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def mean_or_none(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return None if not len(x) else float(x.mean())


def group_summary(mask, mar, me):
    mask = np.asarray(mask, bool)
    mv = mask & mar["valid"]
    out = {"N_carriers": int(mask.sum()), "N_mar": int(mv.sum())}
    for key in ("rw_lab", "dt_lab", "delta_lab", "rw_rgb", "dt_rgb", "delta_rgb"):
        out[key] = mean_or_none(mar[key][mv])
    # Carrier-level ME is mean over every supported transition for that carrier.
    rw_car = np.nanmean(me["rw_epe"], axis=0); dt_car = np.nanmean(me["dt_epe"], axis=0)
    supported = np.isfinite(rw_car) & np.isfinite(dt_car) & mask
    out.update({"N_me": int(supported.sum()), "rw_me": mean_or_none(rw_car[supported]),
                "dt_me": mean_or_none(dt_car[supported]), "delta_me": mean_or_none((dt_car-rw_car)[supported])})
    return out


def runs_and_switches(v):
    switches = np.count_nonzero(v[1:] != v[:-1], axis=0)
    starts = v & np.vstack([np.ones((1, v.shape[1]), bool), ~v[:-1]])
    runs = starts.sum(axis=0)
    reappearance = np.maximum(runs - v[0].astype(np.int64), 0)
    return switches.astype(np.int64), runs.astype(np.int64), reappearance.astype(np.int64)


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3: return {"status": "INSUFFICIENT", "N": int(mask.sum()), "rho": None}
    try:
        from scipy.stats import spearmanr
    except Exception:
        return {"status": "NOT_COMPUTED_DEPENDENCY_UNAVAILABLE", "N": int(mask.sum()), "rho": None}
    r = spearmanr(x[mask], y[mask])
    return {"status": "DIAGNOSTIC_ONLY", "N": int(mask.sum()), "rho": None if np.isnan(r.statistic) else float(r.statistic),
            "pvalue_exploratory_not_formal": None if np.isnan(r.pvalue) else float(r.pvalue)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mar", required=True); ap.add_argument("--me", required=True)
    ap.add_argument("--ids", required=True); ap.add_argument("--tracks", required=True); ap.add_argument("--visibility", required=True)
    ap.add_argument("--source-cells", required=True); ap.add_argument("--funnel", required=True); ap.add_argument("--aligned", required=True)
    ap.add_argument("--out", required=True); args = ap.parse_args(); out = Path(args.out)
    mar = dict(np.load(args.mar)); me = dict(np.load(args.me)); ids = np.load(args.ids).astype(np.int64)
    tracks = np.load(args.tracks)[0].astype(np.float32); visibility = np.load(args.visibility)[0].astype(bool)
    source_cells = np.load(args.source_cells).astype(np.int64)
    rows = list(csv.DictReader(open(args.funnel, newline="")))
    rows = [r for r in rows if r["condition"] == "Correct"]
    by_id = {int(r["material_id"]): r for r in rows}
    missing = [int(i) for i in ids if int(i) not in by_id]
    duplicates = len(by_id) != len(rows)
    joined = [by_id[int(i)] for i in ids if int(i) in by_id]
    if missing or duplicates or len(joined) != 1257:
        report = {"ID_JOIN_COVERAGE": len(joined), "expected": 1257, "missing_ids": missing, "duplicate_ids": duplicates, "status": "UNRESOLVED"}
        (out/"transport_error_join.json").write_text(json.dumps(report, indent=2)+"\n"); raise SystemExit(3)
    iv = lambda key: np.asarray([int(r[key]) for r in joined], np.int64)
    win = iv("TOTAL_WIN_COUNT"); visible_slots = iv("VISIBLE_SLOT_COUNT"); collision_slots = iv("COLLISION_INVOLVED_SLOT_COUNT")
    collision_losses = iv("COLLISION_LOSS_COUNT"); zero_class = np.asarray([r["ZERO_CLASS"] for r in joined])
    contributor = win > 0; no_visible = visible_slots == 0; always_loser = zero_class == "Z3_HAS_CANDIDATES_BUT_ALWAYS_LOSES_COLLISION"
    switches, visible_runs, reappearance = runs_and_switches(visibility)
    disp2 = np.linalg.norm(tracks - tracks[0:1], axis=2)
    source_to_future = np.nanmean(disp2[1:], axis=0)
    trajectory_energy_2d = np.nansum(np.linalg.norm(np.diff(tracks, axis=0), axis=2), axis=0)
    aligned = torch.load(args.aligned, map_location="cpu")
    all_points = aligned["points_3d"].cpu().numpy().astype(np.float64)
    selected3d = all_points[:, ids]
    displacement3d = np.linalg.norm(selected3d - selected3d[0:1], axis=2)
    source_to_future_3d = np.mean(displacement3d[1:], axis=0)
    trajectory_energy_3d = np.sum(np.linalg.norm(np.diff(selected3d, axis=0), axis=2), axis=0)
    # No coarse-bin definition is frozen; do not tune or compute S_i_3D.
    local_deformation = "DEFERRED_TO_F3"
    rw_car = np.nanmean(me["rw_epe"], axis=0); dt_car = np.nanmean(me["dt_epe"], axis=0); delta_me_car = dt_car-rw_car
    delta_mar = mar["delta_lab"]
    np.savez_compressed(out/"transport_error_join.npz", material_id=ids, source_cell=source_cells,
        contributor=contributor, winning_write_count=win, visible_slot_count=visible_slots,
        visibility_switch_count=switches, visible_run_count=visible_runs, reappearance_count=reappearance,
        collision_exposure_count=collision_slots, collision_loss_count=collision_losses,
        always_collision_loser=always_loser, no_visible=no_visible,
        source_to_future_displacement_2d=source_to_future, trajectory_energy_2d=trajectory_energy_2d,
        source_to_future_displacement_3d=source_to_future_3d, trajectory_energy_3d=trajectory_energy_3d,
        delta_mar_lab=delta_mar, delta_me_carrier=delta_me_car)
    join_report = {"status":"DIAGNOSTIC_ONLY","ID_JOIN_COVERAGE":1257,"expected":1257,"missing_ids":[],
        "exact_material_id_join":True,"fields":["TRANSPORT_CONTRIBUTOR","WINNING_WRITE_COUNT","VISIBLE_SLOT_COUNT","VISIBILITY_SWITCH_COUNT","VISIBLE_RUN_COUNT","REAPPEARANCE_COUNT","COLLISION_EXPOSURE_COUNT","ALWAYS_COLLISION_LOSER","NO_VISIBLE","SOURCE_CELL","motion/trajectory"],
        "LOCAL_DEFORMATION_SCORE":local_deformation,"reason":"No exact coarse-bin definition was frozen; no post-hoc bin tuning performed."}
    (out/"transport_error_join.json").write_text(json.dumps(join_report,indent=2)+"\n")
    groups = {
        "contributors": contributor, "zero_contributors": ~contributor,
        "no_visible": no_visible, "has_visible": ~no_visible,
        "always_collision_loser": always_loser, "not_always_collision_loser": ~always_loser,
        "visibility_switch_0": switches == 0, "visibility_switch_1_2": (switches >= 1)&(switches <= 2),
        "visibility_switch_3_4": (switches >= 3)&(switches <= 4), "visibility_switch_ge5": switches >= 5,
    }
    # Fixed timeline halves and quartiles; not selected on outcomes.
    strat = {name:group_summary(mask,mar,me) for name,mask in groups.items()}
    strat["me_temporal"]={"early_0_39":float(np.nanmean(me["delta_epe"][:40])),"late_40_79":float(np.nanmean(me["delta_epe"][40:]))}
    # Descriptive pre-existing exposure quartiles, explicitly exploratory.
    for varname,var in (("winning_write",win),("motion_3d",trajectory_energy_3d)):
        q=np.quantile(var,[.25,.5,.75]); strat[f"{varname}_exploratory_quantile_edges"]=[float(x) for x in q]
        for k,(lo,hi) in enumerate(((-np.inf,q[0]),(q[0],q[1]),(q[1],q[2]),(q[2],np.inf))):
            mask=(var>lo)&(var<=hi); strat[f"{varname}_Q{k+1}"]=group_summary(mask,mar,me)
    (out/"error_stratification.json").write_text(json.dumps({"status":"DIAGNOSTIC_ONLY","groups":strat},indent=2)+"\n")
    variables={"visible_slot_count":visible_slots,"visibility_switch_count":switches,"winning_write_count":win,
        "collision_exposure_count":collision_slots,"collision_loss_count":collision_losses,
        "source_to_future_displacement_2d":source_to_future,"trajectory_energy_2d":trajectory_energy_2d,
        "source_to_future_displacement_3d":source_to_future_3d,"trajectory_energy_3d":trajectory_energy_3d}
    corr={name:{"delta_mar_lab":spearman(var,delta_mar),"delta_me_carrier":spearman(var,delta_me_car)} for name,var in variables.items()}
    corr["delta_mar_vs_delta_me"]=spearman(delta_mar,delta_me_car)
    (out/"error_correlation.json").write_text(json.dumps({"status":"DIAGNOSTIC_ONLY","correlations":corr},indent=2)+"\n")
    print(json.dumps({"ID_JOIN_COVERAGE":1257,"groups":len(strat),"correlations":len(corr)}))


if __name__ == "__main__": main()
