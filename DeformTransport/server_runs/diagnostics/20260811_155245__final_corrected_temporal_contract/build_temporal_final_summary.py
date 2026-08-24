import csv, json
from pathlib import Path

ROOT=Path("/workspace/DeformTransport")
RUN=Path(
    (ROOT/"server_runs/diagnostics/current_final_corrected_temporal.txt")
    .read_text().strip()
)

app=json.loads(
    (RUN/"primary_balanced_tcmar.json").read_text()
)

mot=json.loads(
    (RUN/"temporal_motion.json").read_text()
)

cases=["santa","tree"]
anchors=[str(x) for x in range(4,81,4)]

out={
    "protocol":"Final case-specific temporal diagnostics V2",
    "cross_case_aggregation":False,
    "cases":{}
}

rows=[]

for case in cases:
    ac=app["cases"][case]
    mc=mot["cases"][case]

    na=ac["n_tracks"]
    nm=mc["balanced_motion_tracks"]

    decisions={
        "appearance":{"WIN":0,"TIE":0,"LOSS":0},
        "motion":{"WIN":0,"TIE":0,"LOSS":0},
    }

    for t in anchors:
        aa=ac["anchors"][t]
        mm=mc["anchors"][t]

        decisions["appearance"][aa["decision"]]+=1
        decisions["motion"][mm["decision"]]+=1

        rows.append({
            "case":case,
            "frame":int(t),
            "tcmar_rw_minus_v3d":aa["rw_minus_v3d"],
            "tcmar_decision":aa["decision"],
            "tcme_rw_minus_v3d":mm["rw_minus_v3d"],
            "tcme_decision":mm["decision"],
        })

    out["cases"][case]={
        "appearance_tracks":na,
        "motion_tracks":nm,
        "motion_retention_fraction":nm/na,
        "appearance_early":ac["early_4_40"],
        "appearance_late":ac["late_44_80"],
        "motion_early":mc["early_4_40"],
        "motion_late":mc["late_44_80"],
        "anchor_decision_counts":decisions,
    }

(RUN/"temporal_final_summary.json").write_text(
    json.dumps(out,indent=2)+"\n"
)

with open(RUN/"temporal_final_per_anchor.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("===== FINAL TEMPORAL SUMMARY =====")

for case in cases:
    c=out["cases"][case]

    print("\n",case.upper())
    print(
        "support appearance/motion =",
        c["appearance_tracks"],
        "/",
        c["motion_tracks"]
    )
    print(
        "motion retention =",
        round(c["motion_retention_fraction"],4)
    )
    print("TC-MAR early =",c["appearance_early"])
    print("TC-MAR late  =",c["appearance_late"])
    print("TC-ME early  =",c["motion_early"])
    print("TC-ME late   =",c["motion_late"])
    print("anchor counts =",c["anchor_decision_counts"])

print("\nTEMPORAL_FINAL_SUMMARY_DONE")
