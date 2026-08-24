import json, importlib.util
from pathlib import Path
import numpy as np

ROOT=Path("/workspace/DeformTransport")
RUN=Path.cwd()

PH=ROOT/"server_runs/diagnostics/20260811_002817__final_temporal_pnoise_phase01/phase01_final.py"
CONTRACT=RUN/"final_eval_contract.py"
SUITE=RUN/"suite"

sp=importlib.util.spec_from_file_location("ph",PH)
ph=importlib.util.module_from_spec(sp)
sp.loader.exec_module(ph)

ev=ph.load_ev(CONTRACT)

ANCH=list(range(4,81,4))
EARLY=list(range(4,41,4))
LATE=list(range(44,81,4))

SEM={
    "santa":{
        "rw":"Identity-Shuffled",
        "v3d":"Correct",
        "target":"Correct",
    },
    "tree":{
        "rw":"RealWonder A0",
        "v3d":"Persistent-Noise A2",
        "target":"A2",
    },
}

report={
    "protocol":"Primary anchor-balanced temporal TC-MAR",
    "metric":"frozen phase01_final.py::appearance_errors",
    "bootstrap_unit":"whole material track",
    "lower_is_better":True,
    "difference":"rw_minus_v3d; positive favors target",
    "cases":{},
}

for case in ["santa","tree"]:

    errors,paths,_=ph.appearance_errors(
        ev,ROOT,SUITE,case
    )

    ids=np.load(
        RUN/f"{case}_balanced_track_ids.npy"
    ).astype(np.int64)

    cr={
        "semantics":SEM[case],
        "n_tracks":int(len(ids)),
        "anchors":{},
    }

    for t in ANCH:
        rw=errors["rw"][t][ids]
        vv=errors["v3d"][t][ids]

        assert np.isfinite(rw).all()
        assert np.isfinite(vv).all()

        d=rw-vv
        ci=ev.bootstrap_mean_ci(d)

        cr["anchors"][str(t)]={
            "rw":float(rw.mean()),
            "v3d":float(vv.mean()),
            "rw_minus_v3d":float(d.mean()),
            "ci":ci,
            "decision":
                "WIN" if ci[0]>0
                else "LOSS" if ci[1]<0
                else "TIE",
        }

    for name,ts in [
        ("early_4_40",EARLY),
        ("late_44_80",LATE),
    ]:
        rw=np.stack(
            [errors["rw"][t][ids] for t in ts]
        ).mean(0)

        vv=np.stack(
            [errors["v3d"][t][ids] for t in ts]
        ).mean(0)

        d=rw-vv
        ci=ev.bootstrap_mean_ci(d)

        cr[name]={
            "rw":float(rw.mean()),
            "v3d":float(vv.mean()),
            "rw_minus_v3d":float(d.mean()),
            "ci":ci,
            "decision":
                "WIN" if ci[0]>0
                else "LOSS" if ci[1]<0
                else "TIE",
        }

    report["cases"][case]=cr

(RUN/"primary_balanced_tcmar.json").write_text(
    json.dumps(report,indent=2)+"\n"
)

print("\n===== PRIMARY BALANCED TC-MAR =====")
for case in ["santa","tree"]:
    c=report["cases"][case]
    print("\n",case,"n=",c["n_tracks"])
    print("EARLY",c["early_4_40"])
    print("LATE ",c["late_44_80"])

print("\nPRIMARY_BALANCED_TCMAR_DONE")
