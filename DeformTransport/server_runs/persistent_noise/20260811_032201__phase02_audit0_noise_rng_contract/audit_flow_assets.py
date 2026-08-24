from pathlib import Path
import numpy as np
import json

ROOT=Path("/workspace/DeformTransport")

DIRS={
    "santa":
        ROOT/"server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410",

    "tree":
        ROOT/"server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055",
}

report={}

for case,d in DIRS.items():

    print("\n"+"="*80)
    print(case, d)
    print("="*80)

    rows=[]

    for p in sorted(d.rglob("*")):

        if not p.is_file():
            continue

        n=p.name.lower()

        if not (
            "flow" in n
            or n=="noises.npy"
            or "noise" in n
        ):
            continue

        rec={
            "path":str(p),
            "bytes":p.stat().st_size,
        }

        if p.suffix==".npy":
            try:
                a=np.load(p,mmap_mode="r")
                rec["shape"]=list(a.shape)
                rec["dtype"]=str(a.dtype)

                print(
                    p.name,
                    "shape=",a.shape,
                    "dtype=",a.dtype,
                )

            except Exception as e:
                rec["error"]=repr(e)

        else:
            print(p.name)

        rows.append(rec)

    report[case]=rows

Path("flow_asset_audit.json").write_text(
    json.dumps(report,indent=2)+"\n"
)

print("\nFLOW_ASSET_AUDIT_DONE")
