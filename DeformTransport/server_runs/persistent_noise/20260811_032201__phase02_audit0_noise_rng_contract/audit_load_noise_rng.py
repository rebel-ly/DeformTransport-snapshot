import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0,"/workspace/DeformTransport")
import infer_sim


ROOT=Path("/workspace/DeformTransport")

ASSETS={
    "santa":
        ROOT/"server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/noises.npy",

    "tree":
        ROOT/"server_runs/20260804_234925_autonomous_deformtransport/"
        "prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/noises.npy",
}


def tensor_hash(x):
    a=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def run_once(path,seed):
    trace={
        "randperm":[],
        "randn_like":[]
    }

    real_randperm=torch.randperm
    real_randn_like=torch.randn_like

    def traced_randperm(*args,**kwargs):
        x=real_randperm(*args,**kwargs)
        trace["randperm"].append({
            "shape":list(x.shape),
            "values":x.detach().cpu().tolist(),
        })
        return x

    def traced_randn_like(*args,**kwargs):
        x=real_randn_like(*args,**kwargs)
        trace["randn_like"].append({
            "shape":list(x.shape),
            "dtype":str(x.dtype),
            "sha256":tensor_hash(x),
            "mean":float(x.float().mean()),
            "var":float(x.float().var(unbiased=False)),
        })
        return x

    torch.randperm=traced_randperm
    torch.randn_like=traced_randn_like

    try:
        torch.manual_seed(seed)

        out=infer_sim.load_noise(
            noise_path=str(path),
            target_frames=21,
            channel_dim=16,
            downsample_mode="nearest",
            eval_degradation=0.5,
        )

    finally:
        torch.randperm=real_randperm
        torch.randn_like=real_randn_like

    rec={
        "trace":trace,
        "outputs":{}
    }

    for k,v in out.items():
        if torch.is_tensor(v):
            rec["outputs"][k]={
                "shape":list(v.shape),
                "dtype":str(v.dtype),
                "sha256":tensor_hash(v),
                "mean":float(v.float().mean()),
                "var":float(v.float().var(unbiased=False)),
            }

    return rec,out


report={
    "seed":0,
    "cases":{}
}

for case,path in ASSETS.items():

    print("AUDIT",case,path,flush=True)

    r1,o1=run_once(path,0)
    r2,o2=run_once(path,0)

    keys=sorted(
        set(o1.keys())
        & set(o2.keys())
    )

    exact=True

    for k in keys:
        if torch.is_tensor(o1[k]):
            exact &= torch.equal(o1[k],o2[k])

    r1["repeat_bitwise_exact"]=bool(exact)

    report["cases"][case]=r1

    print(
        case,
        "repeat_bitwise_exact =",
        exact
    )

    print(
        case,
        "randperm_calls =",
        len(r1["trace"]["randperm"])
    )

    print(
        case,
        "randn_like_calls =",
        len(r1["trace"]["randn_like"])
    )

    for k,x in r1["outputs"].items():
        print(
            " ",
            k,
            x["shape"],
            "mean=",round(x["mean"],6),
            "var=",round(x["var"],6),
        )


Path("load_noise_rng_audit.json").write_text(
    json.dumps(report,indent=2)+"\n"
)

print()
print("SAVED load_noise_rng_audit.json")
