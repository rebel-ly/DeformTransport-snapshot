import json, importlib.util, hashlib
from pathlib import Path
import numpy as np
import cv2

ROOT=Path("/workspace/DeformTransport")
RUN=Path.cwd()

EVAL=ROOT/"server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/eval_v3.py"
OLDPH=ROOT/"server_runs/diagnostics/20260811_002817__final_temporal_pnoise_phase01"

A0=ROOT/"server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_203228__tree__realwonder_baseline_seed0/tree_realwonder_baseline_seed0.mp4"

A2RUN=Path(
    (ROOT/"server_runs/persistent_noise_tree_A2_current.txt").read_text().strip()
)
A2=A2RUN/"tree_A2_block3_persistent_seed0.mp4"

s=importlib.util.spec_from_file_location("ev",EVAL)
ev=importlib.util.module_from_spec(s)
s.loader.exec_module(ev)

ANCH=list(range(4,81,4))
EARLY=list(range(4,41,4))
LATE=list(range(44,81,4))

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def boot(x):
    return ev.bootstrap_mean_ci(np.asarray(x,np.float64))

# Exact frozen Phase-01.5 implementation.
def lab_patch(x):
    x=np.asarray(x,np.float32)
    if x.max()>1.5: x=x/255.0
    n,h,w,c=x.shape
    y=cv2.cvtColor(x.reshape(n*h,w,3),cv2.COLOR_RGB2LAB)
    return y.reshape(n,h,w,3)

def tex_errors(src,cur):
    a=lab_patch(src); b=lab_patch(cur)

    patch=np.sqrt(
        np.mean(
            np.sum((b-a)**2,axis=-1),
            axis=(1,2)
        )
    )

    La=a[...,0]; Lb=b[...,0]
    gya,gxa=np.gradient(La,axis=(1,2))
    gyb,gxb=np.gradient(Lb,axis=(1,2))

    grad=np.mean(
        np.sqrt((gxb-gxa)**2+(gyb-gya)**2),
        axis=(1,2)
    )

    return patch.astype(np.float64),grad.astype(np.float64)

def valid_all(tr,vis):
    xy=tr/2.0
    good=(
        vis &
        np.isfinite(tr).all(axis=2) &
        (xy[:,:,0]>=0)&(xy[:,:,0]<=415)&
        (xy[:,:,1]>=0)&(xy[:,:,1]<=239)
    )
    return np.where(good.all(axis=0))[0]

assert A0.is_file(), A0
assert A2.is_file(), A2

cfg=ev.CASES["tree"]

tr=np.load(ROOT/cfg["tracks"])[0].astype(np.float32)
vis=np.load(ROOT/cfg["vis"])[0].astype(bool)

ids=valid_all(tr,vis)

# Hard contract: same exact Tree support as frozen Phase-01.5.
frozen=np.load(OLDPH/"tree_cumulative_drift_track_ids.npy").astype(np.int64)
assert np.array_equal(ids,frozen), (len(ids),len(frozen))
assert len(ids)==121, len(ids)

src_img=ev.read_rgb_image(ROOT/cfg["source"])
src_patch=ev.sample_patches(src_img,tr[0,ids])
src_mean=ev.patch_mean_lab(src_patch)

paths={
    "A0":A0,
    "A2":A2,
}

errs={
    m:{k:{} for k in ["tcmar","patch","grad"]}
    for m in paths
}

for method,path in paths.items():
    print("LOAD",method,path,flush=True)
    vid=ev.read_video_common(path)
    assert len(vid)>=81, (method,len(vid))

    for t in ANCH:
        xy=tr[t,ids].copy()

        # Exact frozen 480 -> 464 video-domain y transform.
        xy[:,1]*=464.0/480.0

        p=ev.sample_patches(vid[t],xy)

        mean=ev.patch_mean_lab(p)
        errs[method]["tcmar"][t]=np.linalg.norm(
            mean-src_mean,axis=1
        ).astype(np.float64)

        patch,grad=tex_errors(src_patch,p)
        errs[method]["patch"][t]=patch
        errs[method]["grad"][t]=grad

def summarize(ts,key):
    a0=np.stack([errs["A0"][key][t] for t in ts]).mean(0)
    a2=np.stack([errs["A2"][key][t] for t in ts]).mean(0)

    # Lower is better. Positive A0-A2 favors A2.
    d=a0-a2
    ci=boot(d)

    return {
        "A0":float(a0.mean()),
        "A2":float(a2.mean()),
        "A0_minus_A2":float(d.mean()),
        "ci":ci,
        "decision":
            "A2_WIN" if ci[0]>0
            else "A2_LOSS" if ci[1]<0
            else "TIE"
    }

report={
    "protocol":"Tree A0 vs A2; exact frozen Phase-01.5 appearance protocol",
    "primary_metric":"TC-MAR",
    "lower_is_better":True,
    "difference_definition":"A0_minus_A2; positive favors A2",
    "n_tracks":int(len(ids)),
    "track_ids":ids.tolist(),
    "sha256":{"A0":sha256(A0),"A2":sha256(A2)},
    "early":{},
    "late":{},
    "anchors":{}
}

for key,label in [
    ("tcmar","TC-MAR"),
    ("patch","TC-Patch-Lab"),
    ("grad","TC-Grad"),
]:
    report["early"][label]=summarize(EARLY,key)
    report["late"][label]=summarize(LATE,key)

for t in ANCH:
    report["anchors"][str(t)]={}
    for key,label in [
        ("tcmar","TC-MAR"),
        ("patch","TC-Patch-Lab"),
        ("grad","TC-Grad"),
    ]:
        a0=errs["A0"][key][t]
        a2=errs["A2"][key][t]
        d=a0-a2
        ci=boot(d)

        report["anchors"][str(t)][label]={
            "A0":float(a0.mean()),
            "A2":float(a2.mean()),
            "A0_minus_A2":float(d.mean()),
            "ci":ci,
            "decision":
                "A2_WIN" if ci[0]>0
                else "A2_LOSS" if ci[1]<0
                else "TIE"
        }

(RUN/"tree_a0_vs_a2_texture_eval.json").write_text(
    json.dumps(report,indent=2)+"\n"
)

print("\n===== TREE A0 VS A2 =====")
print("n_tracks =",len(ids))
print("\nEARLY")
for k,v in report["early"].items(): print(k,v)
print("\nLATE")
for k,v in report["late"].items(): print(k,v)
print("\nTREE_A0_VS_A2_TEXTURE_DONE")
