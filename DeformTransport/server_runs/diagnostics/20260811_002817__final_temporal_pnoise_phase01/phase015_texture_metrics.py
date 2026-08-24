import json, importlib.util
from pathlib import Path
import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    plt = None
    HAS_PLOT = False

ROOT=Path("/workspace/DeformTransport")
RUN=Path.cwd()
EVAL=ROOT/"server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/eval_v3.py"
SUITE=ROOT/"server_runs/wan_move_method_suite/20260810_054423__v3s_v3b_v3c_v3d_v3e_correct_seed0"

s=importlib.util.spec_from_file_location("ev",EVAL)
ev=importlib.util.module_from_spec(s); s.loader.exec_module(ev)

ANCH=list(range(4,81,4))
EARLY=list(range(4,41,4))
LATE=list(range(44,81,4))

def boot(x):
    return ev.bootstrap_mean_ci(np.asarray(x,np.float64))

def lab_patch(x):
    x=np.asarray(x,np.float32)
    if x.max()>1.5: x=x/255.0
    n,h,w,c=x.shape
    y=cv2.cvtColor(x.reshape(n*h,w,3),cv2.COLOR_RGB2LAB)
    return y.reshape(n,h,w,3)

def tex_errors(src,cur):
    a=lab_patch(src); b=lab_patch(cur)

    # RMS pixel-wise Lab discrepancy
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
        np.sqrt(
            (gxb-gxa)**2+
            (gyb-gya)**2
        ),
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

report={
 "protocol":"Phase-01.5 exact-support appearance/texture companions",
 "metrics":{
   "TC-MAR":"8x8 patch mean Lab L2; frozen metric",
   "TC-Patch-Lab":"RMS pixel-wise Lab distance over aligned 8x8 patches",
   "TC-Grad":"mean L-channel gradient-vector discrepancy over aligned 8x8 patches"
 },
 "cases":{}
}

OUT=RUN/"phase015_figures"
OUT.mkdir(exist_ok=True)

for case in ["santa","tree"]:
    cfg=ev.CASES[case]
    tr=np.load(ROOT/cfg["tracks"])[0].astype(np.float32)
    vis=np.load(ROOT/cfg["vis"])[0].astype(bool)

    ids=valid_all(tr,vis)
    if case=="tree":
        frozen=np.load(RUN/"tree_cumulative_drift_track_ids.npy").astype(np.int64)
        assert np.array_equal(ids,frozen), (len(ids),len(frozen))

    src_img=ev.read_rgb_image(ROOT/cfg["source"])
    src_patch=ev.sample_patches(src_img,tr[0,ids])
    src_mean=ev.patch_mean_lab(src_patch)

    paths={
      "rw":ROOT/cfg["rw"],
      "v3d":SUITE/case/"v3d"/f"{case}_v3d_correct_seed0.mp4"
    }

    errs={m:{k:{} for k in ["tcmar","patch","grad"]} for m in paths}

    for method,path in paths.items():
        print("LOAD",case,method,path,flush=True)
        vid=ev.read_video_common(path)

        for t in ANCH:
            xy=tr[t,ids].copy()
            xy[:,1]*=464.0/480.0
            p=ev.sample_patches(vid[t],xy)

            mean=ev.patch_mean_lab(p)
            errs[method]["tcmar"][t]=np.linalg.norm(
                mean-src_mean,axis=1
            ).astype(np.float64)

            patch,grad=tex_errors(src_patch,p)
            errs[method]["patch"][t]=patch
            errs[method]["grad"][t]=grad

    cr={"n_tracks":int(len(ids)),"anchors":{}}

    for t in ANCH:
        cr["anchors"][str(t)]={}
        for key,label in [
            ("tcmar","TC-MAR"),
            ("patch","TC-Patch-Lab"),
            ("grad","TC-Grad")
        ]:
            rw=errs["rw"][key][t]
            vv=errs["v3d"][key][t]
            d=rw-vv
            ci=boot(d)
            cr["anchors"][str(t)][label]={
              "rw":float(rw.mean()),
              "v3d":float(vv.mean()),
              "rw_minus_v3d":float(d.mean()),
              "ci":ci,
              "decision":"WIN" if ci[0]>0 else "LOSS" if ci[1]<0 else "TIE"
            }

    for win,ts in [("early",EARLY),("late",LATE)]:
        cr[win]={}
        for key,label in [
            ("tcmar","TC-MAR"),
            ("patch","TC-Patch-Lab"),
            ("grad","TC-Grad")
        ]:
            rw=np.stack([errs["rw"][key][t] for t in ts]).mean(0)
            vv=np.stack([errs["v3d"][key][t] for t in ts]).mean(0)
            d=rw-vv; ci=boot(d)
            cr[win][label]={
              "rw":float(rw.mean()),
              "v3d":float(vv.mean()),
              "rw_minus_v3d":float(d.mean()),
              "ci":ci,
              "decision":"WIN" if ci[0]>0 else "LOSS" if ci[1]<0 else "TIE"
            }

    report["cases"][case]=cr

    # publication figures
    if HAS_PLOT:
        for key,label,ylabel in [
            ("tcmar","TC-MAR","TC-MAR ↓"),
            ("patch","TC-Patch-Lab","TC-Patch-Lab ↓"),
            ("grad","TC-Grad","TC-Grad ↓")
        ]:
            fig,ax=plt.subplots(figsize=(3.35,2.35))
            ax.plot(ANCH,[errs["rw"][key][t].mean() for t in ANCH],marker="o",label="RealWonder")
            ax.plot(ANCH,[errs["v3d"][key][t].mean() for t in ANCH],marker="s",linestyle="--",label="V3D")
            ax.axvline(42,linestyle=":",linewidth=1)
            ax.set_xlabel("Prediction horizon (frame)")
            ax.set_ylabel(ylabel)
            ax.set_xticks([4,20,40,60,80])
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(OUT/f"{case}_{key}_exact_support.pdf",bbox_inches="tight")
            fig.savefig(OUT/f"{case}_{key}_exact_support.png",dpi=500,bbox_inches="tight")
            plt.close(fig)
    else:
        print("PLOTTING_SKIPPED: matplotlib unavailable", flush=True)

(RUN/"phase015_texture_metrics.json").write_text(json.dumps(report,indent=2)+"\n")

print("\n===== EXACT SUPPORT SUMMARY =====")
for c in ["santa","tree"]:
    print("\n",c,"n=",report["cases"][c]["n_tracks"])
    print("EARLY",report["cases"][c]["early"])
    print("LATE ",report["cases"][c]["late"])

print("\nSAVED phase015_texture_metrics.json")
