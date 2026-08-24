#!/usr/bin/env python3
"""DIAGNOSTIC_ONLY frozen TC-MAR/TC-ME on a lineage-qualified RGB frame directory."""
import argparse, importlib.util, json
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--evaluator',required=True); ap.add_argument('--frames',required=True)
 ap.add_argument('--root',required=True); ap.add_argument('--tracks',required=True); ap.add_argument('--visibility',required=True)
 ap.add_argument('--out',required=True); ap.add_argument('--batch',type=int,default=8); a=ap.parse_args()
 spec=importlib.util.spec_from_file_location('ev',a.evaluator); ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
 root=Path(a.root); files=sorted(Path(a.frames).glob('frame_*.png')); assert len(files)==81
 raw=[]
 for p in files:
  b=cv2.imread(str(p),cv2.IMREAD_COLOR); assert b is not None
  raw.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
 video=np.stack([ev.to_common(x) for x in raw]); assert video.shape==(81,464,832,3)
 tracks=np.load(a.tracks)[0].astype(np.float32); vis=np.load(a.visibility)[0].astype(bool); n=tracks.shape[1]
 source=ev.read_rgb_image(root/ev.CASES['santa']['source']); centers0=tracks[0]
 sv=(centers0[:,0]-3.5>=0)&(centers0[:,0]+3.5<=831)&(centers0[:,1]-3.5>=0)&(centers0[:,1]+3.5<=479)
 sp=np.full((n,8,8,3),np.nan,np.float32); good=np.where(sv)[0]; sp[good]=ev.sample_patches(source,centers0[good])
 sl=np.full((n,3),np.nan,np.float32); sl[good]=ev.patch_mean_lab(sp[good]); lr=[]; rr=[]; obs=0
 for t in ev.ANCHORS:
  c=tracks[t].copy(); c[:,1]*=464/480; fv=(c[:,0]-3.5>=0)&(c[:,0]+3.5<=831)&(c[:,1]-3.5>=0)&(c[:,1]+3.5<=463)
  valid=vis[t]&sv&fv&np.isfinite(c).all(1); ids=np.where(valid)[0]; obs+=len(ids); patch=ev.sample_patches(video[t],c[ids])
  lr.append((ids,np.linalg.norm(ev.patch_mean_lab(patch)-sl[ids],axis=1))); rr.append((ids,np.abs(patch-sp[ids]).mean(axis=(1,2,3))))
 lab,count=ev.aggregate(lr,n); rgb,count2=ev.aggregate(rr,n); assert np.array_equal(count,count2); valid=count>0
 refs=[]
 for t in range(80):
  ok=vis[t]&vis[t+1]&np.isfinite(tracks[t]).all(1)&np.isfinite(tracks[t+1]).all(1); ids=np.where(ok)[0]; c=tracks[t,ids]/2; ref=(tracks[t+1,ids]-tracks[t,ids])/2
  inbound=(c[:,0]>=0)&(c[:,0]<=415)&(c[:,1]>=0)&(c[:,1]<=239); refs.append((c[inbound],ref[inbound]))
 model,transforms=ev.load_raft_cached(torch.device('cuda:0')); vals=[]
 with torch.inference_mode():
  for start in range(0,80,a.batch):
   end=min(80,start+a.batch); x=torch.from_numpy(video[start:end]).permute(0,3,1,2).cuda(); y=torch.from_numpy(video[start+1:end+1]).permute(0,3,1,2).cuda()
   x=F.interpolate(x,size=(240,416),mode='area'); y=F.interpolate(y,size=(240,416),mode='area'); x,y=transforms(x,y); pred=model(x,y)[-1].float().cpu().numpy()
   for j,t in enumerate(range(start,end)):
    c,ref=refs[t]; vals.append(float(np.linalg.norm(ev.bilinear_flow(pred[j],c)-ref,axis=1).mean()))
 vals=np.asarray(vals,np.float64)
 report={'status':'DIAGNOSTIC_ONLY','representation':'81 lineage-qualified RGB simulation/coarse frames; frozen evaluator mapping','frame_count':81,'decoded_shape':list(video.shape),'valid_tracks':int(valid.sum()),'valid_anchor_observations':obs,
  'tc_mar_lab':ev.stats(lab[valid]),'tc_mar_rgb_l1':ev.stats(rgb[valid]),
  'tc_me':{'mean':float(vals.mean()),'median':float(np.median(vals)),'p95':float(np.percentile(vals,95))}}
 Path(a.out).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
