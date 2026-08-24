#!/usr/bin/env python3
"""F4-R3 subgroup diagnostics using frozen evaluator helpers; formal JSON remains primary."""
import importlib.util,json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
R=Path(__file__).parent; B=R.parents[2]; ROOT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport'); F3=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze')
EP=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
spec=importlib.util.spec_from_file_location('ev',EP);ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
SIDE=ROOT/'server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
M={'RW':ROOT/'server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/baseline/aligned_santa_baseline_seed0.mp4','DT-FULL':Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4'),'WM-0':B/'outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4','DT-FRAG-PRUNE':B/'outputs/frag_container_seed0/santa_correct_v3d_seed000.mp4','DT-GRID100-CENTER':B/'outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4'}
t=np.load(SIDE/'santa_material_tracks_correct.npy')[0].astype('float32');v=np.load(SIDE/'santa_material_visibility_correct.npy')[0].astype(bool);n=1257
src=ev.read_rgb_image(ROOT/ev.CASES['santa']['source']); c0=t[0]; sv=(c0[:,0]-3.5>=0)&(c0[:,0]+3.5<=831)&(c0[:,1]-3.5>=0)&(c0[:,1]+3.5<=479); sp=np.full((n,8,8,3),np.nan,'float32');sp[sv]=ev.sample_patches(src,c0[sv]);sl=np.full((n,3),np.nan,'float32');sl[sv]=ev.patch_mean_lab(sp[sv])
vid={k:ev.read_video_common(p) for k,p in M.items()}; labs={}; counts=None
for name,z in vid.items():
 rows=[]
 for a in ev.ANCHORS:
  c=t[a].copy();c[:,1]*=464/480;ok=v[a]&sv&(c[:,0]-3.5>=0)&(c[:,0]+3.5<=831)&(c[:,1]-3.5>=0)&(c[:,1]+3.5<=463)&np.isfinite(c).all(1);ii=np.where(ok)[0]; rows.append((ii,np.linalg.norm(ev.patch_mean_lab(ev.sample_patches(z[a],c[ii]))-sl[ii],axis=1)))
 labs[name],cnt=ev.aggregate(rows,n); counts=cnt if counts is None else counts
support=[];ref=[]
for a in range(80):
 ok=v[a]&v[a+1]&np.isfinite(t[a]).all(1)&np.isfinite(t[a+1]).all(1);ii=np.where(ok)[0];c=t[a,ii]/2;r=(t[a+1,ii]-t[a,ii])/2;inside=(c[:,0]>=0)&(c[:,0]<=415)&(c[:,1]>=0)&(c[:,1]<=239);support.append((ii[inside],c[inside]));ref.append(r[inside])
model,tf=ev.load_raft_cached(torch.device('cuda:0')); epe={}; tm={}
with torch.inference_mode():
 for name,z in vid.items():
  E=np.full((80,n),np.nan);q=[]
  for lo in range(0,80,8):
   hi=min(lo+8,80);x=torch.from_numpy(z[lo:hi]).permute(0,3,1,2).cuda();y=torch.from_numpy(z[lo+1:hi+1]).permute(0,3,1,2).cuda();x=F.interpolate(x,(240,416),mode='area');y=F.interpolate(y,(240,416),mode='area');x,y=tf(x,y);flow=model(x,y)[-1].float().cpu().numpy()
   for j,a in enumerate(range(lo,hi)):
    ii,c=support[a];qv=np.linalg.norm(ev.bilinear_flow(flow[j],c)-ref[a],axis=1);E[a,ii]=qv;q.append(float(qv.mean()))
  epe[name]=E;tm[name]=q
masks={x:np.load(F3/'subgroups'/(x+'_mask.npy')).astype(bool) for x in ['all','high_motion_q4','fragmented_switch_ge3','q4_and_fragmented','q4_and_stable','zero_switch_positive_visible']}
out={'formal_all_diagnostic':{},'subgroups':{}}
for name in M:
 out['formal_all_diagnostic'][name]={'TCMAR_LAB_MEAN':float(np.nanmean(labs[name])),'TCME_TRANSITION_MEAN':float(np.mean(tm[name]))}
 out['subgroups'][name]={}
 for sn,mask in masks.items():
  carr=np.nanmean(epe[name],axis=0); good=mask&np.isfinite(carr)&(counts>0)
  out['subgroups'][name][sn]={'N':int(mask.sum()),'TCMAR_LAB_MEAN':float(np.nanmean(labs[name][mask&(counts>0)])),'TCME_MEAN':float(np.nanmean(carr[good])) if good.any() else None}
(R/'frozen_subgroup_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
