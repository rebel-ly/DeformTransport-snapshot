#!/usr/bin/env python3
"""Appearance-only ZERO62 diagnostic using frozen evaluator helpers; no subgroup TC-ME."""
import importlib.util, json
from pathlib import Path
import numpy as np

R=Path(__file__).parent; ROOT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport')
EP=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
F3=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze')
SIDE=ROOT/'server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
spec=importlib.util.spec_from_file_location('ev',EP); ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
M={'RW':ROOT/'server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/baseline/aligned_santa_baseline_seed0.mp4','DT-FULL':Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4'),'DROP-ZERO62':Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/wave2/20260814_082257/drop_zero62/santa_correct_v3d_seed000.mp4'),'WM-0':Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution/outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4')}
t=np.load(SIDE/'santa_material_tracks_correct.npy')[0].astype(np.float32); v=np.load(SIDE/'santa_material_visibility_correct.npy')[0].astype(bool); n=1257
src=ev.read_rgb_image(ROOT/ev.CASES['santa']['source']); c0=t[0]; sv=(c0[:,0]-3.5>=0)&(c0[:,0]+3.5<=831)&(c0[:,1]-3.5>=0)&(c0[:,1]+3.5<=479)
sp=np.full((n,8,8,3),np.nan,np.float32); sp[sv]=ev.sample_patches(src,c0[sv]); sl=np.full((n,3),np.nan,np.float32); sl[sv]=ev.patch_mean_lab(sp[sv])
zero=np.load(F3/'subgroups/zero_switch_positive_visible_mask.npy').astype(bool); comp=~zero
out={'subgroup_tcme_used_for_decision':False,'zero62_count':int(zero.sum()),'complement_count':int(comp.sum()),'methods':{}}
for name,p in M.items():
 z=ev.read_video_common(p); rows=[]
 for a in ev.ANCHORS:
  c=t[a].copy(); c[:,1]*=464/480
  ok=v[a]&sv&(c[:,0]-3.5>=0)&(c[:,0]+3.5<=831)&(c[:,1]-3.5>=0)&(c[:,1]+3.5<=463)&np.isfinite(c).all(1)
  ii=np.where(ok)[0]; rows.append((ii,np.linalg.norm(ev.patch_mean_lab(ev.sample_patches(z[a],c[ii]))-sl[ii],axis=1)))
 lab,counts=ev.aggregate(rows,n)
 def s(mask):
  x=lab[mask&(counts>0)]; return {'n':int(len(x)),'mean':float(np.mean(x)),'median':float(np.median(x)),'p95':float(np.percentile(x,95))}
 out['methods'][name]={'zero62_tc_mar_lab':s(zero),'complement1195_tc_mar_lab':s(comp)}
(R/'zero62_appearance_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
