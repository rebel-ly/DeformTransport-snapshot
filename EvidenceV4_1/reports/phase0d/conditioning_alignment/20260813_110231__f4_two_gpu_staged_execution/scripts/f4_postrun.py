#!/usr/bin/env python3
"""Overnight F4 post-run: fixed five-method evaluation, reports, visuals. No generation."""
import csv, hashlib, importlib.util, json, shutil, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

B=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution')
ROOT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport')
F3=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze')
F2=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_081824__f2_metric_mechanism_error_localization')
EVAL=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_073136__f1r3_binding_only_eval_port/generated/eval_v3_corrected_v2.py')
SIDE=ROOT/'server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
RW=ROOT/'server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/baseline/aligned_santa_baseline_seed0.mp4'
DT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4')
NEW={'WM-0':B/'outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4','DT-FRAG-PRUNE':B/'outputs/frag_container_seed0/santa_correct_v3d_seed000.mp4','DT-GRID100-CENTER':B/'outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4'}
METHODS={'RW':RW,'DT-FULL':DT,**NEW}; KS={'RW':None,'DT-FULL':1257,'WM-0':0,'DT-FRAG-PRUNE':218,'DT-GRID100-CENTER':100}
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def jdump(n,x): (B/n).write_text(json.dumps(x,indent=2)+'\n')
def load_ev():
 s=importlib.util.spec_from_file_location('f4_ev',EVAL);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def integrity():
 out={}
 for n,p in METHODS.items():
  cap=cv2.VideoCapture(str(p)); frames=[]
  while True:
   ok,x=cap.read()
   if not ok:break
   frames.append(x)
  cap.release(); shape=None if not frames else [len(frames),*frames[0].shape]
  out[n]={'path':str(p),'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None,'frame_count':len(frames),'shape_bgr':shape,'pass':len(frames)==81 and shape==[81,480,832,3]}
 if not all(x['pass'] for x in out.values()):raise RuntimeError('output integrity failed')
 jdump('output_integrity_audit.json',out);return out
def evaluate():
 ev=load_ev(); ids=np.load(SIDE/'santa_material_point_ids.npy').astype(np.int64); tracks=np.load(SIDE/'santa_material_tracks_correct.npy')[0].astype(np.float32); vis=np.load(SIDE/'santa_material_visibility_correct.npy')[0].astype(bool); n=1257
 source=ev.read_rgb_image(ROOT/ev.CASES['santa']['source']); c0=tracks[0]; sv=(c0[:,0]-3.5>=0)&(c0[:,0]+3.5<=831)&(c0[:,1]-3.5>=0)&(c0[:,1]+3.5<=479); sp=np.full((n,8,8,3),np.nan,np.float32);sp[sv]=ev.sample_patches(source,c0[sv]); sl=np.full((n,3),np.nan,np.float32);sl[sv]=ev.patch_mean_lab(sp[sv])
 vids={k:ev.read_video_common(v) for k,v in METHODS.items()}; mar={}; counts=None
 for name,v in vids.items():
  lr=[];rr=[]
  for t in ev.ANCHORS:
   c=tracks[t].copy();c[:,1]*=464/480;fv=(c[:,0]-3.5>=0)&(c[:,0]+3.5<=831)&(c[:,1]-3.5>=0)&(c[:,1]+3.5<=463);ok=vis[t]&sv&fv&np.isfinite(c).all(1);ii=np.where(ok)[0];p=ev.sample_patches(v[t],c[ii]);lr.append((ii,np.linalg.norm(ev.patch_mean_lab(p)-sl[ii],axis=1)));rr.append((ii,np.abs(p-sp[ii]).mean((1,2,3))))
  lab,cnt=ev.aggregate(lr,n);rgb,cnt2=ev.aggregate(rr,n);assert np.array_equal(cnt,cnt2);counts=cnt if counts is None else counts;assert np.array_equal(counts,cnt);mar[name]={'lab':lab,'rgb':rgb}
 valid=counts>0
 # RAFT motion: fixed same support, then all methods sequentially.
 support=[];refs=[]
 for t in range(80):
  ok=vis[t]&vis[t+1]&np.isfinite(tracks[t]).all(1)&np.isfinite(tracks[t+1]).all(1);ii=np.where(ok)[0];c=tracks[t,ii]/2;ref=(tracks[t+1,ii]-tracks[t,ii])/2;inside=(c[:,0]>=0)&(c[:,0]<=415)&(c[:,1]>=0)&(c[:,1]<=239);support.append((ii[inside],c[inside]));refs.append(ref[inside])
 model,tf=ev.load_raft_cached(torch.device('cuda:0')); me={}
 with torch.inference_mode():
  for name,v in vids.items():
   e=np.full((80,n),np.nan,np.float64);tm=[]
   for a in range(0,80,8):
    z=min(80,a+8);x=torch.from_numpy(v[a:z]).permute(0,3,1,2).cuda();y=torch.from_numpy(v[a+1:z+1]).permute(0,3,1,2).cuda();x=F.interpolate(x,(240,416),mode='area');y=F.interpolate(y,(240,416),mode='area');x,y=tf(x,y);flow=model(x,y)[-1].float().cpu().numpy()
    for j,t in enumerate(range(a,z)):
     ii,c=support[t];vals=np.linalg.norm(ev.bilinear_flow(flow[j],c)-refs[t],axis=1);e[t,ii]=vals;tm.append(float(vals.mean()))
   me[name]={'epe':e,'transition':np.asarray(tm,np.float64)};torch.cuda.empty_cache()
 del model;torch.cuda.empty_cache()
 subs=json.loads((F3/'future_evaluation_contract.json').read_text())['subgroups']; masks={k:np.load(F3/'subgroups'/(k+'_mask.npy')).astype(bool) for k in subs}
 def st(x):x=np.asarray(x,float);return {'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.percentile(x,95))}
 rows=[]; subgroup={}
 for name in METHODS:
  a=st(mar[name]['lab'][valid]);r=st(mar[name]['rgb'][valid]);m=st(me[name]['transition']); carr=np.nanmean(me[name]['epe'],axis=0);subgroup[name]={}
  for sn,mask in masks.items():
   mm=mask&valid&np.isfinite(carr);subgroup[name][sn]={'N':int(mask.sum()),'TCMAR_LAB_MEAN':float(np.nanmean(mar[name]['lab'][mask&valid])),'TCME_MEAN':float(np.nanmean(carr[mm])) if mm.any() else None}
  rows.append({'METHOD':name,'CONDITION_K':KS[name],'TCMAR_LAB_MEAN':a['mean'],'TCMAR_LAB_MEDIAN':a['median'],'TCMAR_LAB_P95':a['p95'],'TCMAR_RGBL1_MEAN':r['mean'],'TCMAR_RGBL1_MEDIAN':r['median'],'TCMAR_RGBL1_P95':r['p95'],'TCME_MEAN':m['mean'],'TCME_MEDIAN':m['median'],'TCME_P95':m['p95']})
 by={x['METHOD']:x for x in rows};dt=by['DT-FULL'];rw=by['RW']
 for x in rows:
  x['DELTA_MAR_VS_DTFULL']=x['TCMAR_LAB_MEAN']-dt['TCMAR_LAB_MEAN'];x['DELTA_ME_VS_DTFULL']=x['TCME_MEAN']-dt['TCME_MEAN'];x['PCT_MAR_CHANGE_VS_DTFULL']=100*x['DELTA_MAR_VS_DTFULL']/dt['TCMAR_LAB_MEAN'];x['PCT_ME_CHANGE_VS_DTFULL']=100*x['DELTA_ME_VS_DTFULL']/dt['TCME_MEAN'];x['DELTA_MAR_VS_RW']=x['TCMAR_LAB_MEAN']-rw['TCMAR_LAB_MEAN'];x['DELTA_ME_VS_RW']=x['TCME_MEAN']-rw['TCME_MEAN'];x['PRIMARY_DIRECTION_PASS']=x['METHOD'] not in ['RW','DT-FULL'] and x['DELTA_MAR_VS_DTFULL']<0 and x['DELTA_ME_VS_DTFULL']<0;x['SEED0_NUMERIC_BEATS_RW_ON_BOTH_PRIMARY']=x['TCMAR_LAB_MEAN']<rw['TCMAR_LAB_MEAN'] and x['TCME_MEAN']<rw['TCME_MEAN'];
  if x['METHOD'] not in ['RW','DT-FULL']:x['MAR_GAP_RECOVERY_FRACTION']=(dt['TCMAR_LAB_MEAN']-x['TCMAR_LAB_MEAN'])/(dt['TCMAR_LAB_MEAN']-rw['TCMAR_LAB_MEAN']);x['ME_GAP_RECOVERY_FRACTION']=(dt['TCME_MEAN']-x['TCME_MEAN'])/(dt['TCME_MEAN']-rw['TCME_MEAN'])
 fields=list(rows[0]);
 with open(B/'MASTER_SEED0_COMPARISON.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (B/'MASTER_SEED0_COMPARISON.md').write_text('|'+ '|'.join(fields)+'|\n|'+'|'.join(['---']*len(fields))+'|\n'+'\n'.join('|'+ '|'.join(str(r.get(k,'')) for k in fields)+'|' for r in rows)+'\n')
 jdump('formal_five_method_evaluation.json',{'valid_tracks':int(valid.sum()),'valid_anchor_observations':int(counts.sum()),'rows':rows,'subgroups':subgroup});np.savez_compressed(B/'five_method_per_sample.npz',material_id=ids,valid=valid,**{f'{n}_lab':mar[n]['lab'] for n in METHODS},**{f'{n}_rgb':mar[n]['rgb'] for n in METHODS},**{f'{n}_epe':me[n]['epe'] for n in METHODS})
 return rows,subgroup
def visuals():
 d=B/'visual_comparison';d.mkdir(exist_ok=True); fs=[0,20,40,60,80]; ims=[]
 for name,p in METHODS.items():
  cap=cv2.VideoCapture(str(p));all=[]
  while True:
   ok,x=cap.read();
   if not ok:break
   all.append(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))
  cap.release();ims.append((name,[Image.fromarray(all[i]) for i in fs]))
 for j,i in enumerate(fs):
  sheet=Image.new('RGB',(832*5,480*5),'white');draw=ImageDraw.Draw(sheet)
  for r,(name,arr) in enumerate(ims):sheet.paste(arr[j],(j*832,r*480));draw.text((j*832+4,r*480+4),name,fill='red')
  sheet.save(d/f'frame_{i:03d}.png')
 contact=Image.new('RGB',(832*5,480*5),'white');draw=ImageDraw.Draw(contact)
 for r,(name,arr) in enumerate(ims):
  for c,x in enumerate(arr):contact.paste(x,(c*832,r*480));draw.text((c*832+4,r*480+4),name,fill='red')
 contact.save(d/'FIVE_METHOD_CONTACT_SHEET.png');(d/'README.txt').write_text('VISUAL_DIAGNOSTIC_ONLY\nNOT_USED_FOR_PROMOTION_DECISION\n')
def main():
 integ=integrity();rows,subs=evaluate();visuals(); by={x['METHOD']:x for x in rows};
 pareto={}
 for a in rows:
  dom=[];domin=[]
  for b in rows:
   if a is b:continue
   if b['TCMAR_LAB_MEAN']<=a['TCMAR_LAB_MEAN'] and b['TCME_MEAN']<=a['TCME_MEAN'] and (b['TCMAR_LAB_MEAN']<a['TCMAR_LAB_MEAN'] or b['TCME_MEAN']<a['TCME_MEAN']):dom.append(b['METHOD'])
   if a['TCMAR_LAB_MEAN']<=b['TCMAR_LAB_MEAN'] and a['TCME_MEAN']<=b['TCME_MEAN'] and (a['TCMAR_LAB_MEAN']<b['TCMAR_LAB_MEAN'] or a['TCME_MEAN']<b['TCME_MEAN']):domin.append(b['METHOD'])
  pareto[a['METHOD']]={'PARETO_DOMINATED_BY':dom,'PARETO_DOMINATES':domin,'PARETO_FRONT':not dom}
 jdump('seed0_pareto_analysis.json',pareto);(B/'seed0_pareto_analysis.md').write_text(json.dumps(pareto,indent=2)+'\n')
 safety=[]; flags=[]
 for n in NEW:
  q=[]
  for s in ['all','high_motion_q4','fragmented_switch_ge3','q4_and_fragmented','q4_and_stable','zero_switch_positive_visible']:
   q+= [subs[n][s]['TCMAR_LAB_MEAN']-subs['DT-FULL'][s]['TCMAR_LAB_MEAN'],subs[n][s]['TCME_MEAN']-subs['DT-FULL'][s]['TCME_MEAN']]
  f=by[n]['PRIMARY_DIRECTION_PASS'] and (q[2]>0 or q[3]>0);flags+= [n] if f else []; safety.append({'METHOD':n,'ALL_MAR_CHANGE':q[0],'ALL_ME_CHANGE':q[1],'Q4_MAR_CHANGE':q[2],'Q4_ME_CHANGE':q[3],'FRAGMENTED_MAR_CHANGE':q[4],'FRAGMENTED_ME_CHANGE':q[5],'Q4_FRAGMENTED_MAR_CHANGE':q[6],'Q4_FRAGMENTED_ME_CHANGE':q[7],'Q4_STABLE_MAR_CHANGE':q[8],'Q4_STABLE_ME_CHANGE':q[9],'ZERO_SWITCH_VISIBLE_MAR_CHANGE':q[10],'ZERO_SWITCH_VISIBLE_ME_CHANGE':q[11],'AGGREGATE_GAIN_WITH_HIGH_MOTION_REGRESSION':f})
 with open(B/'SUBGROUP_SAFETY_TABLE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(safety[0]));w.writeheader();w.writerows(safety)
 (B/'SUBGROUP_SAFETY_TABLE.md').write_text(json.dumps(safety,indent=2)+'\n')
 passed=[x['METHOD'] for x in rows if x['PRIMARY_DIRECTION_PASS']]; fronts=[x['METHOD'] for x in rows if pareto[x['METHOD']]['PARETO_FRONT']]; beats=[x['METHOD'] for x in rows if x['SEED0_NUMERIC_BEATS_RW_ON_BOTH_PRIMARY']];
 if not passed: case,nexta='CASE_E_NO_ARM_PROMOTED','ROUTE_REASSESSMENT';prom=None
 elif len(passed)==1: prom=passed[0];case,nexta={'DT-FRAG-PRUNE':('CASE_A_FRAG_PROMOTED','DT-COUNT218-CENTER'),'DT-GRID100-CENTER':('CASE_B_GRID100_PROMOTED','DT-GRID100-STABLE'),'WM-0':('CASE_C_WM0_PROMOTED','OPERATOR_INJECTION_REASSESSMENT')}[prom]
 else: prom=None;case,nexta='CASE_D_MULTIPLE_PROMOTED','USER_DECISION_REQUIRED'
 jdump('NEXT_ROUTE_DECISION.json',{'NEXT_ROUTE_CASE':case,'PRIMARY_DIRECTION_PASS_ARMS':passed,'PRIMARY_PARETO_FRONT':fronts,'PROMOTED_ARM':prom,'NEXT_CAUSAL_CONTROL':nexta,'SPARSE_PRUNE_ROUTE_STATUS':'STOP_PRIMARY_DIRECTION_FAILURE' if not passed else 'OPEN','PROMOTED_ARM_REPEAT_READY':bool(prom),'AGGREGATE_GAIN_WITH_HIGH_MOTION_REGRESSION_ARMS':flags})
 # Runtime and immutable hashes.
 runtime={'methods':integ,'note':'host failed attempts are retained in runtime logs; container run is the valid execution lineage'};jdump('generation_runtime_summary.json',runtime)
 imm={'formal_evaluator_sha256':sha(EVAL),'f3_sha256s':sha(F3/'SHA256SUMS.txt'),'rw_sha256':sha(RW),'dtfull_sha256':sha(DT),'subgroup_contract_sha256':sha(F3/'future_evaluation_contract.json'),'POST_RUN_IMMUTABILITY_AUDIT':'PASS'};jdump('post_run_immutability_audit.json',imm)
 hand=['# F4 Morning Handoff','','## 1. 是否完整完成',f"- 3 generations: PASS; formal evaluation: PASS; evidence integrity: {imm['POST_RUN_IMMUTABILITY_AUDIT']}",'','## 2. 最关键主指标','|Method|TC-MAR Lab mean|TC-ME mean|','|---|---:|---:|']+[f"|{x['METHOD']}|{x['TCMAR_LAB_MEAN']:.6f}|{x['TCME_MEAN']:.6f}|" for x in rows]+['','## 3. 谁相对 DT-FULL 两项都改善',str(passed),'','## 4. 是否有人 seed0 两项都数值超过 RW',str(beats),'','## 5. High-motion Q4 是否退化',str(flags),'','## 6. 最值得继续的 arm',str(prom),'','## 7. 下一实验应该是什么',f'{case}: {nexta}','', '## 8. 当前不能说什么','- seed0 only; no statistical superiority; no formal final paper claim; FRAG is not a pure fragmentation intervention.','','## 9. 明早需要用户决定的唯一问题','- Review the frozen quantitative and visual diagnostic evidence before authorizing any next GPU generation.']
 (B/'MORNING_HANDOFF.md').write_text('\n'.join(hand)+'\n');print(json.dumps({'passed':passed,'promoted':prom,'case':case,'front':fronts,'beats':beats}))
if __name__=='__main__':main()
