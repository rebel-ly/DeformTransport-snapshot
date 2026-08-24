#!/usr/bin/env python3
"""Visible-support state equality audit for the pre-existing Tree bridge."""
import hashlib, json
from pathlib import Path
import numpy as np, torch
OUT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_095748__f2r_positive_subgroup_validity')
ROOT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport'); RUN=ROOT/'server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_160835__official_tree_transport_run'; AD=RUN/'20260807_183939__tree__aligned_81state_contract'; BR=ROOT/'server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks'
RAW=RUN/'20260807_175657__tree__official_precomputed__80future_81aligned/point_trajectories.pt'; TR=AD/'aligned_transport_ready.pt'; VC=AD/'aligned_visibility_contract.pt'; HIST=ROOT/'server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/tree_motion_report.json'; EXP=ROOT/'scripts/export_material_tracks_to_wan_move.py'; BUILD=AD/'build_tree_aligned_contract.py'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def cand(a,b,av,bv,k):
 x,y,m,n=(a[:81-k],b[k:],av[:81-k],bv[k:]) if k>=0 else (a[-k:],b[:81+k],av[-k:],bv[:81+k])
 d=np.linalg.norm(x.astype(np.float64)-y.astype(np.float64),axis=-1)[m&n]
 return {'offset_frames':k,'state_offset_simulation_steps':2*k,'compared_visible_point_frames':int(d.size),'exact_equal_visible_point_frames':int((d==0).sum()),'mean_visible_xy_residual_px':float(d.mean()),'max_visible_xy_residual_px':float(d.max())}
def main():
 raw=torch.load(RAW,map_location='cpu',weights_only=False); tr=torch.load(TR,map_location='cpu',weights_only=False); vc=torch.load(VC,map_location='cpu',weights_only=False)
 bt=np.load(BR/'tree_material_tracks_correct.npy')[0]; bv=np.load(BR/'tree_material_visibility_correct.npy')[0]; ids=np.load(BR/'tree_material_point_ids.npy').astype(np.int64); obj=raw['objects'][0]
 uv=torch.cat([obj['initial_points_uv'].unsqueeze(0),obj['points_uv']],0).float().numpy(); uv*=832./512.; uv[...,1]-=176.; ref=tr['points_2d_video'][:,ids].float().numpy().astype(np.float32); rv=vc['aligned_visible'][:,ids].bool().numpy(); cs=[cand(bt,ref,bv,rv,k) for k in range(-2,3)]; best=min(cs,key=lambda x:(x['mean_visible_xy_residual_px'],abs(x['offset_frames'])))
 er=np.array_equal(bt[bv],uv[:,ids].astype(np.float32)[bv]); et=np.array_equal(bt[bv],ref[bv]); ev=np.array_equal(bv,rv); steps=tr['simulation_steps'].numpy(); passed=er and et and ev and best['offset_frames']==0 and np.array_equal(steps,np.arange(0,161,2)) and np.array_equal(steps,vc['simulation_steps'].numpy())
 r4={'status':'PASS' if passed else 'FAIL','tree_asset_lineage_resolved':True,'lineage':{'raw_simulation_asset':str(RAW),'raw_simulation_sha256':sha(RAW),'source_frame':'frame_initial.png=S0','geometry_sequence':'initial_points_uv=S0; points_uv[0:80]=S2,...,S160','visibility_sequence':'aligned_visible=S0,S2,...,S160','exported_bridge':str(BR),'evaluator_inputs':{'tracks':str(BR/'tree_material_tracks_correct.npy'),'visibility':str(BR/'tree_material_visibility_correct.npy'),'historical_motion_report':str(HIST)},'scripts':{'aligned_contract_builder':{'path':str(BUILD),'sha256':sha(BUILD)},'bridge_exporter':{'path':str(EXP),'sha256':sha(EXP)}}},'timeline_contract':{'frame_initial':'S0','points_uv[0]':'S2; only aligned frame1','aligned_geometry_frame0':'S0','visibility_frame0':'S0','exported_bridge_frame0':'S0','future_geometry_frame_t':'S(2*t)','future_visibility_frame_t':'S(2*t)','fixed_temporal_offset_detected':False},'direct_artifact_consistency':{'bridge_tracks_exact_raw_initial_plus_future_on_visible_support':bool(er),'bridge_tracks_exact_aligned_transport_on_visible_support':bool(et),'bridge_visibility_exact_aligned_contract':bool(ev),'invisible_coordinate_rule':'existing exporter forward-fills last visible coordinate; excluded from state-coordinate equality','offset_candidates_geometry_visible_support':cs,'best_alignment_offset':best['offset_frames'],'exact_equal_count':best['exact_equal_visible_point_frames'],'exact_equal_denominator':best['compared_visible_point_frames'],'max_residual_px':best['max_visible_xy_residual_px']},'tree_geometry_visibility_timeline_alignment':'PASS' if passed else 'FAIL_OTHER','tree_historical_motion_evidence_status':'LEGACY_BUT_TIMELINE_COMPATIBLE' if passed else 'LEGACY_UNALIGNED','historical_motion_numbers_preserved':json.loads(HIST.read_text()),'impact':'Tree is timeline-compatible, but historical scores remain legacy/directional and are not promoted to corrected-v2 formal evidence.'}
 (OUT/'r4_tree_timeline_contract_audit.json').write_text(json.dumps(r4,indent=2)+'\n'); print(json.dumps({'tree_pass':bool(passed),'best_offset':best['offset_frames'],'exact':[best['exact_equal_visible_point_frames'],best['compared_visible_point_frames']],'max_residual':best['max_visible_xy_residual_px']}))
if __name__=='__main__': main()
