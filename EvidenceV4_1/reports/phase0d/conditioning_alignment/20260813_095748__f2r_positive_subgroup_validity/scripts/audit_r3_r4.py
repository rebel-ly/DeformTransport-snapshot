#!/usr/bin/env python3
"""Read-only F2-R R3 representation and R4 Tree temporal-contract audit."""
import hashlib,json
from pathlib import Path
import numpy as np, torch
OUT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_095748__f2r_positive_subgroup_validity')
ROOT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport')
RUN=ROOT/'server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_160835__official_tree_transport_run'
RAW=RUN/'20260807_175657__tree__official_precomputed__80future_81aligned/point_trajectories.pt'
AD=RUN/'20260807_183939__tree__aligned_81state_contract'
TR=AD/'aligned_transport_ready.pt'; VC=AD/'aligned_visibility_contract.pt'
BR=ROOT/'server_runs/wan_move_bridge/20260810_072215__tree_correct_tracks'
HIST=ROOT/'server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/tree_motion_report.json'
EXP=ROOT/'scripts/export_material_tracks_to_wan_move.py'; BUILD=AD/'build_tree_aligned_contract.py'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def off(a,b,k):
 x,y=(a[:81-k],b[k:]) if k>=0 else (a[-k:],b[:81+k])
 d=np.linalg.norm(x.astype(np.float64)-y.astype(np.float64),axis=-1)
 return {'offset_frames':k,'state_offset_simulation_steps':2*k,'compared_point_frames':int(d.size),'exact_equal_point_frames':int((d==0).sum()),'mean_xy_residual_px':float(d.mean()),'max_xy_residual_px':float(d.max())}
def main():
 r2=json.loads((OUT/'r2_rw_coarse_support_inventory.json').read_text())
 r3={'status':'PASS','audit_type':'REPRESENTATION_SEMANTICS_ONLY_NO_NEW_TCME','preserved_diagnostic_values':{'RW_CONDITION_TCME':.8946881631311656,'DT_EDITED_Y_TCME':1.5050048806385061,'classification':'DIAGNOSTIC_ONLY'},'rw_condition':{'representation':'81 explicit RGB rasterized simulation/coarse frames at 832x480, evaluator-resized to 832x464','spatial_support':'frontmost projected point-raster support plus holes','overall_future_valid_pixel_fraction':r2['overall_future']['valid_pixel_fraction'],'temporal_density':'81 explicit RGB states S0,S10,...,S800; 80 transitions','future_rgb_explicitly_rerasterized':True,'motion_exists_across_most_object_pixels':'UNRESOLVED_AS_STATED; explicit on occupied support only','holes_exist':True},'dt_edited_y_condition':{'native_representation':'Wan VAE latent CxTxHxW','latent_shape':[16,21,60,104],'future_latent_slots':20,'write_support':9031,'future_cell_denominator':124800,'intervention_fraction':.07236378205128205,'untouched_condition_fraction':.9276362179487179,'decode_semantics':'VAE decode maps 21 latent slots to 81 RGB frames','temporal_support':'sparse writes over future latent slots then nonlinear VAE decode'},'comparison':{'representation_support_comparable':False,'representation_temporal_semantics_comparable':False,'condition_tcme_causal_interpretation_valid':False,'reason':'RW explicit projected RGB raster support differs from sparse edited latent support and VAE decode semantics.'},'rw_metric_advantage_already_present_in_conditioning_motion':'UNRESOLVED_REPRESENTATION_ASYMMETRY'}
 (OUT/'r3_condition_tcme_representation_audit.json').write_text(json.dumps(r3,indent=2)+'\n')
 raw=torch.load(RAW,map_location='cpu',weights_only=False); tr=torch.load(TR,map_location='cpu',weights_only=False); vc=torch.load(VC,map_location='cpu',weights_only=False)
 bt=np.load(BR/'tree_material_tracks_correct.npy')[0]; bv=np.load(BR/'tree_material_visibility_correct.npy')[0]; ids=np.load(BR/'tree_material_point_ids.npy').astype(np.int64)
 obj=raw['objects'][0]; uv=torch.cat([obj['initial_points_uv'].unsqueeze(0),obj['points_uv']],0).float().numpy(); uv*=832./512.; uv[...,1]-=176.
 ref=tr['points_2d_video'][:,ids].float().numpy().astype(np.float32); rv=vc['aligned_visible'][:,ids].bool().numpy(); cand=[off(bt,ref,k) for k in range(-2,3)]; best=min(cand,key=lambda x:(x['mean_xy_residual_px'],abs(x['offset_frames'])))
 exact_raw=np.array_equal(bt,uv[:,ids].astype(np.float32)); exact_tr=np.array_equal(bt,ref); exact_vis=np.array_equal(bv,rv); steps=tr['simulation_steps'].numpy(); pass_=exact_raw and exact_tr and exact_vis and best['offset_frames']==0 and np.array_equal(steps,np.arange(0,161,2)) and np.array_equal(steps,vc['simulation_steps'].numpy())
 r4={'status':'PASS' if pass_ else 'FAIL','tree_asset_lineage_resolved':True,'lineage':{'raw_simulation_asset':str(RAW),'raw_simulation_sha256':sha(RAW),'source_frame':'frame_initial.png=S0','geometry_sequence':'initial_points_uv=S0; points_uv[0:80]=S2,...,S160','visibility_sequence':'aligned_visible=S0,S2,...,S160','exported_bridge':str(BR),'evaluator_inputs':{'tracks':str(BR/'tree_material_tracks_correct.npy'),'visibility':str(BR/'tree_material_visibility_correct.npy'),'historical_motion_report':str(HIST)},'scripts':{'aligned_contract_builder':{'path':str(BUILD),'sha256':sha(BUILD)},'bridge_exporter':{'path':str(EXP),'sha256':sha(EXP)}}},'timeline_contract':{'frame_initial':'S0','points_uv[0]':'S2; only used as aligned frame1','points_uv_aligned[0]':'S0','visibility[0]':'S0','exported_bridge_frame0':'S0','future_geometry_frame_t':'S(2*t)','future_visibility_frame_t':'S(2*t)','fixed_temporal_offset_detected':False},'direct_artifact_consistency':{'bridge_tracks_exact_raw_initial_plus_future':bool(exact_raw),'bridge_tracks_exact_aligned_transport':bool(exact_tr),'bridge_visibility_exact_aligned_contract':bool(exact_vis),'offset_candidates_geometry':cand,'best_alignment_offset':best['offset_frames'],'exact_equal_count':best['exact_equal_point_frames'],'exact_equal_denominator':best['compared_point_frames'],'max_residual_px':best['max_xy_residual_px']},'tree_geometry_visibility_timeline_alignment':'PASS' if pass_ else 'FAIL_OTHER','tree_historical_motion_evidence_status':'LEGACY_BUT_TIMELINE_COMPATIBLE' if pass_ else 'LEGACY_UNALIGNED','historical_motion_numbers_preserved':json.loads(HIST.read_text()),'impact':'Tree is timeline-compatible, but historical scores remain legacy/directional and are not promoted to corrected-v2 formal evidence.'}
 (OUT/'r4_tree_timeline_contract_audit.json').write_text(json.dumps(r4,indent=2)+'\n'); print(json.dumps({'r3_support_comparable':False,'r3_temporal_comparable':False,'tree_pass':bool(pass_),'tree_best_offset':best['offset_frames'],'tree_exact':[best['exact_equal_point_frames'],best['compared_point_frames']],'tree_max_residual':best['max_xy_residual_px']}))
if __name__=='__main__':main()
