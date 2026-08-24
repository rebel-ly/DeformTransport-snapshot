#!/usr/bin/env python3
"""CPU-only R3 real-grid structural audit driven by current frozen trajectory functions."""
import json,os,sys,importlib.util
from pathlib import Path
import numpy as np,torch
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/post_sc_formal_closure/formal_geometry_counterfactual_20260818T185831Z'); ROOT=Path('/workspace/DeformTransport_EvidenceV4_1'); OVER=ROOT/'experimental/20260814__wanmove_preview_sdedit_overlay'
os.environ['DT_TRANSPORT_VARIANT']='v3d';os.environ['DT_TRACK_IDS_PATH']='/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy';os.environ['DT_TRACK_DEPTH_PATH']='/workspace/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy'
spec=importlib.util.spec_from_file_location('r3_frozen_trajectory',OVER/'wan/modules/trajectory.py');tr=importlib.util.module_from_spec(spec);spec.loader.exec_module(tr)
track=torch.from_numpy(np.load('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy'))
vis=torch.from_numpy(np.load('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy'))
if track.ndim==4:track=track[0]
if vis.ndim==3:vis=vis[0]
track=track.clone();track[...,1]*=464/480
def audit(seed):
 torch.manual_seed(seed); _,tp=tr.create_pos_feature_map(track,vis,(4,8,8),464,832,16,track_num=vis.size(-1),device=torch.device('cpu'))
 c=tr._DT_CONTEXT; sv=tp[:,0]; svalid=(sv[:,0]>=0)&(sv[:,1]>=0); vs=c['visibility'][::4]; dep=c['depth'][::4]; ids=c['ids']; groups={}
 for tau in range(1,tp.shape[1]):
  tv=tp[:,tau]; valid=svalid&(tv[:,0]>=0)&(tv[:,1]>=0)&(dep[tau]>0)&torch.isfinite(dep[tau])
  for i in torch.where(valid)[0].tolist(): groups.setdefault((tau,int(tv[i,0]),int(tv[i,1])),[]).append(i)
 winners={k:(int(ids[min(v,key=lambda i:(float(dep[k[0],i]),int(ids[i]))) ]),float(min(float(dep[k[0],i]) for i in v))) for k,v in groups.items()}
 hist={};
 for v in groups.values():hist[str(len(v))]=hist.get(str(len(v)),0)+1
 y=torch.randn(16,21,58,104); y0=y.clone(); edited=tr.replace_feature(y.unsqueeze(0),tp.unsqueeze(0))[0]; delta=(edited-y0).abs(); support=set(map(tuple,torch.nonzero(delta.sum(0)>0,as_tuple=False).tolist()))
 counts={int(i):0 for i in ids.tolist()}
 for _,(i,_) in winners.items():counts[i]+=1
 return {'groups':groups,'winners':winners,'support':support,'counts':counts,'candidate':sum(map(len,groups.values())),'hist':hist}
a,b=audit(0),audit(1)
def keys(x):return set(x['groups'])
def col(x):return {k:len(v) for k,v in x['groups'].items() if len(v)>1}
res={'runtime_grid':[16,21,58,104],'total_future_cells':20*58*104,'aggregates':{'TOTAL_CANDIDATE_ASSIGNMENTS':a['candidate'],'TOTAL_UNIQUE_TARGET_WRITES':len(a['winners']),'TOTAL_COLLISION_CELLS':len(col(a)),'TOTAL_COLLISION_CARRIERS':sum(len(v)-1 for v in a['groups'].values() if len(v)>1),'GLOBAL_MAX_MULTIPLICITY':max(map(len,a['groups'].values())),'CARRIERS_WITH_ANY_CONTRIBUTION':sum(v>0 for v in a['counts'].values()),'ZERO_CONTRIBUTION_CARRIERS':sum(v==0 for v in a['counts'].values()),'OBSERVED_DIFF_SUPPORT':len(a['support'])},'correct_vs_shuffled_invariants':{'target_cell_structure_mismatch':len(keys(a)^keys(b)),'collision_structure_mismatch':sum(col(a).get(k)!=col(b).get(k) for k in keys(a)|keys(b)),'winner_id_mismatch':sum(a['winners'].get(k,(None,None))[0]!=b['winners'].get(k,(None,None))[0] for k in keys(a)|keys(b)),'winner_depth_mismatch':sum(a['winners'].get(k,(None,None))[1]!=b['winners'].get(k,(None,None))[1] for k in keys(a)|keys(b)),'per_carrier_contribution_mismatch':sum(a['counts'].get(i)!=b['counts'].get(i) for i in set(a['counts'])|set(b['counts'])),'zero_contribution_mismatch':sum((a['counts'].get(i,0)==0)!=(b['counts'].get(i,0)==0) for i in set(a['counts'])|set(b['counts']))},'observed_vs_predicted_residual':len(a['support']^set(a['winners'])),'method':'current frozen create_pos_feature_map + replace_feature on CPU; independent valid-candidate/winner reconstruction from its persisted _DT_CONTEXT'}
(R/'NOVIS_GEOMETRY_REPLAY.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res,sort_keys=True))
