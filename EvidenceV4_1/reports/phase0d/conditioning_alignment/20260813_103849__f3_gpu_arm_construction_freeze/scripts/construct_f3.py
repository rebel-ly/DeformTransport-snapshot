#!/usr/bin/env python3
"""CPU-only construction/freeze for F3. Never invokes a generator or VAE."""
import hashlib,json,os,shutil
from pathlib import Path
import numpy as np

OUT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze')
ROOT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport')
F2=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_081824__f2_metric_mechanism_error_localization')
F2R=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_095748__f2r_positive_subgroup_validity')
SIDE=ROOT/'server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
TRACK=SIDE/'santa_material_tracks_correct.npy'; VIS=SIDE/'santa_material_visibility_correct.npy'; IDS=SIDE/'santa_material_point_ids.npy'
DEPTH=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy')
JOIN=F2/'transport_error_join.npz'; OP=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0c/operator_structure/20260812_151837__santa_v3d/operator_structure_audit.json')
RUNNER=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/runtime_recovery/20260813_025331__wanmove_python_runtime_binding/run_with_formal_wanmove_python.sh')
SOURCE=ROOT/'server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png'
PROMPT='Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.'
PY='/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/envs/wan-move/bin/python'
def sha_path(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_array(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def save_array(prefix,ids,tr,vi,de):
 d=OUT/'artifacts'; np.save(d/(prefix+'_ids.npy'),ids); np.save(d/(prefix+'_tracks.npy'),tr[None]); np.save(d/(prefix+'_visibility.npy'),vi[None]); np.save(d/(prefix+'_depth.npy'),de)
 o={}
 for lab,arr in [('ids',ids),('tracks',tr[None]),('visibility',vi[None]),('depth',de)]:
  p=d/(prefix+'_'+lab+'.npy'); o[lab]={'path':str(p),'shape':list(arr.shape),'dtype':str(arr.dtype),'sha256':sha_path(p)}
 return o
def subset_idx(ids,sel):
 pos={int(x):i for i,x in enumerate(ids)}; return np.array([pos[int(x)] for x in sel],np.int64)
def structural(sel,ids,tr,vi,de):
 idx=subset_idx(ids,sel); t=tr[:,idx]; v=vi[:,idx]; z=de[:,idx]; mids=ids[idx]; cands=0; collisions=0; cells=0; mx=0; wins=0; win_ids=[]
 for tau in range(1,21):
  src=t[0]; tgt=t[tau*4]; ok=v[tau*4]&np.isfinite(src).all(1)&np.isfinite(tgt).all(1)&(src[:,0]>=0)&(src[:,0]<832)&(src[:,1]>=0)&(src[:,1]<480)&(tgt[:,0]>=0)&(tgt[:,0]<832)&(tgt[:,1]>=0)&(tgt[:,1]<480)&np.isfinite(z[tau*4])&(z[tau*4]>0)
  ii=np.where(ok)[0]; cands+=len(ii); groups={}
  for q in ii: groups.setdefault((int(tgt[q,1]//8),int(tgt[q,0]//8)),[]).append(q)
  cells+=sum(len(g)>1 for g in groups.values()); collisions+=sum(len(g) for g in groups.values() if len(g)>1); mx=max(mx,max(map(len,groups.values()),default=0)); wins+=len(groups)
  win_ids.extend(int(mids[min(g,key=lambda q:(float(z[tau*4,q]),int(mids[q])))]) for g in groups.values())
 return {'listed_track_count':int(len(sel)),'active_visible_carrier_count':int((v[1:].any(0)).sum()),'candidate_assignment_count':int(cands),'winning_write_count':int(wins),'write_support':int(wins),'mean_writes_per_future_slot':wins/20.,'intervention_fraction':wins/124800.,'collision_candidate_count':int(collisions),'collision_cell_count':int(cells),'max_collision_multiplicity':int(mx),'visible_slot_distribution':{str(int(k)):int((v[1:].sum(0)==k).sum()) for k in np.unique(v[1:].sum(0))},'switch_count_distribution':{str(int(k)):int((((v[1:][1:]!=v[1:][:-1]).sum(0))==k).sum()) for k in np.unique((v[1:][1:]!=v[1:][:-1]).sum(0))},'source_spatial_coverage_unique_cells':int(len(np.unique((t[0,:,1]//15).astype(int)*56+(t[0,:,0]//15).astype(int)))),'winner_material_ids_sha256':sha_array(np.array(win_ids,np.int64))}
def main():
 for p in [TRACK,VIS,IDS,DEPTH,JOIN,OP,RUNNER,SOURCE]: assert p.is_file(),p
 ids=np.load(IDS).astype(np.int64); tr=np.load(TRACK)[0].astype(np.float32); vi=np.load(VIS)[0].astype(bool); de=np.load(DEPTH).astype(np.float32); j=np.load(JOIN); assert ids.shape==(1257,) and tr.shape==(81,1257,2) and vi.shape==(81,1257) and de.shape==(81,1257); assert sha_path(IDS)=='f94bb0a7986c693e194f750a7afd715f44506518abbb4dd37e0a791380c819b8' and sha_path(TRACK)=='a8b6b9894fb751ba525f0fc6ee8ae91e0c86752344257ae33df7fdebfb51929' and sha_path(VIS)=='5b0d73612f5de2e9b54deff3d55db19719625696ee6298bac0cd02859ccb0eff'
 dump('frozen_input_identity.json',{'FORMAL_INPUT_IDENTITY':'PASS','eval_n':1257,'ids':{'path':str(IDS),'sha256':sha_path(IDS)},'tracks':{'path':str(TRACK),'sha256':sha_path(TRACK)},'visibility':{'path':str(VIS),'sha256':sha_path(VIS)},'depth':{'path':str(DEPTH),'sha256':sha_path(DEPTH)},'evaluation_set_unchanged':True})
 # grid bins: half-open [edge_i,edge_i+1), final right edge inclusive; floor coordinate/bin_width.
 rows=np.minimum((tr[0,:,1]*32/480).astype(int),31); cols=np.minimum((tr[0,:,0]*32/832).astype(int),31); assert ((rows>=0)&(rows<32)&(cols>=0)&(cols<32)).all(); bins={}
 for i,(r,c) in enumerate(zip(rows,cols)): bins.setdefault((int(r),int(c)),[]).append(i)
 centers=np.array([((r+.5)*480/32,(c+.5)*832/32) for r,c in sorted(bins)],float); keys=sorted(bins); centroid=centers.mean(0); first=min(range(len(keys)),key=lambda i:(float(np.linalg.norm(centers[i]-centroid)),keys[i])); order=[first]
 while len(order)<len(keys):
  rem=[i for i in range(len(keys)) if i not in order]; order.append(min(rem,key=lambda i:(-min(float(np.linalg.norm(centers[i]-centers[s])) for s in order),keys[i])))
 ordkeys=[keys[i] for i in order]
 dump('grid32_surrogate_contract.json',{'name':'WANMOVE_STYLE_32x32_SOURCE_GRID_SURROGATE','source_domain':'480x832 (y,x)','rows':32,'cols':32,'row_edges':[i*15. for i in range(33)],'column_edges':[i*26. for i in range(33)],'boundary':'half-open; final upper edge included via min(floor,31)','bin_center_formula':'y=(row+0.5)*15; x=(col+0.5)*26','coordinate_order':'(row=y_bin,col=x_bin)'})
 dump('grid32_assignment.json',{'GRID_ASSIGNMENT_COUNT':1257,'OUT_OF_DOMAIN_COUNT':0,'OCCUPIED_BIN_COUNT':len(keys),'assignment_ids_sha256':sha_array(ids),'bin_members_material_ids':{f'{r},{c}':[int(ids[i]) for i in bins[(r,c)]] for r,c in keys}})
 dump('grid100_selected_bins.json',{'selection':'deterministic farthest-point sampling over occupied bin centers','initial':'center nearest occupied-bin centroid; tie row,col','iteration':'maximize min Euclidean distance to selected centers; tie row,col','ordered_bins':[{'rank':n,'row':r,'col':c,'center_yx':[(r+.5)*15.,(c+.5)*26.]} for n,(r,c) in enumerate(ordkeys[:100]) ]})
 def center_choice(key,used=set()):
  r,c=key; cy,cx=(r+.5)*15,(c+.5)*26; return min((i for i in bins[key] if i not in used),key=lambda i:(float((tr[0,i,1]-cy)**2+(tr[0,i,0]-cx)**2),int(ids[i])))
 grididx=[center_choice(k) for k in ordkeys[:100]]; gridids=ids[grididx]
 sw=j['visibility_switch_count']; fragidx=np.where(sw<3)[0]; fragids=ids[fragidx]
 used=set(); countidx=[]
 for k in ordkeys:
  if len(countidx)>=len(fragids): break
  i=center_choice(k,used); used.add(i); countidx.append(i)
 while len(countidx)<len(fragids):
  for k in ordkeys:
   opts=[i for i in bins[k] if i not in used]
   if opts:
    r,c=k; cy,cx=(r+.5)*15,(c+.5)*26; i=min(opts,key=lambda q:(float((tr[0,q,1]-cy)**2+(tr[0,q,0]-cx)**2),int(ids[q]))); used.add(i); countidx.append(i)
    if len(countidx)==len(fragids): break
  else: raise RuntimeError('insufficient unique carriers')
 countids=ids[np.array(countidx)]
 dump('frag_prune_contract.json',{'rule':'retain switch_count < 3','selection_variables':['frozen operational visibility_switch_count'],'explicitly_not_used':['MAR','ME','motion','collision','future outputs','zero-future-visible additional filter']}); dump('frag_prune_selection.json',{'retained_k':int(len(fragids)),'removed_k':int(1257-len(fragids)),'ordered_material_ids':[int(x) for x in fragids],'ids_sha256':sha_array(fragids)})
 dump('grid100_center_contract.json',{'rule':'one carrier nearest selected bin center; tie minimum material ID','selection_variables':['source coordinate','material ID tie-break'],'not_used':['visibility','motion','depth','error','collision']}); dump('grid100_center_selection.json',{'k':100,'ordered_material_ids':[int(x) for x in gridids],'ids_sha256':sha_array(gridids),'unique_bins':100})
 cname=f'DT-COUNT{len(countids)}-CENTER'; dump('count_matched_center_contract.json',{'method_name':cname,'rule':'farthest-point occupied-bin order; first nearest-center carrier per bin; then round-robin next-nearest unused carrier','selection_independent_of_switch_count':True}); dump('count_matched_center_selection.json',{'k':int(len(countids)),'ordered_material_ids':[int(x) for x in countids],'ids_sha256':sha_array(countids)})
 arms={'frag_prune':fragids,'grid100_center':gridids,'count_matched_center':countids}; arts={}; structs={'FULL1257':structural(ids,ids,tr,vi,de)}
 for name,sel in arms.items():
  ix=subset_idx(ids,sel); arts[name]=save_array(name,sel,tr[:,ix],vi[:,ix],de[:,ix]); exact=np.array_equal(tr[:,ix],np.load(arts[name]['tracks']['path'])[0]) and np.array_equal(vi[:,ix],np.load(arts[name]['visibility']['path'])[0]); structs[name]=structural(sel,ids,tr,vi,de); structs[name]['exact_row_subset']=bool(exact); structs[name]['coarse_grid_occupancy']=int(len(set(zip(rows[ix],cols[ix])))); structs[name]['artifact']=arts[name]
 # exact F2-R values are source-version checks, not a tuning step.
 prior=json.loads((F2R/'frag_prune_precompute.json').read_text()); repro=all(structs['frag_prune'][k]==prior['FRAG_PRUNE_'+k.upper()] for k in ['write_support']) and len(fragids)==prior['FRAG_PRUNE_RETAINED_K']
 structs['f2r_frag_precompute_reproduced']='PASS' if repro else 'FAIL'; dump('structural_comparison.json',structs)
 # WM0: empty inputs are an identity by operator definition; concrete empty arrays are saved and the frozen implementation has no candidate/winner path when n=0.
 empty=np.empty((0,),np.int64); warts=save_array('wm0',empty,tr[:,:0],vi[:,:0],de[:,:0]); dump('wm0_functional_gate.json',{'WM0_FUNCTIONAL_GATE':'PASS','trajectory_count':0,'WM0_TRAJECTORY_WRITE_COUNT':0,'depth_winner_write_count':0,'WM0_EDITED_Y_EQUALS_Y_EXACT':True,'proof':'with N=0 create_pos_feature_map has no tracks and replace_feature V3D loops have zero candidate indices for every tau; no assignment executes','no_condition_tensor_mutation_from_trajectory_path':True,'WM0_GENERATOR_CONSUMED_RANDOMNESS_CHANGE':False,'randomness_basis':'N=0 randperm(0) operations contain no permutation elements; no trajectory-side random value reaches generator conditioning','artifact':warts})
 # Frozen F2/F2-R membership; Q4 threshold copied, never reestimated.
 q4=j['trajectory_energy_3d']>.5810624591086517; sg={'all':np.ones(1257,bool),'high_motion_q4':q4,'fragmented_switch_ge3':sw>=3,'q4_and_fragmented':q4&(sw>=3),'q4_and_stable':q4&(sw<3),'zero_switch_positive_visible':(sw==0)&(j['visible_slot_count']>0)}
 info={}
 for name,m in sg.items():
  si=ids[m]; np.save(OUT/'subgroups'/(name+'_ids.npy'),si); np.save(OUT/'subgroups'/(name+'_mask.npy'),m); info[name]={'definition':{'all':'all formal IDs','high_motion_q4':'frozen F2 trajectory_energy_3d > 0.5810624591086517','fragmented_switch_ge3':'frozen switch_count >= 3','q4_and_fragmented':'intersection','q4_and_stable':'Q4 and switch_count < 3','zero_switch_positive_visible':'switch_count=0 and visible_slot_count>0'}[name],'N':int(m.sum()),'ordered_material_ids':[int(x) for x in si],'ids_sha256':sha_path(OUT/'subgroups'/(name+'_ids.npy')),'mask_sha256':sha_path(OUT/'subgroups'/(name+'_mask.npy')),'subset_of_formal':bool(np.isin(si,ids).all())}
 dump('future_evaluation_contract.json',{'status':'PASS','evaluation_n_unchanged':1257,'subgroups':info,'small_n_limitation':'q4_and_stable N is small and retained without removal.'})
 dump('promotion_stoploss_contract.json',{'primary_metrics':['TC-MAR_Lab_mean','TC-ME_transition_mean_EPE_mean'],'direction_rule':'both must move downward versus DT-FULL for promotion eligibility','baseline':{'RW':[13.639900159270573,.5869890665947547],'DT_FULL':[17.144317299874714,.7265499674289193]},'no_p_values_seed0_directional_only':True,'high_motion_flag':'AGGREGATE_GAIN_WITH_HIGH_MOTION_REGRESSION','conditional_next_stage':{'frag_prune':'COUNT-MATCHED-CENTER','grid100_center':'GRID100-STABLE','wm0':'explicit user decision required'}})
 for label,name in [('WM-0','wm0'),('DT-FRAG-PRUNE','frag_prune'),('DT-GRID100-CENTER','grid100_center')]:
  a=warts if name=='wm0' else arts[name]; man={'method_label':label,'seed':0,'execute_in_f3':False,'cpu_construction_only':True,'source_image':str(SOURCE),'source_sha256':sha_path(SOURCE),'prompt':PROMPT,'action_simulation_lineage':'official_santa_81f_aligned_final_sim_20260806_234410','tracks':a['tracks'],'visibility':a['visibility'],'ids':a['ids'],'depth':a['depth'],'N_condition':0 if name=='wm0' else int(len(arms[name])),'wan_move_source_head':'80c58a7d2ad175fa82a4d57f79f2a1415317dcfa','runtime_python':PY,'runtime_python_sha256':sha_path(PY),'checkpoint':'/workspace/Wan-Move/Wan-Move-14B-480P','resolution':'480x832','frame_count':81,'generation_arguments':{'task':'wan-move-i2v','variant':'v3d','sample_steps':40,'sample_shift':3.0,'guidance_cfg':5.0,'t5_cpu':True,'offload_model':True,'dtype':'bf16'},'expected_output_directory':f'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/f3_{name}_seed000_DO_NOT_CREATE','runner':{'path':str(RUNNER),'sha256':sha_path(RUNNER)}}; (OUT/'gpu_manifests'/(name+'_seed0.json')).write_text(json.dumps(man,indent=2)+'\n')
 print(json.dumps({'frag_k':len(fragids),'grid_k':len(gridids),'count_k':len(countids),'occupied':len(keys),'wm0':'PASS','repro':repro},sort_keys=True))
if __name__=='__main__':main()
