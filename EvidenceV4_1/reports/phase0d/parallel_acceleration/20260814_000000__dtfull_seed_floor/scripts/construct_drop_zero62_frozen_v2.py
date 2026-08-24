#!/usr/bin/env python3
"""CPU-only DROP-ZERO62 construction; frozen IDs, canonical order, V3D subset input."""
import hashlib, json, os
from pathlib import Path
import numpy as np

W=Path('/workspace'); RUN=W/'DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor'
OUT=RUN/'drop_zero62'/os.environ['DROP_ZERO62_STAMP']
SIDE=W/'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
F3=W/'DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze'
F2=W/'DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_081824__f2_metric_mechanism_error_localization'
DEPTH=W/'DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy'
SOURCE=W/'DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png'
TRACK=SIDE/'santa_material_tracks_correct.npy'; VIS=SIDE/'santa_material_visibility_correct.npy'; IDS=SIDE/'santa_material_point_ids.npy'; ZERO=F3/'subgroups/zero_switch_positive_visible_ids.npy'; JOIN=F2/'transport_error_join.npz'
def sh(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,a):
 p=OUT/n; np.save(p,a); return {'path':str(p),'sha256':sh(p),'shape':list(a.shape),'dtype':str(a.dtype)}
def winners(ids,tr,vi,de):
 m={}; c=cc=wc=0
 for tau in range(1,21):
  t=tau*4; src,tgt=tr[0],tr[t]
  ok=vi[t]&np.isfinite(src).all(1)&np.isfinite(tgt).all(1)&(src[:,0]>=0)&(src[:,0]<832)&(src[:,1]>=0)&(src[:,1]<480)&(tgt[:,0]>=0)&(tgt[:,0]<832)&(tgt[:,1]>=0)&(tgt[:,1]<480)&np.isfinite(de[t])&(de[t]>0)
  g={}
  for i in np.where(ok)[0]:g.setdefault((tau,int(tgt[i,1]//8),int(tgt[i,0]//8)),[]).append(i)
  c+=int(ok.sum()); cc+=sum(len(x) for x in g.values() if len(x)>1); wc+=len(g)
  for k,x in g.items():m[k]=int(ids[min(x,key=lambda i:(float(de[t,i]),int(ids[i])))])
 return m,{'candidate_assignment_count':c,'collision_candidate_count':cc,'collision_cell_count':sum(len(x)>1 for x in g.values()),'winning_write_count':wc,'spatial_write_support_count':len(m)}
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 full=np.load(IDS).astype(np.int64); zero=np.load(ZERO).astype(np.int64); tr=np.load(TRACK)[0].astype(np.float32); vi=np.load(VIS)[0].astype(bool); de=np.load(DEPTH).astype(np.float32); j=np.load(JOIN)
 assert full.shape==(1257,) and sh(IDS)=='f94bb0a7986c693e194f750a7afd715f44506518abbb4dd37e0a791380c819b8' and zero.shape==(62,) and len(np.unique(zero))==62
 pos={int(x):i for i,x in enumerate(full)}; ix0=np.array([pos[int(x)] for x in zero]); assert np.array_equal(full[ix0],zero) and np.all(j['visibility_switch_count'][ix0]==0) and np.all(j['visible_slot_count'][ix0]>0)
 keep=~np.isin(full,zero); ix=np.where(keep)[0]; outids=full[ix]; assert len(outids)==1195 and not np.intersect1d(outids,zero).size and set(outids)|set(zero)==set(full)
 art={'ids':save('drop_zero62_ids.npy',outids),'tracks':save('drop_zero62_tracks.npy',tr[:,ix][None]),'visibility':save('drop_zero62_visibility.npy',vi[:,ix][None]),'depth':save('drop_zero62_depth.npy',de[:,ix])}
 fw,fs=winners(full,tr,vi,de); dw,ds=winners(outids,tr[:,ix],vi[:,ix],de[:,ix]); changes=sum(fw.get(k)!=dw.get(k) for k in set(fw)|set(dw))
 ver={'FULL_IDS_COUNT':1257,'FULL_IDS_SHA256':sh(IDS),'ZERO62_COUNT':62,'ZERO62_UNIQUE_COUNT':62,'ZERO62_DUPLICATE_COUNT':0,'ZERO62_ALL_IN_FULL1257':True,'ZERO62_SWITCH_COUNT_ZERO_ALL':True,'ZERO62_POSITIVE_VISIBLE_ALL':True,'DROP_ZERO62_COUNT':1195,'DROP_ZERO62_IDS_SHA256':art['ids']['sha256'],'DROP_ZERO62_INTERSECTION_ZERO62_EMPTY':True,'DROP_ZERO62_UNION_FULL':True,'CANONICAL_ORDERING_PRESERVED':True,'NORMAL_SUBSET_REARBITRATION_CONFIGURED':True,'DROP_ZERO62_GENERATION_INPUT_COUNT':1195,'DROP_ZERO62_MANIFEST_PASS':True}
 x={'experiment_name':'DROP-ZERO62','creation_timestamp':os.environ['DROP_ZERO62_STAMP'],'scientific_status':'FROZEN_PREDEFINED_INTERVENTION','parent_population':'FULL_corrected_v2','parent_N':1257,'removed_subgroup':'ZERO_SWITCH_POSITIVE_VISIBLE','removed_N':62,'retained_N':1195,'authoritative_full_ids_path':str(IDS),'authoritative_full_ids_sha256':sh(IDS),'authoritative_zero62_source_path':str(ZERO),'authoritative_zero62_source_sha256':sh(ZERO),'zero62_ids':[int(q) for q in zero],'zero62_ids_sha256':sh(ZERO),'retained_ids':[int(q) for q in outids],'retained_ids_sha256':art['ids']['sha256'],'selection_rule':'FULL corrected-v2 minus exact frozen ZERO_SWITCH_POSITIVE_VISIBLE N62','ordering_rule':'preserve canonical FULL corrected-v2 material-ID order','subset_rearbitration':'normal subset re-arbitration','source_image':{'path':str(SOURCE),'sha256':sh(SOURCE)},'tracks_source':{'path':str(TRACK),'sha256':sh(TRACK)},'visibility_source':{'path':str(VIS),'sha256':sh(VIS)},'depth_source':{'path':str(DEPTH),'sha256':sh(DEPTH)},'timeline_identity':'corrected-v2 81-frame aligned timeline','corrected_v2_lineage':'F3 frozen subgroup membership + corrected-v2 authoritative inputs','generation_input_artifacts':art,'generator_configuration':{'operator':'V3D','sample_steps':40,'sample_shift':3.0,'dtype':'bf16','seed':0},'METRIC_BASED_SELECTION':False,'POST_HOC_TUNING':False,'STABLEVIS_USED':False,'SCIENTIFIC_CONFIGURATION_CHANGED':False,'verification':ver,'structural_difference_diagnostics':{'removed_carrier_count':62,'retained_carrier_count':1195,'full':fs,'drop_zero62':ds,'winner_changes_vs_full':changes,'diagnostic_only':True}}
 (OUT/'DROP_ZERO62_MANIFEST.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'out':str(OUT),'pass':True}))
if __name__=='__main__':main()
