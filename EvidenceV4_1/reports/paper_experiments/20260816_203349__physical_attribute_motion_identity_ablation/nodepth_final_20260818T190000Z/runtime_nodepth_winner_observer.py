import csv, hashlib, inspect, json, os, shutil, sys
from pathlib import Path
import numpy as np
import torch
import wan.modules.trajectory as trajectory
W=Path('/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/nodepth_final_20260818T190000Z')
EXPECTED=W/'overlay/wan/modules/trajectory.py'; EXPECTED_SHA='69a3a0cd177b9affe15c3e95f70956fd20c39fc71cf2e94f689b25c48c6f20b8'
TRACK=Path('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy')
VIS=Path('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy')
DEPTH=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy')
IDS=Path('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy')
os.environ['DT_TRANSPORT_VARIANT']='v3d'; os.environ['DT_TRACK_DEPTH_PATH']=str(DEPTH); os.environ['DT_TRACK_IDS_PATH']=str(IDS)
actual=Path(inspect.getfile(trajectory)).resolve(); assert actual==EXPECTED and hashlib.sha256(actual.read_bytes()).hexdigest()==EXPECTED_SHA
device=torch.device('cuda:0'); base_track=torch.from_numpy(np.load(TRACK)); base_vis=torch.from_numpy(np.load(VIS))
if base_track.ndim==4: base_track=base_track[0]
if base_vis.ndim==3: base_vis=base_vis[0]
base_track=base_track.clone(); base_track[...,1]*=464/480
def observe(placeholder):
    rows=[]
    def tracer(frame,event,arg):
        try:
            if event=='line' and frame.f_code is trajectory.replace_feature.__code__ and frame.f_lineno==1080:
                q=frame.f_locals; rows.append({'latent_t':int(q['tau']),'latent_y':int(q['hh']),'latent_x':int(q['ww']),'candidate_count':len(q['members']),'is_collision':int(len(q['members'])>1),'winner_material_id':int(q['ids'][q['best']].item())})
        except Exception:
            pass
        return tracer
    track=base_track.to(device); vis=base_vis.to(device)
    torch.manual_seed(0); _,pos=trajectory.create_pos_feature_map(track,vis,(4,8,8),464,832,16,track_num=vis.size(-1),device=device)
    torch.manual_seed(0); sys.settrace(tracer)
    try: trajectory.replace_feature(placeholder,pos.unsqueeze(0))
    finally: sys.settrace(None)
    return sorted(rows,key=lambda r:(r['latent_t'],r['latent_y'],r['latent_x']))
def write(name,rows):
    with (W/name).open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=['latent_t','latent_y','latent_x','candidate_count','is_collision','winner_material_id'],lineterminator='\n'); out.writeheader(); out.writerows(rows)
def read(name):
    with (W/name).open(newline='') as f: return [{k:int(v) for k,v in r.items()} for r in csv.DictReader(f)]
def payload(rows,collision=False):
    return ''.join(f"{r['latent_t']},{r['latent_y']},{r['latent_x']},{r['winner_material_id']}\n" for r in rows if not collision or r['is_collision']).encode()
shape=(1,16,21,58,104); a=torch.zeros(shape,device=device,dtype=torch.float32); b=torch.ones(shape,device=device,dtype=torch.float32)
ra=observe(a); rb=observe(b); write('RUNTIME_NODEPTH_WINNER_MAP_PLACEHOLDER_A.csv',ra); write('RUNTIME_NODEPTH_WINNER_MAP_PLACEHOLDER_B.csv',rb)
assert len(ra)==len(rb)==8962 and ra==rb and len({(r['latent_t'],r['latent_y'],r['latent_x']) for r in ra})==8962
assert sum(r['is_collision'] for r in ra)==788
shutil.copyfile(W/'RUNTIME_NODEPTH_WINNER_MAP_PLACEHOLDER_A.csv',W/'RUNTIME_NODEPTH_CANONICAL_WINNER_MAP.csv')
ref=read('NODEPTH_CANONICAL_WINNER_MAP.csv'); c2=read('C2_CANONICAL_WINNER_MAP.csv'); runtime=read('RUNTIME_NODEPTH_CANONICAL_WINNER_MAP.csv')
assert runtime==ref
assert hashlib.sha256(payload(runtime)).hexdigest()=='edc4908ee63d7a69b0440f770aa596ef7c34fcc03d0e6e242e9260914e403c2a'
assert hashlib.sha256(payload(runtime,True)).hexdigest()=='02d2765c4ab11270ed90413f21cf3104cd3a2fed91db8e12184677663c10201e'
changes=[int(x['winner_material_id']!=y['winner_material_id']) for x,y in zip(c2,runtime)]; non=[i for i,r in enumerate(runtime) if not r['is_collision']]; coll=[i for i,r in enumerate(runtime) if r['is_collision']]
assert len(runtime)==8962 and sum(changes[i] for i in non)==0 and sum(changes[i] for i in coll)==328
result={'actual_runtime_import_lineage':'PASS','actual_trajectory_sha256':EXPECTED_SHA,'trace_observation_line':1080,'trace_source_modified':False,'trace_formal_state_mutated':False,'placeholder_a_rows':len(ra),'placeholder_b_rows':len(rb),'placeholder_a_b_winner_map_identical':ra==rb,'runtime_total_rows':len(runtime),'runtime_collision_rows':sum(r['is_collision'] for r in runtime),'runtime_noncollision_rows':sum(not r['is_collision'] for r in runtime),'runtime_duplicate_keys':len(runtime)-len({(r['latent_t'],r['latent_y'],r['latent_x']) for r in runtime}),'target_cell_key_set_identical':{(r['latent_t'],r['latent_y'],r['latent_x']) for r in runtime}=={(r['latent_t'],r['latent_y'],r['latent_x']) for r in ref},'candidate_count_per_cell_identical':all(x['candidate_count']==y['candidate_count'] for x,y in zip(runtime,ref)),'collision_flag_per_cell_identical':all(x['is_collision']==y['is_collision'] for x,y in zip(runtime,ref)),'winner_material_ids_identical':all(x['winner_material_id']==y['winner_material_id'] for x,y in zip(runtime,ref)),'runtime_nodepth_winner_map_sha256':hashlib.sha256(payload(runtime)).hexdigest(),'runtime_nodepth_collision_winner_map_sha256':hashlib.sha256(payload(runtime,True)).hexdigest(),'noncollision_winner_changed_cells':sum(changes[i] for i in non),'nodepth_winner_changed_cells':sum(changes[i] for i in coll),'nodepth_winner_unchanged_cells':len(coll)-sum(changes[i] for i in coll),'nodepth_winner_changed_fraction_collision':sum(changes[i] for i in coll)/len(coll),'nodepth_winner_changed_fraction_all_writes':sum(changes)/len(runtime),'winner_exposure_replay_runtime_match':True,'nodepth_runtime_crosscheck':'PASS'}
(W/'NODEPTH_RUNTIME_REPLAY_VALIDATION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(W/'G1D2B_RUNTIME_WINNER_CROSSCHECK_REPORT.md').write_text('# PAPER-EXP-A1 / G1-D2B\n\nActual `replace_feature` line-1080 observer reproduced the frozen NODEPTH canonical winner map exactly for zero and deterministic non-zero placeholders.\n')
files=['runtime_nodepth_winner_observer.py','RUNTIME_NODEPTH_WINNER_MAP_PLACEHOLDER_A.csv','RUNTIME_NODEPTH_WINNER_MAP_PLACEHOLDER_B.csv','RUNTIME_NODEPTH_CANONICAL_WINNER_MAP.csv','NODEPTH_RUNTIME_REPLAY_VALIDATION.json','G1D2B_RUNTIME_WINNER_CROSSCHECK_REPORT.md']
(W/'G1D2B_SHA256SUMS.txt').write_text(''.join(f'{hashlib.sha256((W/n).read_bytes()).hexdigest()}  {n}\n' for n in files))
print(json.dumps(result,sort_keys=True))
