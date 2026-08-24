"""Append-only R4C companion: authoritative TC-ME ingredients with per-ID retention."""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

R=Path(__file__).resolve().parent; ROOT=Path('/workspace/DeformTransport')
E=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
V={
 'seed0_C1':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/c1_gpu1/c1_provisional_k0_seed000.mp4',
 'seed0_C2':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/c2_gpu0/c2_provisional_correct_v3d_seed000.mp4',
 'seed0_CS':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_070000__phase0d_4d_r3j_corrected_v2_shuffle/cs_gpu0/cs_preview_shuffled_v3d_seed000.mp4',
 'seed1_C1':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/c1_seed1_gpu0/c1_preview_k0_v3d_seed001.mp4',
 'seed1_C2':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/c2_seed1_gpu0/c2_preview_correct_v3d_seed001.mp4',
 'seed1_CS':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/cs_seed1_gpu0/cs_preview_shuffled_v3d_seed001.mp4',
 'SC':'/workspace/DeformTransport_EvidenceV4_1/reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation/outputs/SC/SC_seed000.mp4'}
ACCEPT={'seed0_C1':0.47438763126111494,'seed0_C2':0.42167144903579584,'seed0_CS':0.5138549546231557,'seed1_C1':0.39610003385941484,'seed1_C2':0.39736386332028023,'seed1_CS':0.40577810212784815,'SC':0.0}

def main(key):
 s=importlib.util.spec_from_file_location('ev',E); ev=importlib.util.module_from_spec(s);s.loader.exec_module(ev)
 cfg=ev.CASES['santa']; tracks=np.load(ROOT/cfg['tracks'])[0].astype(np.float32); vis=np.load(ROOT/cfg['vis'])[0].astype(bool); material_ids=np.load('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy')
 refs=[]
 for t in range(80):
  valid=vis[t]&vis[t+1]&np.isfinite(tracks[t]).all(1)&np.isfinite(tracks[t+1]).all(1); idx=np.where(valid)[0]; centers=tracks[t,idx]/2; ref=(tracks[t+1,idx]-tracks[t,idx])/2; inbound=(centers[:,0]>=0)&(centers[:,0]<=415)&(centers[:,1]>=0)&(centers[:,1]<=239); refs.append((idx[inbound],centers[inbound],ref[inbound]))
 device=torch.device('cuda:0'); model,transforms=ev.load_raft_cached(device); video=ev.read_video_common(Path(V[key])); rows=[]; transition=[]
 with torch.inference_mode():
  for start in range(0,80,8):
   end=min(80,start+8); a=torch.from_numpy(video[start:end]).permute(0,3,1,2).to(device); b=torch.from_numpy(video[start+1:end+1]).permute(0,3,1,2).to(device); a=F.interpolate(a,size=(240,416),mode='area');b=F.interpolate(b,size=(240,416),mode='area');a,b=transforms(a,b); pred=model(a,b)[-1].float().cpu().numpy()
   for j,t in enumerate(range(start,end)):
    idx,centers,ref=refs[t]; err=np.linalg.norm(ev.bilinear_flow(pred[j],centers)-ref,axis=1); transition.append(float(err.mean())); rows.extend((int(material_ids[i]),t,float(x)) for i,x in zip(idx,err))
 out=R/'PER_TRACK_RAW';out.mkdir(exist_ok=True); p=out/(key+'.csv')
 with p.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['material_id','transition','epe']);w.writerows(rows)
 payload={'key':key,'authoritative_evaluator_sha256':'e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5','aggregate_from_retained_rows_formal_transition_mean':float(np.mean(transition)),'accepted_aggregate':ACCEPT[key],'exact_reproduction':float(np.mean(transition))==ACCEPT[key],'abs_difference':abs(float(np.mean(transition))-ACCEPT[key]),'rows':len(rows),'csv':str(p)}
 (out/(key+'.json')).write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
if __name__=='__main__':main(sys.argv[1])
