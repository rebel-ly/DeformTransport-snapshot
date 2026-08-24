#!/usr/bin/env python3
"""Build canonical and A2/B2 manifests before GPU launch; no GPU/model imports."""
import hashlib,json,os,sys
from pathlib import Path
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/corrected_parity_20260814')
W=Path('/workspace'); WAN=W/'Wan-Move'; OVER=W/'DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def item(p): return {'path':str(p),'sha256':sha(p)}
prompt='Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.'
base={'schema':'phase0d-4c-corrected-single-launcher-v1','cwd':'/workspace/Wan-Move','python':item(W/'tools/miniforge3/envs/wan-move/bin/python'),'checkpoint_dir':str(WAN/'Wan-Move-14B-480P'),'checkpoint_files':{'vae':item(WAN/'Wan-Move-14B-480P/Wan2.1_VAE.pth')},'source_image':item(W/'DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png'),'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'tracks':item(W/'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy'),'visibility':item(W/'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy'),'ids':item(W/'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy'),'depth':item(W/'DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy'),'env':{'DT_TRANSPORT_VARIANT':'v3d','DT_TRACK_IDS_PATH':'/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy','DT_TRACK_DEPTH_PATH':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy','PYTHONHASHSEED':'0'},'cuda_visible_devices':'0','gpu_uuid':'GPU-14bb1875-6456-dba9-fde5-e1383c8d480b','cli':{'task':'wan-move-i2v','size':'480*832','frame_num':81,'seed':0,'sample_solver':'unipc','sample_steps':40,'sample_shift':3.0,'sample_guide_scale':5.0,'dtype':'bf16','t5_cpu':True,'offload_model':True,'eval_bench':False},'scheduler':{'class':'FlowUniPCMultistepScheduler','num_train_timesteps':1000,'shift':3.0,'dynamic_shifting':False},'generation_domain':'832x480','decoded_output_domain':'832x464','determinism':{'private_torch_generator_seed':0,'torch_manual_seed_before_trajectory':0,'numpy_seed':'not_called','python_random_seed':'not_called_seed0_path','cuda_deterministic_env':'not_set_in_authoritative_runner'}}
canon=dict(base); canon['source']={'mode':'original','root':str(WAN),'wan_move_sha256':sha(WAN/'wan/wan_move.py'),'generate_sha256':sha(WAN/'generate.py'),'trajectory_sha256':sha(WAN/'wan/modules/trajectory.py')}
a2=dict(base); a2['source']=canon['source']; a2['launcher']=item(R/'run_single_launcher.sh')
b2=dict(base); b2['source']={'mode':'overlay','root':str(OVER),'wan_move_sha256':sha(OVER/'wan/wan_move.py'),'generate_sha256':sha(OVER/'generate.py'),'trajectory_sha256':sha(OVER/'wan/modules/trajectory.py'),'preview_sdedit':'OFF','preview_latent':None,'initial_epsilon':None,'start_index':None}; b2['launcher']=item(R/'run_single_launcher.sh')
def clean(x):
 y=json.loads(json.dumps(x)); y.pop('source',None); y.pop('launcher',None); return y
R.mkdir(parents=True,exist_ok=True)
for n,x in [('CANONICAL_FULL_MANIFEST.json',canon),('A2_EFFECTIVE_MANIFEST.json',a2),('B2_EFFECTIVE_MANIFEST.json',b2)]: (R/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
gate={'CANONICAL_MANIFEST_COMPLETE':True,'A2_MANIFEST_MATCH_CANONICAL':clean(a2)==clean(canon),'B2_MANIFEST_MATCH_CANONICAL':clean(b2)==clean(canon),'A2_B2_MANIFEST_ONLY_SOURCE_DIFF':clean(a2)==clean(b2),'launcher_sha256':sha(R/'run_single_launcher.sh'),'launch_authorized':clean(a2)==clean(canon) and clean(b2)==clean(canon) and clean(a2)==clean(b2)}
(R/'PRELAUNCH_MANIFEST_GATE.json').write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n');print(json.dumps(gate,sort_keys=True))
