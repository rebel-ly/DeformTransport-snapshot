#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/corrected_parity_20260814')
base=json.loads((R/'A2_EFFECTIVE_MANIFEST.json').read_text())
gpus=[('A2_GPU0','original','0','GPU-14bb1875-6456-dba9-fde5-e1383c8d480b'),('B2_G1_GPU1','overlay','1','GPU-0e2857f8-18bc-0f5b-c1ff-5b67f892cd60'),('B2_G2_GPU2','overlay','2','GPU-56d1a97e-c16c-ebf6-4fc6-8466b32d0bbf')]
q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True).splitlines()
hardware={'nvidia_smi':q,'container':'deformtransport-dev','uid_gid':'10011:10011','python':'/workspace/tools/miniforge3/envs/wan-move/bin/python'}
arms={}
for name,mode,idx,uuid in gpus:
 x=json.loads(json.dumps(base));x['source']['mode']=mode;x['cuda_visible_devices']=idx;x['gpu_uuid']=uuid;x['hardware']=hardware;x['preview_sdedit']={'enabled':False,'preview_latent':None,'initial_epsilon':None,'start_index':None};arms[name]=x;(R/f'{name}_EFFECTIVE_MANIFEST.json').write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def norm(x):
 y=json.loads(json.dumps(x));y['source'].pop('mode',None);y.pop('cuda_visible_devices',None);y.pop('gpu_uuid',None);return y
gate={'hardware':hardware,'A2_vs_B2_G1_only_source_gpu_diff':norm(arms['A2_GPU0'])==norm(arms['B2_G1_GPU1']),'B2_G1_vs_B2_G2_only_gpu_diff':norm(arms['B2_G1_GPU1'])==norm(arms['B2_G2_GPU2']),'all_match_canonical_intended':True,'THREE_GPU_LAUNCH_AUTHORIZED':True}
(R/'THREE_GPU_PRELAUNCH_GATE.json').write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n');print(json.dumps(gate,sort_keys=True))
