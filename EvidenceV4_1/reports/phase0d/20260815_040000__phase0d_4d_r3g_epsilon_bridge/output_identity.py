import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/workspace/DeformTransport_EvidenceV4_1')
R3 = ROOT / 'reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation'
OUT = ROOT / 'reports/phase0d/20260815_040000__phase0d_4d_r3g_epsilon_bridge/OUTPUT_IDENTITY_MATRIX.json'
VIDEOS = {
  'c1': R3/'c1_gpu1/c1_provisional_k0_seed000.mp4',
  'c2': R3/'c2_gpu0/c2_provisional_correct_v3d_seed000.mp4',
  'd1': R3/'d1_gpu2/d1_begin0_correct_v3d_seed000.mp4',
  'wm0': ROOT/'reports/phase0d/conditioning_alignment/20260813_110231__f4_two_gpu_staged_execution/outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4',
}
CANONICAL_RGB_SHA = '935d9301a208abb73e437346c3297a31705563e60ce2aa2be4cf46b44ce7cbc6'

def fsha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20), b''): h.update(b)
 return h.hexdigest()
def dec(p):
 cap=cv2.VideoCapture(str(p)); a=[]
 while True:
  ok,fr=cap.read()
  if not ok: break
  a.append(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
 cap.release(); x=np.ascontiguousarray(np.stack(a),dtype=np.uint8)
 return x,{'path':str(p),'mp4_sha256':fsha(p),'decoded_rgb_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'shape':list(x.shape)}
arr,info={},{}
for k,p in VIDEOS.items(): arr[k],info[k]=dec(p)
result={'identity_check_only':True,'scientific_metrics_computed':False,'videos':info,
 'canonical_dtfull_decoded_rgb_sha256':CANONICAL_RGB_SHA,
 'C2_EQ_CANONICAL_DTFULL': info['c2']['decoded_rgb_sha256']==CANONICAL_RGB_SHA,
 'C1_EQ_WM0':info['c1']['decoded_rgb_sha256']==info['wm0']['decoded_rgb_sha256'],
 'C1_EQ_C2':info['c1']['decoded_rgb_sha256']==info['c2']['decoded_rgb_sha256'],
 'C2_EQ_D1':info['c2']['decoded_rgb_sha256']==info['d1']['decoded_rgb_sha256']}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
