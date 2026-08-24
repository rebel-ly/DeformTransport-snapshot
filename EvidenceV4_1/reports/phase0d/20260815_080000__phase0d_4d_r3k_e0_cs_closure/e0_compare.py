import cv2,hashlib,json,numpy as np
from pathlib import Path
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_080000__phase0d_4d_r3k_e0_cs_closure');R.mkdir(parents=True,exist_ok=True)
a=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_040000__phase0d_4d_r3g_epsilon_bridge/e0_gpu2/e0_epsilon_only_correct_v3d_seed000.mp4');b=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4')
def d(p):
 c=cv2.VideoCapture(str(p));x=[]
 while True:
  ok,f=c.read()
  if not ok:break
  x.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
 c.release();x=np.ascontiguousarray(np.stack(x),dtype=np.uint8); return x,hashlib.sha256(x.tobytes()).hexdigest(),hashlib.sha256(p.read_bytes()).hexdigest()
x,xh,xm=d(a);y,yh,ym=d(b);z=np.abs(x.astype(np.int16)-y.astype(np.int16));o={'e0_mp4_sha':xm,'e0_decoded_rgb_sha':xh,'canonical_mp4_sha':ym,'canonical_decoded_rgb_sha':yh,'mp4_exact':xm==ym,'rgb_exact':xh==yh,'different_channel_values':int(np.count_nonzero(z)),'max_abs_diff':int(z.max()),'mean_abs_diff':float(z.mean()),'shape':list(x.shape)};R.joinpath('E0_CANONICAL_EXACT_COMPARISON.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o))
