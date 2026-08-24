import cv2,numpy as np
from pathlib import Path
R=Path(__file__).parent;B=R.parents[1]
vp=[('/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/baseline/aligned_santa_baseline_seed0.mp4','RW'),('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4','DT-FULL'),(str(B/'outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4'),'WM-0'),(str(B/'outputs/frag_container_seed0/santa_correct_v3d_seed000.mp4'),'DT-FRAG-PRUNE'),(str(B/'outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4'),'DT-GRID100-CENTER')]
s=np.zeros((464*5,832*5,3),np.uint8)
for r,(p,n) in enumerate(vp):
 c=cv2.VideoCapture(p);f=[]
 while 1:
  ok,x=c.read()
  if not ok:break
  f.append(x)
 c.release()
 for col,i in enumerate([0,20,40,60,80]):
  x=cv2.resize(f[i],(832,464),interpolation=cv2.INTER_CUBIC) if f[i].shape[:2]!=(464,832) else f[i];cv2.putText(x,n,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);s[r*464:(r+1)*464,col*832:(col+1)*832]=x
cv2.imwrite(str(R/'FIVE_METHOD_CONTACT_SHEET.png'),s)
(R/'VISUAL_DIAGNOSTIC_ONLY.txt').write_text('VISUAL_DIAGNOSTIC_ONLY\nNOT_USED_FOR_PROMOTION_DECISION\nAll frames use indices 0,20,40,60,80; 480x832 inputs are deterministically resized to 464x832 for common layout.\n')
