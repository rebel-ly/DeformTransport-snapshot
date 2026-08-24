import importlib.util,json,cv2,numpy as np
from pathlib import Path
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_060000__phase0d_4d_r3i_interpretation_mechanism'); S=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/scripts/frozen_preview_companion_metrics.py')
sp=importlib.util.spec_from_file_location('f',S); f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
P=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/preview_reconstruction_20260814')
files=sorted(P.glob('frame_*.png')); assert len(files)==81,len(files)
preview=[]
for p in files:
 x=cv2.imread(str(p));assert x is not None;preview.append(f.common(cv2.cvtColor(x,cv2.COLOR_BGR2RGB)))
preview=np.stack(preview)
videos={'C1':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/c1_gpu1/c1_provisional_k0_seed000.mp4','C2':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/c2_gpu0/c2_provisional_correct_v3d_seed000.mp4','RW':'/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/baseline/aligned_santa_baseline_seed0.mp4','DTFULL':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4'}
out={}
for n,v in videos.items():
 cap=cv2.VideoCapture(v);a=[]
 while True:
  ok,x=cap.read()
  if not ok:break
  a.append(f.common(cv2.cvtColor(x,cv2.COLOR_BGR2RGB)))
 cap.release();a=np.stack(a);assert a.shape==preview.shape,(n,a.shape,preview.shape)
 d=np.abs(a-preview);m=np.mean((a-preview)**2);g=(np.diff(a,axis=1)**2).mean()+(np.diff(a,axis=2)**2).mean()
 out[n]={'fullframe_l1':float(d.mean()),'fullframe_psnr':float(10*np.log10(1/max(m,1e-12))),'sharpness':float(g),'domain':'frozen common() 480->464 mapping; native 464 unchanged'}
R.joinpath('MASKLESS_PREVIEW_DIAGNOSTICS.json').write_text(json.dumps({'frozen_functions_reused':True,'posthoc_maskless_diagnostic':True,'track_supported_diagnostic':'UNRESOLVED_NO_AUTHORITATIVE_MASK','results':out},indent=2)+'\n');print(json.dumps(out))
