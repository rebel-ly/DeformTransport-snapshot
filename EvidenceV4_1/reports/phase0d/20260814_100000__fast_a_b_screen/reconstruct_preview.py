#!/usr/bin/env python3
"""Copy the exact canonical RealWonder inference-frame sequence with a hash manifest."""
import hashlib, json, shutil
from pathlib import Path

src=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/frames')
out=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/preview_reconstruction_20260814')
frames=sorted(src.glob('frame_*.png'))
assert len(frames)==81 and [p.name for p in frames]==[f'frame_{i:04d}.png' for i in range(81)]
out.mkdir(exist_ok=False)
rows=[]
for i,p in enumerate(frames):
 q=out/p.name; shutil.copyfile(p,q)
 h=hashlib.sha256(q.read_bytes()).hexdigest()
 rows.append({'frame_index':i,'simulation_step':i*10,'source_path':str(p),'reconstructed_path':str(q),'sha256':h})
(out/'PREVIEW_RECONSTRUCTION_MANIFEST.json').write_text(json.dumps({'producer':'infer_sim.py:load_sim_frames','producer_semantics':'sorted frame_*.png; RGB convert; resize(832,480); normalize [-1,1] for SDEdit VAE input','original_preview_persisted':False,'frame_count':81,'timeline':'frame i -> simulation step 10*i; steps 0..800; no step810','frames':rows},indent=2)+'\n')
