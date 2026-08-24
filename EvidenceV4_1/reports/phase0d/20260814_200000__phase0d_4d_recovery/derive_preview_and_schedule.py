#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import numpy as np

R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_200000__phase0d_4d_recovery')
raw_path=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/preview_wan_vae_latent_e1_832x480.npy')
batched_path=R/'preview_wan_vae_latent_e1_832x480_batched.npy'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
raw=np.load(raw_path); batched=raw[None,...]; np.save(batched_path,batched)
schedule=[]
for i,s in enumerate(np.linspace(1,1/40,40)):
 sigma=float(3*s/(1+2*s)); schedule.append({'index':i,'sigma':sigma,'timestep':sigma*1000})
out={'original_preview_path':str(raw_path),'original_preview_sha256':sha(raw_path),'derived_batched_preview_path':str(batched_path),'derived_batched_preview_sha256':sha(batched_path),'original_shape':list(raw.shape),'runtime_shape':list(batched.shape),'batch_axis_inserted':0,'no_transpose':True,'no_permute':True,'no_resize':True,'no_interpolation':True,'batched_0_exact_original':bool(np.array_equal(batched[0],raw)),'wan_schedule_40_shift3':schedule,'index14':schedule[14],'index15':schedule[15],'index16':schedule[16],'wan_sigma_index0':schedule[0]['sigma'],'wan_sigma_index0_exactly_one':schedule[0]['sigma']==1.0}
(R/'PREVIEW_BATCH_LINEAGE.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
