#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import torch, numpy as np
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_200000__phase0d_4d_recovery')
shape=(16,21,60,104); seed=0
g=torch.Generator(device=torch.device('cuda:0')); g.manual_seed(seed)
x=torch.randn(*shape,dtype=torch.float32,generator=g,device=torch.device('cuda:0'))
cpu=x.detach().contiguous().cpu()
def content(t):
 h=hashlib.sha256(); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes()); return h.hexdigest()
pt=R/'FINAL_C_SHARED_EPSILON.pt'; npy=R/'FINAL_C_SHARED_EPSILON.npy'
torch.save(cpu,pt); np.save(npy,cpu.numpy())
loaded=torch.load(pt,map_location='cpu',weights_only=True).contiguous()
meta={'epsilon_status':'PROVISIONAL','canonical_epsilon_source':'DETERMINISTIC_RECONSTRUCTION','creation_code_path':'wan_move.py:230-238 private CUDA Generator(device=self.device), manual_seed(0), torch.randn(float32, [16,21,60,104])','seed':seed,'shape':list(cpu.shape),'dtype':str(cpu.dtype),'device_at_generation':'cuda:0','file_path':str(pt),'npy_runtime_binding_path':str(npy),'file_sha256':hashlib.sha256(pt.read_bytes()).hexdigest(),'tensor_content_sha256':content(cpu),'min':float(cpu.min()),'max':float(cpu.max()),'mean':float(cpu.mean()),'std':float(cpu.std()),'finite':bool(torch.isfinite(cpu).all()),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'roundtrip_exact':bool(torch.equal(cpu,loaded) and content(cpu)==content(loaded))}
(R/'FINAL_C_SHARED_EPSILON_METADATA.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n'); print(json.dumps(meta,sort_keys=True))
