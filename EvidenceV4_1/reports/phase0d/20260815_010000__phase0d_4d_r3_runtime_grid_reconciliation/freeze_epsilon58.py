#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,torch
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation'); R.mkdir(parents=True,exist_ok=True)
g=torch.Generator(device='cuda');g.manual_seed(0); x=torch.randn(16,21,58,104,dtype=torch.float32,generator=g,device='cuda').cpu().contiguous()
def content(t):
 h=hashlib.sha256();h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.numpy().tobytes());return h.hexdigest()
pt=R/'R3_SHARED_EPSILON_58x104.pt';npy=R/'R3_SHARED_EPSILON_58x104.npy';torch.save(x,pt);np.save(npy,x.numpy()); reload=torch.from_numpy(np.load(npy))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
o={'epsilon_status':'PROVISIONAL_PENDING_BEGIN0','canonical_source':'wan_move.py private Generator(device=self.device), manual_seed(0), torch.randn; actual runtime shape','shape':list(x.shape),'dtype':str(x.dtype),'tensor_content_sha256':content(x),'pt_file_sha256':sha(pt),'npy_file_sha256':sha(npy),'roundtrip_exact':bool(torch.equal(x,reload) and content(x)==content(reload)),'finite':bool(torch.isfinite(x).all()),'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std())}
(R/'R3_SHARED_EPSILON_METADATA.json').write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');print(json.dumps(o,sort_keys=True))
