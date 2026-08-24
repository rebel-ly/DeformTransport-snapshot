import torch,numpy as np,hashlib,json
from pathlib import Path
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_110000__phase0d_4d_r3m_r2_epsilon_validation'); A=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation/R3_SHARED_EPSILON_58x104.npy'); B=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_100000__phase0d_4d_r3m_seed1_wave1/EPSILON_SEED1_58x104.npy')
def gen(s):
 g=torch.Generator(device='cuda:0');g.manual_seed(s);return torch.randn(16,21,58,104,dtype=torch.float32,generator=g,device='cuda:0')
def cmp(x,y):
 d=(x-y).abs();return {'equal':bool(torch.equal(x,y)),'different':int(torch.count_nonzero(x!=y)),'max':float(d.max()),'mean':float(d.mean())}
s0=gen(0);s1=gen(1);a=torch.from_numpy(np.load(A)).cuda();b=torch.from_numpy(np.load(B)).cuda();np.save(R/'seed0_replay.npy',s0.cpu().numpy());np.save(R/'seed1_replay.npy',s1.cpu().numpy());np.save(R/'seed1_roundtrip.npy',b.cpu().numpy());q=torch.from_numpy(np.load(R/'seed1_roundtrip.npy')).cuda()
o={'seed0_vs_authoritative':cmp(s0,a),'seed1_vs_existing':cmp(s1,b),'seed1_vs_seed0':cmp(b,a),'seed1_roundtrip':cmp(b,q),'shape':list(b.shape),'dtype':str(b.dtype),'finite':bool(torch.isfinite(b).all())}
(R/'EPSILON_VALIDATION.json').write_text(json.dumps(o,indent=2));print(json.dumps(o))
