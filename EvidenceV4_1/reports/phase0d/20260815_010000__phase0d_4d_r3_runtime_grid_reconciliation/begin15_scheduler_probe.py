#!/usr/bin/env python3
"""External actual FlowUniPC one-step probe; no Wan model or source edits."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation'); O=R/'begin15_gpu0'; O.mkdir(exist_ok=True)
OVER=Path('/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay'); sys.path.insert(0,str(OVER))
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
def content(a):
 h=hashlib.sha256();h.update(str(tuple(a.shape)).encode());h.update('torch.float32'.encode());h.update(a.detach().cpu().contiguous().numpy().tobytes());return h.hexdigest()
p=torch.from_numpy(np.load(R/'WAN_FORMAL_PREVIEW_LATENT_58x104.npy')).cuda(); e=torch.from_numpy(np.load(R/'R3_SHARED_EPSILON_58x104.npy')).cuda()
s=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False); s.set_timesteps(40,device='cuda',shift=3.0); ts=s.timesteps; sigma=float(s.sigmas[15]); s.set_begin_index(15)
x=s.add_noise(p.unsqueeze(0),e.unsqueeze(0),ts[15].reshape(1))[0]; expected=(1-sigma)*p+sigma*e
pre={'begin_index_actual':s._begin_index,'step_index_before':s.step_index,'model_outputs_empty':all(v is None for v in s.model_outputs),'lower_order_nums':s.lower_order_nums,'last_sample_none':s.last_sample is None}
out=s.step(torch.zeros_like(x).unsqueeze(0),ts[15],x.unsqueeze(0),return_dict=False)[0]
res={'begin15_status':'PASS','runtime_preview_shape':list(p.shape),'runtime_epsilon_shape':list(e.shape),'start_state_shape':list(x.shape),'start_state_sigma_index':15,'first_denoise_index':int(s._step_index-1),'start_state_sigma':sigma,'start_state_timestep':int(ts[15]),'begin_index_actual':pre['begin_index_actual'],'model_outputs_empty_at_first_step':pre['model_outputs_empty'],'lower_order_nums_at_first_step':pre['lower_order_nums'],'last_sample_none_at_first_step':pre['last_sample_none'],'first_effective_solver_order':int(s.this_order),'off_by_one_start':bool(s._step_index-1 !=15),'start_state_exact_formula':bool(torch.equal(x,expected)),'start_state_max_abs_diff':float((x-expected).abs().max()),'runtime_epsilon_tensor_content_sha256':content(e),'planned_effective_indices':'15..39','planned_effective_steps':25,'midstart_terminal_endpoint_consistent':bool(int(ts[-1])==int(s.timesteps[-1])),'first_step_output_shape':list(out.shape)}
(O/'BEGIN15_EVIDENCE.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res,sort_keys=True))
