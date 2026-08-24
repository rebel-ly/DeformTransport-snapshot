#!/usr/bin/env python3
import datetime,hashlib,json,sys
from pathlib import Path
import torch
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation'); O=Path('/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay');sys.path.insert(0,str(O))
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
target=5/6;s=FlowUniPCMultistepScheduler(num_train_timesteps=1000,shift=1,use_dynamic_shifting=False);s.set_timesteps(40,device='cuda',shift=3.0)
rows=[{'index':i,'timestep':int(s.timesteps[i]),'sigma':float(s.sigmas[i]),'abs_error':abs(float(s.sigmas[i])-target)} for i in range(40)]
rank=sorted(rows,key=lambda x:(x['abs_error'],x['index']));best,second=rank[:2]
result={'amendment_timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'c_result_available_before_amendment':False,'old_exact_sigma_gate':'WITHDRAWN_FALSE_PREMISE','sigma_match_policy':'ARGMIN_OVER_ACTUAL_WAN_SCHEDULE','rw_target_sigma':target,'wan_schedule_actual':rows,'wan_sigma_argmin_index':best['index'],'wan_sigma_argmin_value':best['sigma'],'wan_sigma_argmin_timestep':best['timestep'],'sigma_abs_error':best['abs_error'],'signal_coeff_rw':1-target,'signal_coeff_wan':1-best['sigma'],'signal_coeff_rel_error':abs((1-best['sigma'])-(1-target))/(1-target),'wan_second_best_index':second['index'],'wan_second_best_error':second['abs_error'],'argmin_margin':second['abs_error']-best['abs_error'],'argmin_unique':bool(second['abs_error']>best['abs_error']),'rw_wan_sigma_exact_match':False,'wan_native_start_point_audited':True,'old_zero_error_derivation_root_cause':'INCORRECT_LINSPACE_CONSTRUCTION: prior audit used np.linspace(1,1/40,40); actual formal FlowUniPC.set_timesteps uses np.linspace(sigma_max,sigma_min,num_inference_steps+1)[:-1], then float32 sigmas and int64 timesteps.'}
(R/'R3_SIGMA_POLICY_AMENDMENT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
