"""原生RealWonder generator/checkpoint GPU加载探针；不声称完整Baseline。"""
from __future__ import annotations
import argparse,datetime,json,resource,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
REPO=Path(__file__).resolve().parents[4]
import sys
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from vidgen import CausalInferencePipelineSDEdit

def mem_available_gib():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'): return int(line.split()[1])/1048576

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 torch.set_grad_enabled(False); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter(); mem0=mem_available_gib()
 cfg=OmegaConf.create({'independent_first_frame':False,'warp_denoising_step':True,'context_noise':0,'causal':True,'i2v':True,'i2v_flow':True,'height':480,'width':832,'num_frame_per_block':3,'denoising_step_list':[750,500,250],'mask_dropin_step':-1,'model_kwargs':{'sink_size':1,'local_attn_size':21,'timestep_shift':5.0}})
 p0=time.perf_counter(); pipe=CausalInferencePipelineSDEdit(cfg,device=torch.device('cuda')); construct=time.perf_counter()-p0
 p0=time.perf_counter(); state=torch.load(a.checkpoint,map_location='cpu'); checkpoint_load=time.perf_counter()-p0
 if 'generator' not in state or 'generator_ema' not in state: raise KeyError('checkpoint缺失generator/generator_ema')
 gen=state['generator']; p0=time.perf_counter()
 try: pipe.generator.load_state_dict(gen); normalized=False
 except RuntimeError:
  gen={k.replace('._fsdp_wrapped_module',''):v for k,v in gen.items()}; pipe.generator.load_state_dict(gen); normalized=True
 state_load=time.perf_counter()-p0; del state,gen
 pipe=pipe.to(dtype=torch.bfloat16); p0=time.perf_counter(); pipe.generator.to(device='cuda'); torch.cuda.synchronize(); gpu_move=time.perf_counter()-p0
 params=sum(x.numel() for x in pipe.generator.parameters()); first=next(pipe.generator.parameters())
 report={'时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'任务':'原生RealWonder生成端generator与checkpoint GPU加载探针','结论边界':'仅验证原生pipeline构造、checkpoint generator键、state_dict与GPU加载；未调用inference，不是Baseline视频结果','checkpoint':str(a.checkpoint),'checkpoint字节':a.checkpoint.stat().st_size,'generator与generator_ema均存在':True,'FSDP键归一化':normalized,'generator参数量':params,'首参数device':str(first.device),'首参数dtype':str(first.dtype),'sdedit':bool(pipe.sdedit),'num_frame_per_block':pipe.num_frame_per_block,'local_attn_size':pipe.local_attn_size,'耗时秒':{'pipeline构造':construct,'checkpoint_CPU加载':checkpoint_load,'state_dict加载':state_load,'generator搬入GPU':gpu_move,'总计':time.perf_counter()-t0},'资源':{'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'CPU_maxRSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576,'MemAvailable_before_GiB':mem0,'MemAvailable_after_GiB':mem_available_gib()},'通过':True}
 (a.output/'generator_load_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
