"""用真实Santa输入执行原生Baseline预生成路径，并核验transport运行时分支。"""
from __future__ import annotations
import argparse, ast, datetime, inspect, json, resource, sys, time
from pathlib import Path
import torch
from omegaconf import OmegaConf
REPO=Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
import infer_sim
from deform_transport.pipeline_integration import load_precomputed_transport_latent
from vidgen import CausalInferencePipelineSDEdit, load_noise, load_first_frame

def mem_gib():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'): return int(line.split()[1])/1048576

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--final-sim',type=Path,required=True); ap.add_argument('--artifact',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 torch.set_grad_enabled(False); torch.cuda.reset_peak_memory_stats(); torch.manual_seed(0); t0=time.perf_counter(); mem0=mem_gib()
 cfg=OmegaConf.create({'independent_first_frame':False,'warp_denoising_step':True,'context_noise':0,'causal':True,'i2v':True,'i2v_flow':True,'height':480,'width':832,'num_frame_per_block':3,'denoising_step_list':[750,500,250],'mask_dropin_step':-1,'model_kwargs':{'sink_size':1,'local_attn_size':21,'timestep_shift':5.0}})
 p0=time.perf_counter(); pipe=CausalInferencePipelineSDEdit(cfg,device=torch.device('cuda')); construct=time.perf_counter()-p0
 p0=time.perf_counter(); state=torch.load(a.checkpoint,map_location='cpu'); load_ckpt=time.perf_counter()-p0
 gen=state['generator']
 try: pipe.generator.load_state_dict(gen); normalized=False
 except RuntimeError:
  pipe.generator.load_state_dict({k.replace('._fsdp_wrapped_module',''):v for k,v in gen.items()}); normalized=True
 del state,gen
 pipe=pipe.to(dtype=torch.bfloat16); pipe.generator.to('cuda'); pipe.vae.to('cuda')
 p0=time.perf_counter(); noise=load_noise(noise_path=str(a.final_sim/'noises.npy'),target_frames=6,channel_dim=16,downsample_mode='nearest',eval_degradation=0.5); noise_time=time.perf_counter()-p0
 first=load_first_frame(str(a.final_sim/'resized_input_image.png'),height=480,width=832)
 frames=infer_sim.load_sim_frames(a.final_sim/'frames',height=480,width=832)
 p0=time.perf_counter(); encoded=pipe.vae.encode_to_latent(frames.to(device='cuda',dtype=torch.bfloat16)).to(dtype=torch.bfloat16); torch.cuda.synchronize(); encode_time=time.perf_counter()-p0
 baseline=encoded
 correct=load_precomputed_transport_latent(a.artifact,mode='correct',reference_latent=encoded)
 shuffled=load_precomputed_transport_latent(a.artifact,mode='shuffled',reference_latent=encoded)
 source=inspect.getsource(infer_sim.main); tree=ast.parse(source)
 transport_calls=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_precomputed_transport_latent' for n in ast.walk(tree))
 inference_src=inspect.getsource(pipe.inference)
 checks={
  'noise_shape':list(noise['structured_noise'].shape),'first_frame_shape':list(first.shape),'sim_frames_shape':list(frames.shape),
  'encoded_shape':list(encoded.shape),'baseline与原生编码同一对象':baseline is encoded,
  'correct_shape':list(correct.shape),'shuffled_shape':list(shuffled.shape),
  'correct与encoded最大绝对差':float((correct-encoded).abs().max().item()),
  'shuffled与encoded最大绝对差':float((shuffled-encoded).abs().max().item()),
  'Correct与Shuffled最大绝对差':float((correct-shuffled).abs().max().item()),
  'infer_sim_AST_transport加载调用数':transport_calls,
  'pipeline_inference签名含sim_latent':'sim_latent' in inspect.signature(pipe.inference).parameters,
  'pipeline_inference实现读取sim_latent':'sim_latent' in inference_src,
 }
 passed=(checks['noise_shape']==[6,16,60,104] and checks['sim_frames_shape']==[1,3,21,480,832] and checks['encoded_shape']==[1,6,16,60,104] and checks['baseline与原生编码同一对象'] and checks['correct_shape']==checks['encoded_shape'] and checks['shuffled_shape']==checks['encoded_shape'] and checks['Correct与Shuffled最大绝对差']>0 and transport_calls==1 and checks['pipeline_inference签名含sim_latent'])
 report={'时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'任务':'原生Baseline预生成GPU路径等价性与transport运行时分支审计','结论边界':'真实执行pipeline构造、checkpoint加载、structured_noise加载、首帧加载、21帧Wan_VAE编码及Correct/Shuffled payload注入；未执行去噪generator，不是视频结果','检查':checks,'FSDP键归一化':normalized,'耗时秒':{'pipeline构造':construct,'checkpoint加载':load_ckpt,'noise加载':noise_time,'VAE编码':encode_time,'总计':time.perf_counter()-t0},'资源':{'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'CPU_maxRSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576,'MemAvailable_before_GiB':mem0,'MemAvailable_after_GiB':mem_gib()},'通过':passed}
 (a.output/'runtime_audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
 if not passed: raise SystemExit(2)
if __name__=='__main__': main()
