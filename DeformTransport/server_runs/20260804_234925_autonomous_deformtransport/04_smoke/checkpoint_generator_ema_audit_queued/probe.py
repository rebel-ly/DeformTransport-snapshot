"""在GPU上逐张量审计RealWonder checkpoint generator与generator_ema差异。"""
import argparse,datetime,json,resource,time
from pathlib import Path
import torch

def mem_gib():
 for x in Path('/proc/meminfo').read_text().splitlines():
  if x.startswith('MemAvailable:'): return int(x.split()[1])/1048576

def norm(d): return {k.replace('._fsdp_wrapped_module',''):v for k,v in d.items()}
def main():
 p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 torch.set_grad_enabled(False); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter(); m0=mem_gib(); state=torch.load(a.checkpoint,map_location='cpu'); load_t=time.perf_counter()-t0
 g,e=norm(state['generator']),norm(state['generator_ema']); keys_g=set(g); keys_e=set(e); common=sorted(keys_g&keys_e)
 exact=0; changed=0; elems=0; sq=0.0; maxdiff=0.0; top=[]; p0=time.perf_counter()
 for k in common:
  x,y=g[k],e[k]
  if x.shape!=y.shape: top.append((float('inf'),k,'shape')); continue
  n=x.numel(); elems+=n
  if torch.equal(x,y): exact+=1; continue
  changed+=1; xc=x.to('cuda',dtype=torch.float32,non_blocking=False); yc=y.to('cuda',dtype=torch.float32,non_blocking=False); d=(xc-yc).abs(); md=float(d.max()); ss=float((d*d).sum()); maxdiff=max(maxdiff,md); sq+=ss; top.append((md,k,list(x.shape))); del xc,yc,d
 torch.cuda.synchronize(); compare_t=time.perf_counter()-p0; top=sorted(top,reverse=True)[:20]
 report={'时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'任务':'RealWonder_checkpoint_generator与generator_ema_GPU逐张量差异审计','checkpoint':str(a.checkpoint),'键':{'generator':len(keys_g),'generator_ema':len(keys_e),'公共':len(common),'仅generator':sorted(keys_g-keys_e),'仅generator_ema':sorted(keys_e-keys_g)},'张量':{'完全相同':exact,'有差异':changed,'总元素':elems,'全局最大绝对差':maxdiff,'全局RMS差':(sq/elems)**0.5 if elems else None,'最大差Top20':top},'耗时秒':{'checkpoint加载':load_t,'GPU比较':compare_t,'总计':time.perf_counter()-t0},'资源':{'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'CPU_maxRSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576,'MemAvailable_before_GiB':m0,'MemAvailable_after_GiB':mem_gib()},'通过':keys_g==keys_e and len(common)>0 and elems>0}
 (a.output/'generator_ema_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2)); raise SystemExit(0 if report['通过'] else 2)
if __name__=='__main__': main()
