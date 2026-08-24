import argparse,datetime,hashlib,json,resource,time
from pathlib import Path
import torch
from vidgen import WanImageEncoder,load_first_frame

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('image',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); torch.set_grad_enabled(False); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
 enc=WanImageEncoder().to(device='cuda',dtype=torch.float32); img=load_first_frame(str(a.image),height=480,width=832).unsqueeze(0).to('cuda'); p0=time.perf_counter(); out=enc.encode_image([img]); torch.cuda.synchronize(); encode_t=time.perf_counter()-p0; op=a.output/'clip_feature.pt'; torch.save(out.cpu(),op)
 r={'时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'任务':'Santa真实首帧Wan_CLIP图像编码器SDPA端到端回归','输入':str(a.image),'输入SHA256':sha(a.image),'输入shape':list(img.shape),'输出shape':list(out.shape),'输出dtype':str(out.dtype),'输出finite':bool(torch.isfinite(out).all()),'输出SHA256':sha(op),'CLIP编码耗时秒':encode_t,'总耗时秒':time.perf_counter()-t0,'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'CPU_maxRSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576}
 r['通过']=r['输出finite'] and out.numel()>0; (a.output/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(r,ensure_ascii=False,indent=2)); raise SystemExit(0 if r['通过'] else 2)
if __name__=='__main__': main()
