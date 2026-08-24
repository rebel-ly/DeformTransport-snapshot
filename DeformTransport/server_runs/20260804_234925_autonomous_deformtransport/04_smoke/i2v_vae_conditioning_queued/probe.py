import argparse,datetime,hashlib,json,resource,time
from pathlib import Path
import torch
from vidgen import WanVideoVAE,WanVideoUnit_ImageEmbedderVAE,load_first_frame

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('image',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); torch.set_grad_enabled(False); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
 vae=WanVideoVAE().to(device='cuda',dtype=torch.float32); unit=WanVideoUnit_ImageEmbedderVAE(); img=load_first_frame(str(a.image),height=480,width=832).unsqueeze(0); p0=time.perf_counter(); result=unit.process(vae,img,None,21,480,832,torch.device('cuda'),torch.float32); torch.cuda.synchronize(); dt=time.perf_counter()-p0; y=result['y']; op=a.output/'i2v_vae_conditioning.pt'; torch.save(y.cpu(),op)
 r={'时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'任务':'Santa真实首帧原生WanVideoUnit_ImageEmbedderVAE条件编码回归','输入SHA256':sha(a.image),'输入shape':list(img.shape),'输出y_shape':list(y.shape),'输出dtype':str(y.dtype),'输出finite':bool(torch.isfinite(y).all()),'输出SHA256':sha(op),'编码耗时秒':dt,'总耗时秒':time.perf_counter()-t0,'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'CPU_maxRSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576}; r['通过']=r['输出finite'] and y.numel()>0; (a.output/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(r,ensure_ascii=False,indent=2)); raise SystemExit(0 if r['通过'] else 2)
if __name__=='__main__': main()
