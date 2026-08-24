import argparse,datetime,hashlib,json,time
from pathlib import Path
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from simulation.image23D.noise_warp.raft import RaftOpticalFlow

def read(p):
 x=np.stack([np.asarray(f)[...,:3] for f in iio.imiter(p,plugin='FFMPEG')]); t=torch.from_numpy(x.copy()).permute(0,3,1,2).float(); return F.interpolate(t,size=(240,416),mode='area').round().clamp(0,255).to(torch.uint8).permute(0,2,3,1).numpy()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--reference-flow',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('videos',nargs='+',type=Path); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); torch.set_grad_enabled(False); torch.cuda.reset_peak_memory_stats(); ref=torch.from_numpy(np.load(a.reference_flow).astype(np.float32)).cuda(); model=RaftOpticalFlow('cuda','large'); reports={}; t0=time.perf_counter()
 for vp in a.videos:
  frames=read(vp); fs=[]
  for x,y in zip(frames[:-1],frames[1:]): fs.append(model(x,y))
  flow=torch.stack(fs); arr=flow.detach().cpu().numpy().astype(np.float16); op=a.output_dir/(vp.parent.name+'_raft_flow.npy'); np.save(op,arr); epe=((flow-ref).square().sum(1).sqrt()); dot=(flow*ref).sum(1); cos=dot/(flow.square().sum(1).sqrt()*ref.square().sum(1).sqrt()+1e-6); reports[vp.parent.name]={'视频':str(vp),'flow_shape':list(flow.shape),'EPE_mean_px':float(epe.mean()),'EPE_median_px':float(epe.median()),'EPE_p95_px':float(torch.quantile(epe,0.95)),'cosine_mean':float(cos.mean()),'预测flow幅值mean_px':float(flow.square().sum(1).sqrt().mean()),'参考flow幅值mean_px':float(ref.square().sum(1).sqrt().mean()),'finite':bool(torch.isfinite(flow).all())}
 report={'时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'任务':'官方RAFT_Large生成视频相对coarse_proxy光流一致性','说明':'参考是有损coarse proxy的RAFT flow，不是future_GT；只衡量运动引导一致性','结果':reports,'总耗时秒':time.perf_counter()-t0,'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576}; (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
