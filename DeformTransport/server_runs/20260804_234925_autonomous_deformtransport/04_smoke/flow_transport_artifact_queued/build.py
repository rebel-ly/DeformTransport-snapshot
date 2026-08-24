"""以同一coarse帧和官方RAFT构造dense-flow latent对照，不覆盖material-point artifact。"""
import argparse,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from simulation.image23D.noise_warp.raft import RaftOpticalFlow

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def area_img(p):
 x=torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2,0,1).float()[None]
 return F.interpolate(x,size=(240,416),mode='area').round().clamp(0,255)[0].permute(1,2,0).byte().numpy()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source-artifact',type=Path,required=True); ap.add_argument('--frames-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); op=a.output_dir/'flow_transport_artifact_v1.pt';
 if op.exists(): raise FileExistsError(op)
 state=torch.load(a.source_artifact,map_location='cpu',weights_only=True); idx=state['latent_frame_indices'].tolist(); source=state['source_latent'][:,0].cuda(); target=state['target_latent'].cuda(); mask=state['transport_mask'].cuda()[None]; H,W=source.shape[-2:]
 files=sorted(a.frames_dir.glob('frame_*.png')); assert len(files)==21 and idx==[0,4,8,12,16,20]
 frames=[area_img(files[i]) for i in idx]; model=RaftOpticalFlow('cuda','large'); warped=[]; backward=[]; t0=time.perf_counter()
 yy,xx=torch.meshgrid(torch.arange(H,device='cuda'),torch.arange(W,device='cuda'),indexing='ij'); base=torch.stack((2*xx/(W-1)-1,2*yy/(H-1)-1),-1).float()[None]
 for j,frame in enumerate(frames):
  if j==0: flow=torch.zeros((2,240,416),device='cuda')
  else: flow=model(frame,frames[0])
  backward.append(flow); fl=F.interpolate(flow[None],size=(H,W),mode='bilinear',align_corners=True); fl[:,0]*=W/416; fl[:,1]*=H/240; grid=base+torch.stack((2*fl[:,0]/(W-1),2*fl[:,1]/(H-1)),-1); warped.append(F.grid_sample(source,grid,mode='bilinear',padding_mode='border',align_corners=True))
 flow_trans=torch.stack(warped,1); fused=torch.where(mask,flow_trans,target); torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
 new={k:v for k,v in state.items()}; new['flow_transported_latent']=flow_trans.cpu(); new['flow_fused_latent']=fused.cpu(); new['flow_backward_target_to_initial']=torch.stack(backward).cpu().half(); new['flow_baseline_metadata']={'方法':'每个latent时刻coarse RGB到初帧的RAFT-Large直接反向光流；grid_sample初帧latent；同一material transport_mask内替换，mask外保留target_latent','方向':'target_to_initial','padding_mode':'border','align_corners':True,'latent_frame_indices':idx}; torch.save(new,op)
 def ml1(x): return float((x-target).abs().masked_select(mask.expand_as(x)).mean())
 report={'生成时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'任务':'dense-flow latent公平对照artifact','输入artifact':str(a.source_artifact),'输入artifact_SHA256':sha(a.source_artifact),'帧目录':str(a.frames_dir),'latent时刻':idx,'flow_shape':list(torch.stack(backward).shape),'flow_transported_shape':list(flow_trans.shape),'flow_fused_shape':list(fused.shape),'finite':bool(torch.isfinite(fused).all()),'mask支持cell数':[int(x) for x in mask.sum(dim=(0,2,3,4)).tolist()],'masked_L1_vs_coarse_target':{'flow':ml1(fused),'correct':ml1(state['correct_fused_latent'].cuda()),'shuffled':ml1(state['shuffled_fused_latent'].cuda())},'耗时秒':elapsed,'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'输出':str(op),'输出SHA256':sha(op),'结论边界':'光流对照使用有损coarse proxy，不是future GT'}; (a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2)); assert report['finite']
if __name__=='__main__': main()
