"""以项目原NoiseWarper数学实现和torchvision官方RAFT重建structured noise。"""
from __future__ import annotations
import argparse,datetime,hashlib,json,sys,types,time
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
# 原模块的fire只用于CLI、BackgroundRemover仅用于可选去背景；本任务remove_background=False。
sys.modules.setdefault('fire',types.ModuleType('fire'))
bg=types.ModuleType('simulation.image23D.noise_warp.background_remover')
bg.BackgroundRemover=object
sys.modules['simulation.image23D.noise_warp.background_remover']=bg
from simulation.image23D.noise_warp import noise_warp as nw

def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): d.update(b)
 return d.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('final_sim',type=Path); ap.add_argument('--seed',type=int,default=0); a=ap.parse_args()
 root=a.final_sim.resolve(); noise=root/'noises.npy'; flow=root/'flows.npy'
 if noise.exists() or flow.exists(): raise FileExistsError('拒绝覆盖已有noise/flow')
 fps=sorted((root/'frames').glob('frame_*.png'))
 cfg=yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8'))
 expected_frames=(int(cfg['num_output_frames'])-1)*4+1
 if len(fps)!=expected_frames: raise ValueError(f'必须为{expected_frames}帧')
 video_raw=np.stack([np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8) for p in fps])
 # 原实现 resize_frames=0.5 会经 rp 调用 OpenCV；锁定环境无 cv2，故用 Torch area 做同尺寸预缩放。
 video_t=torch.from_numpy(video_raw.copy()).permute(0,3,1,2).float()
 video_t=F.interpolate(video_t,size=(240,416),mode='area')
 video=video_t.round().clamp_(0,255).to(torch.uint8).permute(0,2,3,1).contiguous().numpy()
 torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
 torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
 out=nw.get_noise_from_video(video,noise_channels=32,input_flow=False,output_folder=str(root),visualize=False,resize_frames=None,resize_flow=8,downscale_factor=32,device='cuda',save_files=True,remove_background=False)
 torch.cuda.synchronize(); elapsed=time.perf_counter()-started
 arr=np.load(noise,allow_pickle=False); fl=np.load(flow,allow_pickle=False)
 checks={'noise_shape':list(arr.shape),'noise_dtype':str(arr.dtype),'noise_finite':bool(np.isfinite(arr).all()),'flow_shape':list(fl.shape),'flow_dtype':str(fl.dtype),'flow_finite':bool(np.isfinite(fl).all())}
 if checks['noise_shape']!=[expected_frames,60,104,32] or not checks['noise_finite'] or not checks['flow_finite']: raise RuntimeError(checks)
 rep={'生成时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'seed':a.seed,'算法':'项目simulation/image23D/noise_warp/noise_warp.py::get_noise_from_video','光流模型':'torchvision RAFT-Large C_T_SKHT_V2','权重来源':'https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth','权重SHA256':'ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322','输入帧':len(fps),'原始输入shape':list(video_raw.shape),'RAFT输入shape':list(video.shape),'预处理':'torch.nn.functional.interpolate(mode=area,size=(240,416))；等价替代原resize_frames=0.5并绕过未安装cv2','参数':{'noise_channels':32,'resize_frames':None,'等效预缩放':0.5,'resize_flow':8,'downscale_factor':32,'remove_background':False},'检查':checks,'耗时秒':elapsed,'torch峰值allocated_MiB':torch.cuda.max_memory_allocated()/1048576,'torch峰值reserved_MiB':torch.cuda.max_memory_reserved()/1048576,'noises_SHA256':sha(noise),'flows_SHA256':sha(flow)}
 (root/'noise_generation_report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
