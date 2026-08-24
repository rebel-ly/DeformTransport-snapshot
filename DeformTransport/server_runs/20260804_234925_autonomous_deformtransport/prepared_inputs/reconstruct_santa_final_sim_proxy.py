"""从已有且已标注的 Santa 视频代理构建 infer_sim 最小 final_sim 目录。"""
from __future__ import annotations
import csv, datetime, hashlib, json, shutil
from pathlib import Path
import yaml
from PIL import Image
ROOT=Path('/workspace/DeformTransport')
RUN=ROOT/'server_runs/20260804_234925_autonomous_deformtransport'
SRC_FRAMES=RUN/'prepared_inputs/santa_21f_videoproxy_transport_ready/frames'
SRC_INITIAL=ROOT/'artifacts/stage1_dynamic/santa_cloth_2f_wsl_20260802_retry3/frame_initial.png'
SRC_VIDEO=ROOT/'artifacts/transport_validation/santa_cloth_21f/wan_vae/target_input.mp4'
SRC_CONFIG=ROOT/'cases/santa_cloth/config.yaml'
OUT=RUN/'prepared_inputs/santa_21f_final_sim_proxy_v1'
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): d.update(b)
 return d.hexdigest()
def main():
 if OUT.exists(): raise FileExistsError(f'为防覆盖，目标已存在：{OUT}')
 frames=sorted(SRC_FRAMES.glob('frame_*.png'))
 if len(frames)!=21: raise ValueError(f'要求21帧，实际{len(frames)}')
 OUT.mkdir(parents=True); (OUT/'frames').mkdir()
 rows=[]
 for p in frames:
  im=Image.open(p).convert('RGB')
  if im.size!=(832,480): raise ValueError(f'{p}尺寸{im.size}')
  q=OUT/'frames'/p.name; shutil.copyfile(p,q)
  rows.append([str(p),str(q),list(im.size),'uint8',sha(q)])
 source=Image.open(SRC_INITIAL).convert('RGB')
 if source.size!=(512,512): raise ValueError('初始图不是512x512')
 resized=source.resize((832,832),resample=Image.BILINEAR).crop((0,176,832,656))
 resized.save(OUT/'resized_input_image.png')
 cfg=yaml.safe_load(SRC_CONFIG.read_text(encoding='utf-8'))
 cfg['num_output_frames']=6
 cfg['simulated_frames_num']=21
 cfg['mask_dropin_step']=-1
 cfg['debug']=False
 cfg['output_folder']=str(OUT)
 cfg['deformtransport_proxy']={
  '类型':'有损工程proxy，不是原始final_sim或future GT',
  'pixel帧数':21,'latent帧数':6,'时序公式':'4*T-3=21',
  '未来帧来源':str(SRC_VIDEO),'初始图来源':str(SRC_INITIAL),
 }
 (OUT/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
 (OUT/'prompt.txt').write_text(str(cfg['vgen_prompt']).strip()+'\n',encoding='utf-8')
 shutil.copyfile(SRC_VIDEO,OUT/'simulation_proxy.mp4')
 with (OUT/'source_frames_sha256.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f); w.writerow(['源路径','目标路径','尺寸WH','dtype','SHA256']); w.writerows(rows)
 prov={
  '生成时间':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  '用途':'infer_sim原生Baseline/Correct/Shuffled最小21像素帧闭环与集成验证',
  '结论边界':'有损工程proxy；不可作为future GT、原始完整simulation或正式方法效果结论',
  '源配置':str(SRC_CONFIG),'源配置SHA256':sha(SRC_CONFIG),
  '源初始图':str(SRC_INITIAL),'源初始图SHA256':sha(SRC_INITIAL),
  '初始图处理':'512x512 RGB→双线性832x832→中心裁剪y[176:656]→832x480 PNG',
  '输出初始图shape':[480,832,3],'输出初始图dtype':'uint8','输出初始图SHA256':sha(OUT/'resized_input_image.png'),
  '未来帧来源':str(SRC_VIDEO),'未来视频SHA256':sha(SRC_VIDEO),
  '未来帧处理':'从历史target_input.mp4已解码的21个832x480 PNG逐文件复制；源MP4有损',
  '未来帧数':21,'未来帧shape':[480,832,3],'未来帧dtype':'uint8',
  '配置latent帧数':6,'像素帧公式':'4*6-3=21','denoising_step_list':cfg['denoising_step_list'],
  'mask策略':'mask_dropin_step=-1；不伪造points/mesh mask；infer_sim按源码跳过',
  '缺失待生成':['noises.npy','flows.npy'],
 }
 (OUT/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(prov,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
