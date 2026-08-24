#!/usr/bin/env python3
"""R3 external artifact builder: mirrors formal preprocessing, never edits overlay."""
import hashlib, json, math, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

W=Path('/workspace'); ROOT=W/'DeformTransport_EvidenceV4_1'; R=ROOT/'reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation'
OVER=ROOT/'experimental/20260814__wanmove_preview_sdedit_overlay'; WAN=W/'Wan-Move'
sys.path[:0]=[str(OVER),str(WAN)]
from wan.modules.vae import WanVAE
from wan.modules.trajectory import create_pos_feature_map, replace_feature
IMG=W/'DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png'
FRAMES=ROOT/'reports/phase0d/20260814_100000__fast_a_b_screen/preview_reconstruction_20260814'
TRACK=W/'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy'
VIS=W/'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy'
CKPT=W/'Wan-Move/Wan-Move-14B-480P/Wan2.1_VAE.pth'
def file_sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def tensor_sha(a):
 h=hashlib.sha256(); h.update(str(tuple(a.shape)).encode()); h.update('torch.float32'.encode()); h.update(np.ascontiguousarray(a).tobytes()); return h.hexdigest()
def grid(h,w,max_area=480*832,stride=(4,8,8),patch=(1,2,2)):
 ar=h/w
 lh=round(np.sqrt(max_area*ar)//stride[1]//patch[1]*patch[1])
 lw=round(np.sqrt(max_area/ar)//stride[2]//patch[2]*patch[2])
 return int(lh),int(lw),int(lh*stride[1]),int(lw*stride[2])
R.mkdir(parents=True,exist_ok=True)
im=Image.open(IMG).convert('RGB'); source=TF.to_tensor(im).sub_(.5).div_(.5)
lh,lw,H,Wd=grid(source.shape[1],source.shape[2])
# Exact formal per-frame equivalent of F.interpolate(img[None], mode='bicubic'), no evaluator transform.
paths=sorted(FRAMES.glob('frame_*.png')); assert len(paths)==81
frames=torch.stack([TF.to_tensor(Image.open(p).convert('RGB')).sub_(.5).div_(.5) for p in paths])
formal_frames=F.interpolate(frames,size=(H,Wd),mode='bicubic')
preview_pixels=formal_frames.permute(1,0,2,3).contiguous().cuda()
source_vae_input=torch.concat([F.interpolate(source[None].cpu(),size=(H,Wd),mode='bicubic').transpose(0,1),torch.zeros(3,80,H,Wd)],dim=1).cuda()
vae=WanVAE(vae_pth=str(CKPT),device='cuda')
with torch.no_grad():
 preview_latent=vae.encode([preview_pixels])[0]
 y=vae.encode([source_vae_input])[0]
 track=torch.from_numpy(np.load(TRACK)).cuda(); vis=torch.from_numpy(np.load(VIS)).cuda()
 if track.ndim==4: track=track[0]
 if vis.ndim==3: vis=vis[0]
 track[...,0]*=Wd/source.shape[2]; track[...,1]*=H/source.shape[1]
 _,pos=create_pos_feature_map(track,vis,(4,8,8),H,Wd,y.size(0),track_num=vis.size(-1),device=y.device)
 edited=replace_feature(y.unsqueeze(0),pos.unsqueeze(0))[0]
 msk=torch.ones(1,81,lh,lw,device='cuda'); msk[:,1:]=0; msk=torch.concat([torch.repeat_interleave(msk[:,:1],4,dim=1),msk[:,1:]],dim=1); msk=msk.view(1,msk.shape[1]//4,4,lh,lw).transpose(1,2)[0]
 y_cond=torch.concat([msk,edited],dim=0)
a=preview_latent.float().cpu().contiguous().numpy(); out=R/'WAN_FORMAL_PREVIEW_LATENT_58x104.npy'; np.save(out,a)
contract={'formal_source_rgb_shape':list(source.shape),'formal_target_h':H,'formal_target_w':Wd,'formal_source_vae_input_shape':list(source_vae_input.shape),'source_vae_latent_shape':list(y.shape),'transport_input_y_shape':list(y.shape),'transport_edited_y_shape':list(edited.shape),'y_cond_pre_model_shape':list(y_cond.shape),'y_cond_at_model_shape':list(y_cond.shape),'formal_initial_noise_shape':[16,21,lh,lw],'sdedit_required_preview_shape':[16,21,lh,lw],'sdedit_start_state_shape':[16,21,lh,lw],'grid_contract':'SINGLE_58x104','runtime_latent_grid':[16,21,lh,lw],'source_image_sha256':file_sha(IMG),'formal_overlay_wan_move_sha256':file_sha(OVER/'wan/wan_move.py'),'vae_checkpoint_sha256':file_sha(CKPT),'preprocess':'torchvision TF.to_tensor -> (x-0.5)/0.5; torch.nn.functional.interpolate per RGB frame, mode=bicubic, default align_corners/antialias; no crop/pad/evaluator preprocessing','preprocess_code_site':'wan/wan_move.py formal source VAE input construction'}
lineage={'path':str(out),'file_sha256':file_sha(out),'tensor_content_sha256':tensor_sha(a),'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std()),'source_frames':str(FRAMES),'frame_count':81,'preprocess_provenance':contract['preprocess'],'formal_target':[H,Wd],'vae_checkpoint':str(CKPT),'vae_checkpoint_sha256':file_sha(CKPT)}
(R/'FORMAL_RUNTIME_GRID_CONTRACT.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
(R/'WAN_FORMAL_TARGET_PREPROCESS.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
(R/'WAN_FORMAL_PREVIEW_LATENT_LINEAGE.json').write_text(json.dumps(lineage,indent=2,sort_keys=True)+'\n')
print(json.dumps({'contract':contract,'lineage':lineage},sort_keys=True))
