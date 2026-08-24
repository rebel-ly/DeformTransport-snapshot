#!/usr/bin/env python3
"""E1 clean VAE-only preview encode; no transformer or diffusion."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from wan.modules.vae import WanVAE

OUT = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen')
PREVIEW = OUT / 'preview_reconstruction_20260814'
CKPT = '/workspace/Wan-Move/Wan-Move-14B-480P/Wan2.1_VAE.pth'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b): h.update(b)
 return h.hexdigest()
frames=[]
for path in sorted(PREVIEW.glob('frame_*.png')):
 bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
 assert bgr is not None and bgr.shape[:2]==(480,832), (path, None if bgr is None else bgr.shape)
 frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
assert len(frames)==81
pixels=torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float().div(255).sub(.5).div(.5).cuda()
vae=WanVAE(vae_pth=CKPT,device='cuda')
with torch.no_grad(): latent=vae.encode([pixels])[0]
array=latent.float().cpu().contiguous().numpy()
latent_path=OUT/'preview_wan_vae_latent_e1_832x480.npy'
np.save(latent_path,array)
result={'PREVIEW_VAE_ENCODE':'PASS' if tuple(array.shape)==(16,21,60,104) and np.isfinite(array).all() else 'FAIL','PREVIEW_INPUT_SHAPE':list(pixels.shape),'PREVIEW_VAE_RAW_OUTPUT_SHAPE':list(array.shape),'PREVIEW_VAE_CANONICAL_AXIS_ORDER':'raw C,T,H,W; batched equivalent B,C,T,H,W','PREVIEW_ACTUAL_LATENT_SHAPE':[1,*list(array.shape)],'dtype':str(latent.dtype),'min':float(array.min()),'max':float(array.max()),'mean':float(array.mean()),'finite_count':int(np.isfinite(array).sum()),'nonfinite_count':int((~np.isfinite(array)).sum()),'PREVIEW_LATENT_COMPATIBLE_WITH_WAN':tuple(array.shape)==(16,21,60,104),'NO_EVAL_RESIZE_BEFORE_VAE':True,'latent_path':str(latent_path),'latent_shape':list(array.shape),'latent_dtype':str(array.dtype),'latent_sha256':sha(latent_path),'checkpoint_sha256':sha(CKPT)}
(OUT/'E1_PREVIEW_WAN_VAE_ENCODE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,sort_keys=True),flush=True)
