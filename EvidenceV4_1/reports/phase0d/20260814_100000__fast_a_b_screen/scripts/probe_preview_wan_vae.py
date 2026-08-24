#!/usr/bin/env python3
"""B3: actual frozen Wan VAE encode of the canonical 832x480 preview."""
import hashlib, json
from pathlib import Path
import cv2, numpy as np, torch
from wan.modules.vae import WanVAE

OUT=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen')
FRAMES=OUT/'preview_reconstruction_20260814'
VAE='/workspace/Wan-Move/Wan-Move-14B-480P/Wan2.1_VAE.pth'
fs=sorted(FRAMES.glob('frame_*.png')); assert len(fs)==81
arr=[]
for p in fs:
    bgr=cv2.imread(str(p),cv2.IMREAD_COLOR); assert bgr is not None and bgr.shape[:2]==(480,832)
    arr.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
x=torch.from_numpy(np.stack(arr)).permute(3,0,1,2).float().div(255).sub(.5).div(.5).cuda()
vae=WanVAE(vae_pth=VAE,device='cuda')
with torch.no_grad(): y=vae.encode([x])[0]
z=y.detach().float().cpu().numpy()
report={'status':'PASS' if tuple(z.shape)==(16,21,60,104) and np.isfinite(z).all() else 'FAIL','input_shape':list(x.shape),'input_domain':'canonical preview RGB 832x480, no evaluation resize','latent_shape':list(z.shape),'batched_wan_shape':[1,*list(z.shape)],'axis_semantics':'C,T,H,W; batched B,C,T,H,W','dtype':str(y.dtype),'min':float(z.min()),'max':float(z.max()),'finite_count':int(np.isfinite(z).sum()),'nonfinite_count':int((~np.isfinite(z)).sum()),'compatible_with_wan':tuple(z.shape)==(16,21,60,104),'checkpoint':VAE,'checkpoint_sha256':hashlib.sha256(open(VAE,'rb').read()).hexdigest()}
(OUT/'B3_PREVIEW_WAN_VAE_ENCODE.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
