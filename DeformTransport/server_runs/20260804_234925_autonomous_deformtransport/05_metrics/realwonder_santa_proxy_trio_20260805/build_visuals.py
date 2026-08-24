import imageio.v3 as iio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
root=Path('server_runs/20260804_234925_autonomous_deformtransport')
paths=[root/'prepared_inputs/santa_21f_final_sim_proxy_v1/simulation_proxy.mp4',root/'04_smoke/REALWONDER_SANTA_BASELINE_20260805_032928/santa_baseline_seed0.mp4',root/'04_smoke/REALWONDER_SANTA_CORRECT_20260805_033343/santa_correct_seed0.mp4',root/'04_smoke/REALWONDER_SANTA_SHUFFLED_20260805_033730/santa_shuffled_seed0.mp4']
labels=['Coarse proxy (not GT)','Baseline','Correct transport','Shuffled control']
vids=[np.stack([np.asarray(f)[...,:3] for f in iio.imiter(p,plugin='FFMPEG')]) for p in paths]
assert all(v.shape==(21,480,832,3) for v in vids)
font=ImageFont.load_default()
def tile(a,label,size=(416,240)):
 im=Image.fromarray(a).resize(size,Image.Resampling.LANCZOS); canvas=Image.new('RGB',(size[0],size[1]+30),'black'); canvas.paste(im,(0,30)); ImageDraw.Draw(canvas).text((8,4),label,font=font,fill='white'); return np.asarray(canvas)
frames=[]
for t in range(21):
 z=[tile(v[t],l) for v,l in zip(vids,labels)]; frames.append(np.concatenate([np.concatenate(z[:2],1),np.concatenate(z[2:],1)],0))
iio.imwrite(root/'05_metrics/realwonder_santa_proxy_trio_20260805/trio_2x2.mp4',np.stack(frames),fps=10,plugin='FFMPEG')
sel=[0,5,10,15,20]; small=(249,144); fw,fh=small; sheet=Image.new('RGB',(fw*5, (fh+28)*4),'white'); d=ImageDraw.Draw(sheet)
for r,(v,label) in enumerate(zip(vids,labels)):
 for c,t in enumerate(sel): sheet.paste(Image.fromarray(v[t]).resize(small,Image.Resampling.LANCZOS),(c*fw,r*(fh+28)+28))
 d.text((5,r*(fh+28)+3),label,font=font,fill='black')
sheet.save(root/'05_metrics/realwonder_santa_proxy_trio_20260805/contact_sheet.jpg',quality=94)
print({'视频shape':list(np.stack(frames).shape),'contact_sheet':list(sheet.size)})
