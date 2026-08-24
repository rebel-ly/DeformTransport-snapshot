import imageio.v3 as iio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
r=Path('server_runs/20260804_234925_autonomous_deformtransport')
items=[('Coarse proxy (not GT)',r/'prepared_inputs/santa_21f_final_sim_proxy_v1/simulation_proxy.mp4'),('Baseline seed0',r/'04_smoke/REALWONDER_SANTA_BASELINE_20260805_032928/santa_baseline_seed0.mp4'),('Correct',r/'04_smoke/REALWONDER_SANTA_CORRECT_20260805_033343/santa_correct_seed0.mp4'),('Shuffled',r/'04_smoke/REALWONDER_SANTA_SHUFFLED_20260805_033730/santa_shuffled_seed0.mp4'),('Dense Flow',r/'04_smoke/REALWONDER_SANTA_FLOW_20260805_035223/santa_flow_seed0.mp4'),('Correct blend a=0.5',r/'04_smoke/REALWONDER_SANTA_BLEND_20260805_040147/santa_blend_seed0.mp4')]
vs=[np.stack([np.asarray(x)[...,:3] for x in iio.imiter(p,plugin='FFMPEG')]) for _,p in items]; assert all(v.shape==(21,480,832,3) for v in vs); font=ImageFont.load_default()
def tile(a,label):
 im=Image.fromarray(a).resize((416,240),Image.Resampling.LANCZOS); z=Image.new('RGB',(416,272),'black'); z.paste(im,(0,32)); ImageDraw.Draw(z).text((8,9),label,font=font,fill='white'); return np.asarray(z)
frames=[]
for t in range(21):
 z=[tile(v[t],label) for (label,_),v in zip(items,vs)]; frames.append(np.concatenate([np.concatenate(z[:3],1),np.concatenate(z[3:],1)],0))
iio.imwrite(r/'05_metrics/realwonder_santa_proxy_trio_20260805/methods_3x2.mp4',np.stack(frames),fps=10,plugin='FFMPEG')
sel=[0,5,10,15,20]; w,h=249,144; sheet=Image.new('RGB',(w*5,(h+24)*6),'white'); d=ImageDraw.Draw(sheet)
for row,((label,_),v) in enumerate(zip(items,vs)):
 d.text((5,row*(h+24)+6),label,font=font,fill='black')
 for col,t in enumerate(sel): sheet.paste(Image.fromarray(v[t]).resize((w,h),Image.Resampling.LANCZOS),(col*w,row*(h+24)+24))
sheet.save(r/'05_metrics/realwonder_santa_proxy_trio_20260805/methods_contact_sheet.jpg',quality=94)
print({'视频shape':list(np.stack(frames).shape),'联系表尺寸':list(sheet.size)})
