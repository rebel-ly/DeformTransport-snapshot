#!/usr/bin/env python3
"""One allowed VAE-only reconstruction; no transformer or diffusion execution."""
import argparse, hashlib, json, os
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

def sha_bytes(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--vae',required=True)
 ap.add_argument('--tracks',required=True); ap.add_argument('--visibility',required=True); ap.add_argument('--depth',required=True)
 ap.add_argument('--ids',required=True); ap.add_argument('--operator-audit',required=True); ap.add_argument('--out',required=True)
 a=ap.parse_args(); out=Path(a.out); frames=out/'dt_edited_y_decoded_frames'; frames.mkdir(parents=True,exist_ok=True)
 os.environ['DT_TRANSPORT_VARIANT']='v3d'; os.environ['DT_TRACK_DEPTH_PATH']=a.depth; os.environ['DT_TRACK_IDS_PATH']=a.ids
 from wan.modules.vae import WanVAE
 import wan.modules.trajectory as tr
 image=Image.open(a.source).convert('RGB'); rgb=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().div(255).sub(.5).div(.5)
 source_tensor=torch.cat([rgb[:,None],torch.zeros((3,80,480,832),dtype=rgb.dtype)],dim=1)
 tracks_np=np.load(a.tracks); vis_np=np.load(a.visibility); ids_np=np.load(a.ids).astype(np.int64); depth_np=np.load(a.depth).astype(np.float32)
 tracks=torch.from_numpy(tracks_np).cuda()[0]; vis=torch.from_numpy(vis_np).cuda()[0]
 vae=WanVAE(vae_pth=a.vae,device='cuda'); y=vae.encode([source_tensor.cuda()])[0]
 assert list(y.shape)==[16,21,60,104]
 torch.manual_seed(0); _,track_pos=tr.create_pos_feature_map(tracks,vis,(4,8,8),480,832,y.size(0),track_num=vis.size(-1),device=y.device)
 ctx=tr._DT_CONTEXT; maps=[]; total=0
 for tau in range(1,21):
  th=track_pos[:,tau,0].to(tracks.device); tw=track_pos[:,tau,1].to(tracks.device); sampled=ctx['visibility'][::4][tau]
  valid=(track_pos[:,0,0].to(tracks.device)>=0)&(track_pos[:,0,1].to(tracks.device)>=0)&(th>=0)&(tw>=0)&sampled
  groups={}
  for ii in torch.where(valid)[0].tolist(): groups.setdefault((int(th[ii]),int(tw[ii])),[]).append(ii)
  winners={}
  for cell,members in groups.items():
   cand=[]
   for ii in members:
    dep=float(ctx['depth'][::4][tau,ii]); mid=int(ctx['ids'][ii])
    if np.isfinite(dep) and dep>0: cand.append(((dep,mid),mid))
   if cand: winners[str(cell)]=min(cand)[1]
  maps.append(winners); total+=len(winners)
 audit=json.load(open(a.operator_audit)); expected=audit['per_slot']['Correct']; same=all(maps[k]==expected[k]['winner_material_ids_by_cell'] for k in range(20))
 assert total==9031 and same
 edited=tr.replace_feature(y.unsqueeze(0),track_pos.unsqueeze(0))[0]
 assert torch.equal(edited[:,0],y[:,0])
 y_np=y.float().cpu().numpy(); e_np=edited.float().cpu().numpy(); np.save(out/'real_encoded_y.npy',y_np); np.save(out/'real_edited_y.npy',e_np)
 decoded=vae.decode([edited])[0].float().cpu(); assert list(decoded.shape)==[3,81,480,832]
 u8=((decoded.clamp(-1,1)+1)*.5*255).to(torch.uint8).permute(1,2,3,0).numpy()
 for i,frame in enumerate(u8): cv2.imwrite(str(frames/f'frame_{i:04d}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
 report={'status':'DIAGNOSTIC_ONLY','execution':'VAE_ONLY_NO_TRANSFORMER_NO_DIFFUSION','source_sha256':file_sha(a.source),'vae_checkpoint_sha256':file_sha(a.vae),
  'tracks_sha256':file_sha(a.tracks),'visibility_sha256':file_sha(a.visibility),'depth_sha256':file_sha(a.depth),'ids_sha256':file_sha(a.ids),
  'encoded_shape':list(y_np.shape),'edited_shape':list(e_np.shape),'latent_hw':[60,104],'decoded_shape':list(decoded.shape),
  'structural_write_support':total,'expected_write_support':9031,'winner_map_exact_phase0c':same,'source_slot_exact':True,
  'encoded_tensor_bytes_sha256':sha_bytes(y_np),'edited_tensor_bytes_sha256':sha_bytes(e_np),'decoded_frame_count':len(u8)}
 (out/'dt_real_edited_y_reconstruction.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,sort_keys=True))
if __name__=='__main__': main()
