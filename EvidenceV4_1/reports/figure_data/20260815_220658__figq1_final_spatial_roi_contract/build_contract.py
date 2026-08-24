import csv,json,math,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; A=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline')
T=np.load(A/'santa_material_tracks_correct.npy')[0];V=np.load(A/'santa_material_visibility_correct.npy')[0].astype(bool);I=np.load(A/'santa_material_point_ids.npy')
# Fixed ROI-A selected only by max of prior fixed-frame physics scores.
raw=[420.8,67.0,669.9,179.6];w,h=raw[2]-raw[0],raw[3]-raw[1];padx,pady=.1*w,.1*h;roiA=[max(0,math.floor(raw[0]-padx)),max(0,math.floor(raw[1]-pady)),min(831,math.ceil(raw[2]+padx)),min(463,math.ceil(raw[3]+pady))]
# source median six-NN distance defines fixed reappearance radius.
P=T[0];D=((P[:,None]-P[None,:])**2).sum(2);r=np.sqrt(np.take_along_axis(D,np.argsort(D,axis=1)[:,1:7],axis=1));radius=3*float(np.median(r))
events=[]
for j in range(len(I)):
 z=V[:,j];a=0
 while a<81:
  if not z[a]:
   b=a
   while b<81 and not z[b]:b+=1
   if a>0 and b<81 and z[a-1] and b-a>=4:events.append({'j':j,'id':int(I[j]),'ds':a,'de':b-1,'re':b,'dur':b-a})
   a=b
  else:a+=1
# union-find events sharing reappearance +/-2 and reappearance position within radius
n=len(events);par=list(range(n))
def find(x):
 while par[x]!=x:par[x]=par[par[x]];x=par[x]
 return x
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:par[b]=a
for a in range(n):
 for b in range(a+1,n):
  if abs(events[a]['re']-events[b]['re'])<=2:
   pa=T[events[a]['re'],events[a]['j']];pb=T[events[b]['re'],events[b]['j']]
   if np.isfinite(pa).all() and np.isfinite(pb).all() and np.linalg.norm(pa-pb)<=radius:union(a,b)
groups={}
for q,e in enumerate(events):groups.setdefault(find(q),[]).append(e)
out=[]
for es in groups.values():
 if len(es)<3:continue
 re=int(round(np.median([e['re'] for e in es])));near=min((20,40,60,80),key=lambda x:abs(x-re)); pts=np.array([T[near,e['j']] for e in es]);pts[:,1]*=464/480
 if not np.isfinite(pts).all():continue
 box=[float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,0].max()),float(pts[:,1].max())];score=(len(es),float(np.median([e['dur'] for e in es])),-abs(near-re));out.append({'events':es,'re':re,'near':near,'box':box,'score':score})
out.sort(key=lambda x:x['score'],reverse=True);out=out[:5]
with (R/'FIGQ1_OCCLUSION_EVENT_CLUSTERS.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['cluster_id','num_material_ids','material_ids','median_disappear_start','median_disappear_end','median_reappear_frame','median_duration','nearest_display_frame','frame_distance','raw_bbox_at_display_frame','ranking_score_components'])
 for q,x in enumerate(out,1):
  e=x['events'];w.writerow([q,len(e),';'.join(map(str,[z['id'] for z in e])),np.median([z['ds'] for z in e]),np.median([z['de'] for z in e]),x['re'],np.median([z['dur'] for z in e]),x['near'],abs(x['near']-x['re']),x['box'],x['score']])
roiB=None
if out:
 x=out[0];b=x['box'];bw,bh=b[2]-b[0],b[3]-b[1]
 if len(x['events'])>=3 and bw>=16 and bh>=16:
  px,py=.1*bw,.1*bh;roiB=[max(0,math.floor(b[0]-px)),max(0,math.floor(b[1]-py)),min(831,math.ceil(b[2]+px)),min(463,math.ceil(b[3]+py))]
contract={'RW_480_TO_464_MODE':'BICUBIC_RESIZE','RW_480_TO_464_EXACT_RULE':'torch.nn.functional.interpolate(size=(464,832), mode=bicubic, align_corners=False, antialias=False).clamp_(0,1)','RW_QUALITATIVE_ALIGNMENT_CONFIDENCE':'PASS','PREVIEW_RGB_GEOMETRY':'832x480','PREVIEW_TO_464_RULE':'same formal evaluator/conditioning bicubic 480x832 -> 464x832 mapping; no crop','ROI_A_SELECTION_RULE':'MAX_PHYSICS_ONLY_LOCAL_DEFORMATION_SCORE_AMONG_FIXED_DISPLAY_FRAMES','ROI_A_FRAME':60,'ROI_A_RAW_BBOX':raw,'ROI_A_PADDED_INTEGER_BBOX':roiA,'ROI_A_SCORE':1.0289578437805176,'ROI_B_AVAILABLE':roiB is not None,'ROI_B_SELECTION_RULE':'cluster size desc, median duration desc, frame distance asc; visibility/trajectory only','ROI_B_DISPLAY_FRAME':out[0]['near'] if roiB else None,'ROI_B_RAW_BBOX':out[0]['box'] if roiB else None,'ROI_B_PADDED_INTEGER_BBOX':roiB,'ROI_B_NUM_MATERIAL_IDS':len(out[0]['events']) if roiB else 0,'source_knn_radius':radius,'ROI_SELECTION_OUTCOME_BLIND':True,'FIGQ1_ROWS':['PhysicalPreview','C1','C2','CS','RealWonder'],'FIGQ1_COLUMNS':[0,20,40,60,80],'FIGQ1_UNIFIED_DISPLAY_DOMAIN':'832x464','GENERATED_RAFT_FLOW_PANEL':False}
(R/'FIGQ1_FINAL_CONTRACT.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n');(R/'FIGQ1_FINAL_CONTRACT.md').write_text('# FIG-Q1 final spatial/ROI contract\n\n'+json.dumps(contract,indent=2,sort_keys=True)+'\n')
with (R/'SHA256SUMS.txt').open('w') as f:
 for p in sorted(R.iterdir()):
  if p.is_file():f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print(json.dumps(contract))
