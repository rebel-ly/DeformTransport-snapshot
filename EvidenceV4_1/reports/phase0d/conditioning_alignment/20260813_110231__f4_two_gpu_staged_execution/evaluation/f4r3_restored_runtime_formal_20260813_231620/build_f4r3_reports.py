import json,csv
from pathlib import Path
import cv2,numpy as np
R=Path(__file__).parent;B=R.parents[1]
rows=[
['RW',None,13.639900159270573,9.546106088056899,40.176803651452055,.0971740217773049,.04986683472192713,.3180196253644923,.5869890665947547,.5070961964161795,1.153139444856402],
['DT-FULL',1257,17.144317299874714,14.77844642547139,37.01063968507866,.11649965856279348,.07873046770691872,.3197196369059383,.7265499674289193,.6906036221851227,1.3080249503965733],
['WM-0',0,44.47206866882119,45.50277235633449,74.84032925367356,.2641506528993322,.20115665549279324,.6719350707406798,3.618652456619911,2.441411089765593,10.055949220567522],
['DT-FRAG-PRUNE',218,17.160929784863576,12.816823387145996,47.79447044372557,.11741110898824558,.06575892852353199,.3785046316683292,.6744934747082023,.565172035859079,1.4197383653388151],
['DT-GRID100-CENTER',100,23.224553849614445,18.520685630185262,58.37823755490152,.14880805595837307,.09151995375606359,.45833471240157164,.6989570945353583,.5961485205756436,1.3877246957781877]]
keys=['METHOD','CONDITION_K','TCMAR_LAB_MEAN','TCMAR_LAB_MEDIAN','TCMAR_LAB_P95','TCMAR_RGBL1_MEAN','TCMAR_RGBL1_MEDIAN','TCMAR_RGBL1_P95','TCME_MEAN','TCME_MEDIAN','TCME_P95']
D=[dict(zip(keys,x)) for x in rows];rw,dt=D[0],D[1]
for x in D:
 x['DELTA_MAR_VS_DTFULL']=x['TCMAR_LAB_MEAN']-dt['TCMAR_LAB_MEAN'];x['DELTA_ME_VS_DTFULL']=x['TCME_MEAN']-dt['TCME_MEAN'];x['DELTA_MAR_VS_RW']=x['TCMAR_LAB_MEAN']-rw['TCMAR_LAB_MEAN'];x['DELTA_ME_VS_RW']=x['TCME_MEAN']-rw['TCME_MEAN'];x['MAR_GAP_RECOVERY_FRACTION']=(dt['TCMAR_LAB_MEAN']-x['TCMAR_LAB_MEAN'])/(dt['TCMAR_LAB_MEAN']-rw['TCMAR_LAB_MEAN']) if x['METHOD'] not in ('RW','DT-FULL') else None;x['ME_GAP_RECOVERY_FRACTION']=(dt['TCME_MEAN']-x['TCME_MEAN'])/(dt['TCME_MEAN']-rw['TCME_MEAN']) if x['METHOD'] not in ('RW','DT-FULL') else None;x['PRIMARY_DIRECTION_PASS']=x['METHOD'] not in ('RW','DT-FULL') and x['DELTA_MAR_VS_DTFULL']<0 and x['DELTA_ME_VS_DTFULL']<0;x['SEED0_NUMERIC_BEATS_RW_ON_BOTH_PRIMARY']=x['TCMAR_LAB_MEAN']<rw['TCMAR_LAB_MEAN'] and x['TCME_MEAN']<rw['TCME_MEAN']
fields=list(D[0]);
with open(R/'MASTER_SEED0_COMPARISON.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(D)
(R/'MASTER_SEED0_COMPARISON.md').write_text('|'+ '|'.join(fields)+'|\n|'+'|'.join(['---']*len(fields))+'|\n'+'\n'.join('|'+ '|'.join(str(x[k]) for k in fields)+'|' for x in D)+'\n\nSEED0_DESCRIPTIVE_ONLY; NO_STATISTICAL_SUPERIORITY_CLAIM.\n')
pareto={}
for a in D:
 dom=[b['METHOD'] for b in D if b is not a and b['TCMAR_LAB_MEAN']<=a['TCMAR_LAB_MEAN'] and b['TCME_MEAN']<=a['TCME_MEAN'] and (b['TCMAR_LAB_MEAN']<a['TCMAR_LAB_MEAN'] or b['TCME_MEAN']<a['TCME_MEAN'])]; domin=[b['METHOD'] for b in D if b is not a and a['TCMAR_LAB_MEAN']<=b['TCMAR_LAB_MEAN'] and a['TCME_MEAN']<=b['TCME_MEAN'] and (a['TCMAR_LAB_MEAN']<b['TCMAR_LAB_MEAN'] or a['TCME_MEAN']<b['TCME_MEAN'])];pareto[a['METHOD']]={'PARETO_FRONT':not dom,'PARETO_DOMINATED_BY':dom,'PARETO_DOMINATES':domin}
(R/'seed0_pareto_analysis.json').write_text(json.dumps(pareto,indent=2)+'\n');(R/'seed0_pareto_analysis.md').write_text(json.dumps(pareto,indent=2)+'\n')
sg=json.loads((R/'frozen_subgroup_diagnostics.json').read_text())['subgroups']; order=['all','high_motion_q4','fragmented_switch_ge3','q4_and_fragmented','q4_and_stable','zero_switch_positive_visible']; safety=[]
for m in ['WM-0','DT-FRAG-PRUNE','DT-GRID100-CENTER']:
 z={'METHOD':m}
 for sn,label in [('all','ALL'),('high_motion_q4','Q4'),('fragmented_switch_ge3','FRAGMENTED'),('q4_and_fragmented','Q4_FRAGMENTED'),('q4_and_stable','Q4_STABLE'),('zero_switch_positive_visible','ZERO_SWITCH_VISIBLE')]:
  z[label+'_MAR_CHANGE']=sg[m][sn]['TCMAR_LAB_MEAN']-sg['DT-FULL'][sn]['TCMAR_LAB_MEAN'];z[label+'_ME_CHANGE']=sg[m][sn]['TCME_MEAN']-sg['DT-FULL'][sn]['TCME_MEAN']
 z['AGGREGATE_GAIN_WITH_HIGH_MOTION_REGRESSION']=D[[x['METHOD'] for x in D].index(m)]['PRIMARY_DIRECTION_PASS'] and (z['Q4_MAR_CHANGE']>0 or z['Q4_ME_CHANGE']>0);safety.append(z)
with open(R/'SUBGROUP_SAFETY_TABLE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(safety[0]));w.writeheader();w.writerows(safety)
(R/'SUBGROUP_SAFETY_TABLE.md').write_text(json.dumps({'subgroups':sg,'safety':safety,'Q4_AND_STABLE_SMALL_N':13},indent=2)+'\n')
route={'NEXT_ROUTE_CASE':'CASE_E_NO_ARM_PROMOTED','PROMOTED_ARM':'NONE','PRIMARY_DIRECTION_PASS_ARMS':[],'NEXT_CAUSAL_CONTROL_OR_ACTION':'ROUTE_REASSESSMENT','SPARSE_PRUNE_ROUTE_STATUS':'STOP_PRIMARY_DIRECTION_FAILURE','TRAJECTORY_CONDITIONING_MAY_BE_NET_NEGATIVE':False,'SEED0_DESCRIPTIVE_ONLY':True,'NO_STATISTICAL_SUPERIORITY_CLAIM':True};(R/'NEXT_ROUTE_DECISION_F4R3.json').write_text(json.dumps(route,indent=2)+'\n')
# deterministic visual diagnostic
paths=[ROOT for ROOT in []]
vp=[('/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/baseline/aligned_santa_baseline_seed0.mp4','RW'),('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/formal_runs/santa_correct_v3d_seed000_replayA_2re/santa_correct_v3d_seed000.mp4','DT-FULL'),(str(B/'outputs/wm0_container_seed0/santa_correct_v3d_seed000.mp4'),'WM-0'),(str(B/'outputs/frag_container_seed0/santa_correct_v3d_seed000.mp4'),'DT-FRAG-PRUNE'),(str(B/'outputs/grid100_container_seed0/santa_correct_v3d_seed000.mp4'),'DT-GRID100-CENTER')]
sheet=np.zeros((464*5,832*5,3),np.uint8)
for r,(p,n) in enumerate(vp):
 c=cv2.VideoCapture(p);frames=[]
 while 1:
  ok,x=c.read()
  if not ok:break
  frames.append(x)
 c.release()
 for col,i in enumerate([0,20,40,60,80]):
  x=frames[i];cv2.putText(x,n,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);sheet[r*464:(r+1)*464,col*832:(col+1)*832]=x
cv2.imwrite(str(R/'FIVE_METHOD_CONTACT_SHEET.png'),sheet)
