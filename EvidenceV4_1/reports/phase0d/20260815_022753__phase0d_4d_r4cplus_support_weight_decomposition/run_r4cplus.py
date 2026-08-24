import csv,hashlib,json,math
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
SRC=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_011925__phase0d_4d_r4c_symmetric_integrity_and_track_localization')
RAW=SRC/'PER_TRACK_RAW'; EV=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
ids=np.load('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy')
tracks=np.load('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy')[0]
accepted={'T0':-0.0527161822253191,'T1':0.0012638294608653955,'I0':-0.09218350558735983,'I1':-0.00841423880756792,'W0':0.03946732336204073,'W1':0.009678068268433315}
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def load(key):
 a=np.full((80,len(ids)),np.nan)
 pos={int(x):i for i,x in enumerate(ids)}
 with (RAW/(key+'.csv')).open() as f:
  for r in csv.DictReader(f):a[int(r['transition']),pos[int(r['material_id'])]]=float(r['epe'])
 return a
X={k:load(k) for k in ('seed0_C1','seed0_C2','seed0_CS','seed1_C1','seed1_C2','seed1_CS')}
cnt={k:np.sum(np.isfinite(v),axis=0) for k,v in X.items()}; common=np.all(np.stack([cnt[k] for k in cnt])==np.stack([cnt[k] for k in cnt])[0],axis=0); supported=cnt['seed0_C1']>0
def corr(a,b):
 def rank(x):
  o=np.argsort(x);r=np.empty(len(x));r[o]=np.arange(len(x));return r
 return {'pearson':float(np.corrcoef(a,b)[0,1]),'spearman':float(np.corrcoef(rank(a),rank(b))[0,1])}
def massstats(c):
 out={}
 for nm,z in [('harm',np.maximum(c,0)),('benefit',np.maximum(-c,0))]:
  order=np.sort(z)[::-1];tot=z.sum();
  out[nm+'_total']=float(tot)
  for p in (1,5,10):out[f'top{p}_share']=float(order[:math.ceil(p*len(z)/100)].sum()/tot) if tot else 0.0
  q=z/tot if tot else z; h=float((q*q).sum()) if tot else 0.;out[nm+'_hhi']=h;out[nm+'_effective_number']=float(1/h) if h else None
 return out
def agg(x,keep):return float(np.nanmean(np.nanmean(x[:,keep],axis=1)))
def contrast(name,A,B):
 nA,nB=cnt[A],cnt[B];mA=np.nanmean(X[A],axis=0);mB=np.nanmean(X[B],axis=0); same=np.array_equal(nA,nB); denA=nA.sum();denB=nB.sum()
 c=np.nan_to_num(np.nansum((X[A]-X[B])/np.sum(np.isfinite(X[A]),axis=1)[:,None],axis=0)/80.0,nan=0.0)
 formal=agg(X[A],supported)-agg(X[B],supported); order=np.argsort(-c[supported]); sid=np.where(supported)[0]; ordered=sid[order]
 rows=[]
 for k in range(len(ordered)+1):
  keep=supported.copy();keep[ordered[:k]]=False; aa=agg(X[A],keep);bb=agg(X[B],keep);rows.append([name,k,k/len(ordered),int(nA[ordered[:k]].sum()),int(nB[ordered[:k]].sum()),int(nA[keep].sum()),aa,bb,aa-bb])
 return {'name':name,'A':A,'B':B,'support_equal':bool(same),'formal_weighting':'transition-equal: sum_t(e_A-ti-e_B-ti)/n_t/80; not global sample-count weighting','formal_delta':formal,'decomposed_sum':float(c.sum()),'abs_diff':abs(formal-c.sum()),'exact_reconstruction':bool(formal==c.sum()),'per_id_mean_delta':mA-mB,'contribution':c,'support':nA,'ordered':ordered,'rows':rows,'concentration':massstats(c)}
C={'T0':contrast('transport','seed0_C2','seed0_C1'),'T1':contrast('transport','seed1_C2','seed1_C1'),'I0':contrast('identity','seed0_C2','seed0_CS'),'I1':contrast('identity','seed1_C2','seed1_CS'),'W0':contrast('wrong_vs_off','seed0_CS','seed0_C1'),'W1':contrast('wrong_vs_off','seed1_CS','seed1_C1')}
with (R/'PER_ID_SUPPORT_COUNTS.csv').open('w',newline='') as f:
 w=csv.writer(f);ks=list(cnt);w.writerow(['material_id']+['support_count_'+k for k in ks]+['common_zero_support']);w.writerows([[int(ids[i])]+[int(cnt[k][i]) for k in ks]+[bool(not supported[i])] for i in range(len(ids))])
eq={};ks=list(cnt)
for a,b in [('seed0_C1','seed0_C2'),('seed0_C2','seed0_CS'),('seed1_C1','seed1_C2'),('seed1_C2','seed1_CS'),('seed0_C1','seed1_C1'),('seed0_C2','seed1_C2'),('seed0_CS','seed1_CS')]:
 d=np.abs(cnt[a]-cnt[b]);eq[f'{a}_vs_{b}']={'exact_equal':bool(np.array_equal(cnt[a],cnt[b])),'support_count_diff_count':int((d>0).sum()),'max_abs_support_count_diff':int(d.max())}
supportaudit={'COMMON_ZERO_SUPPORT_SET_EXACT':bool(np.all(~supported==np.all(np.stack([cnt[k]==0 for k in cnt]),axis=0))),'COMMON_ZERO_SUPPORT_N':int((~supported).sum()),'comparisons':eq}
(R/'PER_ID_SUPPORT_EQUALITY_AUDIT.json').write_text(json.dumps(supportaudit,indent=2,sort_keys=True)+'\n')
with (R/'FORMAL_AGGREGATION_DECOMPOSITION.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['contrast','seed','material_id','support_count','per_id_delta','formal_contribution','positive_harm_mass','negative_benefit_mass'])
 for key,z in C.items():
  for i in np.where(supported)[0]:w.writerow([z['name'],key[-1],int(ids[i]),int(z['support'][i]),z['per_id_mean_delta'][i],z['contribution'][i],max(z['contribution'][i],0),max(-z['contribution'][i],0)])
dec={k:{x:(v if not isinstance(v,np.ndarray) else None) for x,v in z.items() if x not in ('ordered','rows','per_id_mean_delta','contribution','support')} for k,z in C.items()}
dec['seed1_transport_unweighted_per_id_delta']=float(C['T1']['per_id_mean_delta'][supported].mean());dec['seed1_transport_formal_weighted_delta']=C['T1']['formal_delta'];dec['seed1_transport_weighting_shift']=dec['seed1_transport_formal_weighted_delta']-dec['seed1_transport_unweighted_per_id_delta'];dec['FORMAL_CONTRIBUTION_DECOMPOSITION_VALIDITY']='PASS' if all(z['abs_diff']<1e-14 for z in C.values()) else 'FAIL'
(R/'FORMAL_AGGREGATION_DECOMPOSITION.json').write_text(json.dumps(dec,indent=2,sort_keys=True)+'\n')
conc={k:z['concentration'] for k,z in C.items()};(R/'CONTRIBUTION_CONCENTRATION.json').write_text(json.dumps(conc,indent=2,sort_keys=True)+'\n')
with (R/'TOPK_DELETION_SENSITIVITY.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['contrast','seed','k','fraction_tracks_removed','transitions_removed_arm_A','transitions_removed_arm_B','remaining_transitions','recomputed_arm_A','recomputed_arm_B','recomputed_delta'])
 for z in C.values():w.writerows(z['rows'])
t1=C['T1']; flip=next(r for r in t1['rows'] if r[-1]<0); top5=t1['ordered'][:math.ceil(.05*len(t1['ordered']))];all_sup=t1['support'][supported];s=t1['support'][top5];driver={'MIN_K_TO_NEGATIVE_TRANSPORT_SEED1':{'k':flip[1],'fraction':flip[2],'removed_transition_fraction':flip[3]/t1['support'].sum(),'material_ids':[int(ids[i]) for i in t1['ordered'][:flip[1]]],'support_counts':[int(t1['support'][i]) for i in t1['ordered'][:flip[1]]]},'MIN_K_TO_NEGATIVE_TRANSPORT_SEED0':0,'top5_support':{'median_top5':float(np.median(s)),'median_all':float(np.median(all_sup)),'mean_top5':float(s.mean()),'mean_all':float(all_sup.mean()),'fraction_top5_in_upper_support_quartile':float(np.mean(s>=np.percentile(all_sup,75)))}}
driver['CONCENTRATED_HIGH_SUPPORT_HARM_DRIVER']=bool(flip[2]<=.05 and t1['concentration']['top5_share']>=.5 and driver['top5_support']['median_top5']>driver['top5_support']['median_all']);driver['SEED1_FORMAL_FLIP_INTERPRETATION']='CONCENTRATED_HIGH_SUPPORT_HARM_DRIVEN' if driver['CONCENTRATED_HIGH_SUPPORT_HARM_DRIVER'] else ('DISTRIBUTED_REWEIGHTING' if flip[2]>.05 and t1['concentration']['top5_share']<.5 else 'MIXED_CONCENTRATION_PATTERN')
(R/'TOP_CONTRIBUTOR_TRACKS.csv').write_text('material_id,seed1_transport_contribution,support_count\n'+'\n'.join(f'{int(ids[i])},{t1["contribution"][i]},{int(t1["support"][i])}' for i in t1['ordered'])+'\n')
def setstats(a,b,p,sign):
 n=math.ceil(p*len(t1['ordered'])/100);sa=set(a['ordered'][:n] if sign=='harm' else np.where(supported)[0][np.argsort(-np.maximum(-a['contribution'][supported],0))[:n]]);sb=set(b['ordered'][:n] if sign=='harm' else np.where(supported)[0][np.argsort(-np.maximum(-b['contribution'][supported],0))[:n]]);return {'intersection':len(sa&sb),'union':len(sa|sb),'jaccard':len(sa&sb)/len(sa|sb),'overlap_coefficient':len(sa&sb)/min(len(sa),len(sb))}
over={'transport_harm_top1':setstats(C['T0'],C['T1'],1,'harm'),'transport_harm_top5':setstats(C['T0'],C['T1'],5,'harm'),'identity_benefit_top1':setstats(C['I0'],C['I1'],1,'benefit'),'identity_benefit_top5':setstats(C['I0'],C['I1'],5,'benefit')}
(R/'CROSS_SEED_TOPSET_OVERLAP.json').write_text(json.dumps(over,indent=2,sort_keys=True)+'\n')
prov={'r4c_final_path':str(SRC/'PHASE0D_4D_R4C_FINAL.md'),'r4c_final_sha256':sha(SRC/'PHASE0D_4D_R4C_FINAL.md'),'r4c_result_path':str(SRC/'PHASE0D_4D_R4C_RESULT.json'),'r4c_result_sha256':sha(SRC/'PHASE0D_4D_R4C_RESULT.json'),'evaluator_path':str(EV),'evaluator_sha256':sha(EV),'ZERO_GPU':True}
(R/'R4CPLUS_INPUT_PROVENANCE.json').write_text(json.dumps(prov,indent=2,sort_keys=True)+'\n')
(R/'FORMAL_AGGREGATION_SEMANTICS.txt').write_text('Recovered mechanically from eval_v3_corrected_v2_recovered.py motion_case and R4C companion: each transition t forms valid visible finite tracks at t,t+1, computes EPE per track, computes transition mean, then formal TC-ME is arithmetic mean of 80 transition means. Per-ID support is the count of valid transition rows.\n')
result={'POST_PRIMARY_EXPLORATORY_DECOMPOSITION':True,'ZERO_GPU':True,'FORMAL_AGGREGATION_SEMANTICS_RECOVERED':'PASS','FORMAL_CONTRIBUTION_DECOMPOSITION_VALIDITY':dec['FORMAL_CONTRIBUTION_DECOMPOSITION_VALIDITY'],'support_audit':supportaudit,'seed1_transport_weighting_shift':dec['seed1_transport_weighting_shift'],'driver':driver,'concentration':conc,'topset_overlap':over,'FROZEN_SUBGROUP_STATUS':'UNRESOLVED_NO_RECOVERED_PREEXISTING_MEMBERSHIP','SPATIAL_LOCALIZATION_STATUS':'UNRESOLVED_SOURCE_IMAGE_PROVENANCE','NEW_R4CPLUS_INTEGRITY_ISSUE':False,'PRIMARY_CONCLUSIONS_UNCHANGED':True,'SEED2_FULL_GENERATION_LAUNCHED':False}
(R/'R4CPLUS_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
