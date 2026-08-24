import csv, hashlib, json, math, statistics
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent; RAW=R/'PER_TRACK_RAW'
accepted={'seed0_C1':0.47438763126111494,'seed0_C2':0.42167144903579584,'seed0_CS':0.5138549546231557,'seed1_C1':0.39610003385941484,'seed1_C2':0.39736386332028023,'seed1_CS':0.40577810212784815}
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def load(k):
 d={}
 with (RAW/(k+'.csv')).open() as f:
  for x in csv.DictReader(f): d.setdefault(int(x['material_id']),[]).append(float(x['epe']))
 return d
def summarise(x):
 q=np.percentile(x,[5,25,50,75,95]); return {'N':int(len(x)),'mean':float(np.mean(x)),'median':float(np.median(x)),'std':float(np.std(x,ddof=1)),'min':float(np.min(x)),'p05':float(q[0]),'p25':float(q[1]),'p50':float(q[2]),'p75':float(q[3]),'p95':float(q[4]),'max':float(np.max(x)),'fraction_lt_zero':float(np.mean(x<0)),'fraction_gt_zero':float(np.mean(x>0)),'fraction_eq_zero':float(np.mean(x==0))}
def rank(a):
 order=np.argsort(a);r=np.empty(len(a),float);r[order]=np.arange(len(a));return r
def corr(a,b):
 return {'pearson':float(np.corrcoef(a,b)[0,1]),'spearman':float(np.corrcoef(rank(a),rank(b))[0,1])}
def boot(a):
 rng=np.random.default_rng(0); n=len(a); means=np.empty(10000)
 for i in range(10000): means[i]=a[rng.integers(0,n,n)].mean()
 return {'mean':float(a.mean()),'p2_5':float(np.percentile(means,2.5)),'p97_5':float(np.percentile(means,97.5)),'repetitions':10000,'rng_seed':0,'resampling_unit':'material track'}

arms={k:load(k) for k in accepted}; ids=set.intersection(*(set(v) for v in arms.values())); bad={k:sorted(set(v)^ids) for k,v in arms.items()}; supports={k:{i:len(v[i]) for i in ids} for k,v in arms.items()}; support_equal=all(supports[k]==supports['seed0_C1'] for k in supports)
vals={k:{i:float(np.mean(v[i])) for i in ids} for k,v in arms.items()}; ids=sorted(ids)
dt0=np.array([vals['seed0_C2'][i]-vals['seed0_C1'][i] for i in ids]);dt1=np.array([vals['seed1_C2'][i]-vals['seed1_C1'][i] for i in ids]);di0=np.array([vals['seed0_C2'][i]-vals['seed0_CS'][i] for i in ids]);di1=np.array([vals['seed1_C2'][i]-vals['seed1_CS'][i] for i in ids]);dw0=np.array([vals['seed0_CS'][i]-vals['seed0_C1'][i] for i in ids]);dw1=np.array([vals['seed1_CS'][i]-vals['seed1_C1'][i] for i in ids])
with (R/'PER_TRACK_DELTAS.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['material_id','transport_delta_seed0','transport_delta_seed1','identity_delta_seed0','identity_delta_seed1','wrong_vs_off_seed0','wrong_vs_off_seed1']);w.writerows(zip(ids,dt0,dt1,di0,di1,dw0,dw1))
validation={'PER_TRACK_EXTRACTION_BATCH_STATUS':'PASS_NORMAL_COMPLETION','accepted_provenance':{'seed0':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_004423__phase0d_4d_r3m_r4a_evaluator_lineage_recovery/SEED0_AUTHORITATIVE_METRIC_PROVENANCE.json','seed1':'/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_005208__phase0d_4d_r3m_r4b_seed1_primary_unseal/SEED1_PRIMARY_TCME_RESULTS.json'},'six_arm':{}}
for k,v in accepted.items():
 x=json.loads((RAW/(k+'.json')).read_text());validation['six_arm'][k]={'reaggregated_tcme':x['aggregate_from_retained_rows_formal_transition_mean'],'authoritative_tcme':v,'abs_diff':abs(x['aggregate_from_retained_rows_formal_transition_mean']-v),'rel_diff':0.0 if v==0 else abs(x['aggregate_from_retained_rows_formal_transition_mean']-v)/abs(v),'exact_reproduction':x['exact_reproduction'],'rows':x['rows']}
validation['PER_TRACK_EXTRACTION_VALIDITY']='PASS' if all(x['exact_reproduction'] for x in validation['six_arm'].values()) else 'FAIL';validation['PER_TRACK_ID_ALIGNMENT']='PASS' if len(ids)==1257 and not any(bad.values()) and support_equal else 'FAIL';validation['N_ALIGNED_MATERIAL_IDS']=len(ids);validation['support_count_equality']=support_equal;validation['missing_id_sets']=bad
(R/'PER_TRACK_EXTRACTION_VALIDATION.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n')
dist={'POST_PRIMARY_EXPLORATORY_ANALYSIS':True,'transport_seed0':summarise(dt0),'transport_seed1':summarise(dt1),'identity_seed0':summarise(di0),'identity_seed1':summarise(di1),'wrong_vs_off_seed0':summarise(dw0),'wrong_vs_off_seed1':summarise(dw1),'formal_aggregate_deltas':{'transport_seed0':accepted['seed0_C2']-accepted['seed0_C1'],'transport_seed1':accepted['seed1_C2']-accepted['seed1_C1'],'identity_seed0':accepted['seed0_C2']-accepted['seed0_CS'],'identity_seed1':accepted['seed1_C2']-accepted['seed1_CS']},'SEED1_TRANSPORT_FLIP_PATTERN':'BROAD_POSITIVE_SHIFT' if np.mean(dt1)>0 and np.median(dt1)>0 and np.mean(dt1>0)>.5 else ('TAIL_DRIVEN_MEAN_REVERSAL' if np.mean(dt1)>0 and np.median(dt1)<0 else 'MIXED_OR_OTHER'),'IDENTITY_BENEFIT_DISTRIBUTION':'MAJORITY_TRACK_DIRECTION_CONSISTENT_ACROSS_BOTH_SEEDS' if np.median(di0)<0 and np.mean(di0<0)>.5 and np.median(di1)<0 and np.mean(di1<0)>.5 else 'NOT_MAJORITY_DIRECTION_CONSISTENT_ACROSS_BOTH_SEEDS'}
(R/'PER_TRACK_DISTRIBUTION_SUMMARY.json').write_text(json.dumps(dist,indent=2,sort_keys=True)+'\n')
assoc={'association_interpretation':'FIXED_MATERIAL_TRACK_CROSS_SEED_ASSOCIATION','transport':corr(dt0,dt1),'identity':corr(di0,di1),'wrong_vs_off':corr(dw0,dw1),'not_diffusion_seed_uncertainty_estimate':True}
(R/'PER_TRACK_CROSS_SEED_ASSOCIATION.json').write_text(json.dumps(assoc,indent=2,sort_keys=True)+'\n')
boots={'transport_seed0':boot(dt0),'transport_seed1':boot(dt1),'identity_seed0':boot(di0),'identity_seed1':boot(di1),'wrong_vs_off_seed0':boot(dw0),'wrong_vs_off_seed1':boot(dw1),'TRACK_BOOTSTRAP_INTERPRETATION':'CONDITIONAL_ON_FIXED_SCENE_AND_FIXED_DIFFUSION_REALIZATION','NOT_A_DIFFUSION_SEED_UNCERTAINTY_INTERVAL':True,'NO_CROSS_SEED_SIGNIFICANCE_CLAIM':True,'TRACKS_NOT_ASSUMED_TO_BE_INDEPENDENT_SCENE_REPLICATES':True}
(R/'PER_TRACK_BOOTSTRAP_CONDITIONAL.json').write_text(json.dumps(boots,indent=2,sort_keys=True)+'\n')
print(json.dumps({'validation':validation,'distribution':dist,'association':assoc,'bootstrap':boots},sort_keys=True))
