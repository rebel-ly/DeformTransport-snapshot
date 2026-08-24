#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

Q4_EDGE=0.5810624591086517

def finite_mean(x):
 x=np.asarray(x); x=x[np.isfinite(x)]; return None if not len(x) else float(x.mean())
def summarize(mask,mar,me):
 mask=np.asarray(mask,bool); valid=mask&mar['valid']; rwc=np.nanmean(me['rw_epe'],axis=0); dtc=np.nanmean(me['dt_epe'],axis=0); mv=mask&np.isfinite(rwc)&np.isfinite(dtc)
 return {'N':int(mask.sum()),'MAR_N':int(valid.sum()),'RW_MAR':finite_mean(mar['rw_lab'][valid]),'DT_MAR':finite_mean(mar['dt_lab'][valid]),'MAR_GAP':finite_mean(mar['delta_lab'][valid]),'ME_N':int(mv.sum()),'RW_ME':finite_mean(rwc[mv]),'DT_ME':finite_mean(dtc[mv]),'ME_GAP':finite_mean((dtc-rwc)[mv])}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mar',required=True); ap.add_argument('--me',required=True); ap.add_argument('--join',required=True); ap.add_argument('--operator',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out)
 mar=dict(np.load(a.mar)); me=dict(np.load(a.me)); j=dict(np.load(a.join)); assert len(j['material_id'])==1257
 sw=j['visibility_switch_count']; vis=j['visible_slot_count']; q4=j['trajectory_energy_3d']>Q4_EDGE; stable=sw<3; fragmented=~stable; zero=sw==0; zero_vis=vis==0; pos_vis=vis>0
 always_visible=np.all(np.load('/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy')[0],axis=0)
 groups={'ZERO_SWITCH_TOTAL':zero,'ZERO_SWITCH_ZERO_VISIBLE':zero&zero_vis,'ZERO_SWITCH_POSITIVE_VISIBLE':zero&pos_vis,'ZERO_SWITCH_ALWAYS_VISIBLE':zero&pos_vis&always_visible,'ZERO_SWITCH_PARTIAL_VISIBLE_NO_SWITCH':zero&pos_vis&~always_visible}
 dec={k:summarize(v,mar,me) for k,v in groups.items()}; (out/'r1_zero_switch_decomposition.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY','groups':dec},indent=2)+'\n')
 cross={'NON_Q4_STABLE':(~q4)&stable,'NON_Q4_FRAGMENTED':(~q4)&fragmented,'Q4_STABLE':q4&stable,'Q4_FRAGMENTED':q4&fragmented,'ZERO_FUTURE_VISIBLE':zero_vis}
 table={k:summarize(v,mar,me) for k,v in cross.items()}; (out/'r1_motion_switch_cross_table.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY','Q4_EDGE_FROZEN':Q4_EDGE,'groups':table},indent=2)+'\n')
 z=dec['ZERO_SWITCH_POSITIVE_VISIBLE']; survives=z['MAR_GAP'] is not None and z['MAR_GAP']<0
 qs,qf=table['Q4_STABLE'],table['Q4_FRAGMENTED']; relation='D_INSUFFICIENT_SAMPLE_OR_UNRESOLVED'
 if qs['N'] and qf['N']:
  stable_bad=(qs['MAR_GAP'] or 0)>0 or (qs['ME_GAP'] or 0)>0; frag_worse=(qf['MAR_GAP'] or 0)>(qs['MAR_GAP'] or 0) and (qf['ME_GAP'] or 0)>(qs['ME_GAP'] or 0)
  relation='C_BOTH_ASSOCIATED' if stable_bad and frag_worse else 'A_FRAGMENTATION_DOMINATES_WITHIN_HIGH_MOTION' if frag_worse else 'B_HIGH_MOTION_REMAINS_BAD_WHEN_STABLE' if stable_bad else relation
 pos={'ZERO_SWITCH_POSITIVE_VISIBLE_MAR_GAP':z['MAR_GAP'],'ZERO_SWITCH_POSITIVE_VISIBLE_ME_GAP':z['ME_GAP'],'POSITIVE_SUBGROUP_SURVIVES_ZERO_VISIBLE_CONTROL':survives,'Q4_STABLE_MAR_GAP':qs['MAR_GAP'],'Q4_FRAGMENTED_MAR_GAP':qf['MAR_GAP'],'Q4_STABLE_ME_GAP':qs['ME_GAP'],'Q4_FRAGMENTED_ME_GAP':qf['ME_GAP'],'MOTION_FRAGMENTATION_RELATION':relation,'FORMAL_POSITIVE_SUBGROUP_EVIDENCE':'PRESENT' if survives else 'NOT_CONFIRMED'}
 (out/'r1_positive_subgroup_summary.json').write_text(json.dumps(pos,indent=2)+'\n')
 retain=stable; removed=~retain; audit=json.load(open(a.operator)); slots=audit['per_slot']['Correct']; keep=set(int(x) for x in j['material_id'][retain]); support=0
 for row in slots: support+=sum(1 for mid in row['winner_material_ids_by_cell'].values() if int(mid) in keep)
 frag={'FRAG_PRUNE_RULE':'retain switch_count < 3','FULL_LISTED_K':1257,'FRAG_PRUNE_RETAINED_K':int(retain.sum()),'FRAG_PRUNE_REMOVED_K':int(removed.sum()),'FRAG_PRUNE_RETAINED_FRACTION':float(retain.mean()),'RETAINED_ACTIVE_VISIBLE_K':int((retain&pos_vis).sum()),'REMOVED_ACTIVE_VISIBLE_K':int((removed&pos_vis).sum()),'FRAG_PRUNE_WRITE_SUPPORT':support,'FRAG_PRUNE_MEAN_WRITES_PER_SLOT':support/20,'FRAG_PRUNE_INTERVENTION_FRACTION':support/(20*60*104),'verification':'count Phase0C frozen FULL winners whose exact material ID is retained; no new operator/video artifact'}
 (out/'frag_prune_precompute.json').write_text(json.dumps(frag,indent=2)+'\n'); print(json.dumps({**pos,**frag},sort_keys=True))
if __name__=='__main__':main()
