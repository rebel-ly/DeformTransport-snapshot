#!/usr/bin/env python3
"""Freeze F3 structural values using the exact Phase0C FULL winner maps."""
import json, hashlib
from pathlib import Path
import numpy as np
OUT=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze')
F2R=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_095748__f2r_positive_subgroup_validity')
OP=Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport_EvidenceV4_1/reports/phase0c/operator_structure/20260812_151837__santa_v3d/operator_structure_audit.json')
def main():
 s=json.loads((OUT/'structural_comparison.json').read_text()); op=json.loads(OP.read_text())['per_slot']['Correct']; prior=json.loads((F2R/'frag_prune_precompute.json').read_text())
 for name in ['FULL1257','frag_prune','grid100_center','count_matched_center']:
  if name=='FULL1257': ids=np.load('/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy')
  else: ids=np.load(s[name]['artifact']['ids']['path'])
  keep=set(map(int,ids)); frozen=sum(sum(1 for mid in slot['winner_material_ids_by_cell'].values() if int(mid) in keep) for slot in op)
  s[name]['frozen_full_winner_write_support']=frozen; s[name]['frozen_full_winner_mean_writes_per_slot']=frozen/20.; s[name]['frozen_full_winner_intervention_fraction']=frozen/124800.; s[name]['structural_metric_primary']='frozen_full_winner_write_support'
 frag=s['frag_prune']; ok=(len(np.load(frag['artifact']['ids']['path']))==prior['FRAG_PRUNE_RETAINED_K'] and frag['frozen_full_winner_write_support']==prior['FRAG_PRUNE_WRITE_SUPPORT'])
 s['f2r_frag_precompute_reproduced']='PASS' if ok else 'FAIL'; s['f2r_reproduction_definition']='exact count of existing Phase0C FULL winner_material_ids_by_cell whose material ID is in the frozen FRAG subset; no subset re-arbitration'
 (OUT/'structural_comparison.json').write_text(json.dumps(s,indent=2)+'\n')
 print(json.dumps({'reproduction':s['f2r_frag_precompute_reproduced'],'frag_frozen_support':frag['frozen_full_winner_write_support'],'prior':prior['FRAG_PRUNE_WRITE_SUPPORT']}))
if __name__=='__main__':main()
