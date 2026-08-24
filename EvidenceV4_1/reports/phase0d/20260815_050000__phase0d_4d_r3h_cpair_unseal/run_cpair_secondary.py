import importlib.util, json, sys
from pathlib import Path
ROOT=Path('/workspace/DeformTransport'); R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_050000__phase0d_4d_r3h_cpair_unseal'); E=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
arm=sys.argv[1]; video=Path(sys.argv[2]); suite=R/('suite_'+arm); target=suite/'santa/dt_full/santa_dt_full_correct_seed0.mp4'; target.parent.mkdir(parents=True,exist_ok=True)
if target.exists() or target.is_symlink(): target.unlink()
target.symlink_to(video)
s=importlib.util.spec_from_file_location('frozen_ev',E); ev=importlib.util.module_from_spec(s); s.loader.exec_module(ev); ev.CANDIDATES=['dt_full']
report=ev.appearance_case(ROOT,suite,'santa'); m=report['methods']['dt_full']['tc_mar_lab']['mean']
payload={'arm':arm,'metric':'TC-MAR','evaluator_sha256':'e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5','value':m,'candidate_mp4_sha256':report['methods']['dt_full']['sha256'],'cross_baseline_interpretation':'NOT_OPENED'}
(R/(arm+'_tcmar_secondary.json')).write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
