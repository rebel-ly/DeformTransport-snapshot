import hashlib, importlib.util, json, os, sys
from pathlib import Path
R=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_005208__phase0d_4d_r3m_r4b_seed1_primary_unseal')
ROOT=Path('/workspace/DeformTransport')
E=Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
V={'C1':Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/c1_seed1_gpu0/c1_preview_k0_v3d_seed001.mp4'),'C2':Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/c2_seed1_gpu0/c2_preview_correct_v3d_seed001.mp4'),'CS':Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/cs_seed1_gpu0/cs_preview_shuffled_v3d_seed001.mp4')}
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
for arm in ('C1','C2','CS'):
 d=R/'PRIMARY_RAW'/arm;d.mkdir(parents=True,exist_ok=True)
 try:
  suite=d/'suite';target=suite/'santa/dt_full/santa_dt_full_correct_seed0.mp4';target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() or target.is_symlink(): target.unlink()
  target.symlink_to(V[arm]);spec=importlib.util.spec_from_file_location('frozen_ev',E);ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev);ev.CANDIDATES=['dt_full'];report=ev.motion_case(ROOT,suite,'santa',batch=8)
  out={'arm':arm,'metric':'TC-ME','value':report['methods']['dt_full']['transition_mean_epe_mean'],'evaluator_sha256':sha(E),'video_sha256':sha(V[arm])};(d/'raw.json').write_text(json.dumps(out,indent=2)+'\n');(d/'exit_code.txt').write_text('0\n')
 except Exception as x:
  (d/'stderr.txt').write_text(repr(x)+'\n');(d/'exit_code.txt').write_text('1\n');raise
