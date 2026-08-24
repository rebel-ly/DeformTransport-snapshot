"""Run frozen corrected-v2 TC-ME one C arm at a time; retain only within-pair values."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path('/workspace/DeformTransport')
REPORT = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_050000__phase0d_4d_r3h_cpair_unseal')
EVAL = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
ARM = sys.argv[1]
VIDEO = Path(sys.argv[2])
SUITE = REPORT / ('suite_' + ARM)
TARGET = SUITE / 'santa/dt_full/santa_dt_full_correct_seed0.mp4'
TARGET.parent.mkdir(parents=True, exist_ok=True)
if TARGET.exists() or TARGET.is_symlink(): TARGET.unlink()
TARGET.symlink_to(VIDEO)
spec = importlib.util.spec_from_file_location('frozen_ev', EVAL)
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
ev.CANDIDATES = ['dt_full']
report = ev.motion_case(ROOT, SUITE, 'santa', batch=8)
value = report['methods']['dt_full']['transition_mean_epe_mean']
payload = {'arm': ARM, 'metric': 'TC-ME', 'evaluator_sha256': 'e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5', 'value': value, 'candidate_mp4_sha256': report['methods']['dt_full']['sha256'], 'cross_baseline_interpretation': 'NOT_OPENED'}
(REPORT / ('%s_tcme_primary.json' % ARM)).write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload))
