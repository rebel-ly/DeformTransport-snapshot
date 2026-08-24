import hashlib
import importlib.util
import json
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = Path('/workspace/DeformTransport')
E = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
V = {a: Path(v['path']) for a, v in json.loads((R / 'SEED1_EVALUATION_CONTRACT.json').read_text())['candidate_videos'].items()}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

for arm in ('C1', 'C2', 'CS'):
    d = R / 'SECONDARY_RAW' / arm
    d.mkdir(parents=True, exist_ok=True)
    try:
        suite = d / 'suite'
        target = suite / 'santa/dt_full/santa_dt_full_correct_seed0.mp4'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink(): target.unlink()
        target.symlink_to(V[arm])
        s = importlib.util.spec_from_file_location('frozen_ev', E)
        ev = importlib.util.module_from_spec(s); s.loader.exec_module(ev); ev.CANDIDATES = ['dt_full']
        report = ev.appearance_case(ROOT, suite, 'santa')
        payload = {'arm': arm, 'metric': 'TC-MAR', 'value': report['methods']['dt_full']['tc_mar_lab']['mean'], 'evaluator_sha256': sha(E), 'video_sha256': sha(V[arm])}
        (d / 'raw.json').write_text(json.dumps(payload, indent=2) + '\n')
        (d / 'exit_code.txt').write_text('0\n')
    except Exception as x:
        (d / 'stderr.txt').write_text(repr(x) + '\n')
        (d / 'exit_code.txt').write_text('1\n')
        raise
