import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

raw = {a: json.loads((R / 'SECONDARY_RAW' / a / 'raw.json').read_text()) for a in ('C1', 'C2', 'CS')}
values = {a: raw[a]['value'] for a in raw}
out = {
    'metric': 'TC-MAR', 'seed': 1, 'N': 1257,
    'values': values,
    'video_sha256': {a: raw[a]['video_sha256'] for a in raw},
    'authoritative_evaluator_sha256': raw['C1']['evaluator_sha256'],
    'raw_result_paths': {a: str(R / 'SECONDARY_RAW' / a / 'raw.json') for a in raw},
    'deltas': {
        'transport_seed1': values['C2'] - values['C1'],
        'identity_seed1': values['C2'] - values['CS'],
        'wrong_identity_vs_off_seed1': values['CS'] - values['C1'],
    },
    'TCMAR_INDEPENDENCE_FROM_PREVIEW': 'NOT_ESTABLISHED',
    'does_not_modify_primary_tcme_conclusion': True,
}
(R / 'SEED1_SECONDARY_TCMAR_RESULTS.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps({'secondary_sha256': sha(R / 'SEED1_SECONDARY_TCMAR_RESULTS.json')}))
