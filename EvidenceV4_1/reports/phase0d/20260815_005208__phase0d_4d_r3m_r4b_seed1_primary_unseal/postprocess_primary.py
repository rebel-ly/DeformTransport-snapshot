import hashlib
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent
R4A = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260815_004423__phase0d_4d_r3m_r4a_evaluator_lineage_recovery/SEED0_AUTHORITATIVE_METRIC_PROVENANCE.json')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

contract = json.loads((R / 'SEED1_EVALUATION_CONTRACT.json').read_text())
seed1_raw = {a: json.loads((R / 'PRIMARY_RAW' / a / 'raw.json').read_text()) for a in ('C1', 'C2', 'CS')}
seed1 = {a: seed1_raw[a]['value'] for a in seed1_raw}
primary = {
    'metric': 'TC-ME', 'seed': 1, 'N': 1257,
    'values': seed1,
    'video_sha256': {a: seed1_raw[a]['video_sha256'] for a in seed1_raw},
    'authoritative_evaluator': contract['authoritative_evaluator'],
    'evaluation_ids': contract['evaluation_ids'],
    'raw_result_paths': contract['primary_raw_paths'],
    'full_precision_preserved': True,
}
(R / 'SEED1_PRIMARY_TCME_RESULTS.json').write_text(json.dumps(primary, indent=2, sort_keys=True) + '\n')
seed0 = json.loads(R4A.read_text())['tcme']
d = {
    'transport_seed0': seed0['C2'] - seed0['C1'], 'transport_seed1': seed1['C2'] - seed1['C1'],
    'identity_seed0': seed0['C2'] - seed0['CS'], 'identity_seed1': seed1['C2'] - seed1['CS'],
    'wrong_identity_vs_off_seed0': seed0['CS'] - seed0['C1'], 'wrong_identity_vs_off_seed1': seed1['CS'] - seed1['C1'],
}
transport = [d['transport_seed0'], d['transport_seed1']]
identity = [d['identity_seed0'], d['identity_seed1']]
out = {
    'seed0_provenance_path': str(R4A), 'seed0_tcme': seed0, 'seed1_tcme': seed1, 'deltas': d,
    'seed1_three_arm_order': [a for a, _ in sorted(seed1.items(), key=lambda x: x[1])],
    'transport_direction_seed1': 'PASS' if seed1['C2'] < seed1['C1'] else 'FAIL',
    'identity_direction_seed1': 'PASS' if seed1['C2'] < seed1['CS'] else 'FAIL',
    'wrong_identity_harm_seed1': 'PASS' if seed1['CS'] > seed1['C1'] else 'FAIL',
    'transport_direction_consistency_2seeds': 'PASS' if all(x < 0 for x in transport) else 'FAIL',
    'identity_direction_consistency_2seeds': 'PASS' if all(x < 0 for x in identity) else 'FAIL',
    'wrong_identity_harm_consistency_2seeds': 'PASS' if d['wrong_identity_vs_off_seed0'] > 0 and d['wrong_identity_vs_off_seed1'] > 0 else 'FAIL',
    'strong_three_arm_order_replication_2seeds': 'PASS' if seed0['C2'] < seed0['C1'] < seed0['CS'] and seed1['C2'] < seed1['C1'] < seed1['CS'] else 'FAIL',
    'transport_paired_delta_mean_2seeds': statistics.mean(transport), 'transport_paired_delta_sample_std_2seeds': statistics.stdev(transport),
    'identity_paired_delta_mean_2seeds': statistics.mean(identity), 'identity_paired_delta_sample_std_2seeds': statistics.stdev(identity),
    'N_SEEDS': 2, 'NO_SIGNIFICANCE_TEST': True, 'NO_EQUIVALENCE_TEST': True, 'NO_EQUIVALENCE_CLAIM': True,
}
(R / 'SEED0_SEED1_MECHANISM_REPLICATION.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(json.dumps({'primary_sha256': sha(R / 'SEED1_PRIMARY_TCME_RESULTS.json'), 'mechanism_sha256': sha(R / 'SEED0_SEED1_MECHANISM_REPLICATION.json')}))
