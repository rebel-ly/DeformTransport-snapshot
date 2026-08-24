import hashlib
import json
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
E = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_075742__f1r4_corrected_v2_preflight_recovery/generated/eval_v3_corrected_v2_recovered.py')
IDS = Path('/mnt/sdbd/home/liuyu_qyh/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy')
V = {
    'C1': Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/c1_seed1_gpu0/c1_preview_k0_v3d_seed001.mp4'),
    'C2': Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/c2_seed1_gpu0/c2_preview_correct_v3d_seed001.mp4'),
    'CS': Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution/cs_seed1_gpu0/cs_preview_shuffled_v3d_seed001.mp4'),
}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def probe(p):
    cp = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(p)], check=True, text=True, capture_output=True)
    return json.loads(cp.stdout)['streams'][0]

payload = {
    'phase': 'Phase 0D-4D-R3M-R4B',
    'contract_frozen_before_primary_result_read': True,
    'authoritative_evaluator': {'path': str(E), 'sha256': sha(E)},
    'corrected_v2_N': 1257,
    'evaluation_ids': {'path': str(IDS), 'sha256': sha(IDS)},
    'primary_metric': 'TC-ME',
    'secondary_metric': 'TC-MAR',
    'blind_primary_order': ['C1', 'C2', 'CS'],
    'secondary_order': ['C1', 'C2', 'CS'],
    'candidate_videos': {a: {'path': str(p), 'sha256': sha(p), 'engineering_decode_verified_previously': True} for a, p in V.items()},
    'primary_raw_paths': {a: str(R / 'PRIMARY_RAW' / a / 'raw.json') for a in V},
}
(R / 'SEED1_EVALUATION_CONTRACT.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
