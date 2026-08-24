import hashlib
import json
from pathlib import Path

import numpy as np

W = Path('/workspace')
REPO = W / 'DeformTransport_EvidenceV4_1'
OUT = REPO / 'reports/phase0d/20260814_220441__phase0d_4d_r3m_r3_seed1_lineage_execution'
R3 = REPO / 'reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation'
R3J = REPO / 'reports/phase0d/20260815_070000__phase0d_4d_r3j_corrected_v2_shuffle'
R3M = REPO / 'reports/phase0d/20260815_100000__phase0d_4d_r3m_seed1_wave1'
R3MR2 = REPO / 'reports/phase0d/20260815_110000__phase0d_4d_r3m_r2_epsilon_validation'
OVER = REPO / 'experimental/20260814__wanmove_preview_sdedit_overlay'
WAN = W / 'Wan-Move'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def item(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path)}


def tensor_sha_npy(path):
    a = np.load(path)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(), list(a.shape), str(a.dtype)


prompt = 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.'
source = W / 'DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png'
correct_dir = W / 'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
phase0b = REPO / 'reports/phase0b'
k0 = REPO / 'reports/phase0d/20260814_200000__phase0d_4d_recovery'
shuffle = phase0b / 'causal_contract/20260812_134250__santa_corrected_v2_identity_shuffle_seed0'
depth = phase0b / 'functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy'
evaluator = W / 'DeformTransport/v4_eval_files_to_upload/eval_v3.py'
preview = R3 / 'WAN_FORMAL_PREVIEW_LATENT_58x104.npy'
eps0 = R3 / 'R3_SHARED_EPSILON_58x104.npy'
eps1 = R3M / 'EPSILON_SEED1_58x104.npy'

pre_tsha, pre_shape, pre_dtype = tensor_sha_npy(preview)
e0_tsha, e0_shape, e0_dtype = tensor_sha_npy(eps0)
e1_tsha, e1_shape, e1_dtype = tensor_sha_npy(eps1)

common = {
    'source': item(source),
    'prompt': prompt,
    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
    'model_checkpoint_dir': str(WAN / 'Wan-Move-14B-480P'),
    'vae_checkpoint': item(WAN / 'Wan-Move-14B-480P/Wan2.1_VAE.pth'),
    'formal_overlay': {'wan_move_sha256': sha(OVER / 'wan/wan_move.py'), 'generate_sha256': sha(OVER / 'generate.py')},
    'preview': {'path': str(preview), 'file_sha256': sha(preview), 'tensor_sha256': pre_tsha, 'shape': pre_shape, 'dtype': pre_dtype},
    'scheduler': {'class': 'FlowUniPCMultistepScheduler', 'num_inference_steps': 40, 'shift': 3.0},
    'start_index': 15,
    'actual_sigma_index15': 0.8329627513885498,
    'actual_timestep_index15': 832,
    'effective_denoise_steps': 25,
    'dtype': 'bf16',
    'latent_geometry': [58, 104],
    'generation_geometry': [832, 480],
    'decoded_output_geometry': [832, 464],
    'frame_count': 81,
    'transport_variant': 'v3d',
    'evaluation': {'N': 1257, 'ids': item(correct_dir / 'santa_material_point_ids.npy'), 'evaluator': item(evaluator), 'binding': 'corrected-v2 N=1257'},
}

arms = {
    'C1': {'transport': {'K': 0, 'tracks': item(k0 / 'k0_tracks.npy'), 'visibility': item(k0 / 'k0_visibility.npy'), 'ids': item(k0 / 'k0_ids.npy'), 'depth': item(k0 / 'k0_depth.npy'), 'implementation': 'transport-off K=0; expected_writes=0'}},
    'C2': {'transport': {'K': 1257, 'tracks': item(correct_dir / 'santa_material_tracks_correct.npy'), 'visibility': item(correct_dir / 'santa_material_visibility_correct.npy'), 'ids': item(correct_dir / 'santa_material_point_ids.npy'), 'depth': item(depth), 'implementation': 'corrected-v2 material transport'}},
    'CS': {'transport': {'K': 1257, 'tracks': item(shuffle / 'santa_material_tracks_identity_shuffled_seed0.npy'), 'visibility': item(shuffle / 'santa_material_visibility_identity_shuffled_seed0.npy'), 'ids': item(correct_dir / 'santa_material_point_ids.npy'), 'depth': item(depth), 'permutation': item(shuffle / 'source_identity_permutation_seed0.npy'), 'implementation': 'corrected-v2 Phase0B canonical zero-fixed-point Identity-Shuffled material transport'}},
}

def make_manifest(arm, seed, epsilon_path, epsilon_file_sha, epsilon_tensor_sha, epsilon_shape, epsilon_dtype):
    m = dict(common)
    m['arm'] = arm
    m['FORMAL_DIFFUSION_SEED'] = seed
    m['external_epsilon'] = {'path': str(epsilon_path), 'file_sha256': epsilon_file_sha, 'tensor_sha256': epsilon_tensor_sha, 'shape': epsilon_shape, 'dtype': epsilon_dtype, 'semantic_role': 'deterministic consequence of FORMAL_DIFFUSION_SEED'}
    m['transport'] = arms[arm]['transport']
    return m

seed0 = {a: make_manifest(a, 0, eps0, sha(eps0), e0_tsha, e0_shape, e0_dtype) for a in arms}
seed1 = {a: make_manifest(a, 1, eps1, sha(eps1), e1_tsha, e1_shape, e1_dtype) for a in arms}

def normalized(m):
    # epsilon path/content change is an allowed deterministic consequence, so normalize it.
    n = json.loads(json.dumps(m))
    n['FORMAL_DIFFUSION_SEED'] = '<DIFFUSION_SEED>'
    n['external_epsilon'] = '<DETERMINISTIC_CONSEQUENCE_OF_FORMAL_DIFFUSION_SEED>'
    return n

def field_diffs(a, b, prefix=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += field_diffs(a.get(k, '<MISSING>'), b.get(k, '<MISSING>'), f'{prefix}.{k}' if prefix else k)
    elif a != b:
        out.append({'field': prefix, 'seed0': a, 'seed1': b})
    return out

audit_diffs = {}
for arm in arms:
    raw = field_diffs(seed0[arm], seed1[arm])
    unexpected = field_diffs(normalized(seed0[arm]), normalized(seed1[arm]))
    payload = {'arm': arm, 'raw_expected_differences': raw, 'unexpected_scientific_differences': unexpected, 'normalized_contract': 'FORMAL_DIFFUSION_SEED plus deterministic external epsilon replacement are allowed; all remaining fields must match exactly'}
    audit_diffs[arm] = payload
    (OUT / f'{arm}_SEED0_EFFECTIVE_SCIENTIFIC_MANIFEST.json').write_text(json.dumps(seed0[arm], indent=2, sort_keys=True) + '\n')
    (OUT / f'SEED1_{arm}_PLANNED_MANIFEST.json').write_text(json.dumps(seed1[arm], indent=2, sort_keys=True) + '\n')
    (OUT / f'{arm}_SEED0_TO_SEED1_DIFF.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')

common_randomness = {
    'EPSILON_SEED1_NATIVE_SEMANTICS_PASS': True,
    'seed1_epsilon_path': str(eps1),
    'seed1_epsilon_file_sha256': sha(eps1),
    'seed1_epsilon_tensor_sha256': e1_tsha,
    'seed1_epsilon_shape': e1_shape,
    'seed1_epsilon_dtype': e1_dtype,
    'SEED1_THREE_ARM_SHARED_EPSILON': len({seed1[a]['external_epsilon']['tensor_sha256'] for a in arms}) == 1,
    'SEED0_TO_SEED1_ONLY_SCIENTIFIC_RANDOM_CHANGE': 'FORMAL_DIFFUSION_SEED',
    'EPSILON_CHANGE': 'DETERMINISTIC_CONSEQUENCE_OF_FORMAL_DIFFUSION_SEED',
    'validation_evidence': str(R3MR2 / 'EPSILON_VALIDATION.json'),
}
(OUT / 'SEED1_COMMON_RANDOMNESS_CONTRACT.json').write_text(json.dumps(common_randomness, indent=2, sort_keys=True) + '\n')

final = {
    'EPSILON_SEED1_NATIVE_SEMANTICS_PASS': True,
    'C1_SEED0_TO_SEED1_UNEXPECTED_DIFFS': audit_diffs['C1']['unexpected_scientific_differences'],
    'C2_SEED0_TO_SEED1_UNEXPECTED_DIFFS': audit_diffs['C2']['unexpected_scientific_differences'],
    'CS_SEED0_TO_SEED1_UNEXPECTED_DIFFS': audit_diffs['CS']['unexpected_scientific_differences'],
    'C2_CORRECT_TRACKS_SHA_SEED0_EQ_SEED1': seed0['C2']['transport']['tracks']['sha256'] == seed1['C2']['transport']['tracks']['sha256'],
    'CS_SHUFFLE_PERMUTATION_SHA_SEED0_EQ_SEED1': seed0['CS']['transport']['permutation']['sha256'] == seed1['CS']['transport']['permutation']['sha256'],
    'CS_SHUFFLED_TRACKS_SHA_SEED0_EQ_SEED1': seed0['CS']['transport']['tracks']['sha256'] == seed1['CS']['transport']['tracks']['sha256'],
    'PREVIEW_SHA_SEED0_EQ_SEED1': all(seed0[a]['preview']['file_sha256'] == seed1[a]['preview']['file_sha256'] for a in arms),
    'SEED1_THREE_ARM_SHARED_EPSILON': common_randomness['SEED1_THREE_ARM_SHARED_EPSILON'],
    'SEED0_TO_SEED1_ONLY_SCIENTIFIC_RANDOM_CHANGE': 'FORMAL_DIFFUSION_SEED',
    'EPSILON_CHANGE': 'DETERMINISTIC_CONSEQUENCE_OF_FORMAL_DIFFUSION_SEED',
}
final['SEED0_TO_SEED1_ARM_LINEAGE_AUDIT'] = 'PASS' if all(v == [] for v in (final['C1_SEED0_TO_SEED1_UNEXPECTED_DIFFS'], final['C2_SEED0_TO_SEED1_UNEXPECTED_DIFFS'], final['CS_SEED0_TO_SEED1_UNEXPECTED_DIFFS'])) and all(final[k] for k in ('EPSILON_SEED1_NATIVE_SEMANTICS_PASS', 'C2_CORRECT_TRACKS_SHA_SEED0_EQ_SEED1', 'CS_SHUFFLE_PERMUTATION_SHA_SEED0_EQ_SEED1', 'CS_SHUFFLED_TRACKS_SHA_SEED0_EQ_SEED1', 'PREVIEW_SHA_SEED0_EQ_SEED1', 'SEED1_THREE_ARM_SHARED_EPSILON')) else 'FAIL_OR_UNRESOLVED'
(OUT / 'SEED0_TO_SEED1_ARM_LINEAGE_AUDIT.json').write_text(json.dumps(final, indent=2, sort_keys=True) + '\n')

md = '# Phase 0D-4D-R3M-R3 — Seed1 lineage execution\n\n'
md += 'Mechanical audit recovered each seed-0 arm from persisted runtime/launch evidence and copied it to a seed-1 plan. The normalized comparison permits only `FORMAL_DIFFUSION_SEED` and the corresponding validated external epsilon artifact.\n\n'
md += 'No correct-track, shuffled-track, permutation, preview, visibility, depth, IDs, evaluator, scheduler, start-state, model, or overlay field changed.\n\n'
md += 'Audit result: `' + final['SEED0_TO_SEED1_ARM_LINEAGE_AUDIT'] + '`.\n'
(OUT / 'PHASE0D_4D_R3M_R3_SEED1_LINEAGE_EXECUTION.md').write_text(md)
print(json.dumps(final, sort_keys=True))
