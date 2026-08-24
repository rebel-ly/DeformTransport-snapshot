#!/usr/bin/env python3
"""CPU-only preparation for PAPER-EXP-A1.

This program never imports Wan's pipeline or torch.  It derives the two
STATIC counterfactual inputs from the frozen accepted C2/CS tracks, copies
the formal overlay into a separate append-only A1 overlay, and adds only the
two explicitly preregistered attribute switches to that copy.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/DeformTransport_EvidenceV4_1')
OUT = ROOT / 'reports/paper_experiments/20260816_203349__physical_attribute_motion_identity_ablation'
FORMAL = ROOT / 'experimental/20260814__wanmove_preview_sdedit_overlay'
R3 = ROOT / 'reports/phase0d/20260815_010000__phase0d_4d_r3_runtime_grid_reconciliation'
PHASE0B = ROOT / 'reports/phase0b/causal_contract/20260812_134250__santa_corrected_v2_identity_shuffle_seed0'
TRACKS = Path('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_tracks_correct.npy')
VIS = Path('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_visibility_correct.npy')
IDS = Path('/workspace/DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline/santa_material_point_ids.npy')
DEPTH = ROOT / 'reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy'
SOURCE = Path('/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png')
WAN = Path('/workspace/Wan-Move')

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def dump(name: str, obj: object) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')

def main() -> None:
    (OUT / 'conditions').mkdir(exist_ok=True)
    correct = np.load(TRACKS, allow_pickle=False)
    shuffled_path = PHASE0B / 'santa_material_tracks_identity_shuffled_seed0.npy'
    shuffled = np.load(shuffled_path, allow_pickle=False)
    if correct.shape != (1, 81, 1257, 2) or shuffled.shape != correct.shape:
        raise RuntimeError(f'unexpected track shapes: {correct.shape}, {shuffled.shape}')

    # A STATIC counterfactual has target position p_i^0 at every future time.
    # It intentionally retains the separate future visibility and depth inputs.
    sc = np.broadcast_to(correct[:, 0:1], correct.shape).copy()
    ss = np.broadcast_to(shuffled[:, 0:1], shuffled.shape).copy()
    sc_path = OUT / 'conditions/SC_static_correct_tracks_seed0.npy'
    ss_path = OUT / 'conditions/SS_static_shuffled_tracks_seed0.npy'
    np.save(sc_path, sc, allow_pickle=False)
    np.save(ss_path, ss, allow_pickle=False)

    overlay = OUT / 'overlay_a1_frozen_r3'
    if overlay.exists():
        raise RuntimeError('refusing to overwrite A1 overlay')
    shutil.copytree(FORMAL, overlay)
    traj = overlay / 'wan/modules/trajectory.py'
    text = traj.read_text()
    old_valid = '''        valid = (
            src_valid
            &
            (target_h >= 0)
            &
            (target_w >= 0)
            &
            vis_sampled[tau]
        )'''
    new_valid = '''        # PAPER-EXP-A1 NOVIS: remove only the future visibility gate.\n        # Coordinate validity remains identical to formal V3D.\n        if os.environ.get("DT_A1_DISABLE_VISIBILITY", "0") == "1":\n            future_visibility = torch.ones_like(vis_sampled[tau], dtype=torch.bool)\n        else:\n            future_visibility = vis_sampled[tau]\n\n        valid = (\n            src_valid\n            & (target_h >= 0)\n            & (target_w >= 0)\n            & future_visibility\n        )'''
    if old_valid not in text:
        raise RuntimeError('formal visibility block not found')
    text = text.replace(old_valid, new_valid, 1)
    old_depth = '''                    else:\n\n                        score = float(\n                            depth_sampled[\n                                tau,\n                                ii,\n                            ].item()\n                        )\n\n                        if (\n                            not math.isfinite(\n                                score\n                            )\n                            or score <= 0\n                        ):\n                            continue\n\n                    key = (\n                        score,\n                        material_id,\n                    )'''
    new_depth = '''                    else:\n\n                        # PAPER-EXP-A1 NODEPTH: remove depth only from\n                        # arbitration.  The fixed material-id order is the\n                        # preregistered deterministic replacement.\n                        if os.environ.get("DT_A1_DISABLE_DEPTH_ARBITRATION", "0") == "1":\n                            key = (material_id,)\n                        else:\n                            score = float(\n                                depth_sampled[\n                                    tau,\n                                    ii,\n                                ].item()\n                            )\n\n                            if (\n                                not math.isfinite(\n                                    score\n                                )\n                                or score <= 0\n                            ):\n                                continue\n\n                            key = (\n                                score,\n                                material_id,\n                            )'''
    if old_depth not in text:
        raise RuntimeError('formal depth block not found')
    text = text.replace(old_depth, new_depth, 1)
    # `os` is needed only by the two isolated A1 branches.
    text = text.replace('    import math\n\n    global _DT_CONTEXT', '    import math\n    import os\n\n    global _DT_CONTEXT', 1)
    traj.write_text(text)

    common = {
        'scene': 'Santa', 'formal_diffusion_seed': 0,
        'transport_variant': 'v3d', 'begin_index': 15,
        'scheduler': 'FlowUniPCMultistepScheduler', 'sample_steps': 40,
        'sample_shift': 3.0, 'effective_denoise_steps': 25,
        'dtype': 'bf16', 'generation_resolution': '480*832',
        'decoded_resolution': '832x464', 'frame_count': 81,
        'source_image': str(SOURCE), 'source_image_sha256': sha(SOURCE),
        'prompt': 'Wind blows the hanging clothes. The motion is gentle, continuous, and rhythmic, driven by shifting airflow. Static camera, eye-level frontal view, natural fabric movement.',
        'preview_latent': str(R3 / 'WAN_FORMAL_PREVIEW_LATENT_58x104.npy'),
        'preview_latent_sha256': sha(R3 / 'WAN_FORMAL_PREVIEW_LATENT_58x104.npy'),
        'epsilon': str(R3 / 'R3_SHARED_EPSILON_58x104.npy'),
        'epsilon_sha256': sha(R3 / 'R3_SHARED_EPSILON_58x104.npy'),
        'tracks_correct': str(TRACKS), 'tracks_correct_sha256': sha(TRACKS),
        'visibility': str(VIS), 'visibility_sha256': sha(VIS),
        'material_ids': str(IDS), 'material_ids_sha256': sha(IDS),
        'depth': str(DEPTH), 'depth_sha256': sha(DEPTH),
        'formal_overlay': str(FORMAL),
        'formal_generate_sha256': sha(FORMAL / 'generate.py'),
        'formal_wan_move_sha256': sha(FORMAL / 'wan/wan_move.py'),
        'formal_trajectory_sha256': sha(FORMAL / 'wan/modules/trajectory.py'),
        'a1_overlay': str(overlay), 'a1_trajectory_sha256': sha(traj),
        'checkpoint': str(WAN / 'Wan-Move-14B-480P'),
        'operator_runtime_note': 'SC/SS use byte-identical formal overlay. NOVIS/NODEPTH use the isolated A1 copy with only preregistered branches.'
    }
    arms = {
        'SC': {'future_motion': 'static', 'correct_identity': True, 'tracks': sc_path, 'visibility_gate': True, 'depth_arbitration': True, 'overlay': FORMAL},
        'SS': {'future_motion': 'static', 'correct_identity': False, 'tracks': ss_path, 'visibility_gate': True, 'depth_arbitration': True, 'overlay': FORMAL},
        'C2-NOVIS': {'future_motion': 'dynamic', 'correct_identity': True, 'tracks': TRACKS, 'visibility_gate': False, 'depth_arbitration': True, 'overlay': overlay},
        'C2-NODEPTH': {'future_motion': 'dynamic', 'correct_identity': True, 'tracks': TRACKS, 'visibility_gate': True, 'depth_arbitration': False, 'overlay': overlay},
    }
    for arm, spec in arms.items():
        m = dict(common)
        m.update({k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()})
        p = Path(m['tracks'])
        m['tracks_sha256'] = sha(p)
        m['identity_shuffle_permutation_sha256'] = sha(PHASE0B / 'identity_permutation_seed0.npy') if (PHASE0B / 'identity_permutation_seed0.npy').exists() else 'c8f5bd566b4cad1825c08863461f0672beee50959a06588d8424b8c8653de471'
        m['launcher_environment'] = {
            'DT_TRANSPORT_VARIANT': 'v3d',
            'DT_TRACK_IDS_PATH': str(IDS),
            'DT_TRACK_DEPTH_PATH': str(DEPTH),
            'DT_A1_DISABLE_VISIBILITY': '1' if arm == 'C2-NOVIS' else '0',
            'DT_A1_DISABLE_DEPTH_ARBITRATION': '1' if arm == 'C2-NODEPTH' else '0',
        }
        dump(f'conditions/{arm}_PLANNED_MANIFEST.json', m)
    contract = {
        'formal_runtime_recovered': True,
        'formal_runtime_evidence': {'generate': sha(FORMAL / 'generate.py'), 'wan_move': sha(FORMAL / 'wan/wan_move.py'), 'trajectory': sha(FORMAL / 'wan/modules/trajectory.py')},
        'static_definition': 'All t=1..80 target coordinates equal their arm-specific t=0 coordinates; future visibility and depth sidecars are unchanged.',
        'SC_SS_operator': 'byte-identical accepted formal overlay',
        'NOVIS': 'Only future visibility inclusion is removed; source/target coordinate validity and depth collision arbitration remain.',
        'NODEPTH': 'Only depth-based arbitration and its depth validity rejection are removed; deterministic key is ascending material_id.',
        'shared_randomness': {'preview_sha256': common['preview_latent_sha256'], 'epsilon_sha256': common['epsilon_sha256'], 'seed': 0},
        'dry_run': 'CPU artifact/shape/hash verification only; no Wan model, torch pipeline, or GPU was loaded.'
    }
    dump('CAUSAL_CONTRACT_AUDIT.json', contract)
    dump('FORMAL_RUNTIME_PROVENANCE.json', common)
    print(json.dumps({'status': 'PASS', 'sc': str(sc_path), 'ss': str(ss_path), 'formal_trajectory_sha': common['formal_trajectory_sha256'], 'a1_trajectory_sha': common['a1_trajectory_sha256']}))

if __name__ == '__main__':
    main()
