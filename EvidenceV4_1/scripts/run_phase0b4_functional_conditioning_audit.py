#!/usr/bin/env python3
"""Frozen CPU-only Phase0B-4 functional conditioning audit.

This harness intentionally invokes the installed Wan-Move functions without
patching them.  All outputs are written below the supplied evidence directory.
"""
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path('/mnt/sdbd/home/liuyu_qyh')
EVIDENCE = ROOT / 'DeformTransport_EvidenceV4_1'
WAN = ROOT / 'Wan-Move'
CORRECTED = ROOT / 'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
B1 = EVIDENCE / 'reports/phase0b/causal_contract/20260812_134250__santa_corrected_v2_identity_shuffle_seed0'
ALIGNED = ROOT / 'DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_contract_20260806_192643/outputs/aligned_transport_ready.pt'
EXPECTED_HEAD = '80c58a7d2ad175fa82a4d57f79f2a1415317dcfa'
EXPECTED_SHA = {
    'wan_move.py': 'aca79f9cc4bf32ea363c4440ed2c7e7d90ef5aa763f3e96ae6c2b8eff35c1857',
    'trajectory.py': '0c6bc94d8ce1f885f0333314a9b201a650163cd209b2a3b3f95b4f3a35a49dae',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def b(value):
    return bool(value)


def exact(a, c):
    return b(torch.equal(a, c))


def array_exact(a, c):
    return b(np.array_equal(a, c))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', None):
        raise RuntimeError('CPU-only protocol requires CUDA_VISIBLE_DEVICES=""')
    if torch.cuda.is_available():
        raise RuntimeError('CPU-only protocol requires torch.cuda.is_available() == False')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out = EVIDENCE / 'reports/phase0b/functional_conditioning' / (stamp + '__santa_v3d_seed0')
    out.mkdir(parents=True, exist_ok=False)
    result = {'phase': 'Phase0B-4', 'timestamp_utc': stamp, 'status': 'ERROR', 'cpu_only': True}
    try:
        # Required upstream archival gate.
        b3 = EVIDENCE / 'reports/phase0b/wanmove_consumption/20260812_135158__installed_source_audit'
        b3_status = json.loads((b3 / 'phase0b3_status.json').read_text())
        result['phase0b3_archive_gate'] = b3_status.get('status') == 'PASS' and (b3 / 'PHASE0B3_SUMMARY.md').is_file()
        if not result['phase0b3_archive_gate']:
            result['status'] = 'UPSTREAM_ARCHIVE_BLOCKED'
            save_json(out / 'phase0b4_status.json', result)
            return 0

        source_paths = {'wan_move.py': WAN / 'wan/wan_move.py', 'trajectory.py': WAN / 'wan/modules/trajectory.py'}
        observed_sha = {k: sha256(v) for k, v in source_paths.items()}
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WAN, text=True).strip()
        result['source_drift_gate'] = {'git_head': head, 'sha256': observed_sha,
                                       'pass': head == EXPECTED_HEAD and observed_sha == EXPECTED_SHA}
        if not result['source_drift_gate']['pass']:
            result['status'] = 'SOURCE_DRIFT'
            save_json(out / 'phase0b4_status.json', result)
            return 0

        tracks_c = np.load(CORRECTED / 'santa_material_tracks_correct.npy')
        tracks_s = np.load(B1 / 'santa_material_tracks_identity_shuffled_seed0.npy')
        visibility = np.load(CORRECTED / 'santa_material_visibility_correct.npy')
        ids = np.load(CORRECTED / 'santa_material_point_ids.npy')
        input_shapes_ok = (tracks_c.shape == (1,81,1257,2) and tracks_s.shape == (1,81,1257,2)
                           and tracks_c.dtype == np.float32 and tracks_s.dtype == np.float32
                           and visibility.shape == (1,81,1257) and visibility.dtype == np.bool_
                           and ids.shape == (1257,))
        result['input_contract'] = {'pass': b(input_shapes_ok), 'correct_shape': list(tracks_c.shape),
                                    'shuffled_shape': list(tracks_s.shape), 'visibility_shape': list(visibility.shape),
                                    'ids_shape': list(ids.shape), 'ids_dtype': str(ids.dtype)}
        if not input_shapes_ok:
            result['status'] = 'FAIL'
            save_json(out / 'phase0b4_status.json', result)
            return 0

        # Torch version compatibility is recorded because the input is trusted frozen evidence.
        try:
            aligned = torch.load(ALIGNED, map_location='cpu', weights_only=False)
        except TypeError:
            aligned = torch.load(ALIGNED, map_location='cpu')
        if not isinstance(aligned, dict) or 'depth' not in aligned or 'point_id' not in aligned:
            raise RuntimeError('aligned_transport_ready.pt lacks required depth/point_id mapping')
        depth_all = aligned['depth'].detach().cpu().numpy() if torch.is_tensor(aligned['depth']) else np.asarray(aligned['depth'])
        point_id = aligned['point_id'].detach().cpu().numpy() if torch.is_tensor(aligned['point_id']) else np.asarray(aligned['point_id'])
        selected_ids = np.asarray(ids)
        ids_integer = np.issubdtype(selected_ids.dtype, np.integer)
        ids_range = ids_integer and selected_ids.min() >= 0 and selected_ids.max() < point_id.shape[0]
        id_lineage = ids_range and array_exact(point_id[selected_ids], selected_ids)
        depth_selected = depth_all[:, selected_ids]
        depth_shape_ok = depth_selected.shape == (81,1257)
        # V3D consumes depth only for candidates that have valid source and target/visibility.
        consumed = visibility[0].copy()
        consumed &= (tracks_c[0, :, :, 0] >= 0) & (tracks_c[0, :, :, 1] >= 0)
        consumed &= (tracks_c[0, 0:1, :, 0] >= 0) & (tracks_c[0, 0:1, :, 1] >= 0)
        depth_consumed_finite = b(np.isfinite(depth_selected[consumed]).all())
        lineage_pass = b(np.unique(selected_ids).size == 1257 and ids_range and id_lineage and depth_shape_ok and depth_consumed_finite)
        np.save(out / 'santa_authoritative_depth_81x1257.npy', depth_selected.astype(np.float32, copy=False))
        np.save(out / 'santa_selected_material_ids_1257.npy', selected_ids)
        result['authoritative_depth_id_lineage'] = {
            'pass': lineage_pass, 'aligned_depth_shape': list(depth_all.shape), 'aligned_point_id_shape': list(point_id.shape),
            'selected_ids_unique': int(np.unique(selected_ids).size), 'selected_ids_in_range': b(ids_range),
            'aligned_point_id_selected_equals_selected_ids': b(id_lineage), 'depth_selected_shape': list(depth_selected.shape),
            'depth_finite_on_v3d_consumed_values': depth_consumed_finite, 'consumed_depth_value_count': int(consumed.sum())}
        if not lineage_pass:
            result['status'] = 'FAIL'
            save_json(out / 'phase0b4_status.json', result)
            return 0

        os.environ['DT_TRANSPORT_VARIANT'] = 'v3d'
        os.environ['DT_TRACK_DEPTH_PATH'] = str(out / 'santa_authoritative_depth_81x1257.npy')
        os.environ['DT_TRACK_IDS_PATH'] = str(out / 'santa_selected_material_ids_1257.npy')
        sys.path.insert(0, str(WAN))
        from wan.modules import trajectory

        # Deterministic spatial source feature; all future slots start identically zero.
        yy, xx = torch.meshgrid(torch.arange(60, dtype=torch.float32), torch.arange(104, dtype=torch.float32), indexing='ij')
        original = torch.zeros((1, 4, 21, 60, 104), dtype=torch.float32, device='cpu')
        original[0, 0, 0] = 1.0
        original[0, 1, 0] = xx / 103.0
        original[0, 2, 0] = yy / 59.0
        original[0, 3, 0] = (yy * 104.0 + xx) / (60.0 * 104.0 - 1.0)

        def run_condition(name, tracks_np):
            torch.manual_seed(0)
            tracks_t = torch.from_numpy(tracks_np[0]).to(dtype=torch.float32, device='cpu')
            vis_t = torch.from_numpy(visibility[0]).to(device='cpu')
            fmap, pos = trajectory.create_pos_feature_map(tracks_t, vis_t, [4,8,8], 480, 832, 4,
                                                           track_num=1257, t_down_strategy='sample', device=torch.device('cpu'), dtype=torch.float32)
            context = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in trajectory._DT_CONTEXT.items()}
            rng_state = torch.get_rng_state().clone()
            expected_third = torch.randperm(1257)
            torch.set_rng_state(rng_state)
            edited = trajectory.replace_feature(original.clone(), pos.unsqueeze(0))
            np.save(out / (name + '_expected_third_randperm.npy'), expected_third.numpy())
            np.save(out / (name + '_track_pos.npy'), pos.numpy())
            for key in ('tracks','visibility','depth','ids'):
                np.save(out / (name + '_context_' + key + '.npy'), context[key].numpy())
            np.save(out / (name + '_edited_y.npy'), edited.detach().cpu().numpy())
            return {'context': context, 'pos': pos.detach().cpu().clone(), 'edited': edited.detach().cpu().clone(),
                    'expected_third': expected_third, 'rng_state_sha256': hashlib.sha256(rng_state.numpy().tobytes()).hexdigest(),
                    'feature_map_shape': list(fmap.shape)}

        correct = run_condition('correct', tracks_c)
        shuffled = run_condition('identity_shuffled', tracks_s)
        ctxc, ctxs = correct['context'], shuffled['context']
        context_checks = {
            'visibility_exact': exact(ctxc['visibility'], ctxs['visibility']),
            'depth_exact': exact(ctxc['depth'], ctxs['depth']),
            'ids_exact': exact(ctxc['ids'], ctxs['ids']),
            'future_tracks_t1_to_80_exact': exact(ctxc['tracks'][1:], ctxs['tracks'][1:]),
            'source_tracks_t0_exact': exact(ctxc['tracks'][0], ctxs['tracks'][0]),
            'future_track_pos_slots1_to20_exact': exact(correct['pos'][:,1:], shuffled['pos'][:,1:]),
            'source_track_pos_slot0_exact': exact(correct['pos'][:,0], shuffled['pos'][:,0]),
            'source_coordinate_mismatch_count': int((ctxc['tracks'][0] != ctxs['tracks'][0]).any(dim=1).sum().item()),
        }
        context_checks['source_tracks_t0_different'] = not context_checks['source_tracks_t0_exact']
        context_checks['source_track_pos_slot0_different'] = not context_checks['source_track_pos_slot0_exact']
        result['paired_rng'] = {'pass': exact(correct['expected_third'], shuffled['expected_third']),
                                'correct_rng_state_sha256_after_two_randperms': correct['rng_state_sha256'],
                                'shuffled_rng_state_sha256_after_two_randperms': shuffled['rng_state_sha256'],
                                'rng_states_equal': correct['rng_state_sha256'] == shuffled['rng_state_sha256']}
        result['functional_context'] = context_checks

        ec, es = correct['edited'], shuffled['edited']
        support_c = ec[0,0,1:] != 0
        support_s = es[0,0,1:] != 0
        common = support_c & support_s
        support_equal = exact(support_c, support_s)
        future_equal_outside = exact(ec[:,:,1:][:,:,~common], es[:,:,1:][:,:,~common])
        source_preserved_c = exact(ec[:,:,0], original[:,:,0])
        source_preserved_s = exact(es[:,:,0], original[:,:,0])
        diff = ec - es
        nonzero = diff != 0
        source_diff_count = int(nonzero[:,:,0].sum().item())
        outside_diff_count = int(nonzero[:,:,1:][:,:,~common].sum().item())
        inside_diff_count = int(nonzero[:,:,1:][:,:,common].sum().item())
        result['edited_y_audit'] = {
            'shape_dtype_exact': list(ec.shape) == list(es.shape) and ec.dtype == es.dtype,
            'edited_shape': list(ec.shape), 'edited_dtype': str(ec.dtype),
            'source_slot_preserved_correct': source_preserved_c, 'source_slot_preserved_shuffled': source_preserved_s,
            'future_write_support_exact': support_equal, 'write_support_cell_count': int(common.sum().item()),
            'outside_common_support_future_exact': future_equal_outside,
            'source_frame_nonzero_difference_count': source_diff_count,
            'outside_support_nonzero_difference_count': outside_diff_count,
            'inside_common_support_nonzero_conditioning_difference_count': inside_diff_count,
        }
        required = [result['paired_rng']['pass'], result['paired_rng']['rng_states_equal'],
                    context_checks['visibility_exact'], context_checks['depth_exact'], context_checks['ids_exact'],
                    context_checks['future_tracks_t1_to_80_exact'], context_checks['future_track_pos_slots1_to20_exact'],
                    context_checks['source_tracks_t0_different'], context_checks['source_track_pos_slot0_different'],
                    support_equal, future_equal_outside, source_preserved_c, source_preserved_s,
                    source_diff_count == 0, outside_diff_count == 0, inside_diff_count > 0]
        result['status'] = 'PASS' if all(required) else 'FAIL'
        result['pass_requirements'] = [b(x) for x in required]
        save_json(out / 'phase0b4_status.json', result)
        lines = ['# Phase0B-4 Functional Conditioning Differential Audit', '', f"Status: **{result['status']}**", '',
                 'This CPU-only function-level audit used the frozen corrected-v2 inputs, authoritative aligned depth sidecar, and frozen patched Wan-Move source.', '',
                 '## Key results', '']
        for section in ('source_drift_gate','authoritative_depth_id_lineage','paired_rng','functional_context','edited_y_audit'):
            lines += [f'### {section}', '', '```json', json.dumps(result[section], indent=2, sort_keys=True), '```', '']
        lines += ['## Interpretation', '', ('Correct and Identity-Shuffled have paired future geometry/support, visibility, depth, IDs, and RNG; the observed conditioning differential is confined to common future target support.' if result['status'] == 'PASS' else 'One or more preregistered functional conditions failed; consult the JSON evidence without adapting the protocol.'), '']
        (out / 'PHASE0B4_SUMMARY.md').write_text('\n'.join(lines))
    except Exception:
        result['status'] = 'CPU_IMPORT_BLOCKED' if 'from wan.modules' in traceback.format_exc() else 'ERROR'
        (out / 'traceback.txt').write_text(traceback.format_exc())
        save_json(out / 'phase0b4_status.json', result)
    finally:
        files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS_PHASE0B4.txt')
        (out / 'SHA256SUMS_PHASE0B4.txt').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files))
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
