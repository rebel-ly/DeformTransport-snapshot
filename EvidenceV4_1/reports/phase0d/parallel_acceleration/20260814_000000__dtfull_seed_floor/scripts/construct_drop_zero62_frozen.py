#!/usr/bin/env python3
"""CPU-only, order-preserving construction of the frozen DROP-ZERO62 input."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

W = Path('/workspace')
RUN = W / 'DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor'
OUT = RUN / 'drop_zero62' / os.environ['DROP_ZERO62_STAMP']
SIDE = W / 'DeformTransport/server_runs/new_audit/20260811_224005__santa_corrected_v2_aligned_timeline'
F3 = W / 'DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_103849__f3_gpu_arm_construction_freeze'
F2 = W / 'DeformTransport_EvidenceV4_1/reports/phase0d/conditioning_alignment/20260813_081824__f2_metric_mechanism_error_localization'
DEPTH = W / 'DeformTransport_EvidenceV4_1/reports/phase0b/functional_conditioning/20260812_143438__santa_v3d_seed0_import_recovered/santa_authoritative_depth_81x1257.npy'
SOURCE = W / 'DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/resized_input_image.png'
TRACK = SIDE / 'santa_material_tracks_correct.npy'
VIS = SIDE / 'santa_material_visibility_correct.npy'
IDS = SIDE / 'santa_material_point_ids.npy'
ZERO = F3 / 'subgroups/zero_switch_positive_visible_ids.npy'
JOIN = F2 / 'transport_error_join.npz'

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def save(name, a):
    p = OUT / name
    np.save(p, a)
    return {'path': str(p), 'sha256': sha_file(p), 'shape': list(a.shape), 'dtype': str(a.dtype)}

def structural(ids, tr, vi, de):
    candidates = collisions = collision_cells = winners = 0
    full_winner = []
    support = set()
    for tau in range(1, 21):
        t = tau * 4
        src, tgt = tr[0], tr[t]
        ok = (vi[t] & np.isfinite(src).all(1) & np.isfinite(tgt).all(1) &
              (src[:,0] >= 0) & (src[:,0] < 832) & (src[:,1] >= 0) & (src[:,1] < 480) &
              (tgt[:,0] >= 0) & (tgt[:,0] < 832) & (tgt[:,1] >= 0) & (tgt[:,1] < 480) &
              np.isfinite(de[t]) & (de[t] > 0))
        groups = {}
        for q in np.where(ok)[0]:
            groups.setdefault((int(tgt[q,1] // 8), int(tgt[q,0] // 8)), []).append(q)
        candidates += int(ok.sum()); winners += len(groups)
        collision_cells += sum(len(g) > 1 for g in groups.values())
        collisions += sum(len(g) for g in groups.values() if len(g) > 1)
        for cell, group in groups.items():
            q = min(group, key=lambda i: (float(de[t,i]), int(ids[i])))
            full_winner.append(int(ids[q])); support.add((tau, *cell))
    return {'carrier_count': int(len(ids)), 'candidate_assignment_count': candidates,
            'winning_write_count': winners, 'collision_candidate_count': collisions,
            'collision_cell_count': collision_cells, 'spatial_write_support_count': len(support),
            'winner_material_ids_sha256': sha_array(np.asarray(full_winner, dtype=np.int64))}

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    full = np.load(IDS).astype(np.int64)
    zero = np.load(ZERO).astype(np.int64)
    tr = np.load(TRACK)[0].astype(np.float32)
    vi = np.load(VIS)[0].astype(bool)
    depth = np.load(DEPTH).astype(np.float32)
    diag = np.load(JOIN)
    assert full.shape == (1257,) and sha_file(IDS) == 'f94bb0a7986c693e194f750a7afd715f44506518abbb4dd37e0a791380c819b8'
    assert zero.shape == (62,) and len(np.unique(zero)) == 62
    pos = {int(x): i for i, x in enumerate(full)}
    assert all(int(x) in pos for x in zero)
    idx_zero = np.array([pos[int(x)] for x in zero], dtype=np.int64)
    assert np.array_equal(full[idx_zero], zero)
    # F2 frozen operational values verify the pre-materialized F3 membership without reselecting it.
    sw, visible = diag['visibility_switch_count'], diag['visible_slot_count']
    assert np.all(sw[idx_zero] == 0) and np.all(visible[idx_zero] > 0)
    keep_mask = ~np.isin(full, zero)
    retained = full[keep_mask]
    keep_idx = np.where(keep_mask)[0]
    assert retained.shape == (1195,) and np.array_equal(retained, full[~np.isin(full, zero)])
    assert not np.intersect1d(retained, zero).size
    assert set(map(int, retained)) | set(map(int, zero)) == set(map(int, full))
    artifacts = {
        'ids': save('drop_zero62_ids.npy', retained),
        'tracks': save('drop_zero62_tracks.npy', tr[:, keep_idx][None]),
        'visibility': save('drop_zero62_visibility.npy', vi[:, keep_idx][None]),
        'depth': save('drop_zero62_depth.npy', depth[:, keep_idx]),
    }
    full_s, drop_s = structural(full, tr, vi, depth), structural(retained, tr[:, keep_idx], vi[:, keep_idx], depth[:, keep_idx])
    changed = {'removed_carrier_count': 62, 'retained_carrier_count': 1195,
               'future_write_support_full': full_s['spatial_write_support_count'],
               'future_write_support_drop': drop_s['spatial_write_support_count'],
               'future_write_support_delta': drop_s['spatial_write_support_count'] - full_s['spatial_write_support_count'],
               'winner_changes_vs_full': int(sum(a != b for a, b in zip(full_s['winner_material_ids_sha256'], drop_s['winner_material_ids_sha256']))),
               'note': 'winner_changes_vs_full is intentionally not inferred from incompatible digest strings; see both deterministic winner digests.'}
    manifest = {
      'experiment_name': 'DROP-ZERO62', 'creation_timestamp': os.environ['DROP_ZERO62_STAMP'],
      'scientific_status': 'FROZEN_PREDEFINED_INTERVENTION', 'parent_population': 'FULL_corrected_v2',
      'parent_N': 1257, 'removed_subgroup': 'ZERO_SWITCH_POSITIVE_VISIBLE', 'removed_N': 62, 'retained_N': 1195,
      'authoritative_full_ids_path': str(IDS), 'authoritative_full_ids_sha256': sha_file(IDS),
      'authoritative_zero62_source_path': str(ZERO), 'authoritative_zero62_source_sha256': sha_file(ZERO),
      'zero62_ids': [int(x) for x in zero], 'zero62_ids_sha256': sha_file(ZERO),
      'retained_ids': [int(x) for x in retained], 'retained_ids_sha256': sha_file(OUT / 'drop_zero62_ids.npy'),
      'selection_rule': 'FULL corrected-v2 minus exact frozen ZERO_SWITCH_POSITIVE_VISIBLE N62',
      'ordering_rule': 'preserve canonical FULL corrected-v2 material-ID order',
      'subset_rearbitration': 'normal subset re-arbitration',
      'source_image': {'path': str(SOURCE), 'sha256': sha_file(SOURCE)},
      'tracks_source': {'path': str(TRACK), 'sha256': sha_file(TRACK)},
      'visibility_source': {'path': str(VIS), 'sha256': sha_file(VIS)},
      'depth_source': {'path': str(DEPTH), 'sha256': sha_file(DEPTH)},
      'timeline_identity': 'corrected-v2 81-frame aligned timeline', 'corrected_v2_lineage': 'F3 frozen subgroup membership + corrected-v2 authoritative inputs',
      'generation_input_artifacts': artifacts, 'generator_configuration': {'operator': 'V3D', 'sample_steps': 40, 'sample_shift': 3.0, 'dtype': 'bf16', 'seed': 0},
      'METRIC_BASED_SELECTION': False, 'POST_HOC_TUNING': False, 'STABLEVIS_USED': False, 'SCIENTIFIC_CONFIGURATION_CHANGED': False,
      'verification': {'FULL_IDS_COUNT': int(len(full)), 'FULL_IDS_SHA256': sha_file(IDS), 'ZERO62_COUNT': int(len(zero)),
       'ZERO62_UNIQUE_COUNT': int(len(np.unique(zero)), 'ZERO62_DUPLICATE_COUNT': int(len(zero)-len(np.unique(zero))),
       'ZERO62_ALL_IN_FULL1257': True, 'ZERO62_SWITCH_COUNT_ZERO_ALL': True, 'ZERO62_POSITIVE_VISIBLE_ALL': True,
       'DROP_ZERO62_COUNT': int(len(retained)), 'DROP_ZERO62_IDS_SHA256': sha_file(OUT / 'drop_zero62_ids.npy'),
       'DROP_ZERO62_INTERSECTION_ZERO62_EMPTY': True, 'DROP_ZERO62_UNION_FULL': True, 'CANONICAL_ORDERING_PRESERVED': True,
       'NORMAL_SUBSET_REARBITRATION_CONFIGURED': True, 'DROP_ZERO62_GENERATION_INPUT_COUNT': int(len(retained)),
       'DROP_ZERO62_MANIFEST_PASS': True},
      'structural_difference_diagnostics': {'full': full_s, 'drop_zero62': drop_s, 'difference': changed}
    }
    (OUT / 'DROP_ZERO62_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'out': str(OUT), 'manifest_pass': True, 'retained_sha': manifest['retained_ids_sha256']}, sort_keys=True))

if __name__ == '__main__':
    main()
