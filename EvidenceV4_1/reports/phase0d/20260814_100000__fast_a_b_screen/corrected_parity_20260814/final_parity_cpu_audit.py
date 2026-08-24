#!/usr/bin/env python3
"""CPU-only decoded-RGB integrity and exact-parity audit; no model imports."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_100000__fast_a_b_screen/corrected_parity_20260814')
VIDEOS = {
    'a2': BASE / 'a2_gpu0_formal_rerun/santa_correct_v3d_seed000.mp4',
    'b2_g2': BASE / 'b2_gpu2_hardgate/santa_correct_v3d_seed000.mp4',
    'canonical': Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/parallel_acceleration/20260814_000000__dtfull_seed_floor/outputs/gpu0_seed0_eligibility/santa_correct_v3d_seed000.mp4'),
}


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    array = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
    return array, {
        'mp4_size_bytes': path.stat().st_size,
        'mp4_sha256': file_sha(path),
        'decode_success': bool(len(frames)),
        'frame_count': int(array.shape[0]),
        'width': int(array.shape[2]),
        'height': int(array.shape[1]),
        'fps': float(fps),
        'shape': list(array.shape),
        'decoded_rgb_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
    }


def compare(left, right):
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    changed = int(np.count_nonzero(delta))
    return {
        'rgb_exact': changed == 0,
        'different_channel_values': changed,
        'different_channel_fraction': changed / delta.size,
        'max_abs_diff': int(delta.max()),
        'mean_abs_diff': float(delta.mean()),
    }


arrays, result = {}, {'videos': {}, 'comparisons': {}}
for name, path in VIDEOS.items():
    arrays[name], result['videos'][name] = decode(path)
    result['videos'][name]['path'] = str(path)
result['comparisons']['a2_vs_canonical'] = compare(arrays['a2'], arrays['canonical'])
result['comparisons']['b2_g2_vs_canonical'] = compare(arrays['b2_g2'], arrays['canonical'])
result['comparisons']['a2_vs_b2_g2'] = compare(arrays['a2'], arrays['b2_g2'])
(BASE / 'PHASE0D_4C_FINAL_PARITY_CPU_AUDIT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
