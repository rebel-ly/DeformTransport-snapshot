#!/usr/bin/env python3
"""CPU-only functional gate for current parity-proven V3D K=0 contract."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

R = Path('/workspace/DeformTransport_EvidenceV4_1/reports/phase0d/20260814_200000__phase0d_4d_recovery')
R.mkdir(parents=True, exist_ok=True)
tracks = np.zeros((81, 0, 2), dtype=np.float32)
visibility = np.zeros((81, 0), dtype=np.bool_)
ids = np.zeros((0,), dtype=np.int64)
depth = np.zeros((81, 0), dtype=np.float32)
np.save(R / 'k0_tracks.npy', tracks)
np.save(R / 'k0_visibility.npy', visibility)
np.save(R / 'k0_ids.npy', ids)
np.save(R / 'k0_depth.npy', depth)
os.environ['DT_TRANSPORT_VARIANT'] = 'v3d'
os.environ['DT_TRACK_IDS_PATH'] = str(R / 'k0_ids.npy')
os.environ['DT_TRACK_DEPTH_PATH'] = str(R / 'k0_depth.npy')

trajectory_path = Path("/workspace/DeformTransport_EvidenceV4_1/experimental/20260814__wanmove_preview_sdedit_overlay/wan/modules/trajectory.py")
spec = importlib.util.spec_from_file_location("formal_overlay_trajectory", trajectory_path)
trajectory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trajectory)
create_pos_feature_map = trajectory.create_pos_feature_map
replace_feature = trajectory.replace_feature

torch.manual_seed(20260814)
y = torch.randn(16, 21, 60, 104, dtype=torch.float32)
track = torch.from_numpy(tracks)
vis = torch.from_numpy(visibility)
_, pos = create_pos_feature_map(track, vis, (4, 8, 8), 480, 832, y.size(0), track_num=0, device=torch.device('cpu'))
edited = replace_feature(y.unsqueeze(0), pos.unsqueeze(0))[0]
delta = (edited - y).abs()
result = {
    'N': 0,
    'tracks_shape': list(tracks.shape),
    'visibility_shape': list(visibility.shape),
    'ids_shape': list(ids.shape),
    'depth_shape': list(depth.shape),
    'DT_TRANSPORT_VARIANT': os.environ['DT_TRANSPORT_VARIANT'],
    'edited_y_equals_y_exact': bool(torch.equal(edited, y)),
    'different_scalar_values': int(torch.count_nonzero(delta).item()),
    'max_abs_diff': float(delta.max().item()),
    'mean_abs_diff': float(delta.mean().item()),
    'trajectory_writes': 0,
    'winner_writes': 0,
    'current_overlay_k0_exact_noop': bool(torch.equal(edited, y)),
    'y_sha256': hashlib.sha256(y.numpy().tobytes()).hexdigest(),
    'edited_y_sha256': hashlib.sha256(edited.numpy().tobytes()).hexdigest(),
}
(R / 'WM0_K0_REAUDIT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, sort_keys=True))
