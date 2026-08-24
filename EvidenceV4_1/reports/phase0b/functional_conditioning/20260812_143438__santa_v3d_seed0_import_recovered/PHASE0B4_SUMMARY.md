# Phase0B-4 Functional Conditioning Differential Audit

Status: **PASS**

This CPU-only function-level audit used the frozen corrected-v2 inputs, authoritative aligned depth sidecar, and frozen patched Wan-Move source.

## Key results

### source_drift_gate

```json
{
  "git_head": "80c58a7d2ad175fa82a4d57f79f2a1415317dcfa",
  "pass": true,
  "sha256": {
    "trajectory.py": "0c6bc94d8ce1f885f0333314a9b201a650163cd209b2a3b3f95b4f3a35a49dae",
    "wan_move.py": "aca79f9cc4bf32ea363c4440ed2c7e7d90ef5aa763f3e96ae6c2b8eff35c1857"
  }
}
```

### authoritative_depth_id_lineage

```json
{
  "aligned_depth_shape": [
    81,
    28264
  ],
  "aligned_point_id_selected_equals_selected_ids": true,
  "aligned_point_id_shape": [
    28264
  ],
  "consumed_depth_value_count": 41230,
  "depth_finite_on_v3d_consumed_values": true,
  "depth_selected_shape": [
    81,
    1257
  ],
  "pass": true,
  "selected_ids_in_range": true,
  "selected_ids_unique": 1257
}
```

### paired_rng

```json
{
  "correct_rng_state_sha256_after_two_randperms": "5154b9719d7b3a7be07d26a398e4da6b6a6e056dcb937346a77d1498df30bfa6",
  "pass": true,
  "rng_states_equal": true,
  "shuffled_rng_state_sha256_after_two_randperms": "5154b9719d7b3a7be07d26a398e4da6b6a6e056dcb937346a77d1498df30bfa6"
}
```

### functional_context

```json
{
  "depth_exact": true,
  "future_track_pos_slots1_to20_exact": true,
  "future_tracks_t1_to_80_exact": true,
  "ids_exact": true,
  "source_coordinate_mismatch_count": 1257,
  "source_track_pos_slot0_different": true,
  "source_track_pos_slot0_exact": false,
  "source_tracks_t0_different": true,
  "source_tracks_t0_exact": false,
  "visibility_exact": true
}
```

### edited_y_audit

```json
{
  "edited_dtype": "torch.float32",
  "edited_shape": [
    1,
    4,
    21,
    60,
    104
  ],
  "future_write_support_exact": true,
  "inside_common_support_nonzero_conditioning_difference_count": 27209,
  "outside_common_support_future_exact": true,
  "outside_support_nonzero_difference_count": 0,
  "shape_dtype_exact": true,
  "source_frame_nonzero_difference_count": 0,
  "source_slot_preserved_correct": true,
  "source_slot_preserved_shuffled": true,
  "write_support_cell_count": 9031
}
```

## Interpretation

Correct and Identity-Shuffled have paired future geometry/support, visibility, depth, IDs, and RNG; the observed conditioning differential is confined to common future target support.
