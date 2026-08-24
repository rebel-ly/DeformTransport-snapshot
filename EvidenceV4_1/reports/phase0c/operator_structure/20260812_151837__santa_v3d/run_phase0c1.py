# Frozen Phase0C-1 reconstruction specification
# Inputs: 0B-4R saved context, track_pos, expected third randperm.
# V3D candidate: valid source/target latent coordinates and sampled visibility.
# Group by target cell; discard non-finite/nonpositive depth; winner=min(depth, material_id).
# Full executed results are in operator_structure_audit.json.
