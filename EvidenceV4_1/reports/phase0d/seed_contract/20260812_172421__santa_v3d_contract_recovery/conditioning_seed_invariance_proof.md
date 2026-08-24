# Conditioning seed-invariance proof

The functional synthetic audit runs seeds 0–4 through frozen `create_pos_feature_map` and `replace_feature` on CPU and obtains bitwise-equal edited_y, support and winner mapping despite different internal randperm sequences.

For general fixed vae_feature, the first permutation jointly reorders tracks and visibility; V3D context applies the same index to depth and IDs. The third permutation jointly reorders track_pos and all associated context fields. Consequently, for each future target cell the candidate multiset of (source coordinate/feature, target coordinate, visibility, depth, material ID) is unchanged by either reordering. Winner selection is min(depth, material_id), so it is order-invariant with unique material IDs. Each winning carrier’s bilinear source feature is carried with that same carrier, and V3D directly replaces the same winning target cell. The second randperm only builds track_feats; frozen `WanMove.generate` does not pass track_feats into arg_c or arg_null/model. Therefore internal permutation order cannot alter generator-consumed edited_y; the seed changes diffusion/random path, not V3D conditioning.

Evidence level: FUNCTIONAL_SYNTHETIC + ALGEBRAIC_GENERAL, not FUNCTIONAL_REAL_LATENT.
