# Conditioning seed-invariance proof

Functional synthetic evidence: seeds 0–4 run the frozen CPU `create_pos_feature_map` then `replace_feature`. Internal first and third randperm sequences differ by seed, while edited_y, future support, and canonical winner mapping `(tau,target_h,target_w,material_id,float32_depth_bits)` are exactly identical.

Algebraic generality: the first permutation jointly reorders tracks/visibility; V3D applies the corresponding index to depth and IDs. The third permutation jointly reorders track_pos and all associated V3D fields. Thus the per-cell candidate multiset is invariant. Winner=min(depth,material_id) is order invariant because IDs are unique. The relevant source feature remains coupled to its material carrier, and V3D directly replaces the same winning target cell. The second randperm affects track_feats only; `WanMove.generate` does not pass track_feats to arg_c/arg_null or the model. Hence internal permutation order does not change generator-consumed edited_y for any fixed vae_feature.

Evidence level: FUNCTIONAL_SYNTHETIC + ALGEBRAIC_GENERAL, not FUNCTIONAL_REAL_LATENT.
