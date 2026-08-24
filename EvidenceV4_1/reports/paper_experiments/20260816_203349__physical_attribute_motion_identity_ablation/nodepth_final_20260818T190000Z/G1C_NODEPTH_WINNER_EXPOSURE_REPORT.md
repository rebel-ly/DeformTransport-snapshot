# PAPER-EXP-A1 / G1-C NODEPTH winner exposure

- Geometry invariant: PASS (`9766 / 8962 / 788 / 804 / 3`).
- Canonical-key join: 8962 rows; key, candidate-count, and collision-flag invariants PASS.
- Non-collision winner changes: 0 / 8174.
- Collision winner changes: 328 / 788 (0.416243654822).
- All-write winner changes: 328 / 8962 (0.036598973443).
- Collision candidates total: 1592.
- Exposure consistency: PASS.

Canonical payload: `latent_t,latent_y,latent_x,winner_material_id\n`, UTF-8, LF, ascending numeric key order.
