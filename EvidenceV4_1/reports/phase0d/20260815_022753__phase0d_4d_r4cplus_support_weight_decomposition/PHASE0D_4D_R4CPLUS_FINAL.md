# Phase 0D-4D-R4C+ final report

## 1. Phase objective

Post-primary, CPU-only decomposition of transition-equal TC-ME support weighting and contribution concentration.

## 2. Audit question

Explain mechanically why seed1's formal transport delta differs from the unweighted per-ID descriptive mean, without changing primary results.

## 3. Data

R4C's six persisted per-transition/per-ID CSVs; R4C final report/result hashes are recorded in `R4CPLUS_INPUT_PROVENANCE.json`.

## 4. Method

Recovered formal semantics: mean EPE within each of 80 transitions, then equal mean across transitions. Contributions use exact transition-level weights `1/(80*n_t)`, not global sample-count weights. Top-k deletion recomputes both arm denominators.

## 5. Key commands/scripts

`run_r4cplus.py` was run with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`, and `MKL_NUM_THREADS=2`.

## 6. Key results

All six arm support counts are exactly equal; the 83 zero-support IDs are common. Formal decomposition reconstructs every contrast up to reduction-order residuals of at most `1.2836953722228372e-16`. Seed1 transport's unweighted per-ID mean is `-0.02121746136281137`, formal delta is `0.00126382946086534`, and descriptive aggregation shift is `0.02248129082367671`. Removing the single top harmful seed1 transport ID (23166; 57 transitions) makes the recomputed formal delta negative, but top-5% harmful mass share is `0.33051785964844455`, below the pre-registered 0.50 threshold; classification is `MIXED_CONCENTRATION_PATTERN`.

## 7. PASS / FAIL / UNRESOLVED

Formal semantics and contribution decomposition PASS. Frozen subgroup membership and authoritative spatial source-image provenance remain unresolved; no substitute subgroup or image was created.

## 8. Impact on future work

Primary conclusions remain unchanged: seed1 transport direction and two-seed transport consistency remain FAIL; identity and wrong-identity direction consistencies remain PASS. No seed2 or generation was launched.

## 9. Residual questions

Subgroup/spatial analysis can proceed only if pre-existing frozen membership and uniquely authoritative source-image provenance are recovered independently.
