# Phase0C-3A-R — Visible-count Variance Clarification

Visible sampled-slot count is not constant. The previously recorded zero-variance explanation was invalid. Spearman rho was not recomputed because the preregistered scipy dependency was unavailable; this statistic is descriptive and non-gating.

The frozen Phase0C-3 core result is unchanged:

- contributors = 1102
- zero contribution = 155
- Z0 = 147
- Z3 = 8
- Correct/Shuffled funnel mismatch = 0
- reconstructed wins exact match 0C-1
- `PHASE0C3_CORE_STATUS = PASS`

CSV verification: N=1257, visible min/max/unique/mean/population variance = 0/20/21/7.769291965/40.5800275814; win min/max/unique/mean/population variance = 0/20/21/7.184566428/36.9126337221. Z0 visible counts are all zero; all Z3 visible counts are positive.

`SPEARMAN_STATUS = NOT_COMPUTED_DEPENDENCY_UNAVAILABLE`. No rho was estimated and no significance or causal claim is made.
