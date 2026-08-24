# Phase0C-4A — Constant-channel Numerical Clarification

Synthetic source channel0 is exactly all one: min=max=1, unique count=1. CPU frozen bilinear lookup nevertheless returns exact float32 values differing from one for Correct=10 and Shuffled=10 source carriers.

The actual maxima are Correct deviation=5.96046447754e-08, Shuffled deviation=5.96046447754e-08; source Correct-Shuffled max/mean absolute difference=5.96046447754e-08/9.48363498843e-10. `numpy.spacing(float32(1.0))=1.19209289551e-07`; spacing ratios are recorded in JSON only as numerical diagnostics, not thresholds.

There are 20 source carriers with exact channel0 difference. The frozen winner mapping uses them 121 times, yielding exactly 121 predicted propagated channel0 nonzeros, which matches the frozen Phase0C-4 observed count 121.

`CHANNEL0_NONZERO_ORIGIN = BILINEAR_FLOAT32_SOURCE_LOOKUP_NUMERICS`. No replace_feature/operator execution was performed. `PHASE0C4_CORE_STATUS = PASS` remains unchanged.
