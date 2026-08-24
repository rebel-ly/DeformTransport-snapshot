# F4-R3 Morning Handoff

## Execution integrity

The R2 policy SHA and exact evaluator identity passed; all five video and corrected-v2 sidecar SHAs passed. A fresh RW/DT-FULL gate under the restored host runtime (`CUDA_VISIBLE_DEVICES=1`) reproduced all stored primary aggregates exactly before candidate unblinding.

## Primary metrics (lower is better)

| Method | TC-MAR Lab mean | TC-ME mean | Pass vs DT-FULL |
|---|---:|---:|---|
| RW | 13.639900159270573 | 0.5869890665947547 | reference |
| DT-FULL | 17.144317299874714 | 0.7265499674289193 | reference |
| WM-0 | 44.47206866882119 | 3.618652456619911 | False |
| DT-FRAG-PRUNE | 17.160929784863576 | 0.6744934747082023 | False |
| DT-GRID100-CENTER | 23.224553849614445 | 0.6989570945353583 | False |

No candidate improves both frozen primary metrics. FRAG improves TC-ME but worsens TC-MAR by `0.01661248498886181`; GRID100 improves TC-ME but worsens TC-MAR by `6.08023654973973`; WM-0 worsens both.

## Frozen decision

`CASE_E_NO_ARM_PROMOTED`; `PROMOTED_ARM = NONE`; next action is `ROUTE_REASSESSMENT`. No replay, seed sweep, COUNT218, GRID100-STABLE, or F5 was run.

## Safety and interpretation

No arm satisfies the aggregate dual-primary prerequisite for the high-motion regression flag. FRAG is not a pure fragmentation intervention: it also reduces count to K=218. GRID100 is a 32x32 source-grid surrogate, not evidence of exact training-distribution matching. Results are seed0 descriptive screening only; no statistical superiority claim.
