# PAPER-EXP-A1  pre-GPU closure

## Outcome

Formal runtime recovery, causal-contract construction, frozen launcher creation, and four GPU-disabled dry-runs PASS. No GPU model inference, video generation, or evaluation was run because the user supplied `GPU_AVAILABLE=False`.

## Formal runtime lineage

The recovered formal overlay is exactly the accepted C2/CS runtime: generate.py `45f7323f...`; wan_move.py `eae7f5a...`; trajectory.py `0c6bc94...`. The preview, epsilon, corrected-v2 tracks/visibility/depth/material IDs, scheduler, seed, prompt, source image, and decode domain are fixed in `FORMAL_RUNTIME_PROVENANCE.json`.

## A1 conditions

- SC: static target positions plus correct identity, byte-identical formal overlay.
- SS: static target positions plus the existing frozen CS derangement, byte-identical formal overlay.
- C2-NOVIS: only future visibility gating disabled in an isolated A1 overlay.
- C2-NODEPTH: only depth collision arbitration removed; ascending material ID is the frozen replacement rule.

SC/SS retain future visibility and depth. No new permutation was generated.

## Dry-run and resource status

All four launch cases passed filesystem-only dry-run. The launcher SHA is recorded in the checksum file. GPU0, GPU1, and GPU2 had external compute processes in the gate snapshot; GPU3 was not used. Model inference was intentionally not reached.

## Next execution point

After a fresh clean-GPU gate sets GPU availability true, run SC and SS with the frozen launcher, validate engineering, then use the recovered corrected-v2 evaluator. C2-NOVIS and C2-NODEPTH remain Priority B and must not run before Priority A closes.
