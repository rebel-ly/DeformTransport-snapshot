# PAPER-EXP-A1 runtime provenance

## Formal runtime recovery

PASS.  The currently available accepted overlay exactly matches the immutable hashes recorded by the accepted C2/CS manifests:

- generate.py: `45f7323f22d7bb7d593b949fa48e6cf764d08cafeaf8863d726df9a663b21b85`
- wan/wan_move.py: `eae7f5a86f39164f3ad1ce3b8db4a974f4a71f42c2898402f029bb9db77c32f7`
- wan/modules/trajectory.py: `0c6bc94d8ce1f885f0333314a9b201a650163cd209b2a3b3f95b4f3a35a49dae`

C2 and CS share the recovered preview latent, seed-0 epsilon, 40-step FlowUniPC scheduler with shift 3.0, begin index 15 (25 effective denoise steps), corrected-v2 N=1257 inputs, source image, prompt, and bf16 480*832 generation / 832*464 decode contract.

## Counterfactual construction

SC and SS use the byte-identical formal overlay. Their tracks have shape [1,81,1257,2] and every future target coordinate is their arm-specific source-frame coordinate. Future visibility and depth sidecars remain the accepted ones.

C2-NOVIS and C2-NODEPTH use only the isolated A1 overlay copy. It adds the preregistered visibility switch and deterministic material-ID collision key switch. It does not modify accepted artifacts or the formal overlay.

## Runtime evidence

Accepted C2 wall time: 2026-08-14T18:13:43Z to 2026-08-14T18:49:40Z = 35.95 min.
Accepted CS wall time: 2026-08-14T20:28:10Z to 2026-08-14T20:59:33Z = 31.38 min.
Mean generation estimate: 33.67 min/arm. Priority-A two new generations estimate: 67.33 min, before evaluation. No persisted accepted evaluator wall-time marker was uniquely recovered, so evaluation time is recorded as unresolved rather than invented. The prior 60-minute ceiling was explicitly cancelled by the user.

## Resource gate

The user supplied GPU_AVAILABLE=False. A contemporaneous read-only snapshot shows external compute processes on GPU0, GPU1 and GPU2; GPU3 remains forbidden. No model inference or GPU generation was attempted. The frozen launcher is ready for a future clean-GPU gate.
