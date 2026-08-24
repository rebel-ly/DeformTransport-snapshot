# Phase0D-2 — Same-Seed Replay Determinism Audit

## 1. 阶段目标

Audit final decoded-video replay determinism for the frozen corrected-v2 Santa Correct V3D seed=0 contract.

## 2. 审计问题

Determine whether two paired seed=0 replays produce exactly identical decoded RGB frames across GPU1 and GPU2.

## 3. 使用的数据

Phase0D-1R frozen corrected-v2 N=1257 runner and formal input contract; Run A/B pre-generation manifests; GPUs 1 and 2.

## 4. 使用的方法

The paired manifests passed before launch, and two authorized seed=0 processes were launched concurrently on the two idle L40 GPUs. No other seed was invoked. Intended decode/comparison was not reached.

## 5. 关键命令/脚本

Frozen runner: `formal_run_corrected_v2_v3d.sh`. Complete stdout/stderr from both run directories are copied to this report; original run directories remain intact.

## 6. 关键结果

Both processes failed identically during `generate.py` import, before model/checkpoint loading or output creation. `dashscope` transitively imported `cryptography`, whose Rust binding requires unavailable `GLIBC_2.18`. GPU processes remained empty. Neither output MP4 exists, so no decoded RGB tensors exist to compare.

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2_STATUS = ENGINEERING_FAIL`. This is not a determinism FAIL and `SAME_SEED_REPLAY` is not evaluated. The launcher did not persist each child wait status, so the child exit code is honestly recorded as `NOT_CAPTURED_BY_LAUNCHER`, not guessed.

## 8. 对后续实验影响

Do not proceed to Phase0D-3. A corrected compatible execution environment or an approved engineering recovery is required before the same frozen seed=0 replay can be reattempted; seed and formal contract must remain unchanged.

## 9. 遗留问题

- glibc/cryptography/dashscope import incompatibility blocks generation.
- no decoded RGB output exists for determinism assessment.
- formal future-video GT/evaluator remains unresolved; Phase0D scope remains pairwise-only.
- no seeds 1–4 were run.
