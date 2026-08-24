# Phase 0D-4D-R3H — C1/C2 within-protocol unseal

## 1. 阶段目标

Open only the C1 versus C2 within-protocol result seal and reconcile the live E0 PID.

## 2. 审计问题

Verify equal corrected-v2 evaluation binding and equal 25-step enabled-path runtime, then evaluate the preregistered primary TC-ME before secondary TC-MAR.

## 3. 使用的数据

C1 K=0 and C2 Correct-K1257 MP4s; frozen N=1257 material IDs; frozen corrected-v2 evaluator.

## 4. 使用的方法

The evaluator SHA was mechanically verified. TC-ME was run C1 then C2 on GPU0; TC-MAR was run CPU-only in the same frozen evaluator. No cross-baseline decision was read or reported.

## 5. 关键命令/脚本

`run_cpair_primary.py`, `run_cpair_secondary.py`, and read-only `/proc`/`nvidia-smi` reconciliation of PID 174965.

## 6. 关键结果

TC-ME: C1 0.47438763126111494; C2 0.42167144903579584; delta -0.0527161822253191. TC-MAR: C1 10.2725907004925; C2 10.261138980605933; delta -0.011451719886567.

## 7. PASS/FAIL/UNRESOLVED

Within-protocol seed-0 incremental transport signal: PASS. E0 remains RUNNING_ACTIVE. Preview-collapse diagnostics are unresolved because the required frozen support mask was not persisted.

## 8. 对后续实验影响

The permitted statement is limited to an incremental seed-0 improvement within the frozen Preview-SDEdit protocol. Cross-baseline attribution remains closed pending the E0 bridge.

## 9. 遗留问题

Do not start further arms. Let E0 continue; user decision is required before any follow-on experiment.
