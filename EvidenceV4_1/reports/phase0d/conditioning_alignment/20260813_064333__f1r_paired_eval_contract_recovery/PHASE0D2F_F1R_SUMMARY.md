# Phase0D-2F / F1-R summary

## 1. 阶段目标

定位 previous F1 的 exact gate/paired-contract failure，并仅在语义保持恢复合法时完成 corrected-v2 配对评分。

## 2. 审计问题

确定失败是 scientific/evaluation-contract incompatibility，还是 manifest/path/comparison engineering issue。

## 3. 使用的数据

Previous F1 archive、冻结 `eval_v3.py`、corrected-v2 N=1257 IDs/tracks/visibility，以及既有 RW canonical 和 DT-FULL Replay A MP4。

## 4. 使用的方法

只读源代码与证据追溯；使用冻结 evaluator 的 `read_video_common` 做 deterministic loader preflight。没有正式评分、生成、资产或 evaluator 修改。

## 5. 关键命令/脚本

`sed`/`nl`/`rg`/`sha256sum` 源码审计；existing wan-move Python 3.11 runtime 执行本目录临时 loader-only preflight；输出 JSON 证据。

## 6. 关键结果

Previous archive 的 exact persisted failing field 是 `evaluation_set_exact_equal=UNVERIFIABLE_WITH_FROZEN_EVALUATOR`。首个可达 executable failure 是 `EVALUATOR_ACCEPTS_CORRECTED_V2_N`: `eval_v3.py:829-832` asserts N equals the Santa configuration's fixed 1277. The source binds N=1277 tracks/visibility (`:65-84`, `:790-806`) and exposes no DT-FULL video binding (`:169-199`, `:2146-2186`). Both MP4s loader-preflight PASS at 81×464×832 and corrected-v2 sidecars are N=1257.

## 7. PASS/FAIL/UNRESOLVED 判断

RW_PREFLIGHT=PASS and DTFULL_PREFLIGHT=PASS, but formal frozen-evaluator invocation is FAIL. This is a scientific/evaluation-contract incompatibility of the frozen executable. A wrapper cannot repair it without changing which sidecars/method map/expected guard the evaluator semantically uses. `PHASE0D2F_F1R_STATUS=FAIL_SCIENTIFIC_EVAL_CONTRACT`.

## 8. 对后续实验影响

No formal corrected-v2 RW-vs-DT-FULL metrics exist. `PROCEED_TO_F2=False`; F2 was not executed. Any N=1277 values remain `LEGACY_HISTORICAL_ONLY`.

## 9. 遗留问题

A separately authorized evaluator revision/configuration that has an explicit corrected-v2 N=1257 binding and two method video bindings would be required. It is outside this phase because evaluator semantics/source/config modification is prohibited.

## Code quota isolation audit

LOCAL_CODEX_ONLY_POLICY=ENABLED. SERVER_ROLE=SSH_EXECUTION_TARGET_ONLY. No server-side Codex CLI, Codex agent, OpenAI API, CPA API LLM use, or LLM inference occurred. No existing remote helper/app-server process was killed. QUOTA_ISOLATION_STATUS=PASS.
