# Phase0D-2F / F1 summary

## 1. 阶段目标

审计已有 Santa RealWonder 视频，并在同一 corrected-v2 N=1257 contract 下与 DT-FULL Replay A 配对重评。

## 2. 审计问题

确认 RealWonder lineage/timeline/N1277 independence，并验证冻结 evaluator 是否可无修改地接受两条视频与 corrected-v2 assets。

## 3. 使用的数据

Santa RealWonder canonical run/log/validation、DT-FULL Replay A evidence、F0 contract，以及 SHA-pinned `eval_v3.py`。

## 4. 使用的方法

有界、只读 hash/manifest/source audit。未重生成，未运行 evaluator，未修改代码或资产。

## 5. 关键命令/脚本

普通 `rg`、`find`、`sha256sum`、`sed`/`nl`。所有推理由本地 Codex 完成；服务器仅作 shell/file target。

## 6. 关键结果

唯一 RW baseline 是 81-frame Santa aligned video，source/action/physics/timeline 均 PASS，N1277 generation consumption=0。DT-FULL Replay A lineage PASS。冻结 evaluator 却固定旧 N=1277 bridge tracks/visibility，且无 DT-FULL video/asset override。

## 7. PASS/FAIL/UNRESOLVED 判断

RW video reusable，但 `EVAL_CONTRACT_GATE=FAIL`、`PAIRED_EVALUATION_CONTRACT=FAIL`。执行会错误消费 N=1277 legacy support，故未评分。

## 8. 对后续实验影响

F1 为 `EVALUATOR_FAIL`，`PROCEED_TO_F2=False`。需用户单独授权一个 corrected-v2 evaluator binding/recovery phase；F1 不修改 evaluator。

## 9. 遗留问题

唯一遗留为建立 SHA-pinned、可接受 N=1257 support 和两条 explicit video paths 的 evaluator contract；不得将旧 N=1277 数字冒充本阶段结果。

## Code quota isolation audit

LOCAL_CODEX_ONLY_POLICY=ENABLED. SERVER_ROLE=SSH_EXECUTION_TARGET_ONLY. No server-side Codex CLI/agent or OpenAI/CPA LLM API was invoked; no helper processes were killed.
