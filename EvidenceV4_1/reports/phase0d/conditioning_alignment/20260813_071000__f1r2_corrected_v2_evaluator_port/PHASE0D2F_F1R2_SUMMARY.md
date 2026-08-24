# Phase0D-2F / F1-R2 summary

## 1. 阶段目标

在 evaluator source SHA 不变的约束下，构造 corrected-v2 N=1257 evaluator contract，并先通过 N=1277 legacy-domain exact semantic equivalence 后进行正式配对重评分。

## 2. 审计问题

确认 Santa N=1277 expectation、sidecar binding 与 method binding 是否在 evaluator 源内，还是可由 external config/wrapper 合法提供。

## 3. 使用的数据

仅使用冻结 `eval_v3.py`（SHA `2b801bda…c08`）、历史 F1/F1-R evidence 与用户指定的 corrected-v2 SHA contract。

## 4. 使用的方法

静态只读源码/CLI 审计和 SHA 重算。未创建 port、未执行 evaluator、未产生 generation 或视频变换。

## 5. 关键命令/脚本

`sha256sum`、`nl -ba`、`rg`；审计 `CASES`、`method_paths`、`appearance_case` 和 `main`。

## 6. 关键结果

`N_EXPECTATION_LOCATION=SOURCE_HARDCODE`。`CASES['santa']['expect']['n']=1277` 位于 `eval_v3.py:77-84`，由 `appearance_case` 的 `assert n == cfg['expect']['n']`（`:808-832`）执行。legacy tracks/visibility paths 同在 `CASES`（`:65-75`）；DT_FULL 无可配置 method binding（`method_paths`, `:169-199`）。CLI 没有 config/sidecar/video/method override（`:2144-2186`）。

## 7. PASS/FAIL/UNRESOLVED 判断

`SOURCE_MODIFICATION_REQUIRED=True`；为了 N=1257 target，至少要编辑 source-resident dataset/method bindings and expectation。F1-R2 禁止 source modification，因此状态为 `FAIL_REQUIRES_METRIC_SOURCE_CHANGE`。Evaluator SHA 保持不变。

## 8. 对后续实验影响

协议规定在此处 STOP：未运行 legacy equivalence、corrected-v2 preflight 或正式评分；`PROCEED_TO_F2=False`，未执行 F2/GRID100/generation。

## 9. 遗留问题

需要单独授权且可审计的 evaluator architecture/configuration change，才能建立 corrected-v2 port；该变更不在本阶段授权范围内。N1277 数值持续仅为 `LEGACY_HISTORICAL_ONLY`。

## Code quota isolation audit

LOCAL_CODEX_ONLY_POLICY=ENABLED. SERVER_ROLE=SSH_EXECUTION_TARGET_ONLY. No server-side Codex CLI/agent, OpenAI/Codex API, CPA API LLM token use, or server-side LLM inference occurred. No existing remote helper/app-server was killed. QUOTA_ISOLATION_STATUS=PASS.
