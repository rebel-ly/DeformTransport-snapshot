# PHASE0D-2F / F1-R3 Summary

## 1. 阶段目标

在不修改历史 `eval_v3.py` 的前提下，构造 binding-only corrected-v2 evaluator，证明静态/legacy 功能等价，并仅在全部门禁通过后重评分 RW 与 DT-FULL。

## 2. 审计问题

核心问题是 derived evaluator 是否只改变 data/method/output binding，且 requested corrected-v2 SHA、视频域与方法映射是否能在正式评分前完整通过 preflight。

## 3. 使用的数据

Legacy N1277 仅用于语义等价 fixture。Corrected-v2 使用 N=1257 IDs/tracks/visibility、RW canonical 与 DT-FULL Replay A。IDs、visibility、两视频 SHA 均通过；requested tracks SHA 为 63 位，现场资产 SHA 为 64 位，精确不相等。

## 4. 使用的方法

确定性 builder 从 SHA-pinned original 生成 byte-identical legacy fixture 与 corrected port。17 个 metric semantic regions 做 source SHA 与 normalized AST 精确比较；legacy appearance 与 Santa TC-ME 用同 runtime/sidecars/videos 执行双跑并逐字比较 JSON。

## 5. 关键命令/脚本

`scripts/build_binding_only_eval_port.py` 生成 derived evaluator；`scripts/audit_eval_semantic_equivalence.py` 审计 diff/source/AST；冻结 Python 3.11 runtime 执行 legacy appearance、Santa motion 与 corrected-v2 loader preflight。

## 6. 关键结果

Original SHA before/after 均为 `2b801b…c08`。Legacy fixture 与 original byte-identical。Corrected port 5 个 allowed diff hunks、0 forbidden hunks；17/17 semantic region SHA 相等且 AST PASS。Legacy appearance/motion 报告逐字相等，max abs diff=0。Preflight 除 tracks SHA 外全部通过。

Requested tracks SHA: `a8b6b9894fb751ba525f0fc6ee8ae91e0c86752344257ae33df7fdebfb51929`（63 hex）。

Actual tracks SHA: `a8b6b9894fb751ba525f0fc6ee8ae91e0c86752344257ae33ddf7fdebfb51929`（64 hex）。

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2F_F1R3_STATUS=FAIL_CORRECTED_V2_PREFLIGHT`。该失败由冻结 contract 与资产 SHA 精确不一致触发，不是 evaluator semantic-equivalence 失败。按止损规则，正式 RW/DT-FULL 指标未运行。

## 8. 对后续实验影响

`PROCEED_TO_F2=False`。F1 尚未恢复；不得将任何 N1277 数字升级为 current formal，也不得声明 RW/DT-FULL numerical winner。

## 9. 遗留问题

唯一阻断项是外部冻结请求中的 malformed tracks SHA。需用户另行明确授权/更正 contract 后才能启动新的、独立时间戳任务；本目录不应被回写或改判。
