# PHASE0D-2F / F1-R4 Summary

## 1. 阶段目标

精确定位 F1-R3 corrected-v2 preflight failure，并判断能否仅通过 binding recovery 恢复 RW/DT-FULL 正式配对评测。

## 2. Previous failure localization

F1-R3 的 RW 与 DT-FULL 首个且唯一 shared failure 均为 `tracks_sha`。Manifest 期望值是 63 位 `…ae33df…`，冻结 tracks 资产实际为 64 位 `…ae33ddf…`。其余 F1-R3 predicates 均 PASS。

## 3. Root-cause classification

根因为 `DATA_BINDING`，不是 path namespace、legacy representation、method binding 或 metric semantic incompatibility。Sidecars 原生 representation 已与 historical loader 一致，无需 adapter、复制、过滤、重采样或坐标修改。

## 4. Binding-only recovery

Builder 未改。Recovery manifest 更正 tracks SHA，重绑定新目录 suite/output，并以冻结 metric code 在 corrected-v2 sidecars 上校准 old-correct reproduction guard 的 Lab/RGB/TC-ME dataset bindings。N、资产内容、视频、mapping、visibility、timeline、patch 与 metric logic 均不变。

## 5. Semantic and legacy gates

Builder SHA 保持 `385b39…a0f0`，previous legacy exact equivalence 可复用。Recovered evaluator 的 forbidden diff hunks=0，17/17 metric region source SHA相等，normalized AST PASS。

## 6. Recovered preflight

RW 与 DT-FULL 的存在性、SHA、container、shape、dtype、N/T、timeline、mapping、patch bounds、expect、method path/label、decode、frame count 与 frame HW 全部 PASS。Attempt1 的三个 container label FAIL 是 preflight harness 字符串标签错误，已保留并修正；未改 evaluator/manifest/资产。

## 7. Formal paired result

Appearance 和 motion 均退出 0。按 evaluator frozen direction（全部 lower-is-better），RW 在 TC-MAR Lab mean/median、TC-MAR RGB-L1 mean/median/p95、TC-ME mean/median/p95 数值更低；DT-FULL 仅在 TC-MAR Lab p95 数值更低。这只是 seed0 numerical comparison，无显著性、superiority 或 non-inferiority 结论。

## 8. PASS/FAIL 与后续影响

`PHASE0D2F_F1R4_STATUS=PASS`，F1 恢复为 `PASS_WITH_BINDING_ONLY_CORRECTED_V2_EVALUATOR`，`PROCEED_TO_F2=True`。本任务没有执行 F2、GRID100、WM-0 或 generation。

## 9. 遗留问题

当前结果仅为单 seed0。Historical evaluator 的报告名称仍含 development 字样，这是冻结输出文本，未修改。后续阶段是否执行由用户另行决定。
