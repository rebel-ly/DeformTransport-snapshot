# Phase 0D-2F / F4-R1 Formal Post-Run Recovery Summary

## 1. 阶段目标

在不重新生成任何视频的条件下，以精确冻结的 corrected-v2 evaluator 恢复 F4 五方法正式评价。

## 2. 审计问题

确认 evaluator 身份、五方法输入身份和 RW/DT-FULL 基线复现是否满足 F4-R1 的强制门。

## 3. 使用的数据

corrected-v2 Santa support N=1257；RW、DT-FULL 与三条已通过完整性审计的 F4 seed0 视频。F3 manifests、IDs、tracks 和 visibility 均已核验。

## 4. 使用的方法

使用 SHA-256 为 `e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5` 的冻结 evaluator。每个新 arm 仅通过独立的符号链接绑定到 evaluator 固定的 `dt_full` 槽位；未修改 evaluator、视频或 conditioning artifacts。

## 5. 关键命令/脚本

`scripts/run_f4r1_frozen_eval.sh` 在验证后的 `/workspace/tools/miniforge3/envs/wan-move/bin/python` 环境中运行 appearance 和 motion。运行日志与 JSON 位于 `evaluation/frozen_evaluator_runs/`。

## 6. 关键结果

RW 与 DT-FULL 的 TC-MAR Lab mean 精确复现。RW TC-ME residual 为 `-0.00001112509291572936`，DT-FULL TC-ME residual 为 `+0.0000072177479011076`。预注册材料没有定义可接受误差。

## 7. PASS/FAIL/UNRESOLVED 判断

`FROZEN_EVALUATOR_RECOVERY = PASS`；`FORMAL_FIVE_METHOD_INPUT_IDENTITY = PASS`；`RW_BASELINE_REPRODUCTION = FAIL`；`DTFULL_BASELINE_REPRODUCTION = FAIL`；`F4_FORMAL_RESULTS_VALID = False`；`PHASE0D2F_F4_STATUS = BLOCKED_BASELINE_REPRODUCTION`。

## 8. 对后续实验影响

必须在明确 baseline TC-ME 的确定性/容差处理前停止；不得基于已保存的候选诊断作 promotion、route 或下一轮 GPU 设计决定。

## 9. 遗留问题

需由用户决定如何处理冻结 evaluator 在当前运行时产生的微小 TC-ME 数值残差。未运行任何 replay、seed1-4、COUNT218、GRID100-STABLE、F5 或新的 generation。
