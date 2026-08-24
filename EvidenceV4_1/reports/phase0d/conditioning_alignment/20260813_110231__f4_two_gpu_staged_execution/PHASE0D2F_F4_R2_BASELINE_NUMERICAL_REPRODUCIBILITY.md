# Phase 0D-2F / F4-R2 Baseline Numerical Reproducibility

## 1. 阶段目标

在候选指标保持盲态时，审计 TC-ME 基线数值复现，并预先冻结后续正式评价运行时政策。

## 2. 审计问题

确定 F4-R1 的微小 TC-ME 偏差是语义/输入错误、同一运行时非确定性，还是运行时漂移。

## 3. 使用的数据

仅使用 RW 和 DT-FULL 的冻结视频、corrected-v2 sidecars，以及 SHA-256 为 `e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5` 的 evaluator。未读取、打印或比较任何 F4 候选指标。

## 4. 使用的方法

恢复 F1-R4 provenance；在一个固定容器 runtime 中对仅基线套件顺序运行 5 次；再以 F1 记录的 host Python 和 `CUDA_VISIBLE_DEVICES=1` 作 baseline-only restoration probe。

## 5. 关键命令/脚本

`scripts/run_f4r2_baseline_repeats_retry.sh`；历史 suite 路径在容器中为零帧的首次失败被单独保留，成功统计只使用 `retry_run01`–`retry_run05`。原始 runtime probe 输出位于 `evaluation/f4r2_original_runtime_probe/`。

## 6. 关键结果

容器五次重复完全一致：RW mean `0.586977941501839`，DT-FULL mean `0.7265571851768204`，两者 range、std、最大 pairwise 差均为 0。它们与历史值固定不同。恢复 host runtime 后，RW `0.5869890665947547`、DT-FULL `0.7265499674289193`，mean、median、p95 均精确复现历史输出。

## 7. PASS/FAIL/UNRESOLVED 判断

`SOURCE_CLASS = CLASS_C_RUNTIME_DRIFT`；无语义 evaluator 变化、无输入变化。历史基线与恢复运行时兼容，政策冻结通过：`F4_R2_STATUS = PASS_REPRODUCTION_POLICY_FROZEN`。

## 8. 对后续实验影响

未来 F4-R3 若获授权，只能在恢复的 host runtime、原 Python、`CUDA_VISIBLE_DEVICES=1`、精确 evaluator SHA 下重新验证 RW/DT-FULL 后，才可对候选解除盲态。不得使用任意 epsilon。

## 9. 遗留问题

历史/当前 per-transition arrays 均未由冻结 evaluator 保存，故 raw-transition audit 为 UNRESOLVED。候选保持未解盲；本阶段未执行任何 generation、replay、promotion 或路由决定。
