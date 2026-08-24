# Phase 0D-2F / F4-R3 Formal Candidate Unblinding

## 1. 阶段目标

在 F4-R2 冻结的恢复 host runtime 上首次正式解盲三条候选。

## 2. 审计问题

检验各 arm 是否同时降低 TC-MAR Lab mean 与 TC-ME mean，相对 DT-FULL。

## 3. 使用的数据

五条 SHA 核验视频、corrected-v2 N=1257 IDs/tracks/visibility、冻结 F3 子组 mask。

## 4. 使用的方法

精确 evaluator SHA、host Wan-Move Python、`CUDA_VISIBLE_DEVICES=1`；候选经隔离符号绑定进入固定 `dt_full` 槽位。RW/DT-FULL exact baseline gate 先于候选运行。

## 5. 关键命令/脚本

`run_candidates_restored.sh`、`run_frozen_subgroup_diagnostics_retry.py`、`build_f4r3_reports.py`。旧容器 runtime 候选数值未参与任何判断。

## 6. 关键结果

WM-0 两项均恶化。FRAG 的 ME 降低 `0.052056492720716996`，但 MAR 上升 `0.01661248498886181`。GRID100 的 ME 降低 `0.02759287289356105`，但 MAR 上升 `6.08023654973973`。无 primary-direction pass。

## 7. PASS/FAIL/UNRESOLVED 判断

输入、runtime、baseline gate、formal evaluation、ALL diagnostic reproduction、子组完成均 PASS；primary direction is FAIL for all candidates. F4-R3 status COMPLETE.

## 8. 对后续实验影响

冻结路线为 CASE_E：停止 sparse-prune primary-direction route，下一动作 ROUTE_REASSESSMENT。不得自动开展额外 GPU 实验。

## 9. 遗留问题

只完成 seed0 screening；不能声称 statistical superiority。Q4_AND_STABLE N=13 不作强证据。FRAG 非纯 fragmentation intervention，GRID100 非 exact train-distribution proof。
