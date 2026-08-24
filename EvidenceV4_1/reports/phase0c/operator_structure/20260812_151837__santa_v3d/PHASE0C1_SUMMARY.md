# Phase0C-1 — Operator Structural Intervention Audit

## 1. 阶段目标
审计冻结 patched Wan-Move V3D transport operator 的 target-cell 写入、collision、winner arbitration 与 source-carrier contribution 结构。

## 2. 审计问题
验证 Correct / Identity-Shuffled 的 target support、collision structure、winner IDs/depth、per-carrier counts 与 zero-contribution set 是否完全相同，判断 structural intervention 是否只保留 source-feature-value 的差异。

## 3. 使用的数据
使用 corrected-v2 Santa（N=1257,T=81）及 0B-4R 已保存 CPU context、track_pos、expected third randperm 与 authoritative depth/IDs；未使用历史 N=1277 bridge。

## 4. 使用的方法
严格重建冻结 `trajectory.py` V3D：slot tau 使用 sampled frame tau×4，要求 source/target 有效并且 visibility 为真；按 target cell 分组；丢弃 non-finite 或 depth<=0；winner 为最小 `(depth, material_id)`。

## 5. 关键命令/脚本
`run_phase0c1.py` 记录冻结重建规范。此阶段是 CPU-only、read-only 分析；source SHA drift gate PASS。

## 6. 关键结果
- 20 future slots 的 candidate assignments=9766，unique target writes=9031（与 0B-4R 9031 个 future support cells 一致）。
- collision cells=722，collision carriers=1457，global max multiplicity=3。
- 1102 / 1257 carriers 有至少一次贡献；zero-contribution=155；mean/p50/p95/max winner count=7.184566/6.000000/19.000000/20。
- 逐项 Correct/Shuffled structure mismatch 均为 0；zero-contribution set 完全相同。

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0C1_STATUS = PASS`。所有预注册结构 invariant exact equal。

## 8. 对后续实验影响
该干预保持 operator structural transport 不变，只有 source feature value 层面不同；不等价于、也不证明最终视频 quality superiority。

## 9. 遗留问题
后续正式性能实验仍需独立验证最终视频效果；本阶段不改变 Phase0B 的 complete polyline kinematic invariance=FAIL，也不允许使用 legacy/rejected N=1277 Santa evidence。
