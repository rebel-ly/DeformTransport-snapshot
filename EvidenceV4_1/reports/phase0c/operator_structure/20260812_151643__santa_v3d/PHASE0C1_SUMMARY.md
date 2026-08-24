# Phase0C-1 — Operator Structural Intervention Audit

## 1. 阶段目标
审计冻结 patched Wan-Move V3D transport operator 的 target-cell 写入、collision arbitration、winner 与 source contribution 结构。

## 2. 审计问题
确认 Correct 与 Identity-Shuffled 的 operator 结构是否完全相同，以及干预是否仅改变 source feature value。

## 3. 使用的数据
corrected-v2 Santa（N=1257,T=81）、0B-4R 保存的 post-create context、track_pos、第三次 randperm、authoritative depth/IDs；未使用历史 N=1277 bridge。

## 4. 使用的方法
按冻结 trajectory.py V3D 逻辑：future slot tau 使用 frame tau×4；候选需 source/target 坐标有效且 visibility 为真；按 target cell 分组；depth 非有限或 <=0 跳过；winner key 为 `(depth, material_id)` 最小值。

## 5. 关键命令/脚本
`run_phase0c1.py`；输入与输出均在本目录及 0B-4R evidence。源码 SHA drift gate PASS。

## 6. 关键结果
- 总候选 assignments=768；总 unique target writes=718。
- collision cells=49；collision carriers=99；global max multiplicity=3。
- carriers with contribution=718；zero=539；mean/p50/p95/max wins=0.571201/1.000000/1.000000/1。
- Correct/Shuffled target support、collision、winner IDs/depth、per-carrier counts 全部 exact equal；write support 为 9031 cells each。

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0C1_STATUS = PASS`。任何结构 mismatch 均为 0；zero-contribution set equal=True。

## 8. 对后续实验影响
可将该 intervention 解释为保持 operator structural transport 不变、仅改变 source feature value 的 correspondence audit；不进入视频生成性能结论。

## 9. 遗留问题
需要后续独立性能实验验证 correspondence 是否改善最终视频质量；本阶段不证明性能 superiority，也不改变 Phase0B 对完整 polyline kinematic invariance=FAIL 的记录。
