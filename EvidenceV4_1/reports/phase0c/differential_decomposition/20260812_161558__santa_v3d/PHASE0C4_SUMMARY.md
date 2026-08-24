# Phase0C-4 — Correct-vs-Shuffled Operator Differential Decomposition

## 1. 阶段目标
独立从 frozen source-feature correspondence、winner mapping 和 write equation 构造 predicted delta，并与 saved observed edited_y delta 完全比较。

## 2. 审计问题
检验所有 Correct-vs-Shuffled conditioning differences 是否完全由不同 source feature 经相同 V3D winners 写入相同 target cells 所解释。

## 3. 使用的数据
corrected-v2 N=1257 的 0B-4R saved context, track_pos, edited_y 和 0C-1 arbitration semantics；未使用 N=1277。

## 4. 使用的方法
CPU-only 使用冻结 `_dt_bilinear_source_features` 对精确 frozen synthetic y 重建 source vectors；使用 min(depth, material_id) 重建 winners；按 direct overwrite equation 生成 predicted delta，而非从 observed delta 反推。

## 5. 关键命令/脚本
`run_phase0c4.py`，`operator_write_equation.txt`。源码 gate PASS；GPU2 仅供 normal package import 可见，所有 functional tensors 为 CPU。

## 6. 关键结果
observed/predicted nonzero scalars=27209/27209；observed-only/predicted-only=0/0；residual=0，max/mean=0.0/0.0。channel counts=[121, 9031, 9026, 9031]。Any/all/equal support cells=9031/121/0。

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0C4_STATUS = PASS`。winner mapping equal=True；observed delta exact equals independently constructed predicted delta=True.

## 8. 对后续实验影响
该 synthetic functional audit shows the operator differential is fully explained by source-feature correspondence propagated through the same frozen V3D write operator. It does not establish real-latent amplitude or video-quality superiority.

## 9. 遗留问题
synthetic channels are harness codes, not real Wan semantic channels; later video/performance claims require independent formal evaluation.
