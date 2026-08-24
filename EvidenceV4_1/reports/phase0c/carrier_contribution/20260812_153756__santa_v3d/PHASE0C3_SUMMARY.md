# Phase0C-3 — Carrier Contribution / Zero-Contribution Cause Audit

## 1. 阶段目标
解释冻结 V3D operator 下 155 个 persistent material carriers 的零贡献机制；不优化 carrier selection。

## 2. 审计问题
按 visibility、target validity、depth validity 与 collision loss gate 分解每个 carrier 的 20-slot funnel，并验证 Correct/Shuffled carrier-level invariance。

## 3. 使用的数据
0B-4R saved context/track_pos/third permutation 与 0C-1 contribution evidence；不使用 legacy N=1277。

## 4. 使用的方法
严格使用冻结 V3D semantics：source/target valid、sampled visibility、finite positive depth 才是 candidate；同 cell winner=min(depth, material_id)。

## 5. 关键命令/脚本
`run_phase0c3.py`；CPU-only read-only reconstruction，无 V3D rerun、GPU 或 carrier policy change。

## 6. 关键结果
- any candidate=1110；contributors=1102；zero=155。
- zero classes: Z0=147, Z1=0, Z2=0, Z3=8, Z_OTHER=0; sum=155。
- collision loss carriers=409；never in collision=640。
- candidate mean/p50/p95=7.769292/6.000000/20.000000; wins=7.184566/6.000000/19.000000。
- descriptive Spearman visible-vs-win is UNDEFINED_CONSTANT_VISIBLE_SLOT_COUNT (the visible-count input has zero variance); no correlation or significance claim is made.

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0C3_STATUS = PASS`；reconstructed win counts exactly match 0C-1; all preregistered Correct/Shuffled carrier invariants exact.

## 8. 对后续实验影响
zero-contribution 是当前 frozen operator gate 的机制描述，不是 performance bottleneck diagnosis，也不授权改变 selection、visibility、depth 或 collision policy。

## 9. 遗留问题
该审计不测量最终视频质量或方法 superiority；future performance work must remain a separate formal protocol.
