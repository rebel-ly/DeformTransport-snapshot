# Phase 0D-2F / F3 GPU Arm Construction & Freeze

## 1. 阶段目标

在不执行 diffusion generation 的前提下，构造并冻结下一轮 seed0 screening 的 conditioning subsets、WM-0 gate、evaluation subgroups 与 run manifests。

## 2. 审计问题

验证 formal N=1257 identity；构造 FRAG-PRUNE、GRID100-CENTER、count-matched spatial control；检查 pure-row-subset 性；精确复现 F2-R FRAG 结构预计算；证明 WM-0 的空轨迹条件为 no-op；在视频存在前冻结评测和 stop-loss 合同。

## 3. 使用的数据

使用 corrected-v2 Santa point IDs、tracks、raster-defined operational visibility、authoritative depth、F2 transport join、Phase0C frozen FULL V3D winner maps，以及既有 formal runtime binding。实际 tracks SHA 为 `a8b6...33ddf7...`；附件中的 SHA 少一个 `d`，仅为 63 位，已与此前 frozen replay 记录交叉确认。

## 4. 使用的方法

采用 480x832 source domain 的 32x32 surrogate grid：row width 15、column width 26、半开区间且最终边界归入 bin 31。100 bins 由 occupied-bin centers 的 deterministic farthest-point ordering 选出；每 bin 选择到中心最近的 source carrier，距离并列取最小 material ID。COUNT218 使用同一 bin ordering 的一轮一 carrier，随后 round-robin next-nearest unused carrier。全部选择不使用 error、motion、depth、collision 或 visibility（FRAG 的 preregistered switch rule 除外）。

## 5. 关键命令/脚本

`scripts/construct_f3.py` 与 `scripts/finalize_f3_structural.py`。两者均为 CPU-only 的 ndarray/JSON 处理；未加载 VAE、transformer 或 diffusion。F2-R write-support 按其原始定义复现：对 Phase0C FULL frozen winner maps 按 retained material IDs 计数；这与重新在 subset 内 arbitration 的辅助数值明确区分。

## 6. 关键结果

Formal identity PASS，EVAL_N=1257。grid assignment=1257、out-of-domain=0、occupied bins=235。FRAG-PRUNE retained=218、removed=1039、frozen-FULL-winner support=1799（89.95/slot，1.441506%）；F2-R 复现 PASS。GRID100 support=836（41.8/slot，0.669872%）；COUNT218 support=1588（79.4/slot，1.272436%）。三种 conditioning artifact 都是 exact row subset。WM-0 的 N=0 path 有 0 trajectory/depth/winner writes 且 `edited_y==y`，gate PASS。

六个 subgroup 已固定：ALL=1257、HIGH_MOTION_Q4=314、FRAGMENTED_SWITCH_GE3=1039、Q4_AND_FRAGMENTED=301、Q4_AND_STABLE=13、ZERO_SWITCH_POSITIVE_VISIBLE=62。Q4 stable 保留且标记 small-N。

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2F_F3_STATUS=PASS`。所有必需 F3 checks 通过；WM-0 GPU arm allowed。R2/R3 的 F2-R 解释限制保持不变，未在本阶段重新解释或修复。

## 8. 对后续实验影响

只允许下一阶段按冻结 manifest 运行 3 个 seed0 videos：WM-0、DT-FRAG-PRUNE、DT-GRID100-CENTER。每个仍须以 full N=1257 evaluation support 评估，而非 conditioning subset。promotion 仅当 TC-MAR 与 TC-ME 都相对 DT-FULL 下降；若 aggregate gain 伴随 Q4 regression，必须标记。COUNT218 只在 FRAG strongest/promoted 时作为后续 count-matched control。

## 9. 遗留问题

F3 不给出任何视频质量结论。FRAG-PRUNE 是 fragmentation-targeted strong sparsification，绝非纯 fragmentation intervention。两个 write-support 口径已分别保存：F2-R reproduction 为 frozen FULL winners 的 retained-ID count；subset re-arbitration 是辅助结构统计，不能替代前者。未执行 GPU/diffusion generation。
