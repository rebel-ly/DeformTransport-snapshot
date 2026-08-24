# PHASE0D-2F / F2 Summary

## 1. 阶段目标

在不优化、不生成新 diffusion 视频的前提下，定位 RW 与 DT-FULL corrected-v2 的 TC-MAR/TC-ME gap机制，并为未来稀疏干预提出受证据约束的假设。

## 2. 审计问题

审计 formal metric定义、逐样本复现、1257-ID operator join、error distribution/分层，以及 RW coarse 与 DT real edited_y conditioning-level表现。

## 3. 使用的数据

使用 F1-R4 frozen evaluator与两个 canonical MP4、corrected-v2 N1257 sidecars、Phase0C carrier funnel/winner evidence、authoritative aligned 3D trajectories、canonical RW exact 81-frame simulation，以及一次允许的 Wan VAE-only DT reconstruction。

## 4. 使用的方法

独立脚本复用 formal patch/Lab/aggregate和 RAFT/bilinear-flow函数。逐 carrier MAR、逐 transition×carrier ME精确复现 F1后，按 material ID join operator变量，采用预先给定 visibility bins、固定时间半区及明确标为 exploratory 的 exposure quartiles。未安装 scipy，rank correlations记为 dependency unavailable。

## 5. 关键命令/脚本

`extract_per_sample_diagnostics.py`、`localize_errors.py`、`diagnose_rgb_condition.py`、`reconstruct_real_edited_y.py`。VAE reconstruction 未加载/运行 transformer、text encoder或 diffusion。

## 6. 关键结果

逐样本 formal 最大差为0；ID join 1257/1257。DT 的 paired Lab delta median为+4.779，只有29.3% carriers更好，但在RW top-5% tail中92.9%更好，解释了mean/median输而marginal p95赢。ME early gap 0.234、late 0.045。3D motion Q1→Q4 的 MAR gap为-1.821、2.511、2.897、9.039，ME gap为-0.026、0.136、0.200、0.321；visibility switch≥5的MAR gap为4.702。Write density非单调，collision-loser仅8个且ME未恶化。

RW coarse与DT decoded edited_y的Lab mean为65.713/65.743，而final为13.640/17.144；appearance final gap主要在generator response中形成。Condition TC-ME为0.895/1.505，RW motion advantage已在conditioning中存在，final gap反而缩小。

## 7. PASS/FAIL/UNRESOLVED 判断

`PHASE0D2F_F2_STATUS=PASS`。TC-MAR/TC-ME存在，TC-CDE不存在。Mechanism结论均为DIAGNOSTIC_ONLY关联；over-conditioning假设为WEAKLY_SUPPORTED。

## 8. 对后续实验影响

允许进入F3，但本阶段未执行F3或GRID100。未来稀疏筛选必须同时观察appearance与motion gap；候选机制应优先覆盖高运动/变形与visibility fragmentation，而非假设motion已优于RW。

## 9. 遗留问题

SciPy在现有runtime不可用，Spearman未计算且未安装依赖。Local deformation coarse-bin定义未冻结，`S_i_3D`延后F3。Condition-level RGB由不同原生conditioning representations解码，仅作机制诊断，不能替代formal endpoint或证明因果。
