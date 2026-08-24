# Phase 0D / 0D-3 formal 5-seed DT-FULL floor and DROP-ZERO62 evaluation

## 1. 阶段目标

在冻结 corrected-v2 evaluator/runtime 下，建立 DT-FULL 的五种子描述性随机尺度，并以 seed0 对 seed0 比较 DROP-ZERO62。

## 2. 审计问题

冻结的 N=62 `ZERO_SWITCH_POSITIVE_VISIBLE` carrier 删除，是否相对 DT-FULL seed0 同时改善 TC-MAR 与 TC-ME；以及 DT-FULL 的随机种子波动幅度。

## 3. 使用的数据

RW seed0、DT-FULL canonical seed0 及 seed1--4、DROP-ZERO62 seed0。DT-FULL 是 N=1257；DROP-ZERO62 为保持 canonical ID 顺序的 N=1195，且使用 normal subset re-arbitration。

## 4. 使用的方法

宿主 `/mnt/sdbd/home/liuyu_qyh/tools/miniforge3/envs/wan-move/bin/python`、`CUDA_VISIBLE_DEVICES=1` 与 evaluator SHA `e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5`。先零容差重现 RW/DT seed0；其后逐候选运行 TC-MAR、RGB-L1 与 TC-ME。ZERO62 仅运行 TC-MAR appearance diagnostic；不使用 subgroup TC-ME。

## 5. 关键命令/脚本

`run_formal_candidate.sh` 为每一候选建立隔离 symbolic binding 并调用冻结 evaluator。`zero62_appearance_diagnostic.py` 复用冻结 evaluator 的 appearance helpers。

## 6. 关键结果

RW 与 DT seed0 的全部指定统计量精确复现，两个 baseline gate 均 PASS。DT 五种子 TC-MAR mean 的 mean/sample-SD/min/max 为 18.367141901067804 / 0.9042641376188056 / 17.144317299874714 / 19.482984302700842；TC-ME 对应 0.7058747544754642 / 0.07113175518249301 / 0.6386018935544951 / 0.8198994616629207。

DROP-ZERO62 seed0 的 TC-MAR mean/median/p95 为 17.310077134304095 / 15.3817307472229 / 38.589810376167286；TC-ME 为 0.6961113799471432 / 0.6675841485414448 / 1.2906829794012262。对 DT seed0 的 signed delta：MAR `+0.16575983442938025`、RGB-L1 `-0.0035445736904753944`、ME `-0.03043858748177608`。因此为 `CASE_DZ_C_MOTION_IMPROVES_APPEARANCE_WORSENS`。

ZERO62 TC-MAR means：DT-FULL 14.168292969801737、DROP 14.963870509497584、RW 23.556203480112934、WM0 27.906852825538763。补集 1195 的 means：DT-FULL 17.32037980795155、DROP 17.448879434626615。

## 7. PASS/FAIL/UNRESOLVED 判断

正式 evaluator/runtime binding、两条 baseline exact reproduction、全部五个 DT seed 与 DROP seed0 evaluation 均 PASS。`SUBGROUP_TCME_USED_FOR_DECISION=False`。

## 8. 对后续实验影响

五种子结果仅是 `DESCRIPTIVE_STOCHASTIC_SCALE_ONLY`，不是等价界或追溯 promotion rule。DROP 删除在同种子上未同时改善主轴，路线为 `ROUTE_2_DROP62_NOT_PROMISING`；停止 ZERO62-specific GPU 投入。未启动任何新 GPU 实验。

## 9. 遗留问题

DROP 只有 seed0；本结果不构成多种子稳健性或等价性结论。若未来另行科学授权，应使用预定义 paired same-seed differences。
