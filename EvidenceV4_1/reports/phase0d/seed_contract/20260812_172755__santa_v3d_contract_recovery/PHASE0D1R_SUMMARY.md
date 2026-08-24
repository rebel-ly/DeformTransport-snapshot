# Phase0D-1R — Formal Corrected-v2 Runner Recovery and Pairwise Seed-Floor Contract

## 1. 阶段目标
恢复 corrected-v2 Santa N=1257 V3D formal runner，功能验证 seed conditioning invariance，并冻结 pairwise-only output stochasticity contract。

## 2. 审计问题
验证 runner 的 corrected-v2 binding、internal randperm 对 generator-consumed V3D conditioning 的影响，及无 GT 时 pairwise output metrics 的可用性。

## 3. 使用的数据
复用 0D-1 frozen corrected-v2 N=1257 tracks/visibility/IDs/depth/image/prompt，及冻结 Wan-Move source。

## 4. 使用的方法
runner 仅 syntax/dry-run；五 seeds 的 CPU synthetic `create_pos_feature_map`/`replace_feature` functional audit；canonical winners 按 `(tau,target cell,material ID,float32 depth bits)` 比较；附代数证明。无 14B generation。

## 5. 关键命令/脚本
`formal_run_corrected_v2_v3d.sh`，`conditioning_seed_invariance.json`，`pairwise_evaluation_contract.json`。

## 6. 关键结果
N1277 path hits=0。五 seed internal first/third randperm sequences each have 5/5 unique sequences，但 edited_y/support/canonical winner map 全部 exact equal。首个 0D-1R 尝试错误地在 winner serialization 中包含 transient permuted carrier index；该工程错误 evidence 保留，本 retry 使用排列不变 canonical identity。Pixel pairwise metrics PASS；SSIM/LPIPS unavailable without frozen implementations。

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0D1R_STATUS = PASS`。formal future quality reference remains unresolved, but permitted scope is `PAIRWISE_OUTPUT_STOCHASTICITY_ONLY`。

## 8. 对后续实验影响
`PROCEED_TO_PHASE0D2=True`；Phase0D-2 may only assess decoded-frame same-seed determinism, not task quality.

## 9. 遗留问题
GT/reference evaluator remains unresolved; real-latent seed test not performed; historical N=1277 remains rejected; pairwise metrics do not establish video quality.
