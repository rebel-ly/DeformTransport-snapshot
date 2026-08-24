# Phase0D-1R — Formal Corrected-v2 Runner Recovery and Pairwise Seed-Floor Contract

## 1. 阶段目标
恢复 corrected-v2 Santa N=1257 V3D formal runner，功能验证 seed conditioning invariance，并冻结 pairwise-only output stochasticity contract。

## 2. 审计问题
验证新 runner 只绑定 corrected-v2 formal assets，internal trajectory randperm 是否影响 generator-consumed V3D conditioning，以及无 GT 时哪些 video-pairwise metrics 可用。

## 3. 使用的数据
只读复用 0D-1 formal input contract：corrected-v2 tracks/visibility/IDs、authoritative selected depth、source image、prompt；N=1257/T=81/steps0..800。

## 4. 使用的方法
新 Evidence runner 使用 legacy invocation syntax only，但不复用 legacy paths。shell syntax/dry-run 验证；五 seed CPU synthetic functional conditioning audit；基于冻结源码给出 algebraic generality proof；审计 pairwise evaluator contract。未启动 14B generation。

## 5. 关键命令/脚本
`formal_run_corrected_v2_v3d.sh` supports `--dry-run SEED OUTPUT_DIR`; shell check and dry run PASS。

## 6. 关键结果
runner has N1277 hits=0 and binds all six corrected-v2 assets/variant/seed/output. Internal first and third randperm sequences vary across seeds (unique counts 5/5), yet synthetic edited_y, support and winner mapping are bitwise equal across seeds. Pairwise contract supports decoded RGB exact mismatch, MAE, RMSE, PSNR; SSIM and LPIPS are unavailable without a frozen implementation.

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0D1R_STATUS = FAIL`; scope=`PAIRWISE_OUTPUT_STOCHASTICITY_ONLY`; formal future-video quality reference remains `UNRESOLVED_NO_GT_REFERENCE`.

## 8. 对后续实验影响
主对话可授权 Phase0D-2 same-seed replay under pairwise-only scope. Seed0 replayA must be reused for Phase0D-3; do not create a third seed0 output.

## 9. 遗留问题
formal future-video GT/evaluator lineage remains unresolved; pairwise metrics cannot establish task quality; seed invariance evidence is synthetic functional plus algebraic general, not real-latent functional; legacy N=1277 Santa remains rejected.
