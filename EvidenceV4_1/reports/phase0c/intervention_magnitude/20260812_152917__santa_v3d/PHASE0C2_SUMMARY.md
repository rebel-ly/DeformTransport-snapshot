# Phase0C-2 — Operator Intervention Magnitude Audit

## 1. 阶段目标
量化 V3D intervention 的 structural magnitude，并与 synthetic harness-only amplitude diagnostics 区分。

## 2. 审计问题
确认写入是否只位于 intended future support、Correct/Shuffled support 是否严格一致，并盘点 cached real latent provenance。

## 3. 使用的数据
只读使用 0B-4R saved edited_y 与冻结 synthetic construction。

## 4. 使用的方法
精确重建 [1,4,21,60,104] synthetic y：source slot channel0=1、channel1=x/103、channel2=y/59、channel3=linear cell ID/(60×104−1)，future slots=0；逐元素比较 saved edited_y。只在 formal Santa/new_audit/evidence 进行有界缓存 latent 搜索。

## 5. 关键命令/脚本
`run_phase0c2.py`；未运行 GPU、V3D、WanVAE 或图像编码。

## 6. 关键结果
future cells=124800；Correct/Shuffled support=9031/9031；fraction=0.072363782。source modified cells=0/0；outside-support=0/0；changed cells=9031/9031；changed scalars=36124/36124。

`SYNTHETIC_HARNESS_ONLY`: Correct/Shuffled L1=22399.617188/22100.326172，L2=128.989517/127.195686，Linf=1.000000/1.000000。它们不是 real Wan latent amplitude。real latent status=`UNRESOLVED_NO_VALID_CACHED_LATENT`。

## 7. PASS/FAIL/UNRESOLVED 判断
`PHASE0C2_STATUS = PASS`。structural audit PASS；real-latent amplitude UNRESOLVED 不构成 structural FAIL。

## 8. 对后续实验影响
可报告 intervention coverage，不能将 synthetic norms 解释为真实 latent amplitude 或 method superiority。

## 9. 遗留问题
尚无 provenance 完整的 cached real Wan latent；未来真实 amplitude 需要独立可审计 image/resize-crop/Wan-VAE-checkpoint lineage。
