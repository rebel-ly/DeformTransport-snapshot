# Phase0D-2F / F0-R2 summary

## 1. 阶段目标

闭合真实 Wan transport latent shape，并尽可能最小闭合 480→464 output origin。

## 2. 审计问题

仅处理 real VAE/transport shape 和 final height discrepancy；N1277 contamination 已由 F0-R 关闭且未重审。

## 3. 使用的数据

前两轮 F0 archives、formal replay logs/output metadata、Wan-Move frozen source/config/VAE code、formal source image contract。

## 4. 使用的方法

静态源码逐层 shape proof；GPU1/GPU2 preflight；一次严格 VAE-only probe，带持久化 stderr/exit code。无 diffusion/T5/CLIP/14B 加载。

## 5. 关键命令/脚本

`nvidia-smi -i 1/-i 2`、container `nl/grep`、`f0r2_vae_probe.py`。探针以错误的 `/tmp` import binding 退出，未修改环境且不重试。

## 6. 关键结果

WanMove 目标公式确定 h=480,w=832；VAE Encoder3d 三个空间 stride-2 下采样对 480→240→120→60、832→416→208→104，temporal 81 按 1+20×4 得 21；z_dim=16。WanVAE.encode squeeze batch 后 y=[16,21,60,104]，transport caller passes [1,16,21,60,104] to replace_feature. 464 output origin 在静态正式路径未出现。

## 7. PASS/FAIL/UNRESOLVED 判断

Real latent resolved from exact deterministic formal code. 464 cause remains `UNRESOLVED_NONBLOCKING_FOR_EVALUATION`; all nonblocking conditions are met.

## 8. 对后续实验影响

F0 recovered status is `PASS_WITH_OUTPUT_ORIGIN_LIMITATION`; F1 may proceed under frozen 464x832 evaluation mapping, but this turn does not execute F1.

## 9. 遗留问题

If exact provenance of 464 is later required, it needs pre-existing runtime intermediate or separately authorized instrumentation; no diffusion rerun is allowed in this phase.
