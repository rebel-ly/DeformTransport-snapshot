# DeformTransport 自主科研运行总摘要（当前阶段）

## 结论先行

当前服务器环境已恢复到可运行 RealWonder 生成依赖的状态，CPU 验收全部通过，模型与 transport 资产已核验，真实 Wan VAE + transport GPU smoke 已完成 CPU 前置门禁和安全启动封装。但截至 2026-08-05 00:55，四张 L40 均存在其他用户 compute PID，本轮没有占用或共享 GPU，也尚未产生新的 GPU 结果。

现有历史 Santa VAE proxy 中，Correct 在 latent L1、解码 masked L1、压缩视频域 PSNR/SSIM 和时序差分上均优于 Shuffled；人工视觉检查同样显示 Correct 更能保留轮廓和运动方向。然而 Correct 仍有明显模糊、灰色拖影和细节损失。这些证据不等于完整 RealWonder 生成提升。

## 环境验收

- `realwonder-gen`：`pip check` 通过。
- CPU 单元测试：25/25 通过。
- `compileall`：通过。
- `git diff --check`：通过。
- 受保护版本未变化；仅补装缺失的固定版本顶层依赖及 omegaconf 必需 runtime。
- 容器 `deformtransport-dev` 持续运行，未重启。

## 资产结论

- RealWonder checkpoint、Wan VAE、Wan diffusion、CLIP、T5 均存在且 SHA256 已记录。
- `transport_ready.pt` 张量合同有效，但外部路径仍指向旧机器。
- `vae_latent_outputs.pt` 完整，足以运行真实 Wan VAE 三路解码 smoke。
- 完整 Santa `final_sim`、原始 21 帧路径资产、未来 GT 和完整机器人 deformable case 缺失。

## 现有 proxy 结果

| 指标 | Correct | Shuffled | 边界 |
|---|---:|---:|---|
| latent masked L1 | 0.357785 | 0.523939 | 历史 VAE proxy |
| decoded masked L1 | 0.121550 | 0.234458 | 历史 VAE proxy |
| 压缩视频 masked PSNR dB | 16.7664 | 11.0733 | coarse target，非未来 GT |
| 压缩视频 masked SSIM | 0.4798 | 0.2964 | 本地 Gaussian SSIM |
| 时序差分 L1 | 0.014853 | 0.015487 | 全帧 proxy |

## 代码审查关键风险

1. transport loader 只比 shape，不能识别同 shape 的错误案例。
2. 历史 fused latent 整体覆盖本次 freshly encoded reference，mask 外一致性没有保证。
3. `final_sim` coarse 帧数与 `4*T-3` 时序不一致时，入口会静默裁剪或重复末 latent。

因此 Baseline 可在完整输入恢复后先跑；Correct/Shuffled 完整生成前应修复 provenance 和时序 fail-fast。

## GPU 调度

所有已见 GPU 计算进程均属于 `pengzhennan_gyj`，分类 A，不干预。GPU3 曾短暂被报告为空，但复核时新 PID 257153 已进入；本轮未创建虚假锁或空转占位。下一张独占空闲卡将启动真实 Wan VAE 三路解码，不使用 sleep、空转或纯显存任务。

## 当前状态

本轮仍在等待独占 GPU。没有完整 RealWonder Baseline/Correct/Shuffled 新视频，因此最终科研主张保持开放，不宣布方法优于 Baseline 或外部工作。
