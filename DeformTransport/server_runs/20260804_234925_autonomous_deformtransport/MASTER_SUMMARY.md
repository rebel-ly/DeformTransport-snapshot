# 主运行摘要（进行中）

## 运行身份

- RUN_ID：`20260804_234925_autonomous_deformtransport`
- 主机：`gpu3`
- 用户：`liuyu_qyh`
- 容器：`deformtransport-dev`，持续运行，未重启
- 项目：`/mnt/sdbd/home/liuyu_qyh/DeformTransport`

## 已完成

1. 远程命令、代理、OpenAI API TCP/TLS、容器挂载和运行状态验证。
2. 四张 L40 及全部 compute 进程只读溯源；已确认进程均属于其他用户 `pengzhennan_gyj`，不属于 Docker/DeformTransport，未作任何干预。
3. `realwonder-gen` 最小依赖恢复：仅增加 imageio、einops、omegaconf、peft 及 omegaconf 必需的 antlr runtime；torch、CUDA、numpy、diffusers、transformers 等保护版本未变化。
4. `pip check` 通过；CPU 单元测试 25/25 通过；compileall 和 `git diff --check` 通过。
5. 模型、软链接、checkpoint、Wan VAE 和 transport artifact 的路径、字节数与 SHA256 审计。
6. Baseline/Correct/Shuffled 命令重建；transport 静态调用链审计；输入验证器、资源监控和视频指标脚本准备并自测。
7. 本地 RealWonder 官方案例与机器人案例搜索；确认没有完整 `final_sim`，也没有可作为未来 GT 的机器人 deformable case。
8. 完成中文方法与文献分析，并准备真实 Wan VAE + transport artifact 单卡 GPU smoke。

## 当前可用证据

现有历史 Wan VAE proxy 报告中，Correct 的 latent 区域平均 L1 为 0.35778485，Shuffled 为 0.52393939；Correct 在 6/6 latent 帧更低。解码域 Correct 平均 masked L1 为 0.12154974，Shuffled 为 0.23445781；Correct 在 21/21 帧不劣。它们是 VAE proxy，不是完整 RealWonder 生成结果。

## 当前阻塞

- 四张 GPU 当前均有其他用户 compute PID，我方没有 GPU 任务。
- Santa 完整无损 `final_sim` 缺失，RealWonder Baseline/Correct/Shuffled 完整生成不能公平启动。
- 现有 `transport_ready.pt` 内部外部路径指向旧机器 `/home/a/DeformTransport/...`；轨迹张量本身完整，但原始 21 帧 PNG/flows 未随资产迁移。
- 未来 GT、LPIPS/FVD 依赖和完整机器人 deformable case 缺失。

## 下一动作

第一张满足独占门禁的 GPU 将用于真实 Wan VAE 模型加载和 Target/Correct/Shuffled 三路解码 smoke；不使用 sleep、空转或纯占显存任务。完整 `final_sim` 恢复后，优先级切回同卡串行 Baseline→Correct→Shuffled。
