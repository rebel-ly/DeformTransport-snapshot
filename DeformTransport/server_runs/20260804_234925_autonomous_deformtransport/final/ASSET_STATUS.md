# 资产状态

| 资产 | 状态 | 证据或限制 |
|---|---|---|
| RealWonder 生成 checkpoint | 完整 | 18,774,247,788 字节，SHA256 已记录 |
| Wan VAE checkpoint | 完整 | 507,609,880 字节，SHA256 `38071a...981` |
| Wan diffusion、CLIP、T5 | 完整 | 文件存在且 SHA256 已记录 |
| Santa `transport_ready.pt` | 张量有效 | 34,229,269 字节，validator 通过；内部外部路径不可迁移 |
| Santa `vae_latent_outputs.pt` | 完整可执行 | Target/Correct/Shuffled、mask/count 完整，SHA256 `f2ad92...345c` |
| Santa 完整 `final_sim` | 缺失 | 无原始 21 帧完整输入目录；不能公平运行生成器 |
| 官方 RealWonder case | 部分完整 | input/inpaint/config 存在，但无预计算 `final_sim` 或未来 GT |
| 机器人 deformable case | 缺失 | 仅发现 Panda XML 与文本提示，不构成完整实验资产 |
| 未来 GT | 缺失 | 当前只允许 proxy 指标 |
| LPIPS/FVD | 未恢复 | 不为无 GT 的阶段引入重依赖 |

## 本轮新增代理资产（2026-08-05）

| 资产 | 状态 | 证据或限制 |
|---|---|---|
| video proxy v1 | 保留失败现场 | initial 与 coarse 均为 832×480，不满足官方加载契约 |
| video proxy v2 | 资产差异审计通过但不可完整运行 | 仅修正 initial；除该字段外完全不变；SHA256 `294f3381...45a6` |
| video proxy v3 | 真实 Wan VAE 闭环通过 | initial 与 21 个 coarse 均为 512×512；SHA256 `2c842162...fd3d`；未来帧仍为有损 proxy |
| 完整 Santa final_sim | 仍缺失 | 因此完整 RealWonder Baseline/Correct/Shuffled 仍不得声称已运行 |
