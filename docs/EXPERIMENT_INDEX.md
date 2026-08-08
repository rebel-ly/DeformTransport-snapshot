# DeformTransport 实验索引

## 当前正式Santa开发实验

| 简洁名称 | 含义 |
|---|---|
| 01_realwonder_baseline.mp4 | RealWonder Baseline |
| 02_point_hard_correct.mp4 | 正确材料点身份Hard Transport |
| 03_point_identity_shuffled.mp4 | 打乱材料点与latent身份 |
| 04_point_blend_a050.mp4 | Correct与Baseline按0.5融合 |
| 05_flow_transport.mp4 | RAFT二维光流latent运输 |

快捷入口：

server_runs/20260804_234925_autonomous_deformtransport/current/santa

## 已成立结论

Correct明显优于Shuffled。

Blend比Hard Correct更合理。

## 尚未成立结论

DeformTransport优于RealWonder。

3D材料点优于2D Flow。

当前Santa能够验证机器人动作。

## 当前开发状态

Hard Transport单元测试：

- 14 tests
- exit code 0
- 2026-08-05

下一项：

Soft Bilinear Point Transport。
