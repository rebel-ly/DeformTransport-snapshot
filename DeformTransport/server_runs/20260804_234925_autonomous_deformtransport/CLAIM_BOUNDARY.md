# Claim Boundary

## Already supported before this run

Persistent point identity carries useful appearance information in the Santa RGB proxy and real Wan VAE latent probes.

## Not yet established

- Improvement of full RealWonder generation over its baseline.
- Superiority to a fair flow-warp baseline.
- Robot deformable-object effectiveness.
- Cross-object or cross-action generalization.
- Robustness to trajectory errors.

Santa is a wind-driven cloth case and must not be described as robot manipulation.

## 本轮新增支持（2026-08-05）

- 真实 Wan VAE 单帧编码解码、21帧编码、Correct/Shuffled transport 与六路解码闭环已在 GPU2 机会式共享下完成。
- 在本轮有损 coarse-RGB/latent proxy 上，Correct 的 latent masked L1 在 6/6 帧优于 Shuffled，decoded masked L1 在 21/21 帧优于 Shuffled。
- 这仍不证明完整 RealWonder 生成优于 Baseline；完整 `final_sim`、未来 GT、机器人案例与人工视觉验收仍缺失。
