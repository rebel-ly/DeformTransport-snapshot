# 历史 Wan VAE 视频域 proxy 审计

## 输入

- 参考：`wan_vae/target_input.mp4`
- Correct：`wan_vae/fused_correct.mp4`
- Shuffled：`wan_vae/fused_shuffled.mp4`
- Mask：`checkpoint_free/transport_mask.mp4`，阈值 0.5
- 三个视频均为 21 帧、480×832。

## 结果

| 变体 | masked 平均 PSNR dB | masked 平均 SSIM | 全帧时序差分 L1 |
|---|---:|---:|---:|
| Correct | 16.76636768 | 0.47977354 | 0.01485253 |
| Shuffled | 11.07328370 | 0.29640957 | 0.01548738 |

在这组三项 proxy 中，Correct 均优于 Shuffled。PSNR 高 5.6931 dB，SSIM 高 0.18336，时序差分 L1 低 0.000635。

## 审计边界

这些视频是历史 Wan VAE transport 解码产物，参考是 coarse target，而非未来真实 GT；视频与 mask 还经过 MP4 编码。该结果只支持 Correct 材料点对应在当前 VAE proxy 中比 Shuffled 更接近 coarse target，不能外推为 RealWonder 完整生成质量或物理真实性提升。

当前 SSIM 为本地 11×11 Gaussian-window 实现，边界使用零填充；适合相同实现下的组间比较，不替代标准评测工具链。LPIPS 和 FVD 未计算。
