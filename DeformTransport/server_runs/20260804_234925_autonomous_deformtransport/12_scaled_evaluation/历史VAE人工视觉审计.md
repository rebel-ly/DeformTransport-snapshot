# 历史 Wan VAE 人工视觉审计

审计对象：首帧、中帧、末帧四联图；每行依次为 Original VAE recon、Correct masked replace、Shuffled masked replace、Coarse RGB input。

## 可见事实

- Shuffled 三个时刻均出现大面积红色块状/糊状纹理，衣服开襟、腰带、白色饰边和袖口结构显著丢失。
- Correct 的整体轮廓、水平位置、倾斜方向和主要红白区域比 Shuffled 更接近 coarse RGB input。
- Correct 并非高质量重建：中帧和末帧出现明显灰色拖影、边界模糊、白色饰边损失及局部透明感。
- Original VAE recon 与 coarse input 整体接近，说明主要退化发生在 transport masked replace，而不是单纯 VAE 重建。

## 与量化结果的关系

人工观察支持 Correct 相对 Shuffled 的 PSNR/SSIM 和 masked L1 优势，但同时否定“Correct 已达到可用生成质量”的强结论。当前结果更适合作为材料点对应有效性的机制 proxy，并暴露硬替换边界与单源 latent 表达不足。

## 限制

图像来自历史 VAE 解码，不是本轮服务器复跑；coarse input 不是未来真实 GT；三张关键帧不能代表全视频所有时刻。结论需要本轮真实 GPU smoke 和完整 RealWonder 对照验证。

审计拼图：`历史VAE视觉审计缩略图.jpg`。
