# 下一实验

1. 第一张通过独占门禁的 L40：运行真实 Wan VAE + Target/Correct/Shuffled transport artifact GPU smoke；记录模型加载、三路解码、有限性、显存和视频。
2. 恢复原始 Santa 21 帧 PNG、flows 和完整 `final_sim`，执行 CPU validator，并固化全部输入 SHA256。
3. 同一张独占 GPU 串行运行 Baseline、Correct、Shuffled，固定 checkpoint、generator、seed 0、noise、prompt、mask、分辨率和编码参数。
4. 首组三组成功后扩展至少 3 个 seed，报告均值、方差、逐帧指标和失败案例。
5. 搜索或构建具备完整模拟轨迹与未来 GT 的机器人 deformable case；不存在有效资产前不做虚构机器人结论。
6. 分析 contribution count 与误差/伪影相关性，再决定是否实现 softmax splatting；检查 transport 影响随去噪衰减后再考虑逐步注入。
