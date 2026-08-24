# 未解决问题

1. 当前没有独占空闲 GPU；所有计算 PID 均属于其他用户。
2. 完整、无损且 provenance 可核验的 Santa `final_sim` 缺失。
3. `transport_ready.pt` 缺少案例输入 hash、目标 latent hash、checkpoint hash 和可迁移路径。
4. 尚无本轮真实 Wan VAE GPU smoke，也无完整 RealWonder Baseline/Correct/Shuffled 输出。
5. 没有未来真实 GT，LPIPS/FVD 也未安装；当前仅报告无需新增重依赖的 proxy 指标。
6. 本地没有完整的机器人 deformable 官方 case，不能构造具备科学有效性的 robot 定量对照。
