# 下一步

1. 继续只读 GPU 门禁；任一卡无其他用户 compute PID、显存接近空闲、利用率接近 0、温度合理时，立即运行已预检的真实 Wan VAE transport GPU smoke。
2. GPU smoke 完成后检查 PID、退出码、stdout/stderr、结果 JSON、三路 MP4、峰值显存和有限性，并更新锁与运行表。
3. 恢复完整、无损、可核验的 Santa `final_sim`；先运行 CPU validator，再运行 Baseline。
4. Correct/Shuffled 前修复同 shape provenance、mask 外 reference 一致性和时序 fail-fast，并补测试。
5. 同卡串行三组和多 seed 后再形成生成质量结论。
