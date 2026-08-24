# GPU 调度摘要

- 唯一 watcher：PID 291997，持续存活；未创建第二个 watcher。
- 采样周期：5 秒；监控 GPU 0、1、2、3。
- 已完成 MICRO：`DT_TRANSPORT_GPU_PARITY_20260805_015718`。
- 已完成 SHORT：`WAN_VAE_E2E_21F_20260805_022223`，GPU2 机会式共享，退出码0。
- SHORT 实测：PyTorch allocated 4320.065MiB、reserved 6190MiB；进程级 nvidia-smi 峰值5810MiB；计算44.715秒。
- 任务峰值时其他用户 PID 260762 为5406MiB，未受干预；整卡最低空闲34151MiB、最高温度66°C，无ECC/OOM/Xid。
- 当前项目 GPU 任务：0；锁已清理。watcher 继续动态监控。
- 已完成 Santa proxy 的完整原生 Baseline、Correct、Shuffled、Flow、Correct-Blend、seed1 与 EMA 对照；首次三组均在 GPU2 同 seed 顺序运行。
- 完整任务单进程采样显存峰值最高 31,844MiB，CPU VmHWM 约46.1GiB；两项完整模型并发时 MemAvailable 最低23.188GiB，因此当前安全并发上限为2个完整模型，不能按四卡盲目并发。
- 结论边界：Santa输入是有损工程 proxy，不是官方原始 final_sim 或 future GT；官方/机器人跨案例仍受完整 simulation 输出缺失阻塞。
