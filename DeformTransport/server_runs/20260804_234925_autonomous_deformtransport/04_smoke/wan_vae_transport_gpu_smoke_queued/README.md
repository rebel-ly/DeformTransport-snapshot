# 真实 Wan VAE transport GPU smoke 队列包

Baseline 因完整 `final_sim` 缺失而阻塞。本任务对应紧急调度优先级 3，使用真实 Wan VAE checkpoint 和现有 Santa transport latent artifact，完成模型加载、Target/Correct/Shuffled 三路解码、有限性、shape、差异、显存和视频输出验证。

本目录当前仅为排队模板，未声称已经运行。获得独占空闲卡后，必须创建新的时间戳运行目录，不覆盖旧结果。

容器内命令模板：

```bash
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
CUDA_VISIBLE_DEVICES=<GPU编号> python -u server_runs/20260804_234925_autonomous_deformtransport/04_smoke/wan_vae_transport_gpu_smoke_queued/run_gpu_smoke.py \
  --artifact artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt \
  --checkpoint wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth \
  --output-dir <新运行目录>/outputs \
  --seed 0
```

启动前必须通过独占门禁；启动后锁文件 `/tmp/deformtransport_gpu_<GPU编号>.lock` 记录 RUN_ID、GPU、任务 PID、时间和任务名称。
