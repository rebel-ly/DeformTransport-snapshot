# 复现命令

以下路径均为容器 `deformtransport-dev` 内路径。

## CPU 验收

```bash
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
pip check
python -m unittest discover -s tests -v
python -m compileall -q deform_transport scripts tests infer_sim.py
```

## 当前可执行的真实 Wan VAE GPU smoke

按最终机会式共享策略，在剩余显存、温度、系统内存和持续负载满足任务实测条件时执行；其他用户 PID 的存在本身不是拒绝条件：

```bash
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
CUDA_VISIBLE_DEVICES=<GPU编号> python -u \
  server_runs/20260804_234925_autonomous_deformtransport/04_smoke/wan_vae_transport_gpu_smoke_queued/run_gpu_smoke.py \
  --artifact artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt \
  --checkpoint wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth \
  --output-dir <唯一新运行目录>/outputs \
  --seed 0
```

安全启动器：

```bash
server_runs/20260804_234925_autonomous_deformtransport/04_smoke/wan_vae_transport_gpu_smoke_queued/launch_if_free.sh <GPU编号>
```

启动器执行两次初始门禁和一次启动前最终门禁，创建唯一运行目录、`/tmp` 锁、manifest、日志、PID 与资源监控；门禁失败时不启动。

## RealWonder Baseline

该命令已从官方入口重建，但当前因完整 `final_sim` 缺失而不可执行：

```bash
CUDA_VISIBLE_DEVICES=<GPU编号> python infer_sim.py \
  --checkpoint_path ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt \
  --sim_data_path <通过CPU验证的Santa_21f_final_sim> \
  --output_path <唯一新运行目录>/baseline_seed0.mp4 \
  --seed 0
```

Baseline 不提供 `--transport_latent_path` 或 `--transport_mode`。Correct/Shuffled 命令见 `03_command_reconstruction/COMMAND_RECONSTRUCTION.md`。

## 历史 VAE 视频域 proxy

```bash
python server_runs/20260804_234925_autonomous_deformtransport/12_scaled_evaluation/evaluate_video_comparison.py \
  --reference artifacts/transport_validation/santa_cloth_21f/wan_vae/target_input.mp4 \
  --prediction artifacts/transport_validation/santa_cloth_21f/wan_vae/fused_correct.mp4 \
  --mask-video artifacts/transport_validation/santa_cloth_21f/checkpoint_free/transport_mask.mp4 \
  --output-json <输出.json> \
  --output-csv <逐帧.csv>
```

## 本轮已成功的21帧 Wan VAE 闭环

以下机会式共享规则覆盖上文“必须无其他用户 PID”的旧表述：只要剩余显存、温度、系统内存和负载满足已实测 SHORT 条件即可运行；只在实际风险时停止我方任务，不干预其他用户。

```bash
bash server_runs/20260804_234925_autonomous_deformtransport/04_smoke/wan_vae_transport_gpu_smoke_queued/launch_shareable_vae_e2e.sh 2
```

任务内部完整命令见 `04_smoke/WAN_VAE_E2E_21F_20260805_022223/command.sh`，输入为 v3 有损 video proxy；该命令复现工程闭环，不等价于完整 RealWonder 生成。
