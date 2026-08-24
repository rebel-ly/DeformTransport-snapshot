# 完整生成命令重建

## 证据来源

命令来自 README.md、infer_sim.py、case_simulation.py、Santa 配置、checkpoint 归档键和现有 VAE artifact；没有猜测未定义参数。

## 固定路径

容器项目目录：/workspace/DeformTransport

Checkpoint：
/workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt

Transport artifact：
/workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt

待补输入：
/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_final_sim

## Baseline

    cd /workspace/DeformTransport
    source /workspace/tools/miniforge3/etc/profile.d/conda.sh
    conda activate realwonder-gen
    CUDA_VISIBLE_DEVICES=<GPU_ID> python infer_sim.py \
      --checkpoint_path '<CHECKPOINT>' \
      --sim_data_path '<SANTA_FINAL_SIM>' \
      --output_path '<RUN_ROOT>/04_smoke/baseline_smoke/baseline_seed0.mp4' \
      --seed 0

Baseline 不得提供 transport_latent_path 或 transport_mode。官方 README 默认加载 generator；checkpoint 元数据同时含 generator 和 generator_ema，本轮先遵循官方默认，不加 --use_ema。

## Correct

    CUDA_VISIBLE_DEVICES=<GPU_ID> python infer_sim.py \
      --checkpoint_path '<CHECKPOINT>' \
      --sim_data_path '<SANTA_FINAL_SIM>' \
      --output_path '<RUN_ROOT>/06_santa_comparison/correct_seed0.mp4' \
      --seed 0 \
      --transport_latent_path '<TRANSPORT_ARTIFACT>' \
      --transport_mode correct

## Shuffled

    CUDA_VISIBLE_DEVICES=<GPU_ID> python infer_sim.py \
      --checkpoint_path '<CHECKPOINT>' \
      --sim_data_path '<SANTA_FINAL_SIM>' \
      --output_path '<RUN_ROOT>/06_santa_comparison/shuffled_seed0.mp4' \
      --seed 0 \
      --transport_latent_path '<TRANSPORT_ARTIFACT>' \
      --transport_mode shuffled

## 必须固定

同一 final_sim、noises.npy、prompt、checkpoint、generator/EMA选择、seed、分辨率、latent帧数、denoising steps、mask、GPU和输出编码。首次三组在同一 GPU 串行运行。

## 时序与 shape 风险

现有 artifact 为 6 个 latent 时刻，对应 21 个 coarse RGB 帧的因果 VAE 编码。loader 要求它与 freshly encoded sim_latent 在 trim/pad 之前完全同 shape。因此待补 final_sim 应明确形成 21 pixel/coarse 帧和 6 latent 帧，并将 config.num_output_frames 设为与 noise 和生成链路一致的 6；不能直接沿用官方 Santa 配置中的 21 latent 帧而不重新制备 transport artifact。

该判断必须由 CPU 输入验证器和首次实际入口日志再次确认。当前 final_sim 缺失，以上命令是已重建但尚不可执行的模板。
