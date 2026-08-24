#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "用法: run_correct.sh GPU编号" >&2; exit 64; fi
gpu_id="$1"
run_root=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport
sim_dir="$run_root/prepared_inputs/santa_21f_final_sim"
checkpoint='/workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt'
artifact=/workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
exec env CUDA_VISIBLE_DEVICES="$gpu_id" python infer_sim.py --checkpoint_path "$checkpoint" --sim_data_path "$sim_dir" --output_path "$run_root/06_santa_comparison/correct_seed0.mp4" --seed 0 --transport_latent_path "$artifact" --transport_mode correct
