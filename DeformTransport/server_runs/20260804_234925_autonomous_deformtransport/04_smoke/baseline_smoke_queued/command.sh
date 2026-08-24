#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "用法: command.sh GPU编号" >&2
  exit 64
fi
gpu_id="$1"
run_root=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport
sim_dir="$run_root/prepared_inputs/santa_21f_final_sim"
checkpoint='/workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt'
for required in config.yaml noises.npy resized_input_image.png prompt.txt; do
  test -r "$sim_dir/$required" || { echo "缺失输入: $sim_dir/$required" >&2; exit 66; }
done
test -d "$sim_dir/frames" || { echo "缺失frames目录" >&2; exit 66; }
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
exec env CUDA_VISIBLE_DEVICES="$gpu_id" python infer_sim.py \
  --checkpoint_path "$checkpoint" \
  --sim_data_path "$sim_dir" \
  --output_path "$run_root/04_smoke/baseline_smoke_queued/baseline_seed0.mp4" \
  --seed 0
