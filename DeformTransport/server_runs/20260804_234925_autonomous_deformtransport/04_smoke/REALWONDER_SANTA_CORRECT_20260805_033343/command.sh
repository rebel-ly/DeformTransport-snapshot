#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
export TORCH_HOME=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/torch_cache
python -u infer_sim.py --checkpoint_path /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt --sim_data_path /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_final_sim_proxy_v1 --output_path /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_CORRECT_20260805_033343/santa_correct_seed0.mp4 --seed 0 --eval_degradation 0.5 --local_attn_size 21 --transport_latent_path /workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt --transport_mode correct  &
child=$!
echo $child > "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_CORRECT_20260805_033343/container_pid.txt"
while kill -0 $child 2>/dev/null; do
  ts=$(date --iso-8601=seconds); rss=$(awk '/VmRSS/{print $2}' /proc/$child/status 2>/dev/null || echo 0); hwm=$(awk '/VmHWM/{print $2}' /proc/$child/status 2>/dev/null || echo 0); printf '%s,%s,%s\n' "$ts" "$rss" "$hwm" >> "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_CORRECT_20260805_033343/process_memory.csv"; sleep 15
done
wait $child
