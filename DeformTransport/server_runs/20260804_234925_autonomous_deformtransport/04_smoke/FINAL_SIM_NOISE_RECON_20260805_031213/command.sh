#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo $$ > "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/FINAL_SIM_NOISE_RECON_20260805_031213/container_pid.txt"
export TORCH_HOME=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/torch_cache
exec timeout --signal=TERM --kill-after=10s 600s python -u server_runs/20260804_234925_autonomous_deformtransport/04_smoke/final_sim_noise_reconstruction_queued/generate_noise.py server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_final_sim_proxy_v1 --seed 0
