#!/usr/bin/env bash
set -eo pipefail

cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen

export CUDA_VISIBLE_DEVICES=2
export TORCH_HOME=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/torch_cache

timeout --signal=TERM --kill-after=10s 600s python -u /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/final_sim_noise_reconstruction_queued/generate_noise.py   /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_ASSEMBLY_MANUAL_20260805_003555/final_sim   --seed 0
