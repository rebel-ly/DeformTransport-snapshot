#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo $$ > "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/DT_TRANSPORT_GPU_PARITY_20260805_015718/container_pid.txt"
exec timeout --signal=TERM --kill-after=10s 120s python -u "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/DT_TRANSPORT_GPU_PARITY_20260805_015718/run_transport_gpu_validation.py" \
  --transport-ready "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy.pt" \
  --latent-artifact "/workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt" \
  --output-dir "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/DT_TRANSPORT_GPU_PARITY_20260805_015718/outputs" \
  --seed 0
