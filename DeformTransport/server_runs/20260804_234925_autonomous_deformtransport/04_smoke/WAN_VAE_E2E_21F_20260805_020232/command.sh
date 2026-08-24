#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo $$ > "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/WAN_VAE_E2E_21F_20260805_020232/container_pid.txt"
exec timeout --signal=TERM --kill-after=10s 300s python -u scripts/run_wan_vae_transport_probe.py \
 --transport-ready "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_videoproxy_transport_ready/transport_ready_videoproxy.pt" \
 --checkpoint wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth \
 --output-dir "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/WAN_VAE_E2E_21F_20260805_020232/outputs" --seed 0 --visual-inspection pending --visual-note "服务器本轮真实GPU闭环；输入为明确标注的MP4派生proxy"
