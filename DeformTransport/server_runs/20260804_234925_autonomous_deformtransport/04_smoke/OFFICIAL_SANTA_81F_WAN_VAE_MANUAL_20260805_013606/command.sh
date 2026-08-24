#!/usr/bin/env bash
set -eo pipefail

cd /workspace/DeformTransport

source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen

export CUDA_VISIBLE_DEVICES=2

timeout --signal=TERM --kill-after=20s 1800s python -u /workspace/DeformTransport/scripts/run_wan_vae_transport_probe.py   --transport-ready /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/official_santa_81f_transport_ready_20260805_011211/transport_ready.pt   --checkpoint /workspace/DeformTransport/wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth   --output-dir /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_WAN_VAE_MANUAL_20260805_013606/outputs   --seed 0   --visual-inspection pending   --visual-note "Official Santa 81-frame Wan VAE transport probe; visual review pending"
