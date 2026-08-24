#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
echo $$ > "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_GENERATOR_LOAD_20260805_030721/container_pid.txt"
exec timeout --signal=TERM --kill-after=15s 600s python -u server_runs/20260804_234925_autonomous_deformtransport/04_smoke/generator_load_probe_queued/generator_load_probe.py --checkpoint /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt --output "/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_GENERATOR_LOAD_20260805_030721/outputs"
