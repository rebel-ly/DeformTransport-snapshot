#!/usr/bin/env bash
set -eo pipefail
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
python -u /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/checkpoint_generator_ema_audit_queued/probe.py /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_CHECKPOINT_GENERATOR_EMA_AUDIT_20260805_033029/outputs
