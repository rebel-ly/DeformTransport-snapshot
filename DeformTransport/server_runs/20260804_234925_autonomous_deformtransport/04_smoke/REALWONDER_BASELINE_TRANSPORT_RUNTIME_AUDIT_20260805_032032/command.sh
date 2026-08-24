#!/usr/bin/env bash
set -eo pipefail
cd /workspace/DeformTransport
source /workspace/tools/miniforge3/etc/profile.d/conda.sh
conda activate realwonder-gen
python -u server_runs/20260804_234925_autonomous_deformtransport/04_smoke/baseline_transport_runtime_audit_queued/probe.py --checkpoint /workspace/DeformTransport/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt --final-sim /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/santa_21f_final_sim_proxy_v1 --artifact /workspace/DeformTransport/artifacts/transport_validation/santa_cloth_21f/wan_vae/vae_latent_outputs.pt --output /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_BASELINE_TRANSPORT_RUNTIME_AUDIT_20260805_032032/outputs
