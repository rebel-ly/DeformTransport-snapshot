#!/usr/bin/env bash
set -euo pipefail
trap 'code=$?; echo "$code" > /workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_SIM_4F_20260805_044638/exit_code.txt' EXIT

cd /workspace/DeformTransport
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/workspace/tools/conda-libs/deformtransport-gl/lib:${LD_LIBRARY_PATH:-}
/workspace/tools/venvs/deformtransport-sim/bin/python -u scripts/run_realwonder_trajectory_probe.py \
  --demo-data demo_web/demo_data/santa_cloth \
  --frames 4 \
  --direction right \
  --strength 1 \
  --seed 0 \
  --output-dir server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_SIM_4F_20260805_044638/outputs
