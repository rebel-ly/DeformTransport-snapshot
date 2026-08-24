#!/usr/bin/env bash
set -euo pipefail

DT=/workspace/DeformTransport
RUN=$(cat "$DT/server_runs/wan_move_method_dev/current_v3d_formal_validation.txt")

echo "[$(date -Iseconds)] START Tree V3D-Shuffled"
bash "$RUN/run_one.sh" tree shuffled 2

echo "[$(date -Iseconds)] DONE Tree V3D-Shuffled"

echo "[$(date -Iseconds)] START SandHouse V3D-Shuffled"
bash "$RUN/run_one.sh" sandhouse shuffled 2

echo "[$(date -Iseconds)] DONE SandHouse V3D-Shuffled"

date -Iseconds \
> "$RUN/GPU2_QUEUE_DONE.txt"

echo "GPU2_QUEUE_COMPLETE"
