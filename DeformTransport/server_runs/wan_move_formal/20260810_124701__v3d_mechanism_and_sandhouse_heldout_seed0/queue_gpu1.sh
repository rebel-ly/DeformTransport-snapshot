#!/usr/bin/env bash
set -euo pipefail

DT=/workspace/DeformTransport
RUN=$(cat "$DT/server_runs/wan_move_method_dev/current_v3d_formal_validation.txt")

echo "[$(date -Iseconds)] START Santa V3D-Shuffled"
bash "$RUN/run_one.sh" santa shuffled 1

echo "[$(date -Iseconds)] DONE Santa V3D-Shuffled"

echo "[$(date -Iseconds)] START SandHouse V3D-Correct"
bash "$RUN/run_one.sh" sandhouse correct 1

echo "[$(date -Iseconds)] DONE SandHouse V3D-Correct"

date -Iseconds \
> "$RUN/GPU1_QUEUE_DONE.txt"

echo "GPU1_QUEUE_COMPLETE"
