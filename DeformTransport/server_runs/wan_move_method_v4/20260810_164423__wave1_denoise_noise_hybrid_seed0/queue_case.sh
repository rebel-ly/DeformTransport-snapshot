#!/usr/bin/env bash
set -euo pipefail

CASE="${1:?case}"
GPU="${2:?gpu}"

DT=/workspace/DeformTransport
V4=$(cat "$DT/server_runs/wan_move_method_dev/current_v4_wave1.txt")

for V in \
    v4a_d20 \
    v4a_d40 \
    v4b_noise \
    v4c_hybrid
do
    echo
    echo "============================================================"
    echo "[$(date -Iseconds)] START $CASE $V GPU$GPU"
    echo "============================================================"

    bash "$V4/run_one.sh" \
        "$CASE" \
        "$V" \
        "$GPU"

    echo "[$(date -Iseconds)] DONE $CASE $V GPU$GPU"
done

date -Iseconds \
    > "$V4/$CASE/QUEUE_DONE.txt"

echo "${CASE^^}_V4_WAVE1_QUEUE_COMPLETE"
