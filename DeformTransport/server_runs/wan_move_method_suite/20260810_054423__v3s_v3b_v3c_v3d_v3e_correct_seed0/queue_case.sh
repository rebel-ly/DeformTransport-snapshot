#!/usr/bin/env bash
set -euo pipefail

CASE="${1:?case}"
GPU="${2:?gpu}"

DT=/workspace/DeformTransport

SUITE=$(cat \
"$DT/server_runs/wan_move_method_dev/current_v3_suite.txt")


for VAR in \
v3s \
v3b \
v3c \
v3d \
v3e
do

    echo
    echo "============================================"
    echo "[$(date -Iseconds)]"
    echo "START case=$CASE variant=$VAR GPU=$GPU"
    echo "============================================"

    bash \
    "$SUITE/run_one.sh" \
    "$CASE" \
    "$VAR" \
    "$GPU"

    echo
    echo "[$(date -Iseconds)]"
    echo "DONE case=$CASE variant=$VAR GPU=$GPU"

done


date -Iseconds \
> "$SUITE/$CASE/QUEUE_DONE.txt"

echo
echo "============================================"
echo "$CASE QUEUE COMPLETE"
echo "============================================"
