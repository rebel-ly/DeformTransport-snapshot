#!/usr/bin/env bash
set -u

DT=/workspace/DeformTransport
V4=$(cat "$DT/server_runs/wan_move_method_dev/current_v4_wave1.txt")

echo "V4=$V4"

echo
nvidia-smi \
    --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader

for CASE in santa tree
do
    echo
    echo "================ $CASE ================"

    for V in \
        v4a_d20 \
        v4a_d40 \
        v4b_noise \
        v4c_hybrid
    do
        D="$V4/$CASE/$V"

        if [ -s "$D/exit_code.txt" ]; then
            EC=$(cat "$D/exit_code.txt")
        else
            EC="RUNNING/WAITING"
        fi

        if [ -s "$D/output_sha256.txt" ]; then
            S="VIDEO_OK"
        else
            S="-"
        fi

        printf '%-14s exit=%-16s %s\n' \
            "$V" "$EC" "$S"
    done

    echo "--- queue tail ---"
    tail -6 "$V4/$CASE/queue.log" 2>/dev/null || true
done
